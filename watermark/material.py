"""Payload and spread-spectrum watermark operations for sparse PBR voxels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import struct
import zlib
from typing import Any, Mapping

import torch


_MAGIC = b"T2WM"
_VERSION = 1
_MAX_CREATOR_BYTES = 80
_FRAME_SIZE = 256


@dataclass(frozen=True)
class WatermarkPayload:
    creator: str
    created_at: str
    method: str = "TRELLIS.2+DistSeal"

    @classmethod
    def now(cls, creator: str, method: str = "TRELLIS.2+DistSeal") -> "WatermarkPayload":
        return cls(creator, datetime.now(timezone.utc).isoformat(timespec="seconds"), method)

    def pack(self) -> bytes:
        if not self.creator.strip():
            raise ValueError("creator must not be empty")
        creator = self.creator.encode("utf-8")
        method = self.method.encode("utf-8")
        created = self.created_at.encode("ascii")
        if len(creator) > _MAX_CREATOR_BYTES or len(method) > 64 or len(created) > 40:
            raise ValueError("payload field exceeds supported size")
        body = json.dumps(
            {"creator": self.creator, "created_at": self.created_at, "method": self.method},
            separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        header = _MAGIC + bytes([_VERSION]) + struct.pack(">H", len(body))
        frame = header + body + struct.pack(">I", zlib.crc32(header + body))
        if len(frame) > _FRAME_SIZE:
            raise ValueError("watermark payload exceeds fixed frame size")
        return frame.ljust(_FRAME_SIZE, b"\0")

    @classmethod
    def unpack(cls, data: bytes) -> "WatermarkPayload":
        if len(data) != _FRAME_SIZE or data[:4] != _MAGIC or data[4] != _VERSION:
            raise ValueError("watermark header not found")
        size = struct.unpack(">H", data[5:7])[0]
        end = 7 + size
        if len(data) < end + 4:
            raise ValueError("truncated watermark payload")
        expected = struct.unpack(">I", data[end:end + 4])[0]
        if zlib.crc32(data[:end]) != expected:
            raise ValueError("watermark checksum mismatch")
        obj = json.loads(data[7:end].decode("utf-8"))
        return cls(obj["creator"], obj["created_at"], obj["method"])


def _payload_bits(payload: bytes) -> list[int]:
    return [(byte >> shift) & 1 for byte in payload for shift in range(7, -1, -1)]


def _canonical_order(coords: torch.Tensor) -> torch.Tensor:
    c = coords.detach().to(device="cpu", dtype=torch.int64).tolist()
    return torch.tensor(sorted(range(len(c)), key=lambda i: tuple(c[i])), dtype=torch.long, device=coords.device)


def _coords_seed(coords: torch.Tensor) -> torch.Tensor:
    """Stable pseudo-random +/-1 carrier in canonical coordinate order."""
    order = _canonical_order(coords).detach().cpu().tolist()
    c = coords.detach().to(device="cpu", dtype=torch.int64).tolist()
    vals = []
    for index in order:
        xyz = c[index]
        digest = hashlib.blake2s(",".join(map(str, xyz)).encode(), digest_size=1).digest()[0]
        vals.append(1.0 if digest & 1 else -1.0)
    return torch.tensor(vals, dtype=torch.float32, device=coords.device)


def embed_material_watermark(
    coords: torch.Tensor,
    attrs: torch.Tensor,
    layout: Mapping[str, Any],
    payload: WatermarkPayload,
    *,
    strength: float = 0.006,
    min_votes_per_bit: int = 8,
) -> torch.Tensor:
    """Return a copy of voxel attrs with a subtle roughness-channel mark.

    Vertices and faces are never touched. The carrier is coordinate-derived so
    reordering sparse rows does not change extraction. This baseline requires
    enough active voxels and is not claimed robust to remeshing or texture loss.
    """
    if "roughness" not in layout:
        raise ValueError("material layout has no roughness channel")
    sl = layout["roughness"]
    if not isinstance(sl, slice) or sl.stop - sl.start != 1:
        raise ValueError("roughness layout must identify one channel")
    bits = _payload_bits(payload.pack())
    per_bit = attrs.shape[0] // len(bits)
    if per_bit < min_votes_per_bit:
        raise ValueError(
            f"not enough active material voxels ({attrs.shape[0]}); "
            f"need at least {len(bits) * min_votes_per_bit}"
        )
    out = attrs.clone()
    order = _canonical_order(coords).to(attrs.device)
    canonical = out.index_select(0, order)
    carrier = _coords_seed(coords).to(device=attrs.device, dtype=attrs.dtype)
    for i, bit in enumerate(bits):
        lo, hi = i * per_bit, (i + 1) * per_bit
        symbol = 1.0 if bit else -1.0
        canonical[lo:hi, sl.start] += (strength * symbol * carrier[lo:hi])
    out.index_copy_(0, order, canonical)
    return out


def extract_material_watermark(
    coords: torch.Tensor,
    attrs: torch.Tensor,
    layout: Mapping[str, Any],
    *,
    strength: float = 0.006,
    min_votes_per_bit: int = 8,
    threshold: float = 0.15,
) -> WatermarkPayload:
    """Decode a marked native sparse material voxel representation."""
    if "roughness" not in layout:
        raise ValueError("material layout has no roughness channel")
    sl = layout["roughness"]
    carrier = _coords_seed(coords).to(device=attrs.device, dtype=attrs.dtype)
    bit_count = _FRAME_SIZE * 8

    def read_bits(n: int) -> list[int]:
        per_bit = attrs.shape[0] // n
        if per_bit < min_votes_per_bit:
            raise ValueError("insufficient voxels for reliable extraction")
        values = []
        order = _canonical_order(coords).to(attrs.device)
        channel = attrs.index_select(0, order)[:, sl.start]
        # Remove the channel DC component; spread-spectrum correlation remains.
        channel = channel - channel.mean()
        for i in range(n):
            lo, hi = i * per_bit, (i + 1) * per_bit
            score = (channel[lo:hi] * carrier[lo:hi]).mean().item() / strength
            if abs(score) < threshold:
                raise ValueError("watermark signal below detection threshold")
            values.append(1 if score > 0 else 0)
        return values

    def bits_to_bytes(bits: list[int]) -> bytes:
        return bytes(sum(bits[i + j] << (7 - j) for j in range(8)) for i in range(0, len(bits), 8))

    raw = bits_to_bytes(read_bits(bit_count))
    return WatermarkPayload.unpack(raw)
