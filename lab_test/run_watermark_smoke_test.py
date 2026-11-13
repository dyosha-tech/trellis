#!/usr/bin/env python3
"""Run TRELLIS.2 twice at the same seed and verify the material watermark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
import o_voxel

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.watermark import extract_material_watermark


def export_glb(mesh, path: Path) -> None:
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1_000_000,
        texture_size=2048,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(str(path), extension_webp=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path, help="Input image")
    parser.add_argument("--creator", required=True, help="Creator name for payload")
    parser.add_argument("--output-dir", type=Path, default=Path("watermark_test_output"))
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pipeline-type", default="512", choices=("512", "1024", "1024_cascade", "1536_cascade"))
    parser.add_argument("--strength", type=float, default=0.006)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot see a CUDA GPU. Activate the TRELLIS.2 environment first.")
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Torch: {torch.__version__}; torch CUDA runtime: {torch.version.cuda}")
    print(f"Loading {args.model}; the first run downloads the model checkpoint.", flush=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    image = Image.open(args.image).convert("RGB")

    # run() seeds Torch internally, so equal seeds give a matched unmarked and
    # marked output. We compare their topology and vertex coordinates below.
    plain_meshes = pipeline.run(image, seed=args.seed, pipeline_type=args.pipeline_type)
    marked_meshes = pipeline.run(
        image,
        seed=args.seed,
        pipeline_type=args.pipeline_type,
        watermark_creator=args.creator,
        watermark_strength=args.strength,
    )
    plain, marked = plain_meshes[0], marked_meshes[0]

    geometry_same = (
        torch.equal(plain.vertices, marked.vertices)
        and torch.equal(plain.faces, marked.faces)
        and torch.equal(plain.coords, marked.coords)
    )
    if not geometry_same:
        raise AssertionError("Matched-seed marked and unmarked geometry/voxel coordinates differ")

    payload = extract_material_watermark(
        marked.coords, marked.attrs, marked.layout, strength=args.strength
    )
    if payload.creator != args.creator:
        raise AssertionError(f"Wrong creator decoded: {payload.creator!r}")

    # Store native O-Voxel attributes: this is what the current extractor
    # accepts. Extracting from the exported GLB is not currently implemented.
    torch.save(
        {
            "coords": marked.coords.detach().cpu(),
            "attrs": marked.attrs.detach().cpu(),
            "layout": marked.layout,
            "payload": payload.__dict__,
        },
        args.output_dir / "marked_native_voxels.pt",
    )
    export_glb(plain, args.output_dir / "unmarked.glb")
    export_glb(marked, args.output_dir / "watermarked.glb")
    report = {
        "model": args.model,
        "input_image": str(args.image.resolve()),
        "seed": args.seed,
        "pipeline_type": args.pipeline_type,
        "geometry_and_voxel_coordinates_identical": geometry_same,
        "watermark_detected": True,
        "decoded_creator": payload.creator,
        "decoded_created_at_utc": payload.created_at,
        "decoded_method": payload.method,
        "watermark_strength": args.strength,
        "extractor_input": "native TRELLIS.2 sparse material voxels",
        "glb_extraction_supported": False,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
