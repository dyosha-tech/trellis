"""Optional material-space watermarking for TRELLIS.2 assets.

This first implementation marks TRELLIS' sparse material voxel attributes. It
does not alter vertex positions or faces. The watermark is a spread-spectrum
prototype; robust learned DistSeal weights must be trained for a deployment
before making robustness claims.
"""

from .material import (
    WatermarkPayload,
    embed_material_watermark,
    extract_material_watermark,
)

__all__ = [
    "WatermarkPayload",
    "embed_material_watermark",
    "extract_material_watermark",
]
