# Optional material watermarking

TRELLIS.2 can optionally add a creator and generation-time payload to the
linear roughness channel of generated sparse material voxels. This first
implementation uses a coordinate-keyed spread-spectrum carrier. It leaves
vertices and faces untouched and leaves base color, metallic, and alpha
channels unchanged. The watermark is disabled unless `watermark_creator` is
provided to `pipeline.run`.

```python
meshes, (shape_slat, tex_slat, resolution) = pipeline.run(
    image,
    return_latent=True,
    watermark_creator="Example Creator",
)
```

The timestamp is generated in UTC. To extract from TRELLIS.2's native
material-voxel representation:

```python
from trellis2.watermark import extract_material_watermark

mesh = meshes[0]
payload = extract_material_watermark(mesh.coords, mesh.attrs, mesh.layout)
print(payload.creator, payload.created_at, payload.method)
```

The mark resides in material roughness, not in `shape_slat` or `tex_slat`, so
the returned latents are not themselves marked. Re-export through O-Voxel may
resample or weaken the signal. Texture removal removes the signal entirely.
This initial spread-spectrum baseline is not trained with DistSeal and does
not claim robustness to remeshing, decimation, texture edits, arbitrary mesh
formats, or render transformations. DistSeal's learned image embedder and
extractor cannot be reused directly for 3D materials; a 3D training corpus and
evaluation protocol are required to train and validate those properties. No
checkpoint is included.

Run deterministic unit tests from the repository root after installing the
existing TRELLIS.2 Python dependencies:

```bash
PYTHONPATH=TRELLIS.2 python -m unittest discover -s TRELLIS.2/tests -v
```
