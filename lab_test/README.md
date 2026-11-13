# Linux GPU smoke test (CUDA Toolkit 12.3)

This is a practical smoke test for TRELLIS.2 generation plus the repository's
current **prototype** material watermark. It uses a private Python virtual
environment and does not change system Conda or require `sudo`. CUDA 12.3 is
used to compile local extensions with the upstream PyTorch 2.6.0/cu124 wheel;
PyTorch identifies that minor-version mismatch as a warning, but a given
extension can still fail to build or run. The setup can take a while.

## 1. Put the code on the lab machine

Commit/push the repository with `lab_test/` included, then on the Linux
machine, clone the repository (or use its existing checkout) and switch into
its root directory. If transferring a ZIP instead, extract it into the
repository root so `lab_test/` and `trellis2/` are siblings.

## 2. Create the environment and install dependencies

Run from the TRELLIS.2 repository root:

```bash
bash lab_test/setup_cuda123.sh
source "$HOME/.venvs/trellis2/bin/activate"
export CUDA_HOME=/usr/local/cuda-12.3
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

The script builds TRELLIS.2's CUDA extensions from source. It stores build
checkouts in `$HOME/trellis2-extensions`; rerunning the script will stop if
those directories already exist. If setup stops partway, keep the environment
and output, share the final error, and do not delete directories blindly.

## 3. Confirm imports and run the existing unit tests

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); import flash_attn, nvdiffrast.torch, cumesh, o_voxel'
python -m unittest discover -s tests -v
```

## 4. Generate an unmarked and marked asset, then extract

Choose an image file and a creator label. Use GPU 1 (the second GPU in
`nvidia-smi`, an RTX 6000 Ada) for this command; it avoids the display GPU.
The first run downloads `microsoft/TRELLIS.2-4B` from Hugging Face. The model
is several gigabytes, so make sure the lab permits the download.

```bash
CUDA_VISIBLE_DEVICES=1 python lab_test/run_watermark_smoke_test.py \
  --image /path/to/input.png \
  --creator "Your Name" \
  --output-dir watermark_test_output \
  --pipeline-type 512
```

Success means the script exits with status 0 and `report.json` says
`geometry_and_voxel_coordinates_identical: true` and `watermark_detected:
true`. It writes `unmarked.glb`, `watermarked.glb`, and
`marked_native_voxels.pt` for inspection. The extractor reads the saved native
voxel coordinates/material attributes; it **does not** extract from `.glb` or
other mesh files. GLB export/resampling may weaken or discard this prototype
mark, and edits such as remeshing, decimation, geometry transforms, and texture
removal are not covered by this test. This is not a trained LoRA or a trained
DISTSEAL implementation.
