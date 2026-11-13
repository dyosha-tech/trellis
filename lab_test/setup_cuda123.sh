#!/usr/bin/env bash
set -euo pipefail

# Install TRELLIS.2 dependencies in a private Python venv. No sudo and no
# changes to the lab's system Conda installation are made.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${TRELLIS2_VENV:-$HOME/.venvs/trellis2}"
EXT="${TRELLIS2_EXTENSIONS:-$HOME/trellis2-extensions}"

export CUDA_HOME=/usr/local/cuda-12.3
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.9"

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "Expected CUDA 12.3 compiler at $CUDA_HOME/bin/nvcc" >&2
    exit 1
fi

python3 - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Expected Python 3.10; got {sys.version.split()[0]}")
PY

mkdir -p "$(dirname "$VENV")" "$EXT"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel packaging ninja

# TRELLIS.2 pins PyTorch 2.6.0/cu124. CUDA 12.3 is used only to compile local
# extensions; PyTorch treats this minor-version difference as a warning.
python -m pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# User-space equivalent of setup.sh --basic. Standard Pillow avoids the
# libjpeg-dev/pillow-simd system-package step, which requires sudo upstream.
python -m pip install imageio imageio-ffmpeg tqdm easydict \
    opencv-python-headless ninja trimesh transformers gradio==6.0.1 \
    tensorboard pandas lpips zstandard pillow kornia timm
python -m pip install \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Match the upstream setup.sh extension versions, keeping build files under
# the user's home directory rather than /tmp or system locations.
git clone --depth 1 --branch v0.4.0 https://github.com/NVlabs/nvdiffrast.git "$EXT/nvdiffrast"
python -m pip install "$EXT/nvdiffrast" --no-build-isolation

git clone --depth 1 --branch renderutils https://github.com/JeffreyXiang/nvdiffrec.git "$EXT/nvdiffrec"
python -m pip install "$EXT/nvdiffrec" --no-build-isolation

git clone --recursive https://github.com/JeffreyXiang/CuMesh.git "$EXT/CuMesh"
python -m pip install "$EXT/CuMesh" --no-build-isolation

git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git "$EXT/FlexGEMM"
python -m pip install "$EXT/FlexGEMM" --no-build-isolation

python -m pip install "$ROOT/o-voxel" --no-build-isolation
MAX_JOBS="${MAX_JOBS:-4}" python -m pip install flash-attn==2.7.3 --no-build-isolation

echo
echo "Environment setup finished. Activate it with:"
echo "source \"$VENV/bin/activate\""
echo "CUDA_HOME=$CUDA_HOME"
