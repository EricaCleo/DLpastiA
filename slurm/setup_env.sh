#!/bin/bash
# Run ONCE on the LOGIN NODE (it has internet). Do NOT run through sbatch, and do NOT pip-install in sbatch:
# compute nodes have no internet, and GPU time is billed while waiting for downloads.
#
#   cd /data/home/<ACCOUNT>/run/DLpastiA && bash slurm/setup_env.sh
set -e

ENV_NAME="${ENV_NAME:-dlpastia}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# PyTorch wheels must match the GPU driver. A800 / H200 need a CUDA 12.x build.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
# If PyPI is slow from the cluster, set e.g. PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
PIP_INDEX="${PIP_INDEX:-}"

echo "Env: $ENV_NAME | Python: $PYTHON_VERSION | Project: $PROJECT_DIR | Host: $(hostname)"
cd "$PROJECT_DIR"

source /etc/profile || true
[ -f ~/.bashrc ] && source ~/.bashrc || true
if command -v module >/dev/null 2>&1; then
  for m in miniforge3 miniforge miniconda3 miniconda anaconda3 anaconda conda; do
    module load "$m" >/dev/null 2>&1 && echo "Loaded module: $m" && break
  done
fi
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found. Try: module avail conda"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env $ENV_NAME already exists."
else
  conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
fi
conda activate "$ENV_NAME"
export PYTHONNOUSERSITE=1   # ignore ~/.local packages that can shadow the conda env
python --version

# 1) PyTorch (CUDA build), 2) everything else
python -m pip install torch torchvision --index-url "$TORCH_INDEX"
if [ -n "$PIP_INDEX" ]; then python -m pip install -r requirements.txt -i "$PIP_INDEX"
else python -m pip install -r requirements.txt; fi

# Cache the LPIPS / backbone weights NOW (login node has internet, compute nodes do not)
python -c "import lpips; lpips.LPIPS(net='alex'); print('LPIPS weights cached')"

echo "Done. Test with: conda activate $ENV_NAME && python -c 'import torch; print(torch.__version__)'"
echo "(torch.cuda.is_available() is False on the login node; that is expected. Check it inside a GPU job.)"
