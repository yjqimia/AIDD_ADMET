#!/usr/bin/env bash
# Run in the local motil-repro conda environment, not the cloud CPU environment.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
study_dir="$repo_dir/.local/motil"
export PYTHONDONTWRITEBYTECODE=1
if [[ -z ${CONDA_PREFIX:-} && -z ${VIRTUAL_ENV:-} ]]; then
    echo 'Activate a dedicated Python 3.9 conda/venv environment first.' >&2
    exit 1
fi
python -c 'import sys; assert sys.version_info[:2] == (3, 9), "Use Python 3.9 (recommended 3.9.25)"'
command -v nvidia-smi > /dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
setup_dir="$study_dir/setup/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$setup_dir"
exec > >(tee "$setup_dir/setup.log") 2>&1
printf 'Setup log: %s/setup.log\n' "$setup_dir"
run() {
    printf '\n$ '
    printf '%q ' "$@"
    printf '\n'
    "$@"
}
run python --version
run nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
run python -m pip install 'torch==2.2.2+cu121' --index-url https://download.pytorch.org/whl/cu121
run python -m pip install -r "$repo_dir/scripts/requirements-motil-local.txt"
# Official binary wheels match torch 2.2.x / CUDA 12.1; no local nvcc build required.
run python -m pip install --no-index --only-binary=:all: 'torch-scatter==2.1.2+pt22cu121' \
    --find-links https://data.pyg.org/whl/torch-2.2.0+cu121.html
run python -c "import torch, torch_scatter, torch_geometric, rdkit, sklearn, tkinter; assert torch.cuda.is_available(), 'CUDA unavailable; do not start the GPU experiment'; x=torch.randn(512,512,device='cuda:0'); y=x@x; torch.cuda.synchronize(); assert torch.isfinite(y).all().item(); print('GPU:',torch.cuda.get_device_name(0)); print('torch:',torch.__version__,'CUDA runtime:',torch.version.cuda,'scatter:',torch_scatter.__version__,'PyG:',torch_geometric.__version__,'RDKit:',rdkit.__version__,'sklearn:',sklearn.__version__)"
run python -c "import torch; from torch_scatter import scatter_add; x=torch.ones(4,device='cuda:0'); i=torch.tensor([0,0,1,1],device='cuda:0'); y=scatter_add(x,i,dim=0); assert torch.equal(y,torch.tensor([2.,2.],device='cuda:0')); print('CUDA scatter kernel OK')"
upstream_commit=2a8f006d1b72d17e5bc6759e72f19b21f2b5aa64
if [[ ! -e "$study_dir/MotiL" ]]; then
    run git clone https://github.com/Young0222/MotiL.git "$study_dir/MotiL"
    run git -C "$study_dir/MotiL" checkout --detach "$upstream_commit"
fi
if [[ $(git -C "$study_dir/MotiL" rev-parse HEAD) != "$upstream_commit" ]]; then
    echo 'Existing author checkout has another HEAD; preserve it and inspect before proceeding.' >&2
    exit 1
fi
run git -C "$study_dir/MotiL" diff --exit-code HEAD
cd "$study_dir/MotiL/MotiL_micromolecule"
run python train.py --help
python -m pip freeze > "$setup_dir/environment.freeze.txt"
printf '%s\n' "$setup_dir" > "$study_dir/last_setup_path.txt"
printf '\nSetup complete. Log: %s/setup.log\n' "$setup_dir"
