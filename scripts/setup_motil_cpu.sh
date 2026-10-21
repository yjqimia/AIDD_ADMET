#!/usr/bin/env bash
# Compatibility environment for this CPU cloud instance, not the paper's exact environment.
set -euo pipefail
study_dir=${MOTIL_STUDY_DIR:-/workspace/motil-work}
mkdir -p "$study_dir/logs"
export UV_CACHE_DIR="$study_dir/cache"
export UV_PYTHON_INSTALL_DIR="$study_dir/python"
export PYTHONDONTWRITEBYTECODE=1
upstream_commit=2a8f006d1b72d17e5bc6759e72f19b21f2b5aa64
if [[ ! -d "$study_dir/MotiL/.git" ]]; then
    git clone https://github.com/Young0222/MotiL.git "$study_dir/MotiL"
fi
if [[ $(git -C "$study_dir/MotiL" rev-parse HEAD) != "$upstream_commit" ]]; then
    echo "Upstream HEAD differs from the studied commit; preserve local work and inspect before proceeding." >&2
    exit 1
fi
if [[ ! -x "$study_dir/venv/bin/python" ]]; then
    uv venv --python 3.9.25 "$study_dir/venv"
fi
uv pip install --python "$study_dir/venv/bin/python" \
    --index-url https://download.pytorch.org/whl/cpu 'torch==2.2.2+cpu'
uv pip install --python "$study_dir/venv/bin/python" \
    'numpy==1.23.5' 'scipy==1.9.3' 'scikit-learn==0.24.2' \
    'rdkit==2022.9.1' 'torch-geometric==2.3.0' 'tensorboardX==2.6.2.2' \
    'protobuf==3.20.3' 'pandas==1.3.5' 'matplotlib==3.7.5' \
    'openpyxl==3.1.5' 'Unidecode==1.3.7' 'tqdm==4.67.1' \
    'setuptools==75.3.2' 'wheel==0.45.1'
# Build against the installed CPU torch; no disabled TLS or substituted package stubs.
MAX_JOBS=2 uv pip install --python "$study_dir/venv/bin/python" \
    --no-build-isolation 'torch-scatter==2.1.2'
"$study_dir/venv/bin/python" -c \
    "import torch, torch_scatter, rdkit, torch_geometric, sklearn, tkinter; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('rdkit', rdkit.__version__, 'pyg', torch_geometric.__version__, 'sklearn', sklearn.__version__, 'scatter', torch_scatter.__version__)"
uv pip freeze --python "$study_dir/venv/bin/python" > "$study_dir/logs/environment.freeze.txt"
cd "$study_dir/MotiL/MotiL_micromolecule"
"$study_dir/venv/bin/python" train.py --help > "$study_dir/logs/train_help.txt"
