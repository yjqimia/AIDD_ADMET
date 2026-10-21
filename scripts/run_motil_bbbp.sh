#!/usr/bin/env bash
# Author's BBBP configuration, with CPU execution and outputs outside upstream checkout.
set -euo pipefail
study_dir=${MOTIL_STUDY_DIR:-/workspace/motil-work}
run_id=${MOTIL_RUN_ID:-bbbp_cpu}
mkdir -p "$study_dir/logs"
if [[ -e "$study_dir/logs/$run_id.txt" ]] || \
   compgen -G "$study_dir/runs/*-finetune/$run_id" > /dev/null; then
    echo "Run $run_id already exists. Choose a new MOTIL_RUN_ID to preserve its outputs." >&2
    exit 1
fi
cd "$study_dir/MotiL/MotiL_micromolecule"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
"$study_dir/venv/bin/python" train.py \
    --data_path ./data/bbbp.csv \
    --exp_id "$run_id" --dataset_type classification \
    --epochs 10 --num_runs 3 --no_cuda --batch_size 16 --seed 2024 \
    --init_lr 1e-5 --final_lr 5e-6 --wd 5e-5 --warmup_epochs 0.0 \
    --split_type scaffold_balanced --exp_name finetune \
    --split_sizes 0.8 0.1 0.1 --step finetune --save_smiles_splits \
    --dump_path "$study_dir/runs" \
    --checkpoint_path ./dumped/pre-train/1-model/original_CMPN_0707_0800_12000th_epoch.pkl \
    2>&1 | tee "$study_dir/logs/$run_id.txt"
