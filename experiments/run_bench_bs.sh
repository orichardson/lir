#!/bin/bash
#SBATCH --job-name=bench-bs
#SBATCH --output=experiments/logs/bench_bs_%j.log
#SBATCH --error=experiments/logs/bench_bs_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=unkillable

set -euo pipefail

_SAVED_ARGS=("$@")
set --
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate lir
set -- "${_SAVED_ARGS[@]}"
unset _SAVED_ARGS

export CUBLAS_WORKSPACE_CONFIG=:4096:8

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
python3 -u experiments/bench_batchsize.py
