#!/bin/bash
#SBATCH --job-name=optuna-gfn
#SBATCH --output=experiments/logs/%x_%j.log
#SBATCH --error=experiments/logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=long
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=joseph@viviano.ca
#
# SLURM wrapper for optuna_sweep.py (search or confirm mode).
# Activates conda, then forwards all arguments.
#
# Usage:
#   sbatch run_optuna.sh search --algo TBGFlowNet --env original --n-trials 50
#   sbatch run_optuna.sh confirm --algo TBGFlowNet --env original

set -euo pipefail

# --- Environment setup (same as run_single.sh) ---
_SAVED_ARGS=("$@")
set --
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate lir
set -- "${_SAVED_ARGS[@]}"
unset _SAVED_ARGS

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# --- Run ---
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

echo "=== Optuna GFlowNet Sweep ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU:  $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")')"
echo "Args: $*"
echo ""

python3 -u experiments/optuna_sweep.py "$@"

echo ""
echo "=== Done ==="
