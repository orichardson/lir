#!/bin/bash
#SBATCH --job-name=gfn-bench
#SBATCH --output=experiments/logs/%x_%j.log
#SBATCH --error=experiments/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=long
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=joseph@viviano.ca
#
# Run a single (env, algo, hyperparameter) slice of the benchmark.
# Designed to be called by launch_sweep.sh or directly.
#
# Usage:
#   sbatch run_single.sh --envs original --algos TBGFlowNet --lr 1e-3 --beta2 0.999
#   bash  run_single.sh --envs original --algos TBGFlowNet --lr 1e-3 --beta2 0.999

set -euo pipefail

# --- Environment setup ---
# Uncomment/modify whichever applies to your cluster:

# Option A: conda (default)
# Save positional args — conda's shell hook can misparse them on some nodes.
_SAVED_ARGS=("$@")
set --
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate lir
set -- "${_SAVED_ARGS[@]}"
unset _SAVED_ARGS

# Option B: module + conda
# module load cuda/12.1
# source activate torchgfn

# Option C: venv
# source /path/to/venv/bin/activate

# --- Deterministic CuBLAS (needed when torch.use_deterministic_algorithms is on) ---
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# --- Run ---
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p experiments/logs

echo "=== GFlowNet Benchmark ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU:  $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")')"
echo "Args: $*"
echo ""

python3 -u -m lir.gflownet.tb_normalize \
    --device cuda \
    "$@"

echo ""
echo "=== Done ==="
