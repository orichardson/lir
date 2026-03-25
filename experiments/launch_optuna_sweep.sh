#!/bin/bash
#
# Launch Phase 1: Optuna HP search (4 SLURM jobs, one per algorithm).
#
# Each job searches all 4 environments in parallel on one GPU.
# Trials use bs=1024, 2000 iterations, 1 seed (~8 min/trial on L40S).
# 50 trials x 4 envs = 200 trials per job; parallel envs ~4x faster.
#
# Usage:
#   bash experiments/launch_optuna_sweep.sh            # submit all 4 jobs
#   bash experiments/launch_optuna_sweep.sh --dry-run  # print commands only
#
# After all jobs finish:
#   python experiments/optuna_sweep.py summary --all
#   bash experiments/launch_optuna_confirmation.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p experiments/optuna_results/db experiments/logs

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# --- Cluster config (edit for your environment) ---
PARTITION="long"
TIME="48:00:00"
# MAIL_USER="you@example.com"

# --- Experiment grid ---
ALGORITHMS=(
    "TBGFlowNet"
    "ModifiedTBGFlowNet"
    "LogPartitionVarianceGFlowNet"
    "ModifiedLogPartitionVarianceGFlowNet"
)
ENVIRONMENTS="original cosine bitwise_xor multiplicative_coprime"

N_TRIALS=50

job_count=0

for algo in "${ALGORITHMS[@]}"; do
    # Short tag for job name.
    case "$algo" in
        TBGFlowNet)                              atag="TB" ;;
        ModifiedTBGFlowNet)                      atag="ModTB" ;;
        LogPartitionVarianceGFlowNet)            atag="LogPV" ;;
        ModifiedLogPartitionVarianceGFlowNet)    atag="ModLP" ;;
        *)                                       atag="${algo:0:6}" ;;
    esac

    job_name="opt-${atag}"

    cmd=(
        sbatch
        --job-name="$job_name"
        --output="experiments/logs/%x_%j.log"
        --error="experiments/logs/%x_%j.err"
        --time="$TIME"
        --gres=gpu:1
        --cpus-per-task=2
        --mem=16G
        --partition="$PARTITION"
        --mail-type=FAIL
        ${MAIL_USER:+--mail-user="$MAIL_USER"}
        experiments/run_optuna.sh
        search
        --algo "$algo"
        --envs $ENVIRONMENTS
        --n-trials "$N_TRIALS"
        --parallel-envs
    )

    if $DRY_RUN; then
        echo "${cmd[*]}"
    else
        "${cmd[@]}"
    fi
    job_count=$((job_count + 1))
done

echo ""
echo "Launched $job_count Optuna search jobs (each covers 4 environments)."
echo "Results will be written to: experiments/optuna_results/"
echo "After all jobs finish, run:"
echo "  python experiments/optuna_sweep.py summary --all"
echo "  bash experiments/launch_optuna_confirmation.sh"
