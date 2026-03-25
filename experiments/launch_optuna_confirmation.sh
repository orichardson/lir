#!/bin/bash
#
# Launch Phase 2: Confirmation runs for top-3 HP configs per (algo, env).
#
# 16 SLURM jobs: one per (algorithm, environment) pair, running in parallel.
# Each job runs 3 configs x 5 seeds x 4000 iterations (~80 min on L40S).
# Total: 4 algos x 4 envs x 3 configs x 5 seeds = 240 runs.
#
# Prerequisites: Phase 1 (launch_optuna_sweep.sh) must be complete.
#
# Usage:
#   bash experiments/launch_optuna_confirmation.sh            # submit all 16 jobs
#   bash experiments/launch_optuna_confirmation.sh --dry-run  # print commands only

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p experiments/logs

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# --- Cluster config (edit for your environment) ---
PARTITION="long"
TIME="18:00:00"
# MAIL_USER="you@example.com"

# --- Experiment grid ---
ALGORITHMS=(
    "TBGFlowNet"
    "ModifiedTBGFlowNet"
    "LogPartitionVarianceGFlowNet"
    "ModifiedLogPartitionVarianceGFlowNet"
)
ENVIRONMENTS=(
    "original"
    "cosine"
    "bitwise_xor"
    "multiplicative_coprime"
)

TOP_K=3
N_SEEDS=5
N_ITERATIONS=4000

job_count=0

for algo in "${ALGORITHMS[@]}"; do
    case "$algo" in
        TBGFlowNet)                              atag="TB" ;;
        ModifiedTBGFlowNet)                      atag="ModTB" ;;
        LogPartitionVarianceGFlowNet)            atag="LogPV" ;;
        ModifiedLogPartitionVarianceGFlowNet)    atag="ModLP" ;;
        *)                                       atag="${algo:0:6}" ;;
    esac

    for env in "${ENVIRONMENTS[@]}"; do
        case "$env" in
            original)                etag="orig" ;;
            cosine)                  etag="cosi" ;;
            bitwise_xor)             etag="bitw" ;;
            multiplicative_coprime)  etag="mult" ;;
            *)                       etag="${env:0:4}" ;;
        esac

        # Check that the search DB exists for this (algo, env).
        if [[ ! -f "experiments/optuna_results/db/${algo}__${env}.db" ]]; then
            echo "SKIP: ${algo} / ${env} — no search DB found"
            continue
        fi

        job_name="optc-${atag}-${etag}"

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
            confirm
            --algo "$algo"
            --envs "$env"
            --top-k "$TOP_K"
            --n-seeds "$N_SEEDS"
            --n-iterations "$N_ITERATIONS"
        )

        if $DRY_RUN; then
            echo "${cmd[*]}"
        else
            "${cmd[@]}"
        fi
        job_count=$((job_count + 1))
    done
done

echo ""
echo "Launched $job_count confirmation jobs (one per algo x env pair)."
echo "Results will be written to: experiments/optuna_results/confirm/"
