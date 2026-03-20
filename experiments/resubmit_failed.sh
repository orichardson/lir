#!/bin/bash
#
# Resubmit the 24 jobs that failed due to preemption.
# Generated 2026-03-14.
#
# Usage:
#   bash experiments/resubmit_failed.sh            # submit all jobs
#   bash experiments/resubmit_failed.sh --dry-run  # print commands without submitting

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p experiments/logs experiments/sweep_results

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# --- Fixed hyperparameters (same as original sweep) ---
N_ITERATIONS=5000
HEIGHT=24
NDIM=4
BATCH_SIZE=256
REPLAY_CAPACITY=10000
REPLAY_BATCH_FRAC=0.5
LOSS_CLAMP=100.0
N_SEEDS=5
OUTPUT_DIR="experiments/sweep_results"

submit_job() {
    local algo="$1" env="$2" lr="$3" beta2="$4" grad_clip="$5" logz_mult="$6"

    # Short tag for job name
    local atag
    case "$algo" in
        TBGFlowNet)                              atag="TB" ;;
        ModifiedTBGFlowNet)                      atag="ModTB" ;;
        LogPartitionVarianceGFlowNet)            atag="LogPV" ;;
        ModifiedLogPartitionVarianceGFlowNet)    atag="ModLP" ;;
        *)                                       atag="${algo:0:6}" ;;
    esac

    local job_name="gfn-${env:0:4}-${atag}-lr${lr}-b${beta2}-gc${grad_clip}-lz${logz_mult}"

    local cmd=(
        sbatch
        --job-name="$job_name"
        experiments/run_single.sh
        --algos "$algo"
        --envs "$env"
        --n_iterations "$N_ITERATIONS"
        --height "$HEIGHT"
        --ndim "$NDIM"
        --batch-size "$BATCH_SIZE"
        --lr "$lr"
        --lr-logz-multiplier "$logz_mult"
        --beta2 "$beta2"
        --grad-clip "$grad_clip"
        --loss-clamp "$LOSS_CLAMP"
        --replay-capacity "$REPLAY_CAPACITY"
        --replay-batch-frac "$REPLAY_BATCH_FRAC"
        --n-seeds "$N_SEEDS"
        --output-dir "$OUTPUT_DIR"
        --lr-schedule linear
        --show-progress
    )

    if $DRY_RUN; then
        echo "${cmd[*]}"
    else
        "${cmd[@]}"
    fi
}

job_count=0

# --- 24 failed jobs ---
# TBGFlowNet (11 jobs)
submit_job TBGFlowNet original           1e-3  0.999   0.1 100;  job_count=$((job_count + 1))
submit_job TBGFlowNet original           1e-3  0.9999  1.0 100;  job_count=$((job_count + 1))
submit_job TBGFlowNet cosine             1e-4  0.9999  0.1 100;  job_count=$((job_count + 1))
submit_job TBGFlowNet bitwise_xor        1e-3  0.999   1.0 100;  job_count=$((job_count + 1))
submit_job TBGFlowNet bitwise_xor        1e-3  0.999   0.1 10;   job_count=$((job_count + 1))
submit_job TBGFlowNet bitwise_xor        1e-3  0.999   0.1 100;  job_count=$((job_count + 1))
submit_job TBGFlowNet bitwise_xor        1e-3  0.9999  1.0 100;  job_count=$((job_count + 1))
submit_job TBGFlowNet bitwise_xor        3e-4  0.9999  1.0 100;  job_count=$((job_count + 1))
submit_job TBGFlowNet bitwise_xor        1e-4  0.999   0.1 100;  job_count=$((job_count + 1))
submit_job TBGFlowNet multiplicative_coprime 1e-3 0.999  1.0 100; job_count=$((job_count + 1))
submit_job TBGFlowNet multiplicative_coprime 1e-4 0.9999 1.0 100; job_count=$((job_count + 1))

# ModifiedTBGFlowNet (3 jobs)
submit_job ModifiedTBGFlowNet cosine             1e-3  0.9999  1.0 100;  job_count=$((job_count + 1))
submit_job ModifiedTBGFlowNet bitwise_xor        1e-4  0.9999  1.0 10;   job_count=$((job_count + 1))
submit_job ModifiedTBGFlowNet multiplicative_coprime 1e-4 0.999 1.0 100; job_count=$((job_count + 1))

# LogPartitionVarianceGFlowNet (5 jobs)
submit_job LogPartitionVarianceGFlowNet original           1e-3  0.9999  1.0 10;   job_count=$((job_count + 1))
submit_job LogPartitionVarianceGFlowNet bitwise_xor        1e-3  0.999   1.0 10;   job_count=$((job_count + 1))
submit_job LogPartitionVarianceGFlowNet bitwise_xor        1e-4  0.9999  0.1 10;   job_count=$((job_count + 1))
submit_job LogPartitionVarianceGFlowNet multiplicative_coprime 1e-3 0.9999 1.0 100; job_count=$((job_count + 1))
submit_job LogPartitionVarianceGFlowNet multiplicative_coprime 1e-4 0.999  1.0 10;  job_count=$((job_count + 1))

# ModifiedLogPartitionVarianceGFlowNet (5 jobs)
submit_job ModifiedLogPartitionVarianceGFlowNet original  1e-4  0.999   1.0 100;  job_count=$((job_count + 1))
submit_job ModifiedLogPartitionVarianceGFlowNet original  1e-4  0.9999  1.0 10;   job_count=$((job_count + 1))
submit_job ModifiedLogPartitionVarianceGFlowNet cosine    3e-4  0.999   1.0 100;  job_count=$((job_count + 1))
submit_job ModifiedLogPartitionVarianceGFlowNet cosine    3e-4  0.9999  1.0 100;  job_count=$((job_count + 1))
submit_job ModifiedLogPartitionVarianceGFlowNet original  1e-4  0.9999  1.0 100;  job_count=$((job_count + 1))

echo ""
echo "Resubmitted $job_count failed jobs."
echo "Results will be written to: $OUTPUT_DIR/"
