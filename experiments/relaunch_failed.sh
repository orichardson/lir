#!/bin/bash
#
# Relaunch the 79 failed/incomplete sweep combos.
#
# Generated from checkpoint analysis of experiments/sweep_results/.
# Uses the same hyperparameters as launch_sweep.sh.
#
# Usage:
#   bash experiments/relaunch_failed.sh            # submit failed jobs
#   bash experiments/relaunch_failed.sh --dry-run  # print commands only

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p experiments/logs experiments/sweep_results

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# --- Fixed hyperparameters (must match launch_sweep.sh) ---
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
    local algo="$1" env="$2" lr="$3" beta2="$4" grad_clip="$5"
    local atag
    case "$algo" in
        TBGFlowNet)                              atag="TB" ;;
        ModifiedTBGFlowNet)                      atag="ModTB" ;;
        LogPartitionVarianceGFlowNet)            atag="LogPV" ;;
        ModifiedLogPartitionVarianceGFlowNet)    atag="ModLP" ;;
        *)                                       atag="${algo:0:6}" ;;
    esac
    local job_name="gfn-${env:0:4}-${atag}-lr${lr}-b${beta2}-gc${grad_clip}"

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
        --lr-logz "$lr"
        --beta2 "$beta2"
        --grad-clip "$grad_clip"
        --loss-clamp "$LOSS_CLAMP"
        --replay-capacity "$REPLAY_CAPACITY"
        --replay-batch-frac "$REPLAY_BATCH_FRAC"
        --n-seeds "$N_SEEDS"
        --output-dir "$OUTPUT_DIR"
        --cosine-schedule
        --show-progress
    )

    if $DRY_RUN; then
        echo "${cmd[*]}"
    else
        "${cmd[@]}"
    fi
}

job_count=0

# --- All 48 ModifiedTBGFlowNet combos (all failed due to torchgfn API bug) ---
for env in original cosine bitwise_xor multiplicative_coprime; do
    for lr in 1e-3 3e-4 1e-4; do
        for beta2 in 0.999 0.9999; do
            for gc in 1.0 0.1; do
                submit_job ModifiedTBGFlowNet "$env" "$lr" "$beta2" "$gc"
                job_count=$((job_count + 1))
            done
        done
    done
done

# --- 19 failed TBGFlowNet combos ---
for combo in \
    "cosine 1e-3 0.9999 1.0" \
    "cosine 1e-3 0.9999 0.1" \
    "cosine 3e-4 0.999 1.0" \
    "cosine 3e-4 0.999 0.1" \
    "cosine 1e-4 0.999 1.0" \
    "cosine 1e-4 0.9999 1.0" \
    "cosine 1e-4 0.9999 0.1" \
    "bitwise_xor 1e-3 0.999 0.1" \
    "bitwise_xor 1e-3 0.9999 1.0" \
    "bitwise_xor 3e-4 0.999 1.0" \
    "bitwise_xor 3e-4 0.9999 1.0" \
    "bitwise_xor 1e-4 0.999 0.1" \
    "bitwise_xor 1e-4 0.9999 1.0" \
    "multiplicative_coprime 1e-3 0.999 1.0" \
    "multiplicative_coprime 1e-3 0.9999 1.0" \
    "multiplicative_coprime 1e-3 0.9999 0.1" \
    "multiplicative_coprime 3e-4 0.999 1.0" \
    "multiplicative_coprime 3e-4 0.9999 0.1" \
    "multiplicative_coprime 1e-4 0.9999 0.1" \
; do
    read -r env lr beta2 gc <<< "$combo"
    submit_job TBGFlowNet "$env" "$lr" "$beta2" "$gc"
    job_count=$((job_count + 1))
done

# --- 5 failed LogPartitionVarianceGFlowNet combos ---
for combo in \
    "original 3e-4 0.9999 0.1" \
    "original 1e-4 0.999 1.0" \
    "cosine 1e-4 0.999 1.0" \
    "cosine 1e-4 0.9999 1.0" \
    "bitwise_xor 3e-4 0.999 1.0" \
; do
    read -r env lr beta2 gc <<< "$combo"
    submit_job LogPartitionVarianceGFlowNet "$env" "$lr" "$beta2" "$gc"
    job_count=$((job_count + 1))
done

# --- 7 failed ModifiedLogPartitionVarianceGFlowNet combos ---
for combo in \
    "original 1e-3 0.999 1.0" \
    "original 1e-4 0.999 1.0" \
    "cosine 1e-3 0.999 1.0" \
    "cosine 1e-3 0.9999 1.0" \
    "bitwise_xor 1e-4 0.9999 1.0" \
    "bitwise_xor 1e-4 0.9999 0.1" \
    "multiplicative_coprime 1e-3 0.999 1.0" \
; do
    read -r env lr beta2 gc <<< "$combo"
    submit_job ModifiedLogPartitionVarianceGFlowNet "$env" "$lr" "$beta2" "$gc"
    job_count=$((job_count + 1))
done

echo ""
echo "Relaunched $job_count failed jobs."
echo "Results will be written to: $OUTPUT_DIR/"
echo "After all jobs finish, run:  python experiments/aggregate_results.py"
