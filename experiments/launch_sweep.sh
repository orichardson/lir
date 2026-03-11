#!/bin/bash
#
# Launch the full experiment sweep as parallel SLURM jobs.
#
# Experiment grid:
#   4 algorithms × 4 environments × 2 beta2 × 2 grad_clip
#   × (4 cosine LRs + 2 linear LRs) = 384 jobs
#
# Each job runs one (algo, env, lr, beta2, grad_clip, schedule) combination
# with 5 seeds internally. All results land in experiments/sweep_results/.
#
# Usage:
#   bash experiments/launch_sweep.sh            # submit all jobs
#   bash experiments/launch_sweep.sh --dry-run  # print commands without submitting
#
# After all jobs finish:
#   python experiments/aggregate_results.py
#
# Estimated runtime per job: ~2-4 hrs on A100 (5000 iters, 5 seeds)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p experiments/logs experiments/sweep_results

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# --- Experiment grid ---
ALGORITHMS=(
    "TBGFlowNet"
    "ModifiedTBGFlowNet"
    "LogPartitionVarianceGFlowNet"
    "ModifiedLogPartitionVarianceGFlowNet"
)
ENVIRONMENTS=("original" "cosine" "bitwise_xor" "multiplicative_coprime")
BETA2_VALUES=("0.999" "0.9999")
GRAD_CLIP_VALUES=("1.0" "0.1")

# Each entry is "lr schedule_flag" where schedule_flag is passed to run_single.sh.
LR_SCHEDULE_COMBOS=(
    "1e-3 --cosine-schedule"
    "3e-4 --cosine-schedule"
    "1e-4 --cosine-schedule"
    "1e-5 --cosine-schedule"
    "1e-4 --no-cosine-schedule"
    "1e-5 --no-cosine-schedule"
)

# --- Fixed hyperparameters ---
N_ITERATIONS=5000
HEIGHT=24
NDIM=4
BATCH_SIZE=256
REPLAY_CAPACITY=10000
REPLAY_BATCH_FRAC=0.5
LOSS_CLAMP=100.0
N_SEEDS=5
OUTPUT_DIR="experiments/sweep_results"

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

    for env in "${ENVIRONMENTS[@]}"; do
        for beta2 in "${BETA2_VALUES[@]}"; do
            for grad_clip in "${GRAD_CLIP_VALUES[@]}"; do
                for lr_sched in "${LR_SCHEDULE_COMBOS[@]}"; do
                    read -r lr sched_flag <<< "$lr_sched"

                    # Tag linear-schedule jobs with "-lin" suffix.
                    sched_suffix=""
                    if [[ "$sched_flag" == "--no-cosine-schedule" ]]; then
                        sched_suffix="-lin"
                    fi
                    job_name="gfn-${env:0:4}-${atag}-lr${lr}-b${beta2}-gc${grad_clip}${sched_suffix}"

                    cmd=(
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
                        "$sched_flag"
                        --show-progress
                    )

                    if $DRY_RUN; then
                        echo "${cmd[*]}"
                    else
                        "${cmd[@]}"
                    fi
                    job_count=$((job_count + 1))
                done
            done
        done
    done
done

echo ""
echo "Launched $job_count jobs."
echo "Results will be written to: $OUTPUT_DIR/"
echo "After all jobs finish, run:  python experiments/aggregate_results.py"
