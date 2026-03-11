#!/bin/bash
#
# Smoke test: run 2 training iterations + 1 validation for every
# (algo, env, lr, schedule) combination in the sweep grid.
# Runs locally (no SLURM), exits on first failure.
#
# Usage:
#   bash experiments/smoke_test.sh          # run on auto-detected device
#   bash experiments/smoke_test.sh cuda     # force cuda
#   bash experiments/smoke_test.sh cpu      # force cpu

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DEVICE="${1:-auto}"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

ALGORITHMS=(
    "TBGFlowNet"
    "ModifiedTBGFlowNet"
    "LogPartitionVarianceGFlowNet"
    "ModifiedLogPartitionVarianceGFlowNet"
)
ENVIRONMENTS=("original" "cosine" "bitwise_xor" "multiplicative_coprime")

# LR + schedule combos matching launch_sweep.sh
LR_SCHEDULE_COMBOS=(
    "1e-3 --cosine-schedule"
    "1e-4 --no-cosine-schedule"
)

BETA2_VALUES=("0.9999")
GRAD_CLIP_VALUES=("1.0")

total=0
passed=0
failed=0

for algo in "${ALGORITHMS[@]}"; do
    for env in "${ENVIRONMENTS[@]}"; do
        for lr_sched in "${LR_SCHEDULE_COMBOS[@]}"; do
            read -r lr sched_flag <<< "$lr_sched"
            for beta2 in "${BETA2_VALUES[@]}"; do
                for grad_clip in "${GRAD_CLIP_VALUES[@]}"; do
                    total=$((total + 1))
                    tag="${algo}/${env}/lr${lr}/b${beta2}/gc${grad_clip}/${sched_flag}"
                    echo -n "[$total] $tag ... "

                    if python3 -u -m lir.gflownet.tb_normalize \
                        --device "$DEVICE" \
                        --algos "$algo" \
                        --envs "$env" \
                        --n_iterations 1 \
                        --batch-size 16 \
                        --lr "$lr" \
                        --lr-logz "$lr" \
                        --beta2 "$beta2" \
                        --grad-clip "$grad_clip" \
                        --loss-clamp 100.0 \
                        --replay-capacity 100 \
                        --replay-batch-frac 0.5 \
                        --n-seeds 1 \
                        --output-dir "$TMPDIR" \
                        "$sched_flag" \
                        > "$TMPDIR/stdout.log" 2>&1
                    then
                        echo "OK"
                        passed=$((passed + 1))
                    else
                        echo "FAIL"
                        echo "--- stderr/stdout ---"
                        tail -20 "$TMPDIR/stdout.log"
                        echo "---------------------"
                        failed=$((failed + 1))
                    fi
                    # Clean up per-run output so checkpoint dirs don't collide.
                    rm -rf "$TMPDIR"/*
                done
            done
        done
    done
done

echo ""
echo "=== Smoke test complete ==="
echo "Total: $total  Passed: $passed  Failed: $failed"

if [[ $failed -gt 0 ]]; then
    exit 1
fi
