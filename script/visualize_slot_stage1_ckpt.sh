#!/bin/bash
# Standalone unified viz for one or more saved slot-stage1 checkpoints.
# By default processes ALL saved steps in each run dir, optionally at
# multiple t values. Output filename includes step + t so nothing
# overwrites: <run>/slot_unified_eval/step_NNNNNNN.tNNN.png
#
# Distributes runs across GPUs in waves of $N_GPUS (rounds-robin).
#
# Usage:
#   # Single run, ALL steps, default t=0.5
#   bash script/visualize_slot_stage1_ckpt.sh <run_dir>
#
#   # Multiple runs in parallel (one per GPU)
#   bash script/visualize_slot_stage1_ckpt.sh <run1> <run2> <run3> <run4>
#
#   # Multiple t values per step (one file per (step, t))
#   T="0.25 0.5 0.75 0.9"  bash script/visualize_slot_stage1_ckpt.sh <run>
#
#   # Last N steps only (faster preview)
#   LAST_N=3  bash script/visualize_slot_stage1_ckpt.sh <run>
#
#   # Pin to one step instead of all
#   STEP=50000  bash script/visualize_slot_stage1_ckpt.sh <run>
#
#   # Larger sample grid / more sampling steps
#   N_SAMPLES=32 NUM_STEPS=100  bash script/visualize_slot_stage1_ckpt.sh <run>
#
#   # GPU subset
#   GPUS=0,1  bash script/visualize_slot_stage1_ckpt.sh <run1> <run2>
#
#   # Re-run only missing files
#   SKIP_EXISTING=1  bash script/visualize_slot_stage1_ckpt.sh <run>

set -u

GPUS=${GPUS:-"0,1,2,3"}
IFS=',' read -ra GPU_ARR <<< "$GPUS"
N_GPUS=${#GPU_ARR[@]}

STEP=${STEP:-}
LAST_N=${LAST_N:-}
N_SAMPLES=${N_SAMPLES:-16}
T=${T:-"0.5"}
NUM_STEPS=${NUM_STEPS:-50}
SKIP_EXISTING=${SKIP_EXISTING:-}

if [ $# -eq 0 ]; then
    sed -n '2,32p' "$0"
    exit 1
fi

RUN_DIRS=("$@")
N_RUNS=${#RUN_DIRS[@]}

LOG_DIR="/tmp/viz_slot_stage1_$$"
mkdir -p "$LOG_DIR"

echo "GPUs available: $GPUS  ($N_GPUS)"
echo "Runs: $N_RUNS   |   t values: $T   |   N samples: $N_SAMPLES"
[ -n "$STEP"   ] && echo "Step (pinned): $STEP"
[ -n "$LAST_N" ] && echo "Last-N steps: $LAST_N"
echo "Logs: $LOG_DIR/"
echo

run_viz() {
    local gpu=$1
    local run_dir=$2
    local logf=$3

    local args=(
        --run_dir "$run_dir"
        --device "cuda:0"
        --n_samples "$N_SAMPLES"
        --num_sampling_steps "$NUM_STEPS"
        --dit_attn_t $T            # intentionally unquoted: nargs='+'
    )
    if [ -n "$STEP"   ]; then args+=(--step "$STEP");   fi
    if [ -n "$LAST_N" ]; then args+=(--last_n "$LAST_N"); fi
    if [ -n "$SKIP_EXISTING" ]; then args+=(--skip_existing); fi

    CUDA_VISIBLE_DEVICES="$gpu" python src/visualize_slot_stage1_ckpt.py \
        "${args[@]}" > "$logf" 2>&1
}

i=0
wave=0
FAILED=()
while [ $i -lt $N_RUNS ]; do
    wave=$((wave + 1))
    echo "─── Wave $wave ───"
    BATCH_PIDS=()
    BATCH_INDICES=()
    BATCH_DIRS=()

    for j in $(seq 0 $((N_GPUS - 1))); do
        if [ $i -ge $N_RUNS ]; then break; fi
        GPU=${GPU_ARR[$j]}
        RUN="${RUN_DIRS[$i]}"
        LOGF="$LOG_DIR/job_${i}.log"
        printf "  [job %d] GPU %s → %s   (log: %s)\n" "$i" "$GPU" "$RUN" "$LOGF"
        run_viz "$GPU" "$RUN" "$LOGF" &
        BATCH_PIDS+=($!)
        BATCH_INDICES+=("$i")
        BATCH_DIRS+=("$RUN")
        i=$((i + 1))
    done

    for k in "${!BATCH_PIDS[@]}"; do
        idx=${BATCH_INDICES[$k]}
        rd=${BATCH_DIRS[$k]}
        if wait "${BATCH_PIDS[$k]}"; then
            done_line=$(grep -E "^\[viz\] done" "$LOG_DIR/job_${idx}.log" | tail -1)
            saved_line=$(grep -E "→  /" "$LOG_DIR/job_${idx}.log" | wc -l)
            echo "  [job $idx] OK   ${done_line}  (${saved_line} files written)"
        else
            echo "  [job $idx] FAIL  — tail of log:"
            tail -n 8 "$LOG_DIR/job_${idx}.log" | sed 's/^/      /'
            FAILED+=("$idx:$rd")
        fi
    done
    echo
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed jobs (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do echo "  $f"; done
    echo "Inspect logs in: $LOG_DIR/"
    exit 1
fi

echo "All $N_RUNS run(s) visualized successfully."
