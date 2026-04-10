#!/bin/bash
# Evaluate CLEVR detection + attribute accuracy on saved checkpoints.
# Multi-GPU via accelerate. Eval set is deterministic per seed.
#
# Usage:
#   # Single GPU
#   bash script/eval_clevr_ckpts.sh <run_dir>
#
#   # Multi-GPU (4 GPUs)
#   GPUS=0,1,2,3 bash script/eval_clevr_ckpts.sh <run_dir>
#
#   # Specific steps, more samples
#   GPUS=0,1 bash script/eval_clevr_ckpts.sh <run_dir> --steps 50000 100000 --num_samples_per_split 50
#
#   # Different guidance scale
#   bash script/eval_clevr_ckpts.sh <run_dir> --guidance_scale 2.0

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

RUN_DIR=${1:?Usage: bash eval_clevr_ckpts.sh <run_dir> [extra args...]}
shift

export CUDA_VISIBLE_DEVICES="${GPUS:-0}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# Pick a random free port to avoid conflicts
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

CLEVR_VAL="../clevr-dataset-gen/output/clevr_256_varied_val/images"

echo "CLEVR eval: GPUs=$CUDA_VISIBLE_DEVICES ($NUM_GPUS), run_dir=$RUN_DIR"

if [ "$NUM_GPUS" -gt 1 ]; then
    accelerate launch \
        --num_processes $NUM_GPUS \
        --multi_gpu \
        --main_process_port $MASTER_PORT \
        "$DIR/eval_clevr_ckpts.py" \
        --run_dir "$RUN_DIR" \
        --val_dir "$CLEVR_VAL" \
        --num_samples_per_split 30 \
        --guidance_scale 3.0 \
        --num_steps 50 \
        --batch_size 32 \
        "$@"
else
    python "$DIR/eval_clevr_ckpts.py" \
        --run_dir "$RUN_DIR" \
        --val_dir "$CLEVR_VAL" \
        --num_samples_per_split 30 \
        --guidance_scale 3.0 \
        --num_steps 50 \
        --batch_size 32 \
        "$@"
fi
