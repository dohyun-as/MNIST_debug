#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  Evaluate all saved checkpoints for 256_dit_vit_flow_cont_out16
#  using CLEVR detector + classifier.
#
#  Usage:
#    bash script/run_eval_cont.sh
#    GPUS=0,1 bash script/run_eval_cont.sh
#    GPUS=0 bash script/run_eval_cont.sh --steps 10000 20000
# ──────────────────────────────────────────────────────────────────
set -e

export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

RUN_DIR="runs/clevr/256_dit_vit_flow_cont_out16"
VAL_DIR="../clevr_output/clevr_256_varied_val/images"

echo "=== CLEVR Eval ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS)"
echo "Run dir: $RUN_DIR"
echo "Extra args: $@"

accelerate launch \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    script/eval_clevr_ckpts.py \
    --run_dir "$RUN_DIR" \
    --val_dir "$VAL_DIR" \
    --num_samples_per_split 50 \
    --num_steps 50 \
    --guidance_scale 3.0 \
    --batch_size 16 \
    --seed 42 \
    --use_ema \
    "$@"
