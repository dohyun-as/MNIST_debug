#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  Sudoku 288×288 — DiT + ViT encoder (multi-resolution: 9×9, 3×3, 1×1)
# ──────────────────────────────────────────────────────────────────
#
#  DiT backbone (JiT-style) for MNIST-Sudoku generation.
#  Encoder: ViT with multi-resolution grid encoding:
#    - 9×9: per-cell (32×32 each)
#    - 3×3: per-block (96×96 each, 3×3 sudoku blocks)
#    - 1×1: global (288×288)
#  Level drop enabled for multi-resolution training.
#
#  Usage:
#    bash script/train_sudoku_dit_multires.sh
#    GPUS=0,1,2,3 bash script/train_sudoku_dit_multires.sh

set -e


# ── GPU / batch config ──
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

BATCH_PER_GPU=32
GRAD_ACCUM=8

echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS), batch/gpu=$BATCH_PER_GPU, accum=$GRAD_ACCUM, effective=$((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))"

SRM_CONFIG="./config/sudoku_config.json"
CLASSIFIER="/workspace/NAS/sr_diffusion/project/MNIST_debug/datasets/mnist_sudoku/mnist_classifier.pth"

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

accelerate launch \
    --main_process_port $PORT \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    src/main_sudoku_dit.py \
    --backbone dit \
    --output_dir runs/sudoku/dit_multires_9x3x1 \
    --sudoku_config "$SRM_CONFIG" \
    --classifier_pth "$CLASSIFIER" \
    --image_size 288 \
    --in_channels 1 \
    --cond_in_channels 1 \
    --level_sizes 9 3 1 \
    --feat_channels 256 \
    --depth_per_level 2 \
    --encoder_type vit \
    --vit_patch_size 4 \
    --vit_depth 4 \
    --vit_num_heads 4 \
    --vit_mlp_ratio 4.0 \
    --vit_use_cnn_stem \
    --vit_cnn_stem_reduction 4 \
    --dit_patch_size 16 \
    --dit_hidden_size 512 \
    --dit_n_heads 8 \
    --dit_n_blocks 8 \
    --dit_mlp_ratio 4.0 \
    --dit_dropout 0.0 \
    --dit_bottleneck_dim 128 \
    --dit_in_context_len 32 \
    --dit_in_context_start 4 \
    --use_flow_matching \
    --flow_P_mean -0.8 \
    --flow_P_std 0.8 \
    --flow_t_eps 0.05 \
    --flow_noise_scale 1.0 \
    --flow_sampling_method euler \
    --max_train_steps 50000 \
    --batch_size $BATCH_PER_GPU \
    --lr 2e-5 \
    --weight_decay 0.05 \
    --warmup_steps 5000 \
    --max_grad_norm 3.0 \
    --grad_accum_steps $GRAD_ACCUM \
    --mixed_precision fp16 \
    --ema_decay 0 \
    --uncond_drop_prob 0.1 \
    --level_drop \
    --min_keep_levels 1 \
    --level_drop_after_steps 10000 \
    --guidance_scale 3.0 \
    --log_every 100 \
    --save_every 10000 \
    --eval_every 1500 \
    --eval_num_steps 50 \
    --eval_num_samples 81 \
    --num_workers 4 \
    --seed 42 \
    --use_fsq \
    --fsq_levels 8 8 8 8 \
    --mae_mask_ratio 0.75 \
