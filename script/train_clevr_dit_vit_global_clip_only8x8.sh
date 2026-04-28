#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  CLEVR 256×256 — Single-Res (8×8 only) DiT + ViT-Global (CLIP init)
#  with encoder_internal_dim bottleneck
# ──────────────────────────────────────────────────────────────────
#
#  Encoder: vit_global (single-forward) with bottleneck, 8×8 level only
#    image 256×256 → patch 16 → 16×16=256 tokens → ViT-B/16 forward (한 번)
#    → reshape (B, 768, 16, 16)
#    → avg_pool kernel=2 → (B, 768, 8, 8)              ← 8×8 단일 레벨
#    → Conv2d(768→16) bottleneck → (B, 16, 8, 8)
#    → + 8×8 grid_pos_emb
#
#    Output: 64 tokens × 16 dim = 1,024 floats/sample
#    Multi-res variant (1,360 floats) 대비 finest 8×8만 유지.
#    Level drop 도 비활성 (--no_level_drop).
#
#    CLIP init: openai/clip-vit-base-patch16의 patch_embed / pos_emb /
#               12 transformer layers / post_layernorm → internal 768 dim.
#               Conv2d(768→16)는 random init (bottleneck head).
#               pos_emb는 14×14 → 16×16 bicubic interpolate.
#
#  DiT backbone: JiT-B/16 (동일)
#
#  Usage:
#    bash script/train_clevr_dit_vit_global_clip_only8x8.sh
#    GPUS=0,1 bash script/train_clevr_dit_vit_global_clip_only8x8.sh

set -e

# ── GPU / batch config ──
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

BATCH_PER_GPU=256
GRAD_ACCUM=1

echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS), batch/gpu=$BATCH_PER_GPU, accum=$GRAD_ACCUM, effective=$((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))"

# ── Data ──
# Relative to the directory from which the script is run (usually MNIST_debug/).
CLEVR_DIR="../clevr_output/clevr_256_varied/images"
CLEVR_VAL="../clevr_output/clevr_256_varied_val/images"

accelerate launch \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    src/main_multires.py \
    --backbone dit \
    --output_dir runs/clevr/backbone/vit_global_clip_out16_only8x8 \
    --train_dir "$CLEVR_DIR" \
    --val_dir "$CLEVR_VAL" \
    --dataset_root "$CLEVR_DIR" \
    --image_size 256 \
    --in_channels 3 \
    --vae_downsample_factor 1 \
    --min_patch_size 32 \
    --feat_channels 16 \
    --encoder_internal_dim 768 \
    --encoder_type vit_global \
    --level_sizes 8 \
    --vit_patch_size 16 \
    --vit_depth 12 \
    --vit_num_heads 12 \
    --vit_mlp_ratio 4.0 \
    --vit_no_cnn_stem \
    --vit_init_clip \
    --clip_model_name openai/clip-vit-base-patch16 \
    --dit_patch_size 16 \
    --dit_hidden_size 768 \
    --dit_n_heads 12 \
    --dit_n_blocks 12 \
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
    --max_train_steps 200000 \
    --batch_size $BATCH_PER_GPU \
    --blr 2.5e-5 \
    --weight_decay 0.05 \
    --warmup_steps 5000 \
    --max_grad_norm 3.0 \
    --grad_accum_steps $GRAD_ACCUM \
    --mixed_precision bf16 \
    --ema_decay 0 \
    --uncond_drop_prob 0.1 \
    --no_level_drop \
    --guidance_scale 3.0 \
    --log_every 100 \
    --save_every 10000 \
    --sample_every 5000 \
    --fid_every 9999999 \
    --eval_num_steps 50 \
    --num_workers 8 \
    --seed 42 \
    --mae_mask_ratio 0.0 \
    --clevr_eval_every 5000 \
    --clevr_eval_samples 50 \
