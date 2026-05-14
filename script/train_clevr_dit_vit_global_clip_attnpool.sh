#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  CLEVR 256×256 — Multi-Res DiT + ViT-Global encoder (CLIP init)
#                  + LEARNED ATTENTION POOL (no FFN)
# ──────────────────────────────────────────────────────────────────
#
#  Base: train_clevr_dit_vit_global_clip.sh
#  Diff: fixed avg_pool → learnable per-level cross-attention pool.
#
#  Pool details (enc_d=768, levels=[8,4,2,1]):
#    Shared : LN_kv + Wkv  (~1.18M)
#    Per-lv : queries (s²·768) + Wq + Wo  (~1.18M each)
#    Total pool params: ~5.98M (+6.9% over avg-pool baseline)
#    No FFN — isolates "learned aggregation" effect from extra MLP capacity.
#    Per-level trainable capacity: ~50K (avg) → ~1.18M (attn no-FFN), ~24×.
#
#  Hypothesis: coarse-level features (level 1/2) should improve the most
#  because they currently are pure averages of fine-grid tokens; learned
#  queries can compute task-specific global summaries instead.
#
#  Usage:
#    bash script/train_clevr_dit_vit_global_clip_attnpool.sh
#    GPUS=0,1 bash script/train_clevr_dit_vit_global_clip_attnpool.sh

set -e

# ── GPU / batch config ──
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

BATCH_PER_GPU=256
GRAD_ACCUM=1

echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS), batch/gpu=$BATCH_PER_GPU, accum=$GRAD_ACCUM, effective=$((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))"

# ── Data ──
CLEVR_DIR="../clevr_output/clevr_256_varied/images"
CLEVR_VAL="../clevr_output/clevr_256_varied_val/images"

accelerate launch \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    src/main_multires.py \
    --backbone dit \
    --output_dir runs/clevr/backbone/vit_global_clip_out16_attnpool \
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
    --vit_patch_size 16 \
    --vit_depth 12 \
    --vit_num_heads 12 \
    --vit_mlp_ratio 4.0 \
    --vit_no_cnn_stem \
    --vit_init_clip \
    --clip_model_name openai/clip-vit-base-patch16 \
    --vit_global_pool_type attn \
    --vit_global_pool_no_ffn \
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
    --level_drop \
    --min_keep_levels 1 \
    --level_drop_after_steps 10000 \
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
