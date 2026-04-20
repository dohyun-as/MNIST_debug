#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  CLEVR 256×256 — Multi-Resolution DiT + ViT encoder
#  (continuous features, no FSQ)
# ──────────────────────────────────────────────────────────────────
#
#  FSQ 없이 continuous feature (dim=16)로 DiT conditioning.
#  Encoder capacity 진단용.
#
#  ── JiT variant configs ──────────────────────────────────────────
#  JiT-B/16 (default):
#    --dit_hidden_size 768  --dit_n_heads 12 --dit_n_blocks 12
#    --dit_bottleneck_dim 128 --dit_in_context_len 32 --dit_in_context_start 4
#
#  JiT-L/16:
#    --dit_hidden_size 1024 --dit_n_heads 16 --dit_n_blocks 24
#    --dit_bottleneck_dim 128 --dit_in_context_len 32 --dit_in_context_start 8
#
#  JiT-H/16:
#    --dit_hidden_size 1280 --dit_n_heads 16 --dit_n_blocks 32
#    --dit_bottleneck_dim 256 --dit_in_context_len 32 --dit_in_context_start 10
#  ──────────────────────────────────────────────────────────────────
#
#  Usage:
#    bash script/train_clevr_dit_continuous.sh
#    GPUS=0,1 bash script/train_clevr_dit_continuous.sh

set -e

# ── GPU / batch config ──
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

BATCH_PER_GPU=256
GRAD_ACCUM=1

# ── Token drop ratio (override with TOK_DROP=... env var) ──
# level_drop 후 각 sample 의 finest kept level 에만 per-token Bernoulli drop.
# p_b ~ U(0, TOK_DROP) 이라 p=0 (inference 분포)도 자연히 커버.
TOK_DROP="${TOK_DROP:-1.0}"

echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS), batch/gpu=$BATCH_PER_GPU, accum=$GRAD_ACCUM, effective=$((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))"
echo "cond_token_drop_prob=$TOK_DROP (finest kept level only)"

# ── Data ──
CLEVR_DIR="../clevr_output/clevr_256_varied/images"
CLEVR_VAL="../clevr_output/clevr_256_varied_val/images"

accelerate launch \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    src/main_multires.py \
    --backbone dit \
    --output_dir runs/clevr/256_dit_vit_flow_cont_out4 \
    --train_dir "$CLEVR_DIR" \
    --val_dir "$CLEVR_VAL" \
    --dataset_root "$CLEVR_DIR" \
    --image_size 256 \
    --in_channels 3 \
    --vae_downsample_factor 1 \
    --min_patch_size 32 \
    --feat_channels 4 \
    --encoder_internal_dim 256 \
    --depth_per_level 2 \
    --cnn_base_channels 64 \
    --encoder_type vit \
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
    --cond_token_drop_prob $TOK_DROP \
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
