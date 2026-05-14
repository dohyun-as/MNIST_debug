#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  CLEVR 256×256 — Multi-Resolution DiT + ViT encoder (5 levels)
#  Adds 16×16 finest level on top of [8,4,2,1].
#  (continuous features, no FSQ) — random per-token drop on ALL levels
#  with LINEAR drop ratio sampling (f(p)=2p, biased toward more masking)
# ──────────────────────────────────────────────────────────────────
#
#  Encoder cost (token-forwards per image, per CellViT pass):
#    Level 16: cell=16, no stem, tps=2, vit_p=8 → 4 tokens × 256 cells = 1024
#    Level  8: cell=32, stem×4,  tps=8, eff_p=1 → 64 tokens × 64 cells = 4096
#    Level  4: cell=64, stem×4,  tps=8, eff_p=2 → 64 × 16 = 1024
#    Level  2: cell=128, stem×4, tps=8, eff_p=4 → 64 × 4  = 256
#    Level  1: cell=256, stem×4, tps=8, eff_p=8 → 64 × 1  = 64
#    Total: 6464  (vs 5440 without 16-level → +19%)
#
#  DiT cond tokens: 256 + 64 + 16 + 4 + 1 = 341 (was 85). Cross-attn matrix
#  256×341 ≈ 4× the previous; ~5 GB extra activation memory per GPU.
#
#  Usage:
#    bash script/train_clevr_dit_our_continuous_random_drop_linear_lvl16.sh
#    GPUS=0,1 bash script/train_clevr_dit_our_continuous_random_drop_linear_lvl16.sh

set -e

# ── GPU / batch config ──
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

BATCH_PER_GPU=256
GRAD_ACCUM=1

echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS), batch/gpu=$BATCH_PER_GPU, accum=$GRAD_ACCUM, effective=$((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))"

# ── Data ──
CLEVR_DIR="../clevr-dataset-gen/output/clevr_256_varied/images"
CLEVR_VAL="../clevr-dataset-gen/output/clevr_256_varied_val/images"

accelerate launch \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    src/main_multires.py \
    --backbone dit \
    --output_dir runs/clevr/backbone/out16_randomdrop_alllvl_linear_multi_res_lvl16 \
    --train_dir "$CLEVR_DIR" \
    --val_dir "$CLEVR_VAL" \
    --dataset_root "$CLEVR_DIR" \
    --image_size 256 \
    --in_channels 3 \
    --vae_downsample_factor 1 \
    --min_patch_size 16 \
    --feat_channels 16 \
    --encoder_internal_dim 256 \
    --depth_per_level 2 \
    --cnn_base_channels 64 \
    --encoder_type vit \
    --vit_patch_size 4 \
    --vit_cnn_stem_reduction 4 \
    --vit_default_tps 8 \
    --vit_small_cell_tps 2 \
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
    --cond_token_drop_prob 1.0 \
    --cond_token_drop_all_levels \
    --cond_token_drop_linear \
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
    --clevr_eval_n_annotated_random 8 \
    --clevr_eval_n_annotated_worst 16 \
    --clevr_eval_annot_thresh 10
