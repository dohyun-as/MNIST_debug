#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  CLEVR 256x256 — Baseline 1D-Conditioned DiT (Semanticist-style)
#  (continuous features, no FSQ)
# ──────────────────────────────────────────────────────────────────
#
#  Encoder: SemanticistViTEncoder (vit_base_patch16, 85 causal slots)
#    - Causal attention on slots (earlier = more informative)
#    - NestedSampler progressive dropping
#    - slot_dim=16 (continuous, no FSQ)
#
#  DiT backbone: Same JiT-B/16 architecture as ours
#    - Conditioning: 1D token concatenation (no spatial masking)
#    - Self-attention only (no cross-attention)
#
#  Purpose: Baseline comparison — spatially-aligned multi-res (ours)
#           vs. 1D unstructured (Semanticist-style), both continuous
#
#  Usage:
#    bash script/train_clevr_dit_baseline_continuous.sh
#    GPUS=0,1 bash script/train_clevr_dit_baseline_continuous.sh

set -e

# ── GPU / batch config ──
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

BATCH_PER_GPU=16
GRAD_ACCUM=4

echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS), batch/gpu=$BATCH_PER_GPU, accum=$GRAD_ACCUM, effective=$((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))"

# ── Data ──
CLEVR_DIR="../clevr-dataset-gen/output/clevr_256_varied/images"
CLEVR_VAL="../clevr-dataset-gen/output/clevr_256_varied_val/images"

accelerate launch \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    src/main_multires.py \
    --backbone baseline_1d \
    --output_dir runs/clevr/backbone/baseline_1d_85_cont_out16_s_batch \
    --train_dir "$CLEVR_DIR" \
    --val_dir "$CLEVR_VAL" \
    --dataset_root "$CLEVR_DIR" \
    --image_size 256 \
    --in_channels 3 \
    --vae_downsample_factor 1 \
    --num_slots 85 \
    --slot_dim 16 \
    --enc_embed_dim 768 \
    --enc_depth 12 \
    --enc_num_heads 12 \
    --enc_drop_path_rate 0.1 \
    --is_causal \
    --enable_nest \
    --enable_nest_after_steps 10000 \
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
    --save_every 50000 \
    --sample_every 5000 \
    --fid_every 9999999 \
    --eval_num_steps 50 \
    --num_workers 8 \
    --seed 42 \
    --mae_mask_ratio 0.0 \
    --clevr_eval_every 10000 \
    --clevr_eval_samples 50 \
    --eval_slot_configs 1 4 16 64 85 \
