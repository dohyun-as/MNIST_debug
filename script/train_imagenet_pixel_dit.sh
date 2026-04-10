#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  ImageNet 256×256 — Multi-Resolution DiT (pixel space, flow matching V-loss)
# ──────────────────────────────────────────────────────────────────
#
#  DiT backbone (JiT-style): RMSNorm, SwiGLU, QK-norm, adaLN-Zero
#  BottleneckPatchEmbed, 2D RoPE, in-context tokens, fixed sin-cos pos embed,
#  middle-50% dropout, cross-attention conditioning (spatial alignment mask)
#
#  Encoder levels: 8×8, 4×4, 2×2, 1×1  (min_patch_size=32)
#  No VAE — pixel-space diffusion
#  Flow matching with V-prediction loss (JiT-style)
#
#  ── JiT variant configs ──────────────────────────────────────────
#  (patch_size=16 for all variants below; use 32 for /32 variants)
#
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
#    bash script/train_imagenet_pixel_dit.sh
#    GPUS=0,1,2,3 bash script/train_imagenet_pixel_dit.sh

set -e

# ── GPU / batch config ──
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

BATCH_PER_GPU=16
GRAD_ACCUM=16

echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS), batch/gpu=$BATCH_PER_GPU, accum=$GRAD_ACCUM, effective=$((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))"

accelerate launch \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    src/main_multires.py \
    --backbone dit \
    --output_dir runs/imagenet_256_pixel_dit_flow_fsq_mask075_CA_bugfix \
    --dataset_root ../imagenet/ILSVRC/Data/CLS-LOC \
    --image_size 256 \
    --in_channels 3 \
    --vae_downsample_factor 1 \
    --min_patch_size 32 \
    --feat_channels 256 \
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
    --level_drop \
    --min_keep_levels 1 \
    --level_drop_after_steps 10000 \
    --guidance_scale 3.0 \
    --log_every 100 \
    --save_every 50000 \
    --sample_every 5000 \
    --fid_every 9999999 \
    --eval_num_steps 50 \
    --num_workers 8 \
    --seed 42 \
    --use_fsq \
    --fsq_levels 8 8 8 5 5 5 \
    --mae_mask_ratio 0.75 \
    # --vit_depth 12 \
    # --vit_num_heads 12 \
    # --feat_channels 768 \

