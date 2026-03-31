#!/bin/bash
# Multi-Resolution Injection Mode Training on ImageNet 256×256 — Large (~500M)
# VAE: xwen99/mar-vae-kl16 (16× downsample, 16ch latent → 16×16)
#
# Model size: ~503M (vs ~170M for the base config)
#   feat_channels: 256 → 384
#   depth_per_level: 2 → 3
#   cnn_base_channels: 64 → 128
#   block_out_channels: [256,256,512,512] → [384,384,768,768]
#   layers_per_block: 2 → 3
#
# Usage:
#   bash script/train_imagenet_injection_large.sh
#   GPUS=0,1,2,3 bash script/train_imagenet_injection_large.sh
#   LATENT_CACHE_DIR=runs/imagenet_256_injection/latent_cache bash script/train_imagenet_injection_large.sh

set -e

# ── GPU / batch config ──
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

BATCH_PER_GPU=128
GRAD_ACCUM=1
if [ $((BATCH_PER_GPU * NUM_GPUS)) -lt 2048 ]; then
    GRAD_ACCUM=$((2048 / (BATCH_PER_GPU * NUM_GPUS)))
fi

EFFECTIVE=$((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS), batch/gpu=$BATCH_PER_GPU, accum=$GRAD_ACCUM, effective=$EFFECTIVE"

# Reuse base-run cache by default to avoid re-running expensive VAE encoding.
LATENT_CACHE_DIR="${LATENT_CACHE_DIR:-runs/imagenet_256_injection_cfg3/latent_cache}"
echo "Using latent cache dir: $LATENT_CACHE_DIR"

accelerate launch \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    src/main_multires.py \
    --output_dir runs/imagenet_256_injection_large_shallow_fsqX \
    --dataset_root ../imagenet/ILSVRC/Data/CLS-LOC \
    --image_size 256 \
    --in_channels 16 \
    --vae_downsample_factor 16 \
    --vae_pretrained xwen99/mar-vae-kl16 \
    --min_patch_size 32 \
    --feat_channels 384 \
    --depth_per_level 3 \
    --cnn_base_channels 128 \
    --block_out_channels 384 384 768 768 \
    --layers_per_block 3 \
    --num_train_timesteps 1000 \
    --beta_schedule scaled_linear \
    --prediction_type epsilon \
    --max_train_steps 250000 \
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
    --level_drop_after_steps 50000 \
    --guidance_scale 3.0 \
    --log_every 100 \
    --save_every 10000 \
    --sample_every 5000 \
    --fid_every 50000 \
    --eval_num_steps 250 \
    --fid_num_samples 50000 \
    --num_workers 8 \
    --seed 42 \
    --cache_latents \
    --latent_cache_dir "$LATENT_CACHE_DIR" \
    --cond_use_latent \
    "$@"

    # --use_fsq \
    # --fsq_levels 8 8 8 8 8 5 \