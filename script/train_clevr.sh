#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  CLEVR 256×256 — Multi-Resolution Injection (pixel space, x0 pred)
# ──────────────────────────────────────────────────────────────────
#
#  Encoder levels: 8×8, 4×4, 2×2, 1×1  (min_patch_size=32)
#  UNet resolution: 256→128→64→32  (4 blocks, mid attention at 32×32)
#  Down-path injection: encoder features added before each down block
#  No VAE — pixel-space diffusion
#  x0 (sample) prediction
#
#  Eval: saves grids for each (level_config × guidance_scale) combo
#    levels: all, 3lv(4×4+2×2+1×1), 2lv(2×2+1×1), 1lv(1×1)
#    cfg: 1.0, 3.0
#
#  Usage:
#    bash script/train_clevr.sh
#    GPUS=0,1 bash script/train_clevr.sh

set -e

# ── GPU / batch config ──
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

BATCH_PER_GPU=16
GRAD_ACCUM=2

echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS), batch/gpu=$BATCH_PER_GPU, accum=$GRAD_ACCUM, effective=$((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))"

# ── Data ──
# CLEVR images: easy/medium/hard subdirs → ImageFolder compatible
CLEVR_DIR="/workspace/NAS/sr_diffusion/project/clevr-dataset-gen/output/clevr_256_varied/images"
CLEVR_VAL="/workspace/NAS/sr_diffusion/project/clevr-dataset-gen/output/clevr_256_varied_val/images"

accelerate launch \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    src/main_multires.py \
    --output_dir runs/clevr_256_varied_multires_fsq \
    --train_dir "$CLEVR_DIR" \
    --val_dir "$CLEVR_VAL" \
    --dataset_root "$CLEVR_DIR" \
    --image_size 256 \
    --in_channels 3 \
    --vae_downsample_factor 1 \
    --min_patch_size 32 \
    --feat_channels 256 \
    --depth_per_level 2 \
    --cnn_base_channels 64 \
    --block_out_channels 128 256 256 512 \
    --layers_per_block 2 \
    --num_train_timesteps 1000 \
    --beta_schedule scaled_linear \
    --prediction_type sample \
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
    --no_mid_attn \
    --use_fsq \
    --fsq_levels 8 8 8 8 8 5 \
    # --attn_resolutions 32 \
