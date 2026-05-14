#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  CLEVR 256×256 — Text-Conditioned DiT (Baseline)
#  Standard text-to-image diffusion approach:
#    CLEVR JSON → natural language → frozen T5-base → DiT → image
#
#  Comparable to PixArt-α / Imagen style conditioning.
#  No multi-resolution encoding, no discrete diffusion stage.
# ──────────────────────────────────────────────────────────────────
#
#  Encoder modes:
#    --encoder_mode pretrained   (default) T5-base, frozen
#    --encoder_mode scratch      From-scratch CLEVR encoder (ablation)
#
#  Text encoder options:
#    --pretrained_model_name google-t5/t5-base    (default, 220M, PixArt-style)
#    --pretrained_model_name google-t5/t5-small   (60M, lighter)
#    --pretrained_model_name openai/clip-vit-base-patch32  (SD-style)
#
#  Freeze/unfreeze:
#    --freeze_text_encoder       (default) T5 frozen, only projection trained
#    --unfreeze_text_encoder     T5 fine-tuned with separate LR (default: 1/10)
#    --text_encoder_lr 1e-5      Override text encoder LR when unfrozen
#
#  Usage:
#    bash script/train_text_conditioned_clevr.sh
#    GPUS=0,1 bash script/train_text_conditioned_clevr.sh

set -e

# ── GPU / batch config ──
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

BATCH_PER_GPU=256
GRAD_ACCUM=1

echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS), batch/gpu=$BATCH_PER_GPU, accum=$GRAD_ACCUM, effective=$((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))"

# ── Data ──
CLEVR_IMAGE_ROOT="../clevr-dataset-gen/output/clevr_256_varied/images"
CLEVR_COND_DIR="../clevr-dataset-gen/output/clevr_256_varied/conditions_text"
CLEVR_VAL_IMAGE_ROOT="../clevr-dataset-gen/output/clevr_256_varied_val/images"
CLEVR_VAL_COND_DIR="../clevr-dataset-gen/output/clevr_256_varied_val/conditions_text"

accelerate launch \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    src/train_text_conditioned.py \
    --output_dir runs/clevr/naive_dit_256_text_cond_clip \
    --clevr_image_root "$CLEVR_IMAGE_ROOT" \
    --clevr_condition_dir "$CLEVR_COND_DIR" \
    --clevr_val_image_root "$CLEVR_VAL_IMAGE_ROOT" \
    --clevr_val_condition_dir "$CLEVR_VAL_COND_DIR" \
    --clevr_train_splits easy medium hard \
    --clevr_val_splits easy medium hard \
    --clevr_cond_type text \
    --image_size 256 \
    --in_channels 3 \
    --encoder_mode pretrained \
    --pretrained_model_name openai/clip-vit-base-patch32 \
    --pretrained_max_length 77 \
    --unfreeze_text_encoder \
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
    --weight_decay 0.0 \
    --warmup_steps 5000 \
    --max_grad_norm 3.0 \
    --grad_accum_steps $GRAD_ACCUM \
    --mixed_precision bf16 \
    --ema_decay 0 \
    --uncond_drop_prob 0.1 \
    --guidance_scale 3.0 \
    --log_every 100 \
    --save_every 50000 \
    --sample_every 5000 \
    --eval_num_steps 50 \
    --num_workers 8 \
    --seed 42 \
    --clevr_eval_every 10000 \
    --clevr_eval_samples 30 \
