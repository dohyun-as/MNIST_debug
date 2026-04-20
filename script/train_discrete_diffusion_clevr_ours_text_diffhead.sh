#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  CLEVR — Discrete Diffusion (mask-based) on CONTINUOUS image tokens
#  with Diffusion Head (MAR-style) — TEXT conditioned
#
#  Architecture:
#    - Pretrained multi-res ViT encoder (continuous, out_dim=16, no FSQ)
#      from runs/clevr/256_dit_vit_flow_cont_out16
#    - Backbone: DiT with mask-based discrete diffusion (MDLM-style)
#      on continuous tokens
#    - Output head: small MLP diffusion head (flow matching) conditioned
#      on backbone hidden states — generates continuous token vectors
#
#  Conditioning: natural language captions (conditions_text/)
#    - CFG via learned null_cond token in diffusion head
#      (diff_head_cond_drop_prob > 0)
#
#  Reference: MAR (Li et al., 2024), Semanticist (Liu et al., 2025)
#
#  Usage:
#    bash script/train_discrete_diffusion_clevr_ours_text_diffhead.sh
#    GPUS=0,1 bash script/train_discrete_diffusion_clevr_ours_text_diffhead.sh
# ──────────────────────────────────────────────────────────────────

set -e

# ── GPU config ──
GPUS=${GPUS:-"0,1,2,3"}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

# ── Paths ──
# Continuous encoder (out_dim=16, no FSQ)
PRETRAINED_DIR=${PRETRAINED_DIR:-"./runs/clevr/256_dit_vit_flow_cont_out16"}
CLEVR_IMAGE_ROOT=${CLEVR_IMAGE_ROOT:-"../clevr_output/clevr_256_varied/images"}
CLEVR_COND_DIR=${CLEVR_COND_DIR:-"../clevr_output/clevr_256_varied/conditions_text"}
CLEVR_VAL_IMAGE_ROOT=${CLEVR_VAL_IMAGE_ROOT:-"../clevr_output/clevr_256_varied_val/images"}
CLEVR_VAL_COND_DIR=${CLEVR_VAL_COND_DIR:-"../clevr_output/clevr_256_varied_val/conditions_text"}
OUTPUT_DIR="./runs/clevr/discrete_diff_ours_text_diffhead"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS="--output_dir ${OUTPUT_DIR} \
  --dataset_type clevr \
  --dataset_root ${CLEVR_IMAGE_ROOT} \
  --clevr_condition_dir ${CLEVR_COND_DIR} \
  --clevr_val_image_root ${CLEVR_VAL_IMAGE_ROOT} \
  --clevr_val_condition_dir ${CLEVR_VAL_COND_DIR} \
  --clevr_cond_type text \
  --clevr_train_splits easy medium hard \
  --clevr_val_splits easy medium hard \
  --pretrained_output_dir ${PRETRAINED_DIR} \
  --image_size 256 \
  --max_train_steps 200000 \
  --batch_size 256 \
  --lr 3e-4 \
  --weight_decay 0.0 \
  --warmup_steps 2000 \
  --hidden_size 768 \
  --n_heads 12 \
  --n_blocks 12 \
  --cond_dim 256 \
  --mlp_ratio 4 \
  --model_dropout 0.1 \
  --pos_emb_type multires \
  --noise_type loglinear \
  --uncond_drop_prob 0.1 \
  --cfg_mode backbone \
  --ema_decay 0.9999 \
  --save_every 50000 \
  --eval_every 10000 \
  --log_every 100 \
  --eval_num_samples 30 \
  --eval_num_steps 128 \
  --decode_num_steps 50 \
  --seed 42 \
  --mixed_precision bf16 \
  --log_with tensorboard \
  --grad_accum_steps 1 \
  --sampler ddpm_cache \
  --use_diffusion_head \
  --diff_head_depth 6 \
  --diff_head_width 1024 \
  --diff_head_num_sampling_steps 100 \
  --diff_head_batch_mul 4 \
  --diff_head_temperature 1.0 \
  --diff_head_cond_drop_prob 0.1 \
  --diff_head_cfg 3.0 \
  --lr_schedule cosine \
  ${RESUME_DIR:+--resume_dir $RESUME_DIR} \
"

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
    --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} \
    src/train_discrete_diffusion_v2.py $COMMON_ARGS
else
  CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
    src/train_discrete_diffusion_v2.py $COMMON_ARGS
fi
