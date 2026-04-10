#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  CLEVR — Discrete Diffusion on Semanticist (baseline_1d) FSQ tokens
#  Same hyperparams as discrete_diff (v1), different pretrained encoder
# ──────────────────────────────────────────────────────────────────
#
#  Pretrained encoder: Baseline1DConditionalDiT (Semanticist-style)
#    - SemanticistViTEncoder → 256 slots × 16-dim → FSQ [8,8,8,5,5,5]
#    - vocab_size = 64000, seq_len = 256 (flat 1D, no spatial hierarchy)
#
#  Discrete diffusion model: DIT with 1d positional embedding
#    - Same arch as discrete_diff v1 (768/12/12)
#    - pos_emb_type = 1d (not multires, since tokens are 1D slots)
# ──────────────────────────────────────────────────────────────────

# ── GPU config ──
GPUS=${GPUS:-"0,1,2,3"}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

# ── Paths ──
PRETRAINED_DIR=${PRETRAINED_DIR:-"./runs/clevr/backbone/clevr_256_dit_baseline_1d_semanticist"}
CLEVR_IMAGE_ROOT=${CLEVR_IMAGE_ROOT:-"../clevr-dataset-gen/output/clevr_256_varied/images"}
CLEVR_COND_DIR=${CLEVR_COND_DIR:-"../clevr-dataset-gen/output/clevr_256_varied/conditions_margin50_augmented"}
CLEVR_VAL_IMAGE_ROOT=${CLEVR_VAL_IMAGE_ROOT:-"../clevr-dataset-gen/output/clevr_256_varied_val/images"}
CLEVR_VAL_COND_DIR=${CLEVR_VAL_COND_DIR:-"../clevr-dataset-gen/output/clevr_256_varied_val/conditions_margin50_augmented"}
OUTPUT_DIR="./runs/clevr/discrete_diff_semanticist"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS="--output_dir ${OUTPUT_DIR} \
  --dataset_type clevr \
  --dataset_root ${CLEVR_IMAGE_ROOT} \
  --clevr_condition_dir ${CLEVR_COND_DIR} \
  --clevr_val_image_root ${CLEVR_VAL_IMAGE_ROOT} \
  --clevr_val_condition_dir ${CLEVR_VAL_COND_DIR} \
  --clevr_train_splits easy medium hard \
  --clevr_val_splits easy medium hard \
  --pretrained_output_dir ${PRETRAINED_DIR} \
  --image_size 256 \
  --max_train_steps 200000 \
  --batch_size 16 \
  --lr 3e-4 \
  --weight_decay 0.01 \
  --warmup_steps 2000 \
  --hidden_size 768 \
  --n_heads 12 \
  --n_blocks 12 \
  --cond_dim 256 \
  --mlp_ratio 4 \
  --model_dropout 0.1 \
  --pos_emb_type 1d \
  --noise_type loglinear \
  --uncond_drop_prob 0.0 \
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
  --grad_accum_steps 4 \
  --sampler ddpm_cache \
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
