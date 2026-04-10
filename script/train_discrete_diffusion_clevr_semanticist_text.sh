#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  CLEVR — Discrete Diffusion on Semanticist (baseline_1d) FSQ tokens
#  TEXT conditioned (natural language captions from conditions_text/)
# ──────────────────────────────────────────────────────────────────

# ── GPU config ──
GPUS=${GPUS:-"0,1,2,3"}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

# ── Paths ──
PRETRAINED_DIR=${PRETRAINED_DIR:-"./runs/clevr/backbone/clevr_256_dit_baseline_1d_semanticist"}
CLEVR_IMAGE_ROOT=${CLEVR_IMAGE_ROOT:-"../clevr-dataset-gen/output/clevr_256_varied/images"}
CLEVR_COND_DIR=${CLEVR_COND_DIR:-"../clevr-dataset-gen/output/clevr_256_varied/conditions_text"}
CLEVR_VAL_IMAGE_ROOT=${CLEVR_VAL_IMAGE_ROOT:-"../clevr-dataset-gen/output/clevr_256_varied_val/images"}
CLEVR_VAL_COND_DIR=${CLEVR_VAL_COND_DIR:-"../clevr-dataset-gen/output/clevr_256_varied_val/conditions_text"}
OUTPUT_DIR="./runs/clevr/discrete_diff_semanticist_text_w_decay0_larger_batch"
RESUME_DIR=${RESUME_DIR:-"${OUTPUT_DIR}/ckpt/step50000"}

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS="--output_dir ${OUTPUT_DIR} \
  --resume_dir ${RESUME_DIR} \
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
  --batch_size 16 \
  --lr 1.5e-4 \
  --weight_decay 0.0 \
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
