#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  CLEVR — Autoregressive (AR) on multi-res FSQ tokens
#  JSON-conditioned (tokenized entity/relation) via prefix concat
# ──────────────────────────────────────────────────────────────────
#  Requires: pretrained continuous diffusion model with FSQ encoder.
#
#  Usage:
#    bash script/train_ar_clevr.sh
#    GPUS=0,1 bash script/train_ar_clevr.sh

GPUS=${GPUS:-"0,1"}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

PRETRAINED_DIR=${PRETRAINED_DIR:-"./runs/clevr_256_dit_vit_flow_fsq_mask075_CA"}
CLEVR_IMAGE_ROOT=${CLEVR_IMAGE_ROOT:-"../clevr-dataset-gen/output/clevr_256_varied/images"}
CLEVR_COND_DIR=${CLEVR_COND_DIR:-"../clevr-dataset-gen/output/clevr_256_varied/conditions_margin50_augmented"}
CLEVR_VAL_IMAGE_ROOT=${CLEVR_VAL_IMAGE_ROOT:-"../clevr-dataset-gen/output/clevr_256_varied_val/images"}
CLEVR_VAL_COND_DIR=${CLEVR_VAL_COND_DIR:-"../clevr-dataset-gen/output/clevr_256_varied_val/conditions_margin50_augmented"}
OUTPUT_DIR="./runs/clevr_ar"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS="--output_dir ${OUTPUT_DIR} \
  --model_type ar \
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
  --batch_size 256 \
  --lr 3e-4 \
  --weight_decay 0.01 \
  --warmup_steps 2000 \
  --hidden_size 512 \
  --n_heads 8 \
  --n_blocks 12 \
  --cond_dim 256 \
  --mlp_ratio 4 \
  --model_dropout 0.1 \
  --pos_emb_type multires \
  --uncond_drop_prob 0.0 \
  --ema_decay 0 \
  --save_every 10000 \
  --eval_every 5000 \
  --log_every 100 \
  --eval_num_samples 32 \
  --eval_num_steps 128 \
  --decode_num_steps 50 \
  --seed 42 \
  --mixed_precision bf16 \
  --log_with tensorboard \
  --grad_accum_steps 1 \
  --ar_temperature 0.9 \
  --ar_top_k 0 \
  --ar_top_p 0.95 \
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
