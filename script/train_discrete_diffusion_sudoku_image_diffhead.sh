#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  Sudoku — Discrete Diffusion (mask-based) on CONTINUOUS image tokens
#  with Diffusion Head (MAR-style)
#
#  Architecture:
#    - Pretrained ViT encoder (from dit_9x9_cont_out16) → continuous
#      feature vectors (81 tokens × 16-dim, no FSQ/VQ quantization)
#    - Backbone: DiT with mask-based discrete diffusion (MDLM-style)
#      operating on continuous tokens (masked positions → learned
#      mask embedding)
#    - Output head: small MLP diffusion head (flow matching) that
#      takes backbone hidden states as conditioning and generates
#      continuous token vectors
#
#  Conditioning: digit grid (81 integers) → SudokuConditionEncoder
#    - prefix concat, random masking during training
#    - mask_ratio 0.0~1.0 uniform → supports both uncond & cond gen
#
#  Reference: MAR (Li et al., 2024), Semanticist (Liu et al., 2025)
#
#  Usage:
#    bash script/train_discrete_diffusion_sudoku_image_diffhead.sh
#    GPUS=0,1 bash script/train_discrete_diffusion_sudoku_image_diffhead.sh
# ──────────────────────────────────────────────────────────────────

set -e

# ── GPU config ──
GPUS=${GPUS:-"0,1,2,3"}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

# ── Paths ──
# Pretrained continuous encoder (NO FSQ — continuous features, dim=16)
PRETRAINED_DIR=${PRETRAINED_DIR:-"./runs/sudoku/dit_9x9_cont_out16"}
SRM_CONFIG="./config/sudoku_config.json"
CLASSIFIER_PTH="./datasets/mnist_sudoku/mnist_classifier.pth"
OUTPUT_DIR="./runs/sudoku/discrete_diff_image_diffhead"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS="--output_dir ${OUTPUT_DIR} \
  --dataset_type sudoku \
  --sudoku_config ${SRM_CONFIG} \
  --pretrained_output_dir ${PRETRAINED_DIR} \
  --classifier_pth ${CLASSIFIER_PTH} \
  --image_size 288 \
  --grid_hw 9 \
  --mask_ratio_min 0.0 \
  --mask_ratio_max 1.0 \
  --max_train_steps 100000 \
  --batch_size 128 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --warmup_steps 2000 \
  --hidden_size 512 \
  --n_heads 8 \
  --n_blocks 8 \
  --cond_dim 256 \
  --mlp_ratio 4 \
  --model_dropout 0.0 \
  --pos_emb_type multires \
  --noise_type loglinear \
  --uncond_drop_prob 0.0 \
  --ema_decay 0.9999 \
  --save_every 50000 \
  --eval_every 5000 \
  --log_every 100 \
  --eval_num_samples 64 \
  --eval_num_steps 128 \
  --decode_num_steps 50 \
  --eval_video_samples 4 \
  --seed 42 \
  --mixed_precision bf16 \
  --log_with tensorboard \
  --grad_accum_steps 2 \
  --use_diffusion_head \
  --diff_head_depth 6 \
  --diff_head_width 1024 \
  --diff_head_num_sampling_steps 100 \
  --diff_head_batch_mul 4 \
  --diff_head_temperature 1.0 \
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
