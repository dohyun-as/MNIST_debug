#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  Sudoku — Discrete Diffusion on image tokens (backbone encoder)
#  Backbone: dit_9x9_v2 (ViT encoder + FSQ + DiT decoder)
#    - level_sizes=[9] → 81 tokens, vocab from FSQ [8,8,8,8,5,5,5]
#
#  Conditioning: digit grid (81 integers) → SudokuConditionEncoder
#    - prefix concat (like CLEVR), random masking during training
#    - mask_ratio 0.0~1.0 uniform → supports both uncond & cond gen
#
#  Eval:
#    - unconditional generation (all-MASK condition) × 3 samplers
#    - easy / medium / hard difficulty (digit hints as condition)
#    - decode tokens → images → MNIST classifier → sudoku rule check
#    - mp4 video of denoising process
# ──────────────────────────────────────────────────────────────────

# ── GPU config ──
GPUS=${GPUS:-"0,1,2,3"}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

# ── Paths ──
PRETRAINED_DIR=${PRETRAINED_DIR:-"./runs/sudoku/dit_9x9_v2_s_fsq"}
SRM_CONFIG="./config/sudoku_config.json"
CLASSIFIER_PTH="./datasets/mnist_sudoku/mnist_classifier.pth"
OUTPUT_DIR="./runs/sudoku/discrete_diff_image_s_fsq_init_embed"

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
  --lr 3e-4 \
  --weight_decay 0.01 \
  --warmup_steps 1000 \
  --hidden_size 512 \
  --n_heads 8 \
  --n_blocks 8 \
  --cond_dim 256 \
  --mlp_ratio 4 \
  --model_dropout 0.1 \
  --pos_emb_type multires \
  --noise_type loglinear \
  --uncond_drop_prob 0.0 \
  --ema_decay 0.9999 \
  --save_every 50000 \
  --eval_every 10000 \
  --log_every 100 \
  --eval_num_samples 64 \
  --eval_num_steps 128 \
  --decode_num_steps 50 \
  --eval_video_samples 4 \
  --eval_save_format mp4 \
  --seed 42 \
  --mixed_precision bf16 \
  --log_with tensorboard \
  --grad_accum_steps 2 \
  --sampler ddpm_cache \
  --init_embed_from_fsq \
  ${RESUME_DIR:+--resume_dir $RESUME_DIR} \
"
# Eval automatically runs:
#   - 3 samplers (ddpm_cache, confidence_top1, confidence_cosine)
#   - easy/medium/hard difficulty conditioned generation
#   - mp4 videos of denoising process

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
    --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} \
    src/train_discrete_diffusion_v2.py $COMMON_ARGS
else
  CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
    src/train_discrete_diffusion_v2.py $COMMON_ARGS
fi
