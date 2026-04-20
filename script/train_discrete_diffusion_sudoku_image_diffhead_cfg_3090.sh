#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  Sudoku — Discrete Diffusion + DiffHead + CFG (3090-friendly)
#
#  Same backbone size as *_pro (hidden=512, 8L/8H) — the experiment
#  here is isolating CFG training on top of the existing setup:
#    - diff_head_cond_drop_prob=0.1, diff_head_cfg=3.0
#    - cosine LR with longer warmup
#    - model_dropout=0 (CFG drop already regularizes)
#
#  Memory (3090 24GB, bf16, no ckpt):
#    batch_size=64 per GPU → fits in ~18-20 GB
#    Effective batch = 64 × NUM_GPUS × grad_accum_steps
#    Default: 4 GPUs × grad_accum=1 → effective 256 (matches _pro)
#    Single GPU: set GPUS=0 and grad_accum=4 → effective 256
#    OOM? drop to batch_size=32.
# ──────────────────────────────────────────────────────────────────

set -e

GPUS=${GPUS:-"0,1,2,3"}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

PRETRAINED_DIR=${PRETRAINED_DIR:-"./runs/sudoku/backbone/dit_9x9_cont_out16_tokdrop1.0"}
SRM_CONFIG="./config/sudoku_config.json"
CLASSIFIER_PTH="./datasets/mnist_sudoku/mnist_classifier.pth"
OUTPUT_DIR="./runs/sudoku/masked_diffusion/masked_diff_image_diffhead_cfg_3090_tokdrop1.0"

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
  --batch_size 64 \
  --grad_accum_steps 1 \
  --lr 2e-4 \
  --weight_decay 0.02 \
  --warmup_steps 2000 \
  --lr_schedule cosine \
  --min_lr_ratio 0.1 \
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
  --save_every 10000 \
  --eval_every 5000 \
  --log_every 100 \
  --eval_num_samples 64 \
  --eval_num_steps 128 \
  --decode_num_steps 50 \
  --eval_video_samples 4 \
  --seed 42 \
  --mixed_precision bf16 \
  --log_with tensorboard \
  --use_diffusion_head \
  --diff_head_depth 6 \
  --diff_head_width 1024 \
  --diff_head_num_sampling_steps 100 \
  --diff_head_batch_mul 4 \
  --diff_head_temperature 1.0 \
  --diff_head_cond_drop_prob 0.1 \
  --diff_head_cfg 3.0 \
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
