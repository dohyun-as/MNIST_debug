#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  Eval-only: load a saved ckpt and run evaluate_and_save once.
#  Mirrors train_discrete_diffusion_sudoku_image_diffhead_pro.sh args
#  so model/data construction matches the trained config.
#
#  Usage:
#    bash script/eval_only_discrete_diffusion_sudoku_image_diffhead_pro.sh
#    GPUS=0 bash ...                               # single GPU
#    RESUME_DIR=/path/to/ckpt/stepNNNNNN bash ...  # specific ckpt
#    OUTPUT_DIR=./runs/sudoku/<your_run> bash ...  # different run
#
#  If RESUME_DIR is unset, the latest ckpt under OUTPUT_DIR/ckpt/ is used.
# ──────────────────────────────────────────────────────────────────

set -e

GPUS=${GPUS:-"0,1,2,3"}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

PRETRAINED_DIR=${PRETRAINED_DIR:-"./runs/sudoku/dit_9x9_cont_out16"}
SRM_CONFIG="./config/sudoku_config.json"
CLASSIFIER_PTH="./datasets/mnist_sudoku/mnist_classifier.pth"
OUTPUT_DIR=${OUTPUT_DIR:-"./runs/sudoku/discrete_diff_image_diffhead_mask_conditioning_coslr"}

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
  --max_train_steps 200000 \
  --batch_size 256 \
  --lr 3e-4 \
  --weight_decay 0.01 \
  --warmup_steps 1000 \
  --hidden_size 768 \
  --n_heads 12 \
  --n_blocks 12 \
  --cond_dim 256 \
  --mlp_ratio 4 \
  --model_dropout 0.1 \
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
  --grad_accum_steps 1 \
  --use_diffusion_head \
  --diff_head_depth 6 \
  --diff_head_width 1024 \
  --diff_head_num_sampling_steps 100 \
  --diff_head_batch_mul 4 \
  --diff_head_temperature 1.0 \
  --diff_head_cond_drop_prob 0.1 \
  --diff_head_cfg 3.0 \
  --lr_schedule cosine \
  --eval_only \
  ${RESUME_DIR:+--resume_dir $RESUME_DIR} \
"

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
    --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} \
    src/train_discrete_diffusion_v2.py $COMMON_ARGS
else
  CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
    --main_process_port $PORT --num_processes 1 \
    src/train_discrete_diffusion_v2.py $COMMON_ARGS
fi
