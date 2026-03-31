#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  ImageNet — Discrete Diffusion (MDLM-style) on multi-res FSQ tokens
#  Class-label conditioned via adaLN
# ──────────────────────────────────────────────────────────────────
#
#  Requires: pretrained continuous diffusion model (e.g., from
#  train_imagenet_pixel_dit.sh) with FSQ encoder + discretizer.
#
#  Token extraction: encoder(image) → FSQ → flat token sequence
#  Conditioning: class label embedding added to timestep via adaLN
#  Positional embedding: multires (level + 2D row/col per level)
#
#  Level sizes: [8, 4, 2, 1] → seq_len = 64+16+4+1 = 85
#  Vocab size: prod(fsq_levels) e.g. 8*8*8*5*5*5 = 64000
# ──────────────────────────────────────────────────────────────────

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1

# ── GPU config ──
GPUS=${GPUS:-"0,1"}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

# ── Paths ──
# Pretrained continuous model output dir (contains checkpoints/ and run_config.json)
PRETRAINED_DIR=${PRETRAINED_DIR:-"./runs/imagenet_256_pixel_dit_flow_fsq_mask075_CA"}
IMAGENET_ROOT=${IMAGENET_ROOT:-"../imagenet/ILSVRC/Data/CLS-LOC"}
OUTPUT_DIR="./runs/imagenet_discrete_diff_v2"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS="--output_dir ${OUTPUT_DIR} \
  --dataset_type imagenet \
  --dataset_root ${IMAGENET_ROOT} \
  --pretrained_output_dir ${PRETRAINED_DIR} \
  --image_size 256 \
  --num_classes 1000 \
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
  --noise_type loglinear \
  --uncond_drop_prob 0.1 \
  --ema_decay 0 \
  --save_every 10000 \
  --eval_every 5000 \
  --log_every 100 \
  --eval_num_samples 64 \
  --eval_num_steps 128 \
  --decode_num_steps 50 \
  --seed 42 \
  --mixed_precision bf16 \
  --log_with tensorboard \
  --grad_accum_steps 1 \
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
