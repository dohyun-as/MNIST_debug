#!/bin/bash
# ──────────────────────────────────────────────────────────────
#  Evaluate discrete diffusion checkpoints on TRAIN + VAL sets
#  Tests both base and EMA weights (where available)
#
#  EMA available: step150000, step200000
#  No EMA:        step50000, step100000
# ──────────────────────────────────────────────────────────────

GPUS=${GPUS:-"0,1,2,3"}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

# Reuse token cache from training run to avoid re-encoding
TOKEN_CACHE="./runs/clevr/discrete_diff/token_cache"

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
  --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} \
  src/eval_discrete_diffusion_ckpts.py \
  --ckpt_root ./runs/clevr/discrete_diff/ckpt \
  --steps 50000 100000 150000 200000 \
  --pretrained_dir ./runs/clevr/dit_vit_flow_fsq_mask075_CA \
  --token_cache_dir ${TOKEN_CACHE} \
  --train_image_root ../clevr-dataset-gen/output/clevr_256_varied/images \
  --train_cond_dir ../clevr-dataset-gen/output/clevr_256_varied/conditions_margin50_augmented \
  --val_image_root ../clevr-dataset-gen/output/clevr_256_varied_val/images \
  --val_cond_dir ../clevr-dataset-gen/output/clevr_256_varied_val/conditions_margin50_augmented \
  --splits easy medium hard \
  --eval_num_samples 100 \
  --eval_num_steps 128 \
  --decode_num_steps 50 \
  --image_size 256 \
  --hidden_size 768 \
  --n_heads 12 \
  --n_blocks 12 \
  --cond_dim 256 \
  --mlp_ratio 4 \
  --model_dropout 0.1 \
  --noise_type loglinear \
  --mixed_precision bf16 \
  --output_dir ./runs/clevr/discrete_diff/ckpt_eval_results
