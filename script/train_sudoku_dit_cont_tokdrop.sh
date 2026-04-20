#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  Sudoku 288×288 — DiT + ViT encoder (continuous features)
#                   + per-token random drop on cond tokens (train-only)
# ──────────────────────────────────────────────────────────────────
#
#  81개 9x9 cond token 중 일부를 매 step 랜덤하게 learned null 로 대체.
#  매 sample 마다 drop ratio p_b ~ U(0, TOK_DROP) 을 뽑아 Bernoulli(p_b)
#  per position. → p=0 (drop 없음, inference 조건) 부터 p=TOK_DROP 까지
#  전 구간이 training 분포에 포함되어 train/test mismatch 방지.
#
#  → DiT 가 sudoku rule 을 이용해 빠진 cell 을 채워야 loss 가 줄어듦
#  → encoder 가 각 cell 의 digit identity 를 semantic 하게 분리해야
#     다른 cell 이 rule 로 추론 가능해짐 (token space 분리 유도).
#
#  CFG uncond drop (--uncond_drop_prob) 과는 독립적으로 동작.
#  Inference 시에는 자동 비활성 (model.training=False → p=0 과 동일).
#
#  Usage:
#    bash script/train_sudoku_dit_cont_tokdrop.sh
#    GPUS=0,1,2,3 bash script/train_sudoku_dit_cont_tokdrop.sh
#    TOK_DROP=0.5 bash script/train_sudoku_dit_cont_tokdrop.sh

set -e

# ── GPU / batch config ──
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

BATCH_PER_GPU=32
GRAD_ACCUM=2

# ── Token drop ratio (override with TOK_DROP=... env var) ──
TOK_DROP="${TOK_DROP:-1.0}"

echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS), batch/gpu=$BATCH_PER_GPU, accum=$GRAD_ACCUM, effective=$((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))"
echo "cond_token_drop_prob=$TOK_DROP"

SRM_CONFIG="./config/sudoku_config.json"
CLASSIFIER="/workspace/NAS/sr_diffusion/project/MNIST_debug/datasets/mnist_sudoku/mnist_classifier.pth"

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

accelerate launch \
    --main_process_port $PORT \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    src/main_sudoku_dit.py \
    --backbone dit \
    --output_dir runs/sudoku/dit_9x9_cont_out16_tokdrop${TOK_DROP} \
    --sudoku_config "$SRM_CONFIG" \
    --classifier_pth "$CLASSIFIER" \
    --image_size 288 \
    --in_channels 1 \
    --cond_in_channels 1 \
    --level_sizes 9 \
    --feat_channels 16 \
    --encoder_internal_dim 256 \
    --depth_per_level 2 \
    --encoder_type vit \
    --vit_patch_size 4 \
    --vit_depth 4 \
    --vit_num_heads 4 \
    --vit_mlp_ratio 4.0 \
    --vit_use_cnn_stem \
    --vit_cnn_stem_reduction 4 \
    --dit_patch_size 16 \
    --dit_hidden_size 512 \
    --dit_n_heads 8 \
    --dit_n_blocks 8 \
    --dit_mlp_ratio 4.0 \
    --dit_dropout 0.0 \
    --dit_bottleneck_dim 128 \
    --dit_in_context_len 32 \
    --dit_in_context_start 4 \
    --cond_token_drop_prob $TOK_DROP \
    --use_flow_matching \
    --flow_P_mean -0.8 \
    --flow_P_std 0.8 \
    --flow_t_eps 0.05 \
    --flow_noise_scale 1.0 \
    --flow_sampling_method euler \
    --max_train_steps 200000 \
    --batch_size $BATCH_PER_GPU \
    --blr 2.5e-5 \
    --lr_schedule constant \
    --weight_decay 0.05 \
    --warmup_steps 5000 \
    --max_grad_norm 3.0 \
    --grad_accum_steps $GRAD_ACCUM \
    --mixed_precision bf16 \
    --ema_decay 0.9999 \
    --uncond_drop_prob 0.1 \
    --no_level_drop \
    --guidance_scale 5.0 \
    --log_every 100 \
    --save_every 10000 \
    --eval_every 5000 \
    --eval_num_steps 50 \
    --eval_num_samples 81 \
    --num_workers 4 \
    --seed 42 \
    --mae_mask_ratio 0.0
