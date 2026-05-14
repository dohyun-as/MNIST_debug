#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  CLEVR — Discrete Diffusion + Diffusion Head (MAR-style)
#  TEXT conditioned via pretrained CLIP text encoder (UNFROZEN),
#  using **styled (coverage-complete) caption families**.
#
#  Difference vs. train_discrete_diffusion_clevr_ours_text_diffhead_clip.sh:
#    Caption source switched from `conditions_text/` (legacy single-style
#    fixed template) to `conditions_text_styled/` (per-image JSON with
#    six prompt families: C, E, R, C+E, C+R, E+R).
#
#  No new CLI flag is needed: the trainer auto-detects styled captions by
#  inspecting whether each loaded caption dict carries an `exposed` mask,
#  and switches eval to `eval_clevr_complex_text` (family-aware metrics).
#
#  Usage:
#    bash script/train_discrete_diffusion_clevr_ours_text_diffhead_clip_styled.sh
#    GPUS=0,1 bash script/train_discrete_diffusion_clevr_ours_text_diffhead_clip_styled.sh
#
#  Detached (survives session disconnect):
#    setsid nohup bash script/train_discrete_diffusion_clevr_ours_text_diffhead_clip_styled.sh \
#        > runs/clevr/masked_diff/styled_run.log 2>&1 < /dev/null &
#    disown
# ──────────────────────────────────────────────────────────────────

set -e

# ── GPU / batch config ──
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

BATCH_PER_GPU=256
GRAD_ACCUM=1

echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS), batch/gpu=$BATCH_PER_GPU, accum=$GRAD_ACCUM, effective=$((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))"

# ── Paths ──
PRETRAINED_DIR=${PRETRAINED_DIR:-"./runs/clevr/backbone/out16_randomdrop_alllvl_linear_multi_res"}
CLEVR_IMAGE_ROOT=${CLEVR_IMAGE_ROOT:-"../clevr-dataset-gen/output/clevr_256_varied_merged/images"}
CLEVR_COND_DIR=${CLEVR_COND_DIR:-"../clevr-dataset-gen/output/clevr_256_varied_merged/conditions_text_styled"}
CLEVR_VAL_IMAGE_ROOT=${CLEVR_VAL_IMAGE_ROOT:-"../clevr-dataset-gen/output/clevr_256_varied_val/images"}
CLEVR_VAL_COND_DIR=${CLEVR_VAL_COND_DIR:-"../clevr-dataset-gen/output/clevr_256_varied_val/conditions_text_styled"}
OUTPUT_DIR=${OUTPUT_DIR:-"./runs/clevr/masked_diff/out16_randomdrop_alllvl_linear_multi_res_merged_full_data_styled"}

mkdir -p "${OUTPUT_DIR}"

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# ── Launch ──
LAUNCH_ARGS=(
  --main_process_port $PORT
  --num_processes $NUM_GPUS
)
if [ $NUM_GPUS -gt 1 ]; then
  LAUNCH_ARGS+=(--multi_gpu)
fi

accelerate launch \
    "${LAUNCH_ARGS[@]}" \
    src/train_discrete_diffusion_v2.py \
    --output_dir "$OUTPUT_DIR" \
    --dataset_type clevr \
    --dataset_root "$CLEVR_IMAGE_ROOT" \
    --clevr_condition_dir "$CLEVR_COND_DIR" \
    --clevr_val_image_root "$CLEVR_VAL_IMAGE_ROOT" \
    --clevr_val_condition_dir "$CLEVR_VAL_COND_DIR" \
    --clevr_cond_type text \
    --use_pretrained_text_encoder \
    --pretrained_text_model_name openai/clip-vit-base-patch32 \
    --pretrained_text_max_length 77 \
    --unfreeze_text_encoder \
    --text_encoder_lr 3e-5 \
    --clevr_train_splits easy medium hard \
    --clevr_val_splits easy medium hard \
    --pretrained_output_dir "$PRETRAINED_DIR" \
    --image_size 256 \
    --max_train_steps 400000 \
    --batch_size $BATCH_PER_GPU \
    --grad_accum_steps $GRAD_ACCUM \
    --lr 3e-4 \
    --weight_decay 0.0 \
    --warmup_steps 2000 \
    --lr_schedule cosine \
    --hidden_size 768 \
    --n_heads 12 \
    --n_blocks 12 \
    --cond_dim 256 \
    --mlp_ratio 4 \
    --model_dropout 0.1 \
    --pos_emb_type multires \
    --noise_type loglinear \
    --uncond_drop_prob 0.1 \
    --cfg_mode backbone \
    --ema_decay 0.9999 \
    --save_every 50000 \
    --eval_every 10000 \
    --log_every 100 \
    --eval_num_samples 30 \
    --eval_num_steps 128 \
    --decode_num_steps 50 \
    --sampler ddpm_cache \
    --use_diffusion_head \
    --diff_head_depth 6 \
    --diff_head_width 1024 \
    --diff_head_num_sampling_steps 100 \
    --diff_head_batch_mul 4 \
    --diff_head_temperature 1.0 \
    --diff_head_cond_drop_prob 0.1 \
    --diff_head_cfg 3.0 \
    --cfg_schedule linear \
    --mixed_precision bf16 \
    --log_with tensorboard \
    --seed 42 \
    ${RESUME_DIR:+--resume_dir $RESUME_DIR}
