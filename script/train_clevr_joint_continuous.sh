#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  CLEVR 256×256 — JOINT TRAINING
#    stage 1 (encoder + DiT, flow matching, continuous feat dim=16)
#    stage 2 (masked discrete diffusion + diff head on continuous tokens,
#             CLIP text-conditioned)
#  in a single optimizer step.
#
#  Stage 2 target tokens come from an EMA teacher of stage 1 (stop-grad).
#  Stage 2 visible context tokens come from the live student encoder, so
#  the masked-prediction loss flows back into the encoder.
#
#  Usage:
#    bash script/train_clevr_joint_continuous.sh
#    GPUS=0,1 bash script/train_clevr_joint_continuous.sh
#
#  Notes:
#    * existing pipelines (train_clevr_dit_our_continuous.sh and
#      train_discrete_diffusion_clevr_ours_text_diffhead_clip.sh) are
#      untouched — this script runs a NEW python entry point
#      src/main_joint_training.py.
#    * --batch_size is per-GPU. Joint step does ~3 encoder forwards
#      (stage1 + student + teacher), so memory footprint is higher than
#      either stage alone — start with a smaller batch than stage 1.
# ──────────────────────────────────────────────────────────────────

set -e

# ── GPU / batch ──
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

BATCH_PER_GPU=${BATCH_PER_GPU:-128}
GRAD_ACCUM=${GRAD_ACCUM:-1}

echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS)  batch/gpu=$BATCH_PER_GPU  accum=$GRAD_ACCUM  effective=$((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))"

# ── data paths (mirror the existing two scripts) ──
CLEVR_TRAIN_IMAGES=${CLEVR_TRAIN_IMAGES:-"../clevr_output/clevr_256_varied/images"}
CLEVR_VAL_IMAGES=${CLEVR_VAL_IMAGES:-"../clevr_output/clevr_256_varied_val/images"}
CLEVR_TRAIN_COND=${CLEVR_TRAIN_COND:-"../clevr_output/clevr_256_varied/conditions_text"}
CLEVR_VAL_COND=${CLEVR_VAL_COND:-"../clevr_output/clevr_256_varied_val/conditions_text"}

OUTPUT_DIR=${OUTPUT_DIR:-"./runs/clevr/joint/256_dit_vit_flow_cont_out16_clip"}
mkdir -p "${OUTPUT_DIR}"

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

ARGS="
  --output_dir ${OUTPUT_DIR}
  --train_dir ${CLEVR_TRAIN_IMAGES}
  --val_dir ${CLEVR_VAL_IMAGES}
  --clevr_condition_dir ${CLEVR_TRAIN_COND}
  --clevr_val_condition_dir ${CLEVR_VAL_COND}
  --clevr_train_splits easy medium hard
  --clevr_val_splits easy medium hard
  --clevr_cond_type text

  --image_size 256
  --in_channels 3
  --vae_downsample_factor 1

  --encoder_type vit
  --min_patch_size 32
  --feat_channels 16
  --encoder_internal_dim 256
  --depth_per_level 2
  --cnn_base_channels 64

  --dit_patch_size 16
  --dit_hidden_size 768
  --dit_n_heads 12
  --dit_n_blocks 12
  --dit_mlp_ratio 4.0
  --dit_dropout 0.0
  --dit_bottleneck_dim 128
  --dit_in_context_len 32
  --dit_in_context_start 4

  --use_flow_matching
  --flow_P_mean -0.8
  --flow_P_std 0.8
  --flow_t_eps 0.05
  --flow_noise_scale 1.0
  --flow_sampling_method euler

  --uncond_drop_prob 0.1
  --level_drop
  --min_keep_levels 1
  --level_drop_after_steps 10000
  --guidance_scale 3.0

  --s2_hidden_size 768
  --s2_n_heads 12
  --s2_n_blocks 12
  --s2_cond_dim 256
  --s2_mlp_ratio 4
  --s2_dropout 0.1
  --s2_pos_emb_type multires
  --s2_noise_type loglinear
  --s2_uncond_drop_prob 0.1

  --diff_head_depth 6
  --diff_head_width 1024
  --diff_head_num_sampling_steps 100
  --diff_head_batch_mul 4
  --diff_head_cond_drop_prob 0.0
  --diff_head_temperature 1.0
  --diff_head_cfg 3.0
  --cfg_schedule linear
  --cfg_mode backbone
  --sampler ddpm_cache

  --use_pretrained_text_encoder
  --pretrained_text_model_name openai/clip-vit-base-patch32
  --pretrained_text_max_length 77
  --unfreeze_text_encoder
  --text_encoder_lr 3e-5

  --lambda_stage1 1.0
  --lambda_stage2 1.0
  --ema_decay 0.9995
  --stage2_warmup_steps 0

  --max_train_steps 200000
  --batch_size ${BATCH_PER_GPU}
  --blr 2.5e-5
  --weight_decay 0.05
  --warmup_steps 5000
  --lr_schedule constant
  --max_grad_norm 3.0
  --grad_accum_steps ${GRAD_ACCUM}
  --mixed_precision bf16

  --log_every 100
  --save_every 10000
  --sample_every 5000

  --clevr_eval_every 5000
  --clevr_eval_samples 50
  --eval_every 5000
  --eval_num_samples 30
  --eval_num_steps 128
  --decode_num_steps 50
  --eval_sample_batch_size 8

  --num_workers 8
  --seed 42
"

if [ ${NUM_GPUS} -gt 1 ]; then
  accelerate launch \
    --main_process_port $PORT \
    --multi_gpu \
    --num_processes ${NUM_GPUS} \
    src/main_joint_training.py ${ARGS}
else
  accelerate launch \
    --main_process_port $PORT \
    src/main_joint_training.py ${ARGS}
fi
