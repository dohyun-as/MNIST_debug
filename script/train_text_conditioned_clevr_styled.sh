#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  CLEVR 256×256 — Text-Conditioned DiT (Baseline)
#  trained on **styled (coverage-complete) caption families**.
#
#  Difference vs. train_text_conditioned_clevr.sh:
#    Caption source switched from `conditions_text/` (legacy fixed
#    template) to `conditions_text_styled/` (per-image JSON with
#    six prompt families: C, E, R, C+E, C+R, E+R).
#
#  No new CLI flag is needed: the trainer auto-detects styled captions
#  by inspecting whether each loaded caption dict carries an `exposed`
#  mask, and switches eval to `eval_clevr_complex_text` (family-aware
#  metrics — count_acc / entity_inv_acc / rel_acc per (split, family)).
#
#  Eval set selection is seeded (random.Random(42) in
#  _select_eval_indices_balanced), so the SAME val indices are picked
#  regardless of GPU count or host.
#
#  Usage:
#    bash script/train_text_conditioned_clevr_styled.sh
#    GPUS=0,1 bash script/train_text_conditioned_clevr_styled.sh
#
#  Detached (survives session disconnect):
#    setsid nohup bash script/train_text_conditioned_clevr_styled.sh \
#        > runs/clevr/naive_dit_styled/run.log 2>&1 < /dev/null &
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
CLEVR_IMAGE_ROOT=${CLEVR_IMAGE_ROOT:-"../clevr-dataset-gen/output/clevr_256_varied_merged/images"}
CLEVR_COND_DIR=${CLEVR_COND_DIR:-"../clevr-dataset-gen/output/clevr_256_varied_merged/conditions_text_styled"}
CLEVR_VAL_IMAGE_ROOT=${CLEVR_VAL_IMAGE_ROOT:-"../clevr-dataset-gen/output/clevr_256_varied_val/images"}
CLEVR_VAL_COND_DIR=${CLEVR_VAL_COND_DIR:-"../clevr-dataset-gen/output/clevr_256_varied_val/conditions_text_styled"}
OUTPUT_DIR=${OUTPUT_DIR:-"runs/clevr/naive_dit_256_text_cond_clip_styled"}

mkdir -p "${OUTPUT_DIR}"

accelerate launch \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    src/train_text_conditioned.py \
    --output_dir "$OUTPUT_DIR" \
    --clevr_image_root "$CLEVR_IMAGE_ROOT" \
    --clevr_condition_dir "$CLEVR_COND_DIR" \
    --clevr_val_image_root "$CLEVR_VAL_IMAGE_ROOT" \
    --clevr_val_condition_dir "$CLEVR_VAL_COND_DIR" \
    --clevr_train_splits easy medium hard \
    --clevr_val_splits easy medium hard \
    --clevr_cond_type text \
    --image_size 256 \
    --in_channels 3 \
    --encoder_mode pretrained \
    --pretrained_model_name openai/clip-vit-base-patch32 \
    --pretrained_max_length 77 \
    --unfreeze_text_encoder \
    --dit_patch_size 16 \
    --dit_hidden_size 768 \
    --dit_n_heads 12 \
    --dit_n_blocks 12 \
    --dit_mlp_ratio 4.0 \
    --dit_dropout 0.0 \
    --dit_bottleneck_dim 128 \
    --dit_in_context_len 32 \
    --dit_in_context_start 4 \
    --use_flow_matching \
    --flow_P_mean -0.8 \
    --flow_P_std 0.8 \
    --flow_t_eps 0.05 \
    --flow_noise_scale 1.0 \
    --flow_sampling_method euler \
    --max_train_steps 200000 \
    --batch_size $BATCH_PER_GPU \
    --blr 2.5e-5 \
    --weight_decay 0.0 \
    --warmup_steps 5000 \
    --max_grad_norm 3.0 \
    --grad_accum_steps $GRAD_ACCUM \
    --mixed_precision bf16 \
    --ema_decay 0 \
    --uncond_drop_prob 0.1 \
    --guidance_scale 3.0 \
    --log_every 100 \
    --save_every 50000 \
    --sample_every 5000 \
    --eval_num_steps 50 \
    --num_workers 8 \
    --seed 42 \
    --clevr_eval_every 10000 \
    --clevr_eval_samples 30
