#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  CLEVR — Discrete-Diffusion (text-conditioned) ckpt evaluation
#
#  Evaluates the two text-conditioned discrete-diffusion runs against
#  the same CLEVR detector/classifier metrics that
#  naive_dit_256_text_cond_clip already logs to its `clevr_eval/*.json`:
#    - count_accuracy
#    - entity_presence_accuracy
#    - rel_accuracy
#
#  Targets (TEXT condition, clevr_256_varied):
#    runs/clevr/discrete_diff_ours_text                 (multires pos_emb)
#    runs/clevr/discrete_diff_semanticist_text_w_decay0_larger_batch
#                                                       (1d pos_emb,
#                                                        semanticist backbone)
#
#  Usage:
#    bash script/eval_discrete_diff_text_clevr.sh
#    GPUS=0,1 RUN=ours bash script/eval_discrete_diff_text_clevr.sh
#    RUN=semanticist bash script/eval_discrete_diff_text_clevr.sh
# ──────────────────────────────────────────────────────────────────

set -e

export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# ── Data (must match training: cond_type=text → conditions_text) ──
TRAIN_IMG="../clevr-dataset-gen/output/clevr_256_varied/images"
TRAIN_COND="../clevr-dataset-gen/output/clevr_256_varied/conditions_text"
VAL_IMG="../clevr-dataset-gen/output/clevr_256_varied_val/images"
VAL_COND="../clevr-dataset-gen/output/clevr_256_varied_val/conditions_text"

EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-100}"
EVAL_NUM_STEPS="${EVAL_NUM_STEPS:-128}"
DECODE_NUM_STEPS="${DECODE_NUM_STEPS:-50}"

run_eval () {
    local NAME="$1"
    local RUN_DIR="$2"
    local PRETRAINED="$3"
    local POS_EMB="$4"
    local STEPS="$5"

    local CKPT_ROOT="${RUN_DIR}/ckpt"
    local OUT_DIR="${RUN_DIR}/ckpt_eval_results"
    local TOK_CACHE="${RUN_DIR}/token_cache"

    echo "=========================================================="
    echo "[eval] ${NAME}"
    echo "  run_dir   = ${RUN_DIR}"
    echo "  pretrained= ${PRETRAINED}"
    echo "  pos_emb   = ${POS_EMB}"
    echo "  steps     = ${STEPS}"
    echo "  out       = ${OUT_DIR}"
    echo "=========================================================="

    accelerate launch \
        --num_processes ${NUM_GPUS} \
        --multi_gpu \
        src/eval_discrete_diffusion_ckpts.py \
        --ckpt_root "${CKPT_ROOT}" \
        --steps ${STEPS} \
        --pretrained_dir "${PRETRAINED}" \
        --output_dir "${OUT_DIR}" \
        --train_image_root "${TRAIN_IMG}" \
        --train_cond_dir "${TRAIN_COND}" \
        --val_image_root "${VAL_IMG}" \
        --val_cond_dir "${VAL_COND}" \
        --splits easy medium hard \
        --token_cache_dir "${TOK_CACHE}" \
        --cond_type text \
        --pos_emb_type "${POS_EMB}" \
        --hidden_size 768 \
        --n_heads 12 \
        --n_blocks 12 \
        --cond_dim 256 \
        --mlp_ratio 4 \
        --model_dropout 0.1 \
        --eval_num_samples ${EVAL_NUM_SAMPLES} \
        --eval_num_steps ${EVAL_NUM_STEPS} \
        --decode_num_steps ${DECODE_NUM_STEPS} \
        --image_size 256 \
        --sampler ddpm_cache \
        --mixed_precision bf16 \
        --seed 42
}

RUN="${RUN:-all}"

if [[ "$RUN" == "ours" || "$RUN" == "all" ]]; then
    run_eval "discrete_diff_ours_text" \
        "./runs/clevr/discrete_diff_ours_text" \
        "./runs/clevr/backbone/clevr_256_dit_vit_flow_fsq_mask0_CA" \
        "multires" \
        "50000 100000 150000 200000"
fi

if [[ "$RUN" == "semanticist" || "$RUN" == "all" ]]; then
    run_eval "discrete_diff_semanticist_text_w_decay0_larger_batch" \
        "./runs/clevr/discrete_diff_semanticist_text_w_decay0_larger_batch" \
        "./runs/clevr/backbone/clevr_256_dit_baseline_1d_semanticist" \
        "1d" \
        "50000 100000 150000"
fi

echo
echo "[done] Per-run summaries: <run_dir>/ckpt_eval_results/eval_summary.json"
echo "       Compare against:   runs/clevr/naive_dit_256_text_cond_clip/clevr_eval/*.json"
