#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  Condition token space t-SNE (per checkpoint step)
#
#  Targets (space-separated list of run dirs):
#    - dit_9x9_cont_out16                (baseline, no cond noise)
#    - dit_9x9_cont_out16_noise_rel0.3   (rel noise 0.3)
#
#  Produces TWO variants per run:
#    - tsne_per_step_ema/   (EMA weights   — what diffusion sampling uses)
#    - tsne_per_step_raw/   (non-EMA "model" key — what the downstream
#                            train_discrete_diffusion_v2.py actually loads
#                            via ckpt.get("model", ...))
#
#  4-GPU checkpoint-wise sharding for per-step plots, then a serial
#  final pass for the combined grid.
# ──────────────────────────────────────────────────────────────────

set -e

RUN_DIRS=(
    "runs/sudoku/dit_9x9_cont_out16"
    "runs/sudoku/dit_9x9_cont_out16_noise_rel0.3"
)
DATA_DIR="datasets/mnist_sudoku"
PY=/opt/conda/bin/python

cd "$(dirname "$0")/.."

run_variant () {
    local run_dir="$1"
    local ema_flag="$2"    # "--use_ema" or "--no_ema"
    local out_subdir="$3"  # "tsne_per_step_ema" or "tsne_per_step_raw"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  $run_dir  |  $ema_flag  →  $out_subdir"
    echo "════════════════════════════════════════════════════════════"

    local STEPS=($(ls "$run_dir/checkpoints" | sed -E 's/step_0*([0-9]+)/\1/' | sort -n))
    echo "Steps: ${STEPS[@]}"

    local SHARD0=() SHARD1=() SHARD2=() SHARD3=()
    for i in "${!STEPS[@]}"; do
        case $((i % 4)) in
            0) SHARD0+=("${STEPS[$i]}");;
            1) SHARD1+=("${STEPS[$i]}");;
            2) SHARD2+=("${STEPS[$i]}");;
            3) SHARD3+=("${STEPS[$i]}");;
        esac
    done

    mkdir -p "$run_dir/$out_subdir"

    _spawn () {
        local gpu=$1; shift
        local steps=("$@")
        [[ ${#steps[@]} -eq 0 ]] && return 0
        CUDA_VISIBLE_DEVICES=$gpu $PY src/visualize_encoder_tsne_per_step.py \
            --run_dir "$run_dir" \
            --data_dir "$DATA_DIR" \
            --steps "${steps[@]}" \
            --max_samples 5000 \
            --skip_grid \
            --out_subdir "$out_subdir" \
            $ema_flag \
            > "$run_dir/$out_subdir/shard_gpu${gpu}.log" 2>&1 &
        echo "  GPU$gpu (pid $!) steps=${steps[*]}"
    }

    echo "─── Stage 1: per-step (4-way parallel) ───"
    _spawn 0 "${SHARD0[@]}"
    _spawn 1 "${SHARD1[@]}"
    _spawn 2 "${SHARD2[@]}"
    _spawn 3 "${SHARD3[@]}"
    wait
    echo "  shards done."

    echo "─── Stage 2: combined grid (GPU0) ───"
    CUDA_VISIBLE_DEVICES=0 $PY src/visualize_encoder_tsne_per_step.py \
        --run_dir "$run_dir" \
        --data_dir "$DATA_DIR" \
        --max_samples 5000 \
        --out_subdir "$out_subdir" \
        $ema_flag \
        2>&1 | tee "$run_dir/$out_subdir/grid.log"
}

for rd in "${RUN_DIRS[@]}"; do
    run_variant "$rd" "--use_ema" "tsne_per_step_ema"
    run_variant "$rd" "--no_ema"   "tsne_per_step_raw"
done

echo ""
echo "DONE."
for rd in "${RUN_DIRS[@]}"; do
    echo "  $rd/tsne_per_step_ema/"
    echo "  $rd/tsne_per_step_raw/"
done
