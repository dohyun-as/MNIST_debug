#!/bin/bash
# Smoke test for the new annotated bbox grid saving in evaluate_clevr().
# Resumes from the latest ckpt in the run dir and runs ONE clevr eval pass,
# saving:
#   runs/.../clevr_eval/step_XXXXXXX_annotated_random.png
#   runs/.../clevr_eval/step_XXXXXXX_annotated_worst.png
set -e

export CUDA_VISIBLE_DEVICES="${GPUS:-0}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

VENV_BIN="/NHNHOME/WORKSPACE/0226010398_A/sr_diffusion/clevr_sudoku/MNIST_debug/venv/bin"
CLEVR_DIR="../clevr-dataset-gen/output/clevr_256_varied/images"
CLEVR_VAL="../clevr-dataset-gen/output/clevr_256_varied_val/images"

"$VENV_BIN/accelerate" launch \
    --num_processes $NUM_GPUS \
    src/main_multires.py \
    --eval_clevr_only \
    --backbone dit \
    --output_dir runs/clevr/backbone/out16_randomdrop_alllvl_multi_res \
    --train_dir "$CLEVR_DIR" \
    --val_dir "$CLEVR_VAL" \
    --dataset_root "$CLEVR_DIR" \
    --image_size 256 \
    --in_channels 3 \
    --vae_downsample_factor 1 \
    --min_patch_size 32 \
    --feat_channels 16 \
    --encoder_internal_dim 256 \
    --depth_per_level 2 \
    --cnn_base_channels 64 \
    --encoder_type vit \
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
    --batch_size 32 \
    --blr 2.5e-5 \
    --weight_decay 0.05 \
    --warmup_steps 5000 \
    --max_grad_norm 3.0 \
    --grad_accum_steps 1 \
    --mixed_precision bf16 \
    --ema_decay 0 \
    --uncond_drop_prob 0.1 \
    --no_level_drop \
    --cond_token_drop_prob 1.0 \
    --cond_token_drop_all_levels \
    --guidance_scale 3.0 \
    --eval_num_steps 50 \
    --num_workers 4 \
    --seed 42 \
    --mae_mask_ratio 0.0 \
    --clevr_eval_samples 30 \
    --clevr_eval_n_annotated_random 8 \
    --clevr_eval_n_annotated_worst 4 \
    --clevr_eval_annot_thresh 10
