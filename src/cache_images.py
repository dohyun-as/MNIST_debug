"""Single-process pre-builder for the local memmap image cache.

When NAS reads stall, the multi-GPU launcher hits NCCL watchdog timeouts
during cache build (only the main rank encodes images, the other ranks sit
on wait_for_everyone past 30 min and the whole job dies). Run this once
beforehand to warm the cache; subsequent training runs will reuse it
instantly because the layout is identical.

Usage (from MNIST_debug/):
    python src/cache_images.py \\
        --train_dir ../clevr-dataset-gen/output/clevr_256_varied/images \\
        --val_dir   ../clevr-dataset-gen/output/clevr_256_varied_val/images \\
        --image_size 256 \\
        --local_cache_dir /workspace/cache \\
        --max_train_samples_per_class 30000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_multires import (
    _resolve_local_cache_subdir,
    build_memmap_image_cache,
)


class _SoloAccelerator:
    """Stand-in for accelerate.Accelerator on a single process."""
    is_main_process = True
    num_processes = 1
    process_index = 0

    def print(self, *args, **kwargs):
        print(*args, **kwargs)

    def wait_for_everyone(self):
        pass


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--val_dir", type=str, default=None)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--local_cache_dir", type=str, default="/workspace/cache")
    p.add_argument("--max_train_samples_per_class", type=int, default=30000,
                   help="Match the training script. Pass 0 to disable cap.")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_val", action="store_true")
    args = p.parse_args()

    cap = args.max_train_samples_per_class or None
    accel = _SoloAccelerator()

    if not args.skip_train:
        train_sub = _resolve_local_cache_subdir(args, args.train_dir, max_per_class=cap)
        meta = build_memmap_image_cache(
            args.train_dir, train_sub, args.image_size, accel, "train",
            max_per_class=cap,
        )
        print(f"[cache] train ready: {meta['bin_path']} ({meta['num_images']} imgs)")

    if args.val_dir and not args.skip_val:
        val_sub = _resolve_local_cache_subdir(args, args.val_dir)
        val_meta = build_memmap_image_cache(
            args.val_dir, val_sub, args.image_size, accel, "val",
        )
        print(f"[cache] val ready: {val_meta['bin_path']} ({val_meta['num_images']} imgs)")


if __name__ == "__main__":
    main()
