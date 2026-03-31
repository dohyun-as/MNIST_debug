#!/usr/bin/env python3
"""Consolidate individual .pt cache files into a single stacked tensor.

Usage:
    python consolidate_cache.py <cache_dir> [--latent_only]

Example:
    python consolidate_cache.py runs/imagenet_256_injection/latent_cache --latent_only
"""
import argparse
import os
import sys

import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=str)
    parser.add_argument("--latent_only", action="store_true")
    args = parser.parse_args()

    suffix = "_latent_only" if args.latent_only else ""
    out_path = os.path.join(args.cache_dir, f"consolidated{suffix}.pt")

    if os.path.isfile(out_path):
        print(f"Already exists: {out_path}")
        sys.exit(0)

    files = sorted([f for f in os.listdir(args.cache_dir)
                    if f.endswith('.pt') and not f.startswith('consolidated')])
    print(f"Consolidating {len(files)} files → {out_path}")

    latents, latents_flip = [], []
    images, images_flip = ([], []) if not args.latent_only else (None, None)

    for f in tqdm(files, desc="Consolidate"):
        data = torch.load(os.path.join(args.cache_dir, f),
                          map_location='cpu', weights_only=True)
        latents.append(data['latent'])
        latents_flip.append(data['latent_flip'])
        if not args.latent_only:
            images.append(data['image'])
            images_flip.append(data['image_flip'])

    out = {
        'latent': torch.stack(latents),
        'latent_flip': torch.stack(latents_flip),
    }
    if not args.latent_only:
        out['image'] = torch.stack(images)
        out['image_flip'] = torch.stack(images_flip)

    print(f"Saving ({out['latent'].shape}) ...")
    torch.save(out, out_path)
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
