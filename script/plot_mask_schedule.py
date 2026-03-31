#!/usr/bin/env python3
"""Plot expected mask ratio/count for loglinear noise schedule.

This script does not require project imports; it is standalone.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def compute_mask_ratio(t, eps):
    # For loglinear schedule: mask_ratio = (1 - eps) * t
    return (1.0 - eps) * t


def main():
    parser = argparse.ArgumentParser(description="Plot loglinear mask schedule.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps.")
    parser.add_argument("--eps", type=float, default=1e-3, help="Noise epsilon.")
    parser.add_argument("--seq_len", type=int, default=81, help="Sequence length.")
    parser.add_argument(
        "--sampling_eps",
        type=float,
        default=1e-3,
        help="Minimum t used during training sampling.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/mask_schedule.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    t_start = args.sampling_eps
    t_end = 1.0
    t = np.linspace(t_start, t_end, args.steps)
    mask_ratio = compute_mask_ratio(t, args.eps)
    mask_count = mask_ratio * args.seq_len

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(t, mask_ratio, label="mask ratio", color="#1f77b4")
    ax1.set_xlabel("t")
    ax1.set_ylabel("mask ratio", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0, 1.0)

    ax2 = ax1.twinx()
    ax2.plot(t, mask_count, label="masked count", color="#ff7f0e")
    ax2.set_ylabel("masked count (of seq_len)", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax2.set_ylim(0, args.seq_len)

    fig.suptitle(
        f"Loglinear mask schedule (eps={args.eps}, seq_len={args.seq_len})"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
