"""
Visualize continuous encoder token distribution (t-SNE) across training steps.

For models WITHOUT FSQ (continuous feat_channels output).
Loads each checkpoint, extracts 16D encoder features, runs t-SNE colored by digit class.
Produces:
  1) Individual t-SNE plot per step
  2) Combined grid showing evolution across steps
"""

import sys, os, json, argparse
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

# ── project imports ──
sys.path.insert(0, os.path.dirname(__file__))
from multi_res_encoder import HierarchicalMultiResEncoder
from visualize_token_space import SudokuImageDataset

# ─────────────────────────────────────────────────────────
#  Colors
# ─────────────────────────────────────────────────────────

DIGIT_COLORS = {
    1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd',
    5: '#8c564b', 6: '#e377c2', 7: '#7f7f7f', 8: '#bcbd22', 9: '#17becf',
}


# ─────────────────────────────────────────────────────────
#  Build encoder from args.json
# ─────────────────────────────────────────────────────────

def build_encoder(args_dict):
    return HierarchicalMultiResEncoder(
        in_channels=args_dict["cond_in_channels"],
        dim=args_dict["feat_channels"],
        image_size=args_dict["image_size"],
        min_patch_size=args_dict["min_patch_size"],
        num_levels=args_dict.get("num_levels", None),
        depth_per_level=args_dict.get("depth_per_level", 2),
        mlp_ratio=args_dict.get("mlp_ratio", 4.0),
        cnn_base_channels=args_dict.get("cnn_base_channels", 64),
        mae_mask_ratio=0.0,
        encoder_type=args_dict.get("encoder_type", "vit"),
        encoder_internal_dim=args_dict.get("encoder_internal_dim", None),
        vit_patch_size=args_dict.get("vit_patch_size", 4),
        vit_depth=args_dict.get("vit_depth", 4),
        vit_num_heads=args_dict.get("vit_num_heads", 4),
        vit_mlp_ratio=args_dict.get("vit_mlp_ratio", 4.0),
        vit_use_cnn_stem=args_dict.get("vit_use_cnn_stem", True) and not args_dict.get("vit_no_cnn_stem", False),
        vit_cnn_stem_reduction=args_dict.get("vit_cnn_stem_reduction", 4),
        level_sizes=args_dict.get("level_sizes", None),
    )


def load_encoder_weights(encoder, ckpt_path):
    """Load encoder weights from full model checkpoint. Returns step number."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Try EMA first, fall back to regular model
    state = ckpt.get("ema", ckpt.get("model", {}))

    enc_state = OrderedDict()
    for k, v in state.items():
        if k.startswith("encoder."):
            enc_state[k[len("encoder."):]] = v
    encoder.load_state_dict(enc_state, strict=True)

    step = ckpt.get("step", "?")
    return step


# ─────────────────────────────────────────────────────────
#  Extract continuous encoder features
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(encoder, dataloader, device, max_samples=5000):
    """Extract continuous encoder features and digit labels.

    Returns:
        features: (N, feat_dim) numpy array
        digit_labels: (N,) numpy array with values 1-9
    """
    encoder.eval().to(device)

    feat_list, digit_list = [], []
    n_collected = 0

    for images, grid_labels in dataloader:
        if n_collected >= max_samples:
            break
        images = images.to(device)

        level_features = encoder.forward_injection(images)

        for s, feat_2d in level_features.items():
            B, D, H, W = feat_2d.shape
            tokens = feat_2d.flatten(2).transpose(1, 2)  # (B, H*W, D)

            feat_list.append(tokens.reshape(-1, D).cpu())
            digit_list.append(grid_labels.reshape(-1))
            n_collected += B * H * W

    features = torch.cat(feat_list, 0)[:max_samples].numpy()
    digits = torch.cat(digit_list, 0)[:max_samples].numpy()

    return features, digits


# ─────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────

def plot_single_tsne(emb, digit_labels, step, ax, title=None):
    """Plot t-SNE embedding on a given axis, colored by digit class."""
    for d in sorted(DIGIT_COLORS.keys()):
        mask = digit_labels == d
        if mask.sum() == 0:
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=DIGIT_COLORS[d], s=8, alpha=0.5,
                   label=f"{d} ({mask.sum()})", rasterized=True)
    ax.set_title(title or f"Step {step}", fontsize=12, fontweight='bold')
    ax.set_xlabel("t-SNE 1", fontsize=9)
    ax.set_ylabel("t-SNE 2", fontsize=9)
    ax.tick_params(labelsize=7)


def run_tsne(features, perplexity=30, random_state=42):
    return TSNE(
        n_components=2, perplexity=perplexity,
        random_state=random_state, max_iter=1000
    ).fit_transform(features)


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize encoder t-SNE across training steps (continuous output)")
    parser.add_argument("--run_dir", required=True, help="Path to run directory")
    parser.add_argument("--data_dir", required=True, help="Path to MNIST sudoku dataset")
    parser.add_argument("--steps", nargs="*", type=int, default=None,
                        help="Specific steps to visualize (default: all checkpoints)")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_ema", action="store_true", default=True,
                        help="Use EMA weights (default: True)")
    parser.add_argument("--no_ema", dest="use_ema", action="store_false")
    parser.add_argument("--ncols", type=int, default=5,
                        help="Number of columns in grid plot")
    parser.add_argument("--skip_grid", action="store_true",
                        help="Skip combined grid plot (useful for sharded runs)")
    parser.add_argument("--out_subdir", type=str, default="tsne_per_step",
                        help="Subdirectory (under run_dir) to save plots into")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_name = run_dir.name

    # Load config
    with open(run_dir / "args.json") as f:
        args_dict = json.load(f)

    feat_dim = args_dict["feat_channels"]
    print(f"\n{'='*60}")
    print(f"  {run_name} — Continuous encoder ({feat_dim}D)")
    print(f"{'='*60}")

    # Find checkpoints
    ckpt_dir = run_dir / "checkpoints"
    all_ckpt_dirs = sorted(ckpt_dir.iterdir())

    if args.steps:
        selected = []
        for s in args.steps:
            ckpt_name = f"step_{s:07d}"
            ckpt_path = ckpt_dir / ckpt_name / "checkpoint.pt"
            if ckpt_path.exists():
                selected.append(ckpt_path)
            else:
                print(f"  WARNING: {ckpt_path} not found, skipping")
        ckpt_paths = selected
    else:
        ckpt_paths = [d / "checkpoint.pt" for d in all_ckpt_dirs if (d / "checkpoint.pt").exists()]

    if not ckpt_paths:
        print("No checkpoints found!")
        return

    print(f"  Found {len(ckpt_paths)} checkpoints")

    # Build dataset (once, reused across steps)
    dataset = SudokuImageDataset(
        root=args.data_dir,
        image_size=args_dict["image_size"],
        top_n=100, max_grids=500
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        generator=torch.Generator().manual_seed(42)  # reproducible sampling
    )

    # Output directory
    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(exist_ok=True)

    # ── Extract features & run t-SNE for each checkpoint ──
    all_embeddings = []  # (step, emb, digits)

    for ckpt_path in ckpt_paths:
        encoder = build_encoder(args_dict)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if args.use_ema and "ema" in ckpt:
            state = ckpt["ema"]
            weight_label = "EMA"
        else:
            state = ckpt.get("model", {})
            weight_label = "model"

        enc_state = OrderedDict()
        for k, v in state.items():
            if k.startswith("encoder."):
                enc_state[k[len("encoder."):]] = v
        encoder.load_state_dict(enc_state, strict=True)

        step = ckpt.get("step", "?")
        print(f"\n  Step {step} ({weight_label}):")

        # Extract features
        features, digits = extract_features(
            encoder, loader, args.device, max_samples=args.max_samples)
        # Filter out digit 0 (background/padding) if present
        mask = digits > 0
        features, digits = features[mask], digits[mask]
        print(f"    {len(features)} tokens, dim={features.shape[1]}")

        # Run t-SNE
        print(f"    Running t-SNE (perplexity={args.perplexity})...")
        emb = run_tsne(features, perplexity=args.perplexity)

        all_embeddings.append((step, emb, digits))

        # Save individual plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        plot_single_tsne(emb, digits, step, ax)
        ax.legend(fontsize=8, ncol=2, title="Digit", markerscale=2, loc='upper right')
        fig.suptitle(
            f"Sudoku {run_name} (step {step}) — {weight_label}\n"
            f"Continuous encoder output ({feat_dim}D) — by digit class",
            fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(str(out_dir / f"tsne_step_{step:07d}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved individual plot")

        # Free memory
        del encoder, ckpt, state
        torch.cuda.empty_cache() if args.device == "cuda" else None

    if args.skip_grid:
        print(f"  Skipping grid plot (--skip_grid)")
        return

    # ── Combined grid plot ──
    n = len(all_embeddings)
    ncols = min(args.ncols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, (step, emb, digits) in enumerate(all_embeddings):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        plot_single_tsne(emb, digits, step, ax, title=f"Step {step:,}")

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    # Add single legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=9,
               fontsize=10, markerscale=2, title="Digit",
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Encoder t-SNE evolution — {run_name}\n"
        f"Continuous {feat_dim}D output, colored by digit class",
        fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    grid_path = str(out_dir / "tsne_grid_all_steps.png")
    fig.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Grid plot saved -> {grid_path}")
    print(f"  All individual plots saved in {out_dir}/")


if __name__ == "__main__":
    main()
