"""
Visualize token space before and after FSQ quantization.

Two modes:
  - Sudoku: color by digit class (0-9) per patch, single level
  - CLEVR:  codebook utilization analysis + multi-level embedding viz
"""

import sys, os, json, argparse, math
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import Dataset as TorchDataset

# ── project imports ──
sys.path.insert(0, os.path.dirname(__file__))
from multi_res_encoder import HierarchicalMultiResEncoder
from Discretizer import FSQDiscretizer


# ─────────────────────────────────────────────────────────
#  Model building helpers
# ─────────────────────────────────────────────────────────

def build_encoder_and_discretizer(args_dict):
    """Build encoder + discretizer from saved args.json dict."""
    image_size = args_dict["image_size"]
    cond_in_channels = args_dict["cond_in_channels"]
    feat_channels = args_dict["feat_channels"]
    min_patch_size = args_dict["min_patch_size"]
    level_sizes = args_dict.get("level_sizes", None)

    encoder = HierarchicalMultiResEncoder(
        in_channels=cond_in_channels,
        dim=feat_channels,
        image_size=image_size,
        min_patch_size=min_patch_size,
        num_levels=args_dict.get("num_levels", None),
        depth_per_level=args_dict.get("depth_per_level", 2),
        mlp_ratio=args_dict.get("mlp_ratio", 4.0),
        cnn_base_channels=args_dict.get("cnn_base_channels", 64),
        mae_mask_ratio=0.0,
        encoder_type=args_dict.get("encoder_type", "vit"),
        vit_patch_size=args_dict.get("vit_patch_size", 4),
        vit_depth=args_dict.get("vit_depth", 4),
        vit_num_heads=args_dict.get("vit_num_heads", 4),
        vit_mlp_ratio=args_dict.get("vit_mlp_ratio", 4.0),
        vit_use_cnn_stem=args_dict.get("vit_use_cnn_stem", True) and not args_dict.get("vit_no_cnn_stem", False),
        vit_cnn_stem_reduction=args_dict.get("vit_cnn_stem_reduction", 4),
        level_sizes=level_sizes,
    )

    fsq_levels = args_dict["fsq_levels"]
    discretizer = FSQDiscretizer(
        slot_dim=feat_channels,
        levels=fsq_levels,
        drop_quant_p=0.0,
        corrupt_tokens_p=0.0,
    )

    return encoder, discretizer


def load_weights(encoder, discretizer, ckpt_path):
    """Load encoder + discretizer weights from full model checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]

    enc_state = OrderedDict()
    for k, v in state.items():
        if k.startswith("encoder."):
            enc_state[k[len("encoder."):]] = v
    encoder.load_state_dict(enc_state, strict=True)

    disc_state = OrderedDict()
    for k, v in state.items():
        if k.startswith("discretizer."):
            disc_state[k[len("discretizer."):]] = v
    discretizer.load_state_dict(disc_state, strict=True)

    step = ckpt.get("step", "?")
    print(f"  Loaded step {step} from {ckpt_path}")
    return step


# ─────────────────────────────────────────────────────────
#  Sudoku dataset (returns digit labels per patch)
# ─────────────────────────────────────────────────────────

class SudokuImageDataset(TorchDataset):
    """Generates 252x252 MNIST sudoku grids. Returns (image, grid_values)."""

    def __init__(self, root, image_size=288, top_n=100, max_grids=500):
        import torchvision
        import pandas as pd

        root = Path(root)
        self.image_size = image_size

        mnist_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True)
        top_csv = pd.read_csv(root / "top_5000_values.csv")

        chosen = []
        for label in range(10):
            sel = top_csv[top_csv.label == label].sort_values("confidence", ascending=False).iloc[:top_n]
            indices = sel["image_index"].values
            all_imgs = mnist_dataset.data[mnist_dataset.targets == label]
            chosen.append(all_imgs[indices].float())
        self.mnist_images = torch.stack(chosen, dim=0)  # (10, top_n, 28, 28)

        all_grids = np.load(root / "sudokus.npy")
        self.grids = torch.tensor(all_grids[:max_grids])

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        grid = self.grids[idx]  # (9, 9) with values 0-9
        rng = np.random.default_rng(idx)
        full_image = torch.empty(252, 252, dtype=torch.uint8)
        for r in range(9):
            for c in range(9):
                cands = self.mnist_images[int(grid[r, c])]
                i = rng.integers(0, cands.size(0))
                full_image[r*28:(r+1)*28, c*28:(c+1)*28] = cands[i].byte()

        from torchvision.transforms import functional as TF
        img = TF.to_pil_image(full_image)
        img = TF.resize(img, [self.image_size, self.image_size])
        img = TF.to_tensor(img)
        img = (img - 0.5) / 0.5
        # Return grid flattened as (81,) digit labels
        return img, grid.reshape(-1).long()  # (1, H, W), (81,)


# ─────────────────────────────────────────────────────────
#  Extraction
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def extract_latents_sudoku(encoder, discretizer, dataloader, device, max_samples=5000):
    """Sudoku-specific: returns pre/post FSQ latents + digit class labels.

    Returns:
        z_pre:       (N, fsq_dim)
        z_post:      (N, fsq_dim)
        token_ids:   (N,)
        digit_labels:(N,)  — 0-9 digit class per patch
    """
    encoder.eval().to(device)
    discretizer.eval().to(device)

    z_pre_list, z_post_list, tok_list, digit_list = [], [], [], []
    n_collected = 0

    for images, grid_labels in dataloader:
        if n_collected >= max_samples:
            break
        images = images.to(device)
        # grid_labels: (B, 81) digit values

        level_features = encoder.forward_injection(images)

        # Sudoku has single level (level_sizes=[9])
        for s, feat_2d in level_features.items():
            B, D, H, W = feat_2d.shape
            assert H == W == 9, f"Expected 9x9, got {H}x{W}"
            tokens_2d = feat_2d.flatten(2).transpose(1, 2)  # (B, 81, D)

            z = discretizer.pre(tokens_2d)          # (B, 81, 6)
            q, t = discretizer.fsq.forward_z(z)     # (B, 81, 6), (B, 81)

            z_pre_list.append(z.reshape(-1, z.shape[-1]).cpu())
            z_post_list.append(q.reshape(-1, q.shape[-1]).cpu())
            tok_list.append(t.reshape(-1).cpu())
            digit_list.append(grid_labels.reshape(-1))  # (B*81,)
            n_collected += B * 81

    z_pre = torch.cat(z_pre_list, 0)[:max_samples]
    z_post = torch.cat(z_post_list, 0)[:max_samples]
    tok_ids = torch.cat(tok_list, 0)[:max_samples]
    digits = torch.cat(digit_list, 0)[:max_samples]

    return z_pre.numpy(), z_post.numpy(), tok_ids.numpy(), digits.numpy()


@torch.no_grad()
def extract_latents_clevr(encoder, discretizer, dataloader, device, max_samples=5000):
    """CLEVR: returns pre/post FSQ latents + level labels.

    Returns:
        z_pre:       (N, fsq_dim)
        z_post:      (N, fsq_dim)
        token_ids:   (N,)
        level_labels:(N,)
    """
    encoder.eval().to(device)
    discretizer.eval().to(device)

    z_pre_list, z_post_list, tok_list, level_list = [], [], [], []
    n_collected = 0

    for images, _ in dataloader:
        if n_collected >= max_samples:
            break
        images = images.to(device)

        level_features = encoder.forward_injection(images)

        for s, feat_2d in sorted(level_features.items(), reverse=True):
            B, D, H, W = feat_2d.shape
            tokens_2d = feat_2d.flatten(2).transpose(1, 2)

            z = discretizer.pre(tokens_2d)
            q, t = discretizer.fsq.forward_z(z)

            n = B * H * W
            z_pre_list.append(z.reshape(-1, z.shape[-1]).cpu())
            z_post_list.append(q.reshape(-1, q.shape[-1]).cpu())
            tok_list.append(t.reshape(-1).cpu())
            level_list.append(torch.full((n,), s, dtype=torch.long))
            n_collected += n

    z_pre = torch.cat(z_pre_list, 0)[:max_samples]
    z_post = torch.cat(z_post_list, 0)[:max_samples]
    tok_ids = torch.cat(tok_list, 0)[:max_samples]
    levels = torch.cat(level_list, 0)[:max_samples]

    return z_pre.numpy(), z_post.numpy(), tok_ids.numpy(), levels.numpy()


# ─────────────────────────────────────────────────────────
#  Dimensionality reduction helper
# ─────────────────────────────────────────────────────────

def reduce_2d(z_pre, z_post, method="tsne", perplexity=30):
    """Reduce pre+post to 2D jointly."""
    n = len(z_pre)
    z_combined = np.concatenate([z_pre, z_post], axis=0)

    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        emb = reducer.fit_transform(z_combined)
    elif method == "pca":
        emb = PCA(n_components=2).fit_transform(z_combined)
    elif method == "umap":
        import umap
        emb = umap.UMAP(n_components=2, random_state=42, n_neighbors=15).fit_transform(z_combined)
    else:
        raise ValueError(f"Unknown method: {method}")

    return emb[:n], emb[n:]


# ─────────────────────────────────────────────────────────
#  Sudoku visualization: digit-class coloring
# ─────────────────────────────────────────────────────────

DIGIT_COLORS = {
    0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd',
    5: '#8c564b', 6: '#e377c2', 7: '#7f7f7f', 8: '#bcbd22', 9: '#17becf',
}
DIGIT_NAMES = {i: str(i) for i in range(10)}


def plot_sudoku_digit_space(z_pre, z_post, digit_labels, token_ids, title, save_path,
                            method="tsne", max_plot=5000):
    """Pre-FSQ vs Post-FSQ colored by digit class (0-9)."""
    n = min(len(z_pre), max_plot)
    z_pre, z_post = z_pre[:n], z_post[:n]
    digit_labels, token_ids = digit_labels[:n], token_ids[:n]

    emb_pre, emb_post = reduce_2d(z_pre, z_post, method=method)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    for ax, emb, subtitle in [(axes[0], emb_pre, "Pre-FSQ (continuous z)"),
                               (axes[1], emb_post, "Post-FSQ (quantized)")]:
        for d in range(10):
            mask = digit_labels == d
            if mask.sum() == 0:
                continue
            ax.scatter(emb[mask, 0], emb[mask, 1],
                      c=DIGIT_COLORS[d], s=12, alpha=0.6,
                      label=f"{d} ({mask.sum()})", rasterized=True)
        ax.set_title(subtitle, fontsize=14)
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        ax.legend(fontsize=9, ncol=2, title="Digit", markerscale=2,
                 loc='upper right')

    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {save_path}")


def plot_sudoku_per_digit_token_dist(token_ids, digit_labels, title, save_path):
    """Per-digit token distribution: which tokens does each digit use?"""
    fig, axes = plt.subplots(2, 5, figsize=(24, 9))

    for d in range(10):
        ax = axes[d // 5, d % 5]
        mask = digit_labels == d
        dtoks = token_ids[mask]
        unique, counts = np.unique(dtoks, return_counts=True)
        order = np.argsort(-counts)

        top_n = min(20, len(unique))
        ax.bar(range(top_n), counts[order[:top_n]], color=DIGIT_COLORS[d])
        ax.set_title(f"Digit {d}  ({len(unique)} unique / {len(dtoks)} total)", fontsize=10)
        ax.set_xlabel("Token rank")
        ax.set_ylabel("Count")

        # Annotate top-3 token IDs
        for i in range(min(3, top_n)):
            ax.text(i, counts[order[i]], f"{unique[order[i]]}", ha='center', va='bottom', fontsize=6)

    fig.suptitle(f"{title} — Per-digit token distribution", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {save_path}")


def plot_sudoku_token_digit_heatmap(token_ids, digit_labels, title, save_path, top_n_tokens=30):
    """Heatmap: token ID (rows) vs digit class (cols). Shows if tokens are digit-specific."""
    unique_toks, counts = np.unique(token_ids, return_counts=True)
    top_toks = unique_toks[np.argsort(-counts)[:top_n_tokens]]

    # Build count matrix: (top_n_tokens, 10)
    mat = np.zeros((top_n_tokens, 10), dtype=int)
    for i, tid in enumerate(top_toks):
        tok_mask = token_ids == tid
        for d in range(10):
            mat[i, d] = ((digit_labels == d) & tok_mask).sum()

    # Normalize each row to show probability
    row_sums = mat.sum(axis=1, keepdims=True)
    mat_norm = mat / np.maximum(row_sums, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # Raw counts
    im1 = ax1.imshow(mat, aspect='auto', cmap='Blues')
    ax1.set_xlabel("Digit class")
    ax1.set_ylabel("Token ID (by frequency rank)")
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(top_n_tokens))
    ax1.set_yticklabels([str(t) for t in top_toks], fontsize=7)
    ax1.set_title("Raw counts")
    plt.colorbar(im1, ax=ax1, shrink=0.6)

    # Normalized per-token
    im2 = ax2.imshow(mat_norm, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=0.5)
    ax2.set_xlabel("Digit class")
    ax2.set_ylabel("Token ID (by frequency rank)")
    ax2.set_xticks(range(10))
    ax2.set_yticks(range(top_n_tokens))
    ax2.set_yticklabels([str(t) for t in top_toks], fontsize=7)
    ax2.set_title("P(digit | token)")
    plt.colorbar(im2, ax=ax2, shrink=0.6)

    fig.suptitle(f"{title} — Token-Digit association", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {save_path}")


# ─────────────────────────────────────────────────────────
#  CLEVR visualization: codebook utilization
# ─────────────────────────────────────────────────────────

def plot_clevr_codebook_utilization(token_ids, level_labels, codebook_size, title, save_path):
    """Comprehensive codebook utilization analysis."""
    unique_levels = np.unique(level_labels)
    total_tokens = len(token_ids)
    unique_used = len(np.unique(token_ids))
    unused = codebook_size - unique_used
    usage_pct = unique_used / codebook_size * 100

    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ── (0,0) Overall usage pie ──
    ax = fig.add_subplot(gs[0, 0])
    ax.pie([unique_used, unused],
           labels=[f"Used\n{unique_used}", f"Unused\n{unused}"],
           colors=['#2196F3', '#E0E0E0'],
           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
    ax.set_title(f"Codebook utilization\n({codebook_size} total)", fontsize=13)

    # ── (0,1) Per-level unique tokens ──
    ax = fig.add_subplot(gs[0, 1])
    level_data = []
    for lv in sorted(unique_levels):
        mask = level_labels == lv
        n_unique = len(np.unique(token_ids[mask]))
        n_total = mask.sum()
        level_data.append((lv, n_unique, n_total))

    lvs = [f"{d[0]}x{d[0]}" for d in level_data]
    uniq_counts = [d[1] for d in level_data]
    total_counts = [d[2] for d in level_data]

    x = np.arange(len(lvs))
    w = 0.35
    bars1 = ax.bar(x - w/2, total_counts, w, label='Total tokens', color='#90CAF9')
    bars2 = ax.bar(x + w/2, uniq_counts, w, label='Unique tokens', color='#1565C0')
    ax.set_xticks(x)
    ax.set_xticklabels(lvs)
    ax.set_xlabel("Spatial level")
    ax.set_ylabel("Count")
    ax.set_title("Tokens per level", fontsize=13)
    ax.legend()
    # Annotate
    for bar, val in zip(bars2, uniq_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                str(val), ha='center', va='bottom', fontsize=9)

    # ── (0,2) Log-rank frequency distribution ──
    ax = fig.add_subplot(gs[0, 2])
    unique_toks, counts = np.unique(token_ids, return_counts=True)
    order = np.argsort(-counts)
    ax.fill_between(np.arange(1, len(unique_toks)+1), counts[order],
                    alpha=0.3, color='steelblue')
    ax.plot(np.arange(1, len(unique_toks)+1), counts[order],
            linewidth=0.8, color='steelblue')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Token rank (log)")
    ax.set_ylabel("Frequency (log)")
    ax.set_title(f"Zipf plot ({unique_used} unique tokens)", fontsize=13)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='count=1')
    singleton_count = (counts == 1).sum()
    ax.legend(title=f"{singleton_count} singletons", fontsize=9)

    # ── (1,0) Cumulative coverage ──
    ax = fig.add_subplot(gs[1, 0])
    sorted_counts = counts[order]
    cumsum = np.cumsum(sorted_counts) / total_tokens * 100
    ax.plot(np.arange(1, len(cumsum)+1), cumsum, linewidth=1.5, color='#FF5722')
    ax.set_xlabel("Number of top tokens")
    ax.set_ylabel("Cumulative coverage (%)")
    ax.set_title("Coverage curve", fontsize=13)
    ax.set_xscale('log')
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=99, color='gray', linestyle=':', alpha=0.5)
    # Find N for 50%, 90%, 99%
    for pct in [50, 90, 99]:
        idx = np.searchsorted(cumsum, pct)
        if idx < len(cumsum):
            ax.annotate(f"{pct}% @ {idx+1} tokens",
                       xy=(idx+1, pct), fontsize=9,
                       arrowprops=dict(arrowstyle='->', color='gray'),
                       xytext=(idx+1 + len(cumsum)*0.1, pct - 5))

    # ── (1,1) Per-level token overlap analysis ──
    ax = fig.add_subplot(gs[1, 1])
    if len(unique_levels) > 1:
        level_token_sets = {}
        for lv in sorted(unique_levels):
            level_token_sets[lv] = set(np.unique(token_ids[level_labels == lv]))

        levels_sorted = sorted(unique_levels)
        n_levels = len(levels_sorted)
        overlap_mat = np.zeros((n_levels, n_levels))
        for i, lv1 in enumerate(levels_sorted):
            for j, lv2 in enumerate(levels_sorted):
                s1, s2 = level_token_sets[lv1], level_token_sets[lv2]
                if len(s1) > 0 and len(s2) > 0:
                    overlap_mat[i, j] = len(s1 & s2) / len(s1 | s2)  # Jaccard

        im = ax.imshow(overlap_mat, cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xticks(range(n_levels))
        ax.set_yticks(range(n_levels))
        labels = [f"{lv}x{lv}" for lv in levels_sorted]
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        # Annotate values
        for i in range(n_levels):
            for j in range(n_levels):
                ax.text(j, i, f"{overlap_mat[i,j]:.2f}", ha='center', va='center', fontsize=10)
        ax.set_title("Token overlap (Jaccard) between levels", fontsize=13)
        plt.colorbar(im, ax=ax, shrink=0.7)
    else:
        ax.text(0.5, 0.5, "Single level", ha='center', va='center', transform=ax.transAxes)

    # ── (1,2) Entropy summary ──
    ax = fig.add_subplot(gs[1, 2])
    probs = counts / counts.sum()
    entropy = -(probs * np.log2(probs)).sum()
    max_entropy = np.log2(codebook_size)
    uniform_ent = np.log2(unique_used) if unique_used > 0 else 0

    bars = ax.bar(['Actual', 'Uniform\n(used only)', 'Max\n(full codebook)'],
                  [entropy, uniform_ent, max_entropy],
                  color=['#4CAF50', '#FF9800', '#F44336'])
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Token entropy", fontsize=13)
    for bar, val in zip(bars, [entropy, uniform_ent, max_entropy]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Summary text
    fig.text(0.5, 0.02,
             f"Total: {total_tokens} tokens | Unique: {unique_used}/{codebook_size} ({usage_pct:.1f}%) | "
             f"Entropy: {entropy:.1f}/{max_entropy:.1f} bits | Singletons: {singleton_count}",
             ha='center', fontsize=12, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {save_path}")


def plot_clevr_level_embedding(z_pre, z_post, level_labels, title, save_path, method="tsne"):
    """CLEVR t-SNE/UMAP colored by spatial level."""
    n = min(len(z_pre), 5000)
    z_pre, z_post = z_pre[:n], z_post[:n]
    level_labels = level_labels[:n]

    emb_pre, emb_post = reduce_2d(z_pre, z_post, method=method)

    unique_levels = sorted(np.unique(level_labels))
    level_cmap = plt.cm.Set1

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    for ax, emb, subtitle in [(axes[0], emb_pre, "Pre-FSQ (continuous z)"),
                               (axes[1], emb_post, "Post-FSQ (quantized)")]:
        for li, lv in enumerate(unique_levels):
            mask = level_labels == lv
            ax.scatter(emb[mask, 0], emb[mask, 1],
                      c=[level_cmap(li)], s=8, alpha=0.5,
                      label=f"level {lv}x{lv} ({mask.sum()})", rasterized=True)
        ax.set_title(subtitle, fontsize=14)
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        ax.legend(fontsize=10, markerscale=2)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {save_path}")


def plot_fsq_dim_hist(z_pre, z_post, fsq_levels, title, save_path):
    """Per-dimension histogram with FSQ grid lines."""
    fsq_dim = z_pre.shape[1]
    n = min(len(z_pre), 5000)
    z_pre, z_post = z_pre[:n], z_post[:n]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for d in range(min(fsq_dim, 6)):
        ax = axes[d // 3, d % 3]
        ax.hist(z_pre[:, d], bins=100, alpha=0.5, label='pre-FSQ', color='steelblue', density=True)
        ax.hist(z_post[:, d], bins=100, alpha=0.5, label='post-FSQ', color='coral', density=True)

        # Draw FSQ grid lines
        L = fsq_levels[d]
        half_w = L // 2
        for k in range(-half_w, half_w + 1):
            ax.axvline(x=k / half_w, color='red', linestyle=':', alpha=0.3, linewidth=0.8)

        ax.set_title(f"Dim {d} (L={L})", fontsize=12)
        ax.legend(fontsize=8)
        ax.set_xlabel("value")

    fig.suptitle(f"{title} — Per-dimension distribution", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {save_path}")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def process_sudoku(run_dir, data_dir, device, method="tsne", max_samples=5000, batch_size=32):
    run_dir = Path(run_dir)
    run_name = run_dir.name

    with open(run_dir / "args.json") as f:
        args_dict = json.load(f)

    image_size = args_dict["image_size"]
    fsq_levels = args_dict["fsq_levels"]
    codebook_size = 1
    for l in fsq_levels:
        codebook_size *= l

    print(f"\n{'='*60}")
    print(f"  [Sudoku] {run_name}")
    print(f"  Image: {image_size}, FSQ: {fsq_levels} (codebook={codebook_size})")
    print(f"{'='*60}")

    encoder, discretizer = build_encoder_and_discretizer(args_dict)

    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.iterdir())
    latest_ckpt = ckpts[-1] / "checkpoint.pt"
    step = load_weights(encoder, discretizer, latest_ckpt)

    dataset = SudokuImageDataset(root=data_dir, image_size=image_size, top_n=100, max_grids=500)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print(f"  Extracting latents...")
    z_pre, z_post, tok_ids, digit_labels = extract_latents_sudoku(
        encoder, discretizer, loader, device, max_samples=max_samples)
    print(f"  {len(z_pre)} tokens, {len(np.unique(tok_ids))} unique, digits: {np.bincount(digit_labels, minlength=10)}")

    out_dir = run_dir / "token_space_viz"
    out_dir.mkdir(exist_ok=True)

    # 1) Digit-class t-SNE
    for m in ([method] if method != "all" else ["tsne", "umap"]):
        print(f"  {m.upper()} by digit class...")
        plot_sudoku_digit_space(
            z_pre, z_post, digit_labels, tok_ids,
            title=f"{run_name} (step {step})",
            save_path=str(out_dir / f"digit_space_{m}.png"),
            method=m)

    # 2) Per-digit token distribution
    plot_sudoku_per_digit_token_dist(
        tok_ids, digit_labels, title=run_name,
        save_path=str(out_dir / "per_digit_tokens.png"))

    # 3) Token-digit heatmap
    plot_sudoku_token_digit_heatmap(
        tok_ids, digit_labels, title=run_name,
        save_path=str(out_dir / "token_digit_heatmap.png"))

    # 4) Per-dim histogram
    plot_fsq_dim_hist(z_pre, z_post, fsq_levels, title=run_name,
                      save_path=str(out_dir / "fsq_dim_hist.png"))

    print(f"  All saved to {out_dir}/")


def process_clevr(run_dir, data_dir, device, method="tsne", max_samples=10000, batch_size=32):
    run_dir = Path(run_dir)
    run_name = run_dir.name

    with open(run_dir / "args.json") as f:
        args_dict = json.load(f)

    image_size = args_dict["image_size"]
    fsq_levels = args_dict["fsq_levels"]
    codebook_size = 1
    for l in fsq_levels:
        codebook_size *= l

    print(f"\n{'='*60}")
    print(f"  [CLEVR] {run_name}")
    print(f"  Image: {image_size}, FSQ: {fsq_levels} (codebook={codebook_size})")
    print(f"{'='*60}")

    encoder, discretizer = build_encoder_and_discretizer(args_dict)

    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.iterdir())
    latest_ckpt = ckpts[-1] / "checkpoint.pt"
    step = load_weights(encoder, discretizer, latest_ckpt)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*args_dict["cond_in_channels"],
                           [0.5]*args_dict["cond_in_channels"]),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print(f"  Extracting latents (max {max_samples})...")
    z_pre, z_post, tok_ids, level_labels = extract_latents_clevr(
        encoder, discretizer, loader, device, max_samples=max_samples)
    print(f"  {len(z_pre)} tokens, {len(np.unique(tok_ids))} unique")

    out_dir = run_dir / "token_space_viz"
    out_dir.mkdir(exist_ok=True)

    # 1) Codebook utilization dashboard
    print(f"  Codebook utilization analysis...")
    plot_clevr_codebook_utilization(
        tok_ids, level_labels, codebook_size,
        title=f"{run_name} (step {step})",
        save_path=str(out_dir / "codebook_utilization.png"))

    # 2) Level embedding
    for m in ([method] if method != "all" else ["tsne", "umap"]):
        print(f"  {m.upper()} by level...")
        plot_clevr_level_embedding(
            z_pre, z_post, level_labels,
            title=f"{run_name} (step {step})",
            save_path=str(out_dir / f"level_space_{m}.png"),
            method=m)

    # 3) Per-dim histogram
    plot_fsq_dim_hist(z_pre, z_post, fsq_levels, title=run_name,
                      save_path=str(out_dir / "fsq_dim_hist.png"))

    print(f"  All saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Visualize FSQ token space")
    parser.add_argument("--mode", required=True, choices=["sudoku", "clevr"],
                       help="Dataset mode")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--method", default="tsne", choices=["tsne", "pca", "umap", "all"])
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.mode == "sudoku":
        process_sudoku(args.run_dir, args.data_dir, args.device,
                      method=args.method, max_samples=args.max_samples,
                      batch_size=args.batch_size)
    elif args.mode == "clevr":
        process_clevr(args.run_dir, args.data_dir, args.device,
                     method=args.method, max_samples=args.max_samples,
                     batch_size=args.batch_size)


if __name__ == "__main__":
    main()
