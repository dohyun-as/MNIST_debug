#!/usr/bin/env python
"""
plot_eval_sweep.py
==================
Reads `runs/eval_sweep.csv` and produces matplotlib trajectory plots.

Layout (single PNG per figure, per metric):
  rows = splits (overall, easy, medium, hard)
  cols = single column with all (run, sampler) lines
  one figure per metric (count_acc, presence_acc, rel_acc) ∈ separate PNGs

Plus a 12-panel summary PNG (4 splits × 3 metrics).

Usage:
  python script/plot_eval_sweep.py
  python script/plot_eval_sweep.py --csv runs/eval_sweep.csv --out_dir runs/eval_plots --all_steps
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── short labels ─────────────────────────────────────────────────────────
_RUN_SHORT = {
    "ours_text_diffhead_clip_dit_vit_flow_cont_out16_only8x8_tokdrop1.0":
        "ours-only8x8",
    "ours_text_diffhead_clip_out16_tokdrop1.0":
        "ours-multilevel",
    "ours_text_diffhead_clip_out16_tokdrop1.0_semiar":
        "ours-multilevel+semiAR",
    "256_slot16_d64_resnet18s_crossattn_clip":
        "slot-stage2",
    "naive_dit_256_text_cond_clip":
        "naive-T2I",
}

_RUN_COLOR = {
    "ours-only8x8":            "#d62728",
    "ours-multilevel":         "#ff7f0e",
    "ours-multilevel+semiAR":  "#bcbd22",
    "slot-stage2":             "#9467bd",
    "naive-T2I":               "#1f77b4",
}

_SAMPLER_LS = {
    "confidence_top1":   "-",
    "confidence_cosine": "--",
    "ddpm_cache":        ":",
    "naive_t2i":         "-",
}

_METRICS = [
    ("count_acc",    "Count accuracy (%)"),
    ("presence_acc", "Entity presence accuracy (%)"),
    ("rel_acc",      "Relation accuracy (%)"),
]
# Overall split only. Per-split plotting was removed — caused confusion.
_SPLITS = ["overall"]
_SAVED_STEPS = (50000, 100000, 150000, 200000)


def _short_run(full: str) -> str:
    name = full.rstrip("/").split("/")[-1]
    return _RUN_SHORT.get(name, name)


def _load(csv_path: str) -> List[Dict]:
    rows = list(csv.DictReader(open(csv_path)))
    out = []
    for r in rows:
        try:
            r["step"] = int(r["step"])
            r["n_samples"] = int(r["n_samples"]) if r["n_samples"] else 0
            for k in ("count_acc", "presence_acc", "rel_acc"):
                r[k] = float(r[k]) if r[k] not in (None, "") else None
            r["run_short"] = _short_run(r["run"])
            out.append(r)
        except (TypeError, ValueError):
            continue
    return out


def _series(rows: List[Dict], split: str, metric: str
            ) -> Dict[Tuple[str, str], List[Tuple[int, float]]]:
    """Return {(run_short, sampler): sorted [(step, value), ...]}."""
    by_key: Dict[Tuple[str, str], List[Tuple[int, float]]] = defaultdict(list)
    for r in rows:
        if r["split"] != split:
            continue
        if r[metric] is None:
            continue
        by_key[(r["run_short"], r["sampler"])].append((r["step"], r[metric]))
    for k in by_key:
        by_key[k].sort()
    return by_key


def _plot_metric(ax, rows: List[Dict], split: str, metric: str, title: str):
    series = _series(rows, split, metric)
    # Stable ordering — runs first, then samplers
    keys = sorted(series.keys(),
                  key=lambda k: (list(_RUN_COLOR).index(k[0])
                                 if k[0] in _RUN_COLOR else 99,
                                 list(_SAMPLER_LS).index(k[1])
                                 if k[1] in _SAMPLER_LS else 99))
    for (run, sampler) in keys:
        xy = series[(run, sampler)]
        if not xy:
            continue
        xs, ys = zip(*xy)
        col = _RUN_COLOR.get(run, "#444444")
        ls = _SAMPLER_LS.get(sampler, "-")
        # Skip naive's redundant sampler legend slot
        if sampler == "naive_t2i":
            label = run
        else:
            label = f"{run} / {sampler}"
        ax.plot(xs, ys, color=col, linestyle=ls,
                marker="o", markersize=3.0, linewidth=1.4, label=label)
    ax.set_title(title)
    ax.set_xlabel("training step")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)


def make_summary(rows: List[Dict], out_path: str) -> None:
    """1 row (overall) × 3 metrics = 3 panels."""
    fig, axes = plt.subplots(1, len(_METRICS),
                             figsize=(20, 6), squeeze=False)
    for ci, (m, label) in enumerate(_METRICS):
        ax = axes[0][ci]
        _plot_metric(ax, rows, "overall", m, label)
        if ci == 0:
            ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
    fig.suptitle("CLEVR eval sweep — overall (corrected post-fix scores)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def make_per_metric(rows: List[Dict], out_dir: str) -> None:
    """One PNG per metric, single 'overall' panel."""
    for metric, label in _METRICS:
        fig, ax = plt.subplots(1, 1, figsize=(11, 6))
        _plot_metric(ax, rows, "overall", metric, label)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
        fig.tight_layout()
        out = os.path.join(out_dir, f"{metric}_overall.png")
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="runs/eval_outputs/eval_sweep.csv")
    p.add_argument("--out_dir", default="runs/eval_outputs/plots/overall")
    p.add_argument("--all_steps", action="store_true",
                   help="Plot every step found in CSV. Default: only the "
                        "saved-ckpt steps (50k/100k/150k/200k) which were "
                        "re-evaluated by the corrected sweep — earlier "
                        "training-time JSONs may use the buggy eval path.")
    args = p.parse_args()

    rows = _load(args.csv)
    if not args.all_steps:
        rows = [r for r in rows if r["step"] in _SAVED_STEPS]
    if not rows:
        raise SystemExit(f"No rows in {args.csv}")

    os.makedirs(args.out_dir, exist_ok=True)
    make_summary(rows, os.path.join(args.out_dir, "summary.png"))
    make_per_metric(rows, args.out_dir)


if __name__ == "__main__":
    main()
