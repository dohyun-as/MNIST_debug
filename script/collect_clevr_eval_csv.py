#!/usr/bin/env python3
"""
Collect CLEVR eval JSONs from runs and emit a tidy CSV.

Sources:
  naive_dit_256_text_cond_clip/clevr_eval/clevr_eval_step*.json
  discrete_diff_ours_text/eval_samples/step_*_clevr_ddpm_cache_cond_eval.json
  discrete_diff_semanticist_text_w_decay0_larger_batch/
      eval_samples/step_*_clevr_ddpm_cache_cond_eval.json

CSV columns:
  run, step, split, n_samples,
  count_accuracy, entity_presence_accuracy, rel_accuracy,
  count_correct, entity_found, entity_total, rel_correct, rel_total
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from typing import Iterable

DEFAULT_RUNS_ROOT = "runs/clevr"

DEFAULT_TARGETS = [
    # (run_label, subdir_glob_pattern, step_regex)
    ("naive_dit_256_text_cond_clip",
     "naive_dit_256_text_cond_clip/clevr_eval/clevr_eval_step*.json",
     r"clevr_eval_step(\d+)\.json$"),
    ("discrete_diff_ours_text",
     "discrete_diff_ours_text/eval_samples/step_*_clevr_ddpm_cache_cond_eval.json",
     r"step_(\d+)_clevr_ddpm_cache_cond_eval\.json$"),
    ("discrete_diff_semanticist_text_w_decay0_larger_batch",
     "discrete_diff_semanticist_text_w_decay0_larger_batch/"
     "eval_samples/step_*_clevr_ddpm_cache_cond_eval.json",
     r"step_(\d+)_clevr_ddpm_cache_cond_eval\.json$"),
]

METRIC_KEYS = [
    "n_samples",
    "count_accuracy",
    "entity_presence_accuracy",
    "rel_accuracy",
    "count_correct",
    "entity_found",
    "entity_total",
    "rel_correct",
    "rel_total",
]


def iter_rows(runs_root: str, targets) -> Iterable[dict]:
    for run_label, pattern, step_re in targets:
        full_pattern = os.path.join(runs_root, pattern)
        files = sorted(glob.glob(full_pattern))
        if not files:
            print(f"[warn] no files for {run_label}: {full_pattern}")
            continue
        rx = re.compile(step_re)
        for fp in files:
            m = rx.search(os.path.basename(fp))
            if not m:
                continue
            step = int(m.group(1))
            try:
                with open(fp) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[warn] failed to read {fp}: {e}")
                continue

            overall = data.get("overall", {})
            if overall:
                row = {"run": run_label, "step": step, "split": "overall"}
                for k in METRIC_KEYS:
                    row[k] = overall.get(k)
                yield row

            for split_name, split_data in (data.get("per_split") or {}).items():
                row = {"run": run_label, "step": step, "split": split_name}
                for k in METRIC_KEYS:
                    row[k] = split_data.get(k)
                yield row


def write_csv(rows, out_path: str) -> int:
    rows = list(rows)
    rows.sort(key=lambda r: (r["run"], r["step"], r["split"]))
    fieldnames = ["run", "step", "split", *METRIC_KEYS]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


def print_overall_table(rows):
    rows = [r for r in rows if r["split"] == "overall"]
    rows.sort(key=lambda r: (r["run"], r["step"]))
    if not rows:
        return
    hdr = f"{'run':<55s} {'step':>7s} {'count':>7s} {'entPres':>8s} {'rel':>7s}"
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['run']:<55s} {r['step']:>7d} "
              f"{(r['count_accuracy'] or 0):>6.2f}% "
              f"{(r['entity_presence_accuracy'] or 0):>7.2f}% "
              f"{(r['rel_accuracy'] or 0):>6.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default=DEFAULT_RUNS_ROOT,
                    help="Root containing per-run dirs (default: runs/clevr)")
    ap.add_argument("--out", default="runs/clevr/clevr_eval_summary.csv",
                    help="Output CSV path")
    ap.add_argument("--no-table", action="store_true",
                    help="Skip printing the overall table")
    args = ap.parse_args()

    rows = list(iter_rows(args.runs_root, DEFAULT_TARGETS))
    n = write_csv(rows, args.out)
    print(f"[done] wrote {n} rows → {args.out}")
    if not args.no_table:
        print_overall_table(rows)


if __name__ == "__main__":
    main()
