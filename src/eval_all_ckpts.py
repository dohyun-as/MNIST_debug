#!/usr/bin/env python
"""
eval_all_ckpts.py
=================
For each saved checkpoint of each run, re-run condition eval (corrected
[-1,1]→[0,1] path) on a chosen number of validation samples. Writes the
per-(run, step, sampler) score files in their normal `eval_samples/` /
`clevr_eval/` locations and (optionally) aggregates everything into a
single CSV + JSON.

Two run families are supported, auto-detected from the run dir contents:
  - v2 (masked_diff/* + slot_stage2/*): has `run_config.json` + `ckpt/`
  - naive T2I (naive_dit_*):           has `args.json` + top-level `step*/`

Both expose `--eval_only --resume_dir <step_subdir>` after the recent
patch, so this script just launches them with the original training
hyper-parameters but a tunable sample count.

Usage (eval, 4 GPUs, 200 samples per split = 600 total per ckpt):
  python src/eval_all_ckpts.py \
    --run_dirs runs/clevr/masked_diff/ours_text_diffhead_clip_dit_vit_flow_cont_out16_only8x8_tokdrop1.0 \
               runs/clevr/masked_diff/ours_text_diffhead_clip_out16_tokdrop1.0 \
               runs/clevr/masked_diff/ours_text_diffhead_clip_out16_tokdrop1.0_semiar \
               runs/clevr/slot_stage2/256_slot16_d64_resnet18s_crossattn_clip \
               runs/clevr/naive_dit_256_text_cond_clip \
    --num_samples_per_split 200 \
    --gpus 0,1,2,3 \
    --output_csv runs/eval_sweep.csv

Aggregate-only (after a sweep finished, just rebuild CSV/JSON):
  python src/eval_all_ckpts.py --aggregate_only \
    --run_dirs ... --output_csv runs/eval_sweep.csv

If you already trust the cond_eval JSON saved during training **at or after
the bug-fix commit**, set `--reuse_existing` to skip launching and just
aggregate them. Otherwise the script always re-runs to be safe.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


# ───────────────────────────────────────── run-type detection ──────────────

def detect_kind(run_dir: str) -> str:
    """Returns one of: 'v2_slot', 'v2', 'naive_t2i'.

    'v2_slot' means the run uses a slot encoder backbone (stage1 has
    use_slot_encoder=true) — needs train_discrete_diffusion_slot.py
    entrypoint, NOT plain train_discrete_diffusion_v2.py, because the slot
    wrapper monkey-patches load_pretrained_model + decode_*.
    """
    if os.path.isfile(os.path.join(run_dir, "run_config.json")):
        # Check the pretrained config for use_slot_encoder
        try:
            with open(os.path.join(run_dir, "run_config.json")) as f:
                cfg = json.load(f).get("args", {})
            pretrained = cfg.get("pretrained_output_dir", "") or ""
            pretrained = pretrained.lstrip("./")
            for sub in ("args.json", "run_config.json"):
                p = os.path.join(pretrained, sub)
                if os.path.isfile(p):
                    pc = json.load(open(p))
                    if "args" in pc:
                        pc = pc["args"]
                    if pc.get("use_slot_encoder", False):
                        return "v2_slot"
                    break
        except Exception:
            pass
        return "v2"
    if os.path.isfile(os.path.join(run_dir, "args.json")):
        return "naive_t2i"
    raise SystemExit(f"[detect] no run_config.json / args.json in {run_dir}")


def load_saved_args(run_dir: str, kind: str) -> Dict:
    if kind in ("v2", "v2_slot"):
        with open(os.path.join(run_dir, "run_config.json")) as f:
            return dict(json.load(f)["args"])
    with open(os.path.join(run_dir, "args.json")) as f:
        return dict(json.load(f))


def find_ckpt_steps(run_dir: str, kind: str) -> List[Tuple[int, str]]:
    """Return sorted list of (step, ckpt_subdir_path) for this run."""
    if kind in ("v2", "v2_slot"):
        root = os.path.join(run_dir, "ckpt")
    else:
        root = run_dir
    if not os.path.isdir(root):
        return []
    by_step: Dict[int, str] = {}
    for d in os.listdir(root):
        if not d.startswith("step"):
            continue
        m = re.search(r"step0*(\d+)", d)
        if m is None:
            continue
        full = os.path.join(root, d)
        # for v2 require model.safetensors; naive requires model.pt
        marker = ("model.safetensors" if kind == "v2" else "model.pt")
        if not os.path.isfile(os.path.join(full, marker)):
            continue
        step = int(m.group(1))
        # If a duplicate exists (e.g. step200000 + step200000_final), prefer
        # the one without "_final" suffix to be deterministic.
        if step in by_step and d.endswith("_final"):
            continue
        by_step[step] = full
    return sorted(by_step.items())


# ─────────────────────────────────────────── argv reconstruction ───────────

# Args the orchestrator overrides — drop these from the saved config so
# our own values win after argparse merging.
_EVAL_OVERRIDE_KEYS = {
    "eval_only", "resume_dir",
    "eval_num_samples", "clevr_eval_samples",
}

# Args known to be store_true flags in v2 / naive scripts (so we emit them
# only when True).
_BOOL_FLAGS_V2 = {
    "use_diffusion_head", "use_pretrained_text_encoder",
    "unfreeze_text_encoder", "factorized_head", "semi_autoregressive",
    "antithetic_sampling", "change_of_variables", "use_loss_weighting",
    "freeze_text_encoder",
    "tie_embeddings",
}
_BOOL_FLAGS_NAIVE = {
    "use_flow_matching", "freeze_text_encoder", "unfreeze_text_encoder",
    "use_pretrained_text_encoder",
}

# When the saved config has dest=X set to False but the parser only exposes a
# `--Y` (action="store_false", dest="X") flag, this map says "if X is False,
# emit --Y instead of nothing". Same flags exist in both v2 and naive.
_BOOL_FALSE_ALIAS = {
    "freeze_text_encoder": "--unfreeze_text_encoder",
}


def args_dict_to_argv(saved_args: Dict, kind: str) -> List[str]:
    """Convert a saved-args dict (run_config.json's "args" or args.json)
    into a CLI argv list compatible with the train script's argparse."""
    bool_flags = _BOOL_FLAGS_V2 if kind == "v2" else _BOOL_FLAGS_NAIVE
    argv: List[str] = []
    for k, v in saved_args.items():
        if k in _EVAL_OVERRIDE_KEYS:
            continue
        if v is None:
            continue
        if isinstance(v, bool):
            if v and k in bool_flags:
                argv.append(f"--{k}")
            elif (not v) and k in _BOOL_FALSE_ALIAS:
                argv.append(_BOOL_FALSE_ALIAS[k])
            elif v and k not in bool_flags:
                argv.extend([f"--{k}", "true"])
            continue
        if isinstance(v, (list, tuple)):
            if not v:
                continue
            argv.append(f"--{k}")
            argv.extend(str(x) for x in v)
            continue
        if isinstance(v, str) and v == "":
            continue
        argv.extend([f"--{k}", str(v)])
    return argv


# ─────────────────────────────────────────────── per-ckpt launcher ─────────

def launch_eval(run_dir: str, kind: str, ckpt_subdir: str,
                base_argv: List[str], num_samples_per_split: int,
                gpus: str, eval_num_steps: Optional[int] = None,
                eval_sample_bs: Optional[int] = None,
                eval_decode_bs: Optional[int] = None,
                force_semi_ar: bool = False,
                shadow_output_dir: Optional[str] = None,
                env_extra: Optional[Dict[str, str]] = None) -> int:
    """Spawn `accelerate launch ... train_script --eval_only`.

    Returns the subprocess exit code (0 = success).
    """
    n_gpus = len(gpus.split(","))
    if kind in ("v2", "v2_slot"):
        # Slot runs MUST use the wrapper which monkey-patches the pretrained
        # loader and the continuous decode path; running v2 directly skips
        # SlotAttentionEncoder loading and crashes in
        # _cont_tokens_to_level_features (s*s reshape fails on 1D slots).
        train_script = ("src/train_discrete_diffusion_slot.py"
                        if kind == "v2_slot"
                        else "src/train_discrete_diffusion_v2.py")
        sample_arg = ["--eval_num_samples", str(num_samples_per_split)]
        # Bigger batch knobs for higher GPU util during the sweep.
        if eval_sample_bs is not None:
            sample_arg += ["--eval_sample_batch_size", str(eval_sample_bs)]
        if eval_decode_bs is not None:
            sample_arg += ["--eval_decode_batch_size", str(eval_decode_bs)]
    else:
        train_script = "src/train_text_conditioned.py"
        sample_arg = ["--clevr_eval_samples", str(num_samples_per_split)]

    eval_steps_arg = []
    if eval_num_steps is not None and kind == "v2":
        eval_steps_arg = ["--eval_num_steps", str(eval_num_steps)]
    if eval_num_steps is not None and kind == "naive_t2i":
        eval_steps_arg = ["--eval_num_steps", str(eval_num_steps)]

    # Pick a free TCP port for this run
    port_cmd = ("import socket;s=socket.socket();s.bind(('',0));"
                "print(s.getsockname()[1]);s.close()")
    port = subprocess.check_output(["python", "-c", port_cmd]).decode().strip()

    accelerate_args = ["accelerate", "launch", "--main_process_port", port]
    if n_gpus > 1:
        accelerate_args += ["--multi_gpu", "--num_processes", str(n_gpus)]

    extra_overrides: List[str] = []
    if force_semi_ar and kind == "v2":
        extra_overrides += ["--semi_autoregressive"]
    if shadow_output_dir is not None:
        # Send eval_samples/ etc. to the shadow dir so the original run's
        # outputs aren't overwritten. Reuse the original run's token_cache
        # to avoid recomputing 9k cached features.
        extra_overrides += ["--output_dir", shadow_output_dir,
                            "--token_cache_dir",
                            os.path.join(run_dir, "token_cache")]

    # Order: base_argv (saved config) FIRST, our override args LAST, so they
    # win the argparse merge.
    cmd = (accelerate_args + [train_script] + base_argv
           + ["--eval_only", "--resume_dir", ckpt_subdir]
           + sample_arg + eval_steps_arg + extra_overrides)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpus
    if env_extra:
        env.update(env_extra)
    print(f"\n[launch] {run_dir}  step={os.path.basename(ckpt_subdir)}  "
          f"gpus={gpus}  N/split={num_samples_per_split}")
    print("  $ " + " ".join(cmd[:14]) + " ...")
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


# ─────────────────────────────────────── result harvesting + aggregation ───

def collect_results(run_dir: str, kind: str) -> List[Dict]:
    """Read all cond_eval JSONs under this run dir. Returns one row per
    (step, sampler) tuple including a "split" sub-rows iteration."""
    rows: List[Dict] = []
    if kind in ("v2", "v2_slot"):
        eval_dir = os.path.join(run_dir, "eval_samples")
        pat = re.compile(r"step_(\d+)_clevr_(.+?)_cond_eval\.json$")
        if os.path.isdir(eval_dir):
            for fn in sorted(os.listdir(eval_dir)):
                m = pat.match(fn)
                if not m:
                    continue
                step = int(m.group(1))
                sampler = m.group(2)
                with open(os.path.join(eval_dir, fn)) as f:
                    d = json.load(f)
                ov = d.get("overall", {})
                rows.append({
                    "run": run_dir, "kind": kind, "step": step,
                    "sampler": sampler, "split": "overall",
                    "n_samples": ov.get("n_samples"),
                    "count_acc": ov.get("count_accuracy"),
                    "presence_acc": ov.get("entity_presence_accuracy"),
                    "rel_acc": ov.get("rel_accuracy"),
                })
                for sp, r in d.get("per_split", {}).items():
                    rows.append({
                        "run": run_dir, "kind": kind, "step": step,
                        "sampler": sampler, "split": sp,
                        "n_samples": r.get("n_samples"),
                        "count_acc": r.get("count_accuracy"),
                        "presence_acc": r.get("entity_presence_accuracy"),
                        "rel_acc": r.get("rel_accuracy"),
                    })
    else:
        eval_dir = os.path.join(run_dir, "clevr_eval")
        pat = re.compile(r"clevr_eval_step(\d+)\.json$")
        if os.path.isdir(eval_dir):
            for fn in sorted(os.listdir(eval_dir)):
                m = pat.match(fn)
                if not m:
                    continue
                step = int(m.group(1))
                with open(os.path.join(eval_dir, fn)) as f:
                    d = json.load(f)
                ov = d.get("overall") or {}
                rows.append({
                    "run": run_dir, "kind": kind, "step": step,
                    "sampler": "naive_t2i", "split": "overall",
                    "n_samples": ov.get("n_samples"),
                    "count_acc": ov.get("count_accuracy"),
                    "presence_acc": ov.get("entity_presence_accuracy"),
                    "rel_acc": ov.get("rel_accuracy"),
                })
                for sp, r in (d.get("per_split") or {}).items():
                    rows.append({
                        "run": run_dir, "kind": kind, "step": step,
                        "sampler": "naive_t2i", "split": sp,
                        "n_samples": r.get("n_samples"),
                        "count_acc": r.get("count_accuracy"),
                        "presence_acc": r.get("entity_presence_accuracy"),
                        "rel_acc": r.get("rel_accuracy"),
                    })
    return rows


def write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        print(f"[csv] no rows; skipping {path}")
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    cols = ["run", "kind", "step", "sampler", "split",
            "n_samples", "count_acc", "presence_acc", "rel_acc"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[csv] wrote {len(rows)} rows → {path}")


def write_json(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[json] wrote {len(rows)} rows → {path}")


# ───────────────────────────────────────────── annotation of last ckpt ─────

# Short labels matching script/plot_eval_sweep.py — keeps the central
# eval_outputs/ tree readable.
_RUN_SHORT = {
    "ours_text_diffhead_clip_dit_vit_flow_cont_out16_only8x8_tokdrop1.0":
        "ours-only8x8",
    "ours_text_diffhead_clip_out16_tokdrop1.0":
        "ours-multilevel",
    "ours_text_diffhead_clip_out16_tokdrop1.0_infer_semiar":
        "ours-multilevel-INFER-semiar",
    "ours_text_diffhead_clip_out16_tokdrop1.0_semiar":
        "ours-multilevel-semiar",
    "256_slot16_d64_resnet18s_crossattn_clip":
        "slot-stage2",
    "naive_dit_256_text_cond_clip":
        "naive-T2I",
}


def _run_short(run_dir: str) -> str:
    name = run_dir.rstrip("/").split("/")[-1]
    return _RUN_SHORT.get(name, name)


def annotate_last_grid(run_dir: str, kind: str, gpus: str,
                       output_root: str) -> None:
    """Find the latest eval_samples grid PNG for this run and run the
    annotator on it (4-GPU sharded). Annotates only the LAST ckpt step.

    Outputs go to: <output_root>/annotated/<run_short>/step_<step>_<sampler>/
    """
    short = _run_short(run_dir)
    out_base = os.path.join(output_root, "annotated", short)

    if kind in ("v2", "v2_slot"):
        eval_dir = os.path.join(run_dir, "eval_samples")
        pat_meta = re.compile(r"step_(\d+)_clevr_meta\.json$")
        meta_steps = []
        if os.path.isdir(eval_dir):
            for fn in os.listdir(eval_dir):
                m = pat_meta.match(fn)
                if m:
                    meta_steps.append((int(m.group(1)), fn))
        meta_steps.sort()
        if not meta_steps:
            print(f"[annotate] no eval_samples meta in {eval_dir}")
            return
        step, meta_fn = meta_steps[-1]
        meta_path = os.path.join(eval_dir, meta_fn)
        for samp in ("confidence_top1", "confidence_cosine", "ddpm_cache"):
            grid = os.path.join(eval_dir, f"step_{step:07d}_clevr_{samp}.png")
            if not os.path.isfile(grid):
                continue
            out = os.path.join(out_base, f"step_{step:07d}_{samp}")
            _run_annotate_sharded(grid, ["--meta_json", meta_path], out, gpus)
    else:
        eval_dir = os.path.join(run_dir, "clevr_eval")
        pat = re.compile(r"clevr_eval_step(\d+)\.png$")
        png_steps = []
        if os.path.isdir(eval_dir):
            for fn in os.listdir(eval_dir):
                m = pat.match(fn)
                if m:
                    png_steps.append((int(m.group(1)), fn))
        png_steps.sort()
        if not png_steps:
            print(f"[annotate] no naive grid PNGs in {eval_dir}")
            return
        step, png_fn = png_steps[-1]
        grid = os.path.join(eval_dir, png_fn)
        meta = os.path.join(eval_dir, f"clevr_eval_step{step:07d}_meta.json")
        out = os.path.join(out_base, f"step_{step:07d}")
        if os.path.isfile(meta):
            _run_annotate_sharded(grid, ["--meta_json", meta], out, gpus)
        else:
            _run_annotate_sharded(grid, ["--derive_naive_t2i"], out, gpus)


def _run_annotate_sharded(grid: str, cond_args: List[str], out: str,
                          gpus: str) -> None:
    n = len(gpus.split(","))
    os.makedirs(out, exist_ok=True)
    procs = []
    if n == 1:
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = gpus
        cmd = (["python", "src/annotate_clevr_eval_grid.py",
                "--grid_png", grid, "--output_dir", out] + cond_args)
        subprocess.run(cmd, env=env)
        return
    gpu_list = gpus.split(",")
    for s, g in enumerate(gpu_list):
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = g
        log = os.path.join(out, f"shard{s}.log")
        cmd = (["python", "src/annotate_clevr_eval_grid.py",
                "--grid_png", grid, "--output_dir", out,
                "--shard_idx", str(s), "--num_shards", str(n)] + cond_args)
        with open(log, "w") as lf:
            procs.append(subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf))
    for p in procs:
        p.wait()
    subprocess.run(["python", "src/annotate_clevr_eval_grid.py",
                    "--merge", "--grid_png", grid, "--output_dir", out])


# ────────────────────────────────────────────────────────── main ──────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dirs", nargs="+", required=True,
                   help="Run directories to evaluate.")
    p.add_argument("--num_samples_per_split", type=int, default=200,
                   help="N samples per split (easy/medium/hard). Setting "
                        "very large (e.g. 5000) will use the full split.")
    p.add_argument("--gpus", default="0,1,2,3",
                   help="GPU ids to use, comma-separated.")
    p.add_argument("--steps", nargs="*", type=int, default=None,
                   help="Restrict to these step values (default: all saved).")
    p.add_argument("--eval_num_steps", type=int, default=None,
                   help="Override sampler num_steps (default: from saved args).")
    p.add_argument("--eval_sample_bs", type=int, default=32,
                   help="(v2 only) Per-rank token-sampling batch. Default 32.")
    p.add_argument("--eval_decode_bs", type=int, default=16,
                   help="(v2 only) Per-rank decode-to-image batch. Default 16.")
    p.add_argument("--output_root", default="runs/eval_outputs",
                   help="Central output directory. CSV/JSON go here, and "
                        "annotated/<run_short>/step_*/ subtrees too. The "
                        "original training run dirs are left untouched.")
    p.add_argument("--output_csv", default=None,
                   help="Default: <output_root>/eval_sweep.csv")
    p.add_argument("--output_json", default=None,
                   help="Default: <output_root>/eval_sweep.json")
    p.add_argument("--aggregate_only", action="store_true",
                   help="Skip launching; just aggregate existing JSONs.")
    p.add_argument("--reuse_existing", action="store_true",
                   help="Skip a (run, step, sampler) if its cond_eval.json "
                        "already exists (assumes you trust them).")
    p.add_argument("--annotate_last", action="store_true",
                   help="After sweep, also run the annotator on the LAST "
                        "ckpt's grid PNG for each run (4-GPU sharded).")
    p.add_argument("--inference_semi_ar_for", nargs="*", default=[],
                   help="(v2 only) For each run dir in this list, run an "
                        "EXTRA eval pass with --semi_autoregressive forced "
                        "on at inference time (model trained without it). "
                        "Outputs go to a sibling '<run>_infer_semiar' dir. "
                        "Existing outputs of the original run are left "
                        "untouched.")
    return p.parse_args()


def main():
    args = parse_args()
    all_rows: List[Dict] = []

    output_root = args.output_root.rstrip("/")
    os.makedirs(output_root, exist_ok=True)
    out_csv = args.output_csv or os.path.join(output_root, "eval_sweep.csv")
    out_json = args.output_json or os.path.join(output_root, "eval_sweep.json")

    inference_semi_ar_set = {rd.rstrip("/") for rd in args.inference_semi_ar_for}

    # ── Phase 1: regular eval per run dir ─────────────────────────────────
    for rd in args.run_dirs:
        rd = rd.rstrip("/")
        kind = detect_kind(rd)
        saved = load_saved_args(rd, kind)
        ckpts = find_ckpt_steps(rd, kind)
        if args.steps is not None:
            ckpts = [(s, p) for s, p in ckpts if s in set(args.steps)]
        print(f"\n========== {rd}  [{kind}]  {len(ckpts)} ckpt(s) ==========")
        for s, p in ckpts:
            print(f"  step={s}  ckpt={p}")

        if not args.aggregate_only:
            base_argv = args_dict_to_argv(saved, kind)
            for step, ckpt_path in ckpts:
                if args.reuse_existing and _ckpt_already_evaled(rd, kind, step):
                    print(f"  [skip] step {step} already has cond_eval JSON(s)")
                    continue
                rc = launch_eval(rd, kind, ckpt_path, base_argv,
                                 num_samples_per_split=args.num_samples_per_split,
                                 gpus=args.gpus,
                                 eval_num_steps=args.eval_num_steps,
                                 eval_sample_bs=args.eval_sample_bs,
                                 eval_decode_bs=args.eval_decode_bs)
                if rc != 0:
                    print(f"  [WARN] launch failed for step {step} (rc={rc})")

        rows = collect_results(rd, kind)
        all_rows.extend(rows)

        if args.annotate_last:
            annotate_last_grid(rd, kind, args.gpus, output_root)

    # ── Phase 2: inference-only semi-AR shadow runs ───────────────────────
    for rd in args.run_dirs:
        rd = rd.rstrip("/")
        if rd not in inference_semi_ar_set:
            continue
        kind = detect_kind(rd)
        if kind != "v2":
            print(f"[infer-semiar] skip {rd}: not a v2 run")
            continue
        shadow = rd + "_infer_semiar"
        os.makedirs(shadow, exist_ok=True)
        # Mirror the run config so collect_results / annotate_last_grid can
        # detect the shadow as a v2 run.
        rc_path = os.path.join(rd, "run_config.json")
        if os.path.isfile(rc_path) and not os.path.isfile(
                os.path.join(shadow, "run_config.json")):
            import shutil
            shutil.copy(rc_path, os.path.join(shadow, "run_config.json"))

        saved = load_saved_args(rd, kind)
        ckpts = find_ckpt_steps(rd, kind)
        if args.steps is not None:
            ckpts = [(s, p) for s, p in ckpts if s in set(args.steps)]
        print(f"\n========== [INFER-SEMI-AR] {rd}  →  {shadow}  "
              f"{len(ckpts)} ckpt(s) ==========")
        if not args.aggregate_only:
            base_argv = args_dict_to_argv(saved, kind)
            for step, ckpt_path in ckpts:
                rc = launch_eval(rd, kind, ckpt_path, base_argv,
                                 num_samples_per_split=args.num_samples_per_split,
                                 gpus=args.gpus,
                                 eval_num_steps=args.eval_num_steps,
                                 eval_sample_bs=args.eval_sample_bs,
                                 eval_decode_bs=args.eval_decode_bs,
                                 force_semi_ar=True,
                                 shadow_output_dir=shadow)
                if rc != 0:
                    print(f"  [WARN] launch failed for step {step} (rc={rc})")

        rows = collect_results(shadow, kind)
        # Tag rows with the shadow run path so they're distinguishable
        for r in rows:
            r["run"] = shadow
        all_rows.extend(rows)

        if args.annotate_last:
            annotate_last_grid(shadow, kind, args.gpus, output_root)

    write_csv(all_rows, out_csv)
    write_json(all_rows, out_json)


def _ckpt_already_evaled(run_dir: str, kind: str, step: int) -> bool:
    if kind in ("v2", "v2_slot"):
        eval_dir = os.path.join(run_dir, "eval_samples")
        return any(
            f.startswith(f"step_{step:07d}_clevr_") and f.endswith("_cond_eval.json")
            for f in (os.listdir(eval_dir) if os.path.isdir(eval_dir) else []))
    eval_dir = os.path.join(run_dir, "clevr_eval")
    return os.path.isfile(
        os.path.join(eval_dir, f"clevr_eval_step{step:07d}.json"))


if __name__ == "__main__":
    main()
