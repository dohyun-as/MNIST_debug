#!/usr/bin/env python
"""
annotate_clevr_eval_grid.py
===========================
Post-hoc annotator for the CLEVR-eval grids produced during training.

Both `train_discrete_diffusion_v2.py` and `train_text_conditioned.py` save
a paired (GT, gen) grid PNG plus a JSON aggregate score, but they do NOT
draw bounding boxes around detected objects, and the user can't tell at a
glance whether a generation is being marked wrong because the model is
actually wrong or because the detector/classifier got confused.

This tool:
  1. Splits the saved grid PNG back into per-sample (GT, gen) cells.
  2. Runs the same CLEVR detector + classifier on each cell.
  3. Draws coloured bounding boxes:
       - green = the detected object matches at least one entity in the
                condition (i.e. its color/shape/size/material align)
       - red   = detected object does not match any entity in the condition
                (extra / spurious / wrong attributes)
  4. Writes per-sample annotated PNGs (with the caption + per-entity and
     per-relation pass/fail status), a combined annotated grid PNG, and a
     `*_annotations.json` containing the same info in machine-readable form.

Two condition sources are supported:
  --meta_json        path to a `step_*_clevr_meta.json` (saved by v2)
  --derive_naive_t2i runs the same deterministic selector used by
                     `train_text_conditioned.py` against the val dataset
                     and rebuilds conditions for the existing grid.

Usage:
  python src/annotate_clevr_eval_grid.py \
      --grid_png runs/.../step_0200000_clevr_confidence_top1.png \
      --meta_json runs/.../step_0200000_clevr_meta.json \
      --output_dir runs/.../annotated/step_0200000_confidence_top1

  python src/annotate_clevr_eval_grid.py \
      --grid_png runs/.../naive_dit_256_text_cond_clip/clevr_eval/clevr_eval_step0200000.png \
      --derive_naive_t2i \
      --val_image_root ../clevr-dataset-gen/output/clevr_256_varied_val/images \
      --val_cond_dir ../clevr-dataset-gen/output/clevr_256_varied_val/conditions_text \
      --num_samples_per_split 30 \
      --output_dir runs/.../naive_dit_256_text_cond_clip/clevr_eval/annotated/step_0200000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

# Wire up clevr_eval / our eval helpers (same trick the training scripts use)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_clevr_condition import (  # noqa: E402
    ATTR_NAMES,
    ATTR_VOCAB,
    RELATION_MARGIN,
    check_relation,
    clevr_text_to_condition_json,
    detect_and_classify,
    evaluate_condition_alignment,
    load_eval_models,
)


_FONT_PATH = "/opt/conda/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"
_FONT_PATH_BOLD = "/opt/conda/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = _FONT_PATH_BOLD if bold else _FONT_PATH
    if os.path.isfile(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ───────────────────────────────────────── grid splitting ──────────────────

def _slice_grid(grid_path: str, image_size: int = 256, padding: int = 2,
                nrow: int = 8) -> Tuple[List[np.ndarray], int, int]:
    """Split a torchvision-style grid PNG into individual cells.

    Returns
    -------
    cells   : list of (H, W, 3) uint8 numpy arrays in row-major order
    n_rows  : the actual number of rows on the grid
    n_cols  : the actual number of cols (== nrow except possibly the last row)
    """
    img = np.array(Image.open(grid_path).convert("RGB"))
    H, W, _ = img.shape
    cell_step = image_size + padding
    n_cols = (W - padding) // cell_step
    n_rows = (H - padding) // cell_step
    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            x = c * cell_step + padding
            y = r * cell_step + padding
            cell = img[y:y + image_size, x:x + image_size]
            if cell.shape[0] != image_size or cell.shape[1] != image_size:
                # last row partial — skip
                continue
            cells.append(cell)
    return cells, n_rows, n_cols


def _is_blank(cell: np.ndarray, std_thresh: float = 1.0) -> bool:
    """A pure-black padding cell will have std ≈ 0."""
    return float(cell.std()) < std_thresh


# ─────────────────────────────────────── detection + matching ──────────────

def _detect_one(cell_np: np.ndarray, detector, classifier, device,
                det_threshold: float = 0.3) -> List[Dict]:
    """Run detector + classifier on a single (H, W, 3) uint8 cell."""
    img_t = torch.from_numpy(cell_np).float().permute(2, 0, 1) / 255.0  # (3, H, W)
    img_t = img_t.unsqueeze(0).to(device)
    objs = detect_and_classify(img_t, detector, classifier,
                               det_threshold=det_threshold)[0]
    return objs


def _det_obj_matches_any_entity(det_obj: Dict,
                                cond_entities: List[Dict]) -> Optional[int]:
    """Return the index of the first condition entity whose specified attrs
    are ALL satisfied by det_obj, or None if none match."""
    for ei, ent in enumerate(cond_entities):
        attrs = {k: v for k, v in ent.get("attrs", {}).items() if k in ATTR_NAMES}
        if not attrs:
            continue
        if all(det_obj.get(k) == v for k, v in attrs.items()):
            return ei
    return None


def _entity_label(ent: Dict) -> str:
    """Human-readable label for an entity dict, e.g. 'large cyan rubber cylinder'.

    Only words that appear in the caption (i.e. that the entity actually
    constrains) are emitted, in canonical order: size → color → material → shape.
    """
    a = ent.get("attrs", {}) if isinstance(ent, dict) else {}
    parts = []
    for k in ("size", "color", "material", "shape"):
        if k in a and a[k] is not None:
            parts.append(str(a[k]))
    return " ".join(parts) if parts else (ent.get("name", "?") if isinstance(ent, dict) else "?")


def _det_label(d: Dict) -> str:
    """Same canonical phrasing for a detector output."""
    parts = []
    for k in ("size", "color", "material", "shape"):
        if d.get(k):
            parts.append(str(d[k]))
    return " ".join(parts) if parts else "?"


def _gt_obj_to_attr_dict(obj: Dict) -> Dict[str, str]:
    """Pick the four attrs out of a CLEVR scene-JSON object dict."""
    return {k: obj[k] for k in ("color", "shape", "size", "material") if k in obj}


def _scene_objs_for_image(image_filename: Optional[str], split: Optional[str],
                          val_scenes_dir: Optional[str]) -> List[Dict]:
    """Load GT scene objects for a given image, returning a list of dicts:
    {center: (x, y), color, shape, size, material}.

    Returns [] when scene JSON can't be located.
    """
    if not image_filename or not val_scenes_dir:
        return []
    candidates = []
    if split:
        candidates.append(os.path.join(val_scenes_dir, split,
                                       image_filename.replace(".png", ".json")))
    candidates.append(os.path.join(val_scenes_dir,
                                   image_filename.replace(".png", ".json")))
    scene_path = next((p for p in candidates if os.path.isfile(p)), None)
    if scene_path is None:
        return []
    with open(scene_path) as f:
        scene = json.load(f)
    out = []
    for o in scene.get("objects", []):
        px = o.get("pixel_coords", [0, 0, 0])
        out.append({"center": (int(px[0]), int(px[1])),
                    "color": o.get("color"),
                    "shape": o.get("shape"),
                    "size": o.get("size"),
                    "material": o.get("material")})
    return out


def _attrs_match(spec: Dict, obj: Dict) -> bool:
    """obj satisfies every attribute that spec specifies."""
    for k, v in spec.items():
        if k in ATTR_NAMES and v is not None and obj.get(k) != v:
            return False
    return True


# ──────────────────────────────── per-sample annotated PNG ─────────────────

def _draw_box_with_label(draw: ImageDraw.ImageDraw, canvas: Image.Image,
                         cx: int, cy: int, label: str, colour: Tuple[int, int, int],
                         font, box_size: int = 48, label_above: bool = True) -> None:
    half = box_size // 2
    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, canvas.width - 1)
    y2 = min(cy + half, canvas.height - 1)
    draw.rectangle([x1, y1, x2, y2], outline=colour, width=2)
    if not label:
        return
    tw = draw.textlength(label, font=font)
    th = 13
    if label_above:
        ty = max(y1 - th - 1, 0)
    else:
        ty = min(y2 + 1, canvas.height - th - 1)
    tx = max(min(x1, canvas.width - int(tw) - 4), 0)
    draw.rectangle([tx, ty, tx + tw + 4, ty + th], fill=colour)
    draw.text((tx + 2, ty), label, fill="white", font=font)


def _draw_gen_annotations(canvas: Image.Image,
                          detected_objs: List[Dict],
                          cond_entities: List[Dict],
                          box_size: int = 48) -> None:
    """GEN side — green = matches a condition entity, red = extra/spurious."""
    draw = ImageDraw.Draw(canvas)
    font = _load_font(11, bold=True)
    for d in detected_objs:
        cx, cy = d["center"]
        match_idx = _det_obj_matches_any_entity(d, cond_entities)
        colour = (16, 200, 64) if match_idx is not None else (235, 50, 50)
        _draw_box_with_label(draw, canvas, int(cx), int(cy),
                             _det_label(d), colour, font, box_size=box_size)


def _draw_gt_annotations(canvas: Image.Image,
                         gt_objs: List[Dict],
                         cond_entities: List[Dict],
                         entity_details: List[Dict],
                         box_size: int = 48) -> List[Dict]:
    """GT side — boxes mark which entities GEN failed to reproduce.

    Strategy:
      - Iterate cond entities. If sample_eval marks the entity as found in GEN,
        skip (we don't need to highlight successful ones on the GT — we draw
        them in green for context but with a thinner outline).
      - If NOT found in GEN, locate the corresponding GT object (by attr match)
        and draw a thick orange box labelled "MISSED: <entity>".

    Returns the list of "missed" annotations actually drawn (for the JSON record).
    """
    draw = ImageDraw.Draw(canvas)
    font_b = _load_font(11, bold=True)
    font_s = _load_font(11)

    used_gt = set()  # so two missed entities don't both claim the same GT obj
    missed_records = []

    # Map entity name -> "found" status from sample_eval
    found_by_name = {e["name"]: e.get("found", False)
                     for e in (entity_details or [])}

    # First pass: missed entities — orange thick boxes on GT
    for ent in cond_entities:
        if found_by_name.get(ent["name"], False):
            continue
        spec = ent.get("attrs", {})
        # find best GT match
        chosen = None
        for gi, g in enumerate(gt_objs):
            if gi in used_gt:
                continue
            if _attrs_match(spec, g):
                chosen = (gi, g)
                break
        if chosen is None:
            missed_records.append({"entity": ent["name"],
                                   "label": _entity_label(ent),
                                   "gt_box": None,
                                   "note": "no_gt_match"})
            continue
        gi, g = chosen
        used_gt.add(gi)
        cx, cy = g["center"]
        _draw_box_with_label(draw, canvas, int(cx), int(cy),
                             "MISS: " + _entity_label(ent),
                             (255, 140, 0), font_b, box_size=box_size)
        missed_records.append({"entity": ent["name"],
                               "label": _entity_label(ent),
                               "gt_box": [int(cx), int(cy)],
                               "note": "missed_in_gen"})

    # Second pass: ENTITIES that were found in GEN — show their GT counterpart
    # in thin green so the user can see "this object the model got right".
    for ent in cond_entities:
        if not found_by_name.get(ent["name"], False):
            continue
        spec = ent.get("attrs", {})
        for gi, g in enumerate(gt_objs):
            if gi in used_gt:
                continue
            if _attrs_match(spec, g):
                used_gt.add(gi)
                cx, cy = g["center"]
                # thin green: draw rect with width=1
                half = box_size // 2
                x1, y1 = max(int(cx) - half, 0), max(int(cy) - half, 0)
                x2, y2 = min(int(cx) + half, canvas.width - 1), min(int(cy) + half, canvas.height - 1)
                draw.rectangle([x1, y1, x2, y2], outline=(40, 170, 60), width=1)
                # tiny label below the box
                label = _entity_label(ent)
                tw = draw.textlength(label, font=font_s)
                ty = min(y2 + 1, canvas.height - 13)
                tx = max(min(x1, canvas.width - int(tw) - 4), 0)
                draw.rectangle([tx, ty, tx + tw + 4, ty + 13], fill=(40, 170, 60))
                draw.text((tx + 2, ty), label, fill="white", font=font_s)
                break

    return missed_records


def _make_status_panel(width: int,
                       caption: str,
                       sample_eval: Dict,
                       cond_entities: List[Dict],
                       cond_relations: List[Dict],
                       split: Optional[str] = None) -> Image.Image:
    """Render a text panel with caption + per-entity / per-relation status.

    Auto-grows in height until everything fits.
    """
    font_cap = _load_font(13, bold=True)
    font_text = _load_font(12)
    font_text_b = _load_font(12, bold=True)

    # Wrap the caption
    def _wrap(text: str, w: int, font) -> List[str]:
        words = text.split()
        lines, line = [], ""
        tmp_img = Image.new("RGB", (w, 16))
        d = ImageDraw.Draw(tmp_img)
        for word in words:
            cand = (line + " " + word).strip()
            if d.textlength(cand, font=font) <= w - 10:
                line = cand
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        return lines

    cap_lines = _wrap(f"[{split or '?'}] " + caption, width, font_cap)

    entity_details = sample_eval.get("entity_details", [])
    rel_details = sample_eval.get("rel_details", [])

    n_lines = (len(cap_lines)
               + 2  # header + count line
               + max(len(entity_details), 1) + 1  # entity header + entries
               + (max(len(rel_details), 1) + 1 if cond_relations else 0))
    height = 18 + n_lines * 16 + 8

    panel = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(panel)

    y = 6
    for ln in cap_lines:
        draw.text((6, y), ln, fill=(0, 0, 0), font=font_cap)
        y += 16
    y += 4

    cp = sample_eval.get("count_pred", -1)
    cg = sample_eval.get("count_gt", -1)
    cc = sample_eval.get("count_correct", False)
    col = (16, 130, 32) if cc else (200, 0, 0)
    draw.text((6, y), f"count: pred={cp}  gt={cg}  {'OK' if cc else 'BAD'}",
              fill=col, font=font_text_b)
    y += 18

    draw.text((6, y), "entities:", fill=(0, 0, 0), font=font_text_b)
    y += 16
    if not entity_details:
        draw.text((20, y), "(none)", fill=(80, 80, 80), font=font_text)
        y += 16
    # Build a quick map name -> entity dict for natural labels
    name_to_entity = {ent["name"]: ent for ent in cond_entities}
    for e in entity_details:
        ent = name_to_entity.get(e["name"], {"attrs": e.get("attrs", {})})
        nl = _entity_label(ent)
        col = (16, 130, 32) if e["found"] else (200, 0, 0)
        mark = "OK " if e["found"] else "MISS"
        draw.text((20, y),
                  f"[{mark}] {e['name']}: {nl}  (matches={e['n_matches']})",
                  fill=col, font=font_text)
        y += 16

    if cond_relations:
        draw.text((6, y), "relations:", fill=(0, 0, 0), font=font_text_b)
        y += 16
        if not rel_details:
            draw.text((20, y), "(none)", fill=(80, 80, 80), font=font_text)
            y += 16
        rel_phrase = {"left_of": "is to the left of",
                      "right_of": "is to the right of",
                      "in_front_of": "is in front of",
                      "behind": "is behind"}
        for r in rel_details:
            ok = r.get("correct", False)
            col = (16, 130, 32) if ok else (200, 0, 0)
            mark = "OK " if ok else "FAIL"
            subj_lbl = _entity_label(name_to_entity.get(r["subj"], {}))
            obj_lbl = _entity_label(name_to_entity.get(r["obj"], {}))
            phrase = rel_phrase.get(r["rel"], r["rel"])
            reason = ("" if ok else f"  [{r.get('reason','no_pair')}]")
            draw.text((20, y),
                      f"[{mark}] ({r['subj']}) {subj_lbl} {phrase} ({r['obj']}) {obj_lbl}{reason}",
                      fill=col, font=font_text)
            y += 16

    return panel


def _compose_sample(gt_cell: np.ndarray, gen_cell: np.ndarray,
                    gen_objs: List[Dict], gt_objs: List[Dict],
                    sample_eval: Dict,
                    cond_entities: List[Dict],
                    cond_relations: List[Dict],
                    caption: str,
                    split: Optional[str]) -> Tuple[Image.Image, List[Dict]]:
    """GT + GEN side-by-side, both annotated, plus a status panel.

    GT side: orange thick box on entities the GEN missed; thin green box on
             entities the GEN reproduced (so the user can match them up).
    GEN side: green box on detections that match a condition entity, red box
              on extras / spurious detections.

    Returns (final_image, missed_records) — missed_records is the structured
    list of "MISS" entries actually drawn on the GT, suitable for the JSON
    record.
    """
    h, w = gt_cell.shape[:2]
    gt_img = Image.fromarray(gt_cell.copy())
    gen_img = Image.fromarray(gen_cell.copy())

    missed_records = _draw_gt_annotations(
        gt_img, gt_objs, cond_entities,
        sample_eval.get("entity_details", []))
    _draw_gen_annotations(gen_img, gen_objs, cond_entities)

    # Header strip with GT / GEN labels
    header_h = 22
    header = Image.new("RGB", (w * 2 + 6, header_h), (235, 235, 235))
    hd = ImageDraw.Draw(header)
    f = _load_font(13, bold=True)
    hd.text((6, 4),
            "GT  (orange=MISSED in gen / thin green=reproduced)",
            fill=(0, 0, 0), font=f)
    hd.text((w + 6 + 6, 4),
            "GEN (green=matches a condition / red=extra)",
            fill=(0, 0, 0), font=f)

    pair = Image.new("RGB", (w * 2 + 6, h), (255, 255, 255))
    pair.paste(gt_img, (0, 0))
    pair.paste(gen_img, (w + 6, 0))

    panel_w = w * 2 + 6
    panel = _make_status_panel(panel_w, caption, sample_eval,
                               cond_entities, cond_relations, split)

    total_h = header_h + h + panel.height
    final = Image.new("RGB", (panel_w, total_h), (255, 255, 255))
    final.paste(header, (0, 0))
    final.paste(pair, (0, header_h))
    final.paste(panel, (0, header_h + h))
    return final, missed_records


# ───────────────────────────────────────── derive naive T2I conditions ─────

def _derive_naive_t2i_conditions(val_image_root: str, val_cond_dir: str,
                                 splits: List[str], n_per_split: int,
                                 image_size: int, seed: int = 42
                                 ) -> Tuple[List[Dict], List[str]]:
    """Replay the deterministic selection used by `train_text_conditioned.py`.

    Returns (cond_jsons, split_labels) with the same length and order as the
    grid was saved. cond_jsons items are {"text", "image_filename", "split"}.
    """
    sys.path.insert(0, _HERE)
    from train_text_conditioned import (  # noqa: E402
        CLEVRTextCondDataset, _select_eval_indices_balanced)

    val_ds = CLEVRTextCondDataset(
        val_image_root, val_cond_dir, splits, image_size,
        augment=False, mode="pretrained", cond_type="text")

    selected_indices, sample_splits = _select_eval_indices_balanced(
        val_ds, n_per_split)

    cond_jsons = []
    for idx, sp in zip(selected_indices, sample_splits):
        text = val_ds.get_condition(idx)
        img_path = val_ds.samples[idx][0]
        cond_jsons.append({
            "text": text if isinstance(text, str) else str(text),
            "image_filename": os.path.basename(img_path),
            "split": sp,
        })
    return cond_jsons, sample_splits


# ─────────────────────────────────────────────────── main runner ───────────

def annotate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"[annotate] loading detector + classifier on {device} …")
    detector, classifier = load_eval_models(device=device)

    # Resolve conditions ----------------------------------------------------
    if args.meta_json:
        with open(args.meta_json) as f:
            meta = json.load(f)
        cond_jsons = meta["conditions"]
        sample_splits = meta.get("splits", [None] * len(cond_jsons))
    else:
        cond_jsons, sample_splits = _derive_naive_t2i_conditions(
            args.val_image_root, args.val_cond_dir, args.splits,
            args.num_samples_per_split, args.image_size)
    n_samples = len(cond_jsons)
    print(f"[annotate] resolved {n_samples} conditions")

    # Slice grid ------------------------------------------------------------
    cells, n_rows, n_cols = _slice_grid(
        args.grid_png, image_size=args.image_size,
        padding=args.grid_padding, nrow=args.grid_nrow)
    print(f"[annotate] grid={args.grid_png}  rows={n_rows} cols={n_cols} "
          f"-> {len(cells)} cells")

    # Filter blank cells in-place (last row may be partially padded)
    cells = [c for c in cells if not _is_blank(c)]
    if len(cells) < 2 * n_samples:
        raise RuntimeError(
            f"Grid has {len(cells)} non-blank cells but expected ≥{2*n_samples} "
            f"(GT + gen pairs). Check --image_size / --grid_padding / --grid_nrow.")

    cells = cells[: 2 * n_samples]  # pair-aligned

    # Optionally restrict to this shard ------------------------------------
    if args.num_shards > 1:
        my_indices = [i for i in range(n_samples)
                      if (i % args.num_shards) == args.shard_idx]
        print(f"[annotate] shard {args.shard_idx}/{args.num_shards} → "
              f"{len(my_indices)} samples")
    else:
        my_indices = list(range(n_samples))

    # For each pair, run detector on both GT and GEN -----------------------
    per_sample_records = []
    annotated_images = []
    agg = {"count_correct": 0, "entity_found": 0, "entity_total": 0,
           "rel_correct": 0, "rel_total": 0, "n": 0}

    for i in my_indices:
        gt_cell = cells[2 * i]
        gen_cell = cells[2 * i + 1]
        cond = cond_jsons[i]
        caption = (cond.get("text") if isinstance(cond, dict)
                   else cond if isinstance(cond, str) else "")
        split = (cond.get("split") if isinstance(cond, dict) else
                 (sample_splits[i] if i < len(sample_splits) else None))
        image_filename = (cond.get("image_filename")
                          if isinstance(cond, dict) else None)

        cond_struct = clevr_text_to_condition_json(caption)
        cond_entities = cond_struct["entities"]
        cond_relations = cond_struct["relations"]

        # GT objects — prefer ground-truth scene JSON (precise), fall back
        # to running the detector on the GT cell (less precise but avoids
        # requiring scene access).
        gt_objs = _scene_objs_for_image(image_filename, split, args.val_scenes_dir)
        if not gt_objs:
            gt_objs = _detect_one(gt_cell, detector, classifier, device,
                                  det_threshold=args.det_threshold)
        gen_objs = _detect_one(gen_cell, detector, classifier, device,
                               det_threshold=args.det_threshold)
        sample_eval = evaluate_condition_alignment(
            gen_objs, cond_struct, relation_margin=RELATION_MARGIN)

        # Aggregate
        agg["n"] += 1
        if sample_eval["count_correct"]:
            agg["count_correct"] += 1
        agg["entity_found"] += sample_eval["entity_found"]
        agg["entity_total"] += sample_eval["entity_total"]
        agg["rel_correct"] += sample_eval["rel_correct"]
        agg["rel_total"] += sample_eval["rel_total"]

        # Compose annotated PNG
        composed, missed_records = _compose_sample(
            gt_cell, gen_cell, gen_objs, gt_objs,
            sample_eval, cond_entities, cond_relations, caption, split)
        out_name = f"sample_{i:03d}_{split or 'na'}.png"
        composed.save(os.path.join(args.output_dir, out_name))
        annotated_images.append(np.array(composed))

        # JSON record
        # Use natural-language labels for entity / relation status so the
        # JSON is human-readable too.
        name_to_entity = {ent["name"]: ent for ent in cond_entities}
        entities_status = []
        for e in sample_eval["entity_details"]:
            ent = name_to_entity.get(e["name"], {"attrs": e.get("attrs", {})})
            entities_status.append({
                "name": e["name"],
                "label": _entity_label(ent),
                "attrs": e["attrs"],
                "found_in_gen": bool(e["found"]),
                "n_gen_matches": e["n_matches"],
            })
        relations_status = []
        rel_phrase = {"left_of": "is to the left of",
                      "right_of": "is to the right of",
                      "in_front_of": "is in front of",
                      "behind": "is behind"}
        for r in sample_eval["rel_details"]:
            relations_status.append({
                "subj": r["subj"],
                "subj_label": _entity_label(name_to_entity.get(r["subj"], {})),
                "rel": r["rel"],
                "rel_phrase": rel_phrase.get(r["rel"], r["rel"]),
                "obj": r["obj"],
                "obj_label": _entity_label(name_to_entity.get(r["obj"], {})),
                "correct": bool(r.get("correct", False)),
                "reason": r.get("reason"),
            })
        per_sample_records.append({
            "index": i,
            "split": split,
            "caption": caption,
            "image_filename": image_filename,
            "parsed_condition": cond_struct,
            "gt_objects": [{**d, "center": list(d["center"])} for d in gt_objs],
            "gen_detected": [{**d, "center": list(d["center"])} for d in gen_objs],
            "missed_in_gen": missed_records,
            "entities_status": entities_status,
            "relations_status": relations_status,
            "eval": {
                "count_pred": sample_eval["count_pred"],
                "count_gt": sample_eval["count_gt"],
                "count_correct": bool(sample_eval["count_correct"]),
                "entity_found": sample_eval["entity_found"],
                "entity_total": sample_eval["entity_total"],
                "rel_correct": sample_eval["rel_correct"],
                "rel_total": sample_eval["rel_total"],
            },
        })

        if (agg["n"] % 10 == 0) or (agg["n"] == len(my_indices)):
            print(f"  [{agg['n']}/{len(my_indices)}]  "
                  f"count={agg['count_correct']}/{agg['n']}  "
                  f"entity={agg['entity_found']}/{agg['entity_total']}  "
                  f"rel={agg['rel_correct']}/{agg['rel_total']}")

    # Aggregate score JSON --------------------------------------------------
    n = agg["n"]
    summary = {
        "grid_png": os.path.abspath(args.grid_png),
        "n_samples": n,
        "count_accuracy": agg["count_correct"] / n * 100 if n else 0.0,
        "entity_presence_accuracy": (
            agg["entity_found"] / agg["entity_total"] * 100
            if agg["entity_total"] else 0.0),
        "rel_accuracy": (agg["rel_correct"] / agg["rel_total"] * 100
                         if agg["rel_total"] else 0.0),
        "raw": agg,
    }
    print(f"[annotate] overall  "
          f"count={summary['count_accuracy']:.1f}%  "
          f"presence={summary['entity_presence_accuracy']:.1f}%  "
          f"rel={summary['rel_accuracy']:.1f}%")

    json_out = {
        "summary": summary,
        "per_sample": per_sample_records,
    }
    if args.num_shards > 1:
        out_json = os.path.join(
            args.output_dir,
            f"annotations.shard{args.shard_idx:02d}_of_{args.num_shards:02d}.json")
    else:
        out_json = os.path.join(args.output_dir, "annotations.json")
    with open(out_json, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"[annotate] wrote {out_json}")

    # Combined annotated grid (1 sample per row, captioned) ----------------
    # Only build it for single-shard runs; for sharded runs the merge step
    # below handles it.
    if args.num_shards == 1 and annotated_images:
        _make_all_annotated_png(args.output_dir, n_samples)


def _make_all_annotated_png(output_dir: str, n_samples: int):
    """Stack the per-sample PNGs that exist in `output_dir` into a single
    `all_annotated.png` (skipping any that aren't on disk)."""
    sample_files = []
    for i in range(n_samples):
        # filename pattern: sample_{i:03d}_{split}.png  — split varies
        for fn in sorted(os.listdir(output_dir)):
            if fn.startswith(f"sample_{i:03d}_") and fn.endswith(".png"):
                sample_files.append(os.path.join(output_dir, fn))
                break
    if not sample_files:
        return
    images = [np.array(Image.open(p)) for p in sample_files]
    max_w = max(im.shape[1] for im in images)
    total_h = sum(im.shape[0] for im in images) + 4 * len(images)
    big = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y = 0
    for im in images:
        tile = Image.fromarray(im)
        big.paste(tile, (0, y))
        y += tile.height + 4
    big_path = os.path.join(output_dir, "all_annotated.png")
    big.save(big_path)
    print(f"[annotate] wrote {big_path} ({big.size})")


def merge_shards(output_dir: str):
    """Combine `annotations.shard*_of_*.json` files into `annotations.json`
    and re-build `all_annotated.png` from the per-sample PNGs."""
    shard_files = sorted(
        f for f in os.listdir(output_dir)
        if f.startswith("annotations.shard") and f.endswith(".json"))
    if not shard_files:
        raise SystemExit(f"No shard JSONs in {output_dir}")
    print(f"[merge] combining {len(shard_files)} shard files")

    per_sample = []
    grid_png = None
    agg = {"count_correct": 0, "entity_found": 0, "entity_total": 0,
           "rel_correct": 0, "rel_total": 0, "n": 0}
    for fn in shard_files:
        with open(os.path.join(output_dir, fn)) as f:
            d = json.load(f)
        per_sample.extend(d["per_sample"])
        s = d["summary"]
        grid_png = grid_png or s.get("grid_png")
        for k in agg:
            agg[k] += s["raw"][k]
    per_sample.sort(key=lambda r: r["index"])

    n = agg["n"]
    summary = {
        "grid_png": grid_png,
        "n_samples": n,
        "count_accuracy": agg["count_correct"] / n * 100 if n else 0.0,
        "entity_presence_accuracy": (
            agg["entity_found"] / agg["entity_total"] * 100
            if agg["entity_total"] else 0.0),
        "rel_accuracy": (agg["rel_correct"] / agg["rel_total"] * 100
                         if agg["rel_total"] else 0.0),
        "raw": agg,
    }
    out_json = os.path.join(output_dir, "annotations.json")
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "per_sample": per_sample},
                  f, indent=2)
    print(f"[merge] wrote {out_json}")
    print(f"[merge] overall  "
          f"count={summary['count_accuracy']:.1f}%  "
          f"presence={summary['entity_presence_accuracy']:.1f}%  "
          f"rel={summary['rel_accuracy']:.1f}%")

    _make_all_annotated_png(output_dir, n)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grid_png", required=True,
                   help="The paired (GT, gen) grid PNG saved during training")
    p.add_argument("--output_dir", required=True)

    # Either --meta_json (v2) or --derive_naive_t2i path
    p.add_argument("--meta_json", default=None,
                   help="step_*_clevr_meta.json (v2 train script)")
    p.add_argument("--derive_naive_t2i", action="store_true",
                   help="Recover conditions by replaying the naive T2I "
                        "deterministic selector against the val dataset")
    p.add_argument("--val_image_root",
                   default="../clevr-dataset-gen/output/clevr_256_varied_val/images")
    p.add_argument("--val_cond_dir",
                   default="../clevr-dataset-gen/output/clevr_256_varied_val/conditions_text")
    p.add_argument("--val_scenes_dir",
                   default="../clevr-dataset-gen/output/clevr_256_varied_val/scenes",
                   help="GT scene JSONs — used to draw precise per-entity boxes "
                        "on the GT image (color/shape/size/material + pixel "
                        "coordinates). Falls back to the detector if missing.")
    p.add_argument("--splits", nargs="+", default=["easy", "medium", "hard"])
    p.add_argument("--num_samples_per_split", type=int, default=30)

    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--grid_padding", type=int, default=2)
    p.add_argument("--grid_nrow", type=int, default=8)

    p.add_argument("--det_threshold", type=float, default=0.3)
    p.add_argument("--device", default="cuda")

    # Sharding (for multi-GPU parallel runs — each process handles a subset)
    p.add_argument("--shard_idx", type=int, default=0,
                   help="0-based shard index for parallel annotation")
    p.add_argument("--num_shards", type=int, default=1,
                   help="If >1, only process samples where i %% num_shards == shard_idx")

    # Merge mode — combines previously-written shard JSONs into a final
    # annotations.json and rebuilds all_annotated.png. No detection runs.
    p.add_argument("--merge", action="store_true",
                   help="Merge shard JSONs in --output_dir; no detection.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.merge:
        merge_shards(args.output_dir)
        return
    if not args.meta_json and not args.derive_naive_t2i:
        raise SystemExit("Provide either --meta_json or --derive_naive_t2i")
    annotate(args)


if __name__ == "__main__":
    main()
