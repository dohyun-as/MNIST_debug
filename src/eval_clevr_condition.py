"""
Evaluate generated CLEVR images against conditions — simplified metrics.

Three headline scores:
  1. count_accuracy       — image-level: correct object count
  2. entity_presence_acc  — per entity: does a detected obj match ALL its attrs?
                            (no 1-to-1 matching; duplicates allowed)
  3. rel_accuracy         — per relation: find detected objs matching subj/obj
                            features, check if any pair satisfies the spatial rel
"""

import os
import sys
import json
from typing import List, Dict, Optional, Tuple, Set

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# Add clevr_eval to path
_CLEVR_EVAL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "clevr_eval"))
if _CLEVR_EVAL_DIR not in sys.path:
    sys.path.insert(0, _CLEVR_EVAL_DIR)

import config as clevr_cfg

# `models` is a package name that also exists under VAR/ and gets cached in
# sys.modules first when this file is imported from VAR/train_clevr_text.py,
# which shadows clevr_eval/models/. Load detector/classifier from file and
# register them under `models.detector` / `models.classifier` in sys.modules
# so both this file and clevr_eval/evaluate.py resolve to the right modules.
import importlib.util as _ilu

def _load_clevr_eval_module(dotted_name, file_path):
    spec = _ilu.spec_from_file_location(dotted_name, file_path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod

_detector_mod = _load_clevr_eval_module(
    "models.detector",
    os.path.join(_CLEVR_EVAL_DIR, "models", "detector.py"))
_classifier_mod = _load_clevr_eval_module(
    "models.classifier",
    os.path.join(_CLEVR_EVAL_DIR, "models", "classifier.py"))
CenterDetector = _detector_mod.CenterDetector
AttributeClassifier = _classifier_mod.AttributeClassifier
from evaluate import extract_peaks

# ── Attribute vocab ──────────────────────────────────────────────────────

ATTR_NAMES = ["color", "shape", "size", "material"]
ATTR_VOCAB = {
    "color": clevr_cfg.COLORS,
    "shape": clevr_cfg.SHAPES,
    "size": clevr_cfg.SIZES,
    "material": clevr_cfg.MATERIALS,
}

RELATION_MARGIN = 50

# Map from text relation phrases to relation keys
_REL_TEXT_MAP = {
    "to the left of": "left_of",
    "to the right of": "right_of",
    "in front of": "in_front_of",
    "behind": "behind",
}

# All known attribute words for parsing
_ALL_COLORS = set(ATTR_VOCAB["color"])
_ALL_SHAPES = set(ATTR_VOCAB["shape"])
_ALL_SIZES = set(ATTR_VOCAB["size"])
_ALL_MATERIALS = set(ATTR_VOCAB["material"])


def clevr_text_to_condition_json(caption: str) -> Dict:
    """Parse a CLEVR natural language caption into structured condition JSON.

    Expected format:
      "There are N objects: a [size] [color] [material] [shape], ... .
       The [attrs] [shape] is [relation] the [attrs] [shape]."

    Returns dict with "entities" and "relations" keys, compatible with
    evaluate_condition_alignment().
    """
    import re

    entities = []
    relations = []

    # Split into sentences
    sentences = [s.strip() for s in caption.split(".") if s.strip()]

    for sent in sentences:
        sent_lower = sent.lower()

        # Parse object list: "There are N objects: a ..., a ..., and a ..."
        if sent_lower.startswith("there"):
            # Extract object descriptions after the colon
            colon_pos = sent.find(":")
            if colon_pos == -1:
                continue
            obj_part = sent[colon_pos + 1:]
            # Split by comma and "and"
            obj_strs = re.split(r',\s*(?:and\s+)?|\s+and\s+', obj_part)
            for obj_str in obj_strs:
                obj_str = obj_str.strip().lower()
                if not obj_str:
                    continue
                # Remove leading "a " or "an "
                obj_str = re.sub(r'^an?\s+', '', obj_str)
                words = obj_str.split()
                attrs = {}
                for w in words:
                    if w in _ALL_SIZES:
                        attrs["size"] = w
                    elif w in _ALL_COLORS:
                        attrs["color"] = w
                    elif w in _ALL_MATERIALS:
                        attrs["material"] = w
                    elif w in _ALL_SHAPES:
                        attrs["shape"] = w
                if attrs:
                    name = chr(ord("A") + len(entities))
                    entities.append({"name": name, "attrs": attrs})

        # Parse relations: "The [attrs] [shape] is [rel] the [attrs] [shape]"
        elif " is " in sent_lower:
            # Try each relation phrase
            rel_type = None
            rel_pos = -1
            for phrase, rkey in _REL_TEXT_MAP.items():
                pos = sent_lower.find(phrase)
                if pos != -1:
                    rel_type = rkey
                    rel_pos = pos
                    phrase_len = len(phrase)
                    break
            if rel_type is None:
                continue

            subj_part = sent[:rel_pos].strip().lower()
            obj_part = sent[rel_pos + phrase_len:].strip().lower()

            # Remove "The " prefix
            subj_part = re.sub(r'^the\s+', '', subj_part)
            # Remove " is" suffix from subject
            subj_part = re.sub(r'\s+is$', '', subj_part)
            obj_part = re.sub(r'^the\s+', '', obj_part)

            def _parse_attrs(text):
                words = text.split()
                attrs = {}
                for w in words:
                    if w in _ALL_SIZES:
                        attrs["size"] = w
                    elif w in _ALL_COLORS:
                        attrs["color"] = w
                    elif w in _ALL_MATERIALS:
                        attrs["material"] = w
                    elif w in _ALL_SHAPES:
                        attrs["shape"] = w
                return attrs

            subj_attrs = _parse_attrs(subj_part)
            obj_attrs = _parse_attrs(obj_part)

            # Find matching entities
            def _find_entity(attrs):
                for ent in entities:
                    if all(ent["attrs"].get(k) == v for k, v in attrs.items()):
                        return ent["name"]
                return None

            subj_name = _find_entity(subj_attrs)
            obj_name = _find_entity(obj_attrs)

            if subj_name and obj_name:
                relations.append({
                    "subj": subj_name, "rel": rel_type, "obj": obj_name
                })

    return {"entities": entities, "relations": relations}


# ── Load eval models ──────────────────────────────────────────────────────

def load_eval_models(device="cuda"):
    """Load pretrained detector + classifier. Returns (detector, classifier)."""
    det_ckpt = os.path.join(clevr_cfg.CHECKPOINT_DIR, "detector_best.pt")
    cls_ckpt = os.path.join(clevr_cfg.CHECKPOINT_DIR, "classifier_best.pt")

    detector = CenterDetector(backbone_name=clevr_cfg.DETECTOR_BACKBONE).to(device)
    detector.load_state_dict(
        torch.load(det_ckpt, map_location=device, weights_only=True)["model"])
    detector.eval()

    classifier = AttributeClassifier().to(device)
    classifier.load_state_dict(
        torch.load(cls_ckpt, map_location=device, weights_only=True)["model"])
    classifier.eval()

    return detector, classifier


# ── Detection + Classification ───────────────────────────────────────────

_det_normalize = transforms.Normalize([0.5]*3, [0.5]*3)


@torch.no_grad()
def detect_and_classify(images: torch.Tensor,
                        detector: torch.nn.Module,
                        classifier: torch.nn.Module,
                        det_threshold: float = 0.3,
                        ) -> List[List[Dict]]:
    """Run detection + classification on a batch of images.

    Args:
        images: (B, 3, H, W) tensor in [0, 1] range.

    Returns:
        List of B lists, each containing dicts per detected object:
        {"center": (x, y), "color": str, "shape": str, "size": str, "material": str}
    """
    device = next(detector.parameters()).device
    B = images.shape[0]

    det_input = torch.stack([_det_normalize(images[i]) for i in range(B)]).to(device)
    heatmaps = detector(det_input).cpu().numpy()[:, 0]

    results = []
    for b in range(B):
        peaks = extract_peaks(heatmaps[b], threshold=det_threshold)
        if not peaks:
            results.append([])
            continue

        half = clevr_cfg.CROP_SIZE // 2
        H, W = images.shape[2], images.shape[3]
        crops = []
        centers = []
        for (px, py, score) in peaks:
            x1, y1 = max(px - half, 0), max(py - half, 0)
            x2, y2 = min(px + half, W), min(py + half, H)
            crop = images[b:b+1, :, y1:y2, x1:x2]
            crop = F.interpolate(crop, size=(clevr_cfg.CROP_SIZE, clevr_cfg.CROP_SIZE),
                                 mode="bilinear", align_corners=False)
            crop = transforms.Normalize([0.5]*3, [0.5]*3)(crop.squeeze(0))
            crops.append(crop)
            centers.append((px, py))

        crop_batch = torch.stack(crops).to(device)
        preds = classifier(crop_batch)

        objs = []
        for k in range(len(peaks)):
            obj = {"center": centers[k]}
            for attr_name in ATTR_NAMES:
                idx = preds[attr_name][k].argmax().item()
                obj[attr_name] = ATTR_VOCAB[attr_name][idx]
            objs.append(obj)
        results.append(objs)

    return results


# ── Spatial relation checking ────────────────────────────────────────────

def check_relation(subj_center, obj_center, rel, margin=RELATION_MARGIN):
    sx, sy = subj_center
    ox, oy = obj_center
    if rel == "left_of":
        return sx < ox - margin
    elif rel == "right_of":
        return sx > ox + margin
    elif rel == "in_front_of":
        return sy > oy + margin
    elif rel == "behind":
        return sy < oy - margin
    return False


# ── Helper: find detected objects matching an entity's attrs ─────────────

def _find_matching_detections(entity_attrs: Dict, detected_objs: List[Dict]) -> List[int]:
    """Return indices of detected objects that match ALL specified attrs of an entity."""
    specified = {k: v for k, v in entity_attrs.items() if k in ATTR_NAMES}
    if not specified:
        # No attrs specified → all detected objects are candidates
        return list(range(len(detected_objs)))
    matches = []
    for di, det_obj in enumerate(detected_objs):
        if all(det_obj.get(k) == v for k, v in specified.items()):
            matches.append(di)
    return matches


# ── Single-sample evaluation ─────────────────────────────────────────────

def evaluate_condition_alignment(
    detected_objs: List[Dict],
    condition: Dict,
    relation_margin: float = RELATION_MARGIN,
) -> Dict:
    """Evaluate a single generated image against its condition (simplified).

    Metrics:
      1. count_correct: detected count == condition entity count
      2. entity_found / entity_total: for each entity, is there ANY detected
         object matching ALL its specified attrs? (duplicates allowed)
      3. rel_correct / rel_total: for each relation, find detected objects
         matching subj/obj features; check if any pair satisfies the relation
    """
    cond_entities = condition.get("entities", [])
    cond_relations = condition.get("relations", [])

    n_det = len(detected_objs)
    n_cond = len(cond_entities)

    result = {
        "count_pred": n_det,
        "count_gt": n_cond,
        "count_correct": n_det == n_cond,
    }

    # ── Entity presence (no matching, just check existence) ──
    name_to_entity = {}
    entity_found = 0
    entity_total = n_cond
    entity_details = []

    for ci, ent in enumerate(cond_entities):
        name = ent.get("name", f"ent_{ci}")
        attrs = ent.get("attrs", {})
        name_to_entity[name] = ent
        matching_dets = _find_matching_detections(attrs, detected_objs)
        found = len(matching_dets) > 0
        if found:
            entity_found += 1
        entity_details.append({
            "name": name,
            "attrs": {k: v for k, v in attrs.items() if k in ATTR_NAMES},
            "found": found,
            "n_matches": len(matching_dets),
        })

    result["entity_found"] = entity_found
    result["entity_total"] = entity_total
    result["entity_details"] = entity_details

    # ── Relation accuracy ──
    rel_correct = 0
    rel_total = len(cond_relations)
    rel_details = []

    for rel in cond_relations:
        subj_name = rel.get("subj", "")
        obj_name = rel.get("obj", "")
        rel_type = rel.get("rel", "")

        subj_ent = name_to_entity.get(subj_name)
        obj_ent = name_to_entity.get(obj_name)

        if subj_ent is None or obj_ent is None:
            rel_details.append({
                "subj": subj_name, "rel": rel_type, "obj": obj_name,
                "correct": False, "reason": "entity_not_in_condition",
            })
            continue

        # Find all detected objects matching subj / obj features
        subj_candidates = _find_matching_detections(
            subj_ent.get("attrs", {}), detected_objs)
        obj_candidates = _find_matching_detections(
            obj_ent.get("attrs", {}), detected_objs)

        if not subj_candidates or not obj_candidates:
            rel_details.append({
                "subj": subj_name, "rel": rel_type, "obj": obj_name,
                "correct": False, "reason": "object_not_found",
            })
            continue

        # Check if ANY (subj_det, obj_det) pair satisfies the relation
        ok = False
        for si in subj_candidates:
            for oi in obj_candidates:
                if si == oi:
                    continue  # same object can't be both subj and obj
                if check_relation(detected_objs[si]["center"],
                                  detected_objs[oi]["center"],
                                  rel_type, margin=relation_margin):
                    ok = True
                    break
            if ok:
                break

        if ok:
            rel_correct += 1
        rel_details.append({
            "subj": subj_name, "rel": rel_type, "obj": obj_name,
            "correct": ok,
        })

    result["rel_correct"] = rel_correct
    result["rel_total"] = rel_total
    result["rel_details"] = rel_details

    return result


# ── Batch evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def eval_clevr_conditions(
    images: torch.Tensor,
    conditions: List[Dict],
    detector: torch.nn.Module,
    classifier: torch.nn.Module,
    det_threshold: float = 0.3,
    relation_margin: float = RELATION_MARGIN,
) -> Dict:
    """Evaluate a batch of generated images against their conditions.

    Returns aggregated metrics dict with 3 headline scores:
        1. count_accuracy
        2. entity_presence_accuracy
        3. rel_accuracy
    """
    B = images.shape[0]
    assert len(conditions) == B

    all_detected = detect_and_classify(images, detector, classifier,
                                       det_threshold=det_threshold)

    per_sample = []
    count_correct_total = 0
    entity_found_agg = 0
    entity_total_agg = 0
    rel_correct_agg = 0
    rel_total_agg = 0

    for b in range(B):
        r = evaluate_condition_alignment(
            all_detected[b], conditions[b],
            relation_margin=relation_margin)
        per_sample.append(r)

        if r["count_correct"]:
            count_correct_total += 1

        entity_found_agg += r["entity_found"]
        entity_total_agg += r["entity_total"]
        rel_correct_agg += r["rel_correct"]
        rel_total_agg += r["rel_total"]

    return {
        "n_samples": B,
        # ── 3 headline scores ──
        "count_accuracy": count_correct_total / B * 100 if B > 0 else 0.0,
        "entity_presence_accuracy": (entity_found_agg / entity_total_agg * 100
                                     if entity_total_agg > 0 else 0.0),
        "rel_accuracy": (rel_correct_agg / rel_total_agg * 100
                         if rel_total_agg > 0 else 0.0),
        # ── raw counts ──
        "count_correct": count_correct_total,
        "entity_found": entity_found_agg,
        "entity_total": entity_total_agg,
        "rel_correct": rel_correct_agg,
        "rel_total": rel_total_agg,
        "per_sample": per_sample,
    }


def format_eval_results(results: Dict) -> str:
    """Format eval results as a readable string."""
    lines = []
    lines.append(f"  Samples: {results['n_samples']}")
    lines.append(f"  [1] Count acc:           {results['count_accuracy']:.1f}% "
                 f"({results['count_correct']}/{results['n_samples']})")
    lines.append(f"  [2] Entity presence acc: {results['entity_presence_accuracy']:.1f}% "
                 f"({results['entity_found']}/{results['entity_total']})")
    lines.append(f"  [3] Relation acc:        {results['rel_accuracy']:.1f}% "
                 f"({results['rel_correct']}/{results['rel_total']})")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
#  Complex-text evaluation (coverage-complete prompt families)
#
#  Works on the styled-caption records produced by
#  `clevr-dataset-gen/generate_styled_captions.py`. Each caption record
#  carries an `exposed` mask describing exactly which GT facts the caption
#  reveals; this evaluator scores ONLY those exposed axes, so the headline
#  metric for any family is unbiased by parts of the scene the caption never
#  promised to convey.
#
#  Families and the axes they expose:
#    C        : total count
#    E        : entity inventory (schema-projected), implicit total count
#    R        : relations + optional entity fallback groups
#    C+E      : total count + partial entity inventory
#    C+R      : total count + relations
#    E+R      : entity inventory (remaining objects) + relations
# ─────────────────────────────────────────────────────────────────────────

from collections import Counter as _Counter


def _find_detections_by_schema(target_attrs: Dict, schema: List[str],
                               detected_objs: List[Dict]) -> List[int]:
    """Detections whose `schema` attrs equal `target_attrs[schema]`.

    Unlike `_find_matching_detections`, this only compares attrs explicitly
    listed in `schema` — necessary when the caption exposed a strict subset
    of attributes (e.g. `{color, shape}`) and we must allow detections with
    any `size` / `material`.
    """
    if not schema:
        return list(range(len(detected_objs)))
    out = []
    for di, det in enumerate(detected_objs):
        if all(det.get(a) == target_attrs.get(a) for a in schema):
            out.append(di)
    return out


def _score_entity_groups(detected_objs: List[Dict],
                         entity_groups: List[Dict],
                         schema: List[str]) -> Tuple[int, int, List[Dict]]:
    """Multiset score over schema-projected (signature, count) groups.

    Returns:
        groups_found_full: how many groups had count_required ≤ count_in_image
        groups_total: len(entity_groups)
        per_group_details: list of {signature, required, observed, found}
    """
    # detections multiset by schema-projected signature
    det_counter = _Counter(
        tuple(det.get(a) for a in schema) for det in detected_objs
    )
    found = 0
    details = []
    for g in entity_groups:
        sig_tuple = tuple(g["signature"].get(a) for a in schema)
        required = g["count"]
        observed = det_counter.get(sig_tuple, 0)
        ok = observed >= required
        if ok:
            found += 1
        details.append({
            "signature": g["signature"],
            "required": required,
            "observed": observed,
            "found": ok,
        })
    return found, len(entity_groups), details


def _score_relations(detected_objs: List[Dict],
                     exposed_relations: List[Dict],
                     relation_margin: float = RELATION_MARGIN) -> Tuple[int, int, List[Dict]]:
    """Relations whose subj/obj candidates exist and satisfy the spatial check.

    `exposed_relations` entries follow generate_styled_captions.py:
        {"subj": <gt_name>, "rel": <rel_key>,
         "obj":  <gt_name>, "subj_ref": [attrs...], "obj_ref": [attrs...]}

    The subj/obj reference attributes (a subset of {size,color,material,shape})
    determine which detections are valid candidates. Pairs satisfying the
    spatial relation are counted (any-pair, like the legacy evaluator).
    """
    correct = 0
    details = []
    for rel in exposed_relations:
        subj_ref = rel.get("subj_ref", ["color", "shape"])
        obj_ref = rel.get("obj_ref", ["color", "shape"])
        subj_attrs = rel.get("subj_attrs") or rel.get("subj") or {}
        obj_attrs = rel.get("obj_attrs") or rel.get("obj") or {}
        # If only GT names are stored, look up via gt_entities (caller-injected).
        subj_candidates = _find_detections_by_schema(subj_attrs, subj_ref, detected_objs)
        obj_candidates = _find_detections_by_schema(obj_attrs, obj_ref, detected_objs)
        if not subj_candidates or not obj_candidates:
            details.append({**rel, "correct": False, "reason": "object_not_found"})
            continue
        ok = False
        for si in subj_candidates:
            for oi in obj_candidates:
                if si == oi:
                    continue
                if check_relation(detected_objs[si]["center"],
                                  detected_objs[oi]["center"],
                                  rel["rel"], margin=relation_margin):
                    ok = True
                    break
            if ok:
                break
        if ok:
            correct += 1
        details.append({**rel, "correct": ok})
    return correct, len(exposed_relations), details


def _resolve_relation_attrs(exposed_relations: List[Dict],
                            gt_entities: List[Dict]) -> List[Dict]:
    """Inline subj/obj GT attrs into each relation record.

    The generator stores `subj`/`obj` as GT entity names (e.g. "A"). For
    scoring we need the actual attribute values; this helper looks them up
    and writes them into `subj_attrs` / `obj_attrs`.
    """
    by_name = {e["name"]: e["attrs"] for e in gt_entities}
    out = []
    for rel in exposed_relations:
        rel = dict(rel)
        if isinstance(rel.get("subj"), str):
            rel["subj_attrs"] = by_name.get(rel["subj"], {})
        if isinstance(rel.get("obj"), str):
            rel["obj_attrs"] = by_name.get(rel["obj"], {})
        out.append(rel)
    return out


def evaluate_complex_text_alignment(
    detected_objs: List[Dict],
    caption_record: Dict,
    relation_margin: float = RELATION_MARGIN,
) -> Dict:
    """Score one image against one styled-caption record (exposed-aware).

    Args:
        detected_objs: detector+classifier output for the generated image.
        caption_record: dict with keys {family, variant, text, exposed,
                                        gt(optional)}. `gt` provides entity
                        attrs needed to resolve relation references.

    Returns: dict with `family` and only the metric fields applicable to
    that family (others omitted):
        count_correct, count_pred, count_gt
        entity_groups_found, entity_groups_total, entity_details
        rel_correct, rel_total, rel_details
    """
    exposed = caption_record.get("exposed", {})
    family = caption_record.get("family", "?")
    gt_entities = caption_record.get("gt", {}).get("entities", [])

    out: Dict = {"family": family}
    n_det = len(detected_objs)

    # ── count ──
    count_info = exposed.get("count") or {}
    ctype = count_info.get("type")
    if ctype in ("total", "implicit_total"):
        gt_count = int(count_info.get("value", 0))
        out["count_pred"] = n_det
        out["count_gt"] = gt_count
        out["count_correct"] = int(n_det == gt_count)

    # ── entity inventory ──
    schema = exposed.get("entity_schema")
    groups = exposed.get("entity_groups") or []
    if schema and groups:
        f, t, det = _score_entity_groups(detected_objs, groups, schema)
        out["entity_groups_found"] = f
        out["entity_groups_total"] = t
        out["entity_details"] = det

    # ── relations ──
    rels = exposed.get("relations") or []
    if rels:
        rels_resolved = _resolve_relation_attrs(rels, gt_entities)
        c, t, det = _score_relations(detected_objs, rels_resolved,
                                     relation_margin=relation_margin)
        out["rel_correct"] = c
        out["rel_total"] = t
        out["rel_details"] = det

    return out


@torch.no_grad()
def eval_clevr_complex_text(
    images: torch.Tensor,
    caption_records: List[Dict],
    detector: torch.nn.Module,
    classifier: torch.nn.Module,
    det_threshold: float = 0.3,
    relation_margin: float = RELATION_MARGIN,
) -> Dict:
    """Batch evaluation against styled-caption records.

    Args:
        images: (B, 3, H, W) tensor in [0, 1].
        caption_records: list of B caption-record dicts (see
            generate_styled_captions.py output). Each must carry `family`,
            `exposed`, and (for R/C+R/E+R) `gt.entities`.

    Returns:
        {
            "n_samples": B,
            "per_sample": [...],  # per-image dicts
            # aggregates keyed by family ("C","E","R","C+E","C+R","E+R","overall"):
            "by_family": {
               family: {
                  "n_samples": int,
                  "count_accuracy", "count_correct",
                  "entity_inv_accuracy", "entity_groups_found", "entity_groups_total",
                  "rel_accuracy", "rel_correct", "rel_total",
               }
            }
        }
    """
    B = images.shape[0]
    assert len(caption_records) == B

    detected_all = detect_and_classify(images, detector, classifier,
                                       det_threshold=det_threshold)

    per_sample = []
    agg: Dict[str, Dict[str, int]] = {}

    def _bump(family: str, r: Dict) -> None:
        slot = agg.setdefault(family, {
            "n_samples": 0,
            "count_has": 0, "count_correct": 0,
            "entity_groups_found": 0, "entity_groups_total": 0,
            "rel_correct": 0, "rel_total": 0,
        })
        slot["n_samples"] += 1
        if "count_correct" in r:
            slot["count_has"] += 1
            slot["count_correct"] += r["count_correct"]
        if "entity_groups_total" in r:
            slot["entity_groups_found"] += r["entity_groups_found"]
            slot["entity_groups_total"] += r["entity_groups_total"]
        if "rel_total" in r:
            slot["rel_correct"] += r["rel_correct"]
            slot["rel_total"] += r["rel_total"]

    for b in range(B):
        r = evaluate_complex_text_alignment(
            detected_all[b], caption_records[b],
            relation_margin=relation_margin)
        per_sample.append(r)
        _bump(r["family"], r)
        _bump("overall", r)

    by_family = {}
    for fam, slot in agg.items():
        out = {"n_samples": slot["n_samples"]}
        if slot["count_has"]:
            out["count_accuracy"] = slot["count_correct"] / slot["count_has"] * 100
            out["count_correct"] = slot["count_correct"]
            out["count_n"] = slot["count_has"]
        if slot["entity_groups_total"]:
            out["entity_inv_accuracy"] = (slot["entity_groups_found"]
                                          / slot["entity_groups_total"] * 100)
            out["entity_groups_found"] = slot["entity_groups_found"]
            out["entity_groups_total"] = slot["entity_groups_total"]
        if slot["rel_total"]:
            out["rel_accuracy"] = slot["rel_correct"] / slot["rel_total"] * 100
            out["rel_correct"] = slot["rel_correct"]
            out["rel_total"] = slot["rel_total"]
        by_family[fam] = out

    return {
        "n_samples": B,
        "per_sample": per_sample,
        "by_family": by_family,
    }


def format_complex_text_results(results: Dict) -> str:
    """Pretty-print eval_clevr_complex_text output."""
    lines = [f"  Samples: {results['n_samples']}"]
    fams = ["C", "E", "R", "C+E", "C+R", "E+R", "overall"]
    for fam in fams:
        if fam not in results["by_family"]:
            continue
        d = results["by_family"][fam]
        bits = [f"n={d['n_samples']}"]
        if "count_accuracy" in d:
            bits.append(f"count={d['count_accuracy']:.1f}%"
                        f" ({d['count_correct']}/{d['count_n']})")
        if "entity_inv_accuracy" in d:
            bits.append(f"entity_inv={d['entity_inv_accuracy']:.1f}%"
                        f" ({d['entity_groups_found']}/{d['entity_groups_total']})")
        if "rel_accuracy" in d:
            bits.append(f"rel={d['rel_accuracy']:.1f}%"
                        f" ({d['rel_correct']}/{d['rel_total']})")
        lines.append(f"    [{fam:7}] {'  '.join(bits)}")
    return "\n".join(lines)
