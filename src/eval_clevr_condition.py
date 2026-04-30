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
