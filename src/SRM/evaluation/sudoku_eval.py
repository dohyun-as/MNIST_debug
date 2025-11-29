# sudoku_image_eval.py
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def _ensure_bchw(img: Tensor) -> Tensor:
    """Accept [H,W], [1,H,W], or [B,1,H,W] → return [B,1,H,W]."""
    if img.ndim == 2:        # [H,W]
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:      # [C,H,W]
        if img.shape[0] != 1:
            raise ValueError("Expected 1 channel image; got C={}".format(img.shape[0]))
        img = img.unsqueeze(0)
    elif img.ndim == 4:      # [B,C,H,W]
        if img.shape[1] != 1:
            raise ValueError("Expected 1 channel image; got C={}".format(img.shape[1]))
    else:
        raise ValueError(f"Unsupported image shape: {tuple(img.shape)}")
    return img


def _to_neg1_pos1(img: Tensor) -> Tensor:
    """Scale image to [-1,1] if input seems in [0,1]. Leave as-is if already in [-1,1]."""
    # heuristic
    mn, mx = float(img.min()), float(img.max())
    if -0.1 <= mn and mx <= 1.1:
        # likely [0,1]
        return img * 2.0 - 1.0
    return img


def _center_crop_to_multiple(img: Tensor, grid: Tuple[int, int]) -> Tensor:
    """
    Center-crop H,W to be divisible by grid_h, grid_w.
    Assumes BCHW. Keeps as much as possible.
    """
    B, C, H, W = img.shape
    gh, gw = grid
    new_h = (H // gh) * gh
    new_w = (W // gw) * gw
    if new_h == H and new_w == W:
        return img
    if new_h == 0 or new_w == 0:
        raise ValueError(f"Image too small for grid {grid}: H={H}, W={W}")
    top = (H - new_h) // 2
    left = (W - new_w) // 2
    return img[:, :, top:top+new_h, left:left+new_w]


def _tile_digits(
    img: Tensor, grid: Tuple[int, int]
) -> Tensor:
    """
    Split [B,1,H,W] into tiles → [B*G, 1, th, tw], where G = gh*gw.
    """
    B, C, H, W = img.shape
    gh, gw = grid
    th, tw = H // gh, W // gw
    if th * gh != H or tw * gw != W:
        raise ValueError(f"Image size must be divisible by grid {grid}, got {(H,W)}.")
    tiles = img.unfold(2, th, th).unfold(3, tw, tw)   # [B,1,gh,gw,th,tw]
    tiles = tiles.permute(0, 2, 3, 1, 4, 5).contiguous().view(B*gh*gw, 1, th, tw)
    return tiles


def _digits_from_classifier(
    tiles: Tensor, classifier: torch.nn.Module
) -> Tensor:
    """
    Run MNIST classifier on tiles → digits in {1..9}.
    - If classifier outputs '0' class, replace with 2nd-best (사용자 코드 관례 유지).
    Returns shape [B, gh, gw] with values in 1..9.
    """
    logits = classifier.forward(tiles)            # [B*G, 10]
    idx_top2 = torch.topk(logits, k=2, dim=1).indices  # [..., 2]
    pred = idx_top2[:, 0]
    zero_mask = pred == 0
    pred = pred.clone()
    pred[zero_mask] = idx_top2[zero_mask, 1]
    # reshape back
    G = tiles.shape[0]
    B = getattr(classifier, "_batch_override", None)  # not used; we infer from call site
    # we'll infer B from context at caller; easier: we pass expected grid into reshape outside
    return pred


def _reshape_digits(pred_flat: Tensor, batch_size: int, grid: Tuple[int, int]) -> Tensor:
    gh, gw = grid
    return pred_flat.view(batch_size, gh, gw).to(torch.int64)


def _sudoku_validity_and_distance(digits_11: Tensor) -> Dict[str, Tensor]:
    """
    digits_11: [B, 9, 9] in {1..9}
    Returns:
      valid: [B] bool,
      distance: [B] float (제약 위반량),
      detail: row/col/subgrid distance(optional)
    """
    x = digits_11.to(torch.int64) - 1  # -> {0..8}
    B, H, W = x.shape
    if H != 9 or W != 9:
        raise ValueError(f"Expected grid 9x9, got {(H,W)}")

    # one-hot over digits: [B, 9, 9, 9] (row, col, digit)
    oh = F.one_hot(x, num_classes=9).to(torch.float32)

    # rows: sum over columns -> counts per (row,digit)
    row_counts = oh.sum(dim=2)        # [B, 9, 9]
    # cols: sum over rows
    col_counts = oh.sum(dim=1)        # [B, 9, 9]
    # subgrids: reshape and sum inside 3x3 blocks
    sub = oh.view(B, 3, 3, 3, 3, 9).sum(dim=(2, 4))  # [B, 3, 3, 9] blocks × digits
    sub_counts = sub.view(B, 9, 9)   # 9 blocks flattened

    target = 1.0
    row_dist = (row_counts - target).abs().sum(dim=(1, 2))
    col_dist = (col_counts - target).abs().sum(dim=(1, 2))
    sub_dist = (sub_counts - target).abs().sum(dim=(1, 2))
    distance = row_dist + col_dist + sub_dist
    valid = distance.eq(0)

    return {
        "valid": valid,                 # [B] bool
        "distance": distance,           # [B] float
        "row_distance": row_dist,       # optional detail
        "col_distance": col_dist,
        "subgrid_distance": sub_dist,
    }


@dataclass
class SudokuImageEvaluator:
    """
    Standalone evaluator: image -> metrics (no Lightning/DataModule).
    - Assumes MNIST-style digits in a 9x9 Sudoku grid.
    - Expects a classifier that outputs logits for digits 0..9 (0 will be mapped to 2nd top).
    """
    classifier: torch.nn.Module
    grid: Tuple[int, int] = (9, 9)
    expect_range: str = "auto"  # "auto" | "[0,1]" | "[-1,1]"

    def __call__(self, image: Tensor) -> Dict[str, Any]:
        """
        image: Tensor [H,W] | [1,H,W] | [B,1,H,W]; dtype float
        returns:
          {
            "digits": LongTensor [B,9,9] in {1..9},
            "valid":  BoolTensor [B],
            "distance": FloatTensor [B],
            # details...
          }
        """
        x = _ensure_bchw(image).to(next(self.classifier.parameters()).device)

        if self.expect_range == "auto":
            x = _to_neg1_pos1(x)
        elif self.expect_range == "[0,1]":
            x = x * 2.0 - 1.0
        # elif "[-1,1]": leave as-is

        x = _center_crop_to_multiple(x, self.grid)
        tiles = _tile_digits(x, self.grid)                       # [B*81,1,th,tw]

        # Run classifier
        logits = self.classifier.forward(tiles)                  # [B*81,10]
        top2 = torch.topk(logits, k=2, dim=1).indices
        pred = top2[:, 0]
        zero_mask = pred.eq(0)
        pred = pred.clone()
        pred[zero_mask] = top2[zero_mask, 1]

        B = x.shape[0]
        digits = _reshape_digits(pred, B, self.grid) + 0  # {0..9}, currently 0..9 with 0 avoided
        # Guarantee 1..9
        digits[digits.eq(0)] = 1
        metrics = _sudoku_validity_and_distance(digits)

        out: Dict[str, Any] = {
            "digits": digits,                      # [B,9,9] in 1..9
            "valid": metrics["valid"],             # [B] bool
            "distance": metrics["distance"],       # [B] float
            "row_distance": metrics["row_distance"],
            "col_distance": metrics["col_distance"],
            "subgrid_distance": metrics["subgrid_distance"],
        }
        return out


# --- convenience function (functional style) ---

def evaluate_sudoku_image(
    image: Tensor,
    classifier: torch.nn.Module,
    grid: Tuple[int, int] = (9, 9),
    expect_range: str = "auto",
) -> Dict[str, Any]:
    return SudokuImageEvaluator(classifier, grid, expect_range)(image)
