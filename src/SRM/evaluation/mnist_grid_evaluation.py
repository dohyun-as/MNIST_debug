from dataclasses import dataclass
from math import prod
from typing import Literal

from jaxtyping import Bool, Integer, Shaped
import torch
from torch import Tensor

from ..dataset import DatasetMnistSudoku3x3Eager

from .mnist_evaluation import MnistEvaluation, MnistEvaluationCfg


@dataclass
class MnistGridEvaluationCfg(MnistEvaluationCfg):
    name: Literal["mnist_grid"] = "mnist_grid"


class MnistGridEvaluation(MnistEvaluation[MnistGridEvaluationCfg]):
    def __init__(
        self, 
        cfg: MnistEvaluationCfg, 
        tag: str, 
        dataset: DatasetMnistSudoku3x3Eager,
        patch_size: int | None = None,
        patch_grid_shape: tuple[int, int] | None = None,
        deterministic: bool = False
    ) -> None:
        super().__init__(cfg, tag, dataset, patch_size, patch_grid_shape, deterministic)
        self.num_cells = prod(self.grid_size)

    def classify(
        self,
        pred: Integer[Tensor, "batch grid_height grid_width"]
    ) -> tuple[
        Bool[Tensor, "batch"],
        dict[str, Shaped[Tensor, "batch"]]
    ]:
        # -1 because we want exactly one in the end for every cnt
        cnt = torch.full(
            (pred.shape[0], self.num_cells), fill_value=-1, 
            dtype=pred.dtype, device=pred.device
        )
        pred = pred.flatten(-2) - 1
        cnt.scatter_add_(
            dim=1, 
            index=pred, 
            src=torch.ones((1,), dtype=pred.dtype, device=pred.device).expand_as(pred)
        )
        cnt.abs_()
        dist = cnt.sum(dim=1)
        label = dist == 0
        return label, {"distance": dist}
