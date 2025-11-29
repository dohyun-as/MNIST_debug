from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod
from typing import Iterator, TypeVar, Sequence

from jaxtyping import Bool, Float, Integer, Shaped
import numpy as np
import torch
from torch import Tensor

from ..dataset import DatasetGrid
from ..global_cfg import get_mnist_classifier_path
from ..misc.mnist_classifier import get_classifier, MNISTClassifier
from ..model import Wrapper
from .types import EvaluationOutput
from .sampling_evaluation import (
    SamplingEvaluation, 
    SamplingEvaluationCfg, 
    UnbatchedSamplingExample,
    BatchedSamplingExample
)


@dataclass
class MnistEvaluationCfg(SamplingEvaluationCfg):
    num_samples: int = 100
    num_fill: int | Sequence[int] | None = None


T = TypeVar("T", bound=MnistEvaluationCfg)


class MnistEvaluation(SamplingEvaluation[T], ABC):
    def __init__(
        self,
        cfg: MnistEvaluationCfg,
        tag: str,
        dataset: DatasetGrid,
        patch_size: int | None = None,
        patch_grid_shape: tuple[int, int] | None = None,
        deterministic: bool = False
    ) -> None:
        super().__init__(cfg, tag, dataset, patch_size, patch_grid_shape, deterministic)
        self.grid_size = dataset.grid_size

    @torch.no_grad()
    def discretize(
        self,
        classifier: MNISTClassifier,
        samples: Float[Tensor, "batch 1 height width"]
    ) -> Integer[Tensor, "batch grid_height grid_width"]:
        batch_size = samples.shape[0]
        tile_shape = tuple(s // g for s, g in zip(samples.shape[-2:], self.grid_size))
        tiles = samples.unfold(2, tile_shape[0], tile_shape[0])\
            .unfold(3, tile_shape[1], tile_shape[1]).reshape(-1, 1, *tile_shape)
        logits: Float[Tensor, "batch 10"] = classifier.forward(tiles)
        idx = torch.topk(logits, k=2, dim=1).indices
        pred = idx[:, 0]
        # Replace zero predictions with second most probable number
        zero_mask = pred == 0
        pred[zero_mask] = idx[zero_mask, 1]
        pred = pred.reshape(batch_size, *self.grid_size)
        return pred
    
    @abstractmethod
    def classify(
        self,
        pred: Integer[Tensor, "batch grid_height grid_width"]
    ) -> tuple[
        Bool[Tensor, "batch"],
        dict[str, Shaped[Tensor, "batch"]]
    ]:
        pass

    def get_metrics(
        self,
        samples: Float[Tensor, "batch 1 height width"]
    ) -> dict[str, Float[Tensor, ""]]:
        classifier = get_classifier(get_mnist_classifier_path(), samples.device)
        discrete = self.discretize(classifier, samples)
        labels, metrics = self.classify(discrete)
        metrics = {k: v.float().mean() for k, v in metrics.items()}
        metrics["accuracy"] = labels.float().mean()
        return metrics

    @torch.no_grad()
    def evaluate(
        self, 
        model: Wrapper, 
        batch: BatchedSamplingExample,
        return_sample: bool = True
    ) -> Iterator[EvaluationOutput]:
        for res in super().evaluate(model, batch, return_sample):
            res["metrics"] = self.get_metrics(res["sample"]["sample"])
            yield res

    def __getitem__(self, idx: int) -> UnbatchedSamplingExample:
        if isinstance(self.cfg.num_fill, int):
            num_given_cells = self.cfg.num_fill
            
        else: # Sequence[int] | None
            if isinstance(self.cfg.num_fill, Sequence):
                interval = (self.cfg.num_fill[0], self.cfg.num_fill[1]+1) # +1 because randint is exclusive
                assert 0 <= interval[0] <= interval[1] <= prod(self.grid_size) + 1, f"Invalid interval for num_fill"
            
            elif self.cfg.num_fill is None: # Random number of cells
                interval = (0, prod(self.grid_size)+1) # +1 because randint is exclusive
                
            else:
                raise ValueError(f"Invalid value for num_fill: {self.cfg.num_fill}, expected int, Sequence[int] or None")
        
            num_given_cells = (
                np.random.default_rng(idx).integers(*interval).item()
                if self.deterministic else np.random.randint(*interval)
            ) 
        
        return super().__getitem__(idx, num_given_cells=num_given_cells)
