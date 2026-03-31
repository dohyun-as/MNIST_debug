from dataclasses import dataclass
from math import prod
from typing import Literal

from jaxtyping import Float
import torch
from torch import Tensor

from .dataset_grid_eager import DatasetGridEager, DatasetGridEagerCfg


@dataclass
class DatasetMnistSudoku3x3EagerCfg(DatasetGridEagerCfg):
    name: Literal["mnist_grid"] = "mnist_grid"
    extension: str = "jpeg"
    grayscale: bool = True


class DatasetMnistSudoku3x3Eager(DatasetGridEager[DatasetMnistSudoku3x3EagerCfg]):
    cell_size = (28, 28)
    grid_size = (3, 3)

    def get_dependency_matrix(
        self, 
        grid_shape: tuple[int, int]
    ) -> Float[Tensor, "num_patches num_patches"] | None:
        """Returns the dependency matrix for the flattened grid structure  -- all connected by default"""
        assert tuple(grid_shape) == (3, 3), "Only 3x3 patch grid supported"

        dependency_matrix = torch.ones(prod(self.grid_size), prod(self.grid_size))
        if self.cfg.mask_self_dependency:
            dependency_matrix.fill_diagonal_(0)
        return dependency_matrix
