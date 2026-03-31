from dataclasses import dataclass
from typing import Literal, Sequence

from jaxtyping import Float
import torch
from torch import Tensor

from .dataset_grid_eager import DatasetGridEager, DatasetGridEagerCfg


@dataclass
class DatasetMnistSudoku9x9EagerCfg(DatasetGridEagerCfg):
    name: Literal["mnist_sudoku"] = "mnist_sudoku"
    extension: str = "jpeg"
    grayscale: bool = True


class DatasetMnistSudoku9x9Eager(DatasetGridEager[DatasetMnistSudoku9x9EagerCfg]):
    cell_size = (28, 28)
    grid_size = (9, 9)
    
    def get_dependency_matrix(
        self,
        grid_shape: tuple[int, int]
    ) -> Float[Tensor, "num_patches num_patches"] | None:
        assert tuple(grid_shape) == (9, 9), "Only 9x9 patch grid supported"
        dependency_matrix = torch.zeros(81, 81, dtype=torch.bool)
        
        for i in range(81):
            r, c = self._from_full_idx(i)
            for j in range(81):
                r_, c_ = self._from_full_idx(j)
                if r == r_: #same row
                    dependency_matrix[i, j] = True

                if c == c_: #same column
                    dependency_matrix[i, j] = True

                if r // 3 == r_ // 3 and c // 3 == c_ // 3: #same subgrid
                    dependency_matrix[i, j] = True

        if self.cfg.mask_self_dependency:
            dependency_matrix = dependency_matrix.logical_xor(torch.eye(81))

        return dependency_matrix.float()
