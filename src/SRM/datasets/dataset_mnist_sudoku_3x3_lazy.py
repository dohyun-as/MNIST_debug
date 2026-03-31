# dataset_mnist_grid_3x3_lazy.py (새 파일로 추가 권장)
from dataclasses import dataclass
from typing import Literal
from pathlib import PosixPath

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

from .dataset_mnist_grid_lazy import (
    DatasetLazyGridCfg,
    DatasetMnistGridLazy,
)
from ..type_extensions import ConditioningCfg, Stage


@dataclass
class DatasetMnist3x3GridLazyCfg(DatasetLazyGridCfg):
    # 식별용 이름(원하면 자유롭게 변경)
    name: Literal["mnist_grid3x3_lazy"] = "mnist_grid3x3_lazy"
    grayscale: bool = True
    # 필요시: top_n, test_samples_num 등은 부모에서 그대로 상속


class DatasetMnist3x3GridLazy(DatasetMnistGridLazy[DatasetMnist3x3GridLazyCfg]):
    # 28×28 MNIST 타일을 3×3으로 붙임
    cell_size = (28, 28)
    grid_size = (3, 3)

    # 3×3 조합 npy 경로 (N,3,3) 정수 배열(각 칸 0~9)
    @property
    def grids_3x3_file_path(self) -> PosixPath:
        return self.mnist_root_path / "mnist_3x3_grids.npy"

    # 부모 __init__가 호출하는 이름을 맞추기 위해 메서드명은 유지하되 내용을 교체
    def get_raw_sudoku_grids(self) -> torch.Tensor:
        all_grids = np.load(self.grids_3x3_file_path)  # shape: (N, 3, 3), dtype: int
        if self.stage == "train":
            return torch.tensor(all_grids[: -self.cfg.test_samples_num])
        return torch.tensor(all_grids[-self.cfg.test_samples_num:])

    # 3×3 그리드 크기에 맞춰 동적으로 합성
    def load_full_image(self, idx: int) -> Image.Image:
        grid = self.sudoku_grids[idx]  # (3, 3) 정수(0~9)
        rng = None
        if self.is_deterministic:
            rng = np.random.default_rng(idx)

        h_cell, w_cell = self.cell_size
        gh, gw = self.grid_size  # (3, 3)
        full_image = torch.empty((h_cell * gh, w_cell * gw), dtype=torch.uint8)

        for j in range(gh):
            for k in range(gw):
                digit = int(grid[j, k])
                candidates = self.mnist_images[digit]  # (top_n, 28, 28)
                # 무작위 후보 선택(테스트면 시드 고정)
                index = (rng.integers(0, candidates.size(0))
                         if rng is not None else np.random.randint(0, candidates.size(0)))
                mnist_patch = candidates[index]
                full_image[
                    j * h_cell: (j + 1) * h_cell,
                    k * w_cell: (k + 1) * w_cell
                ] = mnist_patch

        return F.to_pil_image(full_image)

    # 의존성 행렬이 필요 없으면 부모(Dataset) 기본 로컬리티 버전을 그대로 사용.
    # 행/열 의존성만 쓰고 싶다면 아래 주석 해제해 간단히 대체 가능.
    # def get_dependency_matrix(self, grid_shape: tuple[int, int]):
    #     dep = torch.zeros(9, 9, dtype=torch.bool)
    #     for i in range(9):
    #         r, c = self._from_full_idx(i)  # (0..2, 0..2)
    #         for j in range(9):
    #             r_, c_ = self._from_full_idx(j)
    #             if r == r_ or c == c_:
    #                 dep[i, j] = True
    #     if self.cfg.mask_self_dependency:
    #         eye = torch.eye(9, dtype=torch.bool, device=dep.device)
    #         dep = dep.logical_xor(eye)
    #     return dep.float()
