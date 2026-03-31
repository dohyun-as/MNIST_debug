from abc import ABC
from dataclasses import dataclass
from typing import TypeVar
from pathlib import PosixPath

import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms.functional as F

from .dataset_grid import DatasetGrid, DatasetGridCfg
from ..type_extensions import ConditioningCfg, Stage


@dataclass
class DatasetLazyGridCfg(DatasetGridCfg):
    top_n: int = 100
    test_samples_num: int = 10000


T = TypeVar("T", bound=DatasetLazyGridCfg)


class DatasetMnistGridLazy(DatasetGrid[T], ABC):
    def __init__(
        self, 
        cfg: T, 
        conditioning_cfg: ConditioningCfg,
        stage: Stage
    ) -> None:
        super().__init__(cfg, conditioning_cfg, stage)
        self.mnist_images = self.get_chosen_mnist_images()
        self.sudoku_grids = self.get_raw_sudoku_grids()

        images_shape = self.mnist_images.shape[-2:]
        self.image_shape = (
            images_shape[0] * self.grid_size[0],
            images_shape[1] * self.grid_size[1]
        )
        self.num_crops_per_axis = tuple(
            ds // s for ds, s in zip(self.image_shape, self.crop_size)
        )


    def _pixel_mask_from_reveal_cells(self) -> torch.Tensor:
        """
        1-based 좌표의 셀만 1로, 나머지는 0으로 만드는 픽셀 마스크 (H,W), dtype=float32
        """
        rc = getattr(self.cfg, "reveal_cells", None)
        if not rc:
            # None이거나 빈 시퀀스면 마스크 적용 안 함(전부 1)
            R, C = self.grid_size
            H, W = self.cell_size
            return torch.ones((R * H, C * W), dtype=torch.float32)

        R, C = self.grid_size
        H, W = self.cell_size

        cell_mask = torch.zeros((R, C), dtype=torch.float32)
        for (r1, c1) in rc:
            r0, c0 = r1 - 1, c1 - 1  # 1-based → 0-based
            if 0 <= r0 < R and 0 <= c0 < C:
                cell_mask[r0, c0] = 1.0

        # 각 셀(28×28)을 같은 값으로 확장
        pixel_mask = torch.kron(cell_mask, torch.ones((H, W), dtype=torch.float32))
        return pixel_mask

    @property
    def sudokus_file_path(self) -> PosixPath:
        return self.mnist_root_path / "sudokus.npy"

    @property
    def mnist_root_path(self) -> PosixPath:
        return self.cfg.root if isinstance(self.cfg.root, PosixPath) else PosixPath(self.cfg.root)

    @property
    def top_5000_indices_csv_path(self) -> PosixPath:
        return self.mnist_root_path / "top_5000_values.csv"

    @property
    def _num_available(self) -> int:
        return len(self.sudoku_grids) * self.num_crops_per_image

    def get_chosen_mnist_images(self) -> torch.Tensor:
        mnist_dataset = torchvision.datasets.MNIST(
            root=self.mnist_root_path, train=True, download=True
        )

        top_images_df = pd.read_csv(self.top_5000_indices_csv_path)

        labels = mnist_dataset.targets
        chosen_mnist_images = []
        for target_label in range(10):  # 0-9
            selected_indices_df = top_images_df[top_images_df.label == target_label]
            
            selected_indices_df.sort_values(by="confidence", ascending=False, inplace=True)
            selected_indices_df = selected_indices_df[: self.cfg.top_n]

            selected_indices = selected_indices_df["image_index"].values

            all_label_images = mnist_dataset.data[labels == target_label]
            selected_images = all_label_images[selected_indices]

            assert len(selected_images) == self.cfg.top_n
            chosen_mnist_images.append(selected_images)

        chosen_mnist_images = torch.stack(chosen_mnist_images, dim=0)

        assert chosen_mnist_images.shape == (10, self.cfg.top_n, 28, 28)
        return torch.Tensor(chosen_mnist_images)

    def get_raw_sudoku_grids(self) -> torch.Tensor:
        all_sudoku_grids = np.load(self.sudokus_file_path)

        if self.stage == "train":
            return torch.tensor(all_sudoku_grids[: -self.cfg.test_samples_num])

        return torch.tensor(all_sudoku_grids[-self.cfg.test_samples_num:])

    @property
    def is_deterministic(self) -> bool:
        return self.stage != "train"

    def load_full_image(self, idx: int) -> Image.Image:
        grid = self.sudoku_grids[idx]

        rng = None
        if self.is_deterministic:
            rng = np.random.default_rng(idx)

        full_image = torch.empty((252, 252), dtype=torch.uint8)

        for j in range(9):
            for k in range(9):
                # Get the corresponding MNIST number
                candidates = self.mnist_images[int(grid[j, k])]

                # Randomly select one of the MNIST numbers
                if rng is not None:
                    index = rng.integers(0, candidates.size(0))
                else:
                    index = np.random.randint(0, candidates.size(0))

                mnist_image = candidates[index]

                # Add the MNIST tensor to the grid of MNIST numbers
                full_image[j * 28 : (j + 1) * 28, k * 28 : (k + 1) * 28] = mnist_image

        # ▶ 여기서 이미지 마스킹 적용 (지정 칸 외 0으로)
        # print("reveal_cells", self.cfg.reveal_cells)
        if getattr(self.cfg, "reveal_cells", None) is not None:
            mask = self._pixel_mask_from_reveal_cells()  # (252,252), float32 in {0,1}
            full_image = (full_image.float() * mask).to(torch.uint8)
            
        return F.to_pil_image(full_image)

    def load_full_image_and_meta(self, idx: int):
        grid = self.sudoku_grids[idx]  # (9,9)

        rng = np.random.default_rng(idx) if self.is_deterministic else None

        H, W = self.cell_size          # (28, 28)
        R, C = self.grid_size          # (9, 9)
        full_image = torch.empty((R * H, C * W), dtype=torch.uint8)

        for r in range(R):
            for c in range(C):
                # Sudoku 값이 1..9라고 가정된 기존 로직 유지
                candidates = self.mnist_images[int(grid[r, c])]
                index = rng.integers(0, candidates.size(0)) if rng is not None else np.random.randint(0, candidates.size(0))
                mnist_image = candidates[index]
                y0, x0 = r * H, c * W
                y1, x1 = y0 + H, x0 + W
                full_image[y0:y1, x0:x1] = mnist_image
        
        # ▶ 여기서 이미지 마스킹 적용 (지정 칸 외 0으로)
        # print("reveal_cells", self.cfg.reveal_cells)
        if getattr(self.cfg, "reveal_cells", None) is not None:
            mask = self._pixel_mask_from_reveal_cells()  # (252,252), float32 in {0,1}
            full_image = (full_image.float() * mask).to(torch.uint8)
            

        image = F.to_pil_image(full_image)
        meta = {
            "grid": torch.as_tensor(grid, dtype=torch.long),  # (9,9) → 디퓨전 조건으로 넘길 값
        }
        return image, meta
