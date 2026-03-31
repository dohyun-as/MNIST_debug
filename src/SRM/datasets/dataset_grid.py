from abc import ABC, abstractmethod
from dataclasses import dataclass
from glob import glob
from math import prod
from typing import TypeVar

from jaxtyping import Float
import numpy as np
from PIL import Image

import torchvision.transforms.functional as F

from .dataset import Dataset, DatasetCfg
from .types import Example
from ..type_extensions import ConditioningCfg, Stage


@dataclass
class DatasetGridCfg(DatasetCfg):
    mask_self_dependency: bool = True


T = TypeVar("T", bound=DatasetGridCfg)


class DatasetGrid(Dataset[T], ABC):

    @abstractmethod
    def load_full_image(self, idx: int) -> Image.Image:
        pass

    @property
    def num_crops_per_image(self) -> int:
        return prod(self.num_crops_per_axis)

    @property
    @abstractmethod
    def _num_available(self) -> int:
        pass

    @property
    @abstractmethod
    def cell_size(self) -> tuple[int, int]:
        pass

    @property
    @abstractmethod
    def grid_size(self) -> tuple[int, int]:
        pass

    @property
    def crop_size(self) -> tuple[int, int]:
        return tuple(map(prod, zip(self.grid_size, self.cell_size)))

    @classmethod
    def _from_full_idx(cls, full_idx) -> tuple[int, int]:
        return full_idx // cls.grid_size[0], full_idx % cls.grid_size[1]        

    def get_random_masks(
        self, num_given_cells: int | None = None, rng: np.random.Generator | None = None,
        return_cell_mask: bool = False, 
    ) -> Float[np.ndarray, "height width"]:
        num_cells = prod(self.grid_size)
        if num_given_cells is None:
            num_given_cells = (
                np.random.randint(0, num_cells)
                if rng is None
                else rng.integers(0, num_cells).item()
            )
        cell_mask = np.ones(self.grid_size, dtype=bool)

        grid_idx = (np.random if rng is None else rng).choice(
            num_cells, num_given_cells, replace=False
        )
        row, col = grid_idx // self.grid_size[1], grid_idx % self.grid_size[1]
        cell_mask[row, col] = False
        pixel_mask = np.kron(cell_mask, np.ones(self.cell_size, dtype=bool)).astype(float)
        
        if return_cell_mask:
            return pixel_mask, cell_mask.astype(float)   # (H,W), (9,9)
        return pixel_mask

    def get_mask(
        self,
        image_idx: int,
        crop_params: tuple[int, int, int, int],
        num_given_cells: int | None = None,
        return_cell_mask: bool = False,
    ) -> Image.Image | Float[np.ndarray, "height width"]:
        return self.get_random_masks(
            num_given_cells,
            np.random.default_rng(image_idx) if self.stage == "test" else None,
            return_cell_mask=return_cell_mask,
        )

    def get_crop_params(self, crop_idx: int):
        row_idx, col_idx = (
            crop_idx // self.num_crops_per_axis[1],
            crop_idx % self.num_crops_per_axis[1],
        )
        top, left = row_idx * self.crop_size[0], col_idx * self.crop_size[1]
        bottom, right = top + self.crop_size[0], left + self.crop_size[1]
        return (left, top, right, bottom)

    def load(self, idx: int, num_given_cells: int | None = None) -> Example:
        image_idx, crop_idx = (
            idx // self.num_crops_per_image,
            idx % self.num_crops_per_image,
        )
        if hasattr(self, "load_full_image_and_meta"):
            image, meta = getattr(self, "load_full_image_and_meta")(image_idx)
        else:
            image = self.load_full_image(image_idx)
            meta = {}
        crop_params = self.get_crop_params(crop_idx)
        image = image.crop(crop_params)
        cell_mask = None
        if self.conditioning_cfg.mask or num_given_cells is not None:
            got = self.get_mask(image_idx, crop_params, num_given_cells,
            return_cell_mask=True,)
            if isinstance(got, tuple):
                pixel_mask, cell_mask = got
            else:
                pixel_mask, cell_mask = got, None
            image = self.concat_mask(image, pixel_mask)

        out = {"image": image}
        if "grid" in meta:
            out["grid"] = meta["grid"]
        if cell_mask is not None:
            # bool로 주는 걸 추천 (필요하면 float32로 바꿔도 됨)
            out["cell_mask"] = cell_mask
        return out
