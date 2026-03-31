from abc import ABC
from dataclasses import dataclass
from glob import glob
from typing import TypeVar

from PIL import Image

from .dataset_grid import DatasetGrid, DatasetGridCfg
from ..type_extensions import ConditioningCfg, Stage


@dataclass
class DatasetGridEagerCfg(DatasetGridCfg):
    extension: str = "jpeg"


T = TypeVar("T", bound=DatasetGridEagerCfg)


class DatasetGridEager(DatasetGrid[T], ABC):
    @property
    def split(self) -> str:
        return "train" if self.stage == "train" else "test"

    def __init__(self, cfg: T, conditioning_cfg: ConditioningCfg, patch_size: int | None, stage: Stage) -> None:
        super().__init__(cfg, conditioning_cfg, patch_size, stage)
        self.image_paths = list(
            glob(str(self.cfg.root / self.split / f"*.{self.cfg.extension}"))
        )
        self.image_shape = Image.open(self.image_paths[0]).size[::-1]
        assert all(ds % s == 0 for ds, s in zip(self.image_shape, self.crop_size))
        self.num_crops_per_axis = tuple(
            ds // s for ds, s in zip(self.image_shape, self.crop_size)
        )

    def load_full_image(self, idx: int) -> Image.Image:
        image_path = self.image_paths[idx]
        return Image.open(image_path).convert("L" if self.cfg.grayscale else "RGB")

    @property
    def _num_available(self) -> int:
        return len(self.image_paths) * self.num_crops_per_image
