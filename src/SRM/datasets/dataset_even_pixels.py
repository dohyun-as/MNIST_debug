from dataclasses import dataclass

from jaxtyping import Bool
import numpy as np
from PIL import Image
from typing import Literal, TypeVar

from .dataset import Dataset, DatasetCfg
from .types import Example


@dataclass
class DatasetEvenPixelsCfg(DatasetCfg):
    name: Literal["even_pixels"] = "even_pixels"
    saturation: float = 1.0
    value: float = 0.7
    dataset_size: int = 1000_000
    root: None = None # We don't need a root for this dataset


T = TypeVar("T", bound=DatasetEvenPixelsCfg)


class DatasetEvenPixels(Dataset[T]):

    @staticmethod
    def _get_even_binary_mask(
        w: int, h: int, rng: np.random.Generator | None
    ) -> Bool[np.ndarray, "w h"]:
        num_ones = int(w * h / 2)
        flat_mask = np.zeros(w * h)
        flat_mask[:num_ones] = 1

        if rng is not None:
            rng.shuffle(flat_mask)
        else:
            np.random.shuffle(flat_mask)

        return flat_mask.astype(bool).reshape(w, h)

    def _get_image(self, rng: np.random.Generator | None) -> Image.Image:
        w, h = self.cfg.image_shape

        if rng is not None:
            hue_offset = rng.uniform(0, 0.5)

        else:
            hue_offset = np.random.uniform(0, 0.5)

        data = np.zeros((w, h, 3))
        data[:, :, 0] = (self._get_even_binary_mask(w, h, rng) * 0.5) + hue_offset
        data[:, :, 1] = self.cfg.saturation
        data[:, :, 2] = self.cfg.value

        return Image.fromarray(np.uint8(data * 255), mode="HSV").convert("RGB")

    def load(self, idx: int) -> Example:
        if self.stage == "train":
            rng = None
        else:
            rng = np.random.default_rng(idx)

        image = self._get_image(rng)

        return {"image": image}

    @property
    def _num_available(self) -> int:
        return self.cfg.dataset_size
