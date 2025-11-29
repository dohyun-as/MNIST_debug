from dataclasses import dataclass
from typing import Iterator, TypeVar, Literal


from jaxtyping import Float, Integer
import numpy as np
from matplotlib.colors import rgb_to_hsv
import torch
from torch import Tensor


from ..model import Wrapper
from .types import EvaluationOutput
from .sampling_evaluation import (
    SamplingEvaluation,
    SamplingEvaluationCfg,
    BatchedSamplingExample,
)


@dataclass
class EvenPixelsEvaluationCfg(SamplingEvaluationCfg):
    name: Literal["even_pixels"] = "even_pixels"
    num_samples: int = 100
    num_bins = 256


T = TypeVar("T", bound=EvenPixelsEvaluationCfg)


class EvenPixelsEvaluation(SamplingEvaluation[T]):

    def get_avg_saturation_std(
        self, images_hsv: Float[np.ndarray, "batch height width 3"]
    ) -> float:
        saturation = images_hsv[:, :, :, 1]
        return saturation.std(axis=(1, 2)).mean().item()

    def get_avg_value_std(
        self, images_hsv: Float[np.ndarray, "batch height width 3"]
    ) -> float:
        value = images_hsv[:, :, :, 2]
        return value.std(axis=(1, 2)).mean().item()

    def get_hue_uneven_error_counts(
        self, images_hsv: Float[np.ndarray, "batch height width 3"]
    ) -> Integer[np.ndarray, "batch"]:
        hue = images_hsv[:, :, :, 0]
        total_pixel_count = hue.shape[1] * hue.shape[2]
        uneven_counts = np.zeros(hue.shape[0], dtype=np.int32)

        for batch_idx, hue_img in enumerate(hue):
            histogram = np.histogram(hue_img, bins=self.cfg.num_bins, range=(0, 1))[0]

            peak_1 = histogram.argmax()
            peak_2 = peak_1 + self.cfg.num_bins // 2

            if peak_2 >= self.cfg.num_bins:
                peak_2 -= self.cfg.num_bins

            peaks = [peak_1, peak_2] if peak_1 < peak_2 else [peak_2, peak_1]
            border_1 = (peaks[0] + peaks[1]) // 2
            border_2 = (peaks[0] + peaks[1] + self.cfg.num_bins) // 2

            if border_2 > self.cfg.num_bins:
                border_2 -= self.cfg.num_bins
                
            borders = [border_1, border_2] if border_1 < border_2 else [border_2, border_1]
            in_border_count = histogram[borders[0]:borders[1]].sum()

            uneven_counts[batch_idx] = np.abs(
                in_border_count - total_pixel_count // 2
            ).astype(np.int32)

        return uneven_counts

    @torch.no_grad()
    def get_metrics(
        self,
        samples: Float[Tensor, "batch 3 height width"],
    ) -> dict[str, Float[Tensor, ""]]:

        images = ((samples + 1) / 2).clamp(0, 1)

        np_images = (
            images.cpu().numpy().transpose(0, 2, 3, 1)
        )  # batch height width channels
        hsv = rgb_to_hsv(np_images)

        hue_uneven_errors = self.get_hue_uneven_error_counts(hsv)

        avg_is_color_count_even = (hue_uneven_errors == 0).mean().item()
        avg_hue_uneven_errors = hue_uneven_errors.mean().item()

        values = {
            "saturation_std": self.get_avg_saturation_std(hsv),
            "value_std": self.get_avg_value_std(hsv),
            "hue_uneven_errors": avg_hue_uneven_errors,
            "is_color_count_even": avg_is_color_count_even,
        }

        return {k: torch.tensor(v, device=samples.device) for k, v in values.items()}

    @torch.no_grad()
    def evaluate(
        self, model: Wrapper, batch: BatchedSamplingExample, return_sample: bool = True
    ) -> Iterator[EvaluationOutput]:

        for res in super().evaluate(model, batch, return_sample):
            res["metrics"] = self.get_metrics(res["sample"]["sample"])
            yield res
