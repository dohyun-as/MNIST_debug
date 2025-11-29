from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Iterator, TypedDict, TypeVar

import hydra
from jaxtyping import Float
import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from src.misc.image_io import prep_image, prep_video, save_image, save_video
from src.type_extensions import SamplingOutput
from src.visualization.color_map import apply_color_map_to_image
from src.visualization.layout import add_border, hcat

from ..dataset import Dataset
from ..model import Wrapper
from .types import EvaluationOutput, SamplingVisualization


@dataclass
class EvaluationCfg:
    name: str
    num_log_samples: int | None = 1  # if None: self.__len__
    save_samples: bool = True   # whether to save test samples
    save_frames: bool = False   # whether to save invidual frames of videos
    image_format: str = "png"
    fps: int = 6
    video_format: str = "mp4"


class UnbatchedExample(TypedDict):
    name: str


class BatchedExample(TypedDict):
    name: list[str]


T = TypeVar("T", bound=EvaluationCfg)
U = TypeVar("U", bound=UnbatchedExample)
B = TypeVar("B", bound=BatchedExample)


class Evaluation(TorchDataset, Generic[T, U, B], ABC):
    def __init__(
        self,
        cfg: T,
        tag: str,
        dataset: Dataset,
        patch_size: int | None = None,
        patch_grid_shape: tuple[int, int] | None = None,
        deterministic: bool = False
    ) -> None:
        super(Evaluation, self).__init__()
        self.cfg = cfg
        self.tag = tag
        self.dataset = dataset
        self.deterministic = deterministic
    
    @property
    def num_log_samples(self) -> int:
        return self.__len__() if self.cfg.num_log_samples is None \
            else self.cfg.num_log_samples

    @abstractmethod
    def evaluate(
        self, 
        model: Wrapper, 
        batch: B,
        return_sample: bool = True
    ) -> Iterator[EvaluationOutput]:
        pass

    @staticmethod
    def prep_sample(
        sample: SamplingOutput,
        masked: Float[Tensor, "batch channel height width"] | None = None,
        num_limit: int | None = None
    ) -> SamplingVisualization:
        """
        NOTE expects images to be in [-1, 1]
        """
        s = slice(num_limit)
        images = (sample["sample"][s] + 1) / 2
        vis: SamplingVisualization = {"images": [prep_image(img) for img in images]}
        if masked is not None:
            masked = (masked[s] + 1) / 2
            vis["masked"] = [prep_image(img) for img in masked]
        if "all_z_t" in sample:
            vis["videos"] = []
            for i, video in enumerate(sample["all_z_t"][s]):
                video = (video + 1) / 2
                if "all_t" in sample:
                    video = hcat(video, (1-sample["all_t"][i]).expand_as(video))
                if "all_sigma" in sample:
                    frame_sigma = sample["all_sigma"][i].squeeze(1)
                    mask = frame_sigma == 0
                    sigma_min = torch.where(mask, torch.inf, frame_sigma).min()
                    sigma_max = torch.where(mask, -torch.inf, frame_sigma).max()
                    frame_sigma = (frame_sigma - sigma_min) / (sigma_max - sigma_min)
                    frame_sigma_color = apply_color_map_to_image(frame_sigma)
                    frame_sigma_color.masked_fill_(mask.unsqueeze(1), 1)
                    video = hcat(video.expand(-1, 3, -1, -1), frame_sigma_color)
                if "all_x" in sample:
                    video = hcat(video, ((sample["all_x"][i] + 1) / 2).expand(-1, video.shape[1], -1, -1))
                if any(k in sample for k in ("all_t", "all_sigma", "all_x")):
                    video = add_border(video)
                vis["videos"].append(prep_video(video))
        return vis

    def log_sample(
        self,
        model: Wrapper,
        sample: SamplingOutput,
        key: str,
        names: list[str],
        masked: Float[Tensor, "batch channel height width"] | None = None,
        num_log: int | None = None
    ) -> None:
        names = names[slice(num_log)]
        vis = self.prep_sample(sample, masked, num_log)
        step = model.step_tracker.get_step()
        model.logger.log_image(f"{key}/sample", vis["images"], step=step, caption=names)
        if "videos" in vis:
            num_videos = len(vis["videos"])
            model.logger.log_video(
                f"{key}/video", vis["videos"], step=step, caption=names, 
                fps=num_videos * [self.cfg.fps], format=num_videos * [self.cfg.video_format]
            )

    def save_sample(
        self,
        sample: SamplingOutput,
        key: str,
        names: list[str],
        masked: Float[Tensor, "batch channel height width"] | None = None,
    ) -> None:
        output_path = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]) / Path(key)
        vis = self.prep_sample(sample, masked)
        for i, s in enumerate(vis["images"]):
            save_image(s, output_path / f"{names[i]}.{self.cfg.image_format}")
        if "masked" in vis:
            for i, m in enumerate(vis["masked"]):
                save_image(m, output_path / f"{names[i]}_masked.{self.cfg.image_format}")
        if "videos" in vis:
            for i, v in enumerate(vis["videos"]):
                save_video(v, output_path / f"{names[i]}.{self.cfg.video_format}", fps=self.cfg.fps)
                if self.cfg.save_frames:
                    v = v.transpose(0, 2, 3, 1)
                    for j, f in enumerate(v):
                        save_image(f, output_path / names[i] / f"{j}.{self.cfg.image_format}")

    def validate(
        self,
        model: Wrapper,
        batch: B,
        batch_idx: int
    ) -> None:
        batch_size = len(batch["name"])
        num_log_samples = max(0, min(batch_size, self.num_log_samples - batch_size * batch_idx))
        # NOTE logging only in rank zero process
        log_samples = model.global_rank == 0 and num_log_samples > 0
        for res in self.evaluate(model, batch, return_sample=log_samples):
            key = f"val/{self.tag}/{res['key']}"
            if log_samples:
                self.log_sample(model, res["sample"], key, res["names"], res.get("masked", None), num_log_samples)
            if "metrics" in res:
                # prepend metric keys for logging purposes
                res["metrics"] = {f"{key}/{k}": v for k, v in res["metrics"].items()}
                # NOTE uses pytorch lightning magic, i.e., mean-reduction metric accumulation 
                # and logging at epoch level
                model.log_dict(res["metrics"], batch_size=batch_size, sync_dist=True)

    def test(
        self,
        model: Wrapper,
        batch: B,
        batch_idx: int
    ) -> None:
        batch_size = len(batch["name"])
        for res in self.evaluate(model, batch, return_sample=self.cfg.save_samples):
            key = f"test/{self.tag}/{res['key']}"
            if self.cfg.save_samples:
                self.save_sample(res["sample"], key, res["names"], res.get("masked", None))
            if "metrics" in res:
                # prepend metric keys for logging purposes
                res["metrics"] = {f"{key}/{k}": v for k, v in res["metrics"].items()}
                # NOTE uses pytorch lightning magic, i.e., mean-reduction metric accumulation 
                # and logging at epoch level
                model.log_dict(res["metrics"], add_dataloader_idx=False, batch_size=batch_size, sync_dist=True)

    def predict(
        self,
        model: Wrapper,
        batch: B,
        batch_idx: int
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, idx: int) -> U:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
