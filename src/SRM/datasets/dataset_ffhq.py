from dataclasses import dataclass
import os
from typing import Literal

from torchvision.datasets import ImageFolder

from .dataset import Dataset, DatasetCfg
from .types import Example
from ..type_extensions import ConditioningCfg, Stage



@dataclass
class DatasetFFHQCfg(DatasetCfg):
    name: Literal["ffhq"] = "ffhq"


class DatasetFFHQ(Dataset[DatasetFFHQCfg]):
    def __init__(
        self, 
        cfg: DatasetFFHQCfg,
        conditioning_cfg: ConditioningCfg,
        stage: Stage
    ) -> None:
        super().__init__(cfg, conditioning_cfg, stage)
        self.dataset = ImageFolder(self.cfg.root)

    def load(self, idx: int) -> Example:
        path = os.path.relpath(self.dataset.samples[idx][0], start=self.cfg.root)
        return {"image": self.dataset[idx][0], "path": path}
    
    @property
    def _num_available(self) -> int:
        return len(self.dataset)
