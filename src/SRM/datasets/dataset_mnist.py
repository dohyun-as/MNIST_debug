from dataclasses import dataclass
from typing import Literal

from torchvision.datasets import MNIST

from .dataset import Dataset, DatasetCfg
from .types import Example
from ..type_extensions import ConditioningCfg, Stage



@dataclass
class DatasetMnistCfg(DatasetCfg):
    name: Literal["mnist"] = "mnist"


class DatasetMnist(Dataset[DatasetMnistCfg]):
    includes_download = True
    dataset: MNIST

    def __init__(
        self, 
        cfg: DatasetMnistCfg,
        conditioning_cfg: ConditioningCfg,
        stage: Stage
    ) -> None:
        super().__init__(cfg, conditioning_cfg, stage)
        self.dataset = MNIST(
            self.cfg.root, 
            train=self.stage=="train",
            download=True
        )

    def load(self, idx: int) -> Example:
        return {"image": self.dataset[idx][0]}
    
    @property
    def _num_available(self) -> int:
        return len(self.dataset)
