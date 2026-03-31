from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, default_collate

from ..misc.step_tracker import StepTracker
from . import DatasetCfg, get_dataset, get_dataset_class

from src.env import DEBUG
from src.type_extensions import ConditioningCfg
from ..evaluation import EvaluationCfg, get_evaluation


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool


@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg | None = None
    test: DataLoaderStageCfg | None = None
    val: DataLoaderStageCfg | None = None


class DataModule(LightningDataModule):
    dataset_cfg: DatasetCfg
    data_loader_cfg: DataLoaderCfg
    conditioning_cfg: ConditioningCfg
    patch_size: int | None
    validations: dict[str, EvaluationCfg] | None
    step_tracker: StepTracker | None

    def __init__(
        self,
        dataset_cfg: DatasetCfg,
        data_loader_cfg: DataLoaderCfg,
        conditioning_cfg: ConditioningCfg,
        patch_size: int | None = None,
        patch_grid_shape: tuple[int, int] | None = None,
        validations: dict[str, EvaluationCfg] | None = None,
        tests: dict[str, EvaluationCfg] | None = None,
        step_tracker: StepTracker | None = None,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.data_loader_cfg = data_loader_cfg
        self.conditioning_cfg = conditioning_cfg
        self.patch_size = patch_size
        self.patch_grid_shape = patch_grid_shape
        self.validations = validations
        self.tests = tests
        self.step_tracker = step_tracker

    @staticmethod
    def id_collate(
        batch: Sequence[Any], 
        id: Any, 
        collate: Callable[[Sequence[Any]], Any] = default_collate
    ) -> dict:
        res = collate(batch)
        return dict(id=id, data=res)

    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def prepare_data(self):
        # Download data in single process
        if get_dataset_class(self.dataset_cfg).includes_download:
            get_dataset(self.dataset_cfg, self.conditioning_cfg, "train")
            get_dataset(self.dataset_cfg, self.conditioning_cfg, "val")
            get_dataset(self.dataset_cfg, self.conditioning_cfg, "test")

    def setup_val_data(self):
        if self.validations is not None and self.validations:
            dataset = get_dataset(self.dataset_cfg, self.conditioning_cfg, "val")
            self.val_data = {k: get_evaluation(
                cfg, k, dataset, self.patch_size, self.patch_grid_shape
            ) for k, cfg in self.validations.items()}
        else:
            self.val_data = {}
    
    def setup_test_data(self):
        if self.tests is not None and self.tests:
            dataset = get_dataset(self.dataset_cfg, self.conditioning_cfg, "test")
            self.test_data = {k: get_evaluation(
                cfg, k, dataset, self.patch_size, self.patch_grid_shape, deterministic=True
            ) for k, cfg in self.tests.items()}
        else:
            self.test_data = {}

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = get_dataset(self.dataset_cfg, self.conditioning_cfg, "train")
            self.setup_val_data()
        elif stage == "validate":
            self.setup_val_data()
        elif stage == "test":
            self.setup_test_data()

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            self.data_loader_cfg.train.batch_size,
            shuffle=True,
            num_workers=self.data_loader_cfg.train.num_workers,
            drop_last=not DEBUG,    # drop last incomplete batch to avoid recompilation
            persistent_workers=self.get_persistent(self.data_loader_cfg.train)
        )

    def val_dataloader(self):
        if self.val_data is not None and self.val_data:
            return [DataLoader(
                vd,
                self.data_loader_cfg.val.batch_size,
                num_workers=self.data_loader_cfg.val.num_workers,
                collate_fn=partial(self.id_collate, id=k),
                persistent_workers=self.get_persistent(self.data_loader_cfg.val),
            ) for k, vd in self.val_data.items()]
        return []

    def test_dataloader(self):
        if self.test_data is not None and self.test_data:
            return [DataLoader(
                td,
                self.data_loader_cfg.test.batch_size,
                num_workers=self.data_loader_cfg.test.num_workers,
                collate_fn=partial(self.id_collate, id=k),
                persistent_workers=self.get_persistent(self.data_loader_cfg.test),
            ) for k, td in self.test_data.items()]
        return []
