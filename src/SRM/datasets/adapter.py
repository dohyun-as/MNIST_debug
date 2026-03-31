# # semanticist/utils/datasets/sudoku9x9_lazy_adapter.py
# from torch.utils.data import Dataset
# import numpy as np
# from .dataset_mnist_sudoku_9x9_lazy import (
#     DatasetMnistSudoku9x9LazyCfg, DatasetMnistSudoku9x9Lazy
# )
# from ..type_extensions import ConditioningCfg

# class Sudoku9x9LazyAsImageNet(Dataset):
#     def __init__(
#         self,
#         root: str,
#         split: str = "train",
#         img_size: int = 252,            # 9*28 권장
#         augment: bool = False,
#         grayscale: bool = True,
#         top_n: int = 100,
#         test_samples_num: int = 10000,
#         mask_self_dependency: bool = True,
#         # 힌트 마스크 쓰고 싶을 때
#         mask: bool = False,
#         num_fill=None,                  # int or [lo, hi] or None
#     ):
#         cfg = DatasetMnistSudoku9x9LazyCfg(
#             image_shape=(img_size, img_size),
#             subset_size=None,
#             augment=augment,
#             grayscale=grayscale,
#             root=root,
#             mask_self_dependency=mask_self_dependency,
#             top_n=top_n,
#             test_samples_num=test_samples_num,
#         )
#         cond = ConditioningCfg(label=False, mask=mask)
#         stage = "train" if split == "train" else "test"
#         self.inner = DatasetMnistSudoku9x9Lazy(cfg, cond, stage)
#         self.mask = mask
#         self.num_fill = num_fill

#     def __len__(self):
#         return len(self.inner)

#     # def _num_given_cells(self, idx):
#     #     nf = self.num_fill
#     #     if not self.mask or nf is None:
#     #         return None
#     #     if isinstance(nf, int):
#     #         return nf
#     #     if isinstance(nf, (list, tuple)) and len(nf) == 2:
#     #         lo, hi = nf
#     #         return int(np.random.default_rng(idx).integers(lo, hi + 1))
#     #     raise ValueError(f"bad num_fill: {nf!r}")

#     def __getitem__(self, idx):
#         # ngc = self._num_given_cells(idx)
#         # sample = self.inner.__getitem__(idx, **({} if ngc is None else {"num_given_cells": ngc}))
#         sample = self.inner.__getitem__(idx)
#         img = sample["image"]
#         label = 0             
#         return img, label

# semanticist/utils/datasets/adapter.py

from typing import Type, Union

from dataclasses import is_dataclass
from typing import Type, Union, Dict, Any

from torch.utils.data import Dataset as TorchDataset

from .dataset import Dataset
from .dataset_ffhq import DatasetFFHQ, DatasetFFHQCfg
from .dataset_grid import DatasetGrid   # noqa
from .dataset_mnist import DatasetMnist, DatasetMnistCfg
from .dataset_mnist_sudoku_3x3_eager import DatasetMnistSudoku3x3Eager, DatasetMnistSudoku3x3EagerCfg
from .dataset_mnist_sudoku_9x9_eager import DatasetMnistSudoku9x9Eager, DatasetMnistSudoku9x9EagerCfg
from .dataset_mnist_sudoku_9x9_lazy import DatasetMnistSudoku9x9Lazy, DatasetMnistSudoku9x9LazyCfg
from .dataset_counting_polygons import DatasetCountingPolygonsBlank, DatasetCountingPolygonsBlankCfg
from .dataset_counting_polygons import DatasetCountingPolygonsFFHQ, DatasetCountingPolygonsFFHQCfg
from .dataset_even_pixels import DatasetEvenPixels, DatasetEvenPixelsCfg

from ..type_extensions import ConditioningCfg, Stage


DATASETS: dict[str, Dataset] = {
    "ffhq": DatasetFFHQ,
    "mnist": DatasetMnist,
    "mnist_grid": DatasetMnistSudoku3x3Eager,
    "mnist_sudoku": DatasetMnistSudoku9x9Eager,
    "mnist_sudoku_lazy": DatasetMnistSudoku9x9Lazy,
    "counting_polygons_blank": DatasetCountingPolygonsBlank,
    "counting_polygons_blank_explicit_conditional": DatasetCountingPolygonsBlank,
    "counting_polygons_blank_ambiguous_conditional": DatasetCountingPolygonsBlank,
    "counting_polygons_ffhq": DatasetCountingPolygonsFFHQ,
    "counting_polygons_ffhq_explicit_conditional": DatasetCountingPolygonsFFHQ,
    "counting_polygons_ffhq_ambiguous_conditional": DatasetCountingPolygonsFFHQ,
    "even_pixels": DatasetEvenPixels,
}


DatasetCfg = Union[
    DatasetFFHQCfg,
    DatasetMnistCfg,
    DatasetMnistSudoku3x3EagerCfg,
    DatasetMnistSudoku9x9EagerCfg,
    DatasetMnistSudoku9x9LazyCfg,
    DatasetCountingPolygonsBlankCfg,
    DatasetCountingPolygonsFFHQCfg,
    DatasetEvenPixelsCfg,
]



def get_dataset_class(
    cfg: DatasetCfg
) -> Type[Dataset]:
    return DATASETS[cfg.name]


def get_dataset(
    cfg: DatasetCfg,
    conditioning_cfg: ConditioningCfg,
    stage: Stage,
) -> Dataset:
    return DATASETS[cfg.name](cfg, conditioning_cfg, stage)


class Adapter(TorchDataset):
    """
    얘 하나로 모든 등록된 데이터셋을 동일한 인터페이스로 생성해서 쓸 수 있게 함.

    사용 예:
        ds = Adapter(
            name="mnist_sudoku_lazy",
            split="train",
            conditioning={"label": False, "mask": True},
            cfg_kwargs={
                "image_shape": (252, 252),
                "subset_size": None,
                "augment": False,
                "grayscale": True,
                "root": "./datasets/mnist_sudoku",
                "mask_self_dependency": True,
                "top_n": 100,
                "test_samples_num": 10000,
            }
        )
    """

    def __init__(
        self,
        name: str,
        split: Stage | str = "train",
        conditioning: ConditioningCfg | Dict[str, Any] | None = None,
        cfg: DatasetCfg | None = None,
        cfg_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            name: registry 키 (예: "mnist_sudoku_lazy")
            split: "train" | "val" | "test"
            conditioning: ConditioningCfg 또는 dict(label=..., mask=...)
            cfg: 이미 만들어진 XXXCfg 인스턴스(있으면 그대로 사용)
            cfg_kwargs: Cfg를 새로 만들 때 넘길 파라미터 dict
        """
        super().__init__()

        self.name = name
        stage: Stage = split if split in ("train", "val", "test") else "train"

        # Conditioning
        if isinstance(conditioning, dict) or conditioning is None:
            conditioning = ConditioningCfg(**({"label": False, "mask": False} | (conditioning or {})))
        elif not isinstance(conditioning, ConditioningCfg):
            raise TypeError("conditioning must be ConditioningCfg or dict")

        # Build cfg if not provided
        if cfg is None:
            cfg_cls = get_cfg_class(name)
            cfg_kwargs = cfg_kwargs or {}
            cfg = cfg_cls(**cfg_kwargs)
        else:
            # sanity check
            if not is_dataclass(cfg):
                raise TypeError("cfg must be a dataclass instance (e.g., Dataset*Cfg)")
            # minimal name check if 필드가 있다면
            if hasattr(cfg, "name"):
                expected = name
                actual = getattr(cfg, "name")
                if actual != expected:
                    # 강제 일치가 필요한 건 아니지만, 보통 이름이 안 맞으면 실수이므로 경고/에러 중 택1
                    raise ValueError(f"cfg.name={actual!r} does not match adapter name={expected!r}")

        ds_cls = get_dataset_class(name)
        self.inner: Dataset = ds_cls(cfg, conditioning, stage)

    # Torch Dataset interface
    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx):
        return self.inner.__getitem__(idx)