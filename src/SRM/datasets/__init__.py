from typing import Type, Union

from .dataset import Dataset
from .dataset_ffhq import DatasetFFHQ, DatasetFFHQCfg
from .dataset_grid import DatasetGrid   # noqa
from .dataset_mnist import DatasetMnist, DatasetMnistCfg
from .dataset_mnist_sudoku_3x3_lazy import DatasetMnist3x3GridLazy, DatasetMnist3x3GridLazyCfg
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
    "mnist_grid3x3_lazy": DatasetMnist3x3GridLazy,
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
    DatasetMnist3x3GridLazyCfg,
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
