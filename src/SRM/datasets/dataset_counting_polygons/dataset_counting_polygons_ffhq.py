from dataclasses import dataclass
from typing import Literal

from .dataset_counting_polygons_subdataset import (
    DatasetCountingPolygonsSubdataset,
    DatasetCountingPolygonsSubdatasetCfg,
)
from ..dataset_ffhq import DatasetFFHQCfg, DatasetFFHQ


@dataclass
class DatasetCountingPolygonsFFHQCfg(DatasetCountingPolygonsSubdatasetCfg):
    name: Literal[
        "counting_polygons_ffhq",
        "counting_polygons_ffhq_explicit_conditional",
        "counting_polygons_ffhq_ambiguous_conditional",
    ] = "counting_polygons_ffhq"
    subdataset_cfg: DatasetFFHQCfg | None = None  # should never be None in practice


class DatasetCountingPolygonsFFHQ(
    DatasetCountingPolygonsSubdataset[DatasetCountingPolygonsFFHQCfg]
):
    dataset_class = DatasetFFHQ
