from typing import NotRequired, TypedDict

from jaxtyping import Float, UInt8
import numpy as np
from torch import Tensor

from src.type_extensions import SamplingOutput


class SamplingVisualization(TypedDict, total=False):
    images: list[UInt8[np.ndarray, "height width 3"]]
    videos: NotRequired[list[UInt8[np.ndarray, "frame 3 height width"]]]
    masked: NotRequired[list[UInt8[np.ndarray, "height width 3"]]]


class EvaluationOutput(TypedDict, total=False):
    key: str
    names: list[str]
    sample: SamplingOutput
    masked: NotRequired[Float[Tensor, "batch channel height width"]]
    metrics: NotRequired[dict[str, Float[Tensor, ""]]]
