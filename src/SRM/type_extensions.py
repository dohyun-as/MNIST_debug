from dataclasses import dataclass
from typing import Literal, TypedDict

from jaxtyping import Float, Int64
from torch import Tensor


FullPrecision = Literal[32, 64, "32-true", "64-true", "32", "64"]
HalfPrecision = Literal[16, "16-true", "16-mixed", "bf16-true", "bf16-mixed", "bf16", "16"]
Stage = Literal["train", "val", "test"]

Parameterization = Literal["eps", "ut"]


# NOTE conditioning both required for model and dataset
# therefore define here to avoid circular dependencies
@dataclass
class ConditioningCfg:
    label: bool = False
    mask: bool = False


class UnbatchedExample(TypedDict, total=False):
    index: int
    image: Float[Tensor, "channel height width"]
    label: int
    # if mask == 1: needs to be inpainted
    mask: Float[Tensor, "1 height width"]
    path: str


class BatchedExample(TypedDict, total=False):
    index: Int64[Tensor, "batch"]
    image: Float[Tensor, "batch channel height width"]
    label: Int64[Tensor, "batch"]
    mask: Float[Tensor, "batch 1 height width"]
    path: list[str]


class BatchedEvaluationExample(TypedDict):
    id: str
    data: dict


class SamplingOutput(TypedDict, total=False):
    sample: Float[Tensor, "batch channel height width"]
    all_z_t: list[Float[Tensor, "frame channel height width"]]
    all_t: list[Float[Tensor, "frame 1 height width"]]
    all_sigma: list[Float[Tensor, "frame 1 height width"]]
    all_x: list[Float[Tensor, "frame channel height width"]]
