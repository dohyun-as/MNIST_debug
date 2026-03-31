from typing import TypedDict

from jaxtyping import Float
from numpy import ndarray
from PIL.Image import Image


class Example(TypedDict, total=True):
    # NOTE this Image can include a mask as alpha channel (LA or RGBA mode)
    image: Image
    label: int | dict[str, int]
    path: str
