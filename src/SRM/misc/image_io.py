import io
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torchvision.io import write_video


FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]


FloatVideo = Union[
    Float[Tensor, "frame height width"],
    Float[Tensor, "frame channel height width"]
]


def prep_image(image: FloatImage) -> Union[
    UInt8[np.ndarray, "height width 3"],
    UInt8[np.ndarray, "height width 4"]
]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def prep_video(video: FloatVideo) -> UInt8[np.ndarray, "frame 3 height width"]:
    # Handle single-channel videos.
    if video.ndim == 3:
        video = rearrange(video, "f h w -> f () h w")

    # Ensure that there are 3 channels.
    _, channel, _, _ = video.shape
    if channel == 1:
        video = repeat(video, "f () h w -> f c h w", c=3)
    assert video.shape[1] == 3

    video = (video.detach().clip(min=0, max=1) * 255)\
        .to(dtype=torch.uint8, device="cpu")
    return video.numpy()
    

def save_image(
    image: UInt8[np.ndarray, "height width channel"],
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""
    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    # Save the image.
    Image.fromarray(image).save(path)


def load_image(
    path: Union[Path, str],
) -> Float[Tensor, "3 height width"]:
    return tf.ToTensor()(Image.open(path))[:3]


def save_video(
    video: UInt8[np.ndarray, "frame 3 height width"],
    path: Union[Path, str],
    **kwargs
) -> None:
    """Save an image. Assumed to be in range 0-1."""
    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    # Save the video.
    write_video(str(path), video.transpose(0, 2, 3, 1), **kwargs)
