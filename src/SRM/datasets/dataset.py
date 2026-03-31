from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from math import prod, exp
from typing import Generic, Sequence, TypeVar

from einops import rearrange
from jaxtyping import Float
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.nn.functional import conv2d
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms.v2 import (
    CenterCrop,
    Compose,
    Grayscale,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RGB,
    ToTensor,
)
from torchvision.transforms import ToTensor

from .types import Example
from ..type_extensions import (
    ConditioningCfg,
    Stage,
    UnbatchedExample
)


@dataclass
class DatasetCfg:
    image_shape: Sequence[int]
    subset_size: int | None
    augment: bool
    grayscale: bool
    root: Path | str = ""
    dependency_matrix_sigma: float = 2.0 # Might make sense to increase for larger resolutions
    pad_to_size: Sequence[int] | None = None 
    pad_mode: str = "replicate"
    pad_value: float = 0.0       
    
    crop_shape: Sequence[int] | None = None

    adjust_with_resize: bool = False          # True면 pad 대신 resize 사용
    resize_mode: str = "bicubic"              # "nearest"|"bilinear"|"bicubic"|"area"
    resize_antialias: bool = True             # bilinear/bicubic일 때만 의미 있음
    
    num_fill: int | Sequence[int] | None = None
    return_masked: bool = False

    reveal_cells: Sequence[tuple[int, int]] | None = None   # 예: [(5,5), (1,9)]

T = TypeVar("T", bound=DatasetCfg)


class Dataset(TorchDataset, Generic[T], ABC):
    includes_download: bool = False
    num_classes: int | None = None
    cfg: T
    conditioning_cfg: ConditioningCfg
    stage: Stage

    def __init__(
        self,
        cfg: T,
        conditioning_cfg: ConditioningCfg,
        stage: Stage,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.conditioning_cfg = conditioning_cfg
        self.stage = stage
        self.deterministic = (stage != "train")
        crop_shape = getattr(self.cfg, "crop_shape", None) or self.cfg.image_shape
        # Define transforms
        transforms = [
            Lambda(lambda pil_image: self.relative_resize(pil_image, self.cfg.image_shape)),
            CenterCrop(crop_shape),
            ToTensor()
        ]
        if self.cfg.augment:
            transforms.insert(2, RandomHorizontalFlip())
        self.transform = Compose(transforms)
        self.rgb_transform = Compose([
            Grayscale() if cfg.grayscale else RGB(),
            Normalize(mean=self.d_data * [0.5], std=self.d_data * [0.5], inplace=True)
        ])
        
    @staticmethod
    def _resize_to(
        x: Tensor,
        size: Sequence[int],
        mode: str = "bicubic",
        antialias: bool = True,
    ) -> Tensor:
        """
        x: (C,H,W) tensor -> return same shape with resized H,W
        """
        th, tw = int(size[0]), int(size[1])
        x4 = x.unsqueeze(0)  # (1,C,H,W)
        # align_corners는 bilinear/bicubic일 때만 설정
        align = None
        if mode in ("bilinear", "bicubic"):
            align = False
        x4 = F.interpolate(
            x4, size=(th, tw), mode=mode,
            align_corners=align,
            antialias=(antialias if mode in ("bilinear", "bicubic") else False),
        )
        return x4.squeeze(0)
        
    @staticmethod
    def _pad_to(
        x: Tensor,
        size: Sequence[int],
        mode: str = "replicate",
        value: float = 0.0,
    ) -> tuple[Tensor, tuple[int, int, int, int] | None]:
        """
        x: (C,H,W) tensor
        size: (TH, TW)
        return: (padded_x, (pt, pb, pl, pr)) ; 패딩이 없으면 offsets=None
        """
        C, H, W = x.shape
        TH, TW = int(size[0]), int(size[1])
        if (H, W) == (TH, TW):
            return x, None
        assert TH >= H and TW >= W, f"pad_to_size {size}는 입력 {(H,W)}보다 크거나 같아야 합니다."
        dh, dw = TH - H, TW - W
        pt, pb = dh // 2, dh - dh // 2
        pl, pr = dw // 2, dw - dw // 2
        if mode == "constant":
            x = F.pad(x, (pl, pr, pt, pb), mode="constant", value=value)
        else:
            x = F.pad(x, (pl, pr, pt, pb), mode=mode)
        return x, (pt, pb, pl, pr)
        
    def _get_dependency_2d_gaussian_kernel(
        self,
        grid_shape: tuple[int, int]
    ) -> Float[Tensor, "max_grid_size max_grid_size"]:
        sigma = self.cfg.dependency_matrix_sigma
        
        kernel_size = max(grid_shape)
        
        # make sure kernel size is odd
        kernel_size += 1 - kernel_size % 2
        
        kernel_1d = torch.tensor([exp(-(x - kernel_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(kernel_size)])
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d /= kernel_2d.sum()  # Normalize
        
        assert kernel_2d.shape == (kernel_size, kernel_size)
        return kernel_2d 

    def get_dependency_matrix(
        self,
        grid_shape: tuple[int, int]
    ) -> Float[Tensor, "num_patches num_patches"] | None:
        "The Default dependency matrix is based on the locality assumption"
        kernel = self._get_dependency_2d_gaussian_kernel(grid_shape)
        
        total_patches = prod(grid_shape)
        dep_matrix = torch.eye(total_patches) # Initialy all patches are independent
        
        # (x1 y1) 1 x2 y2 as we want to blur over the last two dimensions, 
        # we have to add a dummy dimension for channels
        dep_tensor = rearrange(
            dep_matrix, 
            "... (x y) -> ... 1 x y", 
            x=grid_shape[0], 
            y=grid_shape[1],
        ) 
        blurred = conv2d(dep_tensor, kernel[None, None], padding="same")
        return rearrange(blurred, "... 1 x y -> ... (x y)")

    @property
    def d_data(self) -> int:
        return 1 if self.cfg.grayscale else 3

    @staticmethod
    def relative_resize(
        image: Image.Image, 
        target_shape: Sequence[int]
    ) -> Image.Image:
        target_shape = np.asarray(target_shape[::-1])
        while np.all(np.asarray(image.size) >= 2 * target_shape):
            image = image.resize(
                tuple(x // 2 for x in image.size), 
                resample=Image.Resampling.BOX
            )

        scale = np.max(target_shape / np.asarray(image.size))
        image = image.resize(
            tuple(round(x * scale) for x in image.size), 
            resample=Image.Resampling.BICUBIC
        )
        return image

    @staticmethod
    def concat_mask(
        image: Image.Image,
        mask: Image.Image | Float[np.ndarray, "height width"]
    ) -> Image.Image:
        assert image.mode in ("L", "RGB")
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(np.uint8(255 * mask), mode="L")
        else:
            assert mask.mode == "L"
        if image.mode == "L":
            return Image.merge("LA", (image, mask))
        r, g, b = image.split()
        return Image.merge("RGBA", (r, g, b, mask))

    @abstractmethod
    def load(self, idx: int, **kwargs) -> Example:
        """
        NOTE image of Example must include a mask as alpha channel 
        (LA or RGBA mode) if conditioning_cfg.mask
        """
        pass

    def __getitem__(self, idx: int, **load_kwargs) -> UnbatchedExample:
        if hasattr(self.cfg, "num_fill"):
            if isinstance(self.cfg.num_fill, int):
                num_given_cells = self.cfg.num_fill
                load_kwargs["num_given_cells"] = num_given_cells
                
            else: # Sequence[int] | None
                if isinstance(self.cfg.num_fill, Sequence):
                    interval = (self.cfg.num_fill[0], self.cfg.num_fill[1]+1) # +1 because randint is exclusive
                    assert 0 <= interval[0] <= interval[1] <= prod(self.grid_size) + 1, f"Invalid interval for num_fill"
                
                elif self.cfg.num_fill is None: # Random number of cells
                    interval = (0, prod(self.grid_size)+1) # +1 because randint is exclusive
                    
                else:
                    raise ValueError(f"Invalid value for num_fill: {self.cfg.num_fill}, expected int, Sequence[int] or None")
            
                num_given_cells = (
                    np.random.default_rng(idx).integers(*interval).item()
                    if self.deterministic else np.random.randint(*interval)
                ) 
            
                load_kwargs["num_given_cells"] = num_given_cells
        
        sample = self.load(idx, **load_kwargs)
        res = UnbatchedExample(index=idx)
        is_mask_given = sample["image"].mode in ("LA", "RGBA")
        assert not self.conditioning_cfg.mask or is_mask_given, "Mask conditioning but no mask given"
        res["image"] = self.transform(sample["image"])
        if is_mask_given:
            res["mask"] = res["image"][-1:]
            res["image"] = res["image"][:-1]
            res["mask"].round_()
            
        # ⬇ 기존: pad만 하던 부분을 'pad 또는 resize'로 선택 가능하게 변경
        if getattr(self.cfg, "pad_to_size", None):
            target_hw = self.cfg.pad_to_size
            if getattr(self.cfg, "adjust_with_resize", False):
                # --- resize로 맞추기 ---
                res["image"] = self._resize_to(
                    res["image"], target_hw,
                    mode=getattr(self.cfg, "resize_mode", "bicubic"),
                    antialias=getattr(self.cfg, "resize_antialias", True),
                )
                if is_mask_given:
                    # 마스크는 최근접 보간으로 (0/1 유지)
                    res["mask"] = self._resize_to(
                        res["mask"], target_hw,
                        mode="nearest", antialias=False
                    )
            else:
                # --- 기존 padding ---
                res["image"], _ = self._pad_to(
                    res["image"],
                    size=target_hw,
                    mode=getattr(self.cfg, "pad_mode", "replicate"),
                    value=getattr(self.cfg, "pad_value", 0.0),
                )
                if is_mask_given:
                    res["mask"], _ = self._pad_to(
                        res["mask"],
                        size=target_hw,
                        mode=getattr(self.cfg, "pad_mode", "replicate"),
                        value=getattr(self.cfg, "pad_value", 0.0),
                    )
        res["image"] = self.rgb_transform(res["image"])
        if "grid" in sample:
            res["grid"] = sample["grid"]
        if "path" in sample:
                res["path"] = sample["path"]
        if "cell_mask" in sample:
            res["cell_mask"] = sample["cell_mask"]
        if self.conditioning_cfg.label:
            res["label"] = sample["label"]
        return res

    @property
    @abstractmethod
    def _num_available(self) -> int:
        pass

    def __len__(self) -> int:
        if self.cfg.subset_size is not None:
            return self.cfg.subset_size
        return self._num_available
