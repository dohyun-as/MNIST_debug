"""
ImageNet Dataset for conditional diffusion training.

image_conditioning 모드에서 사용:
  - (image, label) 반환
  - 이미지를 [-1, 1]로 정규화
  - 기본 64×64 해상도 (조절 가능)
"""

from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class ImageNetConditionalDataset(Dataset):
    """
    ImageNet (또는 ImageNet subset) 데이터셋.

    Parameters
    ----------
    root : str
        ImageNet 데이터 루트 (train/ val/ 디렉토리 포함)
    split : str
        'train' or 'val'
    image_size : int
        이미지를 이 크기로 resize + center crop
    num_classes : int or None
        사용할 클래스 수 (None이면 전체 1000). 앞에서부터 num_classes개만 사용.
    max_samples_per_class : int or None
        클래스당 최대 샘플 수 (디버깅/소규모 실험용)
    """

    def __init__(
        self,
        root: str = "./data/imagenet",
        split: str = "train",
        image_size: int = 64,
        num_classes: Optional[int] = None,
        max_samples_per_class: Optional[int] = None,
        download: bool = False,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),              # [0, 1]
            transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [-1, 1]
        ])

        # ImageFolder 스타일: root/train/class_name/image.JPEG
        data_path = f"{root}/{split}"
        self.dataset = datasets.ImageFolder(data_path, transform=transform)

        # 클래스 수 제한
        if num_classes is not None and num_classes < len(self.dataset.classes):
            # 앞에서부터 num_classes개 클래스만 사용
            valid_classes = set(range(num_classes))
            indices = [
                i for i, (_, label) in enumerate(self.dataset.samples)
                if label in valid_classes
            ]
            self.indices = indices
        else:
            self.indices = list(range(len(self.dataset)))

        # 클래스당 최대 샘플 제한
        if max_samples_per_class is not None:
            from collections import defaultdict
            class_counts = defaultdict(int)
            filtered = []
            for idx in self.indices:
                _, label = self.dataset.samples[idx]
                if class_counts[label] < max_samples_per_class:
                    filtered.append(idx)
                    class_counts[label] += 1
            self.indices = filtered

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.dataset[real_idx]

        # RGB 3채널 보장
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3]

        return image, label


class TinyImageNetDataset(Dataset):
    """
    Tiny ImageNet (64×64, 200 classes) 데이터셋.
    ImageNet 전체가 없을 때 대안으로 사용.

    구조: root/train/class_id/images/image.JPEG
          root/val/images/image.JPEG (+ val_annotations.txt)
    """

    def __init__(
        self,
        root: str = "./data/tiny-imagenet-200",
        split: str = "train",
        image_size: int = 64,
        num_classes: Optional[int] = None,
    ):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ])

        data_path = f"{root}/{split}"
        self.dataset = datasets.ImageFolder(data_path, transform=transform)

        if num_classes is not None and num_classes < 200:
            valid_classes = set(range(num_classes))
            self.indices = [
                i for i, (_, l) in enumerate(self.dataset.samples)
                if l in valid_classes
            ]
        else:
            self.indices = list(range(len(self.dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return image, label
