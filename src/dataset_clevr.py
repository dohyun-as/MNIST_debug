"""
CLEVR Dataset for conditional diffusion training.

image_conditioning 모드에서 사용:
  - (image, label) 반환
  - label: 이미지 내 물체 개수 (counting task) 또는 속성 조합
  - 이미지를 [-1, 1]로 정규화
  - 기본 128×128 또는 64×64 해상도

CLEVR 데이터 구조:
  root/
    images/
      train/  CLEVR_train_000000.png ...
      val/    CLEVR_val_000000.png ...
    scenes/
      CLEVR_train_scenes.json
      CLEVR_val_scenes.json
"""

import os
import json
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CLEVRDataset(Dataset):
    """
    CLEVR 데이터셋.

    Conditioning 전략:
      - 'count': 물체 개수로 conditioning (0~10 등)
      - 'image': 이미지 자체로 conditioning (image-to-image)

    Parameters
    ----------
    root : str
        CLEVR 데이터 루트 디렉토리
    split : str
        'train' or 'val'
    image_size : int
        이미지를 이 크기로 resize
    label_type : str
        'count' (물체 개수), 'num_objects' (same as count),
        'scene_hash' (scene 속성 기반 pseudo-label)
    max_objects : int
        최대 물체 수 (이보다 많은 물체가 있는 이미지는 max_objects로 clamp)
    max_samples : int or None
        최대 샘플 수 (디버깅용)
    """

    def __init__(
        self,
        root: str = "./data/CLEVR_v1.0",
        split: str = "train",
        image_size: int = 128,
        label_type: str = "count",
        max_objects: int = 10,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.image_size = image_size
        self.label_type = label_type
        self.max_objects = max_objects

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ])

        # 이미지 경로 수집
        img_dir = os.path.join(root, "images", split)
        if not os.path.exists(img_dir):
            # 대안: root 자체가 이미지 디렉토리
            img_dir = os.path.join(root, split)

        self.image_paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ])

        # Scene JSON에서 라벨 추출
        self.labels = self._load_labels()

        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]
            if self.labels is not None:
                self.labels = self.labels[:max_samples]

    def _load_labels(self):
        """Scene JSON에서 라벨(물체 개수 등) 추출."""
        scene_file = os.path.join(
            self.root, "scenes", f"CLEVR_{self.split}_scenes.json"
        )
        if not os.path.exists(scene_file):
            # Scene file 없으면 이미지 이름에서 index 기반 dummy label
            return None

        with open(scene_file, "r") as f:
            scenes_data = json.load(f)

        scenes = scenes_data.get("scenes", scenes_data)

        # image_filename → scene 매핑
        filename_to_scene = {}
        for scene in scenes:
            fname = scene.get("image_filename", "")
            filename_to_scene[fname] = scene

        labels = []
        for img_path in self.image_paths:
            fname = os.path.basename(img_path)
            scene = filename_to_scene.get(fname, None)

            if scene is not None:
                n_objects = len(scene.get("objects", []))
                label = min(n_objects, self.max_objects)
            else:
                label = 0

            labels.append(label)

        return labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)

        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = 0

        return img, label


class CLEVRFolderDataset(Dataset):
    """
    CLEVR 또는 CLEVR-like 데이터셋을 ImageFolder 스타일로 로드.
    클래스 폴더 구조: root/class_0/img.png, root/class_1/img.png, ...

    Scene JSON이 없는 경우나 커스텀 CLEVR 변형에 사용.
    """

    def __init__(
        self,
        root: str,
        image_size: int = 128,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        from torchvision.datasets import ImageFolder

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ])

        self.dataset = ImageFolder(root, transform=transform)

        if max_samples is not None:
            self.indices = list(range(min(max_samples, len(self.dataset))))
        else:
            self.indices = list(range(len(self.dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return image, label
