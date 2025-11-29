from typing import Optional

from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTConditionalDataset(Dataset):
    """
    MNIST를 로드해서 [-1, 1] 범위의 (1, H, W) 텐서와 label 반환.

    옵션:
        - resize_to: 먼저 (resize_to, resize_to)로 리사이즈 (PIL 기준)
        - pad_to:    그 다음 (pad_to, pad_to)까지 zero-padding
                     (MNIST 원본은 28x28, 기준 사이즈는 resize_to 또는 28)

    예:
        resize_to=32, pad_to=None  -> 32x32
        resize_to=32, pad_to=40    -> 32x32 → pad → 40x40
        resize_to=None, pad_to=32  -> 28x28 → pad → 32x32
    padding 값은 0 → ToTensor → [-1,1] 스케일에서는 -1 (검정).
    """

    def __init__(
        self,
        root: str = "./data",
        split: str = "train",
        resize_to: Optional[int] = None,
        pad_to: Optional[int] = None,
        download: bool = True,
    ):
        super().__init__()

        assert split in ["train", "test"], "split must be 'train' or 'test'"
        train = (split == "train")

        transform_list = []
        original_size = 28  # MNIST 원본

        # 1) Resize
        if resize_to is not None and resize_to != original_size:
            transform_list.append(
                transforms.Resize((resize_to, resize_to))
            )

        # 2) Pad
        if pad_to is not None:
            # 현재 기준 크기: resize_to 있으면 그걸 기준, 없으면 28
            base_size = resize_to if resize_to is not None else original_size
            pad_total = max(pad_to - base_size, 0)
            pad_each = pad_total // 2
            # (left, top, right, bottom)
            if pad_each > 0:
                transform_list.append(
                    transforms.Pad(pad_each, fill=0)
                )

        # 3) Tensor 변환 (0~1)
        transform_list.append(transforms.ToTensor())

        # 4) [-1, 1]로 스케일링: x -> 2x - 1
        transform_list.append(
            transforms.Lambda(lambda x: x * 2.0 - 1.0)
        )

        transform = transforms.Compose(transform_list)

        mnist = datasets.MNIST(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )

        # 🔹 label 0 제거
        # mnist.targets: (N,) tensor, mnist.data: (N, 28, 28) tensor
        targets = mnist.targets
        mask = targets != 0  # 0 아닌 것만 True

        mnist.data = mnist.data[mask]
        mnist.targets = mnist.targets[mask]

        self.dataset = mnist

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label
