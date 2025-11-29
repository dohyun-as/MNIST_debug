from dataclasses import dataclass
from typing import Literal

from jaxtyping import Bool, Integer, Shaped, Float
import torch
from torch import Tensor

from ..datasets import DatasetMnistSudoku9x9Eager, DatasetMnistSudoku9x9Lazy

from .mnist_evaluation import MnistEvaluation, MnistEvaluationCfg
from ..global_cfg import get_mnist_classifier_path
from ..misc.mnist_classifier import get_classifier


@dataclass
class MnistSudokuEvaluationCfg(MnistEvaluationCfg):
    name: Literal["mnist_sudoku"] = "mnist_sudoku"


class MnistSudokuEvaluation(MnistEvaluation[MnistSudokuEvaluationCfg]):
    def __init__(
        self, 
        cfg: MnistEvaluationCfg, 
        tag: str, 
        dataset: DatasetMnistSudoku9x9Eager | DatasetMnistSudoku9x9Lazy,
        patch_size: int | None = None,
        patch_grid_shape: tuple[int, int] | None = None,
        deterministic: bool = False
    ) -> None:
        super().__init__(cfg, tag, dataset, patch_size, patch_grid_shape, deterministic)
    
    def classify(
        self,
        pred: Integer[Tensor, "batch grid_size grid_size"]
    ) -> tuple[
        Bool[Tensor, "batch"],
        dict[str, Shaped[Tensor, "batch"]]
    ]:
        batch_size, grid_size = pred.shape[:2]
        sub_grid_size = round(grid_size ** 0.5)
        dtype, device = pred.dtype, pred.device
        pred = pred - 1 # Shift [1, 9] to [0, 8] for indices
        ones = torch.ones((1,), dtype=dtype, device=device).expand_as(pred)
        dist = torch.zeros((batch_size,), dtype=dtype, device=device)
        for dim in range(1, 3):
            cnt = torch.full_like(pred, fill_value=-1)
            cnt.scatter_add_(dim=dim, index=pred, src=ones)
            dist.add_(cnt.abs_().sum(dim=(1, 2)))
        # Subgrids
        grids = pred.unfold(1, sub_grid_size, sub_grid_size)\
            .unfold(2, sub_grid_size, sub_grid_size).reshape(-1, grid_size, grid_size)
        cnt = torch.full_like(grids, fill_value=-1)
        cnt.scatter_add_(dim=dim, index=grids, src=ones)
        dist.add_(cnt.abs_().sum(dim=(1, 2)))
        label = dist == 0
        return label, {"distance": dist}

    @torch.no_grad()
    def eval_images(
        self,
        samples: Float[Tensor, "batch 1 height width"],
    ) -> dict[str, Tensor]:
        """
        입력: [-1, 1] 범위로 정규화된 (B,1,H,W) 이미지 배치.
        출력:
          - labels: (B,) bool, 스도쿠 규칙 완벽히 만족 여부
          - distance: (B,) float/int, 규칙 위반 총량(0이면 완벽)
          - accuracy: () float, 배치 평균 정답률
          - discrete: (B, grid_h, grid_w) int, 타일 단위 MNIST 분류 결과(1~9 범위)
        """
        # 1) MNIST digit classifier 로드
        classifier = get_classifier(get_mnist_classifier_path(), samples.device)

        # 2) 타일 단위로 자르고 MNIST 분류해서 1~9 이산화
        discrete = self.discretize(classifier, samples)  # (B, Gh, Gw), 값 ∈ {1..9}

        # 3) 스도쿠 규칙 검사(행/열/서브그리드), distance 계산
        labels, metrics = self.classify(discrete)        # labels: (B,), metrics["distance"]:(B,)

        # 4) 배치 평균 accuracy와 함께 리턴
        out = {
            "labels": labels,                                # (B,)
            "distance": metrics["distance"],                 # (B,)
            "accuracy": labels.float().mean(),               # ()
            "discrete": discrete,                            # (B, Gh, Gw)
        }
        return out