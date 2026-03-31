import torch
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float, Integer, Bool, Shaped

from ..misc.mnist_classifier import get_classifier, MNISTClassifier



class MnistSudokuEvalOnly:
    """
    Dataset 없이, 이미지만 넣으면
    - 타일별 숫자 예측 (1~9)
    - 스도쿠 규칙 만족 여부 (labels)
    - 위반량(distance)
    - 배치 평균 accuracy
    를 반환하는 eval-only evaluator
    """
    def __init__(self, mnist_classifier_path, grid_size = (9,9)):
        self.grid_size = grid_size
        self.mnist_classifier_path= mnist_classifier_path

    @torch.no_grad()
    def discretize(
        self,
        classifier: MNISTClassifier,
        samples: Float[Tensor, "batch 1 height width"],
    ) -> Integer[Tensor, "batch grid_h grid_w"]:
        B, _, H, W = samples.shape
        Gh, Gw = self.grid_size

        assert H % Gh == 0 and W % Gw == 0, \
            f"Image (H,W)=({H},{W}) must be divisible by grid_size (Gh,Gw)=({Gh},{Gw})"

        tile_h, tile_w = H // Gh, W // Gw

        # (B,1,H,W) -> (B*Gh*Gw,1,tile_h,tile_w)
        tiles = (
            samples.unfold(2, tile_h, tile_h)
                   .unfold(3, tile_w, tile_w)
                   .reshape(-1, 1, tile_h, tile_w)
        )

        # tiles: (B*Gh*Gw, 1, tile_h, tile_w)
        if tiles.shape[-1] != 28 or tiles.shape[-2] != 28:
            tiles = F.interpolate(tiles, size=(28, 28), mode="nearest")

        logits: Float[Tensor, "n 10"] = classifier.forward(tiles)

        # top-2 indices
        idx = torch.topk(logits, k=2, dim=1).indices
        pred = idx[:, 0]

        # Replace zero predictions with second most probable number
        zero_mask = pred == 0
        pred[zero_mask] = idx[zero_mask, 1]

        pred = pred.reshape(B, Gh, Gw)
        return pred

    # def classify(
    #     self,
    #     pred: Integer[Tensor, "batch grid_h grid_w"],
    # ) -> tuple[
    #     Bool[Tensor, "batch"],
    #     dict[str, Shaped[Tensor, "batch"]],
    # ]:
    #     """
    #     pred 값은 {1..9}라고 가정.
    #     distance=0이면 행/열/서브그리드 모두 완벽.
    #     """
    #     batch_size, grid_size = pred.shape[:2]
    #     sub_grid_size = round(grid_size ** 0.5)

    #     if sub_grid_size * sub_grid_size != grid_size:
    #         raise ValueError(f"grid_size={grid_size} is not a perfect square (needed for Sudoku subgrids).")

    #     dtype, device = pred.dtype, pred.device

    #     # Shift [1,9] -> [0,8] for scatter indices
    #     pred0 = pred - 1

    #     ones = torch.ones((1,), dtype=dtype, device=device).expand_as(pred0)
    #     dist = torch.zeros((batch_size,), dtype=dtype, device=device)

    #     # rows (dim=1), cols (dim=2)
    #     for dim in (1, 2):
    #         cnt = torch.full_like(pred0, fill_value=-1)
    #         cnt.scatter_add_(dim=dim, index=pred0, src=ones)
    #         dist.add_(cnt.abs_().sum(dim=(1, 2)))

    #     # subgrids
    #     grids = (
    #         pred0.unfold(1, sub_grid_size, sub_grid_size)
    #              .unfold(2, sub_grid_size, sub_grid_size)
    #              .reshape(-1, grid_size, grid_size)
    #     )
    #     cnt = torch.full_like(grids, fill_value=-1)
    #     # 여기서는 subgrid 텐서의 "row"축 기준으로 세는 것과 같게 동작 (원본 코드 그대로 유지)
    #     cnt.scatter_add_(dim=1, index=grids, src=ones.reshape(-1, grid_size, grid_size))
    #     dist.add_(cnt.abs_().sum(dim=(1, 2)))

    #     labels = dist == 0
    #     return labels, {"distance": dist}

    def classify(
        self,
        pred: Integer[Tensor, "batch grid_h grid_w"],
    ) -> tuple[
        Bool[Tensor, "batch"],
        dict[str, Shaped[Tensor, "batch"]],
    ]:
        """
        pred 값은 {1..9}라고 가정.
        distance=0이면 행/열/서브그리드 모두 완벽.
        """
        batch_size, grid_size = pred.shape[:2]
        sub_grid_size = round(grid_size ** 0.5)

        if sub_grid_size * sub_grid_size != grid_size:
            raise ValueError(f"grid_size={grid_size} is not a perfect square (needed for Sudoku subgrids).")

        dtype, device = pred.dtype, pred.device

        # Shift [1,9] -> [0,8] for scatter indices
        pred0 = pred - 1  # (B,9,9) values 0..8

        ones = torch.ones((1,), dtype=dtype, device=device).expand_as(pred0)
        dist = torch.zeros((batch_size,), dtype=dtype, device=device)

        # rows (dim=2 over cols), cols (dim=1 over rows)  <-- 기존 로직 유지
        for dim in (1, 2):
            cnt = torch.full_like(pred0, fill_value=-1)
            cnt.scatter_add_(dim=dim, index=pred0, src=ones)
            dist.add_(cnt.abs_().sum(dim=(1, 2)))

        # ✅ subgrids (3x3 blocks) - FIX
        s = sub_grid_size  # 3 for 9x9
        B = batch_size
        N = grid_size  # 9

        # (B,9,9) -> (B,3,3,3,3) -> blocks: (B*9, 9)
        blocks = pred0.view(B, s, s, s, s).permute(0, 1, 3, 2, 4).reshape(B * (s * s), s * s)

        # count per block per digit: (B*9, 9), start at -1 to measure |count-1|
        cnt = torch.full((B * (s * s), N), -1, dtype=dtype, device=device)
        cnt.scatter_add_(dim=1, index=blocks, src=torch.ones_like(blocks, dtype=dtype, device=device))

        # block distance: sum over digits, then sum over 9 blocks -> (B,)
        block_dist = cnt.abs().sum(dim=1).view(B, s * s).sum(dim=1)
        dist.add_(block_dist)

        labels = dist == 0
        return labels, {"distance": dist}

    @torch.no_grad()
    def eval_images(
        self,
        samples: Float[Tensor, "batch 1 height width"],
    ) -> dict[str, Tensor]:
        """
        입력: [-1, 1] 범위로 정규화된 (B,1,H,W) 이미지 배치.
        출력:
          - labels: (B,) bool, 스도쿠 규칙 완벽히 만족 여부
          - distance: (B,) 규칙 위반 총량(0이면 완벽)
          - accuracy: () float, 배치 평균 정답률
          - discrete: (B, grid_h, grid_w) int, 타일 단위 MNIST 분류 결과(1~9 범위)
        """
        classifier = get_classifier(self.mnist_classifier_path, samples.device)
        discrete = self.discretize(classifier, samples)
        Gh, Gw = self.grid_size
        if (Gh, Gw) == (1, 1):
            B = samples.shape[0]
            return {
                "labels": torch.ones((B,), dtype=torch.bool, device=samples.device),
                "distance": torch.zeros((B,), dtype=torch.long, device=samples.device),
                "accuracy": torch.tensor(1.0, device=samples.device),
                "discrete": discrete,
            }

        labels, metrics = self.classify(discrete)
        return {
            "labels": labels,
            "distance": metrics["distance"],
            "accuracy": labels.float().mean(),
            "discrete": discrete,
        }
