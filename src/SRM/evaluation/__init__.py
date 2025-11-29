from .evaluation import Evaluation
from .mnist_grid_evaluation import MnistGridEvaluation, MnistGridEvaluationCfg
from .mnist_sudoku_evaluation import MnistSudokuEvaluation, MnistSudokuEvaluationCfg
from .sampling_evaluation import SamplingEvaluation, SamplingEvaluationCfg
from .counting_objects_evaluation import CountingObjectsEvaluation, CountingObjectsEvaluationCfg
from .even_pixels_evaluation import EvenPixelsEvaluation, EvenPixelsEvaluationCfg

from ..dataset import Dataset


EVALUATION = {
    "mnist_grid": MnistGridEvaluation,
    "mnist_sudoku": MnistSudokuEvaluation,
    "sampling": SamplingEvaluation,
    "counting_objects": CountingObjectsEvaluation,
    "even_pixels": EvenPixelsEvaluation,
}


EvaluationCfg = MnistGridEvaluationCfg | MnistSudokuEvaluationCfg | SamplingEvaluationCfg | CountingObjectsEvaluationCfg | EvenPixelsEvaluationCfg


def get_evaluation(
    cfg: EvaluationCfg,
    tag: str,
    dataset: Dataset,
    patch_size: int | None = None,
    patch_grid_shape: tuple[int, int] | None = None,
    deterministic: bool = False
) -> Evaluation:
    return EVALUATION[cfg.name](cfg, tag, dataset, patch_size, patch_grid_shape, deterministic)
