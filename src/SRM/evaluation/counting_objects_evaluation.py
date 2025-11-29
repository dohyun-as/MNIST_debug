from dataclasses import dataclass
from typing import Iterator, TypeVar, Literal, Dict

from jaxtyping import Bool, Float, Integer
import torch
from torch import Tensor

from ..dataset.dataset_counting_polygons.dataset_counting_polygons_base import (
    DatasetCountingPolygonsBase,
)
from ..misc.counting_objects_classifier import get_counting_objects_classifier
from ..model import Wrapper
from .types import EvaluationOutput
from .sampling_evaluation import (
    SamplingEvaluation,
    SamplingEvaluationCfg,
    BatchedSamplingExample,
)


@dataclass
class CountingObjectsEvaluationCfg(SamplingEvaluationCfg):
    name: Literal["counting_objects"] = "counting_objects"
    num_samples: int = 100


T = TypeVar("T", bound=CountingObjectsEvaluationCfg)


class CountingObjectsEvaluation(SamplingEvaluation[T]):
    def __init__(
        self,
        cfg: CountingObjectsEvaluationCfg,
        tag: str,
        dataset: DatasetCountingPolygonsBase,
        patch_size: int | None = None,
        patch_grid_shape: tuple[int, int] | None = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__(cfg, tag, dataset, patch_size, patch_grid_shape, deterministic)
        self.classifier_path = dataset.cfg.counting_objects_classifier_path
        self.classifier_model = dataset.cfg.counting_objects_classifier_model_base

    @staticmethod
    def are_ambiguous_numbers_consistent(
        numbers_label: Integer[Tensor, "batch 2"],
        num_polygons: Integer[Tensor, "batch"],
        num_vertices: Integer[Tensor, "batch"],
    ) -> Bool[Tensor, "batch"]:
        return torch.logical_or(
            torch.logical_and(
                numbers_label[:, 0] == num_polygons, numbers_label[:, 1] == num_vertices
            ),
            torch.logical_and(
                numbers_label[:, 0] == num_vertices, numbers_label[:, 1] == num_polygons
            ),
        )

    def get_vertices_counts(
        self, num_vertices: Integer[Tensor, "batch"], counts_range: tuple[int, int]
    ) -> Dict[str, float]:
        return {
            f"relative_vertex_count_{i}": (
                (num_vertices == i).sum() / num_vertices.shape[0]
            ).item()
            for i in range(*counts_range)
        }

    def get_polygons_counts(
        self, num_polygons: Integer[Tensor, "batch"], counts_range: tuple[int, int]
    ) -> Dict[str, float]:
        return {
            f"relative_polygons_count_{i}": (
                (num_polygons == i).sum() / num_polygons.shape[0]
            ).item()
            for i in range(*counts_range)
        }

    def is_class_label_consistent(
        self,
        class_label: Integer[Tensor, "batch"],
        num_polygons: Integer[Tensor, "batch"],
        num_vertices: Integer[Tensor, "batch"],
    ) -> Bool[Tensor, "batch"]:
        assert self.dataset.labeler is not None, "Labeler must be provided"

        generated_label = self.dataset.labeler.get_batch_labels(
            num_polygons, num_vertices
        )
        return class_label == generated_label

    @torch.no_grad()
    def get_metrics(
        self,
        samples: Float[Tensor, "batch 3 height width"],
        label: Integer[Tensor, "batch"] | None = None,
    ) -> dict[str, Float[Tensor, ""]]:
        classifier = get_counting_objects_classifier(
            model_path=self.classifier_path,
            model_base=self.classifier_model,
            device=samples.device,
            are_nums_on_images=self.dataset.cfg.are_nums_on_images,
        )

        outputs, confidences = classifier.predict(samples)

        if label is not None:
            consistency = self.is_class_label_consistent(
                label, outputs["num_polygons"], outputs["num_vertices"]
            )

        else:
            assert (
                self.dataset.cfg.are_nums_on_images
            ), "Label must be provided or numbers must be on images"
            consistency = self.are_ambiguous_numbers_consistent(
                outputs["num_polygons_vertices"],
                outputs["num_polygons"],
                outputs["num_vertices"],
            )

        metrics = {
            "are_numbers_consistent": consistency,
            "are_vertices_uniform": outputs["is_uniform"],
            **{f"{key}_confidence": value for key, value in confidences.items()},
        }
        metrics = {k: v.float().mean() for k, v in metrics.items()}
        
        vertex_value_range = (classifier.min_vertices, classifier.max_vertices + 1)
        polygon_value_range = (
            classifier.min_num_polygons,
            classifier.max_num_polygons + 1,
        )

        vertex_counts = self.get_vertices_counts(
            outputs["num_vertices"], vertex_value_range
        )
        polygon_counts = self.get_polygons_counts(
            outputs["num_polygons"], polygon_value_range
        )

        metrics.update(vertex_counts)
        metrics.update(polygon_counts)

        return metrics

    @torch.no_grad()
    def evaluate(
        self, model: Wrapper, batch: BatchedSamplingExample, return_sample: bool = True
    ) -> Iterator[EvaluationOutput]:
        label = batch.get("label", None)

        for res in super().evaluate(model, batch, return_sample):
            res["metrics"] = self.get_metrics(res["sample"]["sample"], label=label)
            yield res
