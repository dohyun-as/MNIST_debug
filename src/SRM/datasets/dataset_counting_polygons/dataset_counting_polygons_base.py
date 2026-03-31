from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from typing import Literal, Sequence, TypeVar
from jaxtyping import Float

from PIL import Image, ImageDraw
import numpy as np

from .font_cache import FontCache
from .labelers import get_labeler
from ..dataset import Dataset, DatasetCfg
from ..types import Example
from ...type_extensions import ConditioningCfg, Stage


@dataclass
class DatasetCountingPolygonsCfg(DatasetCfg):
    name: Literal["counting_polygons"] = "counting_polygons"
    labeler_name: Literal["explicit", "ambiguous"] | None = None
    are_nums_on_images: bool = False
    supersampling_image_size: Sequence[int] = (512, 512)
    min_vertices: int = 3
    max_vertices: int = 7
    font_name: str = "Roboto-Regular.ttf"
    mismatched_numbers: bool = False  # True only for the classifier
    allow_nonuinform_vertices: bool = (
        False  # Different vertex numbers for each polygon - only for the classifier
    )
    counting_objects_classifier_path: str = (
        ""  # Path to the counting objects classifier
    )
    counting_objects_classifier_model_base: Literal["resnet18", "resnet50"] = "resnet50"


T = TypeVar("T", bound=DatasetCountingPolygonsCfg)


class DatasetCountingPolygonsBase(Dataset[T], ABC):
    training_set_ratio: float = 0.95
    circle_images_per_num_circles: int = 100_000
    circle_num_variants: int = 9
    circle_positions_file_name: str = "circle_position_radius.npy"

    @abstractmethod
    def _get_base_image(self, base_image_idx) -> Image.Image:
        pass

    @property
    @abstractmethod
    def _num_available(self) -> int:
        pass

    @abstractmethod
    def _split_idx(self, idx) -> tuple[int, int, int | None]:
        pass

    def __init__(
        self,
        cfg: DatasetCountingPolygonsCfg,
        conditioning_cfg: ConditioningCfg,
        stage: Stage,
    ) -> None:
        super().__init__(cfg=cfg, conditioning_cfg=conditioning_cfg, stage=stage)

        if self.cfg.allow_nonuinform_vertices:
            assert (
                self.cfg.mismatched_numbers
            ), "Mismatched numbers must be enabled to allow nonuniform vertices"

        self.root_path = Path(self.cfg.root)
        self.min_circle_num = 3 if self.cfg.are_nums_on_images else 1
        self.circle_positions = self._load_circle_positions()

        self.font_cache = FontCache(self.root_path / "Roboto-Regular.ttf")
        self.labeler = (
            get_labeler(
                labeler_name=cfg.labeler_name,
                min_vertices=cfg.min_vertices,
                max_vertices=cfg.max_vertices,
            )
            if cfg.labeler_name
            else None
        )

        assert self.cfg.supersampling_image_size[0] >= self.cfg.image_shape[0]
        assert self.cfg.supersampling_image_size[1] >= self.cfg.image_shape[1]

        self.circle_xyr_scaling_factor = np.array(
            [
                self.cfg.supersampling_image_size[0],  # x
                self.cfg.supersampling_image_size[1],  # y
                min(self.cfg.supersampling_image_size),  # radius
            ]
        )[
            np.newaxis, ...
        ]  # shape: (1, 3)

    @property
    def is_deterministic(self) -> bool:
        return self.stage != "train"

    @property
    def _training_positions_per_circles_num(self) -> int:
        return int(self.training_set_ratio * self.circle_images_per_num_circles)

    def _load_circle_positions(
        self,
    ) -> dict[int, Float[np.ndarray, "num_images num_circles_per_image 3"]]:
        # The data is in the form of a numpy array with shape (num_circles, num_circles, 3)
        # where the last dimension is the x, y, and radius of the circle
        circle_pos_radius = np.load(
            self.root_path / self.circle_positions_file_name, allow_pickle=True
        ).item()

        possible_num_circle_nums = np.arange(
            self.min_circle_num, self.min_circle_num + self.circle_num_variants
        )  # 1-9 or 3-11

        return {
            num_circles: (
                data[: self._training_positions_per_circles_num]
                if self.stage == "train"
                else data[self._training_positions_per_circles_num :]
            )
            for num_circles, data in circle_pos_radius.items()
            if num_circles in possible_num_circle_nums
        }

    def split_circles_idx(self, idx) -> tuple[int, int]:
        num_circles_idx = (
            idx // self._num_positions_per_num_circles + self.min_circle_num
        )
        circles_image_idx = idx % self._num_positions_per_num_circles

        assert circles_image_idx < self._num_overlay_images
        return num_circles_idx, circles_image_idx

    def _get_circle_xyr(
        self, num_circles_idx: int, circles_image_idx: int
    ) -> Float[np.ndarray, "num_circles_per_image 3"]:
        image_circles = self.circle_positions[num_circles_idx][
            circles_image_idx
        ]  # shape: (num_circles, 3)

        return image_circles * self.circle_xyr_scaling_factor

    @abstractmethod
    def _get_color(
        self, rng: np.random.Generator | None, base_image: Image.Image
    ) -> str | tuple[int, int, int]:
        pass

    @staticmethod
    def _get_unit_polygon_vertices(
        points_on_circle: int, angle_offset: Float[np.ndarray, "num_polygons"]
    ) -> Float[np.ndarray, "num_polygons points_on_circle 2"]:
        base = np.arange(points_on_circle) / points_on_circle * 2 * np.pi
        angles = base[np.newaxis, ...] + np.expand_dims(angle_offset, 1)

        x = np.cos(angles)
        y = np.sin(angles)

        return np.stack([x, y], axis=-1)

    @staticmethod
    def random_choice(rng: np.random.Generator | None, *args, **kwargs):
        if rng is not None:
            return rng.choice(*args, **kwargs)
        return np.random.choice(*args, **kwargs)

    @staticmethod
    def random_integers(rng: np.random.Generator | None, *args, **kwargs):
        if rng is not None:
            return rng.integers(*args, **kwargs).item()
        return np.random.randint(*args, **kwargs)

    @staticmethod
    def random_uniform(rng: np.random.Generator | None, *args, **kwargs):
        if rng is not None:
            return rng.uniform(*args, **kwargs)
        return np.random.uniform(*args, **kwargs)

    def _get_overlay_image_w_label(
        self,
        num_circles_idx: int,
        circles_image_idx: int,
        full_idx: int,
        base_image: Image.Image,
    ) -> tuple[Image.Image, int | None]:
        numbers_label = None
        combined_label = None

        circle_xyr = self._get_circle_xyr(num_circles_idx, circles_image_idx)
        num_circles = circle_xyr.shape[0]

        num_polygons = num_circles - 2 if self.cfg.are_nums_on_images else num_circles

        rng = None
        if self.is_deterministic:
            rng = np.random.default_rng(full_idx)
        else:
            circle_xyr = circle_xyr[np.random.permutation(num_circles)]

        num_vertices = self.random_integers(
            rng, self.cfg.min_vertices, self.cfg.max_vertices + 1
        )
        angle_offset = self.random_uniform(
            rng, 0, 2 * np.pi, size=num_polygons
        )  # shape: (num_polygons,)

        polygons_xyr = circle_xyr[:num_polygons]
        color = self._get_color(rng, base_image)

        image = Image.new("RGBA", self.cfg.supersampling_image_size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)

        if self.cfg.are_nums_on_images:
            assert num_polygons == num_circles - 2
            numbers_xyr = circle_xyr[-2:]

            for (x, y, radius), number in zip(
                numbers_xyr, [num_polygons, num_vertices]
            ):
                center = tuple([x, y])
                font = self.font_cache.get_font(int(radius))
                draw.text(center, str(number), font=font, fill=color)

        if self.labeler is not None:
            numbers_label = self.labeler.get_label(num_polygons, num_vertices)

        num_vertex_bounds = [0, num_polygons]
        num_vertex_groups = 1
        num_vertices_in_groups = [num_vertices]
        if self.cfg.mismatched_numbers:
            is_uniform = True
            if self.cfg.allow_nonuinform_vertices and num_polygons > 1:
                valid_vertex_values = np.arange(
                    self.cfg.min_vertices, self.cfg.max_vertices + 1
                )
                max_distinct_vertices = min(num_polygons, len(valid_vertex_values))

                is_uniform = self.random_choice(rng, [True, False])
                num_vertex_groups = (
                    1
                    if is_uniform
                    else self.random_integers(rng, 2, max_distinct_vertices + 1)
                )
                num_vertices_in_groups = self.random_choice(
                    rng, valid_vertex_values, num_vertex_groups, replace=False
                )

            else:  # Just resample the number of vertices -- same for all polygons
                num_vertices = self.random_integers(
                    rng, self.cfg.min_vertices, self.cfg.max_vertices + 1
                )
                num_vertices_in_groups = [num_vertices]

            combined_label = {
                "num_polygons": num_polygons - 1,  # map 1-9 to 0-8 as class labels
                "num_vertices": num_vertices
                - self.cfg.min_vertices,  # map 3-7 to 0-4 as class labels
                "is_uniform": int(is_uniform),
            }

            if numbers_label:
                combined_label["numbers_label"] = numbers_label

            num_vertex_bounds = np.linspace(
                0, num_polygons, num_vertex_groups + 1
            ).astype(int)
            num_vertex_bounds[-1] = num_polygons

        for start, end, num_vertices_in_group in zip(
            num_vertex_bounds[:-1], num_vertex_bounds[1:], num_vertices_in_groups
        ):
            group_angle_offset = angle_offset[start:end]
            group_polygons_xyr = polygons_xyr[start:end]

            vertices = self._get_unit_polygon_vertices(
                num_vertices_in_group, group_angle_offset
            )
            vertices = vertices * group_polygons_xyr[
                :, 2, np.newaxis, np.newaxis
            ] + np.expand_dims(group_polygons_xyr[:, :2], 1)

            for polygon_vertices in vertices:
                polygon_vertices = [(x, y) for x, y in polygon_vertices]
                draw.polygon(polygon_vertices, fill=color)

        resized_image = image.resize(self.cfg.image_shape, resample=Image.BICUBIC)

        label = combined_label if combined_label else numbers_label
        return resized_image, label

    @property
    def num_classes(self) -> int | dict[str, int]:
        if not self.cfg.mismatched_numbers:
            return self.labeler.num_classes if self.labeler else 0

        class_counts = {
            "num_polygons": self.circle_num_variants,
            "num_vertices": self.cfg.max_vertices - self.cfg.min_vertices + 1,
        }

        if self.labeler:
            class_counts["numbers_label"] = self.labeler.num_classes

        if self.cfg.allow_nonuinform_vertices:
            class_counts["is_uniform"] = 2

        return class_counts

    def load(self, idx: int) -> Example:
        num_circles_idx, circles_image_idx, base_image_idx = self._split_idx(idx)

        base_image = self._get_base_image(base_image_idx)

        overlay_image, label = self._get_overlay_image_w_label(
            num_circles_idx, circles_image_idx, full_idx=idx, base_image=base_image
        )

        image = Image.alpha_composite(base_image, overlay_image)
        image = image.convert("RGB")

        return {"image": image} if label is None else {"image": image, "label": label}

    @property
    def _num_positions_per_num_circles(self) -> int:
        return (
            self._training_positions_per_circles_num
            if self.stage == "train"
            else self.circle_images_per_num_circles
            - self._training_positions_per_circles_num
        )

    @property
    def _num_overlay_images(self) -> int:
        """Calculate the number of overlay images based on the number of circle xyrs
        This doesn't include the colors and the number of vertices"""
        return (
            self._num_positions_per_num_circles * self.circle_num_variants
        )  # 10 possible number of circles
