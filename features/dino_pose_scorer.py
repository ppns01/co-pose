from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as F

from core.types import PreparedView
from features.dinov3_extractor import DINOFeatureResult
from features.observed_surface_features import (
    ObservedSurfaceFeatureResult,
)


@dataclass(frozen=True)
class DirectionalDINOScore:
    """
    한 방향의 depth-gated DINO 평가 결과.

    예:
        Reference observed points
        → Query camera
        → Query DINO feature와 비교

    appearance_loss:
        DINO cosine loss와 depth-match coverage를 결합한 값.
        0에 가까울수록 좋다.

    dino_loss:
        depth가 일치한 동일 surface에서 계산한 feature loss.
        (1 - cosine_similarity) / 2

    coverage:
        비교 가능한 point 중 depth가 일치한 point 비율.
    """

    source_view: str
    target_view: str
    available: bool

    appearance_loss: float | None
    dino_loss: float | None
    mean_cosine_similarity: float | None
    coverage: float | None

    source_point_count: int
    positive_depth_count: int
    inside_image_count: int
    target_mask_count: int
    target_depth_count: int
    zbuffer_point_count: int

    matched_surface_count: int
    occluded_count: int
    free_space_violation_count: int

    def __post_init__(self) -> None:
        if self.source_view not in ("reference", "query"):
            raise ValueError(
                f"지원하지 않는 source view입니다: "
                f"{self.source_view}"
            )

        if self.target_view not in ("reference", "query"):
            raise ValueError(
                f"지원하지 않는 target view입니다: "
                f"{self.target_view}"
            )

        if self.source_view == self.target_view:
            raise ValueError(
                "source_view와 target_view는 달라야 합니다."
            )

        count_values = (
            self.source_point_count,
            self.positive_depth_count,
            self.inside_image_count,
            self.target_mask_count,
            self.target_depth_count,
            self.zbuffer_point_count,
            self.matched_surface_count,
            self.occluded_count,
            self.free_space_violation_count,
        )

        if any(value < 0 for value in count_values):
            raise ValueError(
                "Point count는 0 이상이어야 합니다."
            )

        optional_values = (
            self.appearance_loss,
            self.dino_loss,
            self.mean_cosine_similarity,
            self.coverage,
        )

        for value in optional_values:
            if value is not None and not np.isfinite(value):
                raise ValueError(
                    "DINO score에 NaN 또는 Inf가 있습니다."
                )

        if self.available:
            if any(value is None for value in optional_values):
                raise ValueError(
                    "available=True이면 모든 score가 필요합니다."
                )

            assert self.appearance_loss is not None
            assert self.dino_loss is not None
            assert self.mean_cosine_similarity is not None
            assert self.coverage is not None

            if not 0.0 <= self.appearance_loss <= 1.0:
                raise ValueError(
                    "appearance_loss는 [0, 1] 범위여야 합니다."
                )

            if not 0.0 <= self.dino_loss <= 1.0:
                raise ValueError(
                    "dino_loss는 [0, 1] 범위여야 합니다."
                )

            if not -1.0 <= self.mean_cosine_similarity <= 1.0:
                raise ValueError(
                    "Cosine similarity는 [-1, 1] 범위여야 합니다."
                )

            if not 0.0 <= self.coverage <= 1.0:
                raise ValueError(
                    "coverage는 [0, 1] 범위여야 합니다."
                )


@dataclass(frozen=True)
class BidirectionalDINOScore:
    """
    Reference→Query와 Query→Reference의 통합 DINO 점수.

    combined_loss:
        유효한 각 방향의 matched surface point 수로
        가중 평균한 appearance loss.

    both_directions_available:
        양쪽 방향 모두 충분한 DINO support를 가졌는지 표시한다.
    """

    combined_loss: float | None
    available: bool
    both_directions_available: bool

    reference_to_query: DirectionalDINOScore
    query_to_reference: DirectionalDINOScore

    total_matched_surface_count: int

    def __post_init__(self) -> None:
        if self.total_matched_surface_count < 0:
            raise ValueError(
                "total_matched_surface_count는 "
                "0 이상이어야 합니다."
            )

        if self.available:
            if self.combined_loss is None:
                raise ValueError(
                    "available=True이면 combined_loss가 필요합니다."
                )

            if not np.isfinite(self.combined_loss):
                raise ValueError(
                    "combined_loss가 유한하지 않습니다."
                )

            if not 0.0 <= self.combined_loss <= 1.0:
                raise ValueError(
                    "combined_loss는 [0, 1] 범위여야 합니다."
                )
        elif self.combined_loss is not None:
            raise ValueError(
                "available=False이면 combined_loss는 None이어야 합니다."
            )


def _validate_pose(
    pose_target_from_source: NDArray[np.floating],
) -> NDArray[np.float32]:
    """
    Source camera 좌표를 target camera 좌표로 변환하는
    4×4 rigid transform을 검증한다.
    """

    pose = np.asarray(
        pose_target_from_source,
        dtype=np.float32,
    )

    if pose.shape != (4, 4):
        raise ValueError(
            "Relative pose shape은 (4, 4)이어야 합니다: "
            f"{pose.shape}"
        )

    if not np.isfinite(pose).all():
        raise ValueError(
            "Relative pose에 NaN 또는 Inf가 있습니다."
        )

    expected_last_row = np.array(
        [0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )

    if not np.allclose(
        pose[3],
        expected_last_row,
        atol=1e-5,
        rtol=0.0,
    ):
        raise ValueError(
            "Relative pose의 마지막 행이 올바르지 않습니다."
        )

    rotation = pose[:3, :3]

    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3, dtype=np.float32),
        atol=1e-3,
        rtol=0.0,
    ):
        raise ValueError(
            "Relative pose의 회전 행렬이 직교 행렬이 아닙니다."
        )

    determinant = float(
        np.linalg.det(rotation)
    )

    if not np.isclose(
        determinant,
        1.0,
        atol=1e-3,
        rtol=0.0,
    ):
        raise ValueError(
            "Relative pose 회전 행렬의 determinant가 "
            f"1이 아닙니다: {determinant}"
        )

    return np.ascontiguousarray(
        pose,
        dtype=np.float32,
    )


def _validate_direction_inputs(
    *,
    source_surface: ObservedSurfaceFeatureResult,
    target_view: PreparedView,
    target_dino: DINOFeatureResult,
) -> None:
    """한 방향의 DINO 평가 입력을 검증한다."""

    if source_surface.view_name == target_view.view.source.name:
        raise ValueError(
            "Source surface와 target view가 동일합니다."
        )

    if target_dino.view_name != target_view.view.source.name:
        raise ValueError(
            "Target view와 target DINO feature의 view가 다릅니다: "
            f"view={target_view.view.source.name}, "
            f"dino={target_dino.view_name}"
        )

    if (
        source_surface.features.shape[1]
        != target_dino.feature_map.shape[0]
    ):
        raise ValueError(
            "Source point feature와 target DINO feature의 "
            "채널 수가 다릅니다: "
            f"source={source_surface.features.shape[1]}, "
            f"target={target_dino.feature_map.shape[0]}"
        )

    image_size = target_view.view.rgb.shape[:2]

    if target_dino.original_image_size != image_size:
        raise ValueError(
            "Target RGB와 DINO metadata의 이미지 크기가 다릅니다: "
            f"rgb={image_size}, "
            f"dino={target_dino.original_image_size}"
        )


def _transform_points(
    points_source_m: NDArray[np.float32],
    pose_target_from_source: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Source camera point를 target camera frame으로 변환한다."""

    rotation = pose_target_from_source[:3, :3]
    translation = pose_target_from_source[:3, 3]

    points_target_m = (
        points_source_m @ rotation.T
        + translation
    )

    if not np.isfinite(points_target_m).all():
        raise ValueError(
            "변환된 point에 NaN 또는 Inf가 있습니다."
        )

    return np.ascontiguousarray(
        points_target_m,
        dtype=np.float32,
    )


def _project_points(
    points_target_m: NDArray[np.float32],
    camera_matrix: NDArray[np.float32],
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    """Target camera point를 target 이미지 평면에 투영한다."""

    fx = np.float32(camera_matrix[0, 0])
    fy = np.float32(camera_matrix[1, 1])
    cx = np.float32(camera_matrix[0, 2])
    cy = np.float32(camera_matrix[1, 2])

    x = points_target_m[:, 0]
    y = points_target_m[:, 1]
    z = points_target_m[:, 2]

    safe_z = np.maximum(
        z,
        np.float32(1e-8),
    )

    pixel_u = fx * x / safe_z + cx
    pixel_v = fy * y / safe_z + cy

    return (
        np.ascontiguousarray(
            pixel_u,
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            pixel_v,
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            z,
            dtype=np.float32,
        ),
    )


def _keep_nearest_point_per_pixel(
    *,
    point_indices: NDArray[np.int64],
    pixel_u_int: NDArray[np.int32],
    pixel_v_int: NDArray[np.int32],
    projected_depth_m: NDArray[np.float32],
    image_width: int,
) -> NDArray[np.int64]:
    """
    동일 target pixel에 여러 source point가 투영되면
    카메라에 가장 가까운 point만 유지한다.
    """

    linear_pixel_index = (
        pixel_v_int.astype(np.int64)
        * np.int64(image_width)
        + pixel_u_int.astype(np.int64)
    )

    # pixel index를 우선 정렬하고,
    # 같은 pixel에서는 depth가 작은 point를 우선 정렬한다.
    order = np.lexsort(
        (
            projected_depth_m,
            linear_pixel_index,
        )
    )

    sorted_pixels = linear_pixel_index[order]

    first_per_pixel = np.ones(
        sorted_pixels.shape[0],
        dtype=np.bool_,
    )

    if sorted_pixels.shape[0] > 1:
        first_per_pixel[1:] = (
            sorted_pixels[1:]
            != sorted_pixels[:-1]
        )

    selected_order = order[first_per_pixel]

    return np.ascontiguousarray(
        point_indices[selected_order],
        dtype=np.int64,
    )


def _build_sampling_grid(
    pixel_u: NDArray[np.float32],
    pixel_v: NDArray[np.float32],
    image_height: int,
    image_width: int,
) -> NDArray[np.float32]:
    """
    원본 이미지 pixel 좌표를 grid_sample 좌표로 변환한다.

    align_corners=False 규약을 사용한다.
    """

    normalized_x = (
        2.0
        * (pixel_u + 0.5)
        / float(image_width)
        - 1.0
    )

    normalized_y = (
        2.0
        * (pixel_v + 0.5)
        / float(image_height)
        - 1.0
    )

    sampling_grid = np.stack(
        (
            normalized_x,
            normalized_y,
        ),
        axis=1,
    )

    return np.ascontiguousarray(
        sampling_grid,
        dtype=np.float32,
    )


def _compute_cosine_similarity(
    *,
    source_features: NDArray[np.float16],
    target_feature_map: NDArray[np.float16],
    sampling_grid: NDArray[np.float32],
    device: str,
    chunk_size: int,
) -> NDArray[np.float32]:
    """
    Target DINO feature map을 bilinear sampling한 뒤
    source point feature와 cosine similarity를 계산한다.
    """

    if chunk_size < 1:
        raise ValueError(
            "chunk_size는 1 이상이어야 합니다."
        )

    torch_device = torch.device(device)

    if torch_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch에서 CUDA를 사용할 수 없습니다."
            )

        torch.cuda.set_device(
            torch_device
        )

    feature_map_tensor = torch.from_numpy(
        np.asarray(
            target_feature_map,
            dtype=np.float32,
        )
    ).unsqueeze(0)

    feature_map_tensor = feature_map_tensor.to(
        device=torch_device,
        dtype=torch.float32,
        non_blocking=True,
    )

    point_count = source_features.shape[0]

    similarities = np.empty(
        point_count,
        dtype=np.float32,
    )

    with torch.inference_mode():
        for start_index in range(
            0,
            point_count,
            chunk_size,
        ):
            end_index = min(
                start_index + chunk_size,
                point_count,
            )

            source_chunk = torch.from_numpy(
                np.asarray(
                    source_features[
                        start_index:end_index
                    ],
                    dtype=np.float32,
                )
            ).to(
                device=torch_device,
                dtype=torch.float32,
                non_blocking=True,
            )

            source_chunk = F.normalize(
                source_chunk,
                p=2,
                dim=1,
                eps=1e-6,
            )

            grid_chunk = torch.from_numpy(
                sampling_grid[
                    start_index:end_index
                ]
            ).view(
                1,
                end_index - start_index,
                1,
                2,
            ).to(
                device=torch_device,
                dtype=torch.float32,
                non_blocking=True,
            )

            sampled_target = F.grid_sample(
                input=feature_map_tensor,
                grid=grid_chunk,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )

            sampled_target = (
                sampled_target[0, :, :, 0]
                .transpose(0, 1)
                .contiguous()
            )

            sampled_target = F.normalize(
                sampled_target,
                p=2,
                dim=1,
                eps=1e-6,
            )

            similarity_chunk = torch.sum(
                source_chunk * sampled_target,
                dim=1,
            )

            similarities[
                start_index:end_index
            ] = (
                similarity_chunk
                .clamp(-1.0, 1.0)
                .to(
                    device="cpu",
                    dtype=torch.float32,
                )
                .numpy()
            )

            del source_chunk
            del grid_chunk
            del sampled_target
            del similarity_chunk

    del feature_map_tensor

    if torch_device.type == "cuda":
        torch.cuda.empty_cache()

    return np.ascontiguousarray(
        similarities,
        dtype=np.float32,
    )


def score_directional_dino(
    *,
    source_surface: ObservedSurfaceFeatureResult,
    target_view: PreparedView,
    target_dino: DINOFeatureResult,
    pose_target_from_source: NDArray[np.floating],
    depth_absolute_tolerance_m: float = 0.005,
    depth_relative_tolerance: float = 0.02,
    minimum_matched_points: int = 50,
    minimum_coverage: float = 0.05,
    coverage_weight: float = 0.25,
    device: str = "cuda:0",
    feature_chunk_size: int = 8192,
) -> DirectionalDINOScore:
    """
    Source observed surface를 target view로 투영하고,
    target depth와 일치하는 surface에서만 DINO를 비교한다.

    Depth 분류
    ----------
    abs(projected_z - observed_z) <= tolerance
        동일 surface: DINO 비교

    projected_z > observed_z + tolerance
        관측 surface 뒤쪽: occluded로 제외

    projected_z < observed_z - tolerance
        관측 surface 앞쪽: free-space violation
    """

    _validate_direction_inputs(
        source_surface=source_surface,
        target_view=target_view,
        target_dino=target_dino,
    )

    pose = _validate_pose(
        pose_target_from_source
    )

    if depth_absolute_tolerance_m < 0.0:
        raise ValueError(
            "depth_absolute_tolerance_m은 "
            "0 이상이어야 합니다."
        )

    if depth_relative_tolerance < 0.0:
        raise ValueError(
            "depth_relative_tolerance은 "
            "0 이상이어야 합니다."
        )

    if minimum_matched_points < 1:
        raise ValueError(
            "minimum_matched_points는 1 이상이어야 합니다."
        )

    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError(
            "minimum_coverage는 [0, 1] 범위여야 합니다."
        )

    if coverage_weight < 0.0:
        raise ValueError(
            "coverage_weight는 0 이상이어야 합니다."
        )

    source_points = source_surface.points_camera_m
    source_features = source_surface.features

    source_point_count = source_points.shape[0]

    transformed_points = _transform_points(
        points_source_m=source_points,
        pose_target_from_source=pose,
    )

    pixel_u, pixel_v, projected_z = _project_points(
        points_target_m=transformed_points,
        camera_matrix=target_view.view.camera_matrix,
    )

    positive_depth_mask = projected_z > 0.0
    positive_depth_count = int(
        np.count_nonzero(positive_depth_mask)
    )

    image_height, image_width = (
        target_view.view.rgb.shape[:2]
    )

    inside_image_mask = (
        positive_depth_mask
        & (pixel_u >= 0.0)
        & (pixel_u <= float(image_width - 1))
        & (pixel_v >= 0.0)
        & (pixel_v <= float(image_height - 1))
    )

    inside_indices = np.flatnonzero(
        inside_image_mask
    ).astype(
        np.int64,
        copy=False,
    )

    inside_image_count = int(
        inside_indices.size
    )

    if inside_indices.size == 0:
        return DirectionalDINOScore(
            source_view=source_surface.view_name,
            target_view=target_view.view.source.name,
            available=False,
            appearance_loss=None,
            dino_loss=None,
            mean_cosine_similarity=None,
            coverage=None,
            source_point_count=source_point_count,
            positive_depth_count=positive_depth_count,
            inside_image_count=0,
            target_mask_count=0,
            target_depth_count=0,
            zbuffer_point_count=0,
            matched_surface_count=0,
            occluded_count=0,
            free_space_violation_count=0,
        )

    pixel_u_inside = pixel_u[inside_indices]
    pixel_v_inside = pixel_v[inside_indices]

    pixel_u_int = np.clip(
        np.rint(pixel_u_inside),
        0,
        image_width - 1,
    ).astype(
        np.int32,
        copy=False,
    )

    pixel_v_int = np.clip(
        np.rint(pixel_v_inside),
        0,
        image_height - 1,
    ).astype(
        np.int32,
        copy=False,
    )

    target_mask_values = (
        target_view.segmentation.mask_bool[
            pixel_v_int,
            pixel_u_int,
        ]
    )

    mask_indices = inside_indices[
        target_mask_values
    ]

    target_mask_count = int(
        mask_indices.size
    )

    if mask_indices.size == 0:
        return DirectionalDINOScore(
            source_view=source_surface.view_name,
            target_view=target_view.view.source.name,
            available=False,
            appearance_loss=None,
            dino_loss=None,
            mean_cosine_similarity=None,
            coverage=None,
            source_point_count=source_point_count,
            positive_depth_count=positive_depth_count,
            inside_image_count=inside_image_count,
            target_mask_count=0,
            target_depth_count=0,
            zbuffer_point_count=0,
            matched_surface_count=0,
            occluded_count=0,
            free_space_violation_count=0,
        )

    target_pixel_u = np.clip(
        np.rint(pixel_u[mask_indices]),
        0,
        image_width - 1,
    ).astype(
        np.int32,
        copy=False,
    )

    target_pixel_v = np.clip(
        np.rint(pixel_v[mask_indices]),
        0,
        image_height - 1,
    ).astype(
        np.int32,
        copy=False,
    )

    observed_target_depth = (
        target_view.view.depth_m[
            target_pixel_v,
            target_pixel_u,
        ]
    )

    valid_target_depth = (
        np.isfinite(observed_target_depth)
        & (observed_target_depth > 0.0)
    )

    depth_indices = mask_indices[
        valid_target_depth
    ]

    target_depth_count = int(
        depth_indices.size
    )

    if depth_indices.size == 0:
        return DirectionalDINOScore(
            source_view=source_surface.view_name,
            target_view=target_view.view.source.name,
            available=False,
            appearance_loss=None,
            dino_loss=None,
            mean_cosine_similarity=None,
            coverage=None,
            source_point_count=source_point_count,
            positive_depth_count=positive_depth_count,
            inside_image_count=inside_image_count,
            target_mask_count=target_mask_count,
            target_depth_count=0,
            zbuffer_point_count=0,
            matched_surface_count=0,
            occluded_count=0,
            free_space_violation_count=0,
        )

    depth_pixel_u = np.clip(
        np.rint(pixel_u[depth_indices]),
        0,
        image_width - 1,
    ).astype(
        np.int32,
        copy=False,
    )

    depth_pixel_v = np.clip(
        np.rint(pixel_v[depth_indices]),
        0,
        image_height - 1,
    ).astype(
        np.int32,
        copy=False,
    )

    zbuffer_indices = _keep_nearest_point_per_pixel(
        point_indices=depth_indices,
        pixel_u_int=depth_pixel_u,
        pixel_v_int=depth_pixel_v,
        projected_depth_m=projected_z[depth_indices],
        image_width=image_width,
    )

    zbuffer_point_count = int(
        zbuffer_indices.size
    )

    zbuffer_pixel_u = np.clip(
        np.rint(pixel_u[zbuffer_indices]),
        0,
        image_width - 1,
    ).astype(
        np.int32,
        copy=False,
    )

    zbuffer_pixel_v = np.clip(
        np.rint(pixel_v[zbuffer_indices]),
        0,
        image_height - 1,
    ).astype(
        np.int32,
        copy=False,
    )

    target_depth_values = (
        target_view.view.depth_m[
            zbuffer_pixel_v,
            zbuffer_pixel_u,
        ]
    )

    projected_depth_values = (
        projected_z[zbuffer_indices]
    )

    depth_tolerance = np.maximum(
        np.float32(depth_absolute_tolerance_m),
        target_depth_values
        * np.float32(depth_relative_tolerance),
    )

    depth_difference = (
        projected_depth_values
        - target_depth_values
    )

    matched_mask = (
        np.abs(depth_difference)
        <= depth_tolerance
    )

    occluded_mask = (
        depth_difference > depth_tolerance
    )

    free_space_mask = (
        depth_difference < -depth_tolerance
    )

    matched_indices = zbuffer_indices[
        matched_mask
    ]

    matched_surface_count = int(
        matched_indices.size
    )

    occluded_count = int(
        np.count_nonzero(occluded_mask)
    )

    free_space_violation_count = int(
        np.count_nonzero(free_space_mask)
    )

    # Occluded point는 DINO coverage 분모에서 제외한다.
    comparable_count = (
        matched_surface_count
        + free_space_violation_count
    )

    if comparable_count > 0:
        coverage = (
            matched_surface_count
            / comparable_count
        )
    else:
        coverage = 0.0

    score_available = (
        matched_surface_count >= minimum_matched_points
        and coverage >= minimum_coverage
    )

    if not score_available:
        return DirectionalDINOScore(
            source_view=source_surface.view_name,
            target_view=target_view.view.source.name,
            available=False,
            appearance_loss=None,
            dino_loss=None,
            mean_cosine_similarity=None,
            coverage=None,
            source_point_count=source_point_count,
            positive_depth_count=positive_depth_count,
            inside_image_count=inside_image_count,
            target_mask_count=target_mask_count,
            target_depth_count=target_depth_count,
            zbuffer_point_count=zbuffer_point_count,
            matched_surface_count=matched_surface_count,
            occluded_count=occluded_count,
            free_space_violation_count=(
                free_space_violation_count
            ),
        )

    matched_pixel_u = pixel_u[
        matched_indices
    ]

    matched_pixel_v = pixel_v[
        matched_indices
    ]

    sampling_grid = _build_sampling_grid(
        pixel_u=matched_pixel_u,
        pixel_v=matched_pixel_v,
        image_height=image_height,
        image_width=image_width,
    )

    cosine_similarities = _compute_cosine_similarity(
        source_features=source_features[
            matched_indices
        ],
        target_feature_map=target_dino.feature_map,
        sampling_grid=sampling_grid,
        device=device,
        chunk_size=feature_chunk_size,
    )

    mean_cosine_similarity = float(
        np.mean(cosine_similarities)
    )

    dino_loss = float(
        np.clip(
            (
                1.0
                - mean_cosine_similarity
            )
            * 0.5,
            0.0,
            1.0,
        )
    )

    coverage_loss = 1.0 - coverage

    appearance_loss = float(
        (
            dino_loss
            + coverage_weight * coverage_loss
        )
        / (
            1.0 + coverage_weight
        )
    )

    return DirectionalDINOScore(
        source_view=source_surface.view_name,
        target_view=target_view.view.source.name,
        available=True,
        appearance_loss=appearance_loss,
        dino_loss=dino_loss,
        mean_cosine_similarity=(
            mean_cosine_similarity
        ),
        coverage=float(coverage),
        source_point_count=source_point_count,
        positive_depth_count=positive_depth_count,
        inside_image_count=inside_image_count,
        target_mask_count=target_mask_count,
        target_depth_count=target_depth_count,
        zbuffer_point_count=zbuffer_point_count,
        matched_surface_count=matched_surface_count,
        occluded_count=occluded_count,
        free_space_violation_count=(
            free_space_violation_count
        ),
    )


def score_bidirectional_dino(
    *,
    relative_pose_query_from_reference: NDArray[np.floating],
    reference_surface: ObservedSurfaceFeatureResult,
    query_surface: ObservedSurfaceFeatureResult,
    reference_view: PreparedView,
    query_view: PreparedView,
    reference_dino: DINOFeatureResult,
    query_dino: DINOFeatureResult,
    depth_absolute_tolerance_m: float = 0.005,
    depth_relative_tolerance: float = 0.02,
    minimum_matched_points: int = 50,
    minimum_coverage: float = 0.05,
    coverage_weight: float = 0.25,
    device: str = "cuda:0",
    feature_chunk_size: int = 8192,
) -> BidirectionalDINOScore:
    """
    상대 pose 후보를 양방향 depth-gated DINO로 평가한다.

    입력 pose 규약:
        relative_pose_query_from_reference
        = T_query_camera_from_reference_camera
    """

    relative_pose = _validate_pose(
        relative_pose_query_from_reference
    )

    inverse_pose = np.linalg.inv(
        relative_pose.astype(np.float64)
    ).astype(
        np.float32
    )

    reference_to_query = score_directional_dino(
        source_surface=reference_surface,
        target_view=query_view,
        target_dino=query_dino,
        pose_target_from_source=relative_pose,
        depth_absolute_tolerance_m=(
            depth_absolute_tolerance_m
        ),
        depth_relative_tolerance=(
            depth_relative_tolerance
        ),
        minimum_matched_points=(
            minimum_matched_points
        ),
        minimum_coverage=minimum_coverage,
        coverage_weight=coverage_weight,
        device=device,
        feature_chunk_size=feature_chunk_size,
    )

    query_to_reference = score_directional_dino(
        source_surface=query_surface,
        target_view=reference_view,
        target_dino=reference_dino,
        pose_target_from_source=inverse_pose,
        depth_absolute_tolerance_m=(
            depth_absolute_tolerance_m
        ),
        depth_relative_tolerance=(
            depth_relative_tolerance
        ),
        minimum_matched_points=(
            minimum_matched_points
        ),
        minimum_coverage=minimum_coverage,
        coverage_weight=coverage_weight,
        device=device,
        feature_chunk_size=feature_chunk_size,
    )

    valid_directions = [
        result
        for result in (
            reference_to_query,
            query_to_reference,
        )
        if result.available
    ]

    total_matched_surface_count = sum(
        result.matched_surface_count
        for result in valid_directions
    )

    if not valid_directions:
        return BidirectionalDINOScore(
            combined_loss=None,
            available=False,
            both_directions_available=False,
            reference_to_query=reference_to_query,
            query_to_reference=query_to_reference,
            total_matched_surface_count=0,
        )

    weighted_loss_sum = 0.0
    weight_sum = 0

    for result in valid_directions:
        assert result.appearance_loss is not None

        direction_weight = (
            result.matched_surface_count
        )

        weighted_loss_sum += (
            result.appearance_loss
            * direction_weight
        )

        weight_sum += direction_weight

    combined_loss = (
        weighted_loss_sum / weight_sum
    )

    return BidirectionalDINOScore(
        combined_loss=float(combined_loss),
        available=True,
        both_directions_available=(
            len(valid_directions) == 2
        ),
        reference_to_query=reference_to_query,
        query_to_reference=query_to_reference,
        total_matched_surface_count=(
            total_matched_surface_count
        ),
    )


def save_bidirectional_dino_score(
    result: BidirectionalDINOScore,
    output_path: Path,
    *,
    candidate_name: str,
    relative_pose_query_from_reference: NDArray[np.floating],
) -> Path:
    """양방향 DINO 평가 결과를 JSON으로 저장한다."""

    output_path = (
        Path(output_path)
        .expanduser()
        .resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    relative_pose = _validate_pose(
        relative_pose_query_from_reference
    )

    metadata: dict[str, Any] = {
        "candidate_name": candidate_name,
        "pose_convention": (
            "T_query_camera_from_reference_camera"
        ),
        "relative_pose_query_from_reference": (
            relative_pose.tolist()
        ),
        "available": result.available,
        "both_directions_available": (
            result.both_directions_available
        ),
        "combined_loss": result.combined_loss,
        "total_matched_surface_count": (
            result.total_matched_surface_count
        ),
        "reference_to_query": asdict(
            result.reference_to_query
        ),
        "query_to_reference": asdict(
            result.query_to_reference
        ),
        "lower_is_better": True,
    }

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    if not output_path.is_file():
        raise FileNotFoundError(
            "DINO pose score가 저장되지 않았습니다: "
            f"{output_path}"
        )

    return output_path