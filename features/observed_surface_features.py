from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.types import PreparedView, ViewName
from features.dinov3_extractor import DINOFeatureResult


@dataclass(frozen=True)
class ObservedSurfaceFeatureResult:
    """
    RGB-D 관측에서 생성한 가시 표면 point와 DINO feature.

    points_camera_m:
        Camera 좌표계의 3D point.
        shape=(N, 3), 단위=meter.

    pixel_coordinates_xy:
        원본 RGB 영상의 pixel 좌표.
        shape=(N, 2), 순서=(x, y).

    features:
        각 3D point 위치에서 추출한 DINO feature.
        shape=(N, C), L2 정규화.
    """

    view_name: ViewName

    points_camera_m: NDArray[np.float32]
    pixel_coordinates_xy: NDArray[np.float32]
    features: NDArray[np.float32]

    image_height: int
    image_width: int

    feature_height: int
    feature_width: int

    points_path: Path
    pixels_path: Path
    features_path: Path
    metadata_path: Path

    def __post_init__(self) -> None:
        if self.view_name not in (
            "reference",
            "query",
        ):
            raise ValueError(
                "지원하지 않는 view_name입니다: "
                f"{self.view_name}"
            )

        if (
            self.points_camera_m.ndim != 2
            or self.points_camera_m.shape[1] != 3
        ):
            raise ValueError(
                "points_camera_m shape은 (N, 3)이어야 합니다: "
                f"{self.points_camera_m.shape}"
            )

        if self.points_camera_m.dtype != np.float32:
            raise TypeError(
                "points_camera_m dtype은 float32이어야 합니다: "
                f"{self.points_camera_m.dtype}"
            )

        if (
            self.pixel_coordinates_xy.ndim != 2
            or self.pixel_coordinates_xy.shape[1] != 2
        ):
            raise ValueError(
                "pixel_coordinates_xy shape은 "
                "(N, 2)이어야 합니다: "
                f"{self.pixel_coordinates_xy.shape}"
            )

        if self.pixel_coordinates_xy.dtype != np.float32:
            raise TypeError(
                "pixel_coordinates_xy dtype은 "
                "float32이어야 합니다: "
                f"{self.pixel_coordinates_xy.dtype}"
            )

        if self.features.ndim != 2:
            raise ValueError(
                "features shape은 (N, C)이어야 합니다: "
                f"{self.features.shape}"
            )

        if self.features.dtype != np.float32:
            raise TypeError(
                "features dtype은 float32이어야 합니다: "
                f"{self.features.dtype}"
            )

        point_count = self.points_camera_m.shape[0]

        if point_count == 0:
            raise ValueError(
                "Observed surface point가 없습니다."
            )

        if self.pixel_coordinates_xy.shape[0] != point_count:
            raise ValueError(
                "3D point와 pixel 개수가 다릅니다: "
                f"points={point_count}, "
                f"pixels={self.pixel_coordinates_xy.shape[0]}"
            )

        if self.features.shape[0] != point_count:
            raise ValueError(
                "3D point와 DINO feature 개수가 다릅니다: "
                f"points={point_count}, "
                f"features={self.features.shape[0]}"
            )

        if not np.isfinite(
            self.points_camera_m
        ).all():
            raise ValueError(
                "points_camera_m에 NaN 또는 Inf가 있습니다."
            )

        if not np.isfinite(
            self.pixel_coordinates_xy
        ).all():
            raise ValueError(
                "pixel_coordinates_xy에 NaN 또는 Inf가 있습니다."
            )

        if not np.isfinite(self.features).all():
            raise ValueError(
                "features에 NaN 또는 Inf가 있습니다."
            )

        if np.any(
            self.points_camera_m[:, 2] <= 0.0
        ):
            raise ValueError(
                "Camera point의 z 값은 양수여야 합니다."
            )

        if self.image_height < 1 or self.image_width < 1:
            raise ValueError(
                "원본 영상 크기가 올바르지 않습니다."
            )

        if self.feature_height < 1 or self.feature_width < 1:
            raise ValueError(
                "DINO feature map 크기가 올바르지 않습니다."
            )

        for path in (
            self.points_path,
            self.pixels_path,
            self.features_path,
            self.metadata_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Observed surface 출력 파일이 없습니다: {path}"
                )

    @property
    def point_count(self) -> int:
        return int(
            self.points_camera_m.shape[0]
        )

    @property
    def feature_dimension(self) -> int:
        return int(
            self.features.shape[1]
        )

    # 기존 또는 후속 코드와의 호환용 별칭
    @property
    def points_cam_m(
        self,
    ) -> NDArray[np.float32]:
        return self.points_camera_m

    @property
    def camera_points_m(
        self,
    ) -> NDArray[np.float32]:
        return self.points_camera_m

    @property
    def pixels_xy(
        self,
    ) -> NDArray[np.float32]:
        return self.pixel_coordinates_xy

    @property
    def dino_features(
        self,
    ) -> NDArray[np.float32]:
        return self.features

    @property
    def feature_vectors(
        self,
    ) -> NDArray[np.float32]:
        return self.features


def _resolve_dino_feature_map(
    dino_result: DINOFeatureResult,
) -> NDArray[np.float32]:
    """
    DINOFeatureResult에서 dense feature map을 찾는다.

    지원 attribute:
        feature_map
        dense_features
        feature_tensor
        features

    지원 파일 attribute:
        feature_path
        feature_map_path
        features_path
    """

    array_attribute_names = (
        "feature_map",
        "dense_features",
        "feature_tensor",
        "features",
    )

    raw_feature_map: Any | None = None

    for attribute_name in array_attribute_names:
        candidate = getattr(
            dino_result,
            attribute_name,
            None,
        )

        if candidate is None:
            continue

        if isinstance(candidate, np.ndarray):
            raw_feature_map = candidate
            break

        # torch.Tensor를 직접 import하지 않고 처리
        if (
            hasattr(candidate, "detach")
            and hasattr(candidate, "cpu")
            and hasattr(candidate, "numpy")
        ):
            raw_feature_map = (
                candidate.detach().cpu().numpy()
            )
            break

    if raw_feature_map is None:
        path_attribute_names = (
            "feature_path",
            "feature_map_path",
            "features_path",
        )

        for attribute_name in path_attribute_names:
            candidate_path = getattr(
                dino_result,
                attribute_name,
                None,
            )

            if candidate_path is None:
                continue

            resolved_path = (
                Path(candidate_path)
                .expanduser()
                .resolve()
            )

            if resolved_path.is_file():
                raw_feature_map = np.load(
                    resolved_path,
                    allow_pickle=False,
                )
                break

    if raw_feature_map is None:
        raise AttributeError(
            "DINOFeatureResult에서 dense feature map을 "
            "찾지 못했습니다. 지원 필드: "
            "feature_map, dense_features, feature_tensor, "
            "features, feature_path, feature_map_path, "
            "features_path"
        )

    feature_map = np.asarray(
        raw_feature_map
    )

    if (
        feature_map.ndim == 4
        and feature_map.shape[0] == 1
    ):
        feature_map = feature_map[0]

    if feature_map.ndim != 3:
        raise ValueError(
            "DINO dense feature map shape은 "
            "(C,H,W) 또는 (H,W,C)이어야 합니다: "
            f"{feature_map.shape}"
        )

    # DINO 출력은 일반적으로 C×H×W이다.
    if (
        feature_map.shape[0] >= 16
        and feature_map.shape[1] >= 2
        and feature_map.shape[2] >= 2
    ):
        channel_first = feature_map

    elif (
        feature_map.shape[2] >= 16
        and feature_map.shape[0] >= 2
        and feature_map.shape[1] >= 2
    ):
        channel_first = np.transpose(
            feature_map,
            (2, 0, 1),
        )

    else:
        raise ValueError(
            "DINO feature channel 축을 판별하지 못했습니다: "
            f"{feature_map.shape}"
        )

    channel_first = np.asarray(
        channel_first,
        dtype=np.float32,
    )

    if not np.isfinite(channel_first).all():
        raise ValueError(
            "DINO feature map에 NaN 또는 Inf가 있습니다."
        )

    return np.ascontiguousarray(
        channel_first,
        dtype=np.float32,
    )


def _backproject_masked_depth(
    prepared_view: PreparedView,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
]:
    """
    Mask 내부의 유효 depth를 camera 좌표계로 역투영한다.
    """

    mask_bool = np.asarray(
        prepared_view.segmentation.mask_bool,
        dtype=np.bool_,
    )

    depth_m = np.asarray(
        prepared_view.masked_depth_m,
        dtype=np.float32,
    )

    camera_matrix = np.asarray(
        prepared_view.view.camera_matrix,
        dtype=np.float32,
    )

    if mask_bool.shape != depth_m.shape:
        raise ValueError(
            "Mask와 depth 해상도가 다릅니다: "
            f"mask={mask_bool.shape}, depth={depth_m.shape}"
        )

    if camera_matrix.shape != (3, 3):
        raise ValueError(
            "Camera matrix shape은 (3, 3)이어야 합니다."
        )

    valid_mask = (
        mask_bool
        & np.isfinite(depth_m)
        & (depth_m > 0.0)
    )

    ys, xs = np.nonzero(valid_mask)

    if xs.size == 0:
        raise ValueError(
            "Mask 내부에 유효한 depth point가 없습니다."
        )

    z = depth_m[ys, xs].astype(
        np.float32,
        copy=False,
    )

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(
            f"유효하지 않은 focal length입니다: fx={fx}, fy={fy}"
        )

    x_camera = (
        (xs.astype(np.float32) - cx)
        * z
        / fx
    )

    y_camera = (
        (ys.astype(np.float32) - cy)
        * z
        / fy
    )

    points_camera_m = np.stack(
        (
            x_camera,
            y_camera,
            z,
        ),
        axis=1,
    ).astype(
        np.float32,
        copy=False,
    )

    pixel_coordinates_xy = np.stack(
        (
            xs.astype(np.float32),
            ys.astype(np.float32),
        ),
        axis=1,
    )

    return (
        np.ascontiguousarray(
            points_camera_m,
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            pixel_coordinates_xy,
            dtype=np.float32,
        ),
    )


def _subsample_points(
    *,
    points_camera_m: NDArray[np.float32],
    pixel_coordinates_xy: NDArray[np.float32],
    maximum_point_count: int,
    random_seed: int,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
]:
    """Point 수를 재현 가능한 방식으로 제한한다."""

    if maximum_point_count < 1:
        raise ValueError(
            "maximum_point_count는 1 이상이어야 합니다."
        )

    point_count = points_camera_m.shape[0]

    if point_count <= maximum_point_count:
        return (
            points_camera_m,
            pixel_coordinates_xy,
        )

    random_generator = np.random.default_rng(
        random_seed
    )

    selected_indices = random_generator.choice(
        point_count,
        size=maximum_point_count,
        replace=False,
    )

    selected_indices.sort()

    return (
        np.ascontiguousarray(
            points_camera_m[selected_indices],
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            pixel_coordinates_xy[selected_indices],
            dtype=np.float32,
        ),
    )


def _sample_feature_map(
    *,
    feature_map: NDArray[np.float32],
    pixel_coordinates_xy: NDArray[np.float32],
    image_height: int,
    image_width: int,
    device: str,
    chunk_size: int,
) -> NDArray[np.float32]:
    """
    원본 RGB pixel 위치에서 DINO feature map을 bilinear sampling한다.

    align_corners=False 좌표 규약을 사용한다.
    """

    if chunk_size < 1:
        raise ValueError(
            "chunk_size는 1 이상이어야 합니다."
        )

    if image_height < 1 or image_width < 1:
        raise ValueError(
            "원본 영상 해상도가 올바르지 않습니다."
        )

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise ImportError(
            "DINO surface feature sampling에는 PyTorch가 필요합니다."
        ) from error

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device가 요청되었지만 "
                "PyTorch CUDA를 사용할 수 없습니다."
            )

    torch_device = torch.device(device)

    feature_tensor = torch.as_tensor(
        feature_map,
        device=torch_device,
        dtype=torch.float32,
    ).unsqueeze(0)

    sampled_chunks: list[
        NDArray[np.float32]
    ] = []

    point_count = pixel_coordinates_xy.shape[0]

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

            pixels_chunk = torch.as_tensor(
                pixel_coordinates_xy[
                    start_index:end_index
                ],
                device=torch_device,
                dtype=torch.float32,
            )

            normalized_x = (
                (
                    pixels_chunk[:, 0] + 0.5
                )
                / float(image_width)
                * 2.0
                - 1.0
            )

            normalized_y = (
                (
                    pixels_chunk[:, 1] + 0.5
                )
                / float(image_height)
                * 2.0
                - 1.0
            )

            sampling_grid = torch.stack(
                (
                    normalized_x,
                    normalized_y,
                ),
                dim=1,
            ).view(
                1,
                1,
                -1,
                2,
            )

            sampled = functional.grid_sample(
                feature_tensor,
                sampling_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )

            # (1,C,1,N) → (N,C)
            sampled = (
                sampled[0, :, 0, :]
                .transpose(0, 1)
            )

            sampled = functional.normalize(
                sampled,
                p=2.0,
                dim=1,
                eps=1e-8,
            )

            sampled_chunks.append(
                sampled.to(
                    device="cpu",
                    dtype=torch.float32,
                ).numpy()
            )

    sampled_features = np.concatenate(
        sampled_chunks,
        axis=0,
    )

    return np.ascontiguousarray(
        sampled_features,
        dtype=np.float32,
    )


def build_observed_surface_features(
    *,
    prepared_view: PreparedView,
    dino_result: DINOFeatureResult,
    output_directory: Path,
    maximum_point_count: int = 50_000,
    random_seed: int = 0,
    device: str = "cuda:0",
    feature_chunk_size: int = 8192,
) -> ObservedSurfaceFeatureResult:
    """
    RGB-D 관측 표면의 3D point와 DINO feature를 생성한다.

    처리 과정:
        mask + depth
        → camera-space 3D points
        → DINO dense feature sampling
        → point-feature pair 저장
    """

    view_name = prepared_view.view.source.name

    if view_name not in (
        "reference",
        "query",
    ):
        raise ValueError(
            f"지원하지 않는 view입니다: {view_name}"
        )

    dino_view_name = getattr(
        dino_result,
        "view_name",
        None,
    )

    if (
        dino_view_name is not None
        and dino_view_name != view_name
    ):
        raise ValueError(
            "PreparedView와 DINO feature의 view가 다릅니다: "
            f"prepared={view_name}, dino={dino_view_name}"
        )

    image_height = int(
        prepared_view.view.rgb.shape[0]
    )

    image_width = int(
        prepared_view.view.rgb.shape[1]
    )

    points_camera_m, pixel_coordinates_xy = (
        _backproject_masked_depth(
            prepared_view
        )
    )

    (
        points_camera_m,
        pixel_coordinates_xy,
    ) = _subsample_points(
        points_camera_m=points_camera_m,
        pixel_coordinates_xy=(
            pixel_coordinates_xy
        ),
        maximum_point_count=(
            maximum_point_count
        ),
        random_seed=random_seed,
    )

    feature_map = _resolve_dino_feature_map(
        dino_result
    )

    features = _sample_feature_map(
        feature_map=feature_map,
        pixel_coordinates_xy=(
            pixel_coordinates_xy
        ),
        image_height=image_height,
        image_width=image_width,
        device=device,
        chunk_size=feature_chunk_size,
    )

    output_directory = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    points_path = (
        output_directory
        / "points_camera_m.npy"
    )

    pixels_path = (
        output_directory
        / "pixel_coordinates_xy.npy"
    )

    features_path = (
        output_directory
        / "dino_surface_features.npy"
    )

    metadata_path = (
        output_directory
        / "observed_surface_features.json"
    )

    np.save(
        points_path,
        points_camera_m,
        allow_pickle=False,
    )

    np.save(
        pixels_path,
        pixel_coordinates_xy,
        allow_pickle=False,
    )

    np.save(
        features_path,
        features,
        allow_pickle=False,
    )

    metadata = {
        "view_name": view_name,
        "coordinate_frame": (
            f"{view_name}_camera"
        ),
        "translation_unit": "meter",
        "point_count": int(
            points_camera_m.shape[0]
        ),
        "feature_dimension": int(
            features.shape[1]
        ),
        "image_height": image_height,
        "image_width": image_width,
        "feature_height": int(
            feature_map.shape[1]
        ),
        "feature_width": int(
            feature_map.shape[2]
        ),
        "maximum_point_count": int(
            maximum_point_count
        ),
        "random_seed": int(
            random_seed
        ),
        "feature_sampling": {
            "mode": "bilinear",
            "align_corners": False,
            "source": "original_rgb_full_frame",
        },
        "points_path": str(points_path),
        "pixels_path": str(pixels_path),
        "features_path": str(features_path),
    }

    with metadata_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return ObservedSurfaceFeatureResult(
        view_name=view_name,
        points_camera_m=points_camera_m,
        pixel_coordinates_xy=(
            pixel_coordinates_xy
        ),
        features=features,
        image_height=image_height,
        image_width=image_width,
        feature_height=int(
            feature_map.shape[1]
        ),
        feature_width=int(
            feature_map.shape[2]
        ),
        points_path=points_path,
        pixels_path=pixels_path,
        features_path=features_path,
        metadata_path=metadata_path,
    )
