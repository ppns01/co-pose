from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from core.types import (
    LoadedView,
    PreparedView,
    SegmentationResult,
)


def _validate_inputs(
    view: LoadedView,
    segmentation: SegmentationResult,
) -> None:
    """RGB, Depth, Mask의 shape과 dtype을 검증한다."""

    image_shape = view.rgb.shape[:2]

    if segmentation.mask_bool.shape != image_shape:
        raise ValueError(
            "RGB와 SAM3 mask의 해상도가 다릅니다: "
            f"RGB={image_shape}, "
            f"Mask={segmentation.mask_bool.shape}"
        )

    if view.depth_m.shape != image_shape:
        raise ValueError(
            "RGB와 Depth의 해상도가 다릅니다: "
            f"RGB={image_shape}, "
            f"Depth={view.depth_m.shape}"
        )

    if view.rgb.dtype != np.uint8:
        raise TypeError(
            "RGB dtype은 uint8이어야 합니다: "
            f"{view.rgb.dtype}"
        )

    if view.depth_m.dtype != np.float32:
        raise TypeError(
            "Depth dtype은 float32이어야 합니다: "
            f"{view.depth_m.dtype}"
        )

    if segmentation.mask_bool.dtype != np.bool_:
        raise TypeError(
            "Mask dtype은 bool이어야 합니다: "
            f"{segmentation.mask_bool.dtype}"
        )

    if not np.any(segmentation.mask_bool):
        raise ValueError(
            "SAM3 mask가 비어 있습니다."
        )


def _build_segmented_rgb(
    rgb: NDArray[np.uint8],
    mask_bool: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    """
    Mask 외부를 검은색으로 처리한 RGB 이미지를 생성한다.

    객체 영역:
        원본 RGB 유지

    배경 영역:
        [0, 0, 0]
    """

    segmented_rgb = np.zeros_like(
        rgb,
        dtype=np.uint8,
    )

    segmented_rgb[mask_bool] = rgb[mask_bool]

    return np.ascontiguousarray(
        segmented_rgb,
        dtype=np.uint8,
    )


def _build_masked_depth(
    depth_m: NDArray[np.float32],
    mask_bool: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """
    Mask 외부와 유효하지 않은 depth를 0으로 처리한다.

    출력 단위는 meter이다.
    """

    valid_depth_mask = (
        mask_bool
        & np.isfinite(depth_m)
        & (depth_m > 0.0)
    )

    masked_depth_m = np.zeros_like(
        depth_m,
        dtype=np.float32,
    )

    masked_depth_m[valid_depth_mask] = (
        depth_m[valid_depth_mask]
    )

    return np.ascontiguousarray(
        masked_depth_m,
        dtype=np.float32,
    )


def _write_rgb_png(
    output_path: Path,
    rgb: NDArray[np.uint8],
) -> None:
    """
    RGB 이미지를 PNG로 저장한다.

    OpenCV 저장 직전에 RGB를 BGR로 변환한다.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    success, encoded_image = cv2.imencode(
        ".png",
        bgr,
    )

    if not success:
        raise RuntimeError(
            "Segmented RGB PNG 인코딩에 실패했습니다: "
            f"{output_path}"
        )

    encoded_image.tofile(
        str(output_path)
    )

    if not output_path.is_file():
        raise FileNotFoundError(
            "Segmented RGB 파일이 저장되지 않았습니다: "
            f"{output_path}"
        )


def _write_depth_npy(
    output_path: Path,
    masked_depth_m: NDArray[np.float32],
) -> None:
    """Meter 단위 masked depth를 float32 NPY로 저장한다."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        masked_depth_m,
        allow_pickle=False,
    )

    if not output_path.is_file():
        raise FileNotFoundError(
            "Masked depth 파일이 저장되지 않았습니다: "
            f"{output_path}"
        )


def prepare_masked_view(
    view: LoadedView,
    segmentation: SegmentationResult,
    output_directory: Path,
) -> PreparedView:
    """
    하나의 Reference 또는 Query 입력을 전처리한다.

    생성 파일
    ---------
    output_directory/
    ├── segmented_rgb.png
    └── masked_depth_m.npy
    """

    _validate_inputs(
        view=view,
        segmentation=segmentation,
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

    segmented_rgb = _build_segmented_rgb(
        rgb=view.rgb,
        mask_bool=segmentation.mask_bool,
    )

    masked_depth_m = _build_masked_depth(
        depth_m=view.depth_m,
        mask_bool=segmentation.mask_bool,
    )

    segmented_rgb_path = (
        output_directory
        / "segmented_rgb.png"
    )

    masked_depth_path = (
        output_directory
        / "masked_depth_m.npy"
    )

    _write_rgb_png(
        output_path=segmented_rgb_path,
        rgb=segmented_rgb,
    )

    _write_depth_npy(
        output_path=masked_depth_path,
        masked_depth_m=masked_depth_m,
    )

    return PreparedView(
        view=view,
        segmentation=segmentation,
        segmented_rgb=segmented_rgb,
        masked_depth_m=masked_depth_m,
        segmented_rgb_path=segmented_rgb_path,
        masked_depth_path=masked_depth_path,
    )