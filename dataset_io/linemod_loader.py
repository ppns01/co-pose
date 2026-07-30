from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from core.types import LoadedView, ViewInput, ViewName


RGB_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
)


def _require_directory(
    directory_path: Path,
    description: str,
) -> Path:
    """필수 디렉터리가 존재하는지 확인한다."""

    if not directory_path.is_dir():
        raise FileNotFoundError(
            f"{description} 디렉터리가 없습니다: "
            f"{directory_path}"
        )

    return directory_path


def _require_file(
    file_path: Path,
    description: str,
) -> Path:
    """필수 파일이 존재하는지 확인한다."""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"{description} 파일이 없습니다: "
            f"{file_path}"
        )

    return file_path


def _read_image(
    image_path: Path,
    flags: int,
) -> NDArray[np.generic]:
    """
    한글이 포함된 Windows 경로에서도 이미지를 읽는다.

    cv2.imread()는 환경에 따라 한글 경로를 처리하지 못할 수 있으므로
    np.fromfile()과 cv2.imdecode()를 사용한다.
    """

    _require_file(
        image_path,
        "이미지",
    )

    encoded_data = np.fromfile(
        str(image_path),
        dtype=np.uint8,
    )

    if encoded_data.size == 0:
        raise ValueError(
            f"이미지 파일이 비어 있습니다: {image_path}"
        )

    image = cv2.imdecode(
        encoded_data,
        flags,
    )

    if image is None:
        raise ValueError(
            f"이미지를 읽지 못했습니다: {image_path}"
        )

    return image


def _resolve_rgb_path(
    scene_directory: Path,
    image_id: int,
) -> Path:
    """
    LINEMOD RGB 파일을 찾는다.

    test split에서는 일반적으로 PNG를 사용하고,
    train_pbr에서는 JPG가 사용될 수 있으므로 둘 다 지원한다.
    """

    rgb_directory = _require_directory(
        scene_directory / "rgb",
        "LINEMOD RGB",
    )

    image_stem = f"{image_id:06d}"

    matching_paths: list[Path] = []

    for suffix in RGB_SUFFIXES:
        candidate_path = (
            rgb_directory
            / f"{image_stem}{suffix}"
        )

        if candidate_path.is_file():
            matching_paths.append(candidate_path)

    if not matching_paths:
        raise FileNotFoundError(
            "LINEMOD RGB 파일을 찾지 못했습니다: "
            f"{rgb_directory / image_stem}"
        )

    if len(matching_paths) > 1:
        raise RuntimeError(
            "같은 image_id에 해당하는 RGB 파일이 "
            "여러 개 존재합니다: "
            f"{matching_paths}"
        )

    return matching_paths[0]


def _resolve_depth_path(
    scene_directory: Path,
    image_id: int,
) -> Path:
    """LINEMOD 16-bit depth PNG 경로를 결정한다."""

    depth_directory = _require_directory(
        scene_directory / "depth",
        "LINEMOD Depth",
    )

    depth_path = (
        depth_directory
        / f"{image_id:06d}.png"
    )

    return _require_file(
        depth_path,
        "LINEMOD Depth",
    )


def _load_json(
    json_path: Path,
) -> dict[str, Any]:
    """JSON 파일을 읽고 최상위 자료형을 검증한다."""

    _require_file(
        json_path,
        "JSON",
    )

    try:
        with json_path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            json_data = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            "JSON 형식이 올바르지 않습니다: "
            f"{json_path}, "
            f"line={error.lineno}, "
            f"column={error.colno}"
        ) from error

    if not isinstance(json_data, dict):
        raise ValueError(
            "JSON 최상위 구조는 object여야 합니다: "
            f"{json_path}"
        )

    return json_data


def _get_camera_record(
    scene_camera_path: Path,
    image_id: int,
) -> dict[str, Any]:
    """
    scene_camera.json에서 지정한 image_id의 카메라 정보를 찾는다.
    """

    scene_camera = _load_json(
        scene_camera_path
    )

    possible_keys = (
        str(image_id),
        f"{image_id:06d}",
    )

    camera_record: Any = None

    for key in possible_keys:
        if key in scene_camera:
            camera_record = scene_camera[key]
            break

    if camera_record is None:
        raise KeyError(
            "scene_camera.json에 image_id가 없습니다: "
            f"image_id={image_id}, "
            f"path={scene_camera_path}"
        )

    if not isinstance(camera_record, dict):
        raise ValueError(
            "scene_camera의 프레임 정보가 "
            "object 형식이 아닙니다: "
            f"image_id={image_id}"
        )

    return camera_record


def _parse_camera_matrix(
    camera_record: dict[str, Any],
    scene_camera_path: Path,
    image_id: int,
) -> NDArray[np.float32]:
    """cam_K를 float32의 3×3 행렬로 변환한다."""

    if "cam_K" not in camera_record:
        raise KeyError(
            "카메라 정보에 cam_K가 없습니다: "
            f"image_id={image_id}, "
            f"path={scene_camera_path}"
        )

    camera_values = np.asarray(
        camera_record["cam_K"],
        dtype=np.float32,
    )

    if camera_values.size != 9:
        raise ValueError(
            "cam_K의 원소 개수는 9개여야 합니다: "
            f"actual={camera_values.size}, "
            f"image_id={image_id}"
        )

    camera_matrix = np.ascontiguousarray(
        camera_values.reshape(3, 3),
        dtype=np.float32,
    )

    if not np.isfinite(camera_matrix).all():
        raise ValueError(
            "cam_K에 NaN 또는 Inf가 있습니다: "
            f"image_id={image_id}"
        )

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])

    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(
            "cam_K의 fx와 fy는 양수여야 합니다: "
            f"fx={fx}, fy={fy}"
        )

    return camera_matrix


def _parse_depth_scale_to_m(
    camera_record: dict[str, Any],
    scene_camera_path: Path,
    image_id: int,
) -> float:
    """
    LINEMOD depth raw 값을 meter로 변환하는 배율을 반환한다.

    BOP의 depth_scale은 raw depth를 millimeter로 변환하는 값이므로
    meter 변환을 위해 추가로 0.001을 곱한다.
    """

    if "depth_scale" not in camera_record:
        raise KeyError(
            "카메라 정보에 depth_scale이 없습니다: "
            f"image_id={image_id}, "
            f"path={scene_camera_path}"
        )

    try:
        depth_scale_mm = float(
            camera_record["depth_scale"]
        )

    except (TypeError, ValueError) as error:
        raise ValueError(
            "depth_scale을 숫자로 변환할 수 없습니다: "
            f"{camera_record['depth_scale']}"
        ) from error

    if (
        not np.isfinite(depth_scale_mm)
        or depth_scale_mm <= 0.0
    ):
        raise ValueError(
            "depth_scale은 유한한 양수여야 합니다: "
            f"{depth_scale_mm}"
        )

    return depth_scale_mm * 0.001


def _load_rgb(
    rgb_path: Path,
) -> NDArray[np.uint8]:
    """LINEMOD RGB 이미지를 RGB 순서의 uint8 배열로 읽는다."""

    bgr_image = _read_image(
        image_path=rgb_path,
        flags=cv2.IMREAD_COLOR,
    )

    if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
        raise ValueError(
            "RGB 이미지 shape이 올바르지 않습니다: "
            f"{bgr_image.shape}"
        )

    rgb_image = cv2.cvtColor(
        bgr_image,
        cv2.COLOR_BGR2RGB,
    )

    return np.ascontiguousarray(
        rgb_image,
        dtype=np.uint8,
    )


def _load_depth_m(
    depth_path: Path,
    depth_scale_to_m: float,
) -> NDArray[np.float32]:
    """
    LINEMOD depth PNG를 읽고 meter 단위 float32로 변환한다.

    raw depth가 0인 픽셀은 유효하지 않은 depth로 유지한다.
    """

    depth_raw = _read_image(
        image_path=depth_path,
        flags=cv2.IMREAD_UNCHANGED,
    )

    if depth_raw.ndim != 2:
        raise ValueError(
            "Depth 이미지는 단일 채널이어야 합니다: "
            f"shape={depth_raw.shape}"
        )

    if not np.issubdtype(
        depth_raw.dtype,
        np.integer,
    ):
        raise TypeError(
            "LINEMOD depth PNG는 정수형이어야 합니다: "
            f"dtype={depth_raw.dtype}"
        )

    depth_m = (
        depth_raw.astype(np.float32)
        * np.float32(depth_scale_to_m)
    )

    depth_m[depth_raw == 0] = 0.0

    if not np.isfinite(depth_m).all():
        raise ValueError(
            "변환된 Depth에 NaN 또는 Inf가 있습니다."
        )

    if np.any(depth_m < 0.0):
        raise ValueError(
            "변환된 Depth에 음수 값이 있습니다."
        )

    return np.ascontiguousarray(
        depth_m,
        dtype=np.float32,
    )


def load_linemod_view(
    dataset_root: Path,
    view_name: ViewName,
    scene_id: int,
    image_id: int,
    object_name: str,
    object_id: int | None = None,
    split: str = "test",
) -> LoadedView:
    """
    BOP 형식 LINEMOD 데이터셋의 한 프레임을 읽는다.

    Parameters
    ----------
    dataset_root:
        LINEMOD `lm` 데이터셋 루트.
        예: C:/datasets/bop/lm

    view_name:
        "reference" 또는 "query".

    scene_id:
        LINEMOD scene 번호.

    image_id:
        scene 내부 image 번호.

    object_name:
        이후 SAM3에 전달할 객체 이름.

    object_id:
        평가용 객체 ID.
        현재 입력 로딩 과정에서는 사용하지 않는다.

    split:
        기본값은 "test".
        필요하면 "train_pbr"도 사용할 수 있다.

    Returns
    -------
    LoadedView
        RGB, meter 단위 Depth, camera intrinsic K.
    """

    dataset_root = (
        Path(dataset_root)
        .expanduser()
        .resolve()
    )

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            "LINEMOD 데이터셋 루트가 없습니다: "
            f"{dataset_root}"
        )

    if scene_id < 0:
        raise ValueError(
            f"scene_id는 0 이상이어야 합니다: {scene_id}"
        )

    if image_id < 0:
        raise ValueError(
            f"image_id는 0 이상이어야 합니다: {image_id}"
        )

    if not split.strip():
        raise ValueError(
            "split은 빈 문자열일 수 없습니다."
        )

    scene_directory = (
        dataset_root
        / split
        / f"{scene_id:06d}"
    )

    _require_directory(
        scene_directory,
        "LINEMOD scene",
    )

    rgb_path = _resolve_rgb_path(
        scene_directory=scene_directory,
        image_id=image_id,
    )

    depth_path = _resolve_depth_path(
        scene_directory=scene_directory,
        image_id=image_id,
    )

    scene_camera_path = _require_file(
        scene_directory / "scene_camera.json",
        "LINEMOD scene_camera.json",
    )

    camera_record = _get_camera_record(
        scene_camera_path=scene_camera_path,
        image_id=image_id,
    )

    camera_matrix = _parse_camera_matrix(
        camera_record=camera_record,
        scene_camera_path=scene_camera_path,
        image_id=image_id,
    )

    depth_scale_to_m = _parse_depth_scale_to_m(
        camera_record=camera_record,
        scene_camera_path=scene_camera_path,
        image_id=image_id,
    )

    rgb = _load_rgb(
        rgb_path
    )

    depth_m = _load_depth_m(
        depth_path=depth_path,
        depth_scale_to_m=depth_scale_to_m,
    )

    source = ViewInput(
        name=view_name,
        rgb_path=rgb_path,
        depth_path=depth_path,
        intrinsics_path=scene_camera_path,
        object_name=object_name,
        scene_id=scene_id,
        image_id=image_id,
        object_id=object_id,
    )

    return LoadedView(
        source=source,
        rgb=rgb,
        depth_m=depth_m,
        camera_matrix=camera_matrix,
        depth_scale_to_m=depth_scale_to_m,
    )