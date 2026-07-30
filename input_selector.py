# input_selector.py

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
)


@dataclass(frozen=True)
class ViewInput:
    """Reference 또는 Query 한 프레임의 입력 정보."""

    split: str
    scene_id: int
    image_id: int
    object_id: int
    gt_index: int

    rgb_path: Path
    depth_path: Path
    mask_path: Path

    camera_matrix: np.ndarray
    depth_scale_mm_per_unit: float

    @property
    def depth_scale_m_per_unit(self) -> float:
        """Raw depth 값을 meter 단위로 변환하는 배율."""

        return self.depth_scale_mm_per_unit / 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "BOP LINEMOD 데이터셋에서 "
            "Reference와 Query RGB-D 입력을 선택합니다."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="BOP 형식 LINEMOD 데이터셋 루트 경로",
    )

    parser.add_argument(
        "--object-id",
        type=int,
        required=True,
        help="Reference와 Query에서 사용할 LINEMOD 객체 ID",
    )

    parser.add_argument(
        "--reference-split",
        type=str,
        default="test",
        help="Reference 데이터 split. 기본값: test",
    )

    parser.add_argument(
        "--reference-scene-id",
        type=int,
        required=True,
        help="Reference scene ID",
    )

    parser.add_argument(
        "--reference-image-id",
        type=int,
        required=True,
        help="Reference image ID",
    )

    parser.add_argument(
        "--reference-instance-index",
        type=int,
        default=0,
        help=(
            "동일 객체가 여러 개 있을 때 선택할 instance 순서. "
            "기본값: 0"
        ),
    )

    parser.add_argument(
        "--query-split",
        type=str,
        default="test",
        help="Query 데이터 split. 기본값: test",
    )

    parser.add_argument(
        "--query-scene-id",
        type=int,
        required=True,
        help="Query scene ID",
    )

    parser.add_argument(
        "--query-image-id",
        type=int,
        required=True,
        help="Query image ID",
    )

    parser.add_argument(
        "--query-instance-index",
        type=int,
        default=0,
        help=(
            "동일 객체가 여러 개 있을 때 선택할 instance 순서. "
            "기본값: 0"
        ),
    )

    parser.add_argument(
        "--mask-type",
        choices=("mask_visib", "mask"),
        default="mask_visib",
        help=(
            "사용할 BOP mask 종류. "
            "기본값은 실제 보이는 영역인 mask_visib"
        ),
    )

    return parser.parse_args()


def load_json(json_path: Path) -> dict[str, Any]:
    """JSON 파일을 읽고 최상위 객체가 dict인지 확인한다."""

    if not json_path.is_file():
        raise FileNotFoundError(
            f"필수 JSON 파일이 없습니다: {json_path}"
        )

    try:
        with json_path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON 형식이 올바르지 않습니다: {json_path}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            f"JSON 최상위 구조가 dict가 아닙니다: {json_path}"
        )

    return data


def get_frame_record(
    data: dict[str, Any],
    image_id: int,
    json_path: Path,
) -> Any:
    """scene_camera 또는 scene_gt에서 image ID 항목을 찾는다."""

    possible_keys = (
        str(image_id),
        f"{image_id:06d}",
    )

    for key in possible_keys:
        if key in data:
            return data[key]

    raise KeyError(
        f"Image ID {image_id} 항목이 없습니다: {json_path}"
    )


def resolve_frame_file(
    directory: Path,
    image_id: int,
) -> Path:
    """Image ID에 대응하는 이미지 파일 경로를 찾는다."""

    if not directory.is_dir():
        raise FileNotFoundError(
            f"입력 디렉터리가 없습니다: {directory}"
        )

    frame_stem = f"{image_id:06d}"

    for suffix in IMAGE_SUFFIXES:
        candidate = directory / f"{frame_stem}{suffix}"

        if candidate.is_file():
            return candidate

    matches = [
        path
        for path in directory.glob(f"{frame_stem}.*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"동일한 Image ID의 파일이 여러 개입니다: {matches}"
        )

    raise FileNotFoundError(
        f"Image ID {image_id} 파일을 찾지 못했습니다: "
        f"{directory}"
    )


def parse_camera_matrix(
    camera_record: dict[str, Any],
    scene_camera_path: Path,
    image_id: int,
) -> np.ndarray:
    """cam_K를 3×3 camera intrinsic 행렬로 변환한다."""

    if "cam_K" not in camera_record:
        raise KeyError(
            f"Image ID {image_id}에 cam_K가 없습니다: "
            f"{scene_camera_path}"
        )

    camera_values = np.asarray(
        camera_record["cam_K"],
        dtype=np.float64,
    )

    if camera_values.size != 9:
        raise ValueError(
            f"cam_K 원소 개수가 9개가 아닙니다: "
            f"{scene_camera_path}, image_id={image_id}"
        )

    camera_matrix = camera_values.reshape(3, 3)

    if not np.isfinite(camera_matrix).all():
        raise ValueError(
            f"cam_K에 NaN 또는 Inf가 있습니다: "
            f"{scene_camera_path}, image_id={image_id}"
        )

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])

    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(
            f"fx와 fy는 양수여야 합니다: "
            f"fx={fx}, fy={fy}"
        )

    return camera_matrix


def parse_depth_scale(
    camera_record: dict[str, Any],
    scene_camera_path: Path,
    image_id: int,
) -> float:
    """BOP depth_scale을 읽는다."""

    if "depth_scale" not in camera_record:
        raise KeyError(
            f"Image ID {image_id}에 depth_scale이 없습니다: "
            f"{scene_camera_path}"
        )

    depth_scale = float(camera_record["depth_scale"])

    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError(
            f"depth_scale은 유한한 양수여야 합니다: "
            f"{depth_scale}"
        )

    return depth_scale


def find_gt_index(
    scene_gt_record: Any,
    object_id: int,
    instance_index: int,
    scene_gt_path: Path,
    image_id: int,
) -> int:
    """
    해당 프레임에서 object_id에 대응하는 실제 GT index를 찾는다.

    BOP mask 이름:
        image_id_gt_index.png
    """

    if not isinstance(scene_gt_record, list):
        raise ValueError(
            f"scene_gt의 프레임 항목이 list가 아닙니다: "
            f"{scene_gt_path}, image_id={image_id}"
        )

    matching_gt_indices: list[int] = []

    for gt_index, annotation in enumerate(scene_gt_record):
        if not isinstance(annotation, dict):
            continue

        if int(annotation.get("obj_id", -1)) == object_id:
            matching_gt_indices.append(gt_index)

    if not matching_gt_indices:
        raise ValueError(
            f"Image ID {image_id}에 object ID "
            f"{object_id}가 없습니다: {scene_gt_path}"
        )

    if instance_index < 0:
        raise ValueError(
            "instance_index는 0 이상이어야 합니다."
        )

    if instance_index >= len(matching_gt_indices):
        raise IndexError(
            f"선택 가능한 object instance는 "
            f"{len(matching_gt_indices)}개인데 "
            f"instance_index={instance_index}가 입력되었습니다."
        )

    return matching_gt_indices[instance_index]


def build_view_input(
    dataset_root: Path,
    split: str,
    scene_id: int,
    image_id: int,
    object_id: int,
    instance_index: int,
    mask_type: str,
) -> ViewInput:
    """Reference 또는 Query 입력 하나를 생성한다."""

    scene_dir = (
        dataset_root
        / split
        / f"{scene_id:06d}"
    )

    if not scene_dir.is_dir():
        raise FileNotFoundError(
            f"Scene 폴더가 없습니다: {scene_dir}"
        )

    rgb_path = resolve_frame_file(
        scene_dir / "rgb",
        image_id,
    )

    depth_path = resolve_frame_file(
        scene_dir / "depth",
        image_id,
    )

    scene_camera_path = (
        scene_dir / "scene_camera.json"
    )

    scene_gt_path = (
        scene_dir / "scene_gt.json"
    )

    scene_camera = load_json(scene_camera_path)
    scene_gt = load_json(scene_gt_path)

    camera_record = get_frame_record(
        data=scene_camera,
        image_id=image_id,
        json_path=scene_camera_path,
    )

    if not isinstance(camera_record, dict):
        raise ValueError(
            f"scene_camera 프레임 항목이 dict가 아닙니다: "
            f"{scene_camera_path}, image_id={image_id}"
        )

    camera_matrix = parse_camera_matrix(
        camera_record=camera_record,
        scene_camera_path=scene_camera_path,
        image_id=image_id,
    )

    depth_scale = parse_depth_scale(
        camera_record=camera_record,
        scene_camera_path=scene_camera_path,
        image_id=image_id,
    )

    scene_gt_record = get_frame_record(
        data=scene_gt,
        image_id=image_id,
        json_path=scene_gt_path,
    )

    gt_index = find_gt_index(
        scene_gt_record=scene_gt_record,
        object_id=object_id,
        instance_index=instance_index,
        scene_gt_path=scene_gt_path,
        image_id=image_id,
    )

    mask_path = (
        scene_dir
        / mask_type
        / f"{image_id:06d}_{gt_index:06d}.png"
    )

    if not mask_path.is_file():
        raise FileNotFoundError(
            f"Object mask가 없습니다: {mask_path}"
        )

    return ViewInput(
        split=split,
        scene_id=scene_id,
        image_id=image_id,
        object_id=object_id,
        gt_index=gt_index,
        rgb_path=rgb_path,
        depth_path=depth_path,
        mask_path=mask_path,
        camera_matrix=camera_matrix,
        depth_scale_mm_per_unit=depth_scale,
    )


def validate_different_frames(
    reference: ViewInput,
    query: ViewInput,
) -> None:
    """Reference와 Query가 완전히 동일한 입력인지 검사한다."""

    same_frame = (
        reference.split == query.split
        and reference.scene_id == query.scene_id
        and reference.image_id == query.image_id
        and reference.gt_index == query.gt_index
    )

    if same_frame:
        raise ValueError(
            "Reference와 Query가 동일한 프레임과 "
            "동일한 객체 instance입니다."
        )


def print_view_input(
    name: str,
    view: ViewInput,
) -> None:
    print(f"\n[{name}]")
    print(f"Split       : {view.split}")
    print(f"Scene ID    : {view.scene_id}")
    print(f"Image ID    : {view.image_id}")
    print(f"Object ID   : {view.object_id}")
    print(f"GT index    : {view.gt_index}")
    print(f"RGB         : {view.rgb_path}")
    print(f"Depth       : {view.depth_path}")
    print(f"Mask        : {view.mask_path}")
    print(
        "Depth scale : "
        f"{view.depth_scale_mm_per_unit} mm/raw-unit"
    )
    print("K:")
    print(view.camera_matrix)


def main() -> None:
    args = parse_args()

    dataset_root = (
        args.dataset_root
        .expanduser()
        .resolve()
    )

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"LINEMOD 데이터셋 루트가 없습니다: "
            f"{dataset_root}"
        )

    reference = build_view_input(
        dataset_root=dataset_root,
        split=args.reference_split,
        scene_id=args.reference_scene_id,
        image_id=args.reference_image_id,
        object_id=args.object_id,
        instance_index=args.reference_instance_index,
        mask_type=args.mask_type,
    )

    query = build_view_input(
        dataset_root=dataset_root,
        split=args.query_split,
        scene_id=args.query_scene_id,
        image_id=args.query_image_id,
        object_id=args.object_id,
        instance_index=args.query_instance_index,
        mask_type=args.mask_type,
    )

    validate_different_frames(
        reference=reference,
        query=query,
    )

    print("[LINEMOD 입력 선택 완료]")

    print_view_input(
        name="Reference",
        view=reference,
    )

    print_view_input(
        name="Query",
        view=query,
    )


if __name__ == "__main__":
    main()