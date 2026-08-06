from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BOPFrameGTSpec:
    """
    BOP-LINEMOD의 평가용 GT 프레임 지정 정보.

    object_id:
        BOP 객체 ID.

    instance_index:
        동일한 object_id가 한 영상에 여러 번 등장할 때
        몇 번째 instance를 선택할지 지정한다.
    """

    dataset_root: Path
    split: str
    scene_id: int
    image_id: int
    object_id: int
    instance_index: int = 0

    def __post_init__(self) -> None:
        if not self.split.strip():
            raise ValueError(
                "split은 빈 문자열일 수 없습니다."
            )

        if self.scene_id < 0:
            raise ValueError(
                "scene_id는 0 이상이어야 합니다: "
                f"{self.scene_id}"
            )

        if self.image_id < 0:
            raise ValueError(
                "image_id는 0 이상이어야 합니다: "
                f"{self.image_id}"
            )

        if self.object_id <= 0:
            raise ValueError(
                "object_id는 1 이상이어야 합니다: "
                f"{self.object_id}"
            )

        if self.instance_index < 0:
            raise ValueError(
                "instance_index는 0 이상이어야 합니다: "
                f"{self.instance_index}"
            )

    @property
    def scene_directory(self) -> Path:
        return (
            Path(self.dataset_root)
            .expanduser()
            .resolve()
            / self.split
            / f"{self.scene_id:06d}"
        )

    @property
    def scene_gt_path(self) -> Path:
        return self.scene_directory / "scene_gt.json"


@dataclass(frozen=True)
class RelativePoseThreshold:
    """상대 pose 성공 여부를 판단하는 기준."""

    name: str
    maximum_rotation_error_deg: float
    maximum_translation_error_m: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "Threshold 이름은 빈 문자열일 수 없습니다."
            )

        if (
            not np.isfinite(
                self.maximum_rotation_error_deg
            )
            or self.maximum_rotation_error_deg <= 0.0
        ):
            raise ValueError(
                "maximum_rotation_error_deg는 "
                "유한한 양수여야 합니다."
            )

        if (
            not np.isfinite(
                self.maximum_translation_error_m
            )
            or self.maximum_translation_error_m <= 0.0
        ):
            raise ValueError(
                "maximum_translation_error_m은 "
                "유한한 양수여야 합니다."
            )


DEFAULT_THRESHOLDS = (
    RelativePoseThreshold(
        name="5deg_5cm",
        maximum_rotation_error_deg=5.0,
        maximum_translation_error_m=0.05,
    ),
    RelativePoseThreshold(
        name="2deg_2cm",
        maximum_rotation_error_deg=2.0,
        maximum_translation_error_m=0.02,
    ),
    RelativePoseThreshold(
        name="1deg_1cm",
        maximum_rotation_error_deg=1.0,
        maximum_translation_error_m=0.01,
    ),
    RelativePoseThreshold(
        name="5deg_2cm",
        maximum_rotation_error_deg=5.0,
        maximum_translation_error_m=0.02,
    ),
    RelativePoseThreshold(
        name="10deg_5cm",
        maximum_rotation_error_deg=10.0,
        maximum_translation_error_m=0.05,
    ),
    RelativePoseThreshold(
        name="15deg_5cm",
        maximum_rotation_error_deg=15.0,
        maximum_translation_error_m=0.05,
    ),
)


@dataclass(frozen=True)
class ThresholdEvaluation:
    """특정 기준에서의 상대 pose 성공 여부."""

    threshold: RelativePoseThreshold
    rotation_passed: bool
    translation_passed: bool
    success: bool

    def __post_init__(self) -> None:
        expected_success = (
            self.rotation_passed
            and self.translation_passed
        )

        if self.success != expected_success:
            raise ValueError(
                "success 값이 rotation/translation "
                "통과 결과와 일치하지 않습니다."
            )


@dataclass(frozen=True)
class RelativePoseEvaluationResult:
    """
    예측 상대 pose와 BOP-LINEMOD 상대 GT의 평가 결과.

    Pose 규약:
        T_query_camera_from_reference_camera

    Translation 단위:
        meter
    """

    predicted_relative_pose: NDArray[np.float32]
    ground_truth_relative_pose: NDArray[np.float32]

    rotation_error_deg: float

    # 물체 중심 변위(TE). 예측 상대 pose를 GT reference 절대 pose에
    # anchor해서 잰다.
    translation_error_m: float
    translation_error_cm: float
    translation_error_x_m: float
    translation_error_y_m: float
    translation_error_z_m: float
    translation_error_x_cm: float
    translation_error_y_cm: float
    translation_error_z_cm: float

    # 진단용: 상대 pose translation 열의 단순 차이. reference 카메라
    # 원점 기준이라 회전 오차가 섞여 들어가므로 성공/실패 판정에는
    # 쓰지 않는다.
    reference_origin_translation_error_m: float
    reference_origin_translation_error_cm: float

    estimated_rotation_deg: float
    estimated_translation_m: float
    estimated_translation_cm: float
    estimated_tx_m: float
    estimated_ty_m: float
    estimated_tz_m: float

    ground_truth_relative_rotation_deg: float
    ground_truth_relative_translation_m: float
    ground_truth_relative_translation_cm: float
    ground_truth_tx_m: float
    ground_truth_ty_m: float
    ground_truth_tz_m: float

    catastrophic_failure: bool

    threshold_results: tuple[
        ThresholdEvaluation,
        ...
    ]

    predicted_pose_path: Path
    ground_truth_pose_path: Path
    metadata_path: Path

    def __post_init__(self) -> None:
        _validate_rigid_pose(
            self.predicted_relative_pose,
            "predicted_relative_pose",
        )

        _validate_rigid_pose(
            self.ground_truth_relative_pose,
            "ground_truth_relative_pose",
        )

        scalar_values = (
            self.rotation_error_deg,
            self.translation_error_m,
            self.translation_error_cm,
            self.reference_origin_translation_error_m,
            self.reference_origin_translation_error_cm,
            self.estimated_rotation_deg,
            self.estimated_translation_m,
            self.estimated_translation_cm,
            self.ground_truth_relative_rotation_deg,
            self.ground_truth_relative_translation_m,
            self.ground_truth_relative_translation_cm,
        )

        for value in scalar_values:
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    "평가 오차는 유한한 0 이상의 값이어야 합니다: "
                    f"{value}"
                )

        signed_scalar_values = (
            self.translation_error_x_m,
            self.translation_error_y_m,
            self.translation_error_z_m,
            self.translation_error_x_cm,
            self.translation_error_y_cm,
            self.translation_error_z_cm,
            self.estimated_tx_m,
            self.estimated_ty_m,
            self.estimated_tz_m,
            self.ground_truth_tx_m,
            self.ground_truth_ty_m,
            self.ground_truth_tz_m,
        )

        for value in signed_scalar_values:
            if not np.isfinite(value):
                raise ValueError(
                    "Pose component must be finite: "
                    f"{value}"
                )

        if not self.threshold_results:
            raise ValueError(
                "Threshold 평가 결과가 없습니다."
            )

        for output_path in (
            self.predicted_pose_path,
            self.ground_truth_pose_path,
            self.metadata_path,
        ):
            if not output_path.is_file():
                raise FileNotFoundError(
                    f"평가 출력 파일이 없습니다: {output_path}"
                )


def _load_json(
    json_path: Path,
) -> dict[str, Any]:
    """JSON 파일을 읽고 최상위 자료형을 검증한다."""

    if not json_path.is_file():
        raise FileNotFoundError(
            f"BOP JSON 파일이 없습니다: {json_path}"
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


def _get_frame_annotations(
    scene_gt: dict[str, Any],
    image_id: int,
    scene_gt_path: Path,
) -> list[dict[str, Any]]:
    """scene_gt.json에서 지정한 image_id의 annotation을 찾는다."""

    possible_keys = (
        str(image_id),
        f"{image_id:06d}",
    )

    frame_annotations: Any = None

    for key in possible_keys:
        if key in scene_gt:
            frame_annotations = scene_gt[key]
            break

    if frame_annotations is None:
        raise KeyError(
            "scene_gt.json에 image_id가 없습니다: "
            f"image_id={image_id}, "
            f"path={scene_gt_path}"
        )

    if not isinstance(frame_annotations, list):
        raise ValueError(
            "scene_gt의 프레임 annotation은 "
            "list여야 합니다: "
            f"image_id={image_id}"
        )

    normalized_annotations: list[
        dict[str, Any]
    ] = []

    for annotation in frame_annotations:
        if not isinstance(annotation, dict):
            raise ValueError(
                "scene_gt annotation이 object가 아닙니다: "
                f"type={type(annotation)}"
            )

        normalized_annotations.append(annotation)

    return normalized_annotations


def _select_object_annotation(
    annotations: list[dict[str, Any]],
    *,
    object_id: int,
    instance_index: int,
    scene_gt_path: Path,
    image_id: int,
) -> dict[str, Any]:
    """지정한 object_id와 instance_index의 GT를 선택한다."""

    matching_annotations = [
        annotation
        for annotation in annotations
        if int(annotation.get("obj_id", -1))
        == object_id
    ]

    if not matching_annotations:
        raise ValueError(
            "지정한 객체가 프레임에 존재하지 않습니다: "
            f"object_id={object_id}, "
            f"image_id={image_id}, "
            f"path={scene_gt_path}"
        )

    if instance_index >= len(matching_annotations):
        raise IndexError(
            "요청한 instance_index가 범위를 벗어났습니다: "
            f"object_id={object_id}, "
            f"available={len(matching_annotations)}, "
            f"requested={instance_index}"
        )

    return matching_annotations[instance_index]


def _validate_rigid_pose(
    pose: NDArray[np.floating],
    name: str,
) -> NDArray[np.float32]:
    """4×4 SE(3) pose를 검증한다."""

    pose_array = np.asarray(
        pose,
        dtype=np.float32,
    )

    if pose_array.shape != (4, 4):
        raise ValueError(
            f"{name} shape은 (4, 4)이어야 합니다: "
            f"{pose_array.shape}"
        )

    if not np.isfinite(pose_array).all():
        raise ValueError(
            f"{name}에 NaN 또는 Inf가 있습니다."
        )

    expected_last_row = np.array(
        [0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )

    if not np.allclose(
        pose_array[3],
        expected_last_row,
        atol=1e-5,
        rtol=0.0,
    ):
        raise ValueError(
            f"{name}의 마지막 행이 "
            "homogeneous transform 형식이 아닙니다."
        )

    rotation = pose_array[:3, :3]

    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3, dtype=np.float32),
        atol=1e-3,
        rtol=0.0,
    ):
        raise ValueError(
            f"{name}의 회전 행렬이 직교하지 않습니다."
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
            f"{name} 회전 행렬 determinant가 "
            f"1이 아닙니다: {determinant}"
        )

    return np.ascontiguousarray(
        pose_array,
        dtype=np.float32,
    )


def _parse_bop_absolute_pose(
    annotation: dict[str, Any],
    *,
    scene_gt_path: Path,
    image_id: int,
    object_id: int,
) -> NDArray[np.float32]:
    """
    BOP cam_R_m2c, cam_t_m2c를 4×4 pose로 변환한다.

    BOP translation:
        millimeter

    내부 translation:
        meter
    """

    if "cam_R_m2c" not in annotation:
        raise KeyError(
            "BOP annotation에 cam_R_m2c가 없습니다: "
            f"image_id={image_id}, "
            f"object_id={object_id}, "
            f"path={scene_gt_path}"
        )

    if "cam_t_m2c" not in annotation:
        raise KeyError(
            "BOP annotation에 cam_t_m2c가 없습니다: "
            f"image_id={image_id}, "
            f"object_id={object_id}, "
            f"path={scene_gt_path}"
        )

    rotation_values = np.asarray(
        annotation["cam_R_m2c"],
        dtype=np.float64,
    )

    translation_mm = np.asarray(
        annotation["cam_t_m2c"],
        dtype=np.float64,
    )

    if rotation_values.size != 9:
        raise ValueError(
            "cam_R_m2c 원소 개수는 9개여야 합니다: "
            f"actual={rotation_values.size}"
        )

    if translation_mm.size != 3:
        raise ValueError(
            "cam_t_m2c 원소 개수는 3개여야 합니다: "
            f"actual={translation_mm.size}"
        )

    rotation = rotation_values.reshape(3, 3)
    translation_m = (
        translation_mm.reshape(3)
        * 0.001
    )

    pose = np.eye(
        4,
        dtype=np.float64,
    )

    pose[:3, :3] = rotation
    pose[:3, 3] = translation_m

    return _validate_rigid_pose(
        pose,
        "bop_absolute_pose",
    )


def load_bop_linemod_absolute_gt_pose(
    frame_spec: BOPFrameGTSpec,
) -> NDArray[np.float32]:
    """
    BOP-LINEMOD 한 프레임의 절대 GT pose를 읽는다.

    반환 규약:
        T_camera_from_object

    Translation 단위:
        meter
    """

    scene_gt_path = (
        frame_spec.scene_gt_path
        .expanduser()
        .resolve()
    )

    scene_gt = _load_json(
        scene_gt_path
    )

    annotations = _get_frame_annotations(
        scene_gt=scene_gt,
        image_id=frame_spec.image_id,
        scene_gt_path=scene_gt_path,
    )

    selected_annotation = (
        _select_object_annotation(
            annotations=annotations,
            object_id=frame_spec.object_id,
            instance_index=(
                frame_spec.instance_index
            ),
            scene_gt_path=scene_gt_path,
            image_id=frame_spec.image_id,
        )
    )

    return _parse_bop_absolute_pose(
        annotation=selected_annotation,
        scene_gt_path=scene_gt_path,
        image_id=frame_spec.image_id,
        object_id=frame_spec.object_id,
    )


def _invert_rigid_pose(
    pose_target_from_source: NDArray[np.floating],
) -> NDArray[np.float32]:
    """SE(3) 변환을 회전 transpose 방식으로 역변환한다."""

    pose = _validate_rigid_pose(
        pose_target_from_source,
        "pose_target_from_source",
    ).astype(
        np.float64,
        copy=False,
    )

    rotation = pose[:3, :3]
    translation = pose[:3, 3]

    inverse_pose = np.eye(
        4,
        dtype=np.float64,
    )

    inverse_rotation = rotation.T

    inverse_pose[:3, :3] = inverse_rotation
    inverse_pose[:3, 3] = (
        -inverse_rotation @ translation
    )

    return _validate_rigid_pose(
        inverse_pose,
        "inverse_pose",
    )


def build_ground_truth_relative_pose(
    reference_absolute_pose: NDArray[np.floating],
    query_absolute_pose: NDArray[np.floating],
) -> NDArray[np.float32]:
    """
    BOP 절대 GT pose로 Reference→Query 상대 GT를 만든다.

    H_query_from_reference
        = T_query_from_object
          @ inverse(T_reference_from_object)
    """

    reference_pose = _validate_rigid_pose(
        reference_absolute_pose,
        "reference_absolute_pose",
    )

    query_pose = _validate_rigid_pose(
        query_absolute_pose,
        "query_absolute_pose",
    )

    inverse_reference_pose = (
        _invert_rigid_pose(
            reference_pose
        )
    )

    relative_pose = (
        query_pose.astype(np.float64)
        @ inverse_reference_pose.astype(np.float64)
    )

    return _validate_rigid_pose(
        relative_pose,
        "ground_truth_relative_pose",
    )


def _compute_rotation_error_deg(
    predicted_pose: NDArray[np.float32],
    ground_truth_pose: NDArray[np.float32],
) -> float:
    """예측과 GT 사이의 SO(3) geodesic 회전 오차."""

    predicted_rotation = (
        predicted_pose[:3, :3]
        .astype(np.float64)
    )

    ground_truth_rotation = (
        ground_truth_pose[:3, :3]
        .astype(np.float64)
    )

    rotation_error_matrix = (
        predicted_rotation
        @ ground_truth_rotation.T
    )

    cosine_value = (
        np.trace(rotation_error_matrix) - 1.0
    ) * 0.5

    cosine_value = float(
        np.clip(
            cosine_value,
            -1.0,
            1.0,
        )
    )

    error_rad = float(
        np.arccos(cosine_value)
    )

    return float(
        np.degrees(error_rad)
    )


def _compute_translation_error_m(
    predicted_pose: NDArray[np.float32],
    ground_truth_pose: NDArray[np.float32],
) -> float:
    """예측과 GT translation의 Euclidean 오차."""

    predicted_translation = (
        predicted_pose[:3, 3]
        .astype(np.float64)
    )

    ground_truth_translation = (
        ground_truth_pose[:3, 3]
        .astype(np.float64)
    )

    return float(
        np.linalg.norm(
            predicted_translation
            - ground_truth_translation
        )
    )


def _compute_rotation_magnitude_deg(
    pose: NDArray[np.float32],
) -> float:
    """Return the SO(3) rotation magnitude represented by a pose."""

    identity_pose = np.eye(
        4,
        dtype=np.float32,
    )

    return _compute_rotation_error_deg(
        predicted_pose=pose,
        ground_truth_pose=identity_pose,
    )


def _translation_components_m(
    pose: NDArray[np.float32],
) -> tuple[float, float, float]:
    translation = pose[:3, 3].astype(
        np.float64
    )

    return (
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
    )


def _object_anchored_translation_error_components_m(
    predicted_relative_pose: NDArray[np.float32],
    reference_absolute_pose: NDArray[np.float32],
    query_absolute_pose: NDArray[np.float32],
) -> tuple[float, float, float]:
    """물체 중심이 실제로 얼마나 어긋났는지를 query 카메라 좌표계에서 잰다.

    두 상대 pose의 translation 열을 그대로 빼면 회전 오차가 translation
    오차로 섞여 들어간다. 상대 pose의 translation은 reference 카메라
    원점을 기준으로 정의되는데 물체는 그 원점에서 멀리(LINEMOD 기준
    0.8~1.0 m) 떨어져 있으므로, 회전 오차 theta는 그 차이를 대략
    2 * distance * sin(theta / 2)만큼 부풀린다.

    예측을 GT reference 절대 pose에 anchor하면
    T_query_from_object_hat = H_hat @ T_reference_from_object_gt가 되고,
    이것을 GT query 절대 pose와 비교하면 물체가 실제로 놓인 위치의
    오차만 남는다. 6D pose 문헌의 TE 및 ADD와 같은 기준이다.
    """

    predicted_query_absolute = (
        predicted_relative_pose.astype(np.float64)
        @ reference_absolute_pose.astype(np.float64)
    )

    difference = (
        predicted_query_absolute[:3, 3]
        - query_absolute_pose[:3, 3].astype(np.float64)
    )

    return (
        float(difference[0]),
        float(difference[1]),
        float(difference[2]),
    )


def _evaluate_thresholds(
    *,
    rotation_error_deg: float,
    translation_error_m: float,
    thresholds: Sequence[RelativePoseThreshold],
) -> tuple[ThresholdEvaluation, ...]:
    """각 threshold의 성공 여부를 계산한다."""

    if not thresholds:
        raise ValueError(
            "평가 threshold가 하나도 없습니다."
        )

    threshold_names: set[str] = set()
    evaluations: list[
        ThresholdEvaluation
    ] = []

    for threshold in thresholds:
        if threshold.name in threshold_names:
            raise ValueError(
                "중복된 threshold 이름입니다: "
                f"{threshold.name}"
            )

        threshold_names.add(
            threshold.name
        )

        rotation_passed = (
            rotation_error_deg
            <= threshold.maximum_rotation_error_deg
        )

        translation_passed = (
            translation_error_m
            <= threshold.maximum_translation_error_m
        )

        evaluations.append(
            ThresholdEvaluation(
                threshold=threshold,
                rotation_passed=rotation_passed,
                translation_passed=(
                    translation_passed
                ),
                success=(
                    rotation_passed
                    and translation_passed
                ),
            )
        )

    return tuple(evaluations)


def evaluate_relative_pose(
    *,
    predicted_relative_pose: NDArray[np.floating],
    reference_frame: BOPFrameGTSpec,
    query_frame: BOPFrameGTSpec,
    output_directory: Path,
    thresholds: Sequence[RelativePoseThreshold] = (
        DEFAULT_THRESHOLDS
    ),
) -> RelativePoseEvaluationResult:
    """
    최종 예측 상대 pose를 BOP-LINEMOD GT와 평가한다.

    GT와 CAD는 이 함수에서만 사용하며,
    추론 파이프라인에는 전달하지 않는다.
    """

    if reference_frame.object_id != query_frame.object_id:
        raise ValueError(
            "Reference와 Query object_id가 다릅니다: "
            f"reference={reference_frame.object_id}, "
            f"query={query_frame.object_id}"
        )

    same_instance = (
        reference_frame.split == query_frame.split
        and reference_frame.scene_id
        == query_frame.scene_id
        and reference_frame.image_id
        == query_frame.image_id
        and reference_frame.instance_index
        == query_frame.instance_index
    )

    if same_instance:
        raise ValueError(
            "Reference와 Query가 동일한 프레임과 "
            "동일한 instance입니다."
        )

    predicted_pose = _validate_rigid_pose(
        predicted_relative_pose,
        "predicted_relative_pose",
    )

    reference_absolute_pose = (
        load_bop_linemod_absolute_gt_pose(
            reference_frame
        )
    )

    query_absolute_pose = (
        load_bop_linemod_absolute_gt_pose(
            query_frame
        )
    )

    ground_truth_relative_pose = (
        build_ground_truth_relative_pose(
            reference_absolute_pose=(
                reference_absolute_pose
            ),
            query_absolute_pose=(
                query_absolute_pose
            ),
        )
    )

    rotation_error_deg = (
        _compute_rotation_error_deg(
            predicted_pose=predicted_pose,
            ground_truth_pose=(
                ground_truth_relative_pose
            ),
        )
    )

    # 주 지표는 물체 중심 변위(TE)이다. 상대 pose translation 열의
    # 단순 차이는 회전 오차를 translation 오차로 부풀리므로
    # (_object_anchored_translation_error_components_m 설명 참고)
    # 진단용으로만 따로 기록한다.
    (
        translation_error_x_m,
        translation_error_y_m,
        translation_error_z_m,
    ) = _object_anchored_translation_error_components_m(
        predicted_relative_pose=predicted_pose,
        reference_absolute_pose=reference_absolute_pose,
        query_absolute_pose=query_absolute_pose,
    )

    translation_error_m = float(
        np.linalg.norm(
            (
                translation_error_x_m,
                translation_error_y_m,
                translation_error_z_m,
            )
        )
    )

    translation_error_cm = (
        translation_error_m * 100.0
    )

    reference_origin_translation_error_m = (
        _compute_translation_error_m(
            predicted_pose=predicted_pose,
            ground_truth_pose=(
                ground_truth_relative_pose
            ),
        )
    )

    reference_origin_translation_error_cm = (
        reference_origin_translation_error_m * 100.0
    )

    translation_error_x_cm = (
        translation_error_x_m * 100.0
    )
    translation_error_y_cm = (
        translation_error_y_m * 100.0
    )
    translation_error_z_cm = (
        translation_error_z_m * 100.0
    )

    estimated_rotation_deg = (
        _compute_rotation_magnitude_deg(
            predicted_pose
        )
    )
    estimated_translation_m = float(
        np.linalg.norm(predicted_pose[:3, 3])
    )
    estimated_translation_cm = (
        estimated_translation_m * 100.0
    )
    (
        estimated_tx_m,
        estimated_ty_m,
        estimated_tz_m,
    ) = _translation_components_m(predicted_pose)

    ground_truth_relative_rotation_deg = (
        _compute_rotation_magnitude_deg(
            ground_truth_relative_pose
        )
    )
    ground_truth_relative_translation_m = float(
        np.linalg.norm(
            ground_truth_relative_pose[:3, 3]
        )
    )
    ground_truth_relative_translation_cm = (
        ground_truth_relative_translation_m
        * 100.0
    )
    (
        ground_truth_tx_m,
        ground_truth_ty_m,
        ground_truth_tz_m,
    ) = _translation_components_m(
        ground_truth_relative_pose
    )

    catastrophic_failure = (
        rotation_error_deg > 20.0
        or translation_error_m > 0.10
    )

    threshold_results = (
        _evaluate_thresholds(
            rotation_error_deg=(
                rotation_error_deg
            ),
            translation_error_m=(
                translation_error_m
            ),
            thresholds=thresholds,
        )
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

    predicted_pose_path = (
        output_directory
        / "predicted_relative_pose.npy"
    )

    ground_truth_pose_path = (
        output_directory
        / "ground_truth_relative_pose.npy"
    )

    metadata_path = (
        output_directory
        / "relative_pose_evaluation.json"
    )

    np.save(
        predicted_pose_path,
        predicted_pose,
        allow_pickle=False,
    )

    np.save(
        ground_truth_pose_path,
        ground_truth_relative_pose,
        allow_pickle=False,
    )

    metadata = {
        "dataset": "BOP-LINEMOD",
        "evaluation_type": "relative_pose",
        "pose_convention": (
            "T_query_camera_from_reference_camera"
        ),
        "translation_unit": "meter",
        "symmetry_aware": False,
        "reference_frame": {
            "split": reference_frame.split,
            "scene_id": reference_frame.scene_id,
            "image_id": reference_frame.image_id,
            "object_id": reference_frame.object_id,
            "instance_index": (
                reference_frame.instance_index
            ),
            "scene_gt_path": str(
                reference_frame.scene_gt_path
            ),
        },
        "query_frame": {
            "split": query_frame.split,
            "scene_id": query_frame.scene_id,
            "image_id": query_frame.image_id,
            "object_id": query_frame.object_id,
            "instance_index": (
                query_frame.instance_index
            ),
            "scene_gt_path": str(
                query_frame.scene_gt_path
            ),
        },
        "rotation_error_deg": (
            rotation_error_deg
        ),
        "translation_error_m": (
            translation_error_m
        ),
        "translation_error_cm": (
            translation_error_cm
        ),
        "translation_error_x_m": (
            translation_error_x_m
        ),
        "translation_error_y_m": (
            translation_error_y_m
        ),
        "translation_error_z_m": (
            translation_error_z_m
        ),
        "translation_error_x_cm": (
            translation_error_x_cm
        ),
        "translation_error_y_cm": (
            translation_error_y_cm
        ),
        "translation_error_z_cm": (
            translation_error_z_cm
        ),
        "translation_error_definition": (
            "object-centre displacement: "
            "T_query_from_object_hat = H_hat @ "
            "T_reference_from_object_gt, compared against "
            "T_query_from_object_gt"
        ),
        "reference_origin_translation_error_m": (
            reference_origin_translation_error_m
        ),
        "reference_origin_translation_error_cm": (
            reference_origin_translation_error_cm
        ),
        "reference_origin_translation_error_definition": (
            "diagnostic only: raw difference of the two relative "
            "poses' translation columns, which is anchored at the "
            "reference camera origin and therefore inflates with "
            "rotation error; never used for pass/fail"
        ),
        "estimated_rotation_deg": (
            estimated_rotation_deg
        ),
        "estimated_translation_m": (
            estimated_translation_m
        ),
        "estimated_translation_cm": (
            estimated_translation_cm
        ),
        "estimated_translation_m_xyz": [
            estimated_tx_m,
            estimated_ty_m,
            estimated_tz_m,
        ],
        "ground_truth_relative_rotation_deg": (
            ground_truth_relative_rotation_deg
        ),
        "ground_truth_relative_translation_m": (
            ground_truth_relative_translation_m
        ),
        "ground_truth_relative_translation_cm": (
            ground_truth_relative_translation_cm
        ),
        "ground_truth_translation_m_xyz": [
            ground_truth_tx_m,
            ground_truth_ty_m,
            ground_truth_tz_m,
        ],
        "catastrophic_failure": (
            catastrophic_failure
        ),
        "threshold_results": [
            {
                "name": item.threshold.name,
                "maximum_rotation_error_deg": (
                    item
                    .threshold
                    .maximum_rotation_error_deg
                ),
                "maximum_translation_error_m": (
                    item
                    .threshold
                    .maximum_translation_error_m
                ),
                "rotation_passed": (
                    item.rotation_passed
                ),
                "translation_passed": (
                    item.translation_passed
                ),
                "success": item.success,
            }
            for item in threshold_results
        ],
        "predicted_relative_pose": (
            predicted_pose.tolist()
        ),
        "ground_truth_relative_pose": (
            ground_truth_relative_pose.tolist()
        ),
        "predicted_pose_path": str(
            predicted_pose_path
        ),
        "ground_truth_pose_path": str(
            ground_truth_pose_path
        ),
        "gt_usage": (
            "evaluation only; not used during inference"
        ),
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

    return RelativePoseEvaluationResult(
        predicted_relative_pose=predicted_pose,
        ground_truth_relative_pose=(
            ground_truth_relative_pose
        ),
        rotation_error_deg=rotation_error_deg,
        translation_error_m=translation_error_m,
        translation_error_cm=translation_error_cm,
        translation_error_x_m=(
            translation_error_x_m
        ),
        translation_error_y_m=(
            translation_error_y_m
        ),
        translation_error_z_m=(
            translation_error_z_m
        ),
        translation_error_x_cm=(
            translation_error_x_cm
        ),
        translation_error_y_cm=(
            translation_error_y_cm
        ),
        translation_error_z_cm=(
            translation_error_z_cm
        ),
        reference_origin_translation_error_m=(
            reference_origin_translation_error_m
        ),
        reference_origin_translation_error_cm=(
            reference_origin_translation_error_cm
        ),
        estimated_rotation_deg=(
            estimated_rotation_deg
        ),
        estimated_translation_m=(
            estimated_translation_m
        ),
        estimated_translation_cm=(
            estimated_translation_cm
        ),
        estimated_tx_m=estimated_tx_m,
        estimated_ty_m=estimated_ty_m,
        estimated_tz_m=estimated_tz_m,
        ground_truth_relative_rotation_deg=(
            ground_truth_relative_rotation_deg
        ),
        ground_truth_relative_translation_m=(
            ground_truth_relative_translation_m
        ),
        ground_truth_relative_translation_cm=(
            ground_truth_relative_translation_cm
        ),
        ground_truth_tx_m=ground_truth_tx_m,
        ground_truth_ty_m=ground_truth_ty_m,
        ground_truth_tz_m=ground_truth_tz_m,
        catastrophic_failure=(
            catastrophic_failure
        ),
        threshold_results=threshold_results,
        predicted_pose_path=predicted_pose_path,
        ground_truth_pose_path=(
            ground_truth_pose_path
        ),
        metadata_path=metadata_path,
    )
