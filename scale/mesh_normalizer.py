from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import trimesh

from core.types import MeshGenerationResult


@dataclass(frozen=True)
class MeshNormalizationResult:
    """
    Mesh unit-size normalization 결과.

    normalization_transform:
        원본 proxy 좌표를 정규화 proxy 좌표로 변환한다.

        p_normalized =
            normalization_transform @ p_original

    inverse_normalization_transform:
        정규화 proxy 좌표를 원본 proxy 좌표로 복원한다.
    """

    input_mesh_path: Path
    normalized_mesh_path: Path
    metadata_path: Path

    robust_center: NDArray[np.float64]
    robust_diagonal: float

    normalization_transform: NDArray[np.float64]
    inverse_normalization_transform: NDArray[np.float64]

    normalized_robust_diagonal: float

    def __post_init__(self) -> None:
        if self.robust_center.shape != (3,):
            raise ValueError(
                "robust_center shape은 (3,)이어야 합니다: "
                f"{self.robust_center.shape}"
            )

        if self.normalization_transform.shape != (4, 4):
            raise ValueError(
                "normalization_transform shape은 "
                "(4, 4)이어야 합니다: "
                f"{self.normalization_transform.shape}"
            )

        if (
            self.inverse_normalization_transform.shape
            != (4, 4)
        ):
            raise ValueError(
                "inverse_normalization_transform shape은 "
                "(4, 4)이어야 합니다: "
                f"{self.inverse_normalization_transform.shape}"
            )

        if (
            not np.isfinite(self.robust_diagonal)
            or self.robust_diagonal <= 0.0
        ):
            raise ValueError(
                "robust_diagonal은 유한한 양수여야 합니다: "
                f"{self.robust_diagonal}"
            )

        if not np.isfinite(
            self.normalization_transform
        ).all():
            raise ValueError(
                "normalization_transform에 "
                "NaN 또는 Inf가 있습니다."
            )

        if not np.isfinite(
            self.inverse_normalization_transform
        ).all():
            raise ValueError(
                "inverse_normalization_transform에 "
                "NaN 또는 Inf가 있습니다."
            )


def _validate_parameters(
    quantile_low: float,
    quantile_high: float,
    sample_count: int,
    random_seed: int,
    normalization_tolerance: float,
) -> None:
    """Mesh normalization 파라미터를 검증한다."""

    if not 0.0 <= quantile_low < quantile_high <= 1.0:
        raise ValueError(
            "quantile 범위가 올바르지 않습니다: "
            f"low={quantile_low}, high={quantile_high}"
        )

    if isinstance(sample_count, bool):
        raise TypeError(
            "sample_count는 정수여야 합니다."
        )

    if sample_count < 100:
        raise ValueError(
            "sample_count는 100 이상이어야 합니다: "
            f"{sample_count}"
        )

    if isinstance(random_seed, bool):
        raise TypeError(
            "random_seed는 정수여야 합니다."
        )

    if random_seed < 0:
        raise ValueError(
            "random_seed는 0 이상이어야 합니다: "
            f"{random_seed}"
        )

    if (
        not np.isfinite(normalization_tolerance)
        or normalization_tolerance <= 0.0
    ):
        raise ValueError(
            "normalization_tolerance은 "
            "유한한 양수여야 합니다: "
            f"{normalization_tolerance}"
        )


def _load_mesh_scene(
    mesh_path: Path,
) -> trimesh.Scene:
    """
    생성기 출력을 Trimesh Scene으로 읽는다.

    OBJ, GLB 등 여러 geometry가 포함된 출력도
    Scene 구조로 유지한다.
    """

    mesh_path = (
        Path(mesh_path)
        .expanduser()
        .resolve()
    )

    if not mesh_path.is_file():
        raise FileNotFoundError(
            f"Mesh 파일이 없습니다: {mesh_path}"
        )

    if not mesh_path.suffix:
        raise ValueError(
            "Mesh 파일에 확장자가 없습니다: "
            f"{mesh_path}"
        )

    try:
        scene = trimesh.load_scene(
            file_obj=mesh_path,
            process=False,
        )
    except Exception as error:
        raise RuntimeError(
            "Trimesh가 mesh를 읽지 못했습니다: "
            f"{mesh_path}"
        ) from error

    if not isinstance(scene, trimesh.Scene):
        raise TypeError(
            "Mesh loader 결과가 Scene이 아닙니다: "
            f"type={type(scene)}"
        )

    if scene.is_empty:
        raise ValueError(
            f"읽은 mesh scene이 비어 있습니다: {mesh_path}"
        )

    mesh_geometry_count = sum(
        isinstance(geometry, trimesh.Trimesh)
        for geometry in scene.geometry.values()
    )

    if mesh_geometry_count == 0:
        raise ValueError(
            "Scene에 triangular mesh geometry가 없습니다: "
            f"{mesh_path}"
        )

    return scene


def _scene_to_statistics_mesh(
    scene: trimesh.Scene,
) -> trimesh.Trimesh:
    """
    Scene transform을 적용한 geometry를 하나의 통계용 mesh로 만든다.

    이 mesh는 robust center와 diagonal 계산에만 사용한다.
    실제 출력 scene의 material 구조를 변경하지 않는다.
    """

    if not hasattr(scene, "to_mesh"):
        raise RuntimeError(
            "현재 trimesh 버전에 Scene.to_mesh()가 없습니다. "
            "trimesh를 최신 안정 버전으로 업데이트해야 합니다."
        )

    try:
        statistics_mesh = scene.to_mesh()
    except Exception as error:
        raise RuntimeError(
            "Scene geometry를 통계용 mesh로 "
            "결합하지 못했습니다."
        ) from error

    if not isinstance(
        statistics_mesh,
        trimesh.Trimesh,
    ):
        raise TypeError(
            "Scene.to_mesh() 결과가 Trimesh가 아닙니다: "
            f"{type(statistics_mesh)}"
        )

    if statistics_mesh.is_empty:
        raise ValueError(
            "통계용 mesh가 비어 있습니다."
        )

    if len(statistics_mesh.vertices) < 3:
        raise ValueError(
            "Mesh vertex가 너무 적습니다: "
            f"{len(statistics_mesh.vertices)}"
        )

    if len(statistics_mesh.faces) < 1:
        raise ValueError(
            "Mesh face가 없습니다."
        )

    vertices = np.asarray(
        statistics_mesh.vertices,
        dtype=np.float64,
    )

    if not np.isfinite(vertices).all():
        raise ValueError(
            "Mesh vertex에 NaN 또는 Inf가 있습니다."
        )

    if (
        not np.isfinite(statistics_mesh.area)
        or statistics_mesh.area <= 0.0
    ):
        raise ValueError(
            "Mesh 표면적이 유효하지 않습니다: "
            f"{statistics_mesh.area}"
        )

    return statistics_mesh


def _sample_mesh_surface(
    mesh: trimesh.Trimesh,
    sample_count: int,
    random_seed: int,
) -> NDArray[np.float64]:
    """
    Triangle 면적에 비례해 mesh 표면을 균일 샘플링한다.

    Vertex 밀도가 불균일한 생성기 사이의 차이를 줄이기 위해
    raw vertex 대신 surface sample을 사용한다.
    """

    try:
        sampled_points, _ = (
            trimesh.sample.sample_surface(
                mesh=mesh,
                count=sample_count,
                seed=random_seed,
            )
        )
    except TypeError as error:
        raise RuntimeError(
            "설치된 trimesh 버전이 "
            "sample_surface(..., seed=...)를 지원하지 않습니다."
        ) from error
    except Exception as error:
        raise RuntimeError(
            "Mesh 표면 샘플링에 실패했습니다."
        ) from error

    sampled_points = np.asarray(
        sampled_points,
        dtype=np.float64,
    )

    if sampled_points.shape != (
        sample_count,
        3,
    ):
        raise ValueError(
            "표면 sample shape이 올바르지 않습니다: "
            f"expected={(sample_count, 3)}, "
            f"actual={sampled_points.shape}"
        )

    if not np.isfinite(sampled_points).all():
        raise ValueError(
            "표면 sample에 NaN 또는 Inf가 있습니다."
        )

    return np.ascontiguousarray(
        sampled_points,
        dtype=np.float64,
    )


def _compute_robust_bounds(
    sampled_points: NDArray[np.float64],
    quantile_low: float,
    quantile_high: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """표면 sample의 robust axis-aligned bounds를 계산한다."""

    lower_bound = np.quantile(
        sampled_points,
        quantile_low,
        axis=0,
    )

    upper_bound = np.quantile(
        sampled_points,
        quantile_high,
        axis=0,
    )

    lower_bound = np.asarray(
        lower_bound,
        dtype=np.float64,
    )

    upper_bound = np.asarray(
        upper_bound,
        dtype=np.float64,
    )

    if not np.isfinite(lower_bound).all():
        raise ValueError(
            "Robust lower bound가 유효하지 않습니다."
        )

    if not np.isfinite(upper_bound).all():
        raise ValueError(
            "Robust upper bound가 유효하지 않습니다."
        )

    if np.any(upper_bound <= lower_bound):
        raise ValueError(
            "Mesh의 robust extent가 유효하지 않습니다: "
            f"lower={lower_bound}, "
            f"upper={upper_bound}"
        )

    return lower_bound, upper_bound


def _compute_normalization(
    lower_bound: NDArray[np.float64],
    upper_bound: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    float,
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Robust center와 unit-diagonal normalization transform을 계산한다.

    normalized_point =
        (original_point - robust_center)
        / robust_diagonal
    """

    robust_center = (
        lower_bound + upper_bound
    ) * 0.5

    robust_extent = (
        upper_bound - lower_bound
    )

    robust_diagonal = float(
        np.linalg.norm(robust_extent)
    )

    if (
        not np.isfinite(robust_diagonal)
        or robust_diagonal <= 0.0
    ):
        raise ValueError(
            "Robust diagonal이 유효하지 않습니다: "
            f"{robust_diagonal}"
        )

    inverse_diagonal = 1.0 / robust_diagonal

    normalization_transform = np.eye(
        4,
        dtype=np.float64,
    )

    normalization_transform[:3, :3] *= (
        inverse_diagonal
    )

    normalization_transform[:3, 3] = (
        -robust_center * inverse_diagonal
    )

    inverse_normalization_transform = (
        np.linalg.inv(normalization_transform)
    )

    return (
        np.ascontiguousarray(
            robust_center,
            dtype=np.float64,
        ),
        robust_diagonal,
        np.ascontiguousarray(
            normalization_transform,
            dtype=np.float64,
        ),
        np.ascontiguousarray(
            inverse_normalization_transform,
            dtype=np.float64,
        ),
    )


def _transform_points(
    points: NDArray[np.float64],
    transform: NDArray[np.float64],
) -> NDArray[np.float64]:
    """3D point 배열에 4×4 homogeneous transform을 적용한다."""

    rotation_scale = transform[:3, :3]
    translation = transform[:3, 3]

    transformed_points = (
        points @ rotation_scale.T
        + translation
    )

    return np.ascontiguousarray(
        transformed_points,
        dtype=np.float64,
    )


def _compute_robust_diagonal(
    points: NDArray[np.float64],
    quantile_low: float,
    quantile_high: float,
) -> float:
    """Point set의 robust diagonal을 계산한다."""

    lower_bound, upper_bound = (
        _compute_robust_bounds(
            sampled_points=points,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
        )
    )

    robust_diagonal = float(
        np.linalg.norm(
            upper_bound - lower_bound
        )
    )

    return robust_diagonal


def _apply_normalization_to_scene(
    scene: trimesh.Scene,
    normalization_transform: NDArray[np.float64],
) -> trimesh.Scene:
    """
    원본 scene을 복사한 뒤 normalization transform을 적용한다.

    원본 mesh 객체는 수정하지 않는다.
    """

    normalized_scene = scene.copy()

    try:
        normalized_scene.apply_transform(
            normalization_transform
        )
    except Exception as error:
        raise RuntimeError(
            "Mesh scene에 normalization transform을 "
            "적용하지 못했습니다."
        ) from error

    return normalized_scene


def _export_normalized_scene(
    normalized_scene: trimesh.Scene,
    input_mesh_path: Path,
    output_directory: Path,
) -> Path:
    """
    입력 mesh의 확장자를 유지해 정규화 mesh를 저장한다.

    예:
        mesh.obj -> mesh_normalized.obj
        mesh.glb -> mesh_normalized.glb
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_suffix = input_mesh_path.suffix.lower()

    normalized_mesh_path = (
        output_directory
        / f"{input_mesh_path.stem}_normalized"
        f"{input_suffix}"
    )

    if normalized_mesh_path.is_file():
        normalized_mesh_path.unlink()

    try:
        normalized_scene.export(
            file_obj=normalized_mesh_path,
        )
    except Exception as error:
        raise RuntimeError(
            "정규화 mesh를 원본 확장자로 "
            "내보내지 못했습니다: "
            f"{normalized_mesh_path}"
        ) from error

    if not normalized_mesh_path.is_file():
        raise FileNotFoundError(
            "Mesh export가 종료됐지만 대표 출력이 "
            "생성되지 않았습니다: "
            f"{normalized_mesh_path}"
        )

    return normalized_mesh_path.resolve()


def _save_normalization_metadata(
    output_directory: Path,
    *,
    generator_name: str,
    input_mesh_path: Path,
    normalized_mesh_path: Path,
    robust_center: NDArray[np.float64],
    robust_diagonal: float,
    normalization_transform: NDArray[np.float64],
    inverse_normalization_transform: NDArray[np.float64],
    normalized_robust_diagonal: float,
    quantile_low: float,
    quantile_high: float,
    sample_count: int,
    random_seed: int,
) -> Path:
    """Mesh normalization 정보를 JSON으로 저장한다."""

    metadata_path = (
        output_directory
        / "mesh_normalization.json"
    )

    metadata: dict[str, Any] = {
        "generator_name": generator_name,
        "input_mesh_path": str(input_mesh_path),
        "normalized_mesh_path": str(
            normalized_mesh_path
        ),
        "robust_center": robust_center.tolist(),
        "robust_diagonal": robust_diagonal,
        "normalization_factor": (
            1.0 / robust_diagonal
        ),
        "normalization_transform": (
            normalization_transform.tolist()
        ),
        "inverse_normalization_transform": (
            inverse_normalization_transform.tolist()
        ),
        "normalized_robust_diagonal": (
            normalized_robust_diagonal
        ),
        "quantile_low": quantile_low,
        "quantile_high": quantile_high,
        "sample_count": sample_count,
        "random_seed": random_seed,
        "coordinate_rule": (
            "p_normalized = "
            "normalization_transform @ p_original"
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

    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Mesh normalization metadata가 "
            "저장되지 않았습니다: "
            f"{metadata_path}"
        )

    return metadata_path.resolve()


def normalize_generated_mesh(
    mesh_result: MeshGenerationResult,
    output_directory: Path,
    *,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    sample_count: int = 100_000,
    random_seed: int = 0,
    normalization_tolerance: float = 1e-5,
) -> MeshNormalizationResult:
    """
    생성기 mesh를 robust diagonal 1로 정규화한다.

    Parameters
    ----------
    mesh_result:
        InstantMesh, TRELLIS 또는 SAM3D의 생성 결과.

    output_directory:
        정규화 mesh와 metadata를 저장할 폴더.

    quantile_low, quantile_high:
        극단값 제거에 사용할 quantile.
        기본값은 q1과 q99이다.

    sample_count:
        면적 기반 surface sample 개수.

    random_seed:
        결정론적 surface sampling seed.

    normalization_tolerance:
        정규화 후 robust diagonal이 1인지 검사할 허용 오차.

    Returns
    -------
    MeshNormalizationResult
        정규화 mesh 경로, center, scale 및 변환 행렬.
    """

    _validate_parameters(
        quantile_low=quantile_low,
        quantile_high=quantile_high,
        sample_count=sample_count,
        random_seed=random_seed,
        normalization_tolerance=(
            normalization_tolerance
        ),
    )

    input_mesh_path = (
        mesh_result.primary_output_path
        .expanduser()
        .resolve()
    )

    output_directory = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )

    scene = _load_mesh_scene(
        mesh_path=input_mesh_path
    )

    statistics_mesh = (
        _scene_to_statistics_mesh(scene)
    )

    sampled_points = _sample_mesh_surface(
        mesh=statistics_mesh,
        sample_count=sample_count,
        random_seed=random_seed,
    )

    lower_bound, upper_bound = (
        _compute_robust_bounds(
            sampled_points=sampled_points,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
        )
    )

    (
        robust_center,
        robust_diagonal,
        normalization_transform,
        inverse_normalization_transform,
    ) = _compute_normalization(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    normalized_sampled_points = (
        _transform_points(
            points=sampled_points,
            transform=normalization_transform,
        )
    )

    normalized_robust_diagonal = (
        _compute_robust_diagonal(
            points=normalized_sampled_points,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
        )
    )

    if not np.isclose(
        normalized_robust_diagonal,
        1.0,
        atol=normalization_tolerance,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Mesh normalization 검증에 실패했습니다: "
            f"normalized_diagonal="
            f"{normalized_robust_diagonal}, "
            f"expected=1.0"
        )

    normalized_scene = (
        _apply_normalization_to_scene(
            scene=scene,
            normalization_transform=(
                normalization_transform
            ),
        )
    )

    normalized_mesh_path = (
        _export_normalized_scene(
            normalized_scene=normalized_scene,
            input_mesh_path=input_mesh_path,
            output_directory=output_directory,
        )
    )

    metadata_path = (
        _save_normalization_metadata(
            output_directory=output_directory,
            generator_name=mesh_result.generator_name,
            input_mesh_path=input_mesh_path,
            normalized_mesh_path=normalized_mesh_path,
            robust_center=robust_center,
            robust_diagonal=robust_diagonal,
            normalization_transform=(
                normalization_transform
            ),
            inverse_normalization_transform=(
                inverse_normalization_transform
            ),
            normalized_robust_diagonal=(
                normalized_robust_diagonal
            ),
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            sample_count=sample_count,
            random_seed=random_seed,
        )
    )

    return MeshNormalizationResult(
        input_mesh_path=input_mesh_path,
        normalized_mesh_path=normalized_mesh_path,
        metadata_path=metadata_path,
        robust_center=robust_center,
        robust_diagonal=robust_diagonal,
        normalization_transform=(
            normalization_transform
        ),
        inverse_normalization_transform=(
            inverse_normalization_transform
        ),
        normalized_robust_diagonal=(
            normalized_robust_diagonal
        ),
    )