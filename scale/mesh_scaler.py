from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import trimesh

from core.types import ScaleInitializationResult
from scale.mesh_normalizer import MeshNormalizationResult


@dataclass(frozen=True)
class ScaledMeshCandidate:
    """
    하나의 등방성 scale이 적용된 mesh 후보.

    좌표 관계:
        p_scaled = scale_transform @ p_normalized

    정규화 mesh의 robust diagonal이 1이므로,
    scaled mesh의 robust diagonal은 scale_m에 대응한다.
    """

    candidate_index: int
    scale_m: float

    normalized_mesh_path: Path
    scaled_mesh_path: Path
    metadata_path: Path

    scale_transform: NDArray[np.float64]
    artifact_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if self.candidate_index < 0:
            raise ValueError(
                "candidate_index는 0 이상이어야 합니다: "
                f"{self.candidate_index}"
            )

        if (
            not np.isfinite(self.scale_m)
            or self.scale_m <= 0.0
        ):
            raise ValueError(
                "scale_m은 유한한 양수여야 합니다: "
                f"{self.scale_m}"
            )

        if self.scale_transform.shape != (4, 4):
            raise ValueError(
                "scale_transform shape은 "
                "(4, 4)이어야 합니다: "
                f"{self.scale_transform.shape}"
            )

        if not np.isfinite(
            self.scale_transform
        ).all():
            raise ValueError(
                "scale_transform에 NaN 또는 Inf가 있습니다."
            )

        if not self.scaled_mesh_path.is_file():
            raise FileNotFoundError(
                "Scaled mesh 파일이 없습니다: "
                f"{self.scaled_mesh_path}"
            )

        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                "Scale metadata 파일이 없습니다: "
                f"{self.metadata_path}"
            )


def _validate_scale_candidates(
    scale_result: ScaleInitializationResult,
) -> None:
    """Scale 후보값을 검증한다."""

    if not scale_result.scale_candidates_m:
        raise ValueError(
            "Scale 후보가 하나도 없습니다."
        )

    for scale_m in scale_result.scale_candidates_m:
        if (
            not np.isfinite(scale_m)
            or scale_m <= 0.0
        ):
            raise ValueError(
                "Scale 후보는 유한한 양수여야 합니다: "
                f"{scale_m}"
            )


def _load_normalized_scene(
    normalized_mesh_path: Path,
) -> trimesh.Scene:
    """정규화 mesh를 Trimesh Scene으로 읽는다."""

    normalized_mesh_path = (
        Path(normalized_mesh_path)
        .expanduser()
        .resolve()
    )

    if not normalized_mesh_path.is_file():
        raise FileNotFoundError(
            "정규화 mesh 파일이 없습니다: "
            f"{normalized_mesh_path}"
        )

    try:
        scene = trimesh.load_scene(
            file_obj=normalized_mesh_path,
            process=False,
        )
    except Exception as error:
        raise RuntimeError(
            "정규화 mesh를 읽지 못했습니다: "
            f"{normalized_mesh_path}"
        ) from error

    if not isinstance(scene, trimesh.Scene):
        raise TypeError(
            "Mesh loader 결과가 Scene이 아닙니다: "
            f"{type(scene)}"
        )

    if scene.is_empty:
        raise ValueError(
            "정규화 mesh scene이 비어 있습니다: "
            f"{normalized_mesh_path}"
        )

    mesh_count = sum(
        isinstance(geometry, trimesh.Trimesh)
        for geometry in scene.geometry.values()
    )

    if mesh_count == 0:
        raise ValueError(
            "Scene에 triangular mesh가 없습니다: "
            f"{normalized_mesh_path}"
        )

    return scene


def _build_scale_transform(
    scale_m: float,
) -> NDArray[np.float64]:
    """
    등방성 scale의 4×4 변환 행렬을 만든다.

    정규화 mesh 좌표는 무차원이고,
    scale 적용 후 좌표 단위는 meter가 된다.
    """

    transform = np.eye(
        4,
        dtype=np.float64,
    )

    transform[0, 0] = scale_m
    transform[1, 1] = scale_m
    transform[2, 2] = scale_m

    return np.ascontiguousarray(
        transform,
        dtype=np.float64,
    )


def _apply_scale_to_scene(
    source_scene: trimesh.Scene,
    scale_transform: NDArray[np.float64],
) -> trimesh.Scene:
    """원본 scene을 복사한 뒤 등방성 scale을 적용한다."""

    scaled_scene = source_scene.copy()

    try:
        scaled_scene.apply_transform(
            scale_transform
        )
    except Exception as error:
        raise RuntimeError(
            "Mesh scene에 scale transform을 "
            "적용하지 못했습니다."
        ) from error

    return scaled_scene


def _prepare_candidate_directory(
    output_directory: Path,
    candidate_index: int,
) -> Path:
    """
    Scale 후보별 출력 폴더를 준비한다.

    동일 후보를 재실행하면 해당 후보 폴더만 교체한다.
    """

    candidate_directory = (
        output_directory
        / f"candidate_{candidate_index:02d}"
    )

    if candidate_directory.exists():
        if not candidate_directory.is_dir():
            raise RuntimeError(
                "후보 출력 경로가 디렉터리가 아닙니다: "
                f"{candidate_directory}"
            )

        shutil.rmtree(candidate_directory)

    candidate_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    return candidate_directory


def _export_scaled_scene(
    scaled_scene: trimesh.Scene,
    normalized_mesh_path: Path,
    candidate_directory: Path,
) -> Path:
    """
    정규화 mesh와 동일한 확장자로 scaled mesh를 저장한다.

    예:
        mesh_normalized.obj
        → candidate_00/mesh_scaled.obj

        mesh_normalized.glb
        → candidate_00/mesh_scaled.glb
    """

    mesh_suffix = normalized_mesh_path.suffix.lower()

    if not mesh_suffix:
        raise ValueError(
            "정규화 mesh에 확장자가 없습니다: "
            f"{normalized_mesh_path}"
        )

    scaled_mesh_path = (
        candidate_directory
        / f"mesh_scaled{mesh_suffix}"
    )

    try:
        scaled_scene.export(
            file_obj=scaled_mesh_path,
            file_type=mesh_suffix.lstrip("."),
        )
    except Exception as error:
        raise RuntimeError(
            "Scale이 적용된 mesh를 저장하지 못했습니다: "
            f"{scaled_mesh_path}"
        ) from error

    if not scaled_mesh_path.is_file():
        raise FileNotFoundError(
            "Mesh export가 종료됐지만 대표 출력 파일이 "
            "생성되지 않았습니다: "
            f"{scaled_mesh_path}"
        )

    return scaled_mesh_path.resolve()


def _collect_export_artifacts(
    candidate_directory: Path,
    scaled_mesh_path: Path,
) -> tuple[Path, ...]:
    """
    OBJ의 MTL·texture 등 대표 mesh 외의 파일을 수집한다.
    """

    artifact_paths = tuple(
        sorted(
            path.resolve()
            for path in candidate_directory.rglob("*")
            if (
                path.is_file()
                and path.resolve()
                != scaled_mesh_path.resolve()
            )
        )
    )

    return artifact_paths


def _save_candidate_metadata(
    candidate_directory: Path,
    *,
    candidate_index: int,
    scale_m: float,
    normalized_mesh_path: Path,
    scaled_mesh_path: Path,
    scale_transform: NDArray[np.float64],
    visible_diagonal_m: float,
) -> Path:
    """Scale 후보 정보를 JSON으로 저장한다."""

    metadata_path = (
        candidate_directory
        / "scale_candidate.json"
    )

    metadata: dict[str, Any] = {
        "candidate_index": candidate_index,
        "scale_m": scale_m,
        "visible_diagonal_m": visible_diagonal_m,
        "normalized_mesh_path": str(
            normalized_mesh_path
        ),
        "scaled_mesh_path": str(
            scaled_mesh_path
        ),
        "scale_transform": (
            scale_transform.tolist()
        ),
        "coordinate_rule": (
            "p_scaled_m = "
            "scale_transform @ p_normalized"
        ),
        "unit": "meter",
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
            "Scale candidate metadata가 "
            "저장되지 않았습니다: "
            f"{metadata_path}"
        )

    return metadata_path.resolve()


def build_scaled_mesh_candidates(
    normalization_result: MeshNormalizationResult,
    scale_result: ScaleInitializationResult,
    output_directory: Path,
) -> tuple[ScaledMeshCandidate, ...]:
    """
    정규화 mesh에 모든 초기 scale 후보를 적용한다.

    Parameters
    ----------
    normalization_result:
        Robust diagonal이 1로 정규화된 mesh 정보.

    scale_result:
        Masked depth에서 계산한 초기 scale 후보.

    output_directory:
        Scale 후보별 mesh를 저장할 루트 폴더.

    Returns
    -------
    tuple[ScaledMeshCandidate, ...]
        FoundationPose에 순차적으로 전달할 scaled mesh 후보.
    """

    _validate_scale_candidates(
        scale_result
    )

    normalized_mesh_path = (
        normalization_result
        .normalized_mesh_path
        .expanduser()
        .resolve()
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

    source_scene = _load_normalized_scene(
        normalized_mesh_path
    )

    candidates: list[ScaledMeshCandidate] = []

    for candidate_index, scale_m in enumerate(
        scale_result.scale_candidates_m
    ):
        scale_m = float(scale_m)

        candidate_directory = (
            _prepare_candidate_directory(
                output_directory=output_directory,
                candidate_index=candidate_index,
            )
        )

        scale_transform = (
            _build_scale_transform(
                scale_m=scale_m
            )
        )

        scaled_scene = _apply_scale_to_scene(
            source_scene=source_scene,
            scale_transform=scale_transform,
        )

        scaled_mesh_path = _export_scaled_scene(
            scaled_scene=scaled_scene,
            normalized_mesh_path=(
                normalized_mesh_path
            ),
            candidate_directory=(
                candidate_directory
            ),
        )

        artifact_paths = (
            _collect_export_artifacts(
                candidate_directory=(
                    candidate_directory
                ),
                scaled_mesh_path=scaled_mesh_path,
            )
        )

        metadata_path = (
            _save_candidate_metadata(
                candidate_directory=(
                    candidate_directory
                ),
                candidate_index=candidate_index,
                scale_m=scale_m,
                normalized_mesh_path=(
                    normalized_mesh_path
                ),
                scaled_mesh_path=scaled_mesh_path,
                scale_transform=scale_transform,
                visible_diagonal_m=(
                    scale_result.visible_diagonal_m
                ),
            )
        )

        candidate = ScaledMeshCandidate(
            candidate_index=candidate_index,
            scale_m=scale_m,
            normalized_mesh_path=(
                normalized_mesh_path
            ),
            scaled_mesh_path=scaled_mesh_path,
            metadata_path=metadata_path,
            scale_transform=scale_transform,
            artifact_paths=artifact_paths,
        )

        candidates.append(candidate)

    if not candidates:
        raise RuntimeError(
            "Scaled mesh 후보가 생성되지 않았습니다."
        )

    return tuple(candidates)