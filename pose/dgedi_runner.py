from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DGeDiRegistrationResult:
    proxy_pose_query_from_reference: np.ndarray
    relative_pose_query_from_reference: np.ndarray
    proxy_pose_path: Path
    relative_pose_path: Path
    metadata_path: Path


def _transform(
    value: Any,
    name: str,
) -> np.ndarray:
    value = np.asarray(
        value,
        dtype=np.float64,
    )

    if (
        value.shape != (4, 4)
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(
            f"Invalid {name}: shape={value.shape}"
        )

    if not np.allclose(
        value[3],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-5,
        rtol=0.0,
    ):
        raise ValueError(
            f"Invalid homogeneous row in {name}"
        )

    rotation = value[:3, :3]

    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=5e-3,
        rtol=0.0,
    ):
        raise ValueError(
            f"Non-orthonormal rotation in {name}"
        )

    if np.linalg.det(rotation) <= 0.0:
        raise ValueError(
            f"Invalid rotation determinant in {name}"
        )

    return value


def _apply(
    points: np.ndarray,
    pose: np.ndarray,
) -> np.ndarray:
    pose = _transform(
        pose,
        "point transform",
    )

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    return (
        points @ pose[:3, :3].T
        + pose[:3, 3]
    )


def _rotation_deg(
    pose: np.ndarray,
) -> float:
    pose = _transform(
        pose,
        "rotation transform",
    )

    cosine = np.clip(
        (
            np.trace(pose[:3, :3])
            - 1.0
        )
        / 2.0,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(cosine)
        )
    )


def _translation_m(
    pose: np.ndarray,
) -> float:
    pose = _transform(
        pose,
        "translation transform",
    )

    return float(
        np.linalg.norm(
            pose[:3, 3]
        )
    )


def _cloud(
    points: np.ndarray,
    o3d: Any,
) -> Any:
    cloud = o3d.geometry.PointCloud()

    cloud.points = (
        o3d.utility.Vector3dVector(
            np.asarray(
                points,
                dtype=np.float64,
            )
        )
    )

    return cloud


def _mesh_points(
    path: Path,
    sample_count: int,
    o3d: Any,
) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(
        str(path),
        enable_post_processing=True,
    )

    if len(mesh.vertices) == 0:
        raise ValueError(
            f"Mesh has no vertices: {path}"
        )

    if len(mesh.triangles) > 0:
        cloud = mesh.sample_points_uniformly(
            number_of_points=sample_count,
        )
    else:
        cloud = _cloud(
            np.asarray(mesh.vertices),
            o3d,
        )

        if len(cloud.points) > sample_count:
            cloud = (
                cloud
                .farthest_point_down_sample(
                    sample_count
                )
            )

    cloud = cloud.remove_non_finite_points()
    cloud = cloud.remove_duplicated_points()

    points = np.asarray(
        cloud.points,
        dtype=np.float64,
    ).copy()

    if len(points) < 256:
        raise ValueError(
            f"Too few points: {len(points)}"
        )

    return points


def _diameter(
    points: np.ndarray,
    block_size: int = 256,
) -> float:
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    maximum_squared = 0.0

    for start in range(
        0,
        len(points),
        block_size,
    ):
        block = points[
            start : start + block_size
        ]

        delta = (
            block[:, None, :]
            - points[None, :, :]
        )

        squared = np.einsum(
            "ijk,ijk->ij",
            delta,
            delta,
            optimize=True,
        )

        maximum_squared = max(
            maximum_squared,
            float(squared.max()),
        )

    diameter = float(
        np.sqrt(maximum_squared)
    )

    if (
        not np.isfinite(diameter)
        or diameter <= 0.0
    ):
        raise ValueError(
            f"Invalid diameter: {diameter}"
        )

    return diameter


def _feature_cloud(
    points: np.ndarray,
    diameter_m: float,
    o3d: Any,
) -> Any:
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    normalized = (
        points - points.mean(axis=0)
    ) / diameter_m

    return _cloud(
        normalized,
        o3d,
    )


def _unit_features(
    features: np.ndarray,
) -> np.ndarray:
    features = np.asarray(
        features,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        features,
        axis=1,
        keepdims=True,
    )

    return features / np.maximum(
        norms,
        1e-12,
    )


def _one_way_local_match(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_features: np.ndarray,
    target_features: np.ndarray,
    local_neighbors: int,
    nearest_neighbors_type: Any,
) -> tuple[np.ndarray, np.ndarray]:
    neighbor_count = min(
        max(1, local_neighbors),
        len(target_points),
    )

    spatial_index = (
        nearest_neighbors_type(
            n_neighbors=neighbor_count,
            algorithm="auto",
            n_jobs=-1,
        )
        .fit(target_points)
    )

    candidate_ids = (
        spatial_index.kneighbors(
            source_points,
            return_distance=False,
        )
    )

    candidate_features = (
        target_features[candidate_ids]
    )

    similarities = np.einsum(
        "nd,nkd->nk",
        source_features,
        candidate_features,
        optimize=True,
    )

    source_ids = np.arange(
        len(source_points),
        dtype=np.int64,
    )

    best_local = np.argmax(
        similarities,
        axis=1,
    )

    target_ids = candidate_ids[
        source_ids,
        best_local,
    ]

    scores = similarities[
        source_ids,
        best_local,
    ]

    return target_ids, scores


def _local_mutual_matches(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_features: np.ndarray,
    target_features: np.ndarray,
    local_neighbors: int,
    nearest_neighbors_type: Any,
) -> tuple[np.ndarray, np.ndarray]:
    source_features = _unit_features(
        source_features
    )

    target_features = _unit_features(
        target_features
    )

    source_to_target, source_scores = (
        _one_way_local_match(
            source_points,
            target_points,
            source_features,
            target_features,
            local_neighbors,
            nearest_neighbors_type,
        )
    )

    target_to_source, _ = (
        _one_way_local_match(
            target_points,
            source_points,
            target_features,
            source_features,
            local_neighbors,
            nearest_neighbors_type,
        )
    )

    source_ids = np.arange(
        len(source_points),
        dtype=np.int64,
    )

    mutual = (
        target_to_source[source_to_target]
        == source_ids
    )

    pairs = np.column_stack(
        (
            source_ids[mutual],
            source_to_target[mutual],
        )
    ).astype(
        np.int32,
        copy=False,
    )

    return pairs, source_scores[mutual]


def _result_dict(
    result: Any,
) -> dict[str, Any]:
    pose = _transform(
        result.transformation,
        "registration result",
    )

    return {
        "fitness": float(
            result.fitness
        ),
        "inlier_rmse_m": float(
            result.inlier_rmse
        ),
        "correspondence_count": int(
            len(result.correspondence_set)
        ),
        "rotation_deg": (
            _rotation_deg(pose)
        ),
        "translation_m": (
            _translation_m(pose)
        ),
        "pose": pose.tolist(),
    }


def _worker(
    args: argparse.Namespace,
) -> int:
    repository = (
        args.repository
        .expanduser()
        .resolve()
    )

    config_path = (
        args.config
        .expanduser()
        .resolve()
    )

    reference_mesh = (
        args.reference_mesh
        .expanduser()
        .resolve()
    )

    query_mesh = (
        args.query_mesh
        .expanduser()
        .resolve()
    )

    reference_self_path = (
        args.reference_self_pose
        .expanduser()
        .resolve()
    )

    query_self_path = (
        args.query_self_pose
        .expanduser()
        .resolve()
    )

    output = (
        args.output_directory
        .expanduser()
        .resolve()
    )

    for label, path, is_directory in (
        (
            "repository",
            repository,
            True,
        ),
        (
            "config",
            config_path,
            False,
        ),
        (
            "reference mesh",
            reference_mesh,
            False,
        ),
        (
            "query mesh",
            query_mesh,
            False,
        ),
        (
            "reference self pose",
            reference_self_path,
            False,
        ),
        (
            "query self pose",
            query_self_path,
            False,
        ),
    ):
        exists = (
            path.is_dir()
            if is_directory
            else path.is_file()
        )

        if not exists:
            raise FileNotFoundError(
                f"dGeDi {label}: {path}"
            )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    sys.path.insert(
        0,
        str(repository),
    )

    import open3d as o3d
    import torch

    from sklearn.neighbors import (
        NearestNeighbors,
    )

    from core.dgedi_distilled import (
        dgedi,
    )

    from utils import (
        extract_features,
        load_yaml_config,
    )

    mode_config = load_yaml_config(
        str(config_path)
    )[args.mode]

    model_config = dict(
        mode_config["model_config"]
    )

    weights_path = Path(
        mode_config["weights_path"]
    )

    if not weights_path.is_absolute():
        weights_path = (
            repository / weights_path
        )

    weights_path = weights_path.resolve()

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"dGeDi checkpoint: {weights_path}"
        )

    model_config["weights_path"] = str(
        weights_path
    )

    device = torch.device(
        args.device
    )

    model = dgedi(
        {
            "query": model_config,
            "target": model_config,
            "device": args.device,
        }
    )

    reference_self = _transform(
        np.load(
            reference_self_path,
            allow_pickle=False,
        ),
        "reference self pose",
    )

    query_self = _transform(
        np.load(
            query_self_path,
            allow_pickle=False,
        ),
        "query self pose",
    )

    reference_proxy = _mesh_points(
        reference_mesh,
        args.sample_count,
        o3d,
    )

    query_proxy = _mesh_points(
        query_mesh,
        args.sample_count,
        o3d,
    )

    # FoundationPose가 큰 camera motion을 담당한다.
    #
    # A = T_Cr_from_Pr
    # B = T_Cq_from_Pq
    # H0 = B @ inv(A)
    #
    # H0로 reference mesh를 query camera frame에
    # 먼저 배치하고, dGeDi는 residual만 추정한다.
    initial_relative = _transform(
        query_self
        @ np.linalg.inv(reference_self),
        (
            "FoundationPose initial "
            "relative pose"
        ),
    )

    # Reference:
    # Pr -> Cr -> Cq(initial)
    reference_in_query = _apply(
        _apply(
            reference_proxy,
            reference_self,
        ),
        initial_relative,
    )

    # Query:
    # Pq -> Cq
    query_in_query = _apply(
        query_proxy,
        query_self,
    )

    diameter_m = max(
        _diameter(reference_proxy),
        _diameter(query_proxy),
    )

    reference_features = extract_features(
        _feature_cloud(
            reference_in_query,
            diameter_m,
            o3d,
        ),
        model,
        device,
    )

    query_features = extract_features(
        _feature_cloud(
            query_in_query,
            diameter_m,
            o3d,
        ),
        model,
        device,
    )

    # 최종 accept/reject gate는 없다.
    #
    # FoundationPose prior를 실제로 사용하기 위해
    # 각 점의 공간적 KNN 내부에서만
    # dGeDi descriptor 대응을 찾는다.
    matches, similarities = (
        _local_mutual_matches(
            reference_in_query,
            query_in_query,
            reference_features,
            query_features,
            args.local_neighbors,
            NearestNeighbors,
        )
    )

    if len(matches) < 3:
        raise RuntimeError(
            "Too few local mutual "
            f"dGeDi matches: {len(matches)}"
        )

    source = _cloud(
        reference_in_query,
        o3d,
    )

    target = _cloud(
        query_in_query,
        o3d,
    )

    registration = (
        o3d.pipelines.registration
    )

    ransac_distance_m = (
        args.ransac_threshold
        * diameter_m
    )

    icp_distance_m = (
        args.icp_threshold
        * diameter_m
    )

    ransac = (
        registration
        .registration_ransac_based_on_correspondence(
            source,
            target,
            o3d.utility.Vector2iVector(
                matches
            ),
            ransac_distance_m,
            (
                registration
                .TransformationEstimationPointToPoint(
                    False
                )
            ),
            3,
            [
                (
                    registration
                    .CorrespondenceCheckerBasedOnEdgeLength(
                        0.9
                    )
                ),
                (
                    registration
                    .CorrespondenceCheckerBasedOnDistance(
                        ransac_distance_m
                    )
                ),
            ],
            (
                registration
                .RANSACConvergenceCriteria(
                    10000,
                    0.999,
                )
            ),
        )
    )

    if len(
        ransac.correspondence_set
    ) < 3:
        raise RuntimeError(
            "dGeDi residual RANSAC found "
            "fewer than 3 inliers."
        )

    ransac_residual = _transform(
        ransac.transformation,
        "dGeDi RANSAC residual",
    )

    icp = registration.registration_icp(
        source,
        target,
        icp_distance_m,
        ransac_residual,
        (
            registration
            .TransformationEstimationPointToPoint()
        ),
        (
            registration
            .ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=2000,
            )
        ),
    )

    if len(icp.correspondence_set) < 3:
        raise RuntimeError(
            "dGeDi residual ICP found "
            "fewer than 3 correspondences."
        )

    # 고정 실행:
    # FoundationPose initial
    # -> dGeDi RANSAC residual
    # -> ICP residual
    #
    # rotation/translation/fitness
    # accept-reject gate는 없다.
    final_residual = _transform(
        icp.transformation,
        "dGeDi ICP residual",
    )

    ransac_relative = _transform(
        ransac_residual
        @ initial_relative,
        "RANSAC relative pose",
    )

    final_relative = _transform(
        final_residual
        @ initial_relative,
        "final relative pose",
    )

    # 기존 main.py와 호환되는 등가 proxy transform.
    #
    # H = B @ T_proxy @ inv(A)
    # T_proxy = inv(B) @ H @ A
    ransac_proxy_pose = _transform(
        np.linalg.inv(query_self)
        @ ransac_relative
        @ reference_self,
        "equivalent RANSAC proxy pose",
    )

    proxy_pose = _transform(
        np.linalg.inv(query_self)
        @ final_relative
        @ reference_self,
        "equivalent proxy pose",
    )

    paths = {
        "initial_relative": (
            output
            / (
                "foundationpose_self_"
                "initial_relative_pose.npy"
            )
        ),
        "ransac_residual": (
            output
            / (
                "dgedi_ransac_residual_"
                "query_camera.npy"
            )
        ),
        "icp_residual": (
            output
            / (
                "dgedi_icp_residual_"
                "query_camera.npy"
            )
        ),
        "ransac_proxy": (
            output
            / (
                "dgedi_ransac_proxy_pose_"
                "query_from_reference.npy"
            )
        ),
        "proxy": (
            output
            / (
                "dgedi_proxy_pose_"
                "query_from_reference.npy"
            )
        ),
        "relative": (
            output
            / (
                "dgedi_relative_pose_"
                "query_from_reference.npy"
            )
        ),
        "metadata": (
            output
            / "dgedi_registration.json"
        ),
    }

    np.save(
        paths["initial_relative"],
        initial_relative,
        allow_pickle=False,
    )

    np.save(
        paths["ransac_residual"],
        ransac_residual,
        allow_pickle=False,
    )

    np.save(
        paths["icp_residual"],
        final_residual,
        allow_pickle=False,
    )

    np.save(
        paths["ransac_proxy"],
        ransac_proxy_pose,
        allow_pickle=False,
    )

    np.save(
        paths["proxy"],
        proxy_pose,
        allow_pickle=False,
    )

    np.save(
        paths["relative"],
        final_relative,
        allow_pickle=False,
    )

    metadata = {
        "status": "completed",
        "backend": "dgedi",
        "strategy": (
            "foundationpose_coarse_"
            "local_dgedi_residual"
        ),
        "selection_policy": (
            "fixed_ransac_then_icp; "
            "no final accept_reject gate"
        ),
        "pose_convention": (
            "T_query_camera_from_"
            "reference_camera"
        ),
        "translation_unit": "meter",
        "reference_mesh": str(
            reference_mesh
        ),
        "query_mesh": str(
            query_mesh
        ),
        "reference_self_pose": (
            reference_self.tolist()
        ),
        "query_self_pose": (
            query_self.tolist()
        ),
        "initial_relative_pose": (
            initial_relative.tolist()
        ),
        "diameter_m": diameter_m,
        "matching": {
            "method": (
                "local_spatial_knn_then_"
                "dgedi_cosine_mutual"
            ),
            "local_neighbors": (
                args.local_neighbors
            ),
            "correspondence_count": int(
                len(matches)
            ),
            "mean_cosine_similarity": (
                float(
                    similarities.mean()
                )
            ),
            "median_cosine_similarity": (
                float(
                    np.median(
                        similarities
                    )
                )
            ),
        },
        "thresholds": {
            "ransac_ratio": (
                args.ransac_threshold
            ),
            "ransac_distance_m": (
                ransac_distance_m
            ),
            "icp_ratio": (
                args.icp_threshold
            ),
            "icp_distance_m": (
                icp_distance_m
            ),
        },
        "ransac": _result_dict(
            ransac
        ),
        "icp": _result_dict(
            icp
        ),
        "ransac_relative_pose": (
            ransac_relative.tolist()
        ),
        "final_relative_pose": (
            final_relative.tolist()
        ),
        (
            "proxy_pose_"
            "query_from_reference"
        ): proxy_pose.tolist(),
        "paths": {
            key: str(value)
            for key, value in paths.items()
        },
    }

    with paths["metadata"].open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "[FoundationPose initial pose]"
    )
    print(initial_relative)

    print(
        "[dGeDi local correspondences] "
        f"{len(matches)}"
    )

    print(
        "[dGeDi RANSAC residual] "
        f"rotation_deg="
        f"{_rotation_deg(ransac_residual):.6f}, "
        f"translation_m="
        f"{_translation_m(ransac_residual):.6f}"
    )

    print(
        "[dGeDi ICP residual] "
        f"rotation_deg="
        f"{_rotation_deg(final_residual):.6f}, "
        f"translation_m="
        f"{_translation_m(final_residual):.6f}"
    )

    print(
        f"[dGeDi proxy pose] "
        f"{paths['proxy']}"
    )
    print(proxy_pose)

    print(
        f"[dGeDi relative pose] "
        f"{paths['relative']}"
    )
    print(final_relative)

    return 0


def run_dgedi_registration(
    *,
    repository_path: Path,
    python_executable: Path,
    config_path: Path,
    reference_self_alignment: Any,
    query_self_alignment: Any,
    output_directory: Path,
    mode: str = "multi_scale",
    device: str = "cuda",
    sample_count: int = 6000,
    ransac_threshold: float = 0.03,
    icp_threshold: float = 0.03,
    local_neighbors: int = 32,
) -> DGeDiRegistrationResult:
    repository = (
        Path(repository_path)
        .expanduser()
        .resolve()
    )

    python_path = (
        Path(python_executable)
        .expanduser()
        .resolve()
    )

    config_path = (
        Path(config_path)
        .expanduser()
        .resolve()
    )

    output = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )

    reference_mesh = Path(
        reference_self_alignment
        .scaled_mesh_path
    ).expanduser().resolve()

    query_mesh = Path(
        query_self_alignment
        .scaled_mesh_path
    ).expanduser().resolve()

    for label, path, is_directory in (
        (
            "repository",
            repository,
            True,
        ),
        (
            "Python",
            python_path,
            False,
        ),
        (
            "config",
            config_path,
            False,
        ),
        (
            "reference mesh",
            reference_mesh,
            False,
        ),
        (
            "query mesh",
            query_mesh,
            False,
        ),
    ):
        exists = (
            path.is_dir()
            if is_directory
            else path.is_file()
        )

        if not exists:
            raise FileNotFoundError(
                f"dGeDi {label}: {path}"
            )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_self_path = (
        output
        / "reference_self_pose.npy"
    )

    query_self_path = (
        output
        / "query_self_pose.npy"
    )

    np.save(
        reference_self_path,
        _transform(
            (
                reference_self_alignment
                .pose_camera_from_proxy
            ),
            "reference self pose",
        ),
        allow_pickle=False,
    )

    np.save(
        query_self_path,
        _transform(
            (
                query_self_alignment
                .pose_camera_from_proxy
            ),
            "query self pose",
        ),
        allow_pickle=False,
    )

    command = [
        str(python_path),
        str(Path(__file__).resolve()),
        "--worker",
        "--repository",
        str(repository),
        "--config",
        str(config_path),
        "--reference-mesh",
        str(reference_mesh),
        "--query-mesh",
        str(query_mesh),
        "--reference-self-pose",
        str(reference_self_path),
        "--query-self-pose",
        str(query_self_path),
        "--output-directory",
        str(output),
        "--mode",
        mode,
        "--device",
        device,
        "--sample-count",
        str(sample_count),
        "--ransac-threshold",
        str(ransac_threshold),
        "--icp-threshold",
        str(icp_threshold),
        "--local-neighbors",
        str(local_neighbors),
    ]

    environment = os.environ.copy()

    environment.pop(
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD",
        None,
    )

    environment[
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    ] = "1"

    completed = subprocess.run(
        command,
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "dGeDi execution failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    proxy_pose_path = (
        output
        / (
            "dgedi_proxy_pose_"
            "query_from_reference.npy"
        )
    )

    relative_pose_path = (
        output
        / (
            "dgedi_relative_pose_"
            "query_from_reference.npy"
        )
    )

    metadata_path = (
        output
        / "dgedi_registration.json"
    )

    for label, path in (
        (
            "proxy pose",
            proxy_pose_path,
        ),
        (
            "relative pose",
            relative_pose_path,
        ),
        (
            "metadata",
            metadata_path,
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"dGeDi {label}: {path}"
            )

    proxy_pose = _transform(
        np.load(
            proxy_pose_path,
            allow_pickle=False,
        ),
        "dGeDi proxy pose",
    )

    relative_pose = _transform(
        np.load(
            relative_pose_path,
            allow_pickle=False,
        ),
        "dGeDi relative pose",
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    metadata.update(
        {
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    if completed.stdout:
        print(
            completed.stdout,
            end="",
        )

    if completed.stderr:
        print(
            completed.stderr,
            file=sys.stderr,
            end="",
        )

    return DGeDiRegistrationResult(
        proxy_pose_query_from_reference=(
            proxy_pose
        ),
        relative_pose_query_from_reference=(
            relative_pose
        ),
        proxy_pose_path=(
            proxy_pose_path
        ),
        relative_pose_path=(
            relative_pose_path
        ),
        metadata_path=(
            metadata_path
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--worker",
        action="store_true",
    )

    parser.add_argument(
        "--repository",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--reference-mesh",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--query-mesh",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--reference-self-pose",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--query-self-pose",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--mode",
        choices=(
            "single_scale",
            "multi_scale",
        ),
        default="multi_scale",
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--sample-count",
        type=int,
        default=6000,
    )

    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--icp-threshold",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--local-neighbors",
        type=int,
        default=32,
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()

    if not arguments.worker:
        raise SystemExit(
            "Use --worker when executing "
            "this file directly."
        )

    raise SystemExit(
        _worker(arguments)
    )
