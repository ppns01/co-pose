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


def _rigid(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)

    if (
        matrix.shape != (4, 4)
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError(
            f"Invalid {name}: shape={matrix.shape}"
        )

    if not np.allclose(
        matrix[3],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-5,
        rtol=0.0,
    ):
        raise ValueError(
            f"Invalid homogeneous row in {name}"
        )

    rotation = matrix[:3, :3]

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

    return matrix
def _save_self_aligned_mesh(
    *,
    source_mesh_path: Path,
    pose_camera_from_proxy: Any,
    output_mesh_path: Path,
) -> Path:
    """
    FoundationPose self pose를 mesh vertex에 직접 적용하고
    camera 좌표계의 새 mesh 파일로 저장한다.

    output vertex:
        p_camera = T_camera_from_proxy @ p_proxy
    """
    import open3d as o3d

    source_mesh_path = (
        Path(source_mesh_path)
        .expanduser()
        .resolve()
    )

    output_mesh_path = (
        Path(output_mesh_path)
        .expanduser()
        .resolve()
    )

    if not source_mesh_path.is_file():
        raise FileNotFoundError(
            f"Source mesh not found: "
            f"{source_mesh_path}"
        )

    pose = _rigid(
        pose_camera_from_proxy,
        "FoundationPose self pose",
    )

    mesh = o3d.io.read_triangle_mesh(
        str(source_mesh_path),
        enable_post_processing=True,
    )

    if len(mesh.vertices) == 0:
        raise ValueError(
            f"Mesh has no vertices: "
            f"{source_mesh_path}"
        )

    # FoundationPose self pose를 실제 mesh에 bake한다.
    mesh.transform(pose)

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    output_mesh_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = o3d.io.write_triangle_mesh(
        str(output_mesh_path),
        mesh,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    )

    if not saved:
        raise IOError(
            "Failed to save self-aligned mesh: "
            f"{output_mesh_path}"
        )

    if not output_mesh_path.is_file():
        raise FileNotFoundError(
            "Self-aligned mesh was not created: "
            f"{output_mesh_path}"
        )

    return output_mesh_path

def compose_dgedi_relative_pose(
    *,
    reference_pose_camera_from_proxy: Any,
    query_pose_camera_from_proxy: Any,
    proxy_pose_query_from_reference: Any,
) -> np.ndarray:
    reference_self = _rigid(
        reference_pose_camera_from_proxy,
        "reference self pose",
    )
    query_self = _rigid(
        query_pose_camera_from_proxy,
        "query self pose",
    )
    proxy_pose = _rigid(
        proxy_pose_query_from_reference,
        "dGeDi proxy pose",
    )

    # T_Cq_from_Cr =
    # T_Cq_from_Pq
    # @ T_Pq_from_Pr
    # @ inv(T_Cr_from_Pr)
    return _rigid(
        query_self
        @ proxy_pose
        @ np.linalg.inv(reference_self),
        "relative pose",
    )


def _mesh_to_cloud(
    path: Path,
    count: int,
    o3d: Any,
) -> Any:
    mesh = o3d.io.read_triangle_mesh(
        str(path),
        enable_post_processing=True,
    )

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64,
    )

    if len(vertices) == 0:
        raise ValueError(
            f"Mesh has no vertices: {path}"
        )

    cloud = o3d.geometry.PointCloud()
    cloud.points = (
        o3d.utility.Vector3dVector(vertices)
    )

    if len(cloud.points) > count:
        cloud = (
            cloud.farthest_point_down_sample(
                count
            )
        )

    elif (
        len(cloud.points) < count
        and len(mesh.triangles) > 0
    ):
        cloud = mesh.sample_points_uniformly(
            number_of_points=count
        )

    cloud = cloud.remove_non_finite_points()
    cloud = cloud.remove_duplicated_points()

    if len(cloud.points) < 256:
        raise ValueError(
            f"Too few points: {len(cloud.points)}"
        )

    return cloud


def _diameter(
    points: np.ndarray,
    block: int = 256,
) -> float:
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    maximum = 0.0

    for start in range(
        0,
        len(points),
        block,
    ):
        delta = (
            points[
                start : start + block,
                None,
                :,
            ]
            - points[
                None,
                :,
                :,
            ]
        )

        squared = np.einsum(
            "ijk,ijk->ij",
            delta,
            delta,
            optimize=True,
        )

        maximum = max(
            maximum,
            float(squared.max()),
        )

    return float(np.sqrt(maximum))


def _normalize(
    cloud: Any,
    diameter: float,
    o3d: Any,
) -> tuple[Any, np.ndarray]:
    points = np.asarray(
        cloud.points,
        dtype=np.float64,
    )

    center = points.mean(axis=0)

    normalized = (
        o3d.geometry.PointCloud()
    )

    normalized.points = (
        o3d.utility.Vector3dVector(
            (points - center) / diameter
        )
    )

    return normalized, center


def _restore_transform(
    transform: Any,
    source_center: np.ndarray,
    target_center: np.ndarray,
    diameter: float,
) -> np.ndarray:
    transform = np.asarray(
        transform,
        dtype=np.float64,
    )

    rotation = transform[:3, :3]

    restored = np.eye(
        4,
        dtype=np.float64,
    )

    restored[:3, :3] = rotation

    restored[:3, 3] = (
        target_center
        - rotation @ source_center
        + diameter * transform[:3, 3]
    )

    return _rigid(
        restored,
        "restored dGeDi pose",
    )


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

    output = (
        args.output_directory
        .expanduser()
        .resolve()
    )

    if not repository.is_dir():
        raise FileNotFoundError(
            f"dGeDi repository: {repository}"
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"dGeDi config: {config_path}"
        )

    if not reference_mesh.is_file():
        raise FileNotFoundError(
            f"Reference mesh: {reference_mesh}"
        )

    if not query_mesh.is_file():
        raise FileNotFoundError(
            f"Query mesh: {query_mesh}"
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

    from core.dgedi_distilled import (
        dgedi,
    )
    from utils import (
        extract_features,
        load_yaml_config,
        register_one,
    )

    config = load_yaml_config(
        str(config_path)
    )[args.mode]

    model_config = dict(
        config["model_config"]
    )

    weights = Path(
        config["weights_path"]
    )

    if not weights.is_absolute():
        weights = repository / weights

    weights = weights.resolve()

    if not weights.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights}"
        )

    model_config["weights_path"] = (
        str(weights)
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

    # source = reference proxy
    # target = query proxy
    reference_metric = _mesh_to_cloud(
        reference_mesh,
        args.sample_count,
        o3d,
    )

    query_metric = _mesh_to_cloud(
        query_mesh,
        args.sample_count,
        o3d,
    )

    diameter_m = _diameter(
        np.asarray(
            reference_metric.points
        )
    )

    if (
        not np.isfinite(diameter_m)
        or diameter_m <= 0.0
    ):
        raise ValueError(
            f"Invalid diameter: {diameter_m}"
        )

    # 두 proxy mesh 모두 meter 단위이므로
    # 공식 demo.py의 target * 1000은 적용하지 않는다.
    reference_norm, reference_center = (
        _normalize(
            reference_metric,
            diameter_m,
            o3d,
        )
    )

    query_norm, query_center = (
        _normalize(
            query_metric,
            diameter_m,
            o3d,
        )
    )

    reference_features = (
        extract_features(
            reference_norm,
            model,
            device,
        )
    )

    query_features = (
        extract_features(
            query_norm,
            model,
            device,
        )
    )

    ransac, icp = register_one(
        reference_norm,
        reference_features,
        query_norm,
        query_features,
        args.ransac_threshold,
        args.icp_threshold,
    )

    if (
        len(ransac.correspondence_set)
        < 3
    ):
        raise RuntimeError(
            "dGeDi RANSAC found fewer "
            "than 3 correspondences."
        )

    if (
        len(icp.correspondence_set)
        < 3
    ):
        raise RuntimeError(
            "dGeDi ICP found fewer "
            "than 3 correspondences."
        )

    ransac_pose = _restore_transform(
        ransac.transformation,
        reference_center,
        query_center,
        diameter_m,
    )

    final_pose = _restore_transform(
        icp.transformation,
        reference_center,
        query_center,
        diameter_m,
    )

    ransac_path = (
        output
        / (
            "dgedi_ransac_"
            "proxy_pose_query_from_reference.npy"
        )
    )

    final_path = (
        output
        / (
            "dgedi_proxy_pose_"
            "query_from_reference.npy"
        )
    )

    metadata_path = (
        output
        / "dgedi_registration.json"
    )

    np.save(
        ransac_path,
        ransac_pose,
        allow_pickle=False,
    )

    np.save(
        final_path,
        final_pose,
        allow_pickle=False,
    )

    metadata = {
        "status": "completed",
        "backend": "dgedi",
"pose_convention": (
    "T_query_camera_from_"
    "reference_camera"
),
"input_meshes": (
    "FoundationPose self-aligned "
    "meshes with pose baked into vertices"
),
        "translation_unit": "meter",
        "reference_mesh": (
            str(reference_mesh)
        ),
        "query_mesh": str(query_mesh),
        "normalization_diameter_m": (
            diameter_m
        ),
        "ransac": {
            "fitness": float(
                ransac.fitness
            ),
            "inlier_rmse_m": float(
                ransac.inlier_rmse
                * diameter_m
            ),
            "correspondence_count": (
                len(
                    ransac
                    .correspondence_set
                )
            ),
            "pose": ransac_pose.tolist(),
        },
        "icp": {
            "fitness": float(
                icp.fitness
            ),
            "inlier_rmse_m": float(
                icp.inlier_rmse
                * diameter_m
            ),
            "correspondence_count": (
                len(
                    icp
                    .correspondence_set
                )
            ),
            "pose": final_pose.tolist(),
        },
    }

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

    print(
        f"[dGeDi proxy pose] "
        f"{final_path}"
    )

    print(final_pose)

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

    raw_reference_mesh = Path(
        reference_self_alignment
        .scaled_mesh_path
    ).expanduser().resolve()

    raw_query_mesh = Path(
        query_self_alignment
        .scaled_mesh_path
    ).expanduser().resolve()

    reference_self = _rigid(
        reference_self_alignment
        .pose_camera_from_proxy,
        "reference FoundationPose self pose",
    )

    query_self = _rigid(
        query_self_alignment
        .pose_camera_from_proxy,
        "query FoundationPose self pose",
    )

    if not repository.is_dir():
        raise FileNotFoundError(
            f"dGeDi repository: {repository}"
        )

    if not python_path.is_file():
        raise FileNotFoundError(
            f"dGeDi Python: {python_path}"
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"dGeDi config: {config_path}"
        )

    if not raw_reference_mesh.is_file():
        raise FileNotFoundError(
            "Reference generated mesh: "
            f"{raw_reference_mesh}"
        )

    if not raw_query_mesh.is_file():
        raise FileNotFoundError(
            "Query generated mesh: "
            f"{raw_query_mesh}"
        )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    self_aligned_mesh_root = (
        output
        / "self_aligned_meshes"
    )

    # Reference mesh:
    # P_r -> C_r
    reference_mesh = (
        _save_self_aligned_mesh(
            source_mesh_path=(
                raw_reference_mesh
            ),
            pose_camera_from_proxy=(
                reference_self
            ),
            output_mesh_path=(
                self_aligned_mesh_root
                / (
                    "reference_self_aligned_"
                    "in_reference_camera.obj"
                )
            ),
        )
    )

    # Query mesh:
    # P_q -> C_q
    query_mesh = (
        _save_self_aligned_mesh(
            source_mesh_path=(
                raw_query_mesh
            ),
            pose_camera_from_proxy=(
                query_self
            ),
            output_mesh_path=(
                self_aligned_mesh_root
                / (
                    "query_self_aligned_"
                    "in_query_camera.obj"
                )
            ),
        )
    )

    print(
        "[Reference self-aligned mesh] "
        f"{reference_mesh}"
    )

    print(
        "[Query self-aligned mesh] "
        f"{query_mesh}"
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
    ]

    # dGeDi 공식 checkpoint에는 argparse.Namespace 등
    # 일반 pickle 객체가 포함되어 있다.
    # 이 설정은 dGeDi 전용 subprocess에만 적용한다.
    worker_environment = os.environ.copy()
    worker_environment.pop(
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD",
        None,
    )
    worker_environment[
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    ] = "1"

    completed = subprocess.run(
        command,
        cwd=repository,
        env=worker_environment,
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

    metadata_path = (
        output
        / "dgedi_registration.json"
    )

    if not proxy_pose_path.is_file():
        raise FileNotFoundError(
            f"dGeDi pose: {proxy_pose_path}"
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"dGeDi metadata: {metadata_path}"
        )

    # dGeDi 입력 mesh가 이미 각각 C_r, C_q 좌표계에
    # 저장되어 있으므로 dGeDi source->target transform은
    # 곧바로 T_Cq_from_Cr이다.
    direct_relative_pose = _rigid(
        np.load(
            proxy_pose_path,
            allow_pickle=False,
        ),
        (
            "dGeDi direct relative "
            "camera pose"
        ),
    )

    relative_pose = (
        direct_relative_pose.copy()
    )

    relative_pose_path = (
        output
        / (
            "dgedi_relative_pose_"
            "query_from_reference.npy"
        )
    )

    np.save(
        relative_pose_path,
        relative_pose,
        allow_pickle=False,
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    metadata.update(
        {
            "strategy": (
                "foundationpose_self_pose_"
                "baked_into_mesh_then_dgedi"
            ),
            "input_mesh_coordinate_frames": {
                "reference": (
                    "reference_camera"
                ),
                "query": (
                    "query_camera"
                ),
            },
            "raw_reference_mesh": str(
                raw_reference_mesh
            ),
            "raw_query_mesh": str(
                raw_query_mesh
            ),
            "reference_self_aligned_mesh": (
                str(reference_mesh)
            ),
            "query_self_aligned_mesh": (
                str(query_mesh)
            ),
            "reference_pose_camera_from_proxy": (
                reference_self.tolist()
            ),
            "query_pose_camera_from_proxy": (
                query_self.tolist()
            ),
            "relative_pose_convention": (
                "T_query_camera_from_"
                "reference_camera"
            ),
            "relative_pose_query_from_reference": (
                relative_pose.tolist()
            ),
            "relative_pose_path": (
                str(relative_pose_path)
            ),
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

    return DGeDiRegistrationResult(
        # 입력 mesh에 FoundationPose self pose가 이미
        # bake되어 있으므로 dGeDi 출력은 직접
        # T_query_camera_from_reference_camera이다.
        # 필드명은 기존 main.py 호환을 위해 유지한다.
        proxy_pose_query_from_reference=(
            relative_pose
        ),
        relative_pose_query_from_reference=(
            relative_pose
        ),
        proxy_pose_path=proxy_pose_path,
        relative_pose_path=(
            relative_pose_path
        ),
        metadata_path=metadata_path,
    )


def _parse_worker_args() -> (
    argparse.Namespace
):
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

    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_worker_args()

    if not arguments.worker:
        raise SystemExit(
            "Use --worker when executing "
            "this file directly."
        )

    raise SystemExit(
        _worker(arguments)
    )
