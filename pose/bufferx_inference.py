from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Register two scaled proxy meshes with BUFFER-X."
        )
    )
    parser.add_argument(
        "--repository",
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
        "--pose-estimator",
        choices=("ransac", "kiss_matcher"),
        default="ransac",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=30000,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--icp-refine",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _require_file(
    path: Path,
    *,
    name: str,
) -> Path:
    resolved = (
        Path(path)
        .expanduser()
        .resolve()
    )

    if not resolved.is_file():
        raise FileNotFoundError(
            f"{name} not found: {resolved}"
        )

    return resolved


def _load_surface_points(
    *,
    mesh_path: Path,
    sample_count: int,
    open3d_module: Any,
) -> Any:
    mesh = open3d_module.io.read_triangle_mesh(
        str(mesh_path)
    )

    if mesh.has_triangles():
        point_cloud = (
            mesh.sample_points_uniformly(
                number_of_points=sample_count,
            )
        )
    else:
        point_cloud = (
            open3d_module.io.read_point_cloud(
                str(mesh_path)
            )
        )

    if len(point_cloud.points) < 32:
        raise ValueError(
            "Mesh contains too few surface points: "
            f"{mesh_path} ({len(point_cloud.points)})"
        )

    return point_cloud


def _load_model(
    *,
    repository: Path,
    config: Any,
    device: Any,
    torch_module: Any,
) -> Any:
    from models.BUFFERX import BufferX

    model = BufferX(config)

    for stage in config.train.all_stage:
        checkpoint_path = (
            repository
            / "snapshot"
            / "threedmatch"
            / stage
            / "best.pth"
        )
        _require_file(
            checkpoint_path,
            name=f"BUFFER-X {stage} checkpoint",
        )
        state_dict = torch_module.load(
            checkpoint_path,
            map_location=device,
        )

        if (
            isinstance(state_dict, dict)
            and "state_dict" in state_dict
            and isinstance(
                state_dict["state_dict"],
                dict,
            )
        ):
            state_dict = state_dict["state_dict"]

        stage_values = {
            key: value
            for key, value in state_dict.items()
            if stage in key
        }
        model_values = model.state_dict()
        model_values.update(stage_values)
        model.load_state_dict(model_values)

    return model.to(device).eval()


def main() -> int:
    args = parse_args()

    if args.sample_count < 2000:
        raise ValueError(
            "sample_count must be at least 2000."
        )

    repository = (
        args.repository
        .expanduser()
        .resolve()
    )

    if not repository.is_dir():
        raise FileNotFoundError(
            f"BUFFER-X repository not found: {repository}"
        )

    if str(repository) in sys.path:
        sys.path.remove(str(repository))

    sys.path.insert(0, str(repository))

    reference_mesh_path = _require_file(
        args.reference_mesh,
        name="Reference scaled mesh",
    )
    query_mesh_path = _require_file(
        args.query_mesh,
        name="Query scaled mesh",
    )
    output_directory = (
        args.output_directory
        .expanduser()
        .resolve()
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    import numpy as np
    import open3d as o3d
    import torch

    from config import make_cfg
    from utils.tools import (
        sphericity_based_voxel_analysis,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "BUFFER-X inference requires CUDA."
        )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)

    started_at = time.perf_counter()
    reference_cloud = _load_surface_points(
        mesh_path=reference_mesh_path,
        sample_count=args.sample_count,
        open3d_module=o3d,
    )
    query_cloud = _load_surface_points(
        mesh_path=query_mesh_path,
        sample_count=args.sample_count,
        open3d_module=o3d,
    )

    (
        voxel_size,
        sphericity,
        is_aligned_to_global_z,
    ) = sphericity_based_voxel_analysis(
        reference_cloud,
        query_cloud,
    )
    reference_downsampled = (
        reference_cloud.voxel_down_sample(
            voxel_size=voxel_size
        )
    )
    query_downsampled = (
        query_cloud.voxel_down_sample(
            voxel_size=voxel_size
        )
    )
    reference_points = np.asarray(
        reference_downsampled.points,
        dtype=np.float32,
    )
    query_points = np.asarray(
        query_downsampled.points,
        dtype=np.float32,
    )
    common_point_count = min(
        reference_points.shape[0],
        query_points.shape[0],
    )

    if common_point_count < 32:
        raise ValueError(
            "Voxelized meshes contain too few points: "
            f"reference={reference_points.shape[0]}, "
            f"query={query_points.shape[0]}, "
            f"voxel_size={voxel_size}"
        )

    config = make_cfg(
        "3DMatch",
        repository.parent / "datasets",
    )
    config.stage = "test"
    config.match.pose_estimator = (
        args.pose_estimator
    )
    config.patch.num_points_radius_estimate = min(
        int(
            config
            .patch
            .num_points_radius_estimate
        ),
        common_point_count,
    )
    config.patch.num_fps = min(
        int(config.patch.num_fps),
        common_point_count,
    )
    config.match.dist_th = max(
        2.5 * float(voxel_size),
        0.002,
    )
    config.match.kiss_resolution = max(
        float(voxel_size),
        0.001,
    )

    model = _load_model(
        repository=repository,
        config=config,
        device=device,
        torch_module=torch,
    )
    data_source = {
        "src_fds_pcd": torch.as_tensor(
            reference_points,
            dtype=torch.float32,
            device=device,
        ),
        "tgt_fds_pcd": torch.as_tensor(
            query_points,
            dtype=torch.float32,
            device=device,
        ),
        "is_aligned_to_global_z": bool(
            is_aligned_to_global_z
        ),
    }

    inference_started_at = time.perf_counter()

    with torch.inference_mode():
        (
            raw_pose,
            model_times,
            num_inliers,
            num_mutual_inliers,
            num_consensus_inliers,
            scales_used,
        ) = model(data_source)

    torch.cuda.synchronize()
    inference_time_sec = (
        time.perf_counter()
        - inference_started_at
    )

    if raw_pose is None:
        raise RuntimeError(
            "BUFFER-X did not return a pose."
        )

    raw_pose = np.asarray(
        raw_pose,
        dtype=np.float64,
    )
    final_pose = raw_pose.copy()
    icp_fitness = None
    icp_inlier_rmse = None
    icp_threshold_m = None

    if args.icp_refine:
        icp_threshold_m = max(
            2.0 * float(voxel_size),
            0.001,
        )
        icp_result = (
            o3d.pipelines.registration
            .registration_icp(
                reference_downsampled,
                query_downsampled,
                icp_threshold_m,
                raw_pose,
                (
                    o3d.pipelines.registration
                    .TransformationEstimationPointToPoint()
                ),
                (
                    o3d.pipelines.registration
                    .ICPConvergenceCriteria(
                        max_iteration=50
                    )
                ),
            )
        )
        final_pose = np.asarray(
            icp_result.transformation,
            dtype=np.float64,
        )
        icp_fitness = float(icp_result.fitness)
        icp_inlier_rmse = float(
            icp_result.inlier_rmse
        )

    raw_pose_path = (
        output_directory
        / "bufferx_raw_proxy_pose_query_from_reference.npy"
    )
    final_pose_path = (
        output_directory
        / "bufferx_proxy_pose_query_from_reference.npy"
    )
    metadata_path = (
        output_directory
        / "bufferx_registration.json"
    )
    np.save(
        raw_pose_path,
        raw_pose,
        allow_pickle=False,
    )
    np.save(
        final_pose_path,
        final_pose,
        allow_pickle=False,
    )

    metadata = {
        "status": "inference_completed",
        "proxy_pose_convention": (
            "T_query_proxy_from_reference_proxy"
        ),
        "reference_mesh_path": str(
            reference_mesh_path
        ),
        "query_mesh_path": str(query_mesh_path),
        "pose_estimator": args.pose_estimator,
        "sample_count": args.sample_count,
        "reference_voxel_point_count": int(
            reference_points.shape[0]
        ),
        "query_voxel_point_count": int(
            query_points.shape[0]
        ),
        "voxel_size_m": float(voxel_size),
        "ransac_distance_threshold_m": float(
            config.match.dist_th
        ),
        "sphericity": float(sphericity),
        "is_aligned_to_global_z": bool(
            is_aligned_to_global_z
        ),
        "num_inliers": int(num_inliers),
        "num_mutual_inliers": int(
            num_mutual_inliers
        ),
        "num_consensus_inliers": int(
            num_consensus_inliers
        ),
        "scales_used": int(scales_used),
        "model_times_sec": [
            float(value)
            for value in model_times
        ],
        "inference_time_sec": (
            inference_time_sec
        ),
        "total_time_sec": (
            time.perf_counter() - started_at
        ),
        "icp_refine": bool(args.icp_refine),
        "icp_threshold_m": icp_threshold_m,
        "icp_fitness": icp_fitness,
        "icp_inlier_rmse_m": icp_inlier_rmse,
        "raw_proxy_pose_query_from_reference": (
            raw_pose.tolist()
        ),
        "proxy_pose_query_from_reference": (
            final_pose.tolist()
        ),
        "raw_proxy_pose_path": str(
            raw_pose_path
        ),
        "proxy_pose_path": str(
            final_pose_path
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

    print(
        "[BUFFER-X completed]\n"
        f"Pose: {final_pose_path}\n"
        f"Metadata: {metadata_path}"
    )
    print(final_pose)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
