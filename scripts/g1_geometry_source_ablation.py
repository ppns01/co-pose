"""Compare controlled 3D inputs for final G1 registration.

This is a diagnostic replay of a completed production pair. It does not run
the pipeline and does not change its selected pose. The completed run's exact
S*/D*, final A1/B1, masks, depths, meshes, dGeDi configuration, and RANSAC
seed bank are reused. The experimental variable is the Reference/Query G1
cloud construction:

  proxy_hit:
    Reference/Query = depth-consistent self-rendered proxy hits
  full_proxy:
    Reference/Query = uniformly sampled complete proxy surfaces
  ref_observed_query_proxy:
    Reference = observed depth, Query = depth-consistent self-rendered proxy
  observed_depth:
    Reference/Query = observed depth

GT is never passed to dGeDi. It is loaded only after both workers finish to
measure actual observed-cloud overlap, score a virtual G_GT candidate, and
report RANSAC/ICP/H0 error.

Example:
  python scripts/g1_geometry_source_ablation.py \
    --source-output-root output_smoke_test_obj9_ref0_q3 \
    --output-dir output_smoke_test_obj9_ref0_q3/diagnostics/g1_geometry_source_ablation_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_GEOMETRIES = {
    "proxy_hit": {
        "reference": "proxy_hit",
        "query": "proxy_hit",
    },
    "full_proxy": {
        "reference": "full_proxy",
        "query": "full_proxy",
    },
    "ref_observed_query_proxy": {
        "reference": "observed_depth",
        "query": "proxy_hit",
    },
    "observed_depth": {
        "reference": "observed_depth",
        "query": "observed_depth",
    },
}
SOURCES = tuple(SOURCE_GEOMETRIES)
OVERLAP_THRESHOLDS_M = (0.002, 0.005, 0.010)


def _project_to_so3(rotation: np.ndarray) -> np.ndarray:
    u, _singular, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def _rotation_error_deg(estimate: np.ndarray, target: np.ndarray) -> float:
    relative = estimate[:3, :3].T @ target[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _load_camera_matrix(root: Path, view: str) -> tuple[np.ndarray, Path]:
    search_roots = (
        root / "self_evaluation_shared_axis_scale" / view,
        root / "self_evaluation" / view,
    )
    metadata_paths: list[Path] = []
    for search_root in search_roots:
        if search_root.is_dir():
            metadata_paths.extend(sorted(search_root.glob("**/render_metadata.json")))
        if metadata_paths:
            break
    if not metadata_paths:
        raise FileNotFoundError(f"No render_metadata.json found for {view}")

    matrices: list[np.ndarray] = []
    for path in metadata_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "camera_matrix" in payload:
            matrices.append(np.asarray(payload["camera_matrix"], dtype=np.float64))
    if not matrices:
        raise ValueError(f"No camera_matrix found for {view}")
    if any(matrix.shape != (3, 3) for matrix in matrices):
        raise ValueError(f"Invalid camera matrix shape for {view}")
    if any(not np.allclose(matrices[0], value) for value in matrices[1:]):
        raise ValueError(f"Conflicting camera matrices found for {view}")
    return matrices[0], metadata_paths[0]


def _build_paired_clouds(
    *,
    local_mesh_path: Path,
    pose_camera_from_proxy: np.ndarray,
    camera_matrix: np.ndarray,
    mask_bool: np.ndarray,
    depth_m: np.ndarray,
    maximum_depth_residual_m: float,
    sample_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build both clouds from exactly the same depth-consistent pixels."""
    mesh = o3d.io.read_triangle_mesh(
        str(local_mesh_path), enable_post_processing=True
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError(f"Empty proxy mesh: {local_mesh_path}")

    pose = np.asarray(pose_camera_from_proxy, dtype=np.float64)
    intrinsic = np.asarray(camera_matrix, dtype=np.float64)
    mask = np.asarray(mask_bool, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("pose_camera_from_proxy must be (4,4)")
    if intrinsic.shape != (3, 3):
        raise ValueError("camera_matrix must be (3,3)")
    if mask.shape != depth.shape or depth.ndim != 2:
        raise ValueError("mask and depth must share shape (H,W)")

    camera_mesh = o3d.geometry.TriangleMesh(mesh)
    camera_mesh.vertices = o3d.utility.Vector3dVector(
        vertices @ pose[:3, :3].T + pose[:3, 3][None, :]
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(camera_mesh))

    valid_observation = mask & np.isfinite(depth) & (depth > 0.0)
    pixel_v, pixel_u = np.nonzero(valid_observation)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    ray_direction = np.stack(
        (
            (pixel_u.astype(np.float64) - cx) / fx,
            (pixel_v.astype(np.float64) - cy) / fy,
            np.ones_like(pixel_u, dtype=np.float64),
        ),
        axis=-1,
    )
    rays = np.concatenate(
        (np.zeros_like(ray_direction), ray_direction), axis=-1
    ).astype(np.float32)
    raycast = scene.cast_rays(
        o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    )
    rendered_depth = raycast["t_hit"].numpy().astype(np.float64)
    primitive_ids = raycast["primitive_ids"].numpy().astype(np.int64)
    valid_hit = (
        np.isfinite(rendered_depth)
        & (rendered_depth > 0.0)
        & (primitive_ids >= 0)
        & (primitive_ids < len(triangles))
    )
    observed_depth = depth[pixel_v, pixel_u]
    residual = np.full(rendered_depth.shape, np.inf, dtype=np.float64)
    residual[valid_hit] = np.abs(
        rendered_depth[valid_hit] - observed_depth[valid_hit]
    )
    consistent = valid_hit & (residual <= maximum_depth_residual_m)
    consistent_count = int(np.count_nonzero(consistent))
    if consistent_count < 256:
        raise ValueError(f"Too few consistent pixels: {consistent_count}")
    if consistent_count > sample_count:
        raise ValueError(
            "This diagnostic requires all selected pixels to remain identical, "
            f"but {consistent_count} exceeds sample_count={sample_count}."
        )

    selected_rays = ray_direction[consistent]
    camera_proxy = selected_rays * rendered_depth[consistent, None]
    camera_observed = selected_rays * observed_depth[consistent, None]
    proxy_local = (camera_proxy - pose[:3, 3][None, :]) @ pose[:3, :3]
    observed_local = (camera_observed - pose[:3, 3][None, :]) @ pose[:3, :3]

    finite = np.all(np.isfinite(proxy_local), axis=1) & np.all(
        np.isfinite(observed_local), axis=1
    )
    proxy_local = proxy_local[finite]
    observed_local = observed_local[finite]
    selected_u = pixel_u[consistent][finite]
    selected_v = pixel_v[consistent][finite]

    # Apply one common pixel-level deduplication to both geometry sources.
    pixel_keys = np.stack((selected_v, selected_u), axis=1)
    _unique, first = np.unique(pixel_keys, axis=0, return_index=True)
    keep = np.sort(first)
    proxy_local = proxy_local[keep]
    observed_local = observed_local[keep]

    finite_residual = residual[consistent]
    diagnostics = {
        "valid_observed_depth_pixel_count": int(len(pixel_u)),
        "rendered_first_hit_pixel_count": int(np.count_nonzero(valid_hit)),
        "consistent_pixel_count_before_common_dedup": consistent_count,
        "common_pixel_count_saved": int(len(proxy_local)),
        "maximum_depth_residual_m": float(maximum_depth_residual_m),
        "consistent_depth_residual_mean_m": float(np.mean(finite_residual)),
        "consistent_depth_residual_median_m": float(np.median(finite_residual)),
        "maximum_coordinate_shift_render_to_observed_m": float(
            np.linalg.norm(camera_proxy - camera_observed, axis=1).max()
        ),
    }
    return {
        "proxy_hit": proxy_local,
        "observed_depth": observed_local,
    }, diagnostics


def _write_cloud(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64)
    )
    if not o3d.io.write_point_cloud(
        str(path),
        cloud,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    ):
        raise IOError(f"Failed to write point cloud: {path}")


def _sample_full_proxy(local_mesh_path: Path, sample_count: int) -> np.ndarray:
    """Uniformly sample the complete proxy surface in proxy-local metres."""
    mesh = o3d.io.read_triangle_mesh(
        str(local_mesh_path), enable_post_processing=True
    )
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"Empty proxy mesh: {local_mesh_path}")
    o3d.utility.random.seed(0)
    cloud = mesh.sample_points_uniformly(number_of_points=sample_count)
    points = np.asarray(cloud.points, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 256:
        raise ValueError(
            f"Too few full-proxy surface points: {len(points)}"
        )
    return points


def _backproject_observed(
    depth_m: np.ndarray, mask_bool: np.ndarray, camera_matrix: np.ndarray
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float64)
    mask = np.asarray(mask_bool, dtype=bool)
    intrinsic = np.asarray(camera_matrix, dtype=np.float64)
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    pixel_v, pixel_u = np.nonzero(valid)
    z = depth[pixel_v, pixel_u]
    x = (pixel_u.astype(np.float64) - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (pixel_v.astype(np.float64) - intrinsic[1, 2]) * z / intrinsic[1, 1]
    return np.stack((x, y, z), axis=1)


def _gt_overlap(
    reference_camera_points: np.ndarray,
    query_camera_points: np.ndarray,
    h_gt_query_camera_from_reference_camera: np.ndarray,
) -> dict[str, Any]:
    transform = np.asarray(
        h_gt_query_camera_from_reference_camera, dtype=np.float64
    )
    moved_reference = (
        reference_camera_points @ transform[:3, :3].T
        + transform[:3, 3][None, :]
    )
    forward = cKDTree(query_camera_points).query(moved_reference, k=1)[0]
    backward = cKDTree(moved_reference).query(query_camera_points, k=1)[0]
    thresholds: dict[str, Any] = {}
    for threshold in OVERLAP_THRESHOLDS_M:
        reference_to_query = float(np.mean(forward < threshold))
        query_to_reference = float(np.mean(backward < threshold))
        thresholds[f"{int(threshold * 1000)}mm"] = {
            "reference_to_query": reference_to_query,
            "query_to_reference": query_to_reference,
            "symmetric_mean": 0.5 * (
                reference_to_query + query_to_reference
            ),
        }
    return {
        "definition": (
            "Nearest-neighbour fractions after mapping the raw masked "
            "reference observed-depth cloud into the query camera with H_GT"
        ),
        "reference_point_count": int(len(reference_camera_points)),
        "query_point_count": int(len(query_camera_points)),
        "thresholds": thresholds,
    }


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    project_root = str(PROJECT_ROOT)
    entries = [
        entry
        for entry in environment.get("PYTHONPATH", "").split(os.pathsep)
        if entry and entry != project_root
    ]
    if entries:
        environment["PYTHONPATH"] = os.pathsep.join(entries)
    else:
        environment.pop("PYTHONPATH", None)
    return environment


def _replace_argument(command: list[str], name: str, value: Path) -> None:
    try:
        index = command.index(name)
    except ValueError as error:
        raise ValueError(f"Stored dGeDi command has no {name}") from error
    command[index + 1] = str(value)


def _run_worker(
    *,
    stored_command: list[str],
    reference_cloud_path: Path,
    query_cloud_path: Path,
    output_directory: Path,
) -> Path:
    command = list(stored_command)
    if "--candidate-diagnostics" not in command:
        command.append("--candidate-diagnostics")
    _replace_argument(command, "--reference-mesh", reference_cloud_path)
    _replace_argument(command, "--query-mesh", query_cloud_path)
    _replace_argument(command, "--output-directory", output_directory)
    repository = Path(command[command.index("--repository") + 1]).resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    completed = subprocess.run(
        command,
        cwd=repository,
        env=_worker_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    (output_directory / "worker_stdout.txt").write_text(
        completed.stdout, encoding="utf-8"
    )
    (output_directory / "worker_stderr.txt").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"dGeDi worker failed with return code {completed.returncode}:\n"
            f"{completed.stderr[-3000:]}"
        )
    metadata_path = output_directory / "dgedi_registration.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    return metadata_path


def _read_cloud(path: Path) -> np.ndarray:
    points = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return points


def _symmetric_chamfer_m(
    reference: np.ndarray, query: np.ndarray, transform: np.ndarray
) -> float:
    moved = reference @ transform[:3, :3].T + transform[:3, 3][None, :]
    forward = cKDTree(query).query(moved, k=1)[0]
    backward = cKDTree(moved).query(query, k=1)[0]
    return float(forward.mean() + backward.mean())


def _virtual_rank(score: float, candidate_scores: list[float]) -> int:
    return 1 + sum(value < score - 1e-15 for value in candidate_scores)


def _refine_metric_pose_with_same_icp(
    *,
    reference: np.ndarray,
    query: np.ndarray,
    initial_metric_pose: np.ndarray,
    normalization_diameter_m: float,
    icp_threshold_normalized: float,
) -> tuple[np.ndarray, Any]:
    """Apply dGeDi's exact point-to-point ICP to a metric initial pose."""
    reference_center = reference.mean(axis=0)
    query_center = query.mean(axis=0)
    reference_normalized = o3d.geometry.PointCloud()
    reference_normalized.points = o3d.utility.Vector3dVector(
        (reference - reference_center) / normalization_diameter_m
    )
    query_normalized = o3d.geometry.PointCloud()
    query_normalized.points = o3d.utility.Vector3dVector(
        (query - query_center) / normalization_diameter_m
    )

    initial_normalized = np.eye(4, dtype=np.float64)
    initial_normalized[:3, :3] = initial_metric_pose[:3, :3]
    initial_normalized[:3, 3] = (
        initial_metric_pose[:3, 3]
        - query_center
        + initial_metric_pose[:3, :3] @ reference_center
    ) / normalization_diameter_m
    result = o3d.pipelines.registration.registration_icp(
        reference_normalized,
        query_normalized,
        icp_threshold_normalized,
        initial_normalized,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=2000,
        ),
    )
    normalized_pose = np.asarray(result.transformation, dtype=np.float64)
    metric_pose = np.eye(4, dtype=np.float64)
    metric_pose[:3, :3] = normalized_pose[:3, :3]
    metric_pose[:3, 3] = (
        query_center
        - normalized_pose[:3, :3] @ reference_center
        + normalization_diameter_m * normalized_pose[:3, 3]
    )
    return metric_pose, result


def _load_object_centres(root: Path) -> tuple[np.ndarray, np.ndarray]:
    config = json.loads((root / "pipeline_config.json").read_text(encoding="utf-8"))
    dataset_root = Path(config["dataset_root"])
    split = str(config["split"])
    reference = config["reference"]
    query = config["query"]
    scene_id = int(reference["scene_id"])
    if scene_id != int(query["scene_id"]):
        raise ValueError("Reference/query scenes differ")
    scene_gt_path = dataset_root / split / f"{scene_id:06d}" / "scene_gt.json"
    scene_gt = json.loads(scene_gt_path.read_text(encoding="utf-8"))

    def centre(frame: dict[str, Any]) -> np.ndarray:
        records = scene_gt[str(int(frame["image_id"]))]
        record = records[int(frame["instance_index"])]
        return np.asarray(record["cam_t_m2c"], dtype=np.float64) / 1000.0

    return centre(reference), centre(query)


def _h_errors(
    h_estimate: np.ndarray,
    h_gt: np.ndarray,
    reference_object_centre: np.ndarray,
    query_object_centre: np.ndarray,
) -> dict[str, float]:
    predicted_query_centre = (
        h_estimate[:3, :3] @ reference_object_centre
        + h_estimate[:3, 3]
    )
    return {
        "rotation_error_deg": _rotation_error_deg(h_estimate, h_gt),
        "object_center_translation_error_m": float(
            np.linalg.norm(predicted_query_centre - query_object_centre)
        ),
        "object_center_translation_error_cm": float(
            100.0 * np.linalg.norm(predicted_query_centre - query_object_centre)
        ),
    }


def _analyse_source(
    *,
    source: str,
    metadata_path: Path,
    reference_cloud_path: Path,
    query_cloud_path: Path,
    a1: np.ndarray,
    b1: np.ndarray,
    g_gt: np.ndarray,
    h_gt: np.ndarray,
    reference_object_centre: np.ndarray,
    query_object_centre: np.ndarray,
    icp_threshold_normalized: float,
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    reference = _read_cloud(reference_cloud_path)
    query = _read_cloud(query_cloud_path)
    gt_chamfer = _symmetric_chamfer_m(reference, query, g_gt)

    candidate_rows: list[dict[str, Any]] = []
    for candidate in metadata["candidate_selection"]["candidates"]:
        ransac_pose = np.asarray(candidate["ransac_pose"], dtype=np.float64)
        icp_pose = np.asarray(candidate["icp_pose"], dtype=np.float64)
        candidate_rows.append(
            {
                "seed": int(candidate["seed"]),
                "ransac_chamfer_m": float(candidate["ransac_chamfer_m"]),
                "icp_chamfer_m": float(candidate["icp_chamfer_m"]),
                "ransac_g_rotation_error_deg": _rotation_error_deg(
                    ransac_pose, g_gt
                ),
                "icp_g_rotation_error_deg": _rotation_error_deg(icp_pose, g_gt),
                "ransac_pose": ransac_pose.tolist(),
                "icp_pose": icp_pose.tolist(),
            }
        )

    ransac_scores = [row["ransac_chamfer_m"] for row in candidate_rows]
    icp_scores = [row["icp_chamfer_m"] for row in candidate_rows]
    g_gt_after_icp, g_gt_icp_result = _refine_metric_pose_with_same_icp(
        reference=reference,
        query=query,
        initial_metric_pose=g_gt,
        normalization_diameter_m=float(metadata["normalization_diameter_m"]),
        icp_threshold_normalized=icp_threshold_normalized,
    )
    g_gt_after_icp_chamfer = _symmetric_chamfer_m(
        reference, query, g_gt_after_icp
    )
    selected_ransac = np.asarray(metadata["ransac"]["pose"], dtype=np.float64)
    selected_icp = np.asarray(metadata["icp"]["pose"], dtype=np.float64)
    h0_ransac = b1 @ selected_ransac @ np.linalg.inv(a1)
    h0_icp = b1 @ selected_icp @ np.linalg.inv(a1)

    return {
        "geometry_source": source,
        "reference_point_count": int(len(reference)),
        "query_point_count": int(len(query)),
        "chamfer_g_gt_m": gt_chamfer,
        "virtual_rank_g_gt_among_ransac": _virtual_rank(
            gt_chamfer, ransac_scores
        ),
        "virtual_rank_g_gt_among_icp": _virtual_rank(gt_chamfer, icp_scores),
        "g_gt_after_same_icp": {
            "chamfer_m": g_gt_after_icp_chamfer,
            "virtual_rank_among_icp": _virtual_rank(
                g_gt_after_icp_chamfer, icp_scores
            ),
            "rotation_drift_from_g_gt_deg": _rotation_error_deg(
                g_gt_after_icp, g_gt
            ),
            "fitness": float(g_gt_icp_result.fitness),
            "inlier_rmse_m": float(
                g_gt_icp_result.inlier_rmse
                * float(metadata["normalization_diameter_m"])
            ),
            "pose": g_gt_after_icp.tolist(),
            "h0_error": _h_errors(
                b1 @ g_gt_after_icp @ np.linalg.inv(a1),
                h_gt,
                reference_object_centre,
                query_object_centre,
            ),
        },
        "candidate_count": len(candidate_rows),
        "selected_seed": int(metadata["candidate_selection"]["selected_seed"]),
        "selected_icp_chamfer_m": float(
            metadata["candidate_selection"]["selected_chamfer_m"]
        ),
        "selected_ransac_g_rotation_error_deg": _rotation_error_deg(
            selected_ransac, g_gt
        ),
        "selected_icp_g_rotation_error_deg": _rotation_error_deg(
            selected_icp, g_gt
        ),
        "h0_from_selected_ransac": _h_errors(
            h0_ransac,
            h_gt,
            reference_object_centre,
            query_object_centre,
        ),
        "h0_from_selected_icp": _h_errors(
            h0_icp,
            h_gt,
            reference_object_centre,
            query_object_centre,
        ),
        "candidates": candidate_rows,
        "worker_metadata_path": str(metadata_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--full-proxy-sample-count",
        type=int,
        default=6000,
        help="Uniform samples per complete proxy surface (default: 6000).",
    )
    args = parser.parse_args(argv)

    root = Path(args.source_output_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    registration_path = (
        root / "mesh_registration/dgedi/final/dgedi_registration.json"
    )
    source_metadata = json.loads(registration_path.read_text(encoding="utf-8"))
    stored_command = list(source_metadata["command"])
    sample_count = int(
        stored_command[stored_command.index("--sample-count") + 1]
    )
    full_proxy_sample_count = min(args.full_proxy_sample_count, sample_count)
    if full_proxy_sample_count < 256:
        raise ValueError("--full-proxy-sample-count must be at least 256")
    icp_threshold_normalized = float(
        stored_command[stored_command.index("--icp-threshold") + 1]
    )
    maximum_residual = float(
        source_metadata["maximum_surface_depth_residual_m"]
    )
    a1 = np.asarray(
        source_metadata["reference_pose_camera_from_proxy"], dtype=np.float64
    )
    b1 = np.asarray(
        source_metadata["query_pose_camera_from_proxy"], dtype=np.float64
    )
    h_gt = np.asarray(
        np.load(root / "evaluation/ground_truth_relative_pose.npy"),
        dtype=np.float64,
    )
    h_gt[:3, :3] = _project_to_so3(h_gt[:3, :3])
    g_gt = np.linalg.inv(b1) @ h_gt @ a1
    g_gt[:3, :3] = _project_to_so3(g_gt[:3, :3])

    camera_matrices: dict[str, np.ndarray] = {}
    camera_metadata_paths: dict[str, str] = {}
    masks: dict[str, np.ndarray] = {}
    depths: dict[str, np.ndarray] = {}
    clouds: dict[str, dict[str, np.ndarray]] = {}
    cloud_diagnostics: dict[str, Any] = {}
    for view, pose, mesh_key in (
        ("reference", a1, "raw_reference_mesh"),
        ("query", b1, "raw_query_mesh"),
    ):
        camera_matrix, camera_metadata_path = _load_camera_matrix(root, view)
        camera_matrices[view] = camera_matrix
        camera_metadata_paths[view] = str(camera_metadata_path)
        masks[view] = np.load(root / f"views/{view}/segmentation/mask_bool.npy")
        depths[view] = np.load(root / f"views/{view}/prepared/masked_depth_m.npy")
        local_mesh_path = Path(source_metadata[mesh_key])
        clouds[view], cloud_diagnostics[view] = _build_paired_clouds(
            local_mesh_path=local_mesh_path,
            pose_camera_from_proxy=pose,
            camera_matrix=camera_matrix,
            mask_bool=masks[view],
            depth_m=depths[view],
            maximum_depth_residual_m=maximum_residual,
            sample_count=sample_count,
        )
        clouds[view]["full_proxy"] = _sample_full_proxy(
            local_mesh_path, full_proxy_sample_count
        )
        cloud_diagnostics[view]["full_proxy_surface_point_count"] = int(
            len(clouds[view]["full_proxy"])
        )

    cloud_paths: dict[str, dict[str, Path]] = {}
    worker_metadata_paths: dict[str, Path] = {}
    for source in SOURCES:
        source_root = output / source
        geometry = SOURCE_GEOMETRIES[source]
        cloud_paths[source] = {
            "reference": source_root / "reference_proxy_local.ply",
            "query": source_root / "query_proxy_local.ply",
        }
        _write_cloud(
            cloud_paths[source]["reference"],
            clouds["reference"][geometry["reference"]],
        )
        _write_cloud(
            cloud_paths[source]["query"],
            clouds["query"][geometry["query"]],
        )
        worker_metadata_paths[source] = _run_worker(
            stored_command=stored_command,
            reference_cloud_path=cloud_paths[source]["reference"],
            query_cloud_path=cloud_paths[source]["query"],
            output_directory=source_root / "dgedi",
        )

    observed_reference_camera = _backproject_observed(
        depths["reference"], masks["reference"], camera_matrices["reference"]
    )
    observed_query_camera = _backproject_observed(
        depths["query"], masks["query"], camera_matrices["query"]
    )
    overlap = _gt_overlap(observed_reference_camera, observed_query_camera, h_gt)
    reference_object_centre, query_object_centre = _load_object_centres(root)

    results = {
        source: _analyse_source(
            source=source,
            metadata_path=worker_metadata_paths[source],
            reference_cloud_path=cloud_paths[source]["reference"],
            query_cloud_path=cloud_paths[source]["query"],
            a1=a1,
            b1=b1,
            g_gt=g_gt,
            h_gt=h_gt,
            reference_object_centre=reference_object_centre,
            query_object_centre=query_object_centre,
            icp_threshold_normalized=icp_threshold_normalized,
        )
        for source in SOURCES
    }

    summary = {
        "experiment": "g1_geometry_source_ablation",
        "production_behavior_changed": False,
        "source_output_root": str(root),
        "source_registration_json": str(registration_path),
        "controlled_variables": [
            "S*/D*",
            "A1/B1",
            "final Reference/Query proxy meshes",
            "dGeDi model/config",
            "RANSAC seed bank",
            "ICP settings",
        ],
        "independent_variable": (
            "Reference/Query G1 cloud construction; paired visible variants "
            "reuse the same mask, pixels, render hits, and depth threshold"
        ),
        "source_definitions": SOURCE_GEOMETRIES,
        "gt_usage": (
            "Offline diagnostics only; never passed to dGeDi or candidate selection"
        ),
        "maximum_depth_residual_m": maximum_residual,
        "sample_count": sample_count,
        "full_proxy_sample_count": full_proxy_sample_count,
        "icp_threshold_normalized": icp_threshold_normalized,
        "camera_metadata_paths": camera_metadata_paths,
        "cloud_diagnostics": cloud_diagnostics,
        "gt_observed_cloud_overlap": overlap,
        "g_gt_query_proxy_from_reference_proxy": g_gt.tolist(),
        "results": results,
    }
    summary_path = output / "comparison.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    csv_path = output / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "geometry_source",
            "reference_point_count",
            "query_point_count",
            "chamfer_g_gt_m",
            "virtual_rank_g_gt_among_ransac",
            "virtual_rank_g_gt_among_icp",
            "virtual_rank_g_gt_after_same_icp",
            "g_gt_same_icp_rotation_drift_deg",
            "selected_seed",
            "selected_icp_chamfer_m",
            "selected_ransac_g_rotation_error_deg",
            "selected_icp_g_rotation_error_deg",
            "h0_rotation_error_deg",
            "h0_object_center_translation_error_cm",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source in SOURCES:
            result = results[source]
            writer.writerow(
                {
                    "geometry_source": source,
                    "reference_point_count": result["reference_point_count"],
                    "query_point_count": result["query_point_count"],
                    "chamfer_g_gt_m": result["chamfer_g_gt_m"],
                    "virtual_rank_g_gt_among_ransac": result[
                        "virtual_rank_g_gt_among_ransac"
                    ],
                    "virtual_rank_g_gt_among_icp": result[
                        "virtual_rank_g_gt_among_icp"
                    ],
                    "virtual_rank_g_gt_after_same_icp": result[
                        "g_gt_after_same_icp"
                    ]["virtual_rank_among_icp"],
                    "g_gt_same_icp_rotation_drift_deg": result[
                        "g_gt_after_same_icp"
                    ]["rotation_drift_from_g_gt_deg"],
                    "selected_seed": result["selected_seed"],
                    "selected_icp_chamfer_m": result["selected_icp_chamfer_m"],
                    "selected_ransac_g_rotation_error_deg": result[
                        "selected_ransac_g_rotation_error_deg"
                    ],
                    "selected_icp_g_rotation_error_deg": result[
                        "selected_icp_g_rotation_error_deg"
                    ],
                    "h0_rotation_error_deg": result["h0_from_selected_icp"][
                        "rotation_error_deg"
                    ],
                    "h0_object_center_translation_error_cm": result[
                        "h0_from_selected_icp"
                    ]["object_center_translation_error_cm"],
                }
            )

    print(f"[written] {summary_path}")
    print(f"[written] {csv_path}")
    for source in SOURCES:
        result = results[source]
        h0 = result["h0_from_selected_icp"]
        gt_icp = result["g_gt_after_same_icp"]
        print(
            f"[{source}] GT virtual ICP rank "
            f"{result['virtual_rank_g_gt_among_icp']}/{result['candidate_count'] + 1}, "
            "GT-after-ICP rank/drift="
            f"{gt_icp['virtual_rank_among_icp']}/"
            f"{result['candidate_count'] + 1}/"
            f"{gt_icp['rotation_drift_from_g_gt_deg']:.3f} deg, "
            f"selected G rot error={result['selected_icp_g_rotation_error_deg']:.3f} deg, "
            f"H0 rot error={h0['rotation_error_deg']:.3f} deg, "
            f"H0 object-centre translation={h0['object_center_translation_error_cm']:.3f} cm"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
