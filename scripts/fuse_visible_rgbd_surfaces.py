from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_io.linemod_loader import load_linemod_view
from mesh_fusion.visible_surface_fusion import (
    build_depth_surface_mesh,
    fuse_aligned_surface_meshes,
    fuse_masked_rgbd_tsdf,
    transform_surface_mesh,
    write_triangle_mesh_ply,
)


def _load_run_configuration(run_root: Path) -> dict:
    configuration_path = run_root / "pipeline_config.json"
    if not configuration_path.is_file():
        raise FileNotFoundError(
            f"pipeline_config.json이 없습니다: {configuration_path}"
        )
    with configuration_path.open("r", encoding="utf-8") as file:
        configuration = json.load(file)
    for field in ("dataset_root", "split", "object_name", "reference", "query"):
        if field not in configuration:
            raise KeyError(f"pipeline_config.json에 {field}가 없습니다")
    return configuration


def _load_observation(
    *,
    run_root: Path,
    view_name: str,
    configuration: dict,
    mask_erosion_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = configuration[view_name]
    loaded_view = load_linemod_view(
        dataset_root=Path(configuration["dataset_root"]),
        view_name=view_name,
        scene_id=int(frame["scene_id"]),
        image_id=int(frame["image_id"]),
        object_name=str(configuration["object_name"]),
        object_id=configuration.get("object_id"),
        split=str(configuration["split"]),
    )
    view_root = run_root / "views" / view_name
    depth_path = view_root / "prepared" / "masked_depth_m.npy"
    mask_path = view_root / "segmentation" / "mask_bool.npy"
    if not depth_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(
            f"{view_name}의 prepared depth 또는 mask가 없습니다: {view_root}"
        )
    depth = np.load(depth_path, allow_pickle=False).astype(np.float64)
    mask = np.load(mask_path, allow_pickle=False).astype(bool)
    if mask_erosion_px > 0:
        mask = ndimage.binary_erosion(
            mask,
            iterations=mask_erosion_px,
            border_value=0,
        )
    depth = np.where(mask, depth, 0.0)
    return depth, mask, loaded_view.camera_matrix, loaded_view.rgb


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse reference/query masked depth into one open partial mesh "
            "in the reference-camera coordinate frame."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--pose-path",
        type=Path,
        required=True,
        help="4x4 NPY using T_query_camera_from_reference_camera",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pixel-stride", type=int, default=1)
    parser.add_argument("--mask-erosion-px", type=int, default=1)
    parser.add_argument(
        "--max-triangle-depth-delta-m",
        type=float,
        default=0.012,
    )
    parser.add_argument(
        "--max-triangle-edge-length-m",
        type=float,
        default=0.020,
    )
    parser.add_argument("--merge-distance-m", type=float, default=0.003)
    parser.add_argument("--minimum-normal-cosine", type=float, default=0.30)
    parser.add_argument("--tsdf-voxel-length-m", type=float, default=0.0015)
    parser.add_argument("--tsdf-sdf-truncation-m", type=float, default=0.006)
    parser.add_argument(
        "--minimum-component-triangle-count",
        type=int,
        default=20,
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    run_root = arguments.run_root.expanduser().resolve()
    pose_path = arguments.pose_path.expanduser().resolve()
    if not run_root.is_dir():
        raise NotADirectoryError(f"run root가 없습니다: {run_root}")
    if not pose_path.is_file():
        raise FileNotFoundError(f"pose 파일이 없습니다: {pose_path}")
    if arguments.mask_erosion_px < 0:
        raise ValueError("mask-erosion-px는 0 이상이어야 합니다")

    output_directory = (
        run_root / "visible_surface_fusion"
        if arguments.output_dir is None
        else arguments.output_dir.expanduser().resolve()
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    configuration = _load_run_configuration(run_root)

    reference_inputs = _load_observation(
        run_root=run_root,
        view_name="reference",
        configuration=configuration,
        mask_erosion_px=arguments.mask_erosion_px,
    )
    query_inputs = _load_observation(
        run_root=run_root,
        view_name="query",
        configuration=configuration,
        mask_erosion_px=arguments.mask_erosion_px,
    )
    reference_mesh = build_depth_surface_mesh(
        masked_depth_m=reference_inputs[0],
        mask_bool=reference_inputs[1],
        camera_k=reference_inputs[2],
        rgb=reference_inputs[3],
        pixel_stride=arguments.pixel_stride,
        maximum_triangle_depth_delta_m=(
            arguments.max_triangle_depth_delta_m
        ),
        maximum_triangle_edge_length_m=(
            arguments.max_triangle_edge_length_m
        ),
    )
    query_mesh = build_depth_surface_mesh(
        masked_depth_m=query_inputs[0],
        mask_bool=query_inputs[1],
        camera_k=query_inputs[2],
        rgb=query_inputs[3],
        pixel_stride=arguments.pixel_stride,
        maximum_triangle_depth_delta_m=(
            arguments.max_triangle_depth_delta_m
        ),
        maximum_triangle_edge_length_m=(
            arguments.max_triangle_edge_length_m
        ),
    )

    transform_query_from_reference = np.load(
        pose_path,
        allow_pickle=False,
    ).astype(np.float64)
    transform_reference_from_query = np.linalg.inv(
        transform_query_from_reference
    )
    query_mesh_in_reference = transform_surface_mesh(
        query_mesh,
        transform_reference_from_query,
    )
    diagnostic_vertex_fusion = fuse_aligned_surface_meshes(
        reference_mesh=reference_mesh,
        query_mesh_in_reference=query_mesh_in_reference,
        merge_distance_m=arguments.merge_distance_m,
        minimum_normal_cosine=arguments.minimum_normal_cosine,
    )
    fusion = fuse_masked_rgbd_tsdf(
        reference_depth_m=reference_inputs[0],
        reference_rgb=reference_inputs[3],
        reference_camera_k=reference_inputs[2],
        query_depth_m=query_inputs[0],
        query_rgb=query_inputs[3],
        query_camera_k=query_inputs[2],
        transform_query_from_reference=transform_query_from_reference,
        voxel_length_m=arguments.tsdf_voxel_length_m,
        sdf_truncation_m=arguments.tsdf_sdf_truncation_m,
        minimum_component_triangle_count=(
            arguments.minimum_component_triangle_count
        ),
    )

    reference_path = write_triangle_mesh_ply(
        output_directory / "reference_visible.ply",
        reference_mesh,
    )
    query_path = write_triangle_mesh_ply(
        output_directory / "query_visible_in_reference.ply",
        query_mesh_in_reference,
    )
    fused_path = write_triangle_mesh_ply(
        output_directory / "fused_visible_mesh.ply",
        fusion.mesh,
    )
    np.savez_compressed(
        output_directory / "fused_visible_mesh.npz",
        vertices_m=fusion.mesh.vertices_m,
        triangles=fusion.mesh.triangles,
        vertex_colors_rgb=fusion.mesh.vertex_colors_rgb,
        vertex_normals=fusion.mesh.vertex_normals,
    )
    np.save(
        output_directory / "T_query_from_reference.npy",
        transform_query_from_reference,
        allow_pickle=False,
    )
    np.save(
        output_directory / "T_reference_from_query.npy",
        transform_reference_from_query,
        allow_pickle=False,
    )

    diagnostics = dict(fusion.diagnostics)
    diagnostics.update(
        {
            "status": "COMPLETED",
            "surface_type": "open_partial_visible_surface",
            "pose_convention": "T_query_camera_from_reference_camera",
            "pose_source_path": str(pose_path),
            "run_root": str(run_root),
            "pixel_stride": int(arguments.pixel_stride),
            "mask_erosion_px": int(arguments.mask_erosion_px),
            "maximum_triangle_depth_delta_m": float(
                arguments.max_triangle_depth_delta_m
            ),
            "maximum_triangle_edge_length_m": float(
                arguments.max_triangle_edge_length_m
            ),
            "diagnostic_vertex_merge": (
                diagnostic_vertex_fusion.diagnostics
            ),
            "reference_mesh_path": str(reference_path),
            "query_mesh_in_reference_path": str(query_path),
            "fused_mesh_path": str(fused_path),
        }
    )
    diagnostics_path = output_directory / "fusion_diagnostics.json"
    with diagnostics_path.open("w", encoding="utf-8") as file:
        json.dump(diagnostics, file, indent=2, ensure_ascii=False)

    print(f"[Visible surface fusion] {fused_path}")
    print(f"[Fusion diagnostics] {diagnostics_path}")
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
