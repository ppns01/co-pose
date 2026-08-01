from __future__ import annotations

import json
import os

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import scipy.ndimage

from core.types import PreparedView
from mesh_refinement.silhouette_mesh_refiner import (
    refine_mesh_for_silhouette_and_depth,
)
from mesh_refinement.dense_strip_arap_refiner import (
    refine_mesh_with_dense_strip_arap,
)
from mesh_refinement.iterative_contour_arap_refiner import (
    refine_mesh_with_iterative_contour_arap,
)
from mesh_refinement.weighted_visible_arap_refiner import (
    refine_mesh_with_weighted_visible_arap,
)
from mesh_refinement.stage_observation_visualizer import visualize_refinement_stages
from pose.dgedi_runner import _diameter
from pose.relative_pose_builder import SelfAlignmentSelection


@dataclass(frozen=True)
class ObservationRefinementResult:
    self_alignment: SelfAlignmentSelection
    accepted: bool
    reasons: tuple[str, ...]
    diagnostics: dict


def _diagnostic_to_jsonable(
    value: Any,
) -> Any:
    """numpy/Path가 포함된 diagnostics를 JSON-safe 값으로 변환한다."""

    if isinstance(value, dict):
        return {
            str(key): _diagnostic_to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _diagnostic_to_jsonable(item)
            for item in value
        ]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    return repr(value)


def _write_observation_refinement_diagnostics(
    *,
    output_directory: Path,
    filename: str,
    diagnostics: dict,
    accepted: bool | None = None,
    reasons: tuple[str, ...] = (),
) -> Path:
    """Observation refinement diagnostics를 원자적으로 저장한다."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory / filename
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    payload = _diagnostic_to_jsonable(
        diagnostics
    )

    if not isinstance(payload, dict):
        payload = {
            "diagnostics": payload,
        }

    payload["_result"] = {
        "accepted": accepted,
        "reasons": list(reasons),
    }

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        output_path
    )

    print(
        "[Observation refinement diagnostics] "
        f"{output_path}"
    )

    return output_path



def _write_camera_stage_mesh(
    *,
    path: Path,
    points_camera: np.ndarray,
    triangles: np.ndarray,
    vertex_colors: np.ndarray | None = None,
) -> None:
    mesh = o3d.geometry.TriangleMesh()

    mesh.vertices = (
        o3d.utility.Vector3dVector(
            np.asarray(
                points_camera,
                dtype=np.float64,
            )
        )
    )

    mesh.triangles = (
        o3d.utility.Vector3iVector(
            np.asarray(
                triangles,
                dtype=np.int32,
            )
        )
    )

    if vertex_colors is not None:
        colors = np.asarray(
            vertex_colors,
            dtype=np.float64,
        )

        expected_shape = (
            len(points_camera),
            3,
        )

        if colors.shape != expected_shape:
            raise ValueError(
                "Stage vertex color shape mismatch: "
                f"colors={colors.shape}, "
                f"expected={expected_shape}"
            )

        mesh.vertex_colors = (
            o3d.utility.Vector3dVector(
                colors.copy()
            )
        )

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = o3d.io.write_triangle_mesh(
        str(path),
        mesh,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    )

    if not success:
        raise OSError(
            "Failed to write refinement "
            f"stage mesh: {path}"
        )


def _rotation_delta_deg(
    rotation_t0: np.ndarray,
    rotation_t1: np.ndarray,
) -> float:
    """
    동일한 external refined-proxy frame에 대한
    T0와 T1 사이의 누적 geodesic rotation 차이를 계산한다.
    """
    rotation_t0 = np.asarray(
        rotation_t0,
        dtype=np.float64,
    )

    rotation_t1 = np.asarray(
        rotation_t1,
        dtype=np.float64,
    )

    if (
        rotation_t0.shape != (3, 3)
        or rotation_t1.shape != (3, 3)
    ):
        raise ValueError(
            "rotation_t0 and rotation_t1 "
            "must both have shape (3, 3)"
        )

    relative_rotation = (
        rotation_t1
        @ rotation_t0.T
    )

    cosine = np.clip(
        (
            np.trace(
                relative_rotation
            )
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


def _silhouette_quality(
    *,
    points_camera: np.ndarray,
    triangles: np.ndarray,
    mask_bool: np.ndarray,
    camera_k: np.ndarray,
) -> tuple[float, float]:
    """Triangle raster 기준 mask IoU와 contour distance를 계산한다."""
    from mesh_refinement.iterative_contour_arap_refiner import (
        _raycast_mesh,
    )

    image_height, image_width = mask_bool.shape
    raycast = _raycast_mesh(
        points_camera=points_camera,
        triangles=triangles,
        camera_k=camera_k,
        image_height=image_height,
        image_width=image_width,
    )
    rendered_mask = np.asarray(
        raycast["rendered_mask"],
        dtype=bool,
    )
    if not rendered_mask.any():
        return 0.0, float("inf")

    intersection = np.count_nonzero(rendered_mask & mask_bool)
    union = np.count_nonzero(rendered_mask | mask_bool)
    iou = intersection / union if union else 0.0

    rendered_boundary = (
        rendered_mask
        & ~scipy.ndimage.binary_erosion(
            rendered_mask,
            iterations=1,
            border_value=0,
        )
    )
    observed_boundary = (
        mask_bool
        & ~scipy.ndimage.binary_erosion(
            mask_bool,
            iterations=1,
            border_value=0,
        )
    )
    if not rendered_boundary.any() or not observed_boundary.any():
        return float(iou), float("inf")
    distance_to_observed = scipy.ndimage.distance_transform_edt(
        ~observed_boundary
    )
    distance_to_rendered = scipy.ndimage.distance_transform_edt(
        ~rendered_boundary
    )
    boundary_distance = 0.5 * (
        float(distance_to_observed[rendered_boundary].mean())
        + float(distance_to_rendered[observed_boundary].mean())
    )

    return float(iou), float(boundary_distance)


def _acceptance_gate(diagnostics: dict) -> tuple[bool, tuple[str, ...]]:
    """
    Refined mesh를 그대로 채택하지 않고, IoU/boundary/S*/centroid/
    displacement 전부가 안전 범위 안에 있는지 확인한다. 하나라도
    실패하면 호출자가 원본 self_alignment로 되돌린다.
    """
    reasons: list[str] = []

    iou_before = float(
        diagnostics.get(
            "raster_iou_before",
            diagnostics["iou_before"],
        )
    )
    iou_after = float(
        diagnostics.get(
            "raster_iou_after",
            diagnostics["iou_after"],
        )
    )
    if iou_after < iou_before - 1e-6:
        reasons.append(
            "IoU did not improve "
            f"({iou_before:.3f} -> {iou_after:.3f})"
        )

    boundary_before = diagnostics["boundary_distance_before_px"]
    boundary_after = diagnostics["boundary_distance_after_px"]
    if (
        boundary_before is not None
        and boundary_after is not None
        and boundary_after > boundary_before + 0.1
    ):
        reasons.append(
            "boundary distance got worse "
            f"({boundary_before:.2f} -> {boundary_after:.2f} px)"
        )

    iou_improved = iou_after > iou_before + 1e-4
    boundary_improved = (
        boundary_before is not None
        and boundary_after is not None
        and boundary_after < boundary_before - 0.1
    )
    if not iou_improved and not boundary_improved:
        reasons.append(
            "observation fit did not measurably improve "
            f"(IoU {iou_before:.3f} -> {iou_after:.3f})"
        )

    target_scale_m = diagnostics["target_scale_m"]
    if target_scale_m:
        scale_drift = abs(
            diagnostics["scale_after_reprojection_m"] - target_scale_m
        ) / target_scale_m
        if scale_drift > 0.005:
            reasons.append(
                f"S* drift too large ({scale_drift * 100:.2f}%)"
            )

    scale_correction_ratio = diagnostics.get(
        "scale_correction_ratio"
    )
    if (
        scale_correction_ratio is not None
        and abs(float(scale_correction_ratio) - 1.0) > 0.005
    ):
        reasons.append(
            "required S* correction too large "
            f"({(float(scale_correction_ratio) - 1.0) * 100:.2f}%)"
        )

    final_topology = diagnostics.get(
        "final_cumulative_topology"
    )
    if (
        isinstance(final_topology, dict)
        and not final_topology.get("topology_safe", False)
    ):
        reasons.append("final cumulative topology is unsafe")

    if diagnostics["centroid_drift_m"] > 0.003:
        reasons.append(
            "centroid drift too large "
            f"({diagnostics['centroid_drift_m'] * 1000:.2f}mm)"
        )

    if diagnostics["displacement_max_m"] > 0.009:
        reasons.append(
            "max displacement near/at cap "
            f"({diagnostics['displacement_max_m'] * 1000:.2f}mm)"
        )

    roughness_before = diagnostics.get("roughness_before_p95_m")
    roughness_after = diagnostics.get("roughness_after_p95_m")
    if roughness_before is not None and roughness_after is not None:
        roughness_ceiling = max(roughness_before * 1.5, 0.001)
        if roughness_after > roughness_ceiling:
            reasons.append(
                "surface roughness got worse "
                f"({roughness_before * 1000:.2f}mm -> "
                f"{roughness_after * 1000:.2f}mm p95, "
                f"ceiling={roughness_ceiling * 1000:.2f}mm) -- "
                "IoU/boundary distance improved but local surface "
                "quality did not"
            )

    return len(reasons) == 0, tuple(reasons)


def refine_self_alignment_with_observation(
    *,
    self_alignment: SelfAlignmentSelection,
    prepared_view: PreparedView,
    foundationpose_runner: Any,
    output_directory: Path,
    pose_trust_region_max_rotation_delta_deg: float,
    pose_trust_region_max_translation_ratio: float,
    pose_trust_region_max_iou_drop: float,
) -> ObservationRefinementResult:
    """
    Depth+silhouette로 self-aligned proxy mesh의 visible geometry를
    국소 보정하고(interior=depth, boundary=silhouette), 통과하면
    보정된 mesh에 대해 FoundationPose local refinement로 R,t만
    다시 추정한다 -- scale(S*)은 건드리지 않는다.

    처리 순서:
      1. proxy mesh(scaled_mesh_path)를 기존 T0로 camera frame에 배치
      2. depth(interior)/silhouette(boundary) 국소 보정
      3. accept/reject 게이트 -- 실패하면 원본 self_alignment 그대로 반환
      4. 통과하면 보정된 camera-frame mesh를 T0의 역변환으로 다시
         proxy frame으로 복원 (전체 rigid 역변환, rotation+translation)
      5. FoundationPoseRunner.refine_pose_locally()로 T0 근처에서만
         R,t 재추정 (전역 재탐색 아님, register()가 아니라 track_one()
         경로) -> T1
      6. 새 SelfAlignmentSelection(scaled_mesh_path=refined proxy,
         pose_camera_from_proxy=T1) 반환
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    proxy_mesh = o3d.io.read_triangle_mesh(
        str(self_alignment.scaled_mesh_path)
    )
    proxy_points = np.asarray(
        proxy_mesh.vertices,
        dtype=np.float64,
    )

    triangles = np.asarray(
        proxy_mesh.triangles,
        dtype=np.int64,
    )

    proxy_vertex_colors: np.ndarray | None = None

    if proxy_mesh.has_vertex_colors():
        colors = np.asarray(
            proxy_mesh.vertex_colors,
            dtype=np.float64,
        )

        if colors.shape == (
            len(proxy_points),
            3,
        ):
            proxy_vertex_colors = (
                colors.copy()
            )
        else:
            raise ValueError(
                "Proxy vertex color shape mismatch: "
                f"colors={colors.shape}, "
                f"vertices={len(proxy_points)}"
            )

    print(
        "[Observation refinement source colors] "
        f"view={self_alignment.proxy_view} "
        f"has_vertex_colors="
        f"{proxy_vertex_colors is not None} "
        f"vertex_count={len(proxy_points)}"
    )

    pose_camera_from_proxy = np.asarray(
        self_alignment.pose_camera_from_proxy, dtype=np.float64
    )
    rotation = pose_camera_from_proxy[:3, :3]
    translation = pose_camera_from_proxy[:3, 3]

    points_camera = proxy_points @ rotation.T + translation

    camera_k = np.asarray(
        prepared_view.view.camera_matrix, dtype=np.float64
    )
    depth_m = np.asarray(prepared_view.view.depth_m, dtype=np.float64)
    mask_bool = np.asarray(
        prepared_view.segmentation.mask_bool, dtype=np.bool_
    )
    masked_depth_m = np.where(mask_bool, depth_m, 0.0)

    target_scale_m = float(_diameter(points_camera))

    refinement_mode = os.environ.get(
        "COPOSE_REFINEMENT_MODE",
        "dense_strip_arap",
    ).strip().lower()

    if refinement_mode == "dense_strip_arap":
        refinement_function = (
            refine_mesh_with_dense_strip_arap
        )
    elif refinement_mode == "iterative_contour_arap":
        refinement_function = (
            refine_mesh_with_iterative_contour_arap
        )
    elif refinement_mode == "weighted_visible_arap":
        refinement_function = (
            refine_mesh_with_weighted_visible_arap
        )
    elif refinement_mode == "legacy":
        refinement_function = (
            refine_mesh_for_silhouette_and_depth
        )
    else:
        raise ValueError(
            "COPOSE_REFINEMENT_MODE must be "
            "'legacy', 'dense_strip_arap', or "
            "'iterative_contour_arap', or "
            "'weighted_visible_arap', "
            f"got {refinement_mode!r}"
        )

    print(
        "[Observation refinement mode] "
        f"{refinement_mode}"
    )

    refinement = refinement_function(
        points_camera=points_camera,
        triangles=triangles,
        mask_bool=mask_bool,
        camera_k=camera_k,
        masked_depth_m=masked_depth_m,
        target_scale_m=target_scale_m,
        diameter_fn=_diameter,
    )

    if refinement_mode == "dense_strip_arap":
        print(
            "[Dense-strip ARAP topology] "
            f"{refinement.diagnostics.get('topology_safety')}"
        )
    elif refinement_mode == "iterative_contour_arap":
        print(
            "[Iterative-contour ARAP topology] "
            f"{refinement.diagnostics.get('topology_safety')}"
        )
        print(
            "[Iterative-contour ARAP raster IoU] "
            "before="
            f"{refinement.diagnostics.get('raster_iou_before')} "
            "raw="
            f"{refinement.diagnostics.get('raster_iou_raw')} "
            "after="
            f"{refinement.diagnostics.get('raster_iou_after')} "
            "outer_iterations_completed="
            f"{refinement.diagnostics.get('outer_iteration_count_completed')}"
        )
    elif refinement_mode == "weighted_visible_arap":
        print(
            "[Weighted-visible ARAP topology] "
            f"{refinement.diagnostics.get('final_cumulative_topology')}"
        )
        print(
            "[Weighted-visible ARAP raster IoU] "
            "before="
            f"{refinement.diagnostics.get('raster_iou_before')} "
            "pre_scale="
            f"{refinement.diagnostics.get('raster_iou_raw')} "
            "after="
            f"{refinement.diagnostics.get('raster_iou_after')}"
        )

    diagnostics = dict(
        refinement.diagnostics
    )

    _write_observation_refinement_diagnostics(
        output_directory=output_directory,
        filename="refinement_diagnostics_raw.json",
        diagnostics=diagnostics,
        accepted=None,
        reasons=(),
    )

    stage_directory = (
        output_directory
        / "refinement_stages"
    )

    stage_mesh_paths: dict[
        str,
        str,
    ] = {}

    for (
        stage_name,
        stage_points,
    ) in (
        refinement
        .intermediate_points_camera
        .items()
    ):
        stage_path = (
            stage_directory
            / f"{stage_name}.obj"
        )

        _write_camera_stage_mesh(
            path=stage_path,
            points_camera=stage_points,
            triangles=triangles,
            vertex_colors=(
                proxy_vertex_colors
            ),
        )

        stage_mesh_paths[
            stage_name
        ] = str(stage_path)

    diagnostics[
        "intermediate_mesh_paths"
    ] = stage_mesh_paths

    observation_visualization = (
        visualize_refinement_stages(
            output_directory=(
                output_directory
                / "observation_stage_comparison"
            ),
            rgb=np.asarray(
                prepared_view.view.rgb,
                dtype=np.uint8,
            ),
            observed_mask=mask_bool,
            observed_depth_m=depth_m,
            camera_k=camera_k,
            triangles=triangles,
            stages=(
                refinement
                .intermediate_points_camera
            ),
        )
    )

    diagnostics[
        "observation_stage_metrics"
    ] = observation_visualization[
        "metrics"
    ]

    diagnostics[
        "observation_stage_visualization"
    ] = {
        key: value
        for key, value
        in observation_visualization.items()
        if key != "metrics"
    }

    accepted, reasons = _acceptance_gate(
        diagnostics
    )

    if not accepted:
        _write_observation_refinement_diagnostics(
            output_directory=output_directory,
            filename="refinement_diagnostics.json",
            diagnostics=diagnostics,
            accepted=False,
            reasons=reasons,
        )

        return ObservationRefinementResult(
            self_alignment=self_alignment,
            accepted=False,
            reasons=reasons,
            diagnostics=diagnostics,
        )

    refined_proxy_points = (
        refinement.refined_points_camera - translation
    ) @ rotation

    refined_proxy_mesh = o3d.geometry.TriangleMesh()
    refined_proxy_mesh.vertices = o3d.utility.Vector3dVector(
        refined_proxy_points
    )
    refined_proxy_mesh.triangles = (
        proxy_mesh.triangles
    )

    if proxy_vertex_colors is not None:
        refined_proxy_mesh.vertex_colors = (
            o3d.utility.Vector3dVector(
                proxy_vertex_colors.copy()
            )
        )

    refined_proxy_mesh.compute_triangle_normals()
    refined_proxy_mesh.compute_vertex_normals()

    refined_proxy_path = (
        output_directory
        / f"{self_alignment.proxy_view}_refined_proxy.obj"
    )
    saved_refined_proxy = (
        o3d.io.write_triangle_mesh(
            str(refined_proxy_path),
            refined_proxy_mesh,
            write_ascii=False,
            compressed=False,
            print_progress=False,
        )
    )

    if not saved_refined_proxy:
        raise OSError(
            "Failed to write refined proxy: "
            f"{refined_proxy_path}"
        )

    saved_mesh_check = (
        o3d.io.read_triangle_mesh(
            str(refined_proxy_path),
            enable_post_processing=False,
        )
    )

    saved_has_vertex_colors = bool(
        saved_mesh_check.has_vertex_colors()
    )

    if (
        proxy_vertex_colors is not None
        and not saved_has_vertex_colors
    ):
        raise RuntimeError(
            "Refined proxy lost vertex colors "
            "during serialization: "
            f"{refined_proxy_path}"
        )

    diagnostics[
        "source_has_vertex_colors"
    ] = bool(
        proxy_vertex_colors is not None
    )

    diagnostics[
        "refined_proxy_has_vertex_colors"
    ] = saved_has_vertex_colors

    print(
        "[Observation refinement refined colors] "
        f"view={self_alignment.proxy_view} "
        f"has_vertex_colors="
        f"{saved_has_vertex_colors} "
        f"path={refined_proxy_path}"
    )

    refined_pose = foundationpose_runner.refine_pose_locally(
        mesh_path=refined_proxy_path,
        prepared_view=prepared_view,
        initial_pose_camera_from_mesh=pose_camera_from_proxy,
    )

    # --- pose refinement 검증: track_one()은 단일 forward pass라 발산
    #     가능성을 스스로 체크하지 않는다 (dGeDi의 여러-hypothesis
    #     rescoring과 달리, register()의 top_k 재점수화도 없다). 실측
    #     결과(duck Q3 query) self-pose IoU가 낮아 형상 보정이 크게
    #     들어간 경우, refiner가 T0보다 훨씬 나쁜 pose로 발산하는 걸
    #     확인했다 -- T1을 실제 mask와 비교해서 T0보다 나쁘면 되돌린다. ---
    r1 = refined_pose[:3, :3]
    t1 = refined_pose[:3, 3]
    points_at_t1 = refined_proxy_points @ r1.T + t1
    iou_t1, boundary_t1 = _silhouette_quality(
        points_camera=points_at_t1,
        triangles=triangles,
        mask_bool=mask_bool,
        camera_k=camera_k,
    )

    points_at_t0 = refined_proxy_points @ rotation.T + translation
    iou_t0, boundary_t0 = _silhouette_quality(
        points_camera=points_at_t0,
        triangles=triangles,
        mask_bool=mask_bool,
        camera_k=camera_k,
    )

    # --------------------------------------------------------
    # T0와 T1은 모두 같은 refined-proxy frame에 대한
    # external object-to-camera pose이다.
    #
    # track_one()은 local update 용도이므로 IoU만 개선됐더라도
    # T0에서 지나치게 멀리 이동한 T1은 다른 pose branch로
    # 간주하여 거부한다.
    # --------------------------------------------------------
    pose_t0 = np.asarray(
        pose_camera_from_proxy,
        dtype=np.float64,
    )

    pose_t1 = np.asarray(
        refined_pose,
        dtype=np.float64,
    )

    if (
        pose_t0.shape != (4, 4)
        or pose_t1.shape != (4, 4)
    ):
        raise ValueError(
            "T0 and T1 must both have "
            "shape (4, 4)"
        )

    maximum_rotation_delta_deg = float(
        pose_trust_region_max_rotation_delta_deg
    )

    maximum_translation_delta_ratio = float(
        pose_trust_region_max_translation_ratio
    )

    maximum_iou_drop = float(
        pose_trust_region_max_iou_drop
    )

    if (
        not np.isfinite(
            maximum_rotation_delta_deg
        )
        or maximum_rotation_delta_deg < 0.0
    ):
        raise ValueError(
            "COPOSE_TRACK_MAX_ROTATION_DELTA_DEG "
            "must be finite and non-negative"
        )

    if (
        not np.isfinite(
            maximum_translation_delta_ratio
        )
        or maximum_translation_delta_ratio < 0.0
    ):
        raise ValueError(
            "COPOSE_TRACK_MAX_TRANSLATION_RATIO "
            "must be finite and non-negative"
        )

    if (
        not np.isfinite(
            maximum_iou_drop
        )
        or maximum_iou_drop < 0.0
    ):
        raise ValueError(
            "COPOSE_TRACK_MAX_IOU_DROP "
            "must be finite and non-negative"
        )

    pose_rotation_delta_deg = (
        _rotation_delta_deg(
            pose_t0[:3, :3],
            pose_t1[:3, :3],
        )
    )

    pose_translation_delta_m = float(
        np.linalg.norm(
            pose_t1[:3, 3]
            - pose_t0[:3, 3]
        )
    )

    pose_translation_scale_m = float(
        target_scale_m
    )

    if (
        not np.isfinite(
            pose_translation_scale_m
        )
        or pose_translation_scale_m <= 1e-12
    ):
        raise ValueError(
            "Invalid target scale for "
            "pose translation trust region: "
            f"{pose_translation_scale_m}"
        )

    pose_translation_delta_ratio = (
        pose_translation_delta_m
        / pose_translation_scale_m
    )

    pose_iou_gate_passed = bool(
        iou_t1
        >= iou_t0
        - maximum_iou_drop
    )

    pose_rotation_gate_passed = bool(
        pose_rotation_delta_deg
        <= maximum_rotation_delta_deg
    )

    pose_translation_gate_passed = bool(
        pose_translation_delta_ratio
        <= maximum_translation_delta_ratio
    )

    pose_refinement_rejection_reasons: list[str] = []

    if not pose_iou_gate_passed:
        pose_refinement_rejection_reasons.append(
            "IoU gate failed: "
            f"{iou_t0:.6f} -> {iou_t1:.6f}, "
            f"maximum drop={maximum_iou_drop:.6f}"
        )

    if not pose_rotation_gate_passed:
        pose_refinement_rejection_reasons.append(
            "rotation trust region failed: "
            f"{pose_rotation_delta_deg:.3f} deg > "
            f"{maximum_rotation_delta_deg:.3f} deg"
        )

    if not pose_translation_gate_passed:
        pose_refinement_rejection_reasons.append(
            "translation trust region failed: "
            f"{pose_translation_delta_ratio:.6f} S* > "
            f"{maximum_translation_delta_ratio:.6f} S* "
            f"({pose_translation_delta_m:.6f} m / "
            f"{pose_translation_scale_m:.6f} m)"
        )

    pose_refinement_accepted = bool(
        pose_iou_gate_passed
        and pose_rotation_gate_passed
        and pose_translation_gate_passed
    )

    final_pose = (
        pose_t1
        if pose_refinement_accepted
        else pose_t0
    )

    print(
        "[Pose trust-region] "
        f"view={self_alignment.proxy_view} "
        f"accepted={pose_refinement_accepted} "
        f"selected={'T1' if pose_refinement_accepted else 'T0'} "
        f"rotation_delta_deg={pose_rotation_delta_deg:.3f} "
        f"translation_delta_m={pose_translation_delta_m:.6f} "
        f"translation_delta_ratio={pose_translation_delta_ratio:.6f} "
        f"iou={iou_t0:.6f}->{iou_t1:.6f} "
        f"limits=("
        f"{maximum_rotation_delta_deg:.3f}deg, "
        f"{maximum_translation_delta_ratio:.6f}S*, "
        f"iou_drop={maximum_iou_drop:.6f})"
    )

    if pose_refinement_rejection_reasons:
        print(
            "[Pose trust-region rejection] "
            + " | ".join(
                pose_refinement_rejection_reasons
            )
        )

    diagnostics.update(
        {
            "pose_refinement_accepted": pose_refinement_accepted,
            "pose_selected": (
                "T1"
                if pose_refinement_accepted
                else "T0"
            ),
            "pose_refinement_rejection_reasons": list(
                pose_refinement_rejection_reasons
            ),
            "pose_iou_gate_passed": pose_iou_gate_passed,
            "pose_rotation_gate_passed": pose_rotation_gate_passed,
            "pose_translation_gate_passed": pose_translation_gate_passed,
            "pose_rotation_delta_deg": pose_rotation_delta_deg,
            "pose_translation_delta_m": pose_translation_delta_m,
            "pose_translation_delta_ratio": pose_translation_delta_ratio,
            "pose_translation_scale_m": pose_translation_scale_m,
            "pose_maximum_rotation_delta_deg": (
                maximum_rotation_delta_deg
            ),
            "pose_maximum_translation_delta_ratio": (
                maximum_translation_delta_ratio
            ),
            "pose_maximum_iou_drop": maximum_iou_drop,
            "iou_t0": iou_t0,
            "iou_t1": iou_t1,
            "boundary_distance_t0_px": boundary_t0,
            "boundary_distance_t1_px": boundary_t1,
        }
    )

    refined_self_alignment = replace(
        self_alignment,
        scaled_mesh_path=refined_proxy_path,
        pose_camera_from_proxy=final_pose.astype(np.float32),
    )

    _write_observation_refinement_diagnostics(
        output_directory=output_directory,
        filename="refinement_diagnostics.json",
        diagnostics=diagnostics,
        accepted=True,
        reasons=(),
    )

    return ObservationRefinementResult(
        self_alignment=refined_self_alignment,
        accepted=True,
        reasons=(),
        diagnostics=diagnostics,
    )
