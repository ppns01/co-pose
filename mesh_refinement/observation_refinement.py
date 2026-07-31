from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from core.types import PreparedView
from mesh_refinement.silhouette_mesh_refiner import (
    refine_mesh_for_silhouette_and_depth,
)
from pose.dgedi_runner import _diameter
from pose.relative_pose_builder import SelfAlignmentSelection


@dataclass(frozen=True)
class ObservationRefinementResult:
    self_alignment: SelfAlignmentSelection
    accepted: bool
    reasons: tuple[str, ...]
    diagnostics: dict


def _silhouette_quality(
    *,
    points_camera: np.ndarray,
    mask_bool: np.ndarray,
    camera_k: np.ndarray,
) -> tuple[float, float]:
    """camera-frame mesh 정점 투영이 실제 mask와 얼마나 잘 맞는지
    (IoU, 평균 boundary distance)를 계산한다. Pose 재추정(track_one)
    결과를 실제 관측과 비교 검증하는 데 쓴다."""
    from mesh_refinement.depth_anchored_visible_refiner import (
        _visible_vertex_mask,
    )
    from mesh_refinement.silhouette_mesh_refiner import (
        _signed_distance_to_mask,
    )

    image_height, image_width = mask_bool.shape
    visible_mask, u_pixel, v_pixel = _visible_vertex_mask(
        points_camera=points_camera,
        camera_k=camera_k,
        image_height=image_height,
        image_width=image_width,
        mask_bool=np.ones_like(mask_bool),
    )

    if not visible_mask.any():
        return 0.0, float("inf")

    silhouette = np.zeros((image_height, image_width), dtype=bool)
    silhouette[v_pixel[visible_mask], u_pixel[visible_mask]] = True

    intersection = np.count_nonzero(silhouette & mask_bool)
    union = np.count_nonzero(silhouette | mask_bool)
    iou = intersection / union if union else 0.0

    sdf = _signed_distance_to_mask(mask_bool)
    boundary_distance = float(
        np.abs(sdf[v_pixel[visible_mask], u_pixel[visible_mask]]).mean()
    )

    return iou, boundary_distance


def _acceptance_gate(diagnostics: dict) -> tuple[bool, tuple[str, ...]]:
    """
    Refined mesh를 그대로 채택하지 않고, IoU/boundary/S*/centroid/
    displacement 전부가 안전 범위 안에 있는지 확인한다. 하나라도
    실패하면 호출자가 원본 self_alignment로 되돌린다.
    """
    reasons: list[str] = []

    if diagnostics["iou_after"] < diagnostics["iou_before"]:
        reasons.append(
            "IoU did not improve "
            f"({diagnostics['iou_before']:.3f} -> "
            f"{diagnostics['iou_after']:.3f})"
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

    target_scale_m = diagnostics["target_scale_m"]
    if target_scale_m:
        scale_drift = abs(
            diagnostics["scale_after_reprojection_m"] - target_scale_m
        ) / target_scale_m
        if scale_drift > 0.005:
            reasons.append(
                f"S* drift too large ({scale_drift * 100:.2f}%)"
            )

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

    return len(reasons) == 0, tuple(reasons)


def refine_self_alignment_with_observation(
    *,
    self_alignment: SelfAlignmentSelection,
    prepared_view: PreparedView,
    foundationpose_runner: Any,
    output_directory: Path,
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
    proxy_points = np.asarray(proxy_mesh.vertices, dtype=np.float64)
    triangles = np.asarray(proxy_mesh.triangles, dtype=np.int64)

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

    refinement = refine_mesh_for_silhouette_and_depth(
        points_camera=points_camera,
        triangles=triangles,
        mask_bool=mask_bool,
        camera_k=camera_k,
        masked_depth_m=masked_depth_m,
        target_scale_m=target_scale_m,
        diameter_fn=_diameter,
    )

    accepted, reasons = _acceptance_gate(refinement.diagnostics)

    if not accepted:
        return ObservationRefinementResult(
            self_alignment=self_alignment,
            accepted=False,
            reasons=reasons,
            diagnostics=refinement.diagnostics,
        )

    refined_proxy_points = (
        refinement.refined_points_camera - translation
    ) @ rotation

    refined_proxy_mesh = o3d.geometry.TriangleMesh()
    refined_proxy_mesh.vertices = o3d.utility.Vector3dVector(
        refined_proxy_points
    )
    refined_proxy_mesh.triangles = proxy_mesh.triangles
    refined_proxy_mesh.compute_vertex_normals()

    refined_proxy_path = (
        output_directory
        / f"{self_alignment.proxy_view}_refined_proxy.obj"
    )
    o3d.io.write_triangle_mesh(
        str(refined_proxy_path),
        refined_proxy_mesh,
        write_ascii=False,
        compressed=False,
        print_progress=False,
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
        mask_bool=mask_bool,
        camera_k=camera_k,
    )

    points_at_t0 = refined_proxy_points @ rotation.T + translation
    iou_t0, boundary_t0 = _silhouette_quality(
        points_camera=points_at_t0,
        mask_bool=mask_bool,
        camera_k=camera_k,
    )

    pose_refinement_accepted = iou_t1 >= iou_t0 - 0.02
    final_pose = (
        refined_pose if pose_refinement_accepted else pose_camera_from_proxy
    )

    diagnostics = dict(refinement.diagnostics)
    diagnostics.update(
        {
            "pose_refinement_accepted": pose_refinement_accepted,
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

    return ObservationRefinementResult(
        self_alignment=refined_self_alignment,
        accepted=True,
        reasons=(),
        diagnostics=diagnostics,
    )
