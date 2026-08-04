from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import scipy.ndimage
from PIL import Image, ImageDraw


def _render_mesh_depth(
    *,
    points_camera: np.ndarray,
    triangles: np.ndarray,
    camera_k: np.ndarray,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Camera-frame mesh를 Open3D ray casting으로 렌더링한다.

    반환:
        rendered_mask: triangle이 투영된 실제 채워진 mask
        rendered_depth: camera z-depth [m]
    """
    mesh = o3d.geometry.TriangleMesh()

    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(
            points_camera,
            dtype=np.float64,
        )
    )

    mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(
            triangles,
            dtype=np.int32,
        )
    )

    tensor_mesh = (
        o3d.t.geometry.TriangleMesh.from_legacy(
            mesh
        )
    )

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)

    camera_k = np.asarray(
        camera_k,
        dtype=np.float64,
    )

    fx = float(camera_k[0, 0])
    fy = float(camera_k[1, 1])
    cx = float(camera_k[0, 2])
    cy = float(camera_k[1, 2])

    pixel_u, pixel_v = np.meshgrid(
        np.arange(
            image_width,
            dtype=np.float32,
        ),
        np.arange(
            image_height,
            dtype=np.float32,
        ),
    )

    # 방향 벡터의 z를 1로 두므로 t_hit가 camera z-depth와 같다.
    ray_direction = np.stack(
        [
            (pixel_u - cx) / fx,
            (pixel_v - cy) / fy,
            np.ones_like(pixel_u),
        ],
        axis=-1,
    )

    ray_origin = np.zeros_like(
        ray_direction
    )

    rays = np.concatenate(
        [
            ray_origin,
            ray_direction,
        ],
        axis=-1,
    ).astype(np.float32)

    cast_result = scene.cast_rays(
        o3d.core.Tensor(
            rays,
            dtype=o3d.core.Dtype.Float32,
        )
    )

    rendered_depth = (
        cast_result["t_hit"]
        .numpy()
        .astype(np.float64)
    )

    rendered_mask = (
        np.isfinite(rendered_depth)
        & (rendered_depth > 0.0)
    )

    rendered_depth[
        ~rendered_mask
    ] = 0.0

    return (
        rendered_mask,
        rendered_depth,
    )


def _mask_boundary(
    mask_bool: np.ndarray,
) -> np.ndarray:
    mask_bool = np.asarray(
        mask_bool,
        dtype=bool,
    )

    if not mask_bool.any():
        return np.zeros_like(
            mask_bool
        )

    eroded = (
        scipy.ndimage.binary_erosion(
            mask_bool,
            iterations=1,
            border_value=0,
        )
    )

    return mask_bool & ~eroded


def _symmetric_boundary_distance_px(
    *,
    observed_mask: np.ndarray,
    rendered_mask: np.ndarray,
) -> float | None:
    observed_boundary = _mask_boundary(
        observed_mask
    )

    rendered_boundary = _mask_boundary(
        rendered_mask
    )

    if (
        not observed_boundary.any()
        or not rendered_boundary.any()
    ):
        return None

    distance_to_observed = (
        scipy.ndimage
        .distance_transform_edt(
            ~observed_boundary
        )
    )

    distance_to_rendered = (
        scipy.ndimage
        .distance_transform_edt(
            ~rendered_boundary
        )
    )

    rendered_to_observed = float(
        distance_to_observed[
            rendered_boundary
        ].mean()
    )

    observed_to_rendered = float(
        distance_to_rendered[
            observed_boundary
        ].mean()
    )

    return 0.5 * (
        rendered_to_observed
        + observed_to_rendered
    )


def _calculate_stage_metrics(
    *,
    observed_mask: np.ndarray,
    observed_depth_m: np.ndarray,
    rendered_mask: np.ndarray,
    rendered_depth_m: np.ndarray,
    boundary_exclusion_px: int = 2,
) -> dict[str, Any]:
    observed_mask = np.asarray(
        observed_mask,
        dtype=bool,
    )

    observed_depth_m = np.asarray(
        observed_depth_m,
        dtype=np.float64,
    )

    rendered_mask = np.asarray(
        rendered_mask,
        dtype=bool,
    )

    rendered_depth_m = np.asarray(
        rendered_depth_m,
        dtype=np.float64,
    )

    intersection = int(
        np.count_nonzero(
            observed_mask
            & rendered_mask
        )
    )

    union = int(
        np.count_nonzero(
            observed_mask
            | rendered_mask
        )
    )

    mask_iou = (
        float(intersection / union)
        if union > 0
        else 0.0
    )

    if boundary_exclusion_px > 0:
        depth_interior_mask = (
            scipy.ndimage
            .binary_erosion(
                observed_mask,
                iterations=(
                    boundary_exclusion_px
                ),
                border_value=0,
            )
        )
    else:
        depth_interior_mask = (
            observed_mask
        )

    depth_valid_mask = (
        depth_interior_mask
        & rendered_mask
        & np.isfinite(
            observed_depth_m
        )
        & (observed_depth_m > 0.0)
    )

    metrics: dict[str, Any] = {
        "mask_iou": mask_iou,
        "symmetric_boundary_distance_px":
            _symmetric_boundary_distance_px(
                observed_mask=observed_mask,
                rendered_mask=rendered_mask,
            ),
        "observed_mask_pixel_count": int(
            observed_mask.sum()
        ),
        "rendered_mask_pixel_count": int(
            rendered_mask.sum()
        ),
        "depth_overlap_pixel_count": int(
            depth_valid_mask.sum()
        ),
    }

    if not depth_valid_mask.any():
        metrics.update(
            {
                "depth_signed_median_m": None,
                "depth_abs_median_m": None,
                "depth_abs_mean_m": None,
                "depth_abs_p90_m": None,
                "depth_within_5mm_fraction": None,
                "depth_within_10mm_fraction": None,
            }
        )

        return metrics

    depth_residual = (
        observed_depth_m[
            depth_valid_mask
        ]
        - rendered_depth_m[
            depth_valid_mask
        ]
    )

    absolute_residual = np.abs(
        depth_residual
    )

    metrics.update(
        {
            "depth_signed_median_m": float(
                np.median(
                    depth_residual
                )
            ),
            "depth_abs_median_m": float(
                np.median(
                    absolute_residual
                )
            ),
            "depth_abs_mean_m": float(
                absolute_residual.mean()
            ),
            "depth_abs_p90_m": float(
                np.quantile(
                    absolute_residual,
                    0.90,
                )
            ),
            "depth_within_5mm_fraction": float(
                np.mean(
                    absolute_residual
                    <= 0.005
                )
            ),
            "depth_within_10mm_fraction": float(
                np.mean(
                    absolute_residual
                    <= 0.010
                )
            ),
        }
    )

    return metrics


def _save_mask_overlay(
    *,
    output_path: Path,
    rgb: np.ndarray,
    observed_mask: np.ndarray,
    original_rendered_mask: np.ndarray,
    current_rendered_mask: np.ndarray,
    stage_name: str,
) -> None:
    overlay = np.asarray(
        rgb,
        dtype=np.uint8,
    ).copy()

    observed_boundary = (
        scipy.ndimage
        .binary_dilation(
            _mask_boundary(
                observed_mask
            ),
            iterations=1,
        )
    )

    original_boundary = (
        scipy.ndimage
        .binary_dilation(
            _mask_boundary(
                original_rendered_mask
            ),
            iterations=1,
        )
    )

    current_boundary = (
        scipy.ndimage
        .binary_dilation(
            _mask_boundary(
                current_rendered_mask
            ),
            iterations=1,
        )
    )

    # 원본 M0: cyan
    overlay[
        original_boundary
    ] = np.array(
        [0, 255, 255],
        dtype=np.uint8,
    )

    # 현재 단계: orange
    overlay[
        current_boundary
    ] = np.array(
        [255, 128, 0],
        dtype=np.uint8,
    )

    # 실제 SAM mask: green
    overlay[
        observed_boundary
    ] = np.array(
        [0, 255, 0],
        dtype=np.uint8,
    )

    image = Image.fromarray(
        overlay,
        mode="RGB",
    )

    draw = ImageDraw.Draw(
        image
    )

    draw.rectangle(
        (
            0,
            0,
            min(
                620,
                overlay.shape[1] - 1,
            ),
            46,
        ),
        fill=(0, 0, 0),
    )

    draw.text(
        (8, 5),
        (
            f"{stage_name}: "
            "green=observed mask"
        ),
        fill=(255, 255, 255),
    )

    draw.text(
        (8, 24),
        (
            "cyan=M0 original, "
            "orange=current stage"
        ),
        fill=(255, 255, 255),
    )

    image.save(
        output_path
    )


def _save_depth_residual_image(
    *,
    output_path: Path,
    observed_mask: np.ndarray,
    observed_depth_m: np.ndarray,
    rendered_mask: np.ndarray,
    rendered_depth_m: np.ndarray,
    stage_name: str,
    display_limit_m: float = 0.020,
) -> None:
    observed_mask = np.asarray(
        observed_mask,
        dtype=bool,
    )

    observed_depth_m = np.asarray(
        observed_depth_m,
        dtype=np.float64,
    )

    valid_mask = (
        observed_mask
        & rendered_mask
        & np.isfinite(
            observed_depth_m
        )
        & (observed_depth_m > 0.0)
    )

    residual = (
        observed_depth_m
        - rendered_depth_m
    )

    normalized = np.clip(
        residual
        / display_limit_m,
        -1.0,
        1.0,
    )

    heatmap = np.full(
        (
            observed_depth_m.shape[0],
            observed_depth_m.shape[1],
            3,
        ),
        fill_value=30,
        dtype=np.uint8,
    )

    positive_mask = (
        valid_mask
        & (normalized >= 0.0)
    )

    negative_mask = (
        valid_mask
        & (normalized < 0.0)
    )

    positive_strength = (
        normalized[
            positive_mask
        ]
    )

    heatmap[
        positive_mask,
        0,
    ] = 255

    heatmap[
        positive_mask,
        1,
    ] = (
        255.0
        * (
            1.0
            - positive_strength
        )
    ).astype(np.uint8)

    heatmap[
        positive_mask,
        2,
    ] = heatmap[
        positive_mask,
        1,
    ]

    negative_strength = (
        -normalized[
            negative_mask
        ]
    )

    heatmap[
        negative_mask,
        2,
    ] = 255

    heatmap[
        negative_mask,
        0,
    ] = (
        255.0
        * (
            1.0
            - negative_strength
        )
    ).astype(np.uint8)

    heatmap[
        negative_mask,
        1,
    ] = heatmap[
        negative_mask,
        0,
    ]

    image = Image.fromarray(
        heatmap,
        mode="RGB",
    )

    draw = ImageDraw.Draw(
        image
    )

    draw.rectangle(
        (
            0,
            0,
            min(
                630,
                heatmap.shape[1] - 1,
            ),
            46,
        ),
        fill=(0, 0, 0),
    )

    draw.text(
        (8, 5),
        (
            f"{stage_name}: "
            "observed depth - rendered depth"
        ),
        fill=(255, 255, 255),
    )

    draw.text(
        (8, 24),
        (
            "red=mesh in front, "
            "blue=mesh behind, "
            "range=+/-20mm"
        ),
        fill=(255, 255, 255),
    )

    image.save(
        output_path
    )


def _save_contact_sheet(
    *,
    output_path: Path,
    stage_names: list[str],
    overlay_paths: dict[str, Path],
    depth_paths: dict[str, Path],
) -> None:
    cell_width = 300

    image_rows: list[
        list[Image.Image]
    ] = []

    for path_dictionary in (
        overlay_paths,
        depth_paths,
    ):
        image_row: list[
            Image.Image
        ] = []

        for stage_name in stage_names:
            source_image = Image.open(
                path_dictionary[
                    stage_name
                ]
            ).convert("RGB")

            scale = (
                cell_width
                / source_image.width
            )

            resized_image = (
                source_image.resize(
                    (
                        cell_width,
                        max(
                            1,
                            int(
                                round(
                                    source_image.height
                                    * scale
                                )
                            ),
                        ),
                    )
                )
            )

            image_row.append(
                resized_image
            )

        image_rows.append(
            image_row
        )

    row_heights = [
        max(
            image.height
            for image in image_row
        )
        for image_row in image_rows
    ]

    contact_sheet = Image.new(
        "RGB",
        (
            cell_width
            * len(stage_names),
            sum(row_heights),
        ),
        color=(20, 20, 20),
    )

    offset_y = 0

    for (
        image_row,
        row_height,
    ) in zip(
        image_rows,
        row_heights,
    ):
        for (
            column_index,
            image,
        ) in enumerate(
            image_row
        ):
            contact_sheet.paste(
                image,
                (
                    column_index
                    * cell_width,
                    offset_y,
                ),
            )

        offset_y += row_height

    contact_sheet.save(
        output_path
    )


def visualize_refinement_stages(
    *,
    output_directory: Path,
    rgb: np.ndarray,
    observed_mask: np.ndarray,
    observed_depth_m: np.ndarray,
    camera_k: np.ndarray,
    triangles: np.ndarray,
    stages: dict[str, np.ndarray],
) -> dict[str, Any]:
    """
    M0~M5를 실제 RGB, SAM mask, depth와 비교한다.

    생성 파일:
        *_overlay.png
        *_depth_residual.png
        observation_stage_metrics.json
        observation_stage_metrics.csv
        observation_stage_contact_sheet.png
    """
    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    rgb = np.asarray(
        rgb,
        dtype=np.uint8,
    )

    observed_mask = np.asarray(
        observed_mask,
        dtype=bool,
    )

    observed_depth_m = np.asarray(
        observed_depth_m,
        dtype=np.float64,
    )

    image_height, image_width = (
        observed_mask.shape
    )

    if not stages:
        raise ValueError(
            "No refinement stages provided"
        )

    rendered_results: dict[
        str,
        tuple[
            np.ndarray,
            np.ndarray,
        ],
    ] = {}

    for (
        stage_name,
        stage_points,
    ) in stages.items():
        rendered_results[
            stage_name
        ] = _render_mesh_depth(
            points_camera=stage_points,
            triangles=triangles,
            camera_k=camera_k,
            image_height=image_height,
            image_width=image_width,
        )

    stage_names = list(
        stages.keys()
    )

    original_stage_name = (
        "M0_original_camera"
        if (
            "M0_original_camera"
            in rendered_results
        )
        else stage_names[0]
    )

    original_rendered_mask = (
        rendered_results[
            original_stage_name
        ][0]
    )

    metrics: dict[
        str,
        dict[str, Any],
    ] = {}

    overlay_paths: dict[
        str,
        Path,
    ] = {}

    depth_paths: dict[
        str,
        Path,
    ] = {}

    for stage_name in stage_names:
        (
            rendered_mask,
            rendered_depth_m,
        ) = rendered_results[
            stage_name
        ]

        metrics[
            stage_name
        ] = _calculate_stage_metrics(
            observed_mask=observed_mask,
            observed_depth_m=(
                observed_depth_m
            ),
            rendered_mask=rendered_mask,
            rendered_depth_m=(
                rendered_depth_m
            ),
            boundary_exclusion_px=2,
        )

        overlay_path = (
            output_directory
            / (
                f"{stage_name}"
                "_overlay.png"
            )
        )

        depth_path = (
            output_directory
            / (
                f"{stage_name}"
                "_depth_residual.png"
            )
        )

        _save_mask_overlay(
            output_path=overlay_path,
            rgb=rgb,
            observed_mask=(
                observed_mask
            ),
            original_rendered_mask=(
                original_rendered_mask
            ),
            current_rendered_mask=(
                rendered_mask
            ),
            stage_name=stage_name,
        )

        _save_depth_residual_image(
            output_path=depth_path,
            observed_mask=(
                observed_mask
            ),
            observed_depth_m=(
                observed_depth_m
            ),
            rendered_mask=(
                rendered_mask
            ),
            rendered_depth_m=(
                rendered_depth_m
            ),
            stage_name=stage_name,
            display_limit_m=0.020,
        )

        overlay_paths[
            stage_name
        ] = overlay_path

        depth_paths[
            stage_name
        ] = depth_path

    metrics_json_path = (
        output_directory
        / "observation_stage_metrics.json"
    )

    metrics_json_path.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics_csv_path = (
        output_directory
        / "observation_stage_metrics.csv"
    )

    csv_fieldnames = [
        "stage"
    ]

    for stage_metrics in metrics.values():
        for metric_name in stage_metrics:
            if (
                metric_name
                not in csv_fieldnames
            ):
                csv_fieldnames.append(
                    metric_name
                )

    with metrics_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                csv_fieldnames
            ),
        )

        writer.writeheader()

        for stage_name in stage_names:
            writer.writerow(
                {
                    "stage": stage_name,
                    **metrics[
                        stage_name
                    ],
                }
            )

    contact_sheet_path = (
        output_directory
        / (
            "observation_stage_"
            "contact_sheet.png"
        )
    )

    _save_contact_sheet(
        output_path=(
            contact_sheet_path
        ),
        stage_names=stage_names,
        overlay_paths=overlay_paths,
        depth_paths=depth_paths,
    )

    return {
        "metrics": metrics,
        "metrics_json_path": str(
            metrics_json_path
        ),
        "metrics_csv_path": str(
            metrics_csv_path
        ),
        "contact_sheet_path": str(
            contact_sheet_path
        ),
    }
