from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from evaluate_foundationpose_topk import (
    boundary,
    evaluate_depth,
    find_selected_indices,
    load_camera_k,
    load_json,
    load_mesh,
    load_rgb,
    locate_rgb,
    mask_iou,
    pose_delta,
    transform_points,
)


def build_rays(
    height: int,
    width: int,
    camera_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    directions = np.stack(
        (
            (
                u - float(camera_k[0, 2])
            )
            / float(camera_k[0, 0]),
            (
                v - float(camera_k[1, 2])
            )
            / float(camera_k[1, 1]),
            np.ones_like(u),
        ),
        axis=-1,
    ).astype(np.float32)

    origins = np.zeros_like(directions)

    rays = np.concatenate(
        (origins, directions),
        axis=-1,
    )

    return rays, directions


def raycast_with_ids(
    vertices_camera: np.ndarray,
    triangles: np.ndarray,
    camera_k: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    legacy_mesh = o3d.geometry.TriangleMesh()

    legacy_mesh.vertices = (
        o3d.utility.Vector3dVector(
            vertices_camera
        )
    )

    legacy_mesh.triangles = (
        o3d.utility.Vector3iVector(
            triangles.astype(np.int32)
        )
    )

    legacy_mesh.compute_triangle_normals()

    tensor_mesh = (
        o3d.t.geometry.TriangleMesh
        .from_legacy(legacy_mesh)
    )

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)

    height, width = image_shape

    rays, ray_directions = build_rays(
        height,
        width,
        camera_k,
    )

    hit = scene.cast_rays(
        o3d.core.Tensor(
            rays,
            dtype=o3d.core.Dtype.Float32,
        )
    )

    t_hit = (
        hit["t_hit"]
        .numpy()
        .astype(np.float32)
    )

    normals = (
        hit["primitive_normals"]
        .numpy()
        .astype(np.float32)
    )

    primitive_ids = (
        hit["primitive_ids"]
        .numpy()
        .astype(np.int64)
    )

    rendered_mask = np.isfinite(t_hit)

    rendered_depth_m = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    rendered_depth_m[rendered_mask] = (
        t_hit[rendered_mask]
    )

    normals[~rendered_mask] = 0.0

    ray_norm = np.linalg.norm(
        ray_directions,
        axis=-1,
        keepdims=True,
    )

    ray_unit = (
        ray_directions
        / np.maximum(ray_norm, 1e-12)
    )

    # Face winding에 관계없이 visible normal을
    # 카메라 방향으로 통일
    flip = (
        np.sum(
            normals * ray_unit,
            axis=-1,
        )
        > 0.0
    )

    normals[flip] *= -1.0

    return (
        rendered_mask,
        rendered_depth_m,
        normals,
        primitive_ids,
        ray_unit,
    )


def alpha_overlay(
    rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    rendered_mask: np.ndarray,
    observed_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    output = rgb.copy()

    output[rendered_mask] = (
        (
            1.0 - alpha
        )
        * rgb[
            rendered_mask
        ].astype(np.float32)
        + alpha
        * rendered_rgb[
            rendered_mask
        ].astype(np.float32)
    ).clip(
        0,
        255,
    ).astype(np.uint8)

    # 초록: 관측 boundary
    output[
        boundary(observed_mask)
    ] = np.array(
        [0, 255, 0],
        dtype=np.uint8,
    )

    # 주황: 렌더 boundary
    output[
        boundary(rendered_mask)
    ] = np.array(
        [255, 96, 32],
        dtype=np.uint8,
    )

    return output


def shaded_render(
    normals: np.ndarray,
    ray_unit: np.ndarray,
    rendered_mask: np.ndarray,
) -> np.ndarray:
    view_direction = -ray_unit

    diffuse = np.clip(
        np.sum(
            normals * view_direction,
            axis=-1,
        ),
        0.0,
        1.0,
    )

    intensity = (
        0.22
        + 0.78 * diffuse
    )

    base_rgb = np.array(
        [244.0, 182.0, 52.0],
        dtype=np.float32,
    )

    image = (
        base_rgb[
            None,
            None,
            :
        ]
        * intensity[..., None]
    ).clip(
        0,
        255,
    ).astype(np.uint8)

    image[~rendered_mask] = 0

    return image


def normal_render(
    normals: np.ndarray,
    rendered_mask: np.ndarray,
) -> np.ndarray:
    image = (
        (normals + 1.0)
        * 127.5
    ).clip(
        0,
        255,
    ).astype(np.uint8)

    image[~rendered_mask] = 0

    return image


def canonical_triangle_colors(
    vertices_proxy: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """
    Proxy local 좌표를 RGB로 변환한다.

    R = local X
    G = local Y
    B = local Z

    같은 mesh 부위는 모든 hypothesis에서
    항상 같은 색을 가진다.
    """

    centroids = (
        vertices_proxy[
            triangles
        ].mean(axis=1)
    )

    low = np.quantile(
        centroids,
        0.02,
        axis=0,
    )

    high = np.quantile(
        centroids,
        0.98,
        axis=0,
    )

    span = np.maximum(
        high - low,
        1e-9,
    )

    normalized = np.clip(
        (centroids - low) / span,
        0.0,
        1.0,
    )

    return (
        normalized * 255.0
    ).round().astype(np.uint8)


def canonical_render(
    primitive_ids: np.ndarray,
    rendered_mask: np.ndarray,
    triangle_colors: np.ndarray,
) -> np.ndarray:
    image = np.zeros(
        (
            *rendered_mask.shape,
            3,
        ),
        dtype=np.uint8,
    )

    hit_ids = primitive_ids[
        rendered_mask
    ]

    valid = (
        (hit_ids >= 0)
        & (
            hit_ids
            < len(triangle_colors)
        )
    )

    colors = np.zeros(
        (
            len(hit_ids),
            3,
        ),
        dtype=np.uint8,
    )

    colors[valid] = triangle_colors[
        hit_ids[valid]
    ]

    image[rendered_mask] = colors

    return image


def depth_residual_overlay(
    rgb: np.ndarray,
    observed_mask: np.ndarray,
    observed_depth_m: np.ndarray,
    rendered_mask: np.ndarray,
    rendered_depth_m: np.ndarray,
    clip_mm: float,
) -> np.ndarray:
    overlap = (
        observed_mask
        & rendered_mask
        & np.isfinite(
            observed_depth_m
        )
        & (observed_depth_m > 0.0)
        & np.isfinite(
            rendered_depth_m
        )
        & (rendered_depth_m > 0.0)
    )

    output = rgb.copy()

    if not np.any(overlap):
        return output

    residual_mm = (
        rendered_depth_m
        - observed_depth_m
    ) * 1000.0

    normalized = np.clip(
        (
            residual_mm
            + clip_mm
        )
        / (2.0 * clip_mm),
        0.0,
        1.0,
    )

    values = (
        normalized * 255.0
    ).astype(np.uint8)

    color_bgr = cv2.applyColorMap(
        values,
        cv2.COLORMAP_TURBO,
    )

    color_rgb = cv2.cvtColor(
        color_bgr,
        cv2.COLOR_BGR2RGB,
    )

    output[overlap] = (
        0.20
        * rgb[
            overlap
        ].astype(np.float32)
        + 0.80
        * color_rgb[
            overlap
        ].astype(np.float32)
    ).clip(
        0,
        255,
    ).astype(np.uint8)

    output[
        boundary(observed_mask)
    ] = np.array(
        [0, 255, 0],
        dtype=np.uint8,
    )

    output[
        boundary(rendered_mask)
    ] = np.array(
        [255, 96, 32],
        dtype=np.uint8,
    )

    return output


def project_points(
    points_camera: np.ndarray,
    camera_k: np.ndarray,
) -> np.ndarray:
    z = points_camera[:, 2]

    output = np.full(
        (
            len(points_camera),
            2,
        ),
        np.nan,
        dtype=np.float64,
    )

    valid = z > 1e-8

    if not np.any(valid):
        return output

    projected = (
        points_camera[valid]
        @ camera_k.T
    )

    output[valid, 0] = (
        projected[:, 0]
        / projected[:, 2]
    )

    output[valid, 1] = (
        projected[:, 1]
        / projected[:, 2]
    )

    return output


def draw_proxy_axes(
    image: np.ndarray,
    vertices_proxy: np.ndarray,
    pose: np.ndarray,
    camera_k: np.ndarray,
) -> np.ndarray:
    output = image.copy()

    # mesh centroid에 proxy local 축 표시
    center = np.mean(
        vertices_proxy,
        axis=0,
    )

    extent = np.ptp(
        vertices_proxy,
        axis=0,
    )

    axis_length = (
        0.32
        * float(np.max(extent))
    )

    points_proxy = np.stack(
        (
            center,
            center
            + np.array(
                [
                    axis_length,
                    0.0,
                    0.0,
                ]
            ),
            center
            + np.array(
                [
                    0.0,
                    axis_length,
                    0.0,
                ]
            ),
            center
            + np.array(
                [
                    0.0,
                    0.0,
                    axis_length,
                ]
            ),
        ),
        axis=0,
    )

    points_camera = transform_points(
        points_proxy,
        pose,
    )

    pixels = project_points(
        points_camera,
        camera_k,
    )

    if not np.all(
        np.isfinite(pixels[0])
    ):
        return output

    origin = tuple(
        np.round(
            pixels[0]
        ).astype(int)
    )

    height, width = output.shape[:2]

    if not (
        0 <= origin[0] < width
        and 0 <= origin[1] < height
    ):
        return output

    # RGB array 기준
    colors = (
        (255, 0, 0),    # X
        (0, 255, 0),    # Y
        (0, 128, 255),  # Z
    )

    labels = (
        "X",
        "Y",
        "Z",
    )

    cv2.circle(
        output,
        origin,
        4,
        (255, 255, 255),
        -1,
    )

    for index in range(3):
        endpoint_pixel = (
            pixels[index + 1]
        )

        if not np.all(
            np.isfinite(
                endpoint_pixel
            )
        ):
            continue

        endpoint = tuple(
            np.round(
                endpoint_pixel
            ).astype(int)
        )

        cv2.arrowedLine(
            output,
            origin,
            endpoint,
            colors[index],
            2,
            cv2.LINE_AA,
            tipLength=0.18,
        )

        cv2.putText(
            output,
            labels[index],
            endpoint,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            colors[index],
            1,
            cv2.LINE_AA,
        )

    return output


def crop_box_from_mask(
    mask: np.ndarray,
    margin: int,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)

    height, width = image_shape

    if len(xs) == 0:
        return (
            0,
            0,
            width,
            height,
        )

    x0 = max(
        int(xs.min()) - margin,
        0,
    )

    y0 = max(
        int(ys.min()) - margin,
        0,
    )

    x1 = min(
        int(xs.max()) + margin + 1,
        width,
    )

    y1 = min(
        int(ys.max()) + margin + 1,
        height,
    )

    return (
        x0,
        y0,
        x1,
        y1,
    )


def make_panel(
    image: np.ndarray,
    crop_box: tuple[int, int, int, int],
    title_lines: list[str],
    selected: bool,
    panel_width: int = 390,
    panel_height: int = 360,
) -> np.ndarray:
    x0, y0, x1, y1 = crop_box

    crop = image[
        y0:y1,
        x0:x1,
    ]

    header_height = 112
    body_height = (
        panel_height
        - header_height
    )

    scale = min(
        panel_width
        / max(
            crop.shape[1],
            1,
        ),
        body_height
        / max(
            crop.shape[0],
            1,
        ),
    )

    resized_width = max(
        1,
        int(
            round(
                crop.shape[1]
                * scale
            )
        ),
    )

    resized_height = max(
        1,
        int(
            round(
                crop.shape[0]
                * scale
            )
        ),
    )

    resized = cv2.resize(
        crop,
        (
            resized_width,
            resized_height,
        ),
        interpolation=(
            cv2.INTER_NEAREST
        ),
    )

    panel = np.zeros(
        (
            panel_height,
            panel_width,
            3,
        ),
        dtype=np.uint8,
    )

    body_x = (
        panel_width
        - resized_width
    ) // 2

    body_y = (
        header_height
        + (
            body_height
            - resized_height
        )
        // 2
    )

    panel[
        body_y:
        body_y + resized_height,
        body_x:
        body_x + resized_width,
    ] = resized

    for index, line in enumerate(
        title_lines
    ):
        cv2.putText(
            panel,
            line,
            (
                8,
                22
                + 22 * index,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if selected:
        cv2.rectangle(
            panel,
            (2, 2),
            (
                panel_width - 3,
                panel_height - 3,
            ),
            (0, 255, 255),
            3,
        )

    return panel


def save_rgb(
    path: Path,
    image: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(path),
        cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        ),
    )

    if not success:
        raise RuntimeError(
            f"이미지 저장 실패: {path}"
        )


def fmt(
    value: Any,
    digits: int = 2,
) -> str:
    if value is None:
        return "NA"

    return (
        f"{float(value):.{digits}f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--candidate-index",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--depth-clip-mm",
        type=float,
        default=20.0,
    )

    args = parser.parse_args()

    run = (
        args.run
        .expanduser()
        .resolve()
    )

    config = load_json(
        run / "pipeline_config.json"
    )

    selection = load_json(
        run
        / "visible_scale_refinement"
        / "joint_shared"
        / "selection.json"
    )

    (
        auto_candidate,
        selected_rank,
    ) = find_selected_indices(
        selection
    )

    candidate_index = (
        auto_candidate
        if args.candidate_index
        is None
        else args.candidate_index
    )

    fp_path = (
        run
        / "foundationpose"
        / "self_joint_shared_scale"
        / "query"
        / (
            f"candidate_"
            f"{candidate_index:02d}"
        )
        / "foundationpose_result.json"
    )

    fp_data = load_json(fp_path)

    hypothesis_by_rank = {
        int(item["rank"]): item
        for item
        in fp_data["hypotheses"]
    }

    missing = [
        rank
        for rank in args.ranks
        if rank
        not in hypothesis_by_rank
    ]

    if missing:
        raise KeyError(
            "저장된 hypothesis에 없는 "
            f"rank: {missing}"
        )

    selected_pose = np.asarray(
        hypothesis_by_rank[
            selected_rank
        ]["pose_cam_from_proxy"],
        dtype=np.float64,
    )

    mesh_path = Path(
        fp_data["scaled_mesh_path"]
    ).expanduser().resolve()

    (
        vertices_proxy,
        triangles,
    ) = load_mesh(mesh_path)

    triangle_colors = (
        canonical_triangle_colors(
            vertices_proxy,
            triangles,
        )
    )

    dataset_root = Path(
        config["dataset_root"]
    ).expanduser().resolve()

    split = str(config["split"])

    scene_id = int(
        config["query"]["scene_id"]
    )

    image_id = int(
        config["query"]["image_id"]
    )

    scene_dir = (
        dataset_root
        / split
        / f"{scene_id:06d}"
    )

    rgb_path = locate_rgb(
        scene_dir,
        image_id,
    )

    rgb = load_rgb(rgb_path)

    camera_k = load_camera_k(
        scene_dir
        / "scene_camera.json",
        image_id,
    )

    observed_mask = np.load(
        run
        / "views/query/segmentation/"
        "mask_bool.npy",
        allow_pickle=False,
    ).astype(bool)

    observed_depth_m = np.load(
        run
        / "views/query/prepared/"
        "masked_depth_m.npy",
        allow_pickle=False,
    ).astype(np.float32)

    output_dir = (
        run
        / "diagnostics"
        / "foundationpose_branch_comparison"
        / (
            f"candidate_"
            f"{candidate_index:02d}"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rendered: dict[
        int,
        dict[str, Any],
    ] = {}

    crop_union = (
        observed_mask.copy()
    )

    for rank in args.ranks:
        item = hypothesis_by_rank[
            rank
        ]

        pose = np.asarray(
            item[
                "pose_cam_from_proxy"
            ],
            dtype=np.float64,
        )

        vertices_camera = (
            transform_points(
                vertices_proxy,
                pose,
            )
        )

        (
            rendered_mask,
            rendered_depth_m,
            normals,
            primitive_ids,
            ray_unit,
        ) = raycast_with_ids(
            vertices_camera,
            triangles,
            camera_k,
            rgb.shape[:2],
        )

        depth_metrics = (
            evaluate_depth(
                observed_mask,
                observed_depth_m,
                rendered_mask,
                rendered_depth_m,
            )
        )

        (
            rotation_delta_deg,
            translation_delta_mm,
        ) = pose_delta(
            pose,
            selected_pose,
        )

        shaded = alpha_overlay(
            rgb,
            shaded_render(
                normals,
                ray_unit,
                rendered_mask,
            ),
            rendered_mask,
            observed_mask,
            0.62,
        )

        normal_overlay = alpha_overlay(
            rgb,
            normal_render(
                normals,
                rendered_mask,
            ),
            rendered_mask,
            observed_mask,
            0.72,
        )

        canonical_overlay = (
            alpha_overlay(
                rgb,
                canonical_render(
                    primitive_ids,
                    rendered_mask,
                    triangle_colors,
                ),
                rendered_mask,
                observed_mask,
                0.76,
            )
        )

        residual_overlay = (
            depth_residual_overlay(
                rgb,
                observed_mask,
                observed_depth_m,
                rendered_mask,
                rendered_depth_m,
                args.depth_clip_mm,
            )
        )

        shaded = draw_proxy_axes(
            shaded,
            vertices_proxy,
            pose,
            camera_k,
        )

        canonical_overlay = (
            draw_proxy_axes(
                canonical_overlay,
                vertices_proxy,
                pose,
                camera_k,
            )
        )

        rendered[rank] = {
            "rank": rank,
            "selected": (
                rank == selected_rank
            ),
            "score": float(
                item["score"]
            ),
            "pose": pose,
            "mask_iou": mask_iou(
                observed_mask,
                rendered_mask,
            ),
            "rotation_delta_deg": (
                rotation_delta_deg
            ),
            "translation_delta_mm": (
                translation_delta_mm
            ),
            "depth_metrics": (
                depth_metrics
            ),
            "rendered_mask": (
                rendered_mask
            ),
            "shaded": shaded,
            "normal": normal_overlay,
            "canonical": (
                canonical_overlay
            ),
            "residual": (
                residual_overlay
            ),
        }

        crop_union |= rendered_mask

    crop_box = crop_box_from_mask(
        crop_union,
        margin=26,
        image_shape=rgb.shape[:2],
    )

    row_specs = (
        (
            "shaded",
            "SHADED + AXES",
            (
                "X=red Y=green "
                "Z=blue"
            ),
        ),
        (
            "normal",
            "CAMERA NORMAL RGB",
            (
                "surface direction "
                "/ rotation branch"
            ),
        ),
        (
            "canonical",
            "PROXY XYZ COLOR + AXES",
            (
                "same local region "
                "= same color"
            ),
        ),
        (
            "residual",
            "SIGNED DEPTH RESIDUAL",
            (
                "blue=near red=far "
                f"clip=+-"
                f"{args.depth_clip_mm:.0f}mm"
            ),
        ),
    )

    row_images: list[
        np.ndarray
    ] = []

    for (
        image_key,
        row_name,
        row_note,
    ) in row_specs:
        panels: list[
            np.ndarray
        ] = []

        for rank in args.ranks:
            record = rendered[rank]
            depth = record[
                "depth_metrics"
            ]

            title_lines = [
                (
                    f"{row_name} "
                    f"| rank={rank:02d}"
                    + (
                        " SELECTED"
                        if record["selected"]
                        else ""
                    )
                ),
                (
                    f"score="
                    f"{record['score']:.6f} "
                    f"IoU="
                    f"{record['mask_iou']:.4f}"
                ),
                (
                    "abs/debias="
                    f"{fmt(depth['abs_median_mm'])}"
                    "/"
                    f"{fmt(depth['debiased_median_mm'])}"
                    " mm  "
                    f"dR="
                    f"{record['rotation_delta_deg']:.2f}"
                    " deg"
                ),
                row_note,
            ]

            panels.append(
                make_panel(
                    record[image_key],
                    crop_box,
                    title_lines,
                    record["selected"],
                )
            )

        row_images.append(
            np.concatenate(
                panels,
                axis=1,
            )
        )

    sheet = np.concatenate(
        row_images,
        axis=0,
    )

    sheet_path = (
        output_dir
        / "branch_comparison.png"
    )

    save_rgb(
        sheet_path,
        sheet,
    )

    metrics = {
        "run": str(run),
        "candidate_index": (
            candidate_index
        ),
        "selected_rank": (
            selected_rank
        ),
        "compared_ranks": (
            args.ranks
        ),
        "mesh_path": (
            str(mesh_path)
        ),
        "rgb_path": (
            str(rgb_path)
        ),
        "depth_clip_mm": (
            args.depth_clip_mm
        ),
        "records": {
            str(rank): {
                "selected": (
                    rendered[rank][
                        "selected"
                    ]
                ),
                "foundationpose_score": (
                    rendered[rank][
                        "score"
                    ]
                ),
                "mask_iou": (
                    rendered[rank][
                        "mask_iou"
                    ]
                ),
                "rotation_delta_from_selected_deg": (
                    rendered[rank][
                        "rotation_delta_deg"
                    ]
                ),
                "translation_delta_from_selected_mm": (
                    rendered[rank][
                        "translation_delta_mm"
                    ]
                ),
                **rendered[rank][
                    "depth_metrics"
                ],
                "pose_cam_from_proxy": (
                    rendered[rank][
                        "pose"
                    ].tolist()
                ),
            }
            for rank in args.ranks
        },
    }

    metrics_path = (
        output_dir
        / "branch_comparison.json"
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "=== Branch comparison ==="
    )

    for rank in args.ranks:
        record = rendered[rank]
        depth = record[
            "depth_metrics"
        ]

        selected_text = (
            " SELECTED"
            if record["selected"]
            else ""
        )

        print(
            f"rank={rank:02d}"
            f"{selected_text:9s} "
            f"score="
            f"{record['score']:.6f} "
            f"IoU="
            f"{record['mask_iou']:.4f} "
            f"abs="
            f"{fmt(depth['abs_median_mm'])}"
            "mm "
            f"debiased="
            f"{fmt(depth['debiased_median_mm'])}"
            "mm "
            f"dR="
            f"{record['rotation_delta_deg']:.2f}"
            "deg"
        )

    print()
    print("saved:", sheet_path)
    print("saved:", metrics_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
