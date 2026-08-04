from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON 파일이 없습니다: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"JSON root가 object가 아닙니다: {path}")

    return data


def locate_rgb(
    scene_dir: Path,
    image_id: int,
) -> Path:
    stem = f"{image_id:06d}"

    candidates = (
        scene_dir / "rgb" / f"{stem}.png",
        scene_dir / "rgb" / f"{stem}.jpg",
        scene_dir / "rgb" / f"{stem}.jpeg",
        scene_dir / "rgb" / f"{stem}.tif",
        scene_dir / "rgb" / f"{stem}.tiff",
    )

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        "Query RGB를 찾지 못했습니다:\n"
        + "\n".join(str(path) for path in candidates)
    )


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if bgr is None:
        raise RuntimeError(f"RGB를 읽지 못했습니다: {path}")

    return cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )


def load_camera_k(
    scene_camera_path: Path,
    image_id: int,
) -> np.ndarray:
    data = load_json(scene_camera_path)

    record = None

    for key in (
        str(image_id),
        f"{image_id:06d}",
    ):
        if key in data:
            record = data[key]
            break

    if not isinstance(record, dict):
        raise KeyError(
            f"scene_camera에 image_id={image_id}가 없습니다: "
            f"{scene_camera_path}"
        )

    camera_k = np.asarray(
        record["cam_K"],
        dtype=np.float64,
    ).reshape(3, 3)

    return camera_k


def find_selected_indices(
    selection: dict[str, Any],
) -> tuple[int, int]:
    candidate_index = selection.get(
        "selected_candidate_index"
    )

    selected_record = selection.get(
        "selected_record",
        {},
    )

    query_rank = None

    if isinstance(selected_record, dict):
        query_rank = selected_record.get(
            "query_hypothesis_rank"
        )

    # 저장 형식이 바뀐 경우를 위한 재귀 검색
    def recursive_find(
        value: Any,
        key_names: set[str],
    ) -> Any:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in key_names:
                    return item

            for item in value.values():
                found = recursive_find(
                    item,
                    key_names,
                )

                if found is not None:
                    return found

        elif isinstance(value, list):
            for item in value:
                found = recursive_find(
                    item,
                    key_names,
                )

                if found is not None:
                    return found

        return None

    if candidate_index is None:
        candidate_index = recursive_find(
            selection,
            {
                "selected_candidate_index",
                "candidate_index",
            },
        )

    if query_rank is None:
        query_rank = recursive_find(
            selection,
            {
                "query_hypothesis_rank",
                "selected_query_hypothesis_rank",
            },
        )

    if candidate_index is None:
        raise KeyError(
            "selection.json에서 selected candidate를 "
            "찾지 못했습니다."
        )

    if query_rank is None:
        print(
            "[경고] selected query rank를 찾지 못해 "
            "rank=0을 사용합니다."
        )
        query_rank = 0

    return int(candidate_index), int(query_rank)


def load_mesh(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(
        str(path),
        enable_post_processing=True,
    )

    if (
        mesh.is_empty()
        or len(mesh.vertices) == 0
        or len(mesh.triangles) == 0
    ):
        raise ValueError(f"Mesh가 비어 있습니다: {path}")

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64,
    )

    triangles = np.asarray(
        mesh.triangles,
        dtype=np.int64,
    )

    return vertices, triangles


def transform_points(
    points: np.ndarray,
    pose: np.ndarray,
) -> np.ndarray:
    return (
        points @ pose[:3, :3].T
        + pose[:3, 3]
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


def raycast_mesh(
    vertices_camera: np.ndarray,
    triangles: np.ndarray,
    camera_k: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        o3d.t.geometry.TriangleMesh.from_legacy(
            legacy_mesh
        )
    )

    scene = o3d.t.geometry.RaycastingScene()

    scene.add_triangles(tensor_mesh)

    height, width = image_shape

    rays, ray_directions = build_rays(
        height,
        width,
        camera_k,
    )

    result = scene.cast_rays(
        o3d.core.Tensor(
            rays,
            dtype=o3d.core.Dtype.Float32,
        )
    )

    t_hit = (
        result["t_hit"]
        .numpy()
        .astype(np.float32)
    )

    normals = (
        result["primitive_normals"]
        .numpy()
        .astype(np.float32)
    )

    rendered_mask = np.isfinite(t_hit)

    rendered_depth_m = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    # ray direction의 z가 1이므로
    # t_hit가 camera z-depth와 같다.
    rendered_depth_m[rendered_mask] = (
        t_hit[rendered_mask]
    )

    normals[~rendered_mask] = 0.0

    direction_norm = np.linalg.norm(
        ray_directions,
        axis=-1,
        keepdims=True,
    )

    ray_unit = (
        ray_directions
        / np.maximum(direction_norm, 1e-12)
    )

    # Face winding과 관계없이 visible normal을
    # camera 방향으로 통일한다.
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
    )


def mask_iou(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    intersection = int(
        np.count_nonzero(first & second)
    )

    union = int(
        np.count_nonzero(first | second)
    )

    return (
        1.0
        if union == 0
        else intersection / union
    )


def evaluate_depth(
    observed_mask: np.ndarray,
    observed_depth_m: np.ndarray,
    rendered_mask: np.ndarray,
    rendered_depth_m: np.ndarray,
) -> dict[str, float | int | None]:
    valid_observed = (
        observed_mask
        & np.isfinite(observed_depth_m)
        & (observed_depth_m > 0.0)
    )

    overlap = (
        valid_observed
        & rendered_mask
        & np.isfinite(rendered_depth_m)
        & (rendered_depth_m > 0.0)
    )

    overlap_count = int(
        np.count_nonzero(overlap)
    )

    observed_count = int(
        np.count_nonzero(valid_observed)
    )

    if overlap_count == 0:
        return {
            "overlap_count": 0,
            "depth_coverage": 0.0,
            "signed_median_mm": None,
            "abs_median_mm": None,
            "debiased_median_mm": None,
            "p90_abs_mm": None,
        }

    residual_m = (
        rendered_depth_m[overlap]
        - observed_depth_m[overlap]
    ).astype(np.float64)

    signed_median_m = float(
        np.median(residual_m)
    )

    return {
        "overlap_count": overlap_count,
        "depth_coverage": (
            overlap_count
            / max(observed_count, 1)
        ),
        "signed_median_mm": (
            signed_median_m * 1000.0
        ),
        "abs_median_mm": float(
            np.median(np.abs(residual_m))
            * 1000.0
        ),
        "debiased_median_mm": float(
            np.median(
                np.abs(
                    residual_m
                    - signed_median_m
                )
            )
            * 1000.0
        ),
        "p90_abs_mm": float(
            np.quantile(
                np.abs(residual_m),
                0.90,
            )
            * 1000.0
        ),
    }


def rotation_angle_deg(
    rotation: np.ndarray,
) -> float:
    cosine = float(
        np.clip(
            (
                np.trace(rotation) - 1.0
            )
            / 2.0,
            -1.0,
            1.0,
        )
    )

    return math.degrees(
        math.acos(cosine)
    )


def pose_delta(
    pose: np.ndarray,
    selected_pose: np.ndarray,
) -> tuple[float, float]:
    rotation_delta = (
        pose[:3, :3]
        @ selected_pose[:3, :3].T
    )

    translation_delta = (
        pose[:3, 3]
        - selected_pose[:3, 3]
    )

    return (
        rotation_angle_deg(rotation_delta),
        float(
            np.linalg.norm(translation_delta)
            * 1000.0
        ),
    )


def boundary(
    mask: np.ndarray,
) -> np.ndarray:
    eroded = cv2.erode(
        mask.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    return mask & (eroded == 0)


def build_normal_overlay(
    rgb: np.ndarray,
    normals: np.ndarray,
    rendered_mask: np.ndarray,
    observed_mask: np.ndarray,
    alpha: float = 0.72,
) -> np.ndarray:
    normal_rgb = np.clip(
        (normals + 1.0) * 127.5,
        0,
        255,
    ).astype(np.uint8)

    overlay = rgb.copy()

    overlay[rendered_mask] = (
        (
            1.0 - alpha
        )
        * rgb[rendered_mask].astype(
            np.float32
        )
        + alpha
        * normal_rgb[rendered_mask].astype(
            np.float32
        )
    ).clip(
        0,
        255,
    ).astype(np.uint8)

    overlay[boundary(observed_mask)] = (
        0,
        255,
        0,
    )

    overlay[boundary(rendered_mask)] = (
        255,
        96,
        32,
    )

    return overlay


def crop_object(
    image: np.ndarray,
    observed_mask: np.ndarray,
    rendered_mask: np.ndarray,
    margin: int = 24,
) -> np.ndarray:
    union = observed_mask | rendered_mask

    ys, xs = np.where(union)

    if len(xs) == 0:
        return image

    x0 = max(int(xs.min()) - margin, 0)
    y0 = max(int(ys.min()) - margin, 0)

    x1 = min(
        int(xs.max()) + margin + 1,
        image.shape[1],
    )

    y1 = min(
        int(ys.max()) + margin + 1,
        image.shape[0],
    )

    return image[y0:y1, x0:x1]


def fit_panel(
    image: np.ndarray,
    width: int = 360,
    height: int = 330,
) -> np.ndarray:
    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    maximum_image_height = height - 100

    scale = min(
        width / max(image.shape[1], 1),
        maximum_image_height
        / max(image.shape[0], 1),
    )

    resized_width = max(
        1,
        int(round(image.shape[1] * scale)),
    )

    resized_height = max(
        1,
        int(round(image.shape[0] * scale)),
    )

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )

    x0 = (width - resized_width) // 2
    y0 = 96

    canvas[
        y0:y0 + resized_height,
        x0:x0 + resized_width,
    ] = resized

    return canvas


def draw_panel_text(
    panel: np.ndarray,
    lines: list[str],
    selected: bool,
) -> np.ndarray:
    result = panel.copy()

    if selected:
        cv2.rectangle(
            result,
            (1, 1),
            (
                result.shape[1] - 2,
                result.shape[0] - 2,
            ),
            (0, 255, 255),
            3,
        )

    for index, line in enumerate(lines):
        cv2.putText(
            result,
            line,
            (
                8,
                22 + 22 * index,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return result


def make_contact_sheet(
    panels: list[np.ndarray],
    columns: int,
) -> np.ndarray:
    if not panels:
        raise ValueError("Contact sheet panel이 없습니다.")

    panel_height = max(
        panel.shape[0]
        for panel in panels
    )

    panel_width = max(
        panel.shape[1]
        for panel in panels
    )

    rows = math.ceil(
        len(panels) / columns
    )

    sheet = np.zeros(
        (
            rows * panel_height,
            columns * panel_width,
            3,
        ),
        dtype=np.uint8,
    )

    for index, panel in enumerate(panels):
        row = index // columns
        column = index % columns

        y0 = row * panel_height
        x0 = column * panel_width

        sheet[
            y0:y0 + panel.shape[0],
            x0:x0 + panel.shape[1],
        ] = panel

    return sheet


def save_rgb(
    path: Path,
    image_rgb: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(path),
        cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR,
        ),
    )

    if not success:
        raise RuntimeError(
            f"이미지 저장 실패: {path}"
        )


def optional_number(
    value: Any,
) -> str:
    if value is None:
        return "NA"

    return f"{float(value):.2f}"


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--candidate-index",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--selected-rank",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--columns",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    run = (
        args.run
        .expanduser()
        .resolve()
    )

    config_path = (
        run / "pipeline_config.json"
    )

    selection_path = (
        run
        / "visible_scale_refinement"
        / "joint_shared"
        / "selection.json"
    )

    config = load_json(config_path)

    if (
        args.candidate_index is None
        or args.selected_rank is None
    ):
        selection = load_json(selection_path)

        auto_candidate, auto_rank = (
            find_selected_indices(selection)
        )
    else:
        auto_candidate = args.candidate_index
        auto_rank = args.selected_rank

    candidate_index = (
        args.candidate_index
        if args.candidate_index is not None
        else auto_candidate
    )

    selected_rank = (
        args.selected_rank
        if args.selected_rank is not None
        else auto_rank
    )

    candidate_dir = (
        run
        / "foundationpose"
        / "self_joint_shared_scale"
        / "query"
        / f"candidate_{candidate_index:02d}"
    )

    fp_json_path = (
        candidate_dir
        / "foundationpose_result.json"
    )

    fp_data = load_json(fp_json_path)

    mesh_path = Path(
        fp_data["scaled_mesh_path"]
    ).expanduser().resolve()

    hypotheses = fp_data.get(
        "hypotheses",
        [],
    )

    if not isinstance(hypotheses, list):
        raise ValueError(
            "foundationpose_result.json의 "
            "hypotheses가 list가 아닙니다."
        )

    if not hypotheses:
        raise RuntimeError(
            "저장된 FoundationPose hypothesis가 없습니다."
        )

    selected_items = [
        item
        for item in hypotheses
        if int(item["rank"]) == selected_rank
    ]

    if len(selected_items) != 1:
        raise RuntimeError(
            f"selected rank={selected_rank}를 "
            "정확히 하나 찾지 못했습니다."
        )

    selected_pose = np.asarray(
        selected_items[0][
            "pose_cam_from_proxy"
        ],
        dtype=np.float64,
    )

    dataset_root = Path(
        config["dataset_root"]
    ).expanduser().resolve()

    split = str(config["split"])

    query_config = config["query"]

    scene_id = int(
        query_config["scene_id"]
    )

    image_id = int(
        query_config["image_id"]
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

    scene_camera_path = (
        scene_dir
        / "scene_camera.json"
    )

    rgb = load_rgb(rgb_path)

    camera_k = load_camera_k(
        scene_camera_path,
        image_id,
    )

    mask_path = (
        run
        / "views"
        / "query"
        / "segmentation"
        / "mask_bool.npy"
    )

    depth_path = (
        run
        / "views"
        / "query"
        / "prepared"
        / "masked_depth_m.npy"
    )

    if not mask_path.is_file():
        raise FileNotFoundError(
            f"Query mask가 없습니다: {mask_path}"
        )

    if not depth_path.is_file():
        raise FileNotFoundError(
            f"Query masked depth가 없습니다: {depth_path}"
        )

    observed_mask = np.load(
        mask_path,
        allow_pickle=False,
    ).astype(bool)

    observed_depth_m = np.load(
        depth_path,
        allow_pickle=False,
    ).astype(np.float32)

    if observed_mask.shape != rgb.shape[:2]:
        raise ValueError(
            f"Mask shape 불일치: "
            f"{observed_mask.shape} != {rgb.shape[:2]}"
        )

    if observed_depth_m.shape != rgb.shape[:2]:
        raise ValueError(
            f"Depth shape 불일치: "
            f"{observed_depth_m.shape} != {rgb.shape[:2]}"
        )

    vertices_proxy, triangles = load_mesh(
        mesh_path
    )

    output_dir = (
        run
        / "diagnostics"
        / "foundationpose_topk_query"
        / f"candidate_{candidate_index:02d}"
    )

    overlays_dir = (
        output_dir / "normal_overlays"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overlays_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records: list[
        dict[str, Any]
    ] = []

    panels: list[
        np.ndarray
    ] = []

    print()
    print("=== FoundationPose top-K 검사 ===")
    print("RUN             :", run)
    print("candidate       :", candidate_index)
    print("selected rank   :", selected_rank)
    print("top-K count     :", len(hypotheses))
    print("mesh            :", mesh_path)
    print("RGB             :", rgb_path)
    print("mask            :", mask_path)
    print("depth           :", depth_path)
    print()

    for item in hypotheses:
        rank = int(item["rank"])
        score = float(item["score"])

        pose = np.asarray(
            item["pose_cam_from_proxy"],
            dtype=np.float64,
        )

        if pose.shape != (4, 4):
            raise ValueError(
                f"rank={rank} pose shape 오류: "
                f"{pose.shape}"
            )

        vertices_camera = transform_points(
            vertices_proxy,
            pose,
        )

        (
            rendered_mask,
            rendered_depth_m,
            normals,
        ) = raycast_mesh(
            vertices_camera,
            triangles,
            camera_k,
            rgb.shape[:2],
        )

        iou = mask_iou(
            observed_mask,
            rendered_mask,
        )

        depth_metrics = evaluate_depth(
            observed_mask,
            observed_depth_m,
            rendered_mask,
            rendered_depth_m,
        )

        rotation_delta_deg, translation_delta_mm = (
            pose_delta(
                pose,
                selected_pose,
            )
        )

        record = {
            "rank": rank,
            "selected": (
                rank == selected_rank
            ),
            "foundationpose_score": score,
            "mask_iou": iou,
            **depth_metrics,
            "rotation_delta_from_selected_deg": (
                rotation_delta_deg
            ),
            "translation_delta_from_selected_mm": (
                translation_delta_mm
            ),
            "pose_cam_from_proxy": (
                pose.tolist()
            ),
        }

        records.append(record)

        normal_overlay = (
            build_normal_overlay(
                rgb,
                normals,
                rendered_mask,
                observed_mask,
            )
        )

        crop = crop_object(
            normal_overlay,
            observed_mask,
            rendered_mask,
        )

        panel = fit_panel(crop)

        panel = draw_panel_text(
            panel,
            [
                (
                    f"rank={rank:02d}"
                    + (
                        " SELECTED"
                        if rank == selected_rank
                        else ""
                    )
                ),
                f"FP score={score:.6f}",
                f"IoU={iou:.4f}",
                (
                    "abs/debias="
                    f"{optional_number(depth_metrics['abs_median_mm'])}"
                    "/"
                    f"{optional_number(depth_metrics['debiased_median_mm'])}"
                    " mm"
                ),
                (
                    f"dR={rotation_delta_deg:.2f} deg  "
                    f"dt={translation_delta_mm:.1f} mm"
                ),
            ],
            selected=(
                rank == selected_rank
            ),
        )

        panels.append(panel)

        save_rgb(
            overlays_dir
            / f"rank_{rank:02d}_normal.png",
            panel,
        )

    records_by_rank = sorted(
        records,
        key=lambda record: record["rank"],
    )

    valid_depth_records = [
        record
        for record in records
        if record["debiased_median_mm"]
        is not None
    ]

    best_depth = min(
        valid_depth_records,
        key=lambda record: (
            record["debiased_median_mm"],
            -record["mask_iou"],
        ),
    )

    best_iou = max(
        records,
        key=lambda record: record["mask_iou"],
    )

    selected_record = next(
        record
        for record in records
        if record["selected"]
    )

    summary = {
        "run": str(run),
        "candidate_index": candidate_index,
        "selected_rank": selected_rank,
        "hypothesis_count": len(records),
        "mesh_path": str(mesh_path),
        "rgb_path": str(rgb_path),
        "mask_path": str(mask_path),
        "depth_path": str(depth_path),
        "selected_result": selected_record,
        "best_depth_result": best_depth,
        "best_iou_result": best_iou,
        "selected_is_best_depth": (
            selected_rank == best_depth["rank"]
        ),
        "selected_is_best_iou": (
            selected_rank == best_iou["rank"]
        ),
        "records": records_by_rank,
    }

    json_path = (
        output_dir
        / "foundationpose_topk_metrics.json"
    )

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    csv_path = (
        output_dir
        / "foundationpose_topk_metrics.csv"
    )

    csv_fields = [
        "rank",
        "selected",
        "foundationpose_score",
        "mask_iou",
        "overlap_count",
        "depth_coverage",
        "signed_median_mm",
        "abs_median_mm",
        "debiased_median_mm",
        "p90_abs_mm",
        "rotation_delta_from_selected_deg",
        "translation_delta_from_selected_mm",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=csv_fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        for record in records_by_rank:
            writer.writerow(record)

    contact_sheet = make_contact_sheet(
        panels,
        columns=args.columns,
    )

    contact_sheet_path = (
        output_dir
        / "foundationpose_topk_normal_contact_sheet.png"
    )

    save_rgb(
        contact_sheet_path,
        contact_sheet,
    )

    print("=== 선택 결과 ===")
    print(
        f"selected rank={selected_rank:02d}  "
        f"IoU={selected_record['mask_iou']:.4f}  "
        f"abs={optional_number(selected_record['abs_median_mm'])} mm  "
        f"debiased={optional_number(selected_record['debiased_median_mm'])} mm"
    )

    print()
    print("=== Depth 최선 ===")
    print(
        f"best rank={best_depth['rank']:02d}  "
        f"IoU={best_depth['mask_iou']:.4f}  "
        f"abs={optional_number(best_depth['abs_median_mm'])} mm  "
        f"debiased={optional_number(best_depth['debiased_median_mm'])} mm  "
        f"dR={best_depth['rotation_delta_from_selected_deg']:.2f} deg"
    )

    print()
    print("=== IoU 최선 ===")
    print(
        f"best rank={best_iou['rank']:02d}  "
        f"IoU={best_iou['mask_iou']:.4f}  "
        f"abs={optional_number(best_iou['abs_median_mm'])} mm  "
        f"debiased={optional_number(best_iou['debiased_median_mm'])} mm  "
        f"dR={best_iou['rotation_delta_from_selected_deg']:.2f} deg"
    )

    print()
    print("=== Depth 기준 상위 10개 ===")

    for record in sorted(
        valid_depth_records,
        key=lambda row: (
            row["debiased_median_mm"],
            -row["mask_iou"],
        ),
    )[:10]:
        marker = (
            " *SELECTED"
            if record["selected"]
            else ""
        )

        print(
            f"rank={record['rank']:02d} "
            f"score={record['foundationpose_score']:.6f} "
            f"IoU={record['mask_iou']:.4f} "
            f"abs={record['abs_median_mm']:.2f}mm "
            f"debiased={record['debiased_median_mm']:.2f}mm "
            f"dR={record['rotation_delta_from_selected_deg']:.2f}deg"
            f"{marker}"
        )

    print()
    print("저장:")
    print(json_path)
    print(csv_path)
    print(contact_sheet_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
