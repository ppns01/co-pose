from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


OBSERVED_BOUNDARY = np.array(
    [0, 255, 0],
    dtype=np.uint8,
)

RENDERED_BOUNDARY = np.array(
    [255, 96, 32],
    dtype=np.uint8,
)


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if bgr is None:
        raise FileNotFoundError(
            f"RGB image not found: {path}"
        )

    return cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )


def load_mask(
    path: Path,
    shape: tuple[int, int],
) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        mask = np.load(
            path,
            allow_pickle=False,
        )
    else:
        mask = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise FileNotFoundError(
                f"Mask not found: {path}"
            )

    mask = np.asarray(mask).astype(bool)

    if mask.shape != shape:
        raise ValueError(
            "Mask shape mismatch: "
            f"{mask.shape} != {shape}"
        )

    return mask


def load_camera_k(
    camera_json: Path,
    image_id: int,
) -> np.ndarray:
    with camera_json.open(
        "r",
        encoding="utf-8",
    ) as file:
        records = json.load(file)

    key = str(image_id)

    if key not in records:
        raise KeyError(
            f"image_id={image_id} not found "
            f"in {camera_json}"
        )

    return np.asarray(
        records[key]["cam_K"],
        dtype=np.float64,
    ).reshape(3, 3)


def load_mesh_camera(
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
        raise ValueError(
            f"Empty mesh: {path}"
        )

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64,
    )

    triangles = np.asarray(
        mesh.triangles,
        dtype=np.int64,
    )

    return vertices, triangles


def build_rays(
    height: int,
    width: int,
    k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(
        np.arange(
            width,
            dtype=np.float32,
        ),
        np.arange(
            height,
            dtype=np.float32,
        ),
    )

    directions = np.stack(
        (
            (
                u
                - float(k[0, 2])
            )
            / float(k[0, 0]),
            (
                v
                - float(k[1, 2])
            )
            / float(k[1, 1]),
            np.ones_like(u),
        ),
        axis=-1,
    ).astype(np.float32)

    origins = np.zeros_like(
        directions
    )

    rays = np.concatenate(
        (
            origins,
            directions,
        ),
        axis=-1,
    )

    return rays, directions


def raycast(
    vertices_camera: np.ndarray,
    triangles: np.ndarray,
    k: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    legacy_mesh = (
        o3d.geometry.TriangleMesh()
    )

    legacy_mesh.vertices = (
        o3d.utility.Vector3dVector(
            vertices_camera
        )
    )

    legacy_mesh.triangles = (
        o3d.utility.Vector3iVector(
            triangles.astype(
                np.int32
            )
        )
    )

    legacy_mesh.compute_triangle_normals()

    tensor_mesh = (
        o3d.t.geometry.TriangleMesh
        .from_legacy(
            legacy_mesh
        )
    )

    scene = (
        o3d.t.geometry
        .RaycastingScene()
    )

    scene.add_triangles(
        tensor_mesh
    )

    height, width = image_shape

    rays, directions = build_rays(
        height,
        width,
        k,
    )

    hit = scene.cast_rays(
        o3d.core.Tensor(
            rays,
            dtype=(
                o3d.core.Dtype.Float32
            ),
        )
    )

    t_hit = (
        hit["t_hit"]
        .numpy()
    )

    normals = (
        hit["primitive_normals"]
        .numpy()
        .astype(np.float32)
    )

    rendered_mask = (
        np.isfinite(t_hit)
    )

    normals[
        ~rendered_mask
    ] = 0.0

    ray_norm = np.linalg.norm(
        directions,
        axis=-1,
        keepdims=True,
    )

    ray_unit = (
        directions
        / np.maximum(
            ray_norm,
            1e-12,
        )
    )

    # OBJ face winding이 반대여도
    # 보이는 normal이 카메라를 향하도록 통일한다.
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
        normals,
        ray_unit,
    )


def boundary(
    mask: np.ndarray,
) -> np.ndarray:
    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    eroded = cv2.erode(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
    )

    return (
        mask
        & (eroded == 0)
    )


def mask_iou(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    intersection = np.count_nonzero(
        mask_a & mask_b
    )

    union = np.count_nonzero(
        mask_a | mask_b
    )

    if union == 0:
        return 1.0

    return float(
        intersection / union
    )


def shaded_mesh(
    normals: np.ndarray,
    ray_unit: np.ndarray,
    mask: np.ndarray,
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
        0.25
        + 0.75 * diffuse
    )

    base_rgb = np.array(
        [242.0, 181.0, 58.0],
        dtype=np.float32,
    )

    rendered = (
        base_rgb[
            None,
            None,
            :
        ]
        * intensity[..., None]
    )

    rendered = np.clip(
        rendered,
        0,
        255,
    ).astype(np.uint8)

    rendered[~mask] = 0

    return rendered


def normal_mesh(
    normals: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    rendered = (
        (normals + 1.0)
        * 127.5
    )

    rendered = np.clip(
        rendered,
        0,
        255,
    ).astype(np.uint8)

    rendered[~mask] = 0

    return rendered


def overlay(
    rgb: np.ndarray,
    rendered: np.ndarray,
    rendered_mask: np.ndarray,
    observed_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    result = rgb.copy()

    result[rendered_mask] = (
        (
            1.0 - alpha
        )
        * rgb[
            rendered_mask
        ].astype(np.float32)
        + alpha
        * rendered[
            rendered_mask
        ].astype(np.float32)
    ).clip(
        0,
        255,
    ).astype(np.uint8)

    observed_boundary = boundary(
        observed_mask
    )

    rendered_boundary = boundary(
        rendered_mask
    )

    result[
        observed_boundary
    ] = OBSERVED_BOUNDARY

    result[
        rendered_boundary
    ] = RENDERED_BOUNDARY

    return result


def add_header(
    image: np.ndarray,
    lines: list[str],
) -> np.ndarray:
    header_height = (
        30
        + 24 * len(lines)
    )

    canvas = np.zeros(
        (
            header_height
            + image.shape[0],
            image.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        header_height:
    ] = image

    for index, line in enumerate(
        lines
    ):
        cv2.putText(
            canvas,
            line,
            (
                10,
                25
                + 24 * index,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def save_rgb(
    path: Path,
    image: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    success = cv2.imwrite(
        str(path),
        bgr,
    )

    if not success:
        raise RuntimeError(
            f"Failed to save: {path}"
        )


def parse_stage(
    value: str,
) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Stage format must be "
            "NAME=/path/to/mesh.obj"
        )

    name, path_text = value.split(
        "=",
        1,
    )

    name = name.strip()

    if not name:
        raise argparse.ArgumentTypeError(
            "Stage name is empty"
        )

    path = Path(
        path_text
    ).expanduser().resolve()

    return name, path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Project camera-space meshes "
            "onto query RGB as filled "
            "shaded and normal-colored surfaces."
        )
    )

    parser.add_argument(
        "--rgb-image",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--mask-bool",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--camera-json",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--image-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--stage",
        action="append",
        type=parse_stage,
        required=True,
        help=(
            "Repeat this argument: "
            "--stage "
            "M0_original=/path/mesh.obj"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.58,
    )

    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(
            "--alpha must be "
            "between 0 and 1"
        )

    rgb_path = (
        args.rgb_image
        .expanduser()
        .resolve()
    )

    mask_path = (
        args.mask_bool
        .expanduser()
        .resolve()
    )

    camera_json_path = (
        args.camera_json
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    rgb = load_rgb(
        rgb_path
    )

    height, width = rgb.shape[:2]

    observed_mask = load_mask(
        mask_path,
        (
            height,
            width,
        ),
    )

    camera_k = load_camera_k(
        camera_json_path,
        args.image_id,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    surface_panels: list[
        np.ndarray
    ] = []

    normal_panels: list[
        np.ndarray
    ] = []

    metrics: dict[
        str,
        dict[
            str,
            float | int | str
        ],
    ] = {}

    for (
        stage_name,
        mesh_path,
    ) in args.stage:
        vertices_camera, triangles = (
            load_mesh_camera(
                mesh_path
            )
        )

        (
            rendered_mask,
            normals,
            ray_unit,
        ) = raycast(
            vertices_camera,
            triangles,
            camera_k,
            (
                height,
                width,
            ),
        )

        iou = mask_iou(
            observed_mask,
            rendered_mask,
        )

        shaded = shaded_mesh(
            normals,
            ray_unit,
            rendered_mask,
        )

        normal_colors = normal_mesh(
            normals,
            rendered_mask,
        )

        surface_overlay = overlay(
            rgb,
            shaded,
            rendered_mask,
            observed_mask,
            args.alpha,
        )

        normal_overlay = overlay(
            rgb,
            normal_colors,
            rendered_mask,
            observed_mask,
            args.alpha,
        )

        common_lines = [
            (
                f"{stage_name}: "
                f"mask_iou={iou:.4f}"
            ),
            (
                "green=observed boundary | "
                "orange=mesh boundary"
            ),
        ]

        surface_panel = add_header(
            surface_overlay,
            common_lines
            + [
                "shaded projected mesh"
            ],
        )

        normal_panel = add_header(
            normal_overlay,
            common_lines
            + [
                (
                    "camera-normal RGB: "
                    "rotation/branch inspection"
                )
            ],
        )

        save_rgb(
            output_dir
            / (
                f"{stage_name}"
                "_surface_overlay.png"
            ),
            surface_panel,
        )

        save_rgb(
            output_dir
            / (
                f"{stage_name}"
                "_normal_overlay.png"
            ),
            normal_panel,
        )

        surface_panels.append(
            surface_panel
        )

        normal_panels.append(
            normal_panel
        )

        metrics[stage_name] = {
            "mesh_path": (
                str(mesh_path)
            ),
            "mask_iou": iou,
            "observed_pixels": int(
                np.count_nonzero(
                    observed_mask
                )
            ),
            "rendered_pixels": int(
                np.count_nonzero(
                    rendered_mask
                )
            ),
        }

    surface_row = np.concatenate(
        surface_panels,
        axis=1,
    )

    normal_row = np.concatenate(
        normal_panels,
        axis=1,
    )

    combined = np.concatenate(
        (
            surface_row,
            normal_row,
        ),
        axis=0,
    )

    diagnostic_path = (
        output_dir
        / "M0_M3_mesh_surface_diagnostic.png"
    )

    save_rgb(
        diagnostic_path,
        combined,
    )

    metrics_path = (
        output_dir
        / "M0_M3_mesh_surface_metrics.json"
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
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        f"saved: {diagnostic_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
