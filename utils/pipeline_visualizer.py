from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np
from numpy.typing import NDArray


POSE_AXIS_LENGTH_RATIO = 0.50


def _write_rgb_png(
    output_path: Path,
    rgb: NDArray[np.uint8],
) -> Path:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )
    success, encoded = cv2.imencode(
        ".png",
        bgr,
    )

    if not success:
        raise RuntimeError(
            f"Failed to encode visualization: {output_path}"
        )

    encoded.tofile(str(output_path))
    return output_path


def _mask_boundary(
    mask: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    mask_uint8 = mask.astype(np.uint8)
    eroded = cv2.erode(
        mask_uint8,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    return mask & (eroded == 0)


def _save_mask_overlay(
    *,
    prepared_view: Any,
    output_path: Path,
) -> Path:
    observed = prepared_view.view.rgb
    mask = prepared_view.segmentation.mask_bool
    overlay = observed.copy()

    green = np.zeros_like(observed)
    green[:, :, 1] = np.uint8(255)
    overlay[mask] = (
        0.68 * observed[mask].astype(np.float32)
        + 0.32 * green[mask].astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    overlay[_mask_boundary(mask)] = np.array(
        [0, 255, 0],
        dtype=np.uint8,
    )

    return _write_rgb_png(
        output_path,
        overlay,
    )


def _save_depth_colormap(
    *,
    depth_m: NDArray[np.float32],
    output_path: Path,
) -> Path:
    valid = depth_m > 0.0
    normalized = np.zeros(
        depth_m.shape,
        dtype=np.uint8,
    )

    if np.any(valid):
        valid_depth = depth_m[valid]
        near = float(
            np.percentile(valid_depth, 2.0)
        )
        far = float(
            np.percentile(valid_depth, 98.0)
        )

        if far <= near:
            far = near + 1e-6

        scaled = (
            (depth_m - near)
            / (far - near)
        )
        normalized[valid] = (
            255.0
            * (1.0 - np.clip(scaled[valid], 0.0, 1.0))
        ).astype(np.uint8)

    colored_bgr = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_TURBO,
    )
    colored_bgr[~valid] = 0
    colored_rgb = cv2.cvtColor(
        colored_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return _write_rgb_png(
        output_path,
        colored_rgb,
    )


def _evaluation_render_index(
    evaluation: Any,
) -> int:
    for render_index, hypothesis in enumerate(
        evaluation.candidate_result.hypotheses
    ):
        if hypothesis.rank == evaluation.hypothesis.rank:
            return render_index

    raise RuntimeError(
        "Selected hypothesis is missing from its render batch."
    )


def _project_proxy_point(
    *,
    point_proxy: NDArray[np.float64],
    pose_camera_from_proxy: NDArray[np.float64],
    camera_matrix: NDArray[np.float64],
) -> tuple[int, int] | None:
    point_camera = (
        pose_camera_from_proxy[:3, :3]
        @ point_proxy
        + pose_camera_from_proxy[:3, 3]
    )

    if (
        not np.isfinite(point_camera).all()
        or point_camera[2] <= 1e-6
    ):
        return None

    projected = camera_matrix @ point_camera
    pixel = projected[:2] / projected[2]

    safe_coordinate_limit = float(
        np.iinfo(np.int32).max - 1
    )

    if (
        not np.isfinite(pixel).all()
        or np.any(np.abs(pixel) > safe_coordinate_limit)
    ):
        return None

    return (
        int(np.rint(pixel[0])),
        int(np.rint(pixel[1])),
    )


def _draw_pose_center_and_axes(
    *,
    rgb: NDArray[np.uint8],
    pose_camera_from_proxy: NDArray[np.floating],
    camera_matrix: NDArray[np.floating],
    axis_length_m: float,
) -> NDArray[np.uint8]:
    """
    Proxy robust center와 XYZ 축을 RGB 영상에 투영한다.

    정규화·scale된 proxy의 원점은 robust center이며,
    축 길이는 scaled mesh robust diagonal의 일정 비율이다.
    """

    pose = np.asarray(
        pose_camera_from_proxy,
        dtype=np.float64,
    )
    intrinsics = np.asarray(
        camera_matrix,
        dtype=np.float64,
    )

    if pose.shape != (4, 4):
        raise ValueError(
            "pose_camera_from_proxy shape은 "
            f"(4, 4)여야 합니다: {pose.shape}"
        )

    if intrinsics.shape != (3, 3):
        raise ValueError(
            "camera_matrix shape은 "
            f"(3, 3)이어야 합니다: {intrinsics.shape}"
        )

    if (
        not np.isfinite(axis_length_m)
        or axis_length_m <= 0.0
    ):
        raise ValueError(
            "axis_length_m은 유한한 양수여야 합니다: "
            f"{axis_length_m}"
        )

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            "rgb shape은 (H, W, 3)이어야 합니다: "
            f"{rgb.shape}"
        )

    image_height, image_width = rgb.shape[:2]
    origin_proxy = np.zeros(3, dtype=np.float64)
    origin_pixel = _project_proxy_point(
        point_proxy=origin_proxy,
        pose_camera_from_proxy=pose,
        camera_matrix=intrinsics,
    )

    if (
        origin_pixel is None
        or not (
            0 <= origin_pixel[0] < image_width
            and 0 <= origin_pixel[1] < image_height
        )
    ):
        return rgb.copy()

    axis_specs = (
        (
            "X",
            np.array(
                [axis_length_m, 0.0, 0.0],
                dtype=np.float64,
            ),
            (255, 0, 0),
        ),
        (
            "Y",
            np.array(
                [0.0, axis_length_m, 0.0],
                dtype=np.float64,
            ),
            (0, 255, 0),
        ),
        (
            "Z",
            np.array(
                [0.0, 0.0, axis_length_m],
                dtype=np.float64,
            ),
            (0, 0, 255),
        ),
    )

    output = rgb.copy()

    for label, endpoint_proxy, color_rgb in axis_specs:
        endpoint_pixel = _project_proxy_point(
            point_proxy=endpoint_proxy,
            pose_camera_from_proxy=pose,
            camera_matrix=intrinsics,
        )

        if endpoint_pixel is None:
            continue

        is_visible, clipped_origin, clipped_endpoint = (
            cv2.clipLine(
                (
                    0,
                    0,
                    image_width,
                    image_height,
                ),
                origin_pixel,
                endpoint_pixel,
            )
        )

        if not is_visible:
            continue

        cv2.arrowedLine(
            output,
            clipped_origin,
            clipped_endpoint,
            color_rgb,
            thickness=3,
            line_type=cv2.LINE_AA,
            tipLength=0.12,
        )
        cv2.putText(
            output,
            label,
            (
                min(
                    image_width - 1,
                    max(0, clipped_endpoint[0] + 4),
                ),
                min(
                    image_height - 1,
                    max(0, clipped_endpoint[1] - 4),
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color_rgb,
            2,
            cv2.LINE_AA,
        )

    cv2.circle(
        output,
        origin_pixel,
        6,
        (0, 0, 0),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(
        output,
        origin_pixel,
        4,
        (255, 255, 0),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    return output


def _save_alignment_overlay(
    *,
    prepared_view: Any,
    evaluation: Any,
    output_path: Path,
) -> Path:
    render_index = _evaluation_render_index(
        evaluation
    )
    rendered_rgb = np.load(
        evaluation.render_directory
        / "rendered_rgb.npy",
        allow_pickle=False,
    )[render_index]
    rendered_mask = np.load(
        evaluation.render_directory
        / "rendered_masks.npy",
        allow_pickle=False,
    )[render_index].astype(np.bool_)
    observed = prepared_view.view.rgb

    if rendered_rgb.shape != observed.shape:
        raise ValueError(
            "Observed and rendered RGB shapes differ: "
            f"{observed.shape} vs {rendered_rgb.shape}"
        )

    overlay = observed.copy()
    overlay[rendered_mask] = (
        0.50
        * observed[rendered_mask].astype(np.float32)
        + 0.50
        * rendered_rgb[rendered_mask].astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    overlay[
        _mask_boundary(rendered_mask)
    ] = np.array(
        [255, 64, 64],
        dtype=np.uint8,
    )
    overlay = _draw_pose_center_and_axes(
        rgb=overlay,
        pose_camera_from_proxy=(
            evaluation.hypothesis.pose_cam_from_proxy
        ),
        camera_matrix=(
            prepared_view.view.camera_matrix
        ),
        axis_length_m=(
            POSE_AXIS_LENGTH_RATIO
            * float(
                evaluation.candidate_result.scale_m
            )
        ),
    )

    return _write_rgb_png(
        output_path,
        overlay,
    )


def _find_alignment_evaluation(
    *,
    alignment_evaluation: Any,
    candidate_index: int,
    hypothesis_rank: int,
) -> Any:
    matches = [
        evaluation
        for evaluation
        in alignment_evaluation.evaluations
        if (
            evaluation
            .candidate_result
            .candidate_index
            == candidate_index
            and evaluation.hypothesis.rank
            == hypothesis_rank
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one cross visualization "
            f"for candidate={candidate_index}, "
            f"rank={hypothesis_rank}; found {len(matches)}."
        )

    return matches[0]


def _mesh_preview(
    mesh_result: Any,
) -> Path | None:
    image_artifacts = [
        Path(path)
        for path in mesh_result.artifact_paths
        if (
            Path(path).suffix.lower()
            in {".png", ".jpg", ".jpeg"}
        )
    ]

    preferred = [
        path
        for path in image_artifacts
        if path.parent.name == "images"
    ]

    if preferred:
        return sorted(preferred)[0]

    if image_artifacts:
        return sorted(image_artifacts)[0]

    return None


def _relative_url(
    path: Path,
    report_directory: Path,
) -> str:
    relative = os.path.relpath(
        path,
        start=report_directory,
    )
    return quote(
        Path(relative).as_posix()
    )


def _image_card(
    *,
    title: str,
    path: Path | None,
    report_directory: Path,
) -> str:
    if path is None or not path.is_file():
        return (
            '<div class="card">'
            f"<h3>{html.escape(title)}</h3>"
            "<p>Not available.</p>"
            "</div>"
        )

    url = _relative_url(
        path,
        report_directory,
    )

    return (
        '<div class="card">'
        f"<h3>{html.escape(title)}</h3>"
        f'<a href="{url}">'
        f'<img src="{url}" alt="{html.escape(title)}">'
        "</a>"
        "</div>"
    )


def _file_link(
    *,
    label: str,
    path: Path,
    report_directory: Path,
) -> str:
    url = _relative_url(
        path,
        report_directory,
    )
    return (
        f'<a href="{url}">'
        f"{html.escape(label)}</a>"
    )


def _pose_table(
    pose: NDArray[np.float32] | None,
) -> str:
    if pose is None:
        return "<p>No final pose was accepted.</p>"

    rows = []

    for row in pose:
        cells = "".join(
            f"<td>{float(value):.6f}</td>"
            for value in row
        )
        rows.append(f"<tr>{cells}</tr>")

    return (
        '<table class="matrix">'
        + "".join(rows)
        + "</table>"
    )


def _consistency_table(
    result: Any,
) -> str:
    rows = result.reference_candidate_count
    columns = result.query_candidate_count
    losses = np.full(
        (rows, columns),
        np.nan,
        dtype=np.float32,
    )
    accepted = np.zeros(
        (rows, columns),
        dtype=np.bool_,
    )

    for pair in result.pairs:
        row = pair.reference_candidate_index
        column = pair.query_candidate_index
        losses[row, column] = pair.consistency_loss
        accepted[row, column] = (
            pair.passes_hard_gate
        )

    finite = losses[np.isfinite(losses)]
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    span = max(maximum - minimum, 1e-8)

    header = (
        "<tr><th>Ref \\ Query</th>"
        + "".join(
            f"<th>Q{column}</th>"
            for column in range(columns)
        )
        + "</tr>"
    )
    body_rows = []

    for row in range(rows):
        cells = [f"<th>R{row}</th>"]

        for column in range(columns):
            loss = float(losses[row, column])
            ratio = (loss - minimum) / span
            red = int(55 + 170 * ratio)
            green = int(190 - 120 * ratio)
            marker = (
                "PASS"
                if accepted[row, column]
                else "FAIL"
            )
            cells.append(
                "<td style="
                f'"background: rgb({red},{green},65)">'
                f"{loss:.4f}<br>{marker}</td>"
            )

        body_rows.append(
            "<tr>" + "".join(cells) + "</tr>"
        )

    return (
        '<table class="heatmap">'
        + header
        + "".join(body_rows)
        + "</table>"
    )


def save_pipeline_visualization_report(
    *,
    output_root: Path,
    reference_view: Any,
    query_view: Any,
    reference_mesh_result: Any,
    query_mesh_result: Any,
    reference_self_evaluation: Any,
    query_self_evaluation: Any,
    cross_evidence: Any,
    consistency_result: Any,
    final_result: Any,
) -> Path:
    output_root = (
        Path(output_root)
        .expanduser()
        .resolve()
    )
    report_directory = (
        output_root / "visualizations"
    )
    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_mask_overlay = (
        _save_mask_overlay(
            prepared_view=reference_view,
            output_path=(
                report_directory
                / "stage_01_reference_mask_overlay.png"
            ),
        )
    )
    query_mask_overlay = _save_mask_overlay(
        prepared_view=query_view,
        output_path=(
            report_directory
            / "stage_01_query_mask_overlay.png"
        ),
    )
    reference_depth = _save_depth_colormap(
        depth_m=reference_view.masked_depth_m,
        output_path=(
            report_directory
            / "stage_01_reference_depth.png"
        ),
    )
    query_depth = _save_depth_colormap(
        depth_m=query_view.masked_depth_m,
        output_path=(
            report_directory
            / "stage_01_query_depth.png"
        ),
    )

    reference_self_overlay = (
        _save_alignment_overlay(
            prepared_view=reference_view,
            evaluation=(
                reference_self_evaluation.best
            ),
            output_path=(
                report_directory
                / "stage_04_reference_self_overlay.png"
            ),
        )
    )
    query_self_overlay = (
        _save_alignment_overlay(
            prepared_view=query_view,
            evaluation=query_self_evaluation.best,
            output_path=(
                report_directory
                / "stage_04_query_self_overlay.png"
            ),
        )
    )

    reference_cross_result = (
        cross_evidence.reference_proxy
    )
    query_cross_result = cross_evidence.query_proxy
    reference_cross_overlay = (
        _save_alignment_overlay(
            prepared_view=query_view,
            evaluation=(
                reference_cross_result
                .alignment_evaluation
                .best
            ),
            output_path=(
                report_directory
                / "stage_07_reference_proxy_to_query.png"
            ),
        )
    )
    query_cross_overlay = (
        _save_alignment_overlay(
            prepared_view=reference_view,
            evaluation=(
                query_cross_result
                .alignment_evaluation
                .best
            ),
            output_path=(
                report_directory
                / "stage_07_query_proxy_to_reference.png"
            ),
        )
    )

    final_overlay: Path | None = None
    selected_candidate = (
        final_result.selected_candidate
    )

    if selected_candidate is not None:
        if (
            selected_candidate.path_name
            == "reference_proxy"
        ):
            selected_path_result = (
                reference_cross_result
            )
            selected_target_view = query_view
        else:
            selected_path_result = (
                query_cross_result
            )
            selected_target_view = reference_view

        selected_evaluation = (
            _find_alignment_evaluation(
                alignment_evaluation=(
                    selected_path_result
                    .alignment_evaluation
                ),
                candidate_index=(
                    selected_candidate
                    .cross_candidate_index
                ),
                hypothesis_rank=(
                    selected_candidate
                    .cross_hypothesis_rank
                ),
            )
        )
        final_overlay = _save_alignment_overlay(
            prepared_view=selected_target_view,
            evaluation=selected_evaluation,
            output_path=(
                report_directory
                / "stage_08_final_selected_overlay.png"
            ),
        )

    reference_preview = _mesh_preview(
        reference_mesh_result
    )
    query_preview = _mesh_preview(
        query_mesh_result
    )
    pose = (
        final_result
        .selected_relative_pose_query_from_reference
    )

    manifest_path = (
        report_directory / "stage_visuals.json"
    )
    manifest = {
        "stage_01_prepare": {
            "reference_mask_overlay": str(
                reference_mask_overlay
            ),
            "query_mask_overlay": str(
                query_mask_overlay
            ),
            "reference_depth": str(
                reference_depth
            ),
            "query_depth": str(query_depth),
        },
        "stage_02_proxy": {
            "reference_preview": (
                str(reference_preview)
                if reference_preview is not None
                else None
            ),
            "query_preview": (
                str(query_preview)
                if query_preview is not None
                else None
            ),
            "reference_mesh": str(
                reference_mesh_result
                .primary_output_path
            ),
            "query_mesh": str(
                query_mesh_result
                .primary_output_path
            ),
        },
        "stage_03_self_foundationpose": {
            "reference_summary": str(
                reference_self_evaluation
                .summary_path
            ),
            "query_summary": str(
                query_self_evaluation
                .summary_path
            ),
        },
        "stage_04_self_evaluation": {
            "reference_overlay": str(
                reference_self_overlay
            ),
            "query_overlay": str(
                query_self_overlay
            ),
        },
        "stage_05_cross_alignment": {
            "summary": str(
                output_root
                / "cross_alignment"
                / "bidirectional_cross_alignment.json"
            ),
        },
        "stage_06_relative_pose": {
            "summary": str(
                output_root
                / "relative_pose_candidates"
                / "relative_pose_candidates.json"
            ),
        },
        "stage_07_cross_evidence": {
            "consistency_summary": str(
                output_root
                / "consistency"
                / "bidirectional_consistency.json"
            ),
            "cross_evidence_summary": str(
                cross_evidence.summary_path
            ),
            "reference_proxy_to_query": str(
                reference_cross_overlay
            ),
            "query_proxy_to_reference": str(
                query_cross_overlay
            ),
        },
        "stage_08_final": {
            "status": final_result.status,
            "selected_path": (
                final_result.selected_path_name
            ),
            "overlay": (
                str(final_overlay)
                if final_overlay is not None
                else None
            ),
            "pose_query_from_reference": (
                pose.tolist()
                if pose is not None
                else None
            ),
        },
        "pose_overlay_convention": {
            "center": (
                "scaled proxy robust center (proxy origin)"
            ),
            "axis_length_ratio_of_robust_diagonal": (
                POSE_AXIS_LENGTH_RATIO
            ),
            "colors_rgb": {
                "center": [255, 255, 0],
                "x": [255, 0, 0],
                "y": [0, 255, 0],
                "z": [0, 0, 255],
            },
        },
    }

    with manifest_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    stage_01_cards = "".join(
        (
            _image_card(
                title="Reference mask overlay",
                path=reference_mask_overlay,
                report_directory=report_directory,
            ),
            _image_card(
                title="Reference masked depth",
                path=reference_depth,
                report_directory=report_directory,
            ),
            _image_card(
                title="Query mask overlay",
                path=query_mask_overlay,
                report_directory=report_directory,
            ),
            _image_card(
                title="Query masked depth",
                path=query_depth,
                report_directory=report_directory,
            ),
        )
    )
    stage_02_cards = "".join(
        (
            _image_card(
                title="Reference InstantMesh views",
                path=reference_preview,
                report_directory=report_directory,
            ),
            _image_card(
                title="Query InstantMesh views",
                path=query_preview,
                report_directory=report_directory,
            ),
        )
    )
    stage_04_cards = "".join(
        (
            _image_card(
                title="Reference self alignment",
                path=reference_self_overlay,
                report_directory=report_directory,
            ),
            _image_card(
                title="Query self alignment",
                path=query_self_overlay,
                report_directory=report_directory,
            ),
        )
    )
    stage_07_cards = "".join(
        (
            _image_card(
                title="Reference proxy -> Query",
                path=reference_cross_overlay,
                report_directory=report_directory,
            ),
            _image_card(
                title="Query proxy -> Reference",
                path=query_cross_overlay,
                report_directory=report_directory,
            ),
        )
    )
    final_card = _image_card(
        title="Final selected cross alignment",
        path=final_overlay,
        report_directory=report_directory,
    )

    report_path = report_directory / "index.html"
    report_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pose pipeline visualization</title>
<style>
body {{ margin: 0 auto; max-width: 1280px; padding: 28px;
       color: #e5e7eb; background: #111827;
       font-family: system-ui, sans-serif; }}
h1, h2, h3 {{ color: #f9fafb; }}
section {{ margin: 28px 0; padding: 20px; background: #1f2937;
          border-radius: 12px; }}
.grid {{ display: grid; grid-template-columns:
         repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
.card {{ padding: 12px; background: #111827; border-radius: 9px; }}
img {{ width: 100%; height: auto; border-radius: 6px; }}
a {{ color: #7dd3fc; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #4b5563; padding: 9px; text-align: center; }}
.matrix td {{ font-family: ui-monospace, monospace; }}
.status {{ font-size: 1.2rem; font-weight: 700; }}
</style>
</head>
<body>
<h1>Pose pipeline visualization</h1>
<p>Every image below is derived from artifacts already produced by the
pipeline; this report does not rerun FoundationPose.</p>
<p>Pose overlay: <span style="color:#facc15">center</span>,
<span style="color:#ef4444">X</span>,
<span style="color:#22c55e">Y</span>,
<span style="color:#3b82f6">Z</span>.
Axis length is 50% of the scaled proxy robust diagonal.</p>

<section><h2>1. RGB-D preparation and masks</h2>
<div class="grid">{stage_01_cards}</div></section>

<section><h2>2. InstantMesh proxy generation</h2>
<div class="grid">{stage_02_cards}</div>
<p>{_file_link(label="Reference OBJ",
path=Path(reference_mesh_result.primary_output_path),
report_directory=report_directory)} |
{_file_link(label="Query OBJ",
path=Path(query_mesh_result.primary_output_path),
report_directory=report_directory)}</p></section>

<section><h2>3. FoundationPose self hypotheses</h2>
<p>{_file_link(label="Reference hypothesis scores",
path=Path(reference_self_evaluation.summary_path),
report_directory=report_directory)} |
{_file_link(label="Query hypothesis scores",
path=Path(query_self_evaluation.summary_path),
report_directory=report_directory)}</p></section>

<section><h2>4. Self-alignment mask/depth selection</h2>
<div class="grid">{stage_04_cards}</div></section>

<section><h2>5. Bidirectional cross-alignment</h2>
<p>{_file_link(label="Cross-alignment metadata",
path=output_root / "cross_alignment" /
"bidirectional_cross_alignment.json",
report_directory=report_directory)}</p></section>

<section><h2>6. Relative pose candidates</h2>
<p>{_file_link(label="Relative pose candidates",
path=output_root / "relative_pose_candidates" /
"relative_pose_candidates.json",
report_directory=report_directory)}</p></section>

<section><h2>7. Consistency and cross evidence</h2>
{_consistency_table(consistency_result)}
<div class="grid">{stage_07_cards}</div></section>

<section><h2>8. Final selection</h2>
<p class="status">Status: {html.escape(final_result.status)}</p>
<p>Selected path: {html.escape(str(final_result.selected_path_name))}</p>
{_pose_table(pose)}
<div class="grid">{final_card}</div></section>

<p>{_file_link(label="Visualization manifest",
path=manifest_path,
report_directory=report_directory)}</p>
</body>
</html>
"""

    report_path.write_text(
        report_html,
        encoding="utf-8",
    )

    return report_path
