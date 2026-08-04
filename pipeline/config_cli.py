from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

from pipeline.config_defaults import (
    DEFAULT_PIPELINE_CONFIG,
    DINOV3_CHECKPOINT_FILENAMES,
    POSE_PATH_CHOICES,
    PROJECT_ROOT,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    from config_loader import (
        load_pipeline_config_file,
    )

    materialized_argv = (
        list(argv) if argv is not None else None
    )

    bootstrap_parser = argparse.ArgumentParser(
        add_help=False,
    )
    bootstrap_parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=DEFAULT_PIPELINE_CONFIG,
    )
    bootstrap_args, _ = (
        bootstrap_parser.parse_known_args(
            materialized_argv
        )
    )
    selected_config_path = bootstrap_args.config_path.expanduser().resolve()
    default_config_path = (
        DEFAULT_PIPELINE_CONFIG.resolve()
    )

    merged_values = load_pipeline_config_file(
        default_config_path,
        project_root=PROJECT_ROOT,
        require_complete=True,
    )

    if (
        selected_config_path
        != default_config_path
    ):
        override_values = (
            load_pipeline_config_file(
                selected_config_path,
                project_root=PROJECT_ROOT,
                require_complete=False,
            )
        )

        if (
            override_values.get("query_image_ids")
            is not None
        ):
            merged_values["query_image_id"] = None

        if (
            override_values.get("query_image_id")
            is not None
        ):
            merged_values["query_image_ids"] = (
                None
            )

        if (
            "object_id" in override_values
            and override_values["object_id"]
            != merged_values["object_id"]
        ):
            if (
                "object_name"
                not in override_values
            ):
                merged_values["object_name"] = (
                    None
                )

            if (
                "sam3_prompt"
                not in override_values
            ):
                merged_values["sam3_prompt"] = (
                    None
                )

        if (
            (
                "dinov3_model" in override_values
                or "dinov3_repository"
                in override_values
            )
            and "dinov3_checkpoint"
            not in override_values
        ):
            merged_values["dinov3_checkpoint"] = (
                None
            )

        merged_values.update(override_values)

    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS,
        description=(
            "LINEMOD Reference/Query RGB-D에서 두 proxy mesh를 "
            "생성하고 양방향 상대 pose를 선택합니다."
        ),
    )

    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        help=(
            "Pipeline YAML file. Values override configs/pipeline.yaml."
        ),
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        help=(
            "BOP LINEMOD root. 기본값: <project>/datasets"
        ),
    )

    parser.add_argument(
        "--pose-path",
        choices=POSE_PATH_CHOICES,
        help=(
            "Pose experiment path. Method-specific main files set this automatically."
        ),
    )

    parser.add_argument(
        "--split",
        help="BOP split 이름. 기본값: test",
    )

    object_group = (
        parser.add_mutually_exclusive_group()
    )
    object_group.add_argument(
        "--object-id",
        type=int,
        help="BOP LINEMOD object ID. Default: 8",
    )
    object_group.add_argument(
        "--object-ids",
        type=int,
        nargs="+",
        help=(
            "Run multiple LINEMOD objects sequentially in "
            "separate Python processes. Each object's scene "
            "ID is set to its object ID, and every object "
            "uses the same query image option(s). With "
            "--all-linemod, this filters the full-dataset run "
            "to the listed objects."
        ),
    )

    parser.add_argument(
        "--object-name",
        help=(
            "Object name. Default: driller for object 8, otherwise object_<ID>."
        ),
    )

    parser.add_argument(
        "--sam3-prompt",
        help=(
            "SAM3 text prompt. Default: 'power drill' "
            "for object 8, otherwise object name."
        ),
    )

    parser.add_argument(
        "--reference-scene-id",
        type=int,
        help="Reference scene ID. Default: 8",
    )

    parser.add_argument(
        "--reference-image-id",
        type=int,
        help="Reference image ID. Default: 0",
    )

    parser.add_argument(
        "--reference-instance-index",
        type=int,
    )

    parser.add_argument(
        "--query-scene-id",
        type=int,
        help="Query scene ID. Default: 8",
    )

    query_image_group = (
        parser.add_mutually_exclusive_group()
    )

    query_image_group.add_argument(
        "--query-image-id",
        type=int,
        help="Query image ID. Default: 1",
    )

    query_image_group.add_argument(
        "--query-image-ids",
        type=int,
        nargs="+",
        help=(
            "Multiple query image IDs. Reference preparation "
            "and self-alignment are reused once."
        ),
    )

    parser.add_argument(
        "--query-instance-index",
        type=int,
    )

    parser.add_argument(
        "--mask-type",
        choices=(
            "mask_visib",
            "mask",
        ),
    )

    parser.add_argument(
        "--instantmesh-repository",
        type=Path,
    )

    parser.add_argument(
        "--instantmesh-python",
        type=Path,
        help=(
            "InstantMesh environment Python executable. "
            "By default, auto-detects the sibling "
            "instantmesh_clean conda environment."
        ),
    )

    parser.add_argument(
        "--instantmesh-config",
        type=Path,
        help=(
            "InstantMesh YAML. Default: project 16 GB proxy config with grid_res=64."
        ),
    )

    offline_group = (
        parser.add_mutually_exclusive_group()
    )
    offline_group.add_argument(
        "--instantmesh-offline",
        action="store_true",
        help=(
            "Hugging Face 자동 다운로드를 차단합니다. "
            "모델이 캐시에 있을 때만 사용하세요."
        ),
    )
    offline_group.add_argument(
        "--instantmesh-online",
        "--no-instantmesh-offline",
        dest="instantmesh_offline",
        action="store_false",
        help=(
            "Allow InstantMesh network access when its cache is missing."
        ),
    )

    parser.add_argument(
        "--foundationpose-repository",
        type=Path,
    )

    resume_group = (
        parser.add_mutually_exclusive_group()
    )
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help=(
            "output_root/batch_summary.json에서 이미 "
            "status=completed인 쿼리를 건너뛰고 이어서 "
            "실행합니다. Shared reference는 항상 다시 "
            "계산합니다."
        ),
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="모든 쿼리를 처음부터 다시 실행합니다.",
    )

    dino_group = (
        parser.add_mutually_exclusive_group()
    )
    dino_group.add_argument(
        "--enable-dino",
        dest="dino_enabled",
        action="store_true",
        help="Enable DINOv3 appearance evaluation.",
    )
    dino_group.add_argument(
        "--disable-dino",
        dest="dino_enabled",
        action="store_false",
        help=(
            "DINOv3 양방향 appearance 평가를 비활성화합니다."
        ),
    )

    parser.add_argument(
        "--dinov3-repository",
        type=Path,
    )

    parser.add_argument(
        "--dinov3-checkpoint",
        type=Path,
        help=(
            "DINOv3 .pth 파일 또는 공식 HF 모델 "
            "디렉터리. 기본값: "
            "<dinov3-repository>/weights/"
            "<dinov3-...-pretrain-lvd1689m>"
        ),
    )

    parser.add_argument(
        "--dinov3-model",
        choices=tuple(
            DINOV3_CHECKPOINT_FILENAMES
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
    )

    parser.add_argument(
        "--device",
    )

    parser.add_argument(
        "--top-k",
        type=int,
    )

    parser.add_argument(
        "--foundationpose-rotation-diversity-threshold-deg",
        type=float,
        help=(
            "Minimum SO(3) separation used to keep score-ranked "
            "FoundationPose hypotheses rotation-diverse."
        ),
    )

    parser.add_argument(
        "--refine-iterations",
        type=int,
    )

    parser.add_argument(
        "--foundationpose-workers",
        type=int,
        help=(
            "Independent FoundationPose process count. Default: 1 (sequential)."
        ),
    )

    visible_scale_group = (
        parser.add_mutually_exclusive_group()
    )
    visible_scale_group.add_argument(
        "--enable-visible-scale-refinement",
        dest="visible_scale_refinement_enabled",
        action="store_true",
        help=(
            "Self pose의 visible proxy/RGB-D 대응점으로 "
            "등방성 scale을 보정한 뒤 FoundationPose를 "
            "좁은 scale 범위에서 다시 실행합니다."
        ),
    )
    visible_scale_group.add_argument(
        "--disable-visible-scale-refinement",
        dest="visible_scale_refinement_enabled",
        action="store_false",
        help=(
            "Visible correspondence scale 보정을 "
            "비활성화하고 기존 coarse scale 결과를 "
            "그대로 사용합니다."
        ),
    )

    shared_axis_scale_group = (
        parser.add_mutually_exclusive_group()
    )
    shared_axis_scale_group.add_argument(
        "--enable-shared-axis-scale-refinement",
        dest="enable_shared_axis_scale_refinement",
        action="store_true",
        help=(
            "A0/B0 기준 1차 dGeDi(G0)의 reference->query 회전 R0로 "
            "reference/query proxy에 동일한 물리 변형 D를 공유 적용합니다"
            "(Cr=D, Cq=R0 D R0^T). G0는 R0를 얻기 위한 용도로 단 한 번만 "
            "실행되며, 폴백이나 게이트 없이 보정된 A1/B1을 반환합니다. "
            "최종 등록(G)은 이후 단계에서 한 번만 계산됩니다. "
            "기본값: 비활성화."
        ),
    )
    shared_axis_scale_group.add_argument(
        "--disable-shared-axis-scale-refinement",
        dest="enable_shared_axis_scale_refinement",
        action="store_false",
        help="공유 D 탐색을 비활성화하고 기존 independent axis-scale만 사용합니다.",
    )

    explicit_values = vars(
        parser.parse_args(materialized_argv)
    )
    explicit_values.pop(
        "config_path",
        None,
    )

    if "query_image_id" in explicit_values:
        merged_values["query_image_ids"] = None

    if "query_image_ids" in explicit_values:
        merged_values["query_image_id"] = None

    if (
        "object_id" in explicit_values
        and explicit_values["object_id"]
        != merged_values["object_id"]
    ):
        if "object_name" not in explicit_values:
            merged_values["object_name"] = None

        if "sam3_prompt" not in explicit_values:
            merged_values["sam3_prompt"] = None

    if (
        (
            "dinov3_model" in explicit_values
            or "dinov3_repository"
            in explicit_values
        )
        and "dinov3_checkpoint"
        not in explicit_values
    ):
        merged_values["dinov3_checkpoint"] = None

    merged_values.update(explicit_values)
    merged_values["config_path"] = (
        selected_config_path
    )

    return argparse.Namespace(**merged_values)


def _batch_query_output_label(
    image_ids: tuple[int, ...],
) -> str:
    joined_ids = "-".join(
        f"{image_id:06d}"
        for image_id in image_ids
    )

    if len(joined_ids) <= 96:
        return joined_ids

    digest = hashlib.sha256(
        joined_ids.encode("ascii")
    ).hexdigest()[:12]

    return f"{image_ids[0]:06d}-{image_ids[-1]:06d}-n{len(image_ids):04d}-{digest}"

