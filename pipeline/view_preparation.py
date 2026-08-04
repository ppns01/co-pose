from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

from core.types import FrameSpec, PipelineConfig
from pipeline.config_build import _pose_path_uses_dino


def _prepared_mask_source(
    prepared_view: Any,
) -> str:
    segmentation = getattr(
        prepared_view, "segmentation", None
    )
    source = getattr(
        segmentation, "source", "sam3"
    )
    normalized = str(source).strip()
    return normalized or "sam3"


def _prepare_linemod_view(
    *,
    config: PipelineConfig,
    view_name: str,
    frame: FrameSpec,
    output_root: Path | None = None,
) -> Any:
    from dataset_io.linemod_loader import (
        load_linemod_view,
    )
    from input_selector import build_view_input
    from mask_provider import (
        generate_sam3_segmentation,
        load_existing_segmentation,
    )
    from preprocessing.mask_processing import (
        prepare_masked_view,
    )

    selected_input = build_view_input(
        dataset_root=config.dataset_root,
        split=config.split,
        scene_id=frame.scene_id,
        image_id=frame.image_id,
        object_id=config.object_id,
        instance_index=frame.instance_index,
        mask_type=config.mask_type,
    )

    loaded_view = load_linemod_view(
        dataset_root=config.dataset_root,
        view_name=view_name,
        scene_id=frame.scene_id,
        image_id=frame.image_id,
        object_name=config.object_name,
        object_id=config.object_id,
        split=config.split,
    )

    if (
        selected_input.rgb_path.resolve()
        != loaded_view.source.rgb_path.resolve()
        or selected_input.depth_path.resolve()
        != loaded_view.source.depth_path.resolve()
    ):
        raise RuntimeError(
            "Input selector and LINEMOD loader resolved "
            f"different files for {view_name}."
        )

    resolved_output_root = (
        config.output_root
        if output_root is None
        else output_root
    )

    view_output = (
        resolved_output_root / "views" / view_name
    )

    sam3_error: Exception | None = None
    try:
        segmentation = generate_sam3_segmentation(
            view=loaded_view,
            output_directory=(
                view_output / "segmentation"
            ),
            text_prompt=config.sam3_prompt,
            repository_path=(
                config.sam3_repository
            ),
            checkpoint_path=(
                config.sam3_checkpoint
            ),
            bpe_path=config.sam3_bpe,
            device=config.sam3_device,
            use_amp=config.sam3_use_amp,
            confidence_threshold=(
                config.sam3_confidence_threshold
            ),
        )
        prepared_view = prepare_masked_view(
            view=loaded_view,
            segmentation=segmentation,
            output_directory=(
                view_output / "prepared"
            ),
        )
    except Exception as error:
        sam3_error = error
        if not config.gt_mask_fallback_on_sam3_failure:
            raise

        print(
            "[SAM3 mask failed -> GT fallback] "
            f"view={view_name} scene={frame.scene_id} "
            f"image={frame.image_id} "
            f"error={type(error).__name__}: {error}",
            file=sys.stderr,
        )
        try:
            segmentation = load_existing_segmentation(
                view=loaded_view,
                mask_path=selected_input.mask_path,
                output_directory=(
                    view_output
                    / "segmentation_gt_fallback"
                ),
                source="gt_fallback",
                fallback_reason=(
                    f"{type(error).__name__}: {error}"
                ),
            )
            prepared_view = prepare_masked_view(
                view=loaded_view,
                segmentation=segmentation,
                output_directory=(
                    view_output / "prepared"
                ),
            )
        except Exception as fallback_error:
            raise RuntimeError(
                "Both SAM3 segmentation and GT-mask fallback failed: "
                f"sam3={type(error).__name__}: {error}; "
                "gt_fallback="
                f"{type(fallback_error).__name__}: {fallback_error}"
            ) from fallback_error

    selection_path = (
        view_output
        / "segmentation_selection.json"
    )
    selection_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    selection_path.write_text(
        json.dumps(
            {
                "view_name": view_name,
                "selected_source": segmentation.source,
                "gt_assisted": segmentation.source
                == "gt_fallback",
                "sam3_error": (
                    f"{type(sam3_error).__name__}: {sam3_error}"
                    if sam3_error is not None
                    else None
                ),
                "configured_gt_fallback": (
                    config.gt_mask_fallback_on_sam3_failure
                ),
                "gt_mask_path": (
                    str(selected_input.mask_path)
                    if segmentation.source
                    == "gt_fallback"
                    else None
                ),
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return prepared_view


def _extract_dino_inputs(
    *,
    config: PipelineConfig,
    prepared_views: Sequence[Any],
    output_roots: Sequence[Path],
) -> tuple[
    tuple[Any | None, Any | None],
    ...,
]:
    """
    관측 RGB-D별 DINOv3 dense feature와
    mask/depth 표면 feature를 생성한다.

    반환 tuple의 각 원소는
    (DINOFeatureResult, ObservedSurfaceFeatureResult)이다.
    """

    views = tuple(prepared_views)
    roots = tuple(output_roots)

    if len(views) != len(roots):
        raise ValueError(
            "prepared_views와 output_roots 개수가 다릅니다."
        )

    if not _pose_path_uses_dino(config):
        return tuple((None, None) for _ in views)

    from features.dinov3_extractor import (
        DINOv3Extractor,
    )
    from features.observed_surface_features import (
        build_observed_surface_features,
    )

    print(
        "[DINOv3] Extract dense features: "
        + ", ".join(
            view.view.source.name
            for view in views
        )
    )

    dense_results: list[Any] = []

    with DINOv3Extractor(
        repository_path=(
            config.dinov3_repository
        ),
        checkpoint_path=(
            config.dinov3_checkpoint
        ),
        model_name=config.dinov3_model,
        device=config.device,
        target_long_side=(
            config.dinov3_target_long_side
        ),
        use_amp=config.dinov3_use_amp,
        save_dtype=config.dinov3_save_dtype,
    ) as extractor:
        for prepared_view, output_root in zip(
            views,
            roots,
            strict=True,
        ):
            view_name = (
                prepared_view.view.source.name
            )
            feature_root = (
                Path(output_root)
                / "features"
                / "dinov3"
                / view_name
            )

            dense_results.append(
                extractor.extract(
                    view=prepared_view.view,
                    output_directory=(
                        feature_root / "dense"
                    ),
                )
            )

    results: list[tuple[Any, Any]] = []

    for (
        prepared_view,
        dense_result,
        output_root,
    ) in zip(
        views,
        dense_results,
        roots,
        strict=True,
    ):
        view_name = prepared_view.view.source.name
        feature_root = (
            Path(output_root)
            / "features"
            / "dinov3"
            / view_name
        )

        surface_result = build_observed_surface_features(
            prepared_view=prepared_view,
            dino_result=dense_result,
            output_directory=(
                feature_root / "surface"
            ),
            maximum_point_count=(
                config.dinov3_maximum_surface_points
            ),
            random_seed=config.random_seed,
            device=config.device,
            feature_chunk_size=(
                config.dinov3_feature_chunk_size
            ),
        )

        results.append(
            (
                dense_result,
                surface_result,
            )
        )

    return tuple(results)
