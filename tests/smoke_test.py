from __future__ import annotations

import importlib
import py_compile
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModuleCheck:
    """검사할 내부 Python 모듈과 필수 symbol."""

    module_name: str
    required_symbols: tuple[str, ...] = ()


MODULE_CHECKS = (
    ModuleCheck(
        module_name="main",
        required_symbols=(
            "PipelineConfig",
            "run_pipeline",
        ),
    ),
    ModuleCheck(
        module_name="input_selector",
        required_symbols=(
            "build_view_input",
        ),
    ),
    ModuleCheck(
        module_name="mask_provider",
        required_symbols=(
            "load_existing_segmentation",
        ),
    ),
    ModuleCheck(
        module_name="core.types",
        required_symbols=(
            "ViewInput",
            "LoadedView",
            "SegmentationResult",
            "PreparedView",
            "MeshGenerationResult",
            "ScaleInitializationResult",
        ),
    ),
    ModuleCheck(
        module_name="dataset_io.linemod_loader",
        required_symbols=(
            "load_linemod_view",
        ),
    ),
    ModuleCheck(
        module_name="segmentation.sam3_masker",
        required_symbols=(
            "SAM3Masker",
        ),
    ),
    ModuleCheck(
        module_name="preprocessing.mask_processing",
        required_symbols=(
            "prepare_masked_view",
        ),
    ),
    ModuleCheck(
        module_name="generators.base",
        required_symbols=(
            "MeshGenerationRequest",
            "BaseMeshGenerator",
        ),
    ),
    ModuleCheck(
        module_name="generators.instantmesh_generator",
        required_symbols=(
            "InstantMeshGenerator",
        ),
    ),
    ModuleCheck(
        module_name="scale.depth_scale_initializer",
        required_symbols=(
            "initialize_scale_from_depth",
        ),
    ),
    ModuleCheck(
        module_name="scale.mesh_normalizer",
        required_symbols=(
            "MeshNormalizationResult",
            "normalize_generated_mesh",
        ),
    ),
    ModuleCheck(
        module_name="scale.mesh_scaler",
        required_symbols=(
            "ScaledMeshCandidate",
            "build_scaled_mesh_candidates",
        ),
    ),
    ModuleCheck(
        module_name="scale.local_scale_refiner",
        required_symbols=(
            "LocalScaleRefinementResult",
            "refine_self_scale_locally",
        ),
    ),
    ModuleCheck(
        module_name="pose.foundationpose_runner",
        required_symbols=(
            "FoundationPoseRunner",
            "FoundationPoseHypothesis",
            "FoundationPoseCandidateResult",
        ),
    ),
    ModuleCheck(
        module_name="pose.mesh_renderer",
        required_symbols=(
            "FoundationPoseMeshRenderer",
            "MeshRenderResult",
        ),
    ),
    ModuleCheck(
        module_name="pose.alignment_scorer",
        required_symbols=(
            "AlignmentScoreResult",
            "AlignmentScoreWeights",
            "score_alignment",
        ),
    ),
    ModuleCheck(
        module_name="pose.alignment_evaluator",
        required_symbols=(
            "AlignmentEvaluationResult",
            "evaluate_foundationpose_alignments",
            "select_best_self_alignment",
        ),
    ),
    ModuleCheck(
        module_name="pose.relative_pose_builder",
        required_symbols=(
            "SelfAlignmentSelection",
            "select_self_alignment",
        ),
    ),
    ModuleCheck(
        module_name="features.dinov3_extractor",
        required_symbols=(
            "DINOv3Extractor",
            "DINOFeatureResult",
        ),
    ),
    ModuleCheck(
        module_name="features.observed_surface_features",
        required_symbols=(
            "ObservedSurfaceFeatureResult",
            "build_observed_surface_features",
        ),
    ),
    ModuleCheck(
        module_name="features.dino_pose_scorer",
        required_symbols=(
            "BidirectionalDINOScore",
            "score_bidirectional_dino",
        ),
    ),
    ModuleCheck(
        module_name="evaluation.relative_pose_evaluator",
        required_symbols=(
            "BOPFrameGTSpec",
            "RelativePoseEvaluationResult",
            "evaluate_relative_pose",
        ),
    ),
)


SOURCE_DIRECTORIES = (
    "core",
    "dataset_io",
    "segmentation",
    "preprocessing",
    "generators",
    "scale",
    "pose",
    "features",
    "evaluation",
    "tests",
)

ROOT_PYTHON_FILES = (
    "main.py",
    "input_selector.py",
    "mask_provider.py",
)


def find_python_files() -> tuple[Path, ...]:
    """프로젝트 내부 Python 파일을 찾는다."""

    python_files: list[Path] = []

    python_files.extend(
        file_path
        for file_name in ROOT_PYTHON_FILES
        if (
            file_path := PROJECT_ROOT / file_name
        ).is_file()
    )

    for directory_name in SOURCE_DIRECTORIES:
        directory_path = PROJECT_ROOT / directory_name

        if not directory_path.is_dir():
            continue

        python_files.extend(
            path
            for path in directory_path.rglob("*.py")
            if "__pycache__" not in path.parts
        )

    return tuple(
        sorted(python_files)
    )


def check_python_syntax() -> list[str]:
    """모든 Python 파일의 문법을 검사한다."""

    errors: list[str] = []

    for python_file in find_python_files():
        try:
            py_compile.compile(
                str(python_file),
                doraise=True,
            )

            relative_path = python_file.relative_to(
                PROJECT_ROOT
            )

            print(
                f"[문법 정상] {relative_path}"
            )

        except py_compile.PyCompileError as error:
            errors.append(
                f"{python_file}: {error.msg}"
            )

    return errors


def check_module_imports() -> list[str]:
    """내부 모듈 import와 필수 symbol을 검사한다."""

    errors: list[str] = []

    for module_check in MODULE_CHECKS:
        try:
            module = importlib.import_module(
                module_check.module_name
            )

        except Exception as error:
            errors.append(
                f"{module_check.module_name}: "
                f"{type(error).__name__}: {error}"
            )
            continue

        missing_symbols = [
            symbol
            for symbol in module_check.required_symbols
            if not hasattr(module, symbol)
        ]

        if missing_symbols:
            errors.append(
                f"{module_check.module_name}: "
                f"필수 symbol 없음: {missing_symbols}"
            )
            continue

        print(
            f"[Import 정상] "
            f"{module_check.module_name}"
        )

    return errors


def check_core_data_contracts() -> list[str]:
    """
    실제 모델 없이 core 자료형 계약을 간단히 검사한다.
    """

    errors: list[str] = []

    try:
        from core.types import (
            LoadedView,
            PreparedView,
            SegmentationResult,
            ViewInput,
        )

        height = 32
        width = 48

        rgb = np.zeros(
            (height, width, 3),
            dtype=np.uint8,
        )

        depth_m = np.ones(
            (height, width),
            dtype=np.float32,
        )

        camera_matrix = np.array(
            [
                [500.0, 0.0, width / 2.0],
                [0.0, 500.0, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        mask_bool = np.zeros(
            (height, width),
            dtype=np.bool_,
        )

        mask_bool[
            height // 4 : height * 3 // 4,
            width // 4 : width * 3 // 4,
        ] = True

        mask_rgb = np.repeat(
            (
                mask_bool.astype(np.uint8)
                * np.uint8(255)
            )[:, :, None],
            repeats=3,
            axis=2,
        )

        segmented_rgb = np.zeros_like(
            rgb,
            dtype=np.uint8,
        )

        segmented_rgb[mask_bool] = (
            rgb[mask_bool]
        )

        masked_depth_m = np.zeros_like(
            depth_m,
            dtype=np.float32,
        )

        masked_depth_m[mask_bool] = (
            depth_m[mask_bool]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)

            rgb_path = temp_root / "rgb.png"
            depth_path = temp_root / "depth.png"
            intrinsics_path = (
                temp_root / "scene_camera.json"
            )

            mask_bool_path = (
                temp_root / "mask_bool.npy"
            )

            mask_rgb_path = (
                temp_root / "mask_rgb.png"
            )

            segmented_rgb_path = (
                temp_root / "segmented_rgb.png"
            )

            masked_depth_path = (
                temp_root / "masked_depth_m.npy"
            )

            for file_path in (
                rgb_path,
                depth_path,
                intrinsics_path,
                mask_rgb_path,
                segmented_rgb_path,
            ):
                file_path.touch()

            np.save(
                mask_bool_path,
                mask_bool,
                allow_pickle=False,
            )

            np.save(
                masked_depth_path,
                masked_depth_m,
                allow_pickle=False,
            )

            view_input = ViewInput(
                name="reference",
                rgb_path=rgb_path,
                depth_path=depth_path,
                intrinsics_path=intrinsics_path,
                object_name="driller",
                scene_id=1,
                image_id=0,
                object_id=1,
            )

            loaded_view = LoadedView(
                source=view_input,
                rgb=rgb,
                depth_m=depth_m,
                camera_matrix=camera_matrix,
                depth_scale_to_m=0.001,
            )

            segmentation = SegmentationResult(
                mask_bool=mask_bool,
                mask_rgb=mask_rgb,
                mask_bool_path=mask_bool_path,
                mask_rgb_path=mask_rgb_path,
            )

            PreparedView(
                view=loaded_view,
                segmentation=segmentation,
                segmented_rgb=segmented_rgb,
                masked_depth_m=masked_depth_m,
                segmented_rgb_path=(
                    segmented_rgb_path
                ),
                masked_depth_path=(
                    masked_depth_path
                ),
            )

        print(
            "[자료형 정상] "
            "ViewInput → LoadedView → "
            "SegmentationResult → PreparedView"
        )

    except Exception as error:
        errors.append(
            "core 자료형 계약 실패: "
            f"{type(error).__name__}: {error}"
        )

    return errors


def print_error_section(
    section_name: str,
    errors: list[str],
) -> None:
    """오류 목록을 출력한다."""

    if not errors:
        return

    print(f"\n[{section_name} 오류]")

    for error_index, error in enumerate(
        errors,
        start=1,
    ):
        print(
            f"{error_index}. {error}"
        )


def main() -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(
            0,
            str(PROJECT_ROOT),
        )

    print(
        f"[프로젝트 루트] {PROJECT_ROOT}\n"
    )

    syntax_errors = check_python_syntax()

    import_errors = check_module_imports()

    contract_errors = check_core_data_contracts()

    print_error_section(
        "문법",
        syntax_errors,
    )

    print_error_section(
        "Import",
        import_errors,
    )

    print_error_section(
        "자료형 계약",
        contract_errors,
    )

    total_error_count = (
        len(syntax_errors)
        + len(import_errors)
        + len(contract_errors)
    )

    if total_error_count > 0:
        print(
            f"\n[검사 실패] "
            f"총 오류 수: {total_error_count}"
        )

        return 1

    print(
        "\n[검사 완료] "
        "현재 내부 모듈의 문법, import, "
        "기본 자료형 계약이 정상입니다."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
