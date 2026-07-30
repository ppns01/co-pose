from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from core.types import (
    MeshGenerationResult,
    MeshGeneratorName,
)
from generators.base import (
    BaseMeshGenerator,
    MeshGenerationRequest,
)


class InstantMeshGenerator(BaseMeshGenerator):
    """
    공식 InstantMesh run.py를 별도 Python 프로세스로 실행한다.

    V1 설계:
    - 입력: SAM3 mask가 적용된 segmented_rgb.png
    - 출력: InstantMesh가 생성한 OBJ mesh
    - Reference와 Query를 순차적으로 처리
    - 각 실행 종료 후 subprocess의 GPU 메모리가 자동 반환됨
    - InstantMesh 내부 API를 직접 수정하지 않음
    """

    def __init__(
        self,
        repository_path: Path,
        python_executable: Path,
        config_path: Path | None = None,
        seed: int = 42,
        diffusion_steps: int = 75,
        view_count: int = 6,
        model_scale: float = 1.0,
        render_distance: float = 4.5,
        use_rembg: bool = True,
        export_texture_map: bool = False,
        save_video: bool = False,
        offline: bool = True,
    ) -> None:
        super().__init__()

        self._repository_path = (
            Path(repository_path)
            .expanduser()
            .resolve()
        )

        self._python_executable = (
            Path(python_executable)
            .expanduser()
            .resolve()
        )

        if config_path is None:
            config_path = (
                self._repository_path
                / "configs"
                / "instant-mesh-large.yaml"
            )

        self._config_path = (
            Path(config_path)
            .expanduser()
            .resolve()
        )

        self._seed = self._validate_integer(
            value=seed,
            name="seed",
            minimum=0,
        )

        self._diffusion_steps = self._validate_integer(
            value=diffusion_steps,
            name="diffusion_steps",
            minimum=1,
        )

        if view_count not in (4, 6):
            raise ValueError(
                "view_count는 4 또는 6이어야 합니다: "
                f"{view_count}"
            )

        self._view_count = view_count

        self._model_scale = self._validate_positive_float(
            value=model_scale,
            name="model_scale",
        )

        self._render_distance = self._validate_positive_float(
            value=render_distance,
            name="render_distance",
        )

        self._use_rembg = bool(use_rembg)
        self._export_texture_map = bool(export_texture_map)
        self._save_video = bool(save_video)
        self._offline = bool(offline)

    @property
    def name(self) -> MeshGeneratorName:
        return "instantmesh"

    @staticmethod
    def _validate_integer(
        value: int,
        name: str,
        minimum: int,
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name}은 정수여야 합니다: {value}"
            )

        if value < minimum:
            raise ValueError(
                f"{name}은 {minimum} 이상이어야 합니다: "
                f"{value}"
            )

        return value

    @staticmethod
    def _validate_positive_float(
        value: float,
        name: str,
    ) -> float:
        try:
            converted_value = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"{name}을 실수로 변환할 수 없습니다: "
                f"{value}"
            ) from error

        if converted_value <= 0.0:
            raise ValueError(
                f"{name}은 양수여야 합니다: "
                f"{converted_value}"
            )

        return converted_value

    def _load_model(self) -> None:
        """
        subprocess 실행에 필요한 파일만 확인한다.

        실제 모델은 InstantMesh run.py 내부에서 로드된다.
        """

        if not self._repository_path.is_dir():
            raise FileNotFoundError(
                "InstantMesh 저장소가 없습니다: "
                f"{self._repository_path}"
            )

        run_script_path = (
            self._repository_path / "run.py"
        )

        if not run_script_path.is_file():
            raise FileNotFoundError(
                "InstantMesh run.py가 없습니다: "
                f"{run_script_path}"
            )

        if not self._config_path.is_file():
            raise FileNotFoundError(
                "InstantMesh 설정 파일이 없습니다: "
                f"{self._config_path}"
            )

        if not self._python_executable.is_file():
            raise FileNotFoundError(
                "InstantMesh Python 실행 파일이 없습니다: "
                f"{self._python_executable}"
            )

    def _validate_request(
        self,
        request: MeshGenerationRequest,
    ) -> None:
        super()._validate_request(request)

        supported_suffixes = {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }

        input_suffix = (
            request.segmented_rgb_path
            .suffix
            .lower()
        )

        if input_suffix not in supported_suffixes:
            raise ValueError(
                "InstantMesh가 지원하지 않는 입력 이미지 "
                "형식입니다: "
                f"{request.segmented_rgb_path}"
            )

        if request.mask_bool_path is None:
            raise ValueError(
                "InstantMesh trusted-alpha 입력에는 "
                "mask_bool_path가 필요합니다."
            )

    def _prepare_trusted_alpha_input(
        self,
        request: MeshGenerationRequest,
    ) -> Path:
        """
        SAM3 mask를 alpha로 넣은 RGBA PNG를 만든다.

        공식 InstantMesh 전처리는 유효 alpha가 있으면 rembg 추론을
        건너뛰고 foreground crop/resize는 그대로 수행한다.
        """

        if request.mask_bool_path is None:
            raise ValueError(
                "mask_bool_path가 필요합니다."
            )

        with Image.open(
            request.segmented_rgb_path
        ) as image:
            image_rgb = np.asarray(
                image.convert("RGB"),
                dtype=np.uint8,
            )

        mask_bool = np.asarray(
            np.load(
                request.mask_bool_path,
                allow_pickle=False,
            ),
            dtype=np.bool_,
        )

        if mask_bool.shape != image_rgb.shape[:2]:
            raise ValueError(
                "Segmented RGB와 SAM3 mask 해상도가 다릅니다: "
                f"rgb={image_rgb.shape[:2]}, "
                f"mask={mask_bool.shape}"
            )

        if not np.any(mask_bool):
            raise ValueError(
                "SAM3 mask에 foreground가 없습니다."
            )

        if np.all(mask_bool):
            raise ValueError(
                "SAM3 mask가 전체 이미지를 덮어 trusted alpha로 "
                "사용할 수 없습니다."
            )

        alpha = (
            mask_bool.astype(np.uint8)
            * np.uint8(255)
        )
        image_rgba = np.dstack(
            (image_rgb, alpha)
        )

        trusted_input_path = (
            request.output_directory
            / "input"
            / request.segmented_rgb_path.name
        ).resolve()
        trusted_input_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        Image.fromarray(
            image_rgba,
            mode="RGBA",
        ).save(trusted_input_path)

        return trusted_input_path

    def _build_command(
        self,
        request: MeshGenerationRequest,
        input_image_path: Path | None = None,
    ) -> list[str]:
        run_script_path = (
            self._repository_path / "run.py"
        )

        resolved_input_path = (
            request.segmented_rgb_path
            if input_image_path is None
            else input_image_path
        ).resolve()

        command = [
            str(self._python_executable),
            str(run_script_path),
            str(self._config_path),
            str(resolved_input_path),
            "--output_path",
            str(request.output_directory.resolve()),
            "--diffusion_steps",
            str(self._diffusion_steps),
            "--seed",
            str(self._seed),
            "--scale",
            str(self._model_scale),
            "--distance",
            str(self._render_distance),
            "--view",
            str(self._view_count),
        ]

        # use_rembg=False이면 공식 run.py의 rembg 처리를 생략한다.
        if not self._use_rembg:
            command.append("--no_rembg")

        if self._export_texture_map:
            command.append("--export_texmap")

        if self._save_video:
            command.append("--save_video")

        return command

    def _build_environment(self) -> dict[str, str]:
        environment = os.environ.copy()

        # 여러 GPU가 있어도 첫 번째 GPU만 사용한다.
        environment.setdefault(
            "CUDA_VISIBLE_DEVICES",
            "0",
        )

        # rembg/Numba가 오래된 system TBB를 발견해도
        # 별도 shell 설정 없이 안전한 threading layer를 사용한다.
        environment.setdefault(
            "NUMBA_THREADING_LAYER",
            "workqueue",
        )

        if self._offline:
            # 로컬 캐시에 없는 모델을 외부에서 받지 않도록 한다.
            environment["HF_HUB_OFFLINE"] = "1"
            environment["TRANSFORMERS_OFFLINE"] = "1"
            environment["DIFFUSERS_OFFLINE"] = "1"

        return environment

    def _run_instantmesh(
        self,
        command: list[str],
    ) -> None:
        try:
            subprocess.run(
                command,
                cwd=self._repository_path,
                env=self._build_environment(),
                check=True,
            )

        except subprocess.CalledProcessError as error:
            command_text = subprocess.list2cmdline(
                command
            )

            raise RuntimeError(
                "InstantMesh 실행에 실패했습니다.\n"
                f"종료 코드: {error.returncode}\n"
                f"실행 명령: {command_text}"
            ) from error

        except OSError as error:
            raise RuntimeError(
                "InstantMesh 프로세스를 시작하지 못했습니다: "
                f"{error}"
            ) from error

    def _get_config_name(self) -> str:
        return self._config_path.stem

    def _get_expected_mesh_path(
        self,
        request: MeshGenerationRequest,
    ) -> Path:
        """
        공식 run.py의 기본 OBJ 출력 위치를 계산한다.

        output_directory/
        └── config_name/
            └── meshes/
                └── input_name.obj
        """

        input_name = (
            request.segmented_rgb_path.stem
        )

        return (
            request.output_directory
            / self._get_config_name()
            / "meshes"
            / f"{input_name}.obj"
        ).resolve()

    def _collect_artifacts(
        self,
        request: MeshGenerationRequest,
        primary_mesh_path: Path,
    ) -> tuple[Path, ...]:
        """
        OBJ 외에 생성된 MTL, texture, multiview image,
        video 등을 수집한다.
        """

        config_output_directory = (
            request.output_directory
            / self._get_config_name()
        ).resolve()

        if not config_output_directory.is_dir():
            return ()

        artifact_paths = tuple(
            sorted(
                path.resolve()
                for path
                in config_output_directory.rglob("*")
                if (
                    path.is_file()
                    and path.resolve()
                    != primary_mesh_path.resolve()
                )
            )
        )

        return artifact_paths

    def _generate_mesh(
        self,
        request: MeshGenerationRequest,
    ) -> MeshGenerationResult:
        request.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        expected_mesh_path = (
            self._get_expected_mesh_path(request)
        )

        # 이전 결과를 새 결과로 오인하지 않도록 삭제한다.
        if expected_mesh_path.is_file():
            expected_mesh_path.unlink()

        trusted_input_path = (
            self._prepare_trusted_alpha_input(
                request
            )
        )

        command = self._build_command(
            request=request,
            input_image_path=trusted_input_path,
        )

        self._run_instantmesh(
            command=command,
        )

        if not expected_mesh_path.is_file():
            raise FileNotFoundError(
                "InstantMesh 실행은 종료됐지만 "
                "예상한 OBJ mesh가 생성되지 않았습니다: "
                f"{expected_mesh_path}"
            )

        artifact_paths = self._collect_artifacts(
            request=request,
            primary_mesh_path=expected_mesh_path,
        )
        artifact_paths = (
            trusted_input_path,
            *artifact_paths,
        )

        return MeshGenerationResult(
            generator_name=self.name,
            output_dir=(
                request.output_directory.resolve()
            ),
            primary_output_path=expected_mesh_path,
            artifact_paths=artifact_paths,
            metadata_path=None,
        )

    def _release_model(self) -> None:
        """
        실제 모델은 subprocess 종료와 함께 해제된다.

        현재 프로세스에서 해제할 CUDA 객체는 없다.
        """

        return None
