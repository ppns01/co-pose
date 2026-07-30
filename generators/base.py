from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from core.types import (
    MeshGenerationResult,
    MeshGeneratorName,
    ViewName,
)


@dataclass(frozen=True)
class MeshGenerationRequest:
    """
    3D mesh 생성기의 공통 입력.

    segmented_rgb_path:
        SAM3 mask를 적용한 RGB 이미지.
        모든 생성기의 필수 입력이다.

    mask_bool_path, mask_rgb_path:
        생성기가 원래 객체 경계를 보존해야 할 때 사용한다.
        InstantMesh adapter는 Boolean mask를 RGBA alpha로 전달한다.
    """

    view_name: ViewName

    segmented_rgb_path: Path
    output_directory: Path

    mask_bool_path: Path | None = None
    mask_rgb_path: Path | None = None

    def __post_init__(self) -> None:
        if self.view_name not in (
            "reference",
            "query",
        ):
            raise ValueError(
                "지원하지 않는 view 이름입니다: "
                f"{self.view_name}"
            )

        if not self.segmented_rgb_path.is_file():
            raise FileNotFoundError(
                "Segmented RGB 파일이 없습니다: "
                f"{self.segmented_rgb_path}"
            )

        if (
            self.mask_bool_path is not None
            and not self.mask_bool_path.is_file()
        ):
            raise FileNotFoundError(
                "Boolean mask 파일이 없습니다: "
                f"{self.mask_bool_path}"
            )

        if (
            self.mask_rgb_path is not None
            and not self.mask_rgb_path.is_file()
        ):
            raise FileNotFoundError(
                "RGB mask 파일이 없습니다: "
                f"{self.mask_rgb_path}"
            )


class BaseMeshGenerator(ABC):
    """
    InstantMesh, TRELLIS, SAM3D 공통 기반 클래스.

    모델은 첫 generate() 호출에서 한 번만 로드한다.
    Reference와 Query를 연속으로 생성한 뒤 release()를 호출한다.
    """

    def __init__(self) -> None:
        self._is_loaded = False

    @property
    @abstractmethod
    def name(self) -> MeshGeneratorName:
        """생성기 이름을 반환한다."""

        raise NotImplementedError

    @property
    def is_loaded(self) -> bool:
        """모델이 현재 메모리에 로드되어 있는지 반환한다."""

        return self._is_loaded

    def load(self) -> None:
        """
        생성기 모델을 메모리에 로드한다.

        이미 로드된 경우 다시 로드하지 않는다.
        """

        if self._is_loaded:
            return

        self._load_model()
        self._is_loaded = True

    def generate(
        self,
        request: MeshGenerationRequest,
    ) -> MeshGenerationResult:
        """
        Segmented RGB로 proxy mesh를 생성한다.

        모델이 로드되지 않았다면 자동으로 로드한다.
        생성 후 모델은 유지되므로 Query 생성에 재사용할 수 있다.
        """

        self._validate_request(request)

        request.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.load()

        result = self._generate_mesh(request)

        self._validate_result(
            request=request,
            result=result,
        )

        return result

    def release(self) -> None:
        """
        모델과 생성기 내부 자원을 해제한다.

        RTX 5060 Ti 16GB 환경에서는 Reference와 Query mesh를
        모두 생성한 직후 호출해야 한다.
        """

        if not self._is_loaded:
            return

        try:
            self._release_model()
        finally:
            self._is_loaded = False

    def _validate_request(
        self,
        request: MeshGenerationRequest,
    ) -> None:
        """
        공통 입력을 검증한다.

        생성기별 추가 검증이 필요하면 하위 클래스에서
        이 메서드를 override할 수 있다.
        """

        if not request.segmented_rgb_path.is_file():
            raise FileNotFoundError(
                "Segmented RGB 파일이 없습니다: "
                f"{request.segmented_rgb_path}"
            )

    def _validate_result(
        self,
        request: MeshGenerationRequest,
        result: MeshGenerationResult,
    ) -> None:
        """생성기가 반환한 결과를 검증한다."""

        if result.generator_name != self.name:
            raise ValueError(
                "생성기 이름과 결과 이름이 다릅니다: "
                f"generator={self.name}, "
                f"result={result.generator_name}"
            )

        if not result.output_dir.is_dir():
            raise FileNotFoundError(
                "생성기 출력 폴더가 없습니다: "
                f"{result.output_dir}"
            )

        if not result.primary_output_path.is_file():
            raise FileNotFoundError(
                "대표 mesh 출력 파일이 없습니다: "
                f"{result.primary_output_path}"
            )

        for artifact_path in result.artifact_paths:
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    "생성기 부가 출력 파일이 없습니다: "
                    f"{artifact_path}"
                )

        if (
            result.metadata_path is not None
            and not result.metadata_path.is_file()
        ):
            raise FileNotFoundError(
                "생성기 metadata 파일이 없습니다: "
                f"{result.metadata_path}"
            )

        expected_output_directory = (
            request.output_directory.resolve()
        )

        actual_output_directory = (
            result.output_dir.resolve()
        )

        if actual_output_directory != expected_output_directory:
            raise ValueError(
                "요청한 출력 폴더와 실제 출력 폴더가 다릅니다: "
                f"expected={expected_output_directory}, "
                f"actual={actual_output_directory}"
            )

    @abstractmethod
    def _load_model(self) -> None:
        """생성기별 모델을 로드한다."""

        raise NotImplementedError

    @abstractmethod
    def _generate_mesh(
        self,
        request: MeshGenerationRequest,
    ) -> MeshGenerationResult:
        """생성기별 실제 mesh 생성을 수행한다."""

        raise NotImplementedError

    def _release_model(self) -> None:
        """
        생성기별 모델과 GPU 메모리를 해제한다.

        GPU 모델을 사용하는 하위 클래스에서는 반드시 override한다.
        """

    def __enter__(self) -> BaseMeshGenerator:
        """with 문 진입 시 모델을 로드한다."""

        self.load()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        """with 문 종료 시 모델을 해제한다."""

        self.release()
