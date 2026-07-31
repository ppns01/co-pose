from __future__ import annotations

import gc
import importlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Sequence,
)

import numpy as np
from numpy.typing import NDArray

from core.types import PreparedView, ViewName

if TYPE_CHECKING:
    from scale.mesh_scaler import (
        ScaledMeshCandidate,
    )


@dataclass(frozen=True)
class FoundationPoseHypothesis:
    """
    FoundationPose pose 후보 하나.

    pose_cam_from_proxy:
        T_camera_from_proxy

        입력 scaled mesh의 원래 local 좌표를
        camera 좌표로 변환한다.
    """

    rank: int
    score: float
    pose_cam_from_proxy: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError(
                f"rank는 0 이상이어야 합니다: {self.rank}"
            )

        if not np.isfinite(self.score):
            raise ValueError(
                f"FoundationPose score가 유한하지 않습니다: {self.score}"
            )

        _validate_rigid_pose(
            self.pose_cam_from_proxy,
            "pose_cam_from_proxy",
        )


@dataclass(frozen=True)
class FoundationPoseCandidateResult:
    """
    Scale candidate 하나에 대한 FoundationPose top-K 결과.
    """

    view_name: ViewName

    candidate_index: int
    scale_m: float
    scaled_mesh_path: Path

    hypotheses: tuple[
        FoundationPoseHypothesis,
        ...
    ]

    output_directory: Path
    metadata_path: Path

    def __post_init__(self) -> None:
        if self.view_name not in (
            "reference",
            "query",
        ):
            raise ValueError(
                f"지원하지 않는 view입니다: {self.view_name}"
            )

        if self.candidate_index < 0:
            raise ValueError(
                "candidate_index는 0 이상이어야 합니다: "
                f"{self.candidate_index}"
            )

        if (
            not np.isfinite(self.scale_m)
            or self.scale_m <= 0.0
        ):
            raise ValueError(
                f"scale_m은 유한한 양수여야 합니다: {self.scale_m}"
            )

        if not self.scaled_mesh_path.is_file():
            raise FileNotFoundError(
                "Scaled mesh가 없습니다: "
                f"{self.scaled_mesh_path}"
            )

        if not self.hypotheses:
            raise ValueError(
                "FoundationPose hypothesis가 없습니다."
            )

        expected_ranks = list(
            range(len(self.hypotheses))
        )

        actual_ranks = [
            hypothesis.rank
            for hypothesis in self.hypotheses
        ]

        if actual_ranks != expected_ranks:
            raise ValueError(
                "FoundationPose hypothesis rank가 "
                "연속적이지 않습니다: "
                f"expected={expected_ranks}, "
                f"actual={actual_ranks}"
            )

        if not self.output_directory.is_dir():
            raise FileNotFoundError(
                "FoundationPose 출력 폴더가 없습니다: "
                f"{self.output_directory}"
            )

        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                "FoundationPose metadata가 없습니다: "
                f"{self.metadata_path}"
            )


def _validate_rigid_pose(
    pose: NDArray[np.floating],
    name: str,
) -> NDArray[np.float32]:
    """4×4 SE(3) pose를 검증한다."""

    pose_array = np.asarray(
        pose,
        dtype=np.float32,
    )

    if pose_array.shape != (4, 4):
        raise ValueError(
            f"{name} shape은 (4, 4)이어야 합니다: "
            f"{pose_array.shape}"
        )

    if not np.isfinite(pose_array).all():
        raise ValueError(
            f"{name}에 NaN 또는 Inf가 있습니다."
        )

    expected_last_row = np.array(
        [0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )

    if not np.allclose(
        pose_array[3],
        expected_last_row,
        atol=1e-5,
        rtol=0.0,
    ):
        raise ValueError(
            f"{name}의 마지막 행이 올바르지 않습니다."
        )

    rotation = pose_array[:3, :3]

    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3, dtype=np.float32),
        atol=1e-3,
        rtol=0.0,
    ):
        raise ValueError(
            f"{name}의 회전 행렬이 직교하지 않습니다."
        )

    determinant = float(
        np.linalg.det(rotation)
    )

    if not np.isclose(
        determinant,
        1.0,
        atol=1e-3,
        rtol=0.0,
    ):
        raise ValueError(
            f"{name}의 회전 determinant가 1이 아닙니다: "
            f"{determinant}"
        )

    return np.ascontiguousarray(
        pose_array,
        dtype=np.float32,
    )


def _sanitize_rigid_pose(
    pose: Any,
    name: str,
) -> NDArray[np.float32]:
    """
    FoundationPose 출력의 작은 수치 오차만 SO(3)로 보정한다.

    큰 회전 행렬 오류는 보정하지 않고 예외를 발생시킨다.
    """

    pose_array = np.asarray(
        pose,
        dtype=np.float64,
    )

    if pose_array.shape != (4, 4):
        raise ValueError(
            f"{name} shape은 (4, 4)이어야 합니다: "
            f"{pose_array.shape}"
        )

    if not np.isfinite(pose_array).all():
        raise ValueError(
            f"{name}에 NaN 또는 Inf가 있습니다."
        )

    rotation = pose_array[:3, :3]

    orthogonality_error = float(
        np.linalg.norm(
            rotation.T @ rotation
            - np.eye(3, dtype=np.float64)
        )
    )

    determinant = float(
        np.linalg.det(rotation)
    )

    if (
        orthogonality_error > 0.05
        or abs(abs(determinant) - 1.0) > 0.05
    ):
        raise ValueError(
            f"{name}의 회전 행렬 오차가 너무 큽니다: "
            f"orthogonality_error={orthogonality_error}, "
            f"determinant={determinant}"
        )

    u_matrix, _, vh_matrix = np.linalg.svd(
        rotation
    )

    projected_rotation = (
        u_matrix @ vh_matrix
    )

    if np.linalg.det(projected_rotation) < 0.0:
        u_matrix[:, -1] *= -1.0

        projected_rotation = (
            u_matrix @ vh_matrix
        )

    cleaned_pose = np.eye(
        4,
        dtype=np.float64,
    )

    cleaned_pose[:3, :3] = (
        projected_rotation
    )

    cleaned_pose[:3, 3] = (
        pose_array[:3, 3]
    )

    return _validate_rigid_pose(
        cleaned_pose,
        name,
    )


def _to_numpy(
    value: Any,
) -> NDArray[np.generic]:
    """Torch tensor 또는 numpy 배열을 numpy로 변환한다."""

    if isinstance(value, np.ndarray):
        return value

    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        return value.numpy()

    return np.asarray(value)


@contextmanager
def _temporary_working_directory(
    directory: Path,
) -> Iterator[None]:
    """FoundationPose 초기화 중에만 repository를 cwd로 사용한다."""

    previous_directory = Path.cwd()

    os.chdir(directory)

    try:
        yield
    finally:
        os.chdir(previous_directory)


def _is_path_inside(
    child_path: Path,
    parent_path: Path,
) -> bool:
    """child_path가 parent_path 내부인지 검사한다."""

    try:
        child_path.relative_to(parent_path)
        return True

    except ValueError:
        return False


class FoundationPoseRunner:
    """
    NVlabs FoundationPose model-based pose estimator adapter.

    외부 패키지는 module import 시 로드하지 않는다.
    실제 load() 또는 run_candidate() 호출 시에만
    FoundationPose와 CUDA 모델을 로드한다.
    """

    def __init__(
        self,
        repository_path: Path,
        output_root: Path,
        *,
        top_k: int = 3,
        refine_iterations: int = 5,
        debug: int = 0,
        device: str = "cuda:0",
    ) -> None:
        self._repository_path = (
            Path(repository_path)
            .expanduser()
            .resolve()
        )

        self._output_root = (
            Path(output_root)
            .expanduser()
            .resolve()
        )

        if (
            isinstance(top_k, bool)
            or top_k < 1
        ):
            raise ValueError(
                f"top_k는 1 이상이어야 합니다: {top_k}"
            )

        if (
            isinstance(refine_iterations, bool)
            or refine_iterations < 1
        ):
            raise ValueError(
                "refine_iterations는 1 이상이어야 합니다: "
                f"{refine_iterations}"
            )

        if (
            isinstance(debug, bool)
            or debug < 0
        ):
            raise ValueError(
                f"debug는 0 이상이어야 합니다: {debug}"
            )

        if not device.startswith("cuda"):
            raise ValueError(
                "FoundationPose 공식 구현은 CUDA를 필요로 합니다: "
                f"{device}"
            )

        self._top_k = int(top_k)
        self._refine_iterations = int(
            refine_iterations
        )
        self._debug = int(debug)
        self._device_string = device

        self._torch: ModuleType | None = None
        self._trimesh: ModuleType | None = None
        self._dr: ModuleType | None = None
        self._estimater_module: ModuleType | None = None

        self._foundationpose_class: Any | None = None
        self._scorer: Any | None = None
        self._refiner: Any | None = None
        self._glctx: Any | None = None
        self._estimator: Any | None = None

        self._repository_added_to_path = False
        self._current_mesh_path: Path | None = None

    @property
    def is_loaded(self) -> bool:
        return (
            self._scorer is not None
            and self._refiner is not None
            and self._glctx is not None
        )

    def _validate_repository(self) -> None:
        """FoundationPose 공식 repository 구조를 확인한다."""

        if not self._repository_path.is_dir():
            raise FileNotFoundError(
                "FoundationPose repository가 없습니다: "
                f"{self._repository_path}"
            )

        required_paths = (
            self._repository_path / "estimater.py",
            self._repository_path / "Utils.py",
            self._repository_path / "weights",
        )

        for required_path in required_paths:
            if not required_path.exists():
                raise FileNotFoundError(
                    "FoundationPose 필수 파일 또는 폴더가 "
                    "없습니다: "
                    f"{required_path}"
                )

    def _check_existing_module_collision(
        self,
        module_name: str,
    ) -> None:
        """동일 이름의 다른 Python module 충돌을 검사한다."""

        existing_module = sys.modules.get(
            module_name
        )

        if existing_module is None:
            return

        module_file = getattr(
            existing_module,
            "__file__",
            None,
        )

        if module_file is None:
            return

        resolved_module_file = (
            Path(module_file)
            .expanduser()
            .resolve()
        )

        if not _is_path_inside(
            resolved_module_file,
            self._repository_path,
        ):
            raise RuntimeError(
                "FoundationPose import 이름과 충돌하는 "
                "다른 module이 이미 로드되어 있습니다: "
                f"module={module_name}, "
                f"path={resolved_module_file}"
            )

    def _import_estimater_module(
        self,
    ) -> ModuleType:
        """공식 repository의 estimater.py를 import한다."""

        for module_name in (
            "estimater",
            "Utils",
            "datareader",
        ):
            self._check_existing_module_collision(
                module_name
            )

        repository_string = str(
            self._repository_path
        )

        if repository_string not in sys.path:
            sys.path.insert(
                0,
                repository_string,
            )

            self._repository_added_to_path = True

        importlib.invalidate_caches()

        with _temporary_working_directory(
            self._repository_path
        ):
            try:
                estimater_module = (
                    importlib.import_module(
                        "estimater"
                    )
                )

            except Exception as error:
                raise ImportError(
                    "FoundationPose estimater.py를 import하지 "
                    "못했습니다. build_all.sh, mycpp, "
                    "nvdiffrast 및 Python 환경을 확인해야 합니다."
                ) from error

        module_file = getattr(
            estimater_module,
            "__file__",
            None,
        )

        if module_file is None:
            raise ImportError(
                "Import된 estimater module의 경로를 "
                "확인할 수 없습니다."
            )

        resolved_module_file = (
            Path(module_file)
            .expanduser()
            .resolve()
        )

        if not _is_path_inside(
            resolved_module_file,
            self._repository_path,
        ):
            raise ImportError(
                "다른 위치의 estimater module이 import되었습니다: "
                f"{resolved_module_file}"
            )

        required_symbols = (
            "FoundationPose",
            "ScorePredictor",
            "PoseRefinePredictor",
        )

        for symbol_name in required_symbols:
            if not hasattr(
                estimater_module,
                symbol_name,
            ):
                raise AttributeError(
                    "FoundationPose estimater.py에 필수 "
                    f"symbol이 없습니다: {symbol_name}"
                )

        return estimater_module

    def load(self) -> None:
        """FoundationPose scorer, refiner 및 renderer를 로드한다."""

        if self.is_loaded:
            return

        self._validate_repository()

        try:
            import torch
            import trimesh
            import nvdiffrast.torch as dr

        except Exception as error:
            raise ImportError(
                "FoundationPose 실행에 필요한 torch, trimesh "
                "또는 nvdiffrast를 import하지 못했습니다."
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch에서 CUDA를 사용할 수 없습니다."
            )

        cuda_device = torch.device(
            self._device_string
        )

        torch.cuda.set_device(
            cuda_device
        )

        self._torch = torch
        self._trimesh = trimesh
        self._dr = dr

        estimater_module = (
            self._import_estimater_module()
        )

        self._estimater_module = (
            estimater_module
        )

        self._foundationpose_class = (
            estimater_module.FoundationPose
        )

        try:
            with _temporary_working_directory(
                self._repository_path
            ):
                self._scorer = (
                    estimater_module.ScorePredictor()
                )

                self._refiner = (
                    estimater_module.PoseRefinePredictor()
                )

                try:
                    self._glctx = (
                        dr.RasterizeCudaContext(
                            device=cuda_device
                        )
                    )

                except TypeError:
                    self._glctx = (
                        dr.RasterizeCudaContext()
                    )

        except Exception as error:
            self.release()

            raise RuntimeError(
                "FoundationPose scorer/refiner 또는 "
                "nvdiffrast context 초기화에 실패했습니다."
            ) from error

        self._output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _load_mesh(
        self,
        mesh_path: Path,
    ) -> Any:
        """Scaled proxy mesh를 Trimesh로 읽는다."""

        if self._trimesh is None:
            raise RuntimeError(
                "FoundationPoseRunner가 로드되지 않았습니다."
            )

        mesh_path = (
            Path(mesh_path)
            .expanduser()
            .resolve()
        )

        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"Scaled mesh가 없습니다: {mesh_path}"
            )

        try:
            loaded_mesh = self._trimesh.load(
                mesh_path,
                process=False,
                maintain_order=True,
            )

        except Exception as error:
            raise RuntimeError(
                f"Mesh를 읽지 못했습니다: {mesh_path}"
            ) from error

        if isinstance(
            loaded_mesh,
            self._trimesh.Scene,
        ):
            if loaded_mesh.is_empty:
                raise ValueError(
                    f"Mesh scene이 비어 있습니다: {mesh_path}"
                )

            geometries = tuple(
                loaded_mesh.geometry.values()
            )

            if len(geometries) == 1:
                mesh = geometries[0].copy()

            else:
                try:
                    mesh = (
                        self._trimesh.util.concatenate(
                            geometries
                        )
                    )

                except Exception as error:
                    raise RuntimeError(
                        "여러 geometry의 mesh scene을 "
                        "결합하지 못했습니다."
                    ) from error

        elif isinstance(
            loaded_mesh,
            self._trimesh.Trimesh,
        ):
            mesh = loaded_mesh

        else:
            raise TypeError(
                "지원하지 않는 mesh 자료형입니다: "
                f"{type(loaded_mesh)}"
            )

        if mesh.is_empty:
            raise ValueError(
                f"Mesh가 비어 있습니다: {mesh_path}"
            )

        vertices = np.asarray(
            mesh.vertices,
            dtype=np.float32,
        )

        faces = np.asarray(
            mesh.faces
        )

        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or vertices.shape[0] < 3
        ):
            raise ValueError(
                "Mesh vertex shape이 올바르지 않습니다: "
                f"{vertices.shape}"
            )

        if (
            faces.ndim != 2
            or faces.shape[1] != 3
            or faces.shape[0] < 1
        ):
            raise ValueError(
                "Mesh triangle face가 없습니다: "
                f"{faces.shape}"
            )

        if not np.isfinite(vertices).all():
            raise ValueError(
                "Mesh vertex에 NaN 또는 Inf가 있습니다."
            )

        vertex_normals = np.asarray(
            mesh.vertex_normals,
            dtype=np.float32,
        )

        if vertex_normals.shape != vertices.shape:
            raise ValueError(
                "Mesh vertex normal shape이 vertex와 다릅니다: "
                f"vertices={vertices.shape}, "
                f"normals={vertex_normals.shape}"
            )

        if not np.isfinite(vertex_normals).all():
            raise ValueError(
                "Mesh normal에 NaN 또는 Inf가 있습니다."
            )

        return mesh

    def _set_object(
        self,
        *,
        mesh: Any,
        mesh_path: Path,
        debug_directory: Path,
    ) -> None:
        """Estimator에 현재 scaled proxy mesh를 설정한다."""

        if self._foundationpose_class is None:
            raise RuntimeError(
                "FoundationPose class가 로드되지 않았습니다."
            )

        vertices = np.asarray(
            mesh.vertices,
            dtype=np.float32,
        )

        normals = np.asarray(
            mesh.vertex_normals,
            dtype=np.float32,
        )

        debug_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self._estimator is None:
            self._estimator = (
                self._foundationpose_class(
                    model_pts=vertices,
                    model_normals=normals,
                    mesh=mesh,
                    scorer=self._scorer,
                    refiner=self._refiner,
                    glctx=self._glctx,
                    debug=self._debug,
                    debug_dir=str(
                        debug_directory
                    ),
                )
            )

        else:
            # reset_object()가 기존 GPU mesh tensor를 교체한다.
            # 내부 attribute를 임의로 삭제하지 않는다.
            self._estimator.reset_object(
                model_pts=vertices,
                model_normals=normals,
                mesh=mesh,
            )

            self._estimator.debug = (
                self._debug
            )

            self._estimator.debug_dir = str(
                debug_directory
            )

            self._estimator.glctx = (
                self._glctx
            )

        self._current_mesh_path = (
            mesh_path
        )

    @staticmethod
    def _prepare_inputs(
        prepared_view: PreparedView,
    ) -> tuple[
        NDArray[np.uint8],
        NDArray[np.float32],
        NDArray[np.float64],
        NDArray[np.uint8],
    ]:
        """PreparedView를 FoundationPose register 입력으로 변환한다."""

        rgb = np.asarray(
            prepared_view.view.rgb,
            dtype=np.uint8,
        )

        depth_m = np.asarray(
            prepared_view.view.depth_m,
            dtype=np.float32,
        ).copy()

        camera_matrix = np.asarray(
            prepared_view.view.camera_matrix,
            dtype=np.float64,
        )

        mask_bool = np.asarray(
            prepared_view.segmentation.mask_bool,
            dtype=np.bool_,
        )

        if (
            rgb.ndim != 3
            or rgb.shape[2] != 3
        ):
            raise ValueError(
                "RGB shape은 (H, W, 3)이어야 합니다: "
                f"{rgb.shape}"
            )

        if depth_m.shape != rgb.shape[:2]:
            raise ValueError(
                "RGB와 depth 해상도가 다릅니다: "
                f"rgb={rgb.shape[:2]}, "
                f"depth={depth_m.shape}"
            )

        if mask_bool.shape != depth_m.shape:
            raise ValueError(
                "Mask와 depth 해상도가 다릅니다: "
                f"mask={mask_bool.shape}, "
                f"depth={depth_m.shape}"
            )

        if camera_matrix.shape != (3, 3):
            raise ValueError(
                "Camera matrix shape은 (3, 3)이어야 합니다."
            )

        depth_m[
            ~np.isfinite(depth_m)
            | (depth_m < 0.0)
        ] = 0.0

        valid_object_depth = (
            mask_bool
            & (depth_m >= 0.001)
        )

        if np.count_nonzero(
            valid_object_depth
        ) < 4:
            raise ValueError(
                "FoundationPose 등록에 필요한 mask 내부 "
                "유효 depth pixel이 4개 미만입니다."
            )

        mask_uint8 = (
            mask_bool.astype(np.uint8)
        )

        return (
            np.ascontiguousarray(
                rgb,
                dtype=np.uint8,
            ),
            np.ascontiguousarray(
                depth_m,
                dtype=np.float32,
            ),
            np.ascontiguousarray(
                camera_matrix,
                dtype=np.float64,
            ),
            np.ascontiguousarray(
                mask_uint8,
                dtype=np.uint8,
            ),
        )

    def _extract_hypotheses(
        self,
        *,
        returned_best_pose: Any,
    ) -> tuple[
        FoundationPoseHypothesis,
        ...
    ]:
        """Estimator 내부의 sorted poses와 scores에서 top-K를 만든다."""

        if self._estimator is None:
            raise RuntimeError(
                "FoundationPose estimator가 없습니다."
            )

        raw_poses = getattr(
            self._estimator,
            "poses",
            None,
        )

        raw_scores = getattr(
            self._estimator,
            "scores",
            None,
        )

        if raw_poses is None or raw_scores is None:
            # 일부 구현에서 top-K 내부 배열을 제공하지 않는 경우
            # register() 반환 best pose 하나만 사용한다.
            best_pose = _sanitize_rigid_pose(
                returned_best_pose,
                "returned_best_pose",
            )

            return (
                FoundationPoseHypothesis(
                    rank=0,
                    score=0.0,
                    pose_cam_from_proxy=(
                        best_pose
                    ),
                ),
            )

        poses_centered = np.asarray(
            _to_numpy(raw_poses),
            dtype=np.float64,
        )

        scores = np.asarray(
            _to_numpy(raw_scores),
            dtype=np.float64,
        ).reshape(-1)

        if (
            poses_centered.ndim != 3
            or poses_centered.shape[1:] != (4, 4)
        ):
            raise ValueError(
                "FoundationPose 내부 poses shape이 "
                "올바르지 않습니다: "
                f"{poses_centered.shape}"
            )

        if scores.size != poses_centered.shape[0]:
            raise ValueError(
                "FoundationPose pose와 score 개수가 다릅니다: "
                f"poses={poses_centered.shape[0]}, "
                f"scores={scores.size}"
            )

        tf_to_centered_mesh = np.asarray(
            _to_numpy(
                self._estimator
                .get_tf_to_centered_mesh()
            ),
            dtype=np.float64,
        )

        if tf_to_centered_mesh.shape != (4, 4):
            raise ValueError(
                "get_tf_to_centered_mesh() 결과 shape이 "
                "올바르지 않습니다: "
                f"{tf_to_centered_mesh.shape}"
            )

        valid_items: list[
            tuple[float, NDArray[np.float32]]
        ] = []

        for pose_index in range(
            poses_centered.shape[0]
        ):
            score = float(
                scores[pose_index]
            )

            if not np.isfinite(score):
                continue

            # FoundationPose 내부 pose:
            # T_camera_from_centered_proxy
            #
            # 최종 pose:
            # T_camera_from_original_proxy
            pose_camera_from_proxy = (
                poses_centered[pose_index]
                @ tf_to_centered_mesh
            )

            try:
                cleaned_pose = (
                    _sanitize_rigid_pose(
                        pose_camera_from_proxy,
                        (
                            "pose_camera_from_proxy"
                            f"[{pose_index}]"
                        ),
                    )
                )

            except ValueError:
                continue

            valid_items.append(
                (
                    score,
                    cleaned_pose,
                )
            )

        if not valid_items:
            raise RuntimeError(
                "FoundationPose가 유효한 pose와 score를 "
                "반환하지 않았습니다."
            )

        sorted_items = sorted(
            valid_items,
            key=lambda item: item[0],
            reverse=True,
        )

        selected_items = sorted_items[
            : self._top_k
        ]

        return tuple(
            FoundationPoseHypothesis(
                rank=rank,
                score=float(score),
                pose_cam_from_proxy=pose,
            )
            for rank, (score, pose)
            in enumerate(selected_items)
        )

    @staticmethod
    def _save_result(
        *,
        result_directory: Path,
        view_name: ViewName,
        candidate: ScaledMeshCandidate,
        hypotheses: tuple[
            FoundationPoseHypothesis,
            ...
        ],
        refine_iterations: int,
    ) -> Path:
        """FoundationPose top-K 결과를 저장한다."""

        result_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        poses = np.stack(
            [
                hypothesis.pose_cam_from_proxy
                for hypothesis in hypotheses
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        scores = np.asarray(
            [
                hypothesis.score
                for hypothesis in hypotheses
            ],
            dtype=np.float32,
        )

        poses_path = (
            result_directory
            / "foundationpose_topk_poses.npy"
        )

        scores_path = (
            result_directory
            / "foundationpose_topk_scores.npy"
        )

        metadata_path = (
            result_directory
            / "foundationpose_result.json"
        )

        np.save(
            poses_path,
            poses,
            allow_pickle=False,
        )

        np.save(
            scores_path,
            scores,
            allow_pickle=False,
        )

        hypothesis_metadata: list[
            dict[str, object]
        ] = []

        for hypothesis in hypotheses:
            pose_path = (
                result_directory
                / (
                    f"hypothesis_"
                    f"{hypothesis.rank:02d}.npy"
                )
            )

            np.save(
                pose_path,
                hypothesis.pose_cam_from_proxy,
                allow_pickle=False,
            )

            hypothesis_metadata.append(
                {
                    "rank": hypothesis.rank,
                    "score": hypothesis.score,
                    "pose_path": str(
                        pose_path
                    ),
                    "pose_cam_from_proxy": (
                        hypothesis
                        .pose_cam_from_proxy
                        .tolist()
                    ),
                }
            )

        metadata = {
            "view_name": view_name,
            "candidate_index": (
                candidate.candidate_index
            ),
            "scale_m": candidate.scale_m,
            "scaled_mesh_path": str(
                candidate.scaled_mesh_path
            ),
            "pose_convention": (
                "T_camera_from_proxy"
            ),
            "proxy_coordinate_frame": (
                "original coordinates of the exact "
                "scaled mesh file"
            ),
            "translation_unit": "meter",
            "refine_iterations": (
                refine_iterations
            ),
            "top_k": len(hypotheses),
            "poses_path": str(
                poses_path
            ),
            "scores_path": str(
                scores_path
            ),
            "hypotheses": (
                hypothesis_metadata
            ),
        }

        with metadata_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return metadata_path

    def run_candidate(
        self,
        *,
        candidate: ScaledMeshCandidate,
        prepared_view: PreparedView,
    ) -> FoundationPoseCandidateResult:
        """Scale candidate 하나를 관측 RGB-D에 등록한다."""

        self.load()

        view_name = (
            prepared_view.view.source.name
        )

        if view_name not in (
            "reference",
            "query",
        ):
            raise ValueError(
                f"지원하지 않는 view입니다: {view_name}"
            )

        candidate_mesh_path = (
            Path(candidate.scaled_mesh_path)
            .expanduser()
            .resolve()
        )

        result_directory = (
            self._output_root
            / view_name
            / (
                f"candidate_"
                f"{candidate.candidate_index:02d}"
            )
        )

        debug_directory = (
            result_directory
            / "debug"
        )

        mesh = self._load_mesh(
            candidate_mesh_path
        )

        self._set_object(
            mesh=mesh,
            mesh_path=candidate_mesh_path,
            debug_directory=debug_directory,
        )

        (
            rgb,
            depth_m,
            camera_matrix,
            mask_uint8,
        ) = self._prepare_inputs(
            prepared_view
        )

        if self._estimator is None:
            raise RuntimeError(
                "FoundationPose estimator가 생성되지 않았습니다."
            )

        try:
            with _temporary_working_directory(
                self._repository_path
            ):
                returned_best_pose = (
                    self._estimator.register(
                        K=camera_matrix,
                        rgb=rgb,
                        depth=depth_m,
                        ob_mask=mask_uint8,
                        iteration=(
                            self._refine_iterations
                        ),
                    )
                )

        except Exception as error:
            raise RuntimeError(
                "FoundationPose register() 실행에 실패했습니다: "
                f"view={view_name}, "
                f"candidate={candidate.candidate_index}, "
                f"mesh={candidate_mesh_path}"
            ) from error

        hypotheses = self._extract_hypotheses(
            returned_best_pose=(
                returned_best_pose
            )
        )

        metadata_path = self._save_result(
            result_directory=(
                result_directory
            ),
            view_name=view_name,
            candidate=candidate,
            hypotheses=hypotheses,
            refine_iterations=(
                self._refine_iterations
            ),
        )

        return FoundationPoseCandidateResult(
            view_name=view_name,
            candidate_index=(
                candidate.candidate_index
            ),
            scale_m=float(
                candidate.scale_m
            ),
            scaled_mesh_path=(
                candidate_mesh_path
            ),
            hypotheses=hypotheses,
            output_directory=(
                result_directory
            ),
            metadata_path=metadata_path,
        )

    def refine_pose_locally(
        self,
        *,
        mesh_path: Path,
        prepared_view: PreparedView,
        initial_pose_camera_from_mesh: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        이미 알려진 초기 pose(T0) 근처에서만 지역적으로 pose를 재추정한다.

        register()와 달리 회전 그리드 전체(~90개 이상)를 다시 샘플링하고
        재점수화하지 않는다 -- FoundationPose 자체의 refiner 네트워크만
        한 번 호출하는 track_one() 경로를 쓴다. Depth/silhouette로 mesh
        형상을 국소 보정한 뒤, 그 형상 변화 때문에 더 이상 최적이 아닐
        수 있는 T0를 약간만 조정하려는 용도다 -- 대칭 물체에서 완전히
        다른 pose branch로 튈 위험이 있는 전역 재탐색과는 다르다.

        Scale은 건드리지 않는다 -- FoundationPose는애초에 scale을
        추정하지 않고 입력 mesh의 vertex 크기를 그대로 쓰므로, 호출자가
        이미 최종 S*로 스케일된 mesh를 넘겨주면 자동으로 고정된다.

        FoundationPose는 reset_object() 호출마다 mesh AABB 중심을
        원점으로 재배치한다(get_tf_to_centered_mesh()). pose_last는
        이 "centered mesh" 컨벤션으로 표현되어야 하므로, 호출자가 들고
        있는 T0(원본 mesh 좌표계 기준)를 변환해서 넣고, 결과도 다시
        원본 좌표계로 되돌려서 반환한다.
        """

        self.load()

        view_name = (
            prepared_view.view.source.name
        )

        if view_name not in (
            "reference",
            "query",
        ):
            raise ValueError(
                f"지원하지 않는 view입니다: {view_name}"
            )

        debug_directory = (
            self._output_root
            / view_name
            / "local_pose_refine"
            / "debug"
        )

        mesh = self._load_mesh(mesh_path)

        self._set_object(
            mesh=mesh,
            mesh_path=mesh_path,
            debug_directory=debug_directory,
        )

        (
            rgb,
            depth_m,
            camera_matrix,
            _mask_uint8,
        ) = self._prepare_inputs(
            prepared_view
        )

        if self._estimator is None:
            raise RuntimeError(
                "FoundationPose estimator가 생성되지 않았습니다."
            )

        torch = self._torch

        tf_to_centered = (
            self._estimator
            .get_tf_to_centered_mesh()
            .data.cpu().numpy()
            .astype(np.float64)
        )

        pose_last_centered = (
            initial_pose_camera_from_mesh
            @ np.linalg.inv(tf_to_centered)
        )

        self._estimator.pose_last = torch.as_tensor(
            pose_last_centered,
            device="cuda",
            dtype=torch.float,
        )

        try:
            with _temporary_working_directory(
                self._repository_path
            ):
                self._estimator.track_one(
                    rgb=rgb,
                    depth=depth_m,
                    K=camera_matrix,
                    iteration=(
                        self._refine_iterations
                    ),
                )

        except Exception as error:
            raise RuntimeError(
                "FoundationPose track_one() 실행에 실패했습니다: "
                f"view={view_name}, mesh={mesh_path}"
            ) from error

        refined_pose_centered = (
            self._estimator.pose_last
            .reshape(4, 4)
            .data.cpu().numpy()
            .astype(np.float64)
        )

        refined_pose_camera_from_mesh = (
            refined_pose_centered @ tf_to_centered
        )

        # track_one()의 refiner 네트워크가 반환하는 회전 행렬은 완벽한
        # SO(3)가 아닐 수 있다(부동소수점 누적 오차) -- SVD로 가장
        # 가까운 회전 행렬에 투영해서 downstream 검증을 통과시킨다.
        # 오차가 큰 경우(진짜 깨진 pose)는 여기서도 예외가 난다.
        sanitized = _sanitize_rigid_pose(
            refined_pose_camera_from_mesh,
            "refine_pose_locally() 결과",
        )

        return sanitized.astype(np.float64)

    def run_candidates(
        self,
        *,
        candidates: Sequence[
            ScaledMeshCandidate
        ],
        prepared_view: PreparedView,
    ) -> tuple[
        FoundationPoseCandidateResult,
        ...
    ]:
        """여러 scale candidate를 순차적으로 실행한다."""

        if not candidates:
            raise ValueError(
                "FoundationPose를 실행할 scale candidate가 "
                "없습니다."
            )

        seen_indices: set[int] = set()
        results: list[
            FoundationPoseCandidateResult
        ] = []

        for candidate in candidates:
            if (
                candidate.candidate_index
                in seen_indices
            ):
                raise ValueError(
                    "중복된 scale candidate index입니다: "
                    f"{candidate.candidate_index}"
                )

            seen_indices.add(
                candidate.candidate_index
            )

            result = self.run_candidate(
                candidate=candidate,
                prepared_view=prepared_view,
            )

            results.append(result)

        return tuple(results)

    def release(self) -> None:
        """FoundationPose GPU model과 context 참조를 해제한다."""

        self._estimator = None
        self._current_mesh_path = None

        self._scorer = None
        self._refiner = None
        self._glctx = None

        self._foundationpose_class = None
        self._estimater_module = None
        self._dr = None
        self._trimesh = None

        gc.collect()

        if self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
                self._torch.cuda.ipc_collect()

            except RuntimeError:
                pass

        self._torch = None

        if self._repository_added_to_path:
            repository_string = str(
                self._repository_path
            )

            try:
                sys.path.remove(
                    repository_string
                )

            except ValueError:
                pass

            self._repository_added_to_path = False

    def __enter__(
        self,
    ) -> FoundationPoseRunner:
        self.load()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.release()
