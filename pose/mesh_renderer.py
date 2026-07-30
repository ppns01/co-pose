from __future__ import annotations

import gc
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
import trimesh

from pose.torch_state import (
    preserve_torch_default_tensor_type,
)


@dataclass(frozen=True)
class MeshRenderResult:
    """
    여러 pose 후보의 렌더링 결과.

    rendered_masks:
        shape = (N, H, W)
        dtype = bool

    rendered_depth_m:
        shape = (N, H, W)
        dtype = float32
        단위 = meter
        배경 = 0

    rendered_rgb:
        shape = (N, H, W, 3)
        dtype = uint8
    """

    mesh_path: Path
    poses_camera_from_proxy: NDArray[np.float32]

    rendered_masks: NDArray[np.bool_]
    rendered_depth_m: NDArray[np.float32]
    rendered_rgb: NDArray[np.uint8]

    output_directory: Path | None = None
    metadata_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.mesh_path.is_file():
            raise FileNotFoundError(
                f"렌더링 mesh가 없습니다: {self.mesh_path}"
            )

        if (
            self.poses_camera_from_proxy.ndim != 3
            or self.poses_camera_from_proxy.shape[1:] != (4, 4)
        ):
            raise ValueError(
                "Pose 배열 shape은 (N, 4, 4)이어야 합니다: "
                f"{self.poses_camera_from_proxy.shape}"
            )

        pose_count = self.poses_camera_from_proxy.shape[0]

        if self.poses_camera_from_proxy.dtype != np.float32:
            raise TypeError(
                "Pose dtype은 float32이어야 합니다: "
                f"{self.poses_camera_from_proxy.dtype}"
            )

        if (
            self.rendered_masks.ndim != 3
            or self.rendered_masks.shape[0] != pose_count
        ):
            raise ValueError(
                "Rendered mask shape이 올바르지 않습니다: "
                f"{self.rendered_masks.shape}"
            )

        if self.rendered_masks.dtype != np.bool_:
            raise TypeError(
                "Rendered mask dtype은 bool이어야 합니다: "
                f"{self.rendered_masks.dtype}"
            )

        if self.rendered_depth_m.shape != self.rendered_masks.shape:
            raise ValueError(
                "Rendered mask와 depth shape이 다릅니다: "
                f"mask={self.rendered_masks.shape}, "
                f"depth={self.rendered_depth_m.shape}"
            )

        if self.rendered_depth_m.dtype != np.float32:
            raise TypeError(
                "Rendered depth dtype은 float32이어야 합니다: "
                f"{self.rendered_depth_m.dtype}"
            )

        expected_rgb_shape = (
            pose_count,
            self.rendered_masks.shape[1],
            self.rendered_masks.shape[2],
            3,
        )

        if self.rendered_rgb.shape != expected_rgb_shape:
            raise ValueError(
                "Rendered RGB shape이 올바르지 않습니다: "
                f"expected={expected_rgb_shape}, "
                f"actual={self.rendered_rgb.shape}"
            )

        if self.rendered_rgb.dtype != np.uint8:
            raise TypeError(
                "Rendered RGB dtype은 uint8이어야 합니다: "
                f"{self.rendered_rgb.dtype}"
            )

        if not np.isfinite(self.rendered_depth_m).all():
            raise ValueError(
                "Rendered depth에 NaN 또는 Inf가 있습니다."
            )

        if np.any(self.rendered_depth_m < 0.0):
            raise ValueError(
                "Rendered depth에 음수 값이 있습니다."
            )


def _validate_camera_matrix(
    camera_matrix: NDArray[np.floating],
) -> NDArray[np.float32]:
    """3×3 camera intrinsic을 검증한다."""

    camera_matrix = np.asarray(
        camera_matrix,
        dtype=np.float32,
    )

    if camera_matrix.shape != (3, 3):
        raise ValueError(
            "Camera intrinsic shape은 (3, 3)이어야 합니다: "
            f"{camera_matrix.shape}"
        )

    if not np.isfinite(camera_matrix).all():
        raise ValueError(
            "Camera intrinsic에 NaN 또는 Inf가 있습니다."
        )

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])

    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(
            "fx와 fy는 양수여야 합니다: "
            f"fx={fx}, fy={fy}"
        )

    return np.ascontiguousarray(
        camera_matrix,
        dtype=np.float32,
    )


def _validate_poses(
    poses_camera_from_proxy: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Pose 배열을 (N, 4, 4) SE(3) 형식으로 검증한다."""

    poses = np.asarray(
        poses_camera_from_proxy,
        dtype=np.float32,
    )

    if poses.ndim == 2:
        poses = poses[None, :, :]

    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(
            "Pose shape은 (4, 4) 또는 "
            "(N, 4, 4)이어야 합니다: "
            f"{poses.shape}"
        )

    if poses.shape[0] == 0:
        raise ValueError(
            "렌더링할 pose가 하나도 없습니다."
        )

    if not np.isfinite(poses).all():
        raise ValueError(
            "Pose에 NaN 또는 Inf가 있습니다."
        )

    expected_last_row = np.array(
        [0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )

    for pose_index, pose in enumerate(poses):
        if not np.allclose(
            pose[3],
            expected_last_row,
            atol=1e-5,
            rtol=0.0,
        ):
            raise ValueError(
                "Pose 마지막 행이 올바르지 않습니다: "
                f"index={pose_index}"
            )

        rotation = pose[:3, :3]

        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3, dtype=np.float32),
            atol=1e-3,
            rtol=0.0,
        ):
            raise ValueError(
                "Pose 회전 행렬이 직교하지 않습니다: "
                f"index={pose_index}"
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
                "Pose 회전 determinant가 1이 아닙니다: "
                f"index={pose_index}, det={determinant}"
            )

    return np.ascontiguousarray(
        poses,
        dtype=np.float32,
    )


def _load_trimesh(
    mesh_path: Path,
) -> trimesh.Trimesh:
    """OBJ, GLB 등의 mesh를 단일 Trimesh로 읽는다."""

    mesh_path = (
        Path(mesh_path)
        .expanduser()
        .resolve()
    )

    if not mesh_path.is_file():
        raise FileNotFoundError(
            f"Mesh 파일이 없습니다: {mesh_path}"
        )

    try:
        loaded = trimesh.load(
            mesh_path,
            process=False,
        )
    except Exception as error:
        raise RuntimeError(
            f"Mesh를 읽지 못했습니다: {mesh_path}"
        ) from error

    if isinstance(loaded, trimesh.Scene):
        if loaded.is_empty:
            raise ValueError(
                f"Mesh scene이 비어 있습니다: {mesh_path}"
            )

        try:
            mesh = loaded.to_mesh()
        except Exception as error:
            raise RuntimeError(
                "Scene을 단일 mesh로 변환하지 못했습니다."
            ) from error

    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded

    else:
        raise TypeError(
            "지원하지 않는 trimesh 로딩 결과입니다: "
            f"{type(loaded)}"
        )

    if mesh.is_empty:
        raise ValueError(
            f"Mesh가 비어 있습니다: {mesh_path}"
        )

    if len(mesh.vertices) < 3:
        raise ValueError(
            "Mesh vertex가 너무 적습니다: "
            f"{len(mesh.vertices)}"
        )

    if len(mesh.faces) < 1:
        raise ValueError(
            "Mesh triangle face가 없습니다."
        )

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float32,
    )

    if not np.isfinite(vertices).all():
        raise ValueError(
            "Mesh vertex에 NaN 또는 Inf가 있습니다."
        )

    return mesh


def _write_rgb_png(
    output_path: Path,
    rgb: NDArray[np.uint8],
) -> None:
    """Windows 한글 경로를 지원하도록 RGB PNG를 저장한다."""

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
            f"RGB PNG 인코딩에 실패했습니다: {output_path}"
        )

    encoded.tofile(
        str(output_path)
    )


def _write_mask_png(
    output_path: Path,
    mask: NDArray[np.bool_],
) -> None:
    """Boolean mask를 0/255 RGB PNG로 저장한다."""

    mask_gray = (
        mask.astype(np.uint8)
        * np.uint8(255)
    )

    mask_rgb = np.repeat(
        mask_gray[:, :, None],
        repeats=3,
        axis=2,
    )

    _write_rgb_png(
        output_path=output_path,
        rgb=mask_rgb,
    )


class FoundationPoseMeshRenderer:
    """
    FoundationPose 공식 nvdiffrast renderer adapter.

    동일 mesh의 여러 pose를 batch로 렌더링한다.

    RTX 5060 Ti 16GB 기준:
    - texture 크기 제한
    - pose batch 분할
    - 렌더링 완료 후 명시적 GPU 해제
    """

    def __init__(
        self,
        foundationpose_repository_path: Path,
        *,
        device: str = "cuda:0",
        render_batch_size: int = 3,
        maximum_texture_size: int = 1024,
    ) -> None:
        self._repository_path = (
            Path(foundationpose_repository_path)
            .expanduser()
            .resolve()
        )

        if not device.startswith("cuda"):
            raise ValueError(
                "FoundationPose renderer는 CUDA가 필요합니다: "
                f"{device}"
            )

        if (
            isinstance(render_batch_size, bool)
            or render_batch_size < 1
        ):
            raise ValueError(
                "render_batch_size는 1 이상이어야 합니다: "
                f"{render_batch_size}"
            )

        if (
            isinstance(maximum_texture_size, bool)
            or maximum_texture_size < 1
        ):
            raise ValueError(
                "maximum_texture_size는 1 이상이어야 합니다: "
                f"{maximum_texture_size}"
            )

        self._device_string = device
        self._render_batch_size = int(
            render_batch_size
        )
        self._maximum_texture_size = int(
            maximum_texture_size
        )

        self._torch: ModuleType | None = None
        self._dr: ModuleType | None = None
        self._utils: ModuleType | None = None

        self._glctx: Any | None = None
        self._mesh_path: Path | None = None
        self._mesh_tensors: dict[str, Any] | None = None

    @property
    def is_loaded(self) -> bool:
        return self._utils is not None

    def _load_utils_module(self) -> ModuleType:
        """지정한 FoundationPose 저장소의 Utils.py를 로드한다."""

        utils_path = (
            self._repository_path / "Utils.py"
        )

        if not utils_path.is_file():
            raise FileNotFoundError(
                "FoundationPose Utils.py가 없습니다: "
                f"{utils_path}"
            )

        module_name = (
            "_local_foundationpose_utils"
        )

        spec = importlib.util.spec_from_file_location(
            module_name,
            utils_path,
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                "FoundationPose Utils.py import spec을 "
                "생성하지 못했습니다."
            )

        module = importlib.util.module_from_spec(
            spec
        )

        try:
            spec.loader.exec_module(module)
        except Exception as error:
            raise ImportError(
                "FoundationPose Utils.py를 로드하지 "
                "못했습니다. FoundationPose 환경과 "
                "nvdiffrast 빌드를 확인해야 합니다."
            ) from error

        required_functions = (
            "make_mesh_tensors",
            "nvdiffrast_render",
        )

        for function_name in required_functions:
            if not hasattr(module, function_name):
                raise AttributeError(
                    "FoundationPose Utils.py에 필수 함수가 "
                    f"없습니다: {function_name}"
                )

        return module

    def load(self) -> None:
        """렌더링 runtime과 CUDA context를 준비한다."""

        if self.is_loaded:
            return

        if not self._repository_path.is_dir():
            raise FileNotFoundError(
                "FoundationPose 저장소가 없습니다: "
                f"{self._repository_path}"
            )

        try:
            import torch
            import nvdiffrast.torch as dr
        except Exception as error:
            raise ImportError(
                "torch 또는 nvdiffrast를 import하지 "
                "못했습니다."
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch에서 CUDA를 사용할 수 없습니다."
            )

        device = torch.device(
            self._device_string
        )

        torch.cuda.set_device(device)

        self._torch = torch
        self._dr = dr
        self._utils = (
            self._load_utils_module()
        )

        try:
            self._glctx = (
                dr.RasterizeCudaContext(
                    device=device
                )
            )
        except TypeError:
            # nvdiffrast 버전에 따라 device 인자를
            # 지원하지 않을 수 있다.
            self._glctx = (
                dr.RasterizeCudaContext()
            )
        except Exception as error:
            self.release()

            raise RuntimeError(
                "nvdiffrast CUDA context 생성에 실패했습니다."
            ) from error

    def set_mesh(
        self,
        mesh_path: Path,
    ) -> None:
        """
        렌더링할 scaled proxy mesh를 GPU에 올린다.

        같은 mesh가 이미 설정되어 있으면 재사용한다.
        """

        self.load()

        resolved_mesh_path = (
            Path(mesh_path)
            .expanduser()
            .resolve()
        )

        if (
            self._mesh_path == resolved_mesh_path
            and self._mesh_tensors is not None
        ):
            return

        self._release_mesh_tensors()

        mesh = _load_trimesh(
            resolved_mesh_path
        )

        assert self._utils is not None

        try:
            mesh_tensors = (
                self._utils.make_mesh_tensors(
                    mesh=mesh,
                    device=self._device_string,
                    max_tex_size=(
                        self._maximum_texture_size
                    ),
                )
            )
        except Exception as error:
            raise RuntimeError(
                "Mesh를 FoundationPose 렌더링 tensor로 "
                "변환하지 못했습니다: "
                f"{resolved_mesh_path}"
            ) from error

        if not isinstance(
            mesh_tensors,
            dict,
        ):
            raise TypeError(
                "make_mesh_tensors() 결과가 dict가 아닙니다: "
                f"{type(mesh_tensors)}"
            )

        required_keys = {
            "pos",
            "faces",
            "vnormals",
        }

        missing_keys = (
            required_keys
            - set(mesh_tensors.keys())
        )

        if missing_keys:
            raise KeyError(
                "Mesh tensor에 필수 항목이 없습니다: "
                f"{sorted(missing_keys)}"
            )

        self._mesh_path = resolved_mesh_path
        self._mesh_tensors = mesh_tensors

    def _render_chunk(
        self,
        *,
        pose_chunk: NDArray[np.float32],
        camera_matrix: NDArray[np.float32],
        image_height: int,
        image_width: int,
    ) -> tuple[
        NDArray[np.uint8],
        NDArray[np.float32],
        NDArray[np.bool_],
    ]:
        """Pose 배치 일부를 GPU에서 렌더링한다."""

        if self._torch is None:
            raise RuntimeError(
                "Renderer runtime이 로드되지 않았습니다."
            )

        if self._utils is None:
            raise RuntimeError(
                "FoundationPose Utils가 로드되지 않았습니다."
            )

        if self._mesh_tensors is None:
            raise RuntimeError(
                "렌더링 mesh가 설정되지 않았습니다."
            )

        if self._glctx is None:
            raise RuntimeError(
                "nvdiffrast context가 없습니다."
            )

        torch = self._torch

        pose_tensor = torch.as_tensor(
            pose_chunk,
            device=self._device_string,
            dtype=torch.float32,
        )

        with (
            preserve_torch_default_tensor_type(
                torch
            ),
            torch.inference_mode(),
        ):
            try:
                color_tensor, depth_tensor, _ = (
                    self._utils.nvdiffrast_render(
                        K=camera_matrix,
                        H=image_height,
                        W=image_width,
                        ob_in_cams=pose_tensor,
                        glctx=self._glctx,
                        context="cuda",
                        get_normal=False,
                        mesh_tensors=self._mesh_tensors,
                        mesh=None,
                        output_size=(
                            image_height,
                            image_width,
                        ),
                        use_light=False,
                    )
                )
            except Exception as error:
                raise RuntimeError(
                    "FoundationPose nvdiffrast 렌더링에 "
                    "실패했습니다."
                ) from error

        if color_tensor.ndim != 4:
            raise ValueError(
                "Rendered color tensor shape이 "
                "올바르지 않습니다: "
                f"{tuple(color_tensor.shape)}"
            )

        if depth_tensor.ndim != 3:
            raise ValueError(
                "Rendered depth tensor shape이 "
                "올바르지 않습니다: "
                f"{tuple(depth_tensor.shape)}"
            )

        valid_depth_tensor = (
            torch.isfinite(depth_tensor)
            & (depth_tensor > 0.0)
        )

        cleaned_depth_tensor = torch.where(
            valid_depth_tensor,
            depth_tensor,
            torch.zeros_like(depth_tensor),
        )

        rendered_rgb = (
            color_tensor
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(
                device="cpu",
                dtype=torch.uint8,
            )
            .numpy()
        )

        rendered_depth_m = (
            cleaned_depth_tensor
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .numpy()
        )

        rendered_masks = (
            valid_depth_tensor
            .to(
                device="cpu",
                dtype=torch.bool,
            )
            .numpy()
        )

        del pose_tensor
        del color_tensor
        del depth_tensor
        del valid_depth_tensor
        del cleaned_depth_tensor

        return (
            np.ascontiguousarray(
                rendered_rgb,
                dtype=np.uint8,
            ),
            np.ascontiguousarray(
                rendered_depth_m,
                dtype=np.float32,
            ),
            np.ascontiguousarray(
                rendered_masks,
                dtype=np.bool_,
            ),
        )

    def render(
        self,
        *,
        mesh_path: Path,
        poses_camera_from_proxy: NDArray[np.floating],
        camera_matrix: NDArray[np.floating],
        image_height: int,
        image_width: int,
        output_directory: Path | None = None,
        filename_prefix: str = "hypothesis",
    ) -> MeshRenderResult:
        """
        동일 mesh의 여러 pose 후보를 렌더링한다.

        Parameters
        ----------
        mesh_path:
            Scale이 적용된 proxy mesh.

        poses_camera_from_proxy:
            shape=(N, 4, 4)
            각 pose는 T_camera_from_proxy.

        camera_matrix:
            Target view의 camera intrinsic K.

        image_height, image_width:
            Target RGB-D 해상도.
        """

        if image_height < 1 or image_width < 1:
            raise ValueError(
                "렌더링 해상도는 양수여야 합니다: "
                f"{(image_height, image_width)}"
            )

        if not filename_prefix.strip():
            raise ValueError(
                "filename_prefix는 빈 문자열일 수 없습니다."
            )

        poses = _validate_poses(
            poses_camera_from_proxy
        )

        camera_matrix = (
            _validate_camera_matrix(
                camera_matrix
            )
        )

        self.set_mesh(
            mesh_path
        )

        rendered_rgb_chunks: list[
            NDArray[np.uint8]
        ] = []

        rendered_depth_chunks: list[
            NDArray[np.float32]
        ] = []

        rendered_mask_chunks: list[
            NDArray[np.bool_]
        ] = []

        for start_index in range(
            0,
            poses.shape[0],
            self._render_batch_size,
        ):
            end_index = min(
                start_index
                + self._render_batch_size,
                poses.shape[0],
            )

            (
                rgb_chunk,
                depth_chunk,
                mask_chunk,
            ) = self._render_chunk(
                pose_chunk=poses[
                    start_index:end_index
                ],
                camera_matrix=camera_matrix,
                image_height=image_height,
                image_width=image_width,
            )

            rendered_rgb_chunks.append(
                rgb_chunk
            )

            rendered_depth_chunks.append(
                depth_chunk
            )

            rendered_mask_chunks.append(
                mask_chunk
            )

        rendered_rgb = np.concatenate(
            rendered_rgb_chunks,
            axis=0,
        )

        rendered_depth_m = np.concatenate(
            rendered_depth_chunks,
            axis=0,
        )

        rendered_masks = np.concatenate(
            rendered_mask_chunks,
            axis=0,
        )

        resolved_output_directory: Path | None = None
        metadata_path: Path | None = None

        if output_directory is not None:
            resolved_output_directory = (
                Path(output_directory)
                .expanduser()
                .resolve()
            )

            metadata_path = self._save_result(
                mesh_path=(
                    Path(mesh_path)
                    .expanduser()
                    .resolve()
                ),
                poses=poses,
                camera_matrix=camera_matrix,
                rendered_rgb=rendered_rgb,
                rendered_depth_m=rendered_depth_m,
                rendered_masks=rendered_masks,
                output_directory=(
                    resolved_output_directory
                ),
                filename_prefix=filename_prefix,
            )

        return MeshRenderResult(
            mesh_path=(
                Path(mesh_path)
                .expanduser()
                .resolve()
            ),
            poses_camera_from_proxy=poses,
            rendered_masks=rendered_masks,
            rendered_depth_m=rendered_depth_m,
            rendered_rgb=rendered_rgb,
            output_directory=(
                resolved_output_directory
            ),
            metadata_path=metadata_path,
        )

    @staticmethod
    def _save_result(
        *,
        mesh_path: Path,
        poses: NDArray[np.float32],
        camera_matrix: NDArray[np.float32],
        rendered_rgb: NDArray[np.uint8],
        rendered_depth_m: NDArray[np.float32],
        rendered_masks: NDArray[np.bool_],
        output_directory: Path,
        filename_prefix: str,
    ) -> Path:
        """렌더링 배열과 확인용 이미지를 저장한다."""

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        poses_path = (
            output_directory / "render_poses.npy"
        )

        masks_path = (
            output_directory / "rendered_masks.npy"
        )

        depth_path = (
            output_directory / "rendered_depth_m.npy"
        )

        rgb_path = (
            output_directory / "rendered_rgb.npy"
        )

        metadata_path = (
            output_directory / "render_metadata.json"
        )

        np.save(
            poses_path,
            poses,
            allow_pickle=False,
        )

        np.save(
            masks_path,
            rendered_masks,
            allow_pickle=False,
        )

        np.save(
            depth_path,
            rendered_depth_m,
            allow_pickle=False,
        )

        np.save(
            rgb_path,
            rendered_rgb,
            allow_pickle=False,
        )

        hypothesis_files: list[
            dict[str, Any]
        ] = []

        for hypothesis_index in range(
            poses.shape[0]
        ):
            file_stem = (
                f"{filename_prefix}_"
                f"{hypothesis_index:02d}"
            )

            mask_png_path = (
                output_directory
                / f"{file_stem}_mask.png"
            )

            rgb_png_path = (
                output_directory
                / f"{file_stem}_rgb.png"
            )

            depth_npy_path = (
                output_directory
                / f"{file_stem}_depth_m.npy"
            )

            _write_mask_png(
                output_path=mask_png_path,
                mask=rendered_masks[
                    hypothesis_index
                ],
            )

            _write_rgb_png(
                output_path=rgb_png_path,
                rgb=rendered_rgb[
                    hypothesis_index
                ],
            )

            np.save(
                depth_npy_path,
                rendered_depth_m[
                    hypothesis_index
                ],
                allow_pickle=False,
            )

            hypothesis_files.append(
                {
                    "hypothesis_index": (
                        hypothesis_index
                    ),
                    "mask_png": str(
                        mask_png_path
                    ),
                    "rgb_png": str(
                        rgb_png_path
                    ),
                    "depth_npy": str(
                        depth_npy_path
                    ),
                    "rendered_pixel_count": int(
                        np.count_nonzero(
                            rendered_masks[
                                hypothesis_index
                            ]
                        )
                    ),
                }
            )

        metadata = {
            "mesh_path": str(mesh_path),
            "pose_convention": (
                "T_camera_from_proxy"
            ),
            "translation_unit": "meter",
            "pose_count": int(
                poses.shape[0]
            ),
            "image_height": int(
                rendered_masks.shape[1]
            ),
            "image_width": int(
                rendered_masks.shape[2]
            ),
            "camera_matrix": (
                camera_matrix.tolist()
            ),
            "poses_path": str(poses_path),
            "masks_path": str(masks_path),
            "depth_path": str(depth_path),
            "rgb_path": str(rgb_path),
            "hypotheses": hypothesis_files,
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

    def _release_mesh_tensors(self) -> None:
        """현재 mesh의 GPU tensor를 해제한다."""

        self._mesh_tensors = None
        self._mesh_path = None

        gc.collect()

        if self._torch is not None:
            self._torch.cuda.empty_cache()

    def release(self) -> None:
        """Mesh tensor와 CUDA context를 해제한다."""

        self._release_mesh_tensors()

        self._glctx = None
        self._utils = None
        self._dr = None

        gc.collect()

        if self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
                self._torch.cuda.ipc_collect()
            except RuntimeError:
                pass

        self._torch = None

    def __enter__(
        self,
    ) -> FoundationPoseMeshRenderer:
        self.load()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.release()
