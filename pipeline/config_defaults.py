from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PIPELINE_CONFIG = (
    PROJECT_ROOT / "configs" / "pipeline.yaml"
)
DEFAULT_INSTANTMESH_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "instantmesh_16gb"
    / "instant-mesh-large.yaml"
)
DEFAULT_DINOV3_REPOSITORY = (
    PROJECT_ROOT / "dinov3"
)
DINOV3_CHECKPOINT_FILENAMES = {
    "dinov3_vits16": (
        "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    ),
    "dinov3_vits16plus": (
        "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
    ),
    "dinov3_vitb16": (
        "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    ),
    "dinov3_vitl16": (
        "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    ),
    "dinov3_vith16plus": (
        "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
    ),
}
DINOV3_HF_MODEL_DIRECTORIES = {
    "dinov3_vits16": (
        "dinov3-vits16-pretrain-lvd1689m"
    ),
    "dinov3_vits16plus": (
        "dinov3-vits16plus-pretrain-lvd1689m"
    ),
    "dinov3_vitb16": (
        "dinov3-vitb16-pretrain-lvd1689m"
    ),
    "dinov3_vitl16": (
        "dinov3-vitl16-pretrain-lvd1689m"
    ),
    "dinov3_vith16plus": (
        "dinov3-vith16plus-pretrain-lvd1689m"
    ),
}
LINEMOD_OBJECT_METADATA: dict[
    int, tuple[str, str]
] = {
    1: ("ape", "brown toy"),
    2: (
        "benchvise",
        "a metallic bench vise tool with a screw handle",
    ),
    3: ("bowl", "a white bowl"),
    4: ("camera", "camera"),
    5: ("can", "white watering can"),
    6: ("cat", "pink cat figurine"),
    7: ("cup", "cup"),
    8: ("driller", "power drill"),
    9: ("duck", "a small yellow rubber duck"),
    10: ("eggbox", "plastic case"),
    11: (
        "glue",
        "a small white glue bottle with a nozzle",
    ),
    12: (
        "holepuncher",
        "blue paper hole punch",
    ),
    13: ("iron", "clothes iron"),
    14: ("lamp", "white desk lamp"),
    15: ("phone", "phone handset"),
}
POSE_PATH_CHOICES = ("self_mesh",)


def _default_dinov3_weights_path(
    repository_path: Path,
    model_name: str,
) -> Path:
    weights_directory = (
        repository_path / "weights"
    )
    huggingface_directory = (
        weights_directory
        / DINOV3_HF_MODEL_DIRECTORIES[model_name]
    )
    legacy_checkpoint = (
        weights_directory
        / DINOV3_CHECKPOINT_FILENAMES[model_name]
    )

    if huggingface_directory.is_dir():
        return huggingface_directory

    if legacy_checkpoint.is_file():
        return legacy_checkpoint

    return huggingface_directory


def _default_instantmesh_python() -> Path:
    candidates: list[Path] = []

    conda_prefix = os.environ.get("CONDA_PREFIX")

    if conda_prefix:
        candidates.append(
            Path(conda_prefix)
            .expanduser()
            .resolve()
            .parent
            / "instantmesh_clean"
            / "bin"
            / "python"
        )

    candidates.extend(
        (
            (
                Path.home()
                / "miniforge3"
                / "envs"
                / "instantmesh_clean"
                / "bin"
                / "python"
            ),
            (
                Path.home()
                / "miniconda3"
                / "envs"
                / "instantmesh_clean"
                / "bin"
                / "python"
            ),
        )
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return Path(sys.executable)
