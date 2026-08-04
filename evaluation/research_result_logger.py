from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


POSE_DIRECTION = "query_from_reference"
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class ResearchRunContext:
    run_id: str
    started_at: str
    output_directory: Path
    config_path: Path
    config_hash: str
    git_commit: str
    instantmesh_commit: str
    foundationpose_commit: str
    python_version: str
    torch_version: str
    cuda_version: str
    gpu_name: str
    random_seed: int = DEFAULT_RANDOM_SEED


@dataclass(frozen=True)
class ResearchLoggingResult:
    pair_results_path: Path
    path_results_path: Path
    proxy_results_path: Path
    reference_pose_path: Path
    query_pose_path: Path
    final_pose_path: Path | None
    ground_truth_pose_path: Path


def _now_local_iso() -> str:
    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def _canonical_config_hash(
    config_path: Path,
) -> str:
    with config_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(canonical).hexdigest()


def _git_commit(repository_path: Path) -> str:
    repository_path = Path(repository_path)

    if not (repository_path / ".git").exists():
        return ""

    try:
        completed = subprocess.run(
            (
                "git",
                "-C",
                str(repository_path),
                "rev-parse",
                "HEAD",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return ""

    return completed.stdout.strip()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _cuda_version_from_torch_version(
    torch_version: str,
) -> str:
    marker = "+cu"

    if marker not in torch_version:
        return ""

    encoded = torch_version.split(
        marker,
        maxsplit=1,
    )[1]

    if len(encoded) < 3 or not encoded.isdigit():
        return encoded

    return (
        f"{int(encoded[:-1])}."
        f"{int(encoded[-1])}"
    )


def _gpu_name_from_nvidia_smi() -> str:
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return ""

    names = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    return ";".join(names)


def initialize_research_run(
    *,
    output_root: Path,
    config_path: Path,
    project_root: Path,
    instantmesh_repository: Path,
    foundationpose_repository: Path,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> ResearchRunContext:
    config_path = Path(config_path).resolve()
    config_hash = _canonical_config_hash(
        config_path
    )
    started_at = _now_local_iso()
    compact_time = (
        datetime.now()
        .astimezone()
        .strftime("%Y%m%dT%H%M%S%f%z")
    )
    run_id = (
        f"run_{compact_time}_{config_hash[:8]}"
    )
    research_root = (
        Path(output_root).resolve() / "research"
    )
    research_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch_version = _package_version("torch")
    context = ResearchRunContext(
        run_id=run_id,
        started_at=started_at,
        output_directory=research_root,
        config_path=config_path,
        config_hash=config_hash,
        git_commit=_git_commit(project_root),
        instantmesh_commit=_git_commit(
            instantmesh_repository
        ),
        foundationpose_commit=_git_commit(
            foundationpose_repository
        ),
        python_version=sys.version.split()[0],
        torch_version=torch_version,
        cuda_version=(
            _cuda_version_from_torch_version(
                torch_version
            )
        ),
        gpu_name=_gpu_name_from_nvidia_smi(),
        random_seed=random_seed,
    )

    metadata_directory = (
        research_root / "run_metadata"
    )
    metadata_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    metadata_path = (
        metadata_directory
        / f"{run_id}.json"
    )
    metadata = {
        **asdict(context),
        "output_directory": str(
            context.output_directory
        ),
        "config_path": str(context.config_path),
        "peak_gpu_memory_mb": None,
        "peak_gpu_memory_note": (
            "Unavailable: FoundationPose and InstantMesh "
            "run in separate processes."
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

    definitions_path = (
        research_root
        / "metric_definitions.json"
    )
    definitions = {
        "pose_direction": POSE_DIRECTION,
        "pose_equation": (
            "X_query = H_query_from_reference "
            "@ X_reference"
        ),
        "translation_error_axis": (
            "signed estimate minus ground truth"
        ),
        "catastrophic_failure": (
            "rotation_error_deg > 20 or "
            "translation_error_cm > 10"
        ),
        "mask_score": "cross-render mask IoU",
        "depth_score": (
            "1 - normalized cross depth loss"
        ),
        "source_validation_score": (
            "1 - self-alignment total loss"
        ),
        "bidirectional_consistency_score": (
            "1 - normalized consistency loss"
        ),
        "selector_correct": (
            "Pareto dominance on GT rotation and "
            "translation errors; blank for a tie or "
            "rotation/translation trade-off"
        ),
        "confidence_raw": (
            "relative candidate confidence; "
            "not a calibrated probability"
        ),
        "runtime_scope": (
            "Measured end-to-end timings are recorded "
            "only on the dual method row. Ref-only and "
            "query-only rows are analytical path "
            "projections, not separately timed runs."
        ),
        "batch_runtime_accounting": (
            "In batch mode total_time_sec is the "
            "observed query wall time excluding the "
            "once-only shared reference. Use "
            "shared_reference_time_sec once per batch, "
            "or standalone_equivalent_time_sec for an "
            "independent-pair cost estimate."
        ),
        "csv_join_model": (
            "Join pair_results, path_results, "
            "proxy_results, and run_metadata by run_id "
            "and pair_id. Full provenance lives in the "
            "pair table and run metadata."
        ),
        "path_candidate_scope": (
            "Reference/query path rows are the two "
            "components of the best accepted dual pair, "
            "or the best overall pair when rejected."
        ),
        "add_adds_scope": (
            "ADD, ADD-S, and symmetry-aware ADD(-S)-0.1d are stored "
            "in compact result.json and linemod_all_results outputs. "
            "The legacy research pair columns remain unpopulated."
        ),
        "intentionally_unavailable": [
            "confidence_calibrated until calibration "
            "is fitted on multiple pairs",
            "trusted_surface_ratio until trusted "
            "surface construction is implemented",
            "cross-process peak GPU memory",
        ],
    }

    with definitions_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            definitions,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return context


