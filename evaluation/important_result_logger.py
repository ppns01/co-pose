from __future__ import annotations

"""
Compact, Git-friendly research result logger.

목적
----
- 전체 디버그 산출물은 outputs/에 유지한다.
- 논문·GitHub에서 확인할 핵심 결과만 published_results/에 저장한다.
- CONSISTENT뿐 아니라 REJECT도 모두 저장한다.
- visible-scale OFF/ON 결과를 독립 행으로 보존한다.
- 같은 pair의 OFF/ON 결과가 모두 존재하면 비교 CSV/JSON을 자동 갱신한다.
- 최종/경로별/GT pose 행렬을 JSON 안에도 기록하고, 압축 NPZ로도 저장한다.

외부 의존성
-----------
- numpy
- Linux에서는 fcntl 파일 잠금을 사용한다. 다른 OS에서는 잠금 없이 동작한다.
"""

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

try:  # Linux/Unix batch 실행에서 CSV 동시 쓰기 보호
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


SCHEMA_VERSION = 1
POSE_CONVENTION = "T_query_camera_from_reference_camera"
TRANSLATION_UNIT = "meter"
VALID_STATUSES = {"CONSISTENT", "REJECT"}
VALID_VARIANTS = {"OFF", "ON"}


# CSV는 GitHub에서 바로 비교할 수 있도록 핵심 scalar만 유지한다.
PAIR_CSV_FIELDS = (
    "schema_version",
    "result_key",
    "timestamp_utc",
    "experiment_name",
    "comparison_group",
    "variant",
    "visible_scale_enabled",
    "run_id",
    "pair_id",
    "dataset",
    "split",
    "object_id",
    "object_name",
    "reference_scene_id",
    "reference_image_id",
    "reference_instance_index",
    "query_scene_id",
    "query_image_id",
    "query_instance_index",
    "method",
    "status",
    "rejection_reason",
    "selected_path",
    "git_commit",
    "config_hash",
    "config_snapshot",
    "pose_archive",
    "pair_json",
    # Reference visible-scale diagnostics
    "reference_initial_scale_m",
    "reference_estimated_scale_m",
    "reference_selected_scale_m",
    "reference_alpha",
    "reference_log_scale_delta",
    "reference_correspondence_count",
    "reference_inlier_count",
    "reference_inlier_ratio",
    "reference_spatial_coverage",
    "reference_median_residual_3d_mm",
    "reference_p90_residual_3d_mm",
    "reference_median_abs_residual_object_x_mm",
    "reference_median_abs_residual_object_y_mm",
    "reference_median_abs_residual_object_z_mm",
    "reference_correction_accepted",
    "reference_scale_rejection_reason",
    # Query visible-scale diagnostics
    "query_initial_scale_m",
    "query_estimated_scale_m",
    "query_selected_scale_m",
    "query_alpha",
    "query_log_scale_delta",
    "query_correspondence_count",
    "query_inlier_count",
    "query_inlier_ratio",
    "query_spatial_coverage",
    "query_median_residual_3d_mm",
    "query_p90_residual_3d_mm",
    "query_median_abs_residual_object_x_mm",
    "query_median_abs_residual_object_y_mm",
    "query_median_abs_residual_object_z_mm",
    "query_correction_accepted",
    "query_scale_rejection_reason",
    "proxy_scale_log_difference",
    "proxy_scale_difference_percent",
    # Path consistency / selection
    "path_rotation_difference_deg",
    "path_translation_difference_cm",
    "normalized_translation_difference",
    "selection_score",
    "top1_top2_margin",
    # GT errors
    "reference_path_rotation_error_deg",
    "reference_path_translation_error_cm",
    "reference_path_translation_error_x_cm",
    "reference_path_translation_error_y_cm",
    "reference_path_translation_error_z_cm",
    "query_path_rotation_error_deg",
    "query_path_translation_error_cm",
    "query_path_translation_error_x_cm",
    "query_path_translation_error_y_cm",
    "query_path_translation_error_z_cm",
    "final_rotation_error_deg",
    "final_translation_error_cm",
    "final_translation_error_x_cm",
    "final_translation_error_y_cm",
    "final_translation_error_z_cm",
    # Runtime
    "total_time_sec",
    "visible_scale_time_sec",
    "foundationpose_time_sec",
)


COMPARISON_CSV_FIELDS = (
    "schema_version",
    "comparison_key",
    "comparison_group",
    "pair_id",
    "method",
    "off_result_key",
    "on_result_key",
    "status_off",
    "status_on",
    "status_transition",
    "reference_path_rotation_error_deg_off",
    "reference_path_rotation_error_deg_on",
    "reference_path_rotation_error_deg_change",
    "reference_path_translation_error_cm_off",
    "reference_path_translation_error_cm_on",
    "reference_path_translation_error_cm_change",
    "query_path_rotation_error_deg_off",
    "query_path_rotation_error_deg_on",
    "query_path_rotation_error_deg_change",
    "query_path_translation_error_cm_off",
    "query_path_translation_error_cm_on",
    "query_path_translation_error_cm_change",
    "final_rotation_error_deg_off",
    "final_rotation_error_deg_on",
    "final_rotation_error_deg_change",
    "final_translation_error_cm_off",
    "final_translation_error_cm_on",
    "final_translation_error_cm_change",
    "path_rotation_difference_deg_off",
    "path_rotation_difference_deg_on",
    "path_rotation_difference_deg_change",
    "path_translation_difference_cm_off",
    "path_translation_difference_cm_on",
    "path_translation_difference_cm_change",
    "normalized_translation_difference_off",
    "normalized_translation_difference_on",
    "normalized_translation_difference_change",
    "proxy_scale_difference_percent_off",
    "proxy_scale_difference_percent_on",
    "proxy_scale_difference_percent_change",
    "total_time_sec_off",
    "total_time_sec_on",
    "total_time_sec_change",
)


@dataclass(frozen=True)
class ScaleDiagnostics:
    """한 proxy의 visible-correspondence scale 진단값."""

    initial_scale_m: float | None = None
    estimated_scale_m: float | None = None
    selected_scale_m: float | None = None

    alpha: float | None = None
    log_scale_delta: float | None = None

    correspondence_count: int | None = None
    inlier_count: int | None = None
    inlier_ratio: float | None = None
    spatial_coverage: float | None = None

    median_residual_3d_m: float | None = None
    p90_residual_3d_m: float | None = None

    median_abs_residual_object_x_m: float | None = None
    median_abs_residual_object_y_m: float | None = None
    median_abs_residual_object_z_m: float | None = None

    correction_accepted: bool | None = None
    rejection_reason: str | None = None

    def normalized(self) -> "ScaleDiagnostics":
        """가능한 경우 alpha와 log delta를 자동 계산한다."""

        alpha = self.alpha
        log_delta = self.log_scale_delta

        if (
            alpha is None
            and self.initial_scale_m is not None
            and self.estimated_scale_m is not None
            and self.initial_scale_m > 0.0
        ):
            alpha = self.estimated_scale_m / self.initial_scale_m

        if log_delta is None and alpha is not None and alpha > 0.0:
            log_delta = math.log(alpha)

        return ScaleDiagnostics(
            initial_scale_m=self.initial_scale_m,
            estimated_scale_m=self.estimated_scale_m,
            selected_scale_m=self.selected_scale_m,
            alpha=alpha,
            log_scale_delta=log_delta,
            correspondence_count=self.correspondence_count,
            inlier_count=self.inlier_count,
            inlier_ratio=self.inlier_ratio,
            spatial_coverage=self.spatial_coverage,
            median_residual_3d_m=self.median_residual_3d_m,
            p90_residual_3d_m=self.p90_residual_3d_m,
            median_abs_residual_object_x_m=(
                self.median_abs_residual_object_x_m
            ),
            median_abs_residual_object_y_m=(
                self.median_abs_residual_object_y_m
            ),
            median_abs_residual_object_z_m=(
                self.median_abs_residual_object_z_m
            ),
            correction_accepted=self.correction_accepted,
            rejection_reason=self.rejection_reason,
        )


@dataclass(frozen=True)
class PoseErrorMetrics:
    rotation_error_deg: float | None = None
    translation_error_cm: float | None = None
    translation_error_x_cm: float | None = None
    translation_error_y_cm: float | None = None
    translation_error_z_cm: float | None = None


@dataclass(frozen=True)
class ConsistencyMetrics:
    rotation_difference_deg: float | None = None
    translation_difference_cm: float | None = None
    normalized_translation_difference: float | None = None
    selection_score: float | None = None
    top1_top2_margin: float | None = None


@dataclass(frozen=True)
class RuntimeMetrics:
    total_time_sec: float | None = None
    visible_scale_time_sec: float | None = None
    foundationpose_time_sec: float | None = None


@dataclass(frozen=True)
class ImportantPairResult:
    """GitHub에 저장할 pair 단위 핵심 결과."""

    experiment_name: str
    comparison_group: str
    variant: str
    visible_scale_enabled: bool

    run_id: str
    pair_id: str
    dataset: str
    split: str
    object_id: int
    object_name: str

    reference_scene_id: int
    reference_image_id: int
    reference_instance_index: int
    query_scene_id: int
    query_image_id: int
    query_instance_index: int

    method: str
    status: str
    rejection_reason: str | None = None
    selected_path: str | None = None

    git_commit: str | None = None
    config_hash: str | None = None

    reference_scale: ScaleDiagnostics = field(
        default_factory=ScaleDiagnostics
    )
    query_scale: ScaleDiagnostics = field(
        default_factory=ScaleDiagnostics
    )

    consistency: ConsistencyMetrics = field(
        default_factory=ConsistencyMetrics
    )

    reference_path_error: PoseErrorMetrics = field(
        default_factory=PoseErrorMetrics
    )
    query_path_error: PoseErrorMetrics = field(
        default_factory=PoseErrorMetrics
    )
    final_error: PoseErrorMetrics = field(
        default_factory=PoseErrorMetrics
    )

    runtime: RuntimeMetrics = field(
        default_factory=RuntimeMetrics
    )

    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_status = self.status.upper()
        normalized_variant = self.variant.upper()

        if normalized_status not in VALID_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(VALID_STATUSES)}: "
                f"{self.status!r}"
            )

        if normalized_variant not in VALID_VARIANTS:
            raise ValueError(
                f"variant must be OFF or ON: {self.variant!r}"
            )

        if self.visible_scale_enabled != (normalized_variant == "ON"):
            raise ValueError(
                "variant and visible_scale_enabled disagree: "
                f"variant={self.variant}, "
                f"visible_scale_enabled={self.visible_scale_enabled}"
            )

        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty.")
        if not self.comparison_group.strip():
            raise ValueError("comparison_group must not be empty.")
        if not self.pair_id.strip():
            raise ValueError("pair_id must not be empty.")


@dataclass(frozen=True)
class ImportantResultSaveResult:
    pair_json_path: Path
    pose_archive_path: Path | None
    pair_csv_path: Path
    comparison_csv_path: Path
    comparison_json_path: Path
    run_summary_path: Path
    config_snapshot_path: Path | None


class _FileLock:
    """작은 CSV/JSON 파일 갱신을 위한 프로세스 잠금."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: Any = None

    def __enter__(self) -> "_FileLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a+", encoding="utf-8")
        if fcntl is not None:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._file is not None and fcntl is not None:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        if self._file is not None:
            self._file.close()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(value).strip()
    )
    normalized = normalized.strip("_.")
    if not normalized:
        raise ValueError(f"Cannot build a safe slug from {value!r}.")
    return normalized


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(
            _json_safe(value),
            temporary,
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
        )
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        writer = csv.DictWriter(
            temporary,
            fieldnames=list(fields),
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _csv_value(row.get(field)) for field in fields}
            )
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _csv_value(value: Any) -> Any:
    value = _json_safe(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None:
        return None
    return int(number)


def _boolean(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _validate_pose_matrix(
    matrix: Any,
    *,
    name: str,
) -> np.ndarray | None:
    if matrix is None:
        return None
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4): {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    if not np.allclose(array[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous last row.")
    return array


def detect_git_commit(repo_root: Path | str = ".") -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = output.strip()
    return value or None


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def _build_result_key(result: ImportantPairResult) -> str:
    commit = (result.git_commit or "nogit")[:8]
    config = (result.config_hash or "noconfig")[:8]
    return _slug(
        "__".join(
            (
                result.comparison_group,
                result.pair_id,
                result.method,
                result.variant.upper(),
                commit,
                config,
            )
        )
    )


def _scale_to_flat(prefix: str, value: ScaleDiagnostics) -> dict[str, Any]:
    value = value.normalized()
    return {
        f"{prefix}_initial_scale_m": value.initial_scale_m,
        f"{prefix}_estimated_scale_m": value.estimated_scale_m,
        f"{prefix}_selected_scale_m": value.selected_scale_m,
        f"{prefix}_alpha": value.alpha,
        f"{prefix}_log_scale_delta": value.log_scale_delta,
        f"{prefix}_correspondence_count": value.correspondence_count,
        f"{prefix}_inlier_count": value.inlier_count,
        f"{prefix}_inlier_ratio": value.inlier_ratio,
        f"{prefix}_spatial_coverage": value.spatial_coverage,
        f"{prefix}_median_residual_3d_mm": (
            None
            if value.median_residual_3d_m is None
            else value.median_residual_3d_m * 1000.0
        ),
        f"{prefix}_p90_residual_3d_mm": (
            None
            if value.p90_residual_3d_m is None
            else value.p90_residual_3d_m * 1000.0
        ),
        f"{prefix}_median_abs_residual_object_x_mm": (
            None
            if value.median_abs_residual_object_x_m is None
            else value.median_abs_residual_object_x_m * 1000.0
        ),
        f"{prefix}_median_abs_residual_object_y_mm": (
            None
            if value.median_abs_residual_object_y_m is None
            else value.median_abs_residual_object_y_m * 1000.0
        ),
        f"{prefix}_median_abs_residual_object_z_mm": (
            None
            if value.median_abs_residual_object_z_m is None
            else value.median_abs_residual_object_z_m * 1000.0
        ),
        f"{prefix}_correction_accepted": value.correction_accepted,
        f"{prefix}_scale_rejection_reason": value.rejection_reason,
    }


def _pose_error_to_flat(prefix: str, value: PoseErrorMetrics) -> dict[str, Any]:
    return {
        f"{prefix}_rotation_error_deg": value.rotation_error_deg,
        f"{prefix}_translation_error_cm": value.translation_error_cm,
        f"{prefix}_translation_error_x_cm": value.translation_error_x_cm,
        f"{prefix}_translation_error_y_cm": value.translation_error_y_cm,
        f"{prefix}_translation_error_z_cm": value.translation_error_z_cm,
    }


def _proxy_scale_values(
    reference: ScaleDiagnostics,
    query: ScaleDiagnostics,
) -> tuple[float | None, float | None]:
    reference = reference.normalized()
    query = query.normalized()

    reference_scale = (
        reference.estimated_scale_m
        if reference.estimated_scale_m is not None
        else reference.selected_scale_m
    )
    query_scale = (
        query.estimated_scale_m
        if query.estimated_scale_m is not None
        else query.selected_scale_m
    )

    if (
        reference_scale is None
        or query_scale is None
        or reference_scale <= 0.0
        or query_scale <= 0.0
    ):
        return None, None

    log_difference = math.log(reference_scale / query_scale)
    percent_difference = (math.exp(abs(log_difference)) - 1.0) * 100.0
    return log_difference, percent_difference


def _upsert_row(
    rows: list[dict[str, Any]],
    new_row: Mapping[str, Any],
    *,
    key_field: str,
) -> list[dict[str, Any]]:
    key = str(new_row[key_field])
    filtered = [row for row in rows if str(row.get(key_field)) != key]
    filtered.append(dict(new_row))
    return sorted(filtered, key=lambda row: str(row.get(key_field, "")))


def _difference(on_value: Any, off_value: Any) -> float | None:
    on_number = _number(on_value)
    off_number = _number(off_value)
    if on_number is None or off_number is None:
        return None
    return on_number - off_number


def _comparison_key(row: Mapping[str, Any]) -> str:
    return "__".join(
        (
            str(row.get("comparison_group", "")),
            str(row.get("pair_id", "")),
            str(row.get("method", "")),
        )
    )


def _build_comparison_rows(
    pair_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}

    # 같은 variant가 여러 번 존재하면 timestamp상 마지막 행을 사용한다.
    sorted_rows = sorted(
        pair_rows,
        key=lambda row: str(row.get("timestamp_utc", "")),
    )
    for row in sorted_rows:
        variant = str(row.get("variant", "")).upper()
        if variant not in VALID_VARIANTS:
            continue
        grouped.setdefault(_comparison_key(row), {})[variant] = row

    comparisons: list[dict[str, Any]] = []
    metrics = (
        "reference_path_rotation_error_deg",
        "reference_path_translation_error_cm",
        "query_path_rotation_error_deg",
        "query_path_translation_error_cm",
        "final_rotation_error_deg",
        "final_translation_error_cm",
        "path_rotation_difference_deg",
        "path_translation_difference_cm",
        "normalized_translation_difference",
        "proxy_scale_difference_percent",
        "total_time_sec",
    )

    for key, variants in sorted(grouped.items()):
        off = variants.get("OFF")
        on = variants.get("ON")
        if off is None or on is None:
            continue

        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "comparison_key": key,
            "comparison_group": on.get("comparison_group"),
            "pair_id": on.get("pair_id"),
            "method": on.get("method"),
            "off_result_key": off.get("result_key"),
            "on_result_key": on.get("result_key"),
            "status_off": off.get("status"),
            "status_on": on.get("status"),
            "status_transition": (
                f"{off.get('status')}->{on.get('status')}"
            ),
        }

        for metric in metrics:
            row[f"{metric}_off"] = off.get(metric)
            row[f"{metric}_on"] = on.get(metric)
            row[f"{metric}_change"] = _difference(
                on.get(metric), off.get(metric)
            )

        comparisons.append(row)

    return comparisons


def _aggregate(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _build_run_summary(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "updated_at_utc": _now_utc_iso(),
        "pair_result_count": len(pair_rows),
        "variants": {},
    }

    for variant in ("OFF", "ON"):
        rows = [
            row
            for row in pair_rows
            if str(row.get("variant", "")).upper() == variant
        ]
        consistent = sum(
            1 for row in rows if row.get("status") == "CONSISTENT"
        )
        rejected = sum(1 for row in rows if row.get("status") == "REJECT")

        metric_names = (
            "reference_path_rotation_error_deg",
            "reference_path_translation_error_cm",
            "query_path_rotation_error_deg",
            "query_path_translation_error_cm",
            "final_rotation_error_deg",
            "final_translation_error_cm",
            "path_rotation_difference_deg",
            "path_translation_difference_cm",
            "normalized_translation_difference",
            "proxy_scale_difference_percent",
            "total_time_sec",
        )

        summary["variants"][variant] = {
            "count": len(rows),
            "consistent_count": consistent,
            "reject_count": rejected,
            "consistency_rate": (
                consistent / len(rows) if rows else None
            ),
            "metrics": {
                metric: _aggregate(
                    value
                    for row in rows
                    if (value := _number(row.get(metric))) is not None
                )
                for metric in metric_names
            },
        }

    comparison_rows = _build_comparison_rows(pair_rows)
    summary["off_on_comparison_count"] = len(comparison_rows)
    summary["status_transitions"] = {
        transition: sum(
            1
            for row in comparison_rows
            if row.get("status_transition") == transition
        )
        for transition in (
            "REJECT->REJECT",
            "REJECT->CONSISTENT",
            "CONSISTENT->REJECT",
            "CONSISTENT->CONSISTENT",
        )
    }
    return summary


def save_important_pair_result(
    *,
    output_root: Path | str,
    result: ImportantPairResult,
    reference_pose_query_from_reference: Any = None,
    query_pose_query_from_reference: Any = None,
    final_pose_query_from_reference: Any = None,
    ground_truth_pose_query_from_reference: Any = None,
    config_path: Path | str | None = None,
) -> ImportantResultSaveResult:
    """
    핵심 결과를 published_results/<experiment_name>/ 아래에 저장한다.

    저장되는 것
    -----------
    1. pair별 JSON: scalar 결과 + 모든 pose matrix
    2. pair별 압축 NPZ: reference/query/final/GT pose
    3. pair_results.csv: OFF/ON 및 REJECT를 포함한 독립 행
    4. off_on_comparison.csv/json: 같은 pair의 OFF/ON 자동 비교
    5. run_summary.json: variant별 성공률과 주요 통계
    6. 선택적 config snapshot
    """

    output_root = Path(output_root).expanduser().resolve()
    experiment_root = output_root / _slug(result.experiment_name)
    pairs_root = experiment_root / "pairs"
    poses_root = experiment_root / "poses"
    configs_root = experiment_root / "configs"

    for directory in (pairs_root, poses_root, configs_root):
        directory.mkdir(parents=True, exist_ok=True)

    reference_pose = _validate_pose_matrix(
        reference_pose_query_from_reference,
        name="reference_pose_query_from_reference",
    )
    query_pose = _validate_pose_matrix(
        query_pose_query_from_reference,
        name="query_pose_query_from_reference",
    )
    final_pose = _validate_pose_matrix(
        final_pose_query_from_reference,
        name="final_pose_query_from_reference",
    )
    ground_truth_pose = _validate_pose_matrix(
        ground_truth_pose_query_from_reference,
        name="ground_truth_pose_query_from_reference",
    )

    status = result.status.upper()
    variant = result.variant.upper()

    if status == "CONSISTENT" and final_pose is None:
        raise ValueError("CONSISTENT result requires a final pose matrix.")
    if status == "REJECT" and final_pose is not None:
        raise ValueError("REJECT result must not contain a final pose matrix.")

    normalized_reference_scale = result.reference_scale.normalized()
    normalized_query_scale = result.query_scale.normalized()
    proxy_log_difference, proxy_percent_difference = _proxy_scale_values(
        normalized_reference_scale,
        normalized_query_scale,
    )

    result_key = _build_result_key(result)
    pair_json_path = pairs_root / f"{result_key}.json"

    pose_arrays = {
        "reference_pose_query_from_reference": reference_pose,
        "query_pose_query_from_reference": query_pose,
        "final_pose_query_from_reference": final_pose,
        "ground_truth_pose_query_from_reference": ground_truth_pose,
    }
    valid_pose_arrays = {
        key: value for key, value in pose_arrays.items() if value is not None
    }

    pose_archive_path: Path | None = None
    if valid_pose_arrays:
        pose_archive_path = poses_root / f"{result_key}.npz"
        np.savez_compressed(pose_archive_path, **valid_pose_arrays)

    config_snapshot_path: Path | None = None
    config_hash = result.config_hash
    if config_path is not None:
        source_config = Path(config_path).expanduser().resolve()
        if not source_config.is_file():
            raise FileNotFoundError(f"Config file not found: {source_config}")
        config_hash = config_hash or sha256_file(source_config)
        suffix = source_config.suffix or ".yaml"
        config_snapshot_path = configs_root / f"{config_hash[:16]}{suffix}"
        if not config_snapshot_path.exists():
            shutil.copy2(source_config, config_snapshot_path)

    timestamp = _now_utc_iso()
    pair_payload = {
        "schema_version": SCHEMA_VERSION,
        "result_key": result_key,
        "timestamp_utc": timestamp,
        "experiment": {
            "name": result.experiment_name,
            "comparison_group": result.comparison_group,
            "variant": variant,
            "visible_scale_enabled": result.visible_scale_enabled,
            "run_id": result.run_id,
            "method": result.method,
            "git_commit": result.git_commit,
            "config_hash": config_hash,
            "config_snapshot": _relative_path(
                config_snapshot_path, experiment_root
            ),
        },
        "pair": {
            "pair_id": result.pair_id,
            "dataset": result.dataset,
            "split": result.split,
            "object_id": result.object_id,
            "object_name": result.object_name,
            "reference": {
                "scene_id": result.reference_scene_id,
                "image_id": result.reference_image_id,
                "instance_index": result.reference_instance_index,
            },
            "query": {
                "scene_id": result.query_scene_id,
                "image_id": result.query_image_id,
                "instance_index": result.query_instance_index,
            },
        },
        "selection": {
            "status": status,
            "rejection_reason": result.rejection_reason,
            "selected_path": result.selected_path,
            **asdict(result.consistency),
        },
        "scale": {
            "reference": asdict(normalized_reference_scale),
            "query": asdict(normalized_query_scale),
            "proxy_scale_log_difference": proxy_log_difference,
            "proxy_scale_difference_percent": proxy_percent_difference,
        },
        "gt_error": {
            "reference_path": asdict(result.reference_path_error),
            "query_path": asdict(result.query_path_error),
            "final": asdict(result.final_error),
        },
        "runtime": asdict(result.runtime),
        "pose": {
            "convention": POSE_CONVENTION,
            "translation_unit": TRANSLATION_UNIT,
            "archive": _relative_path(pose_archive_path, experiment_root),
            "reference_path_matrix": (
                None if reference_pose is None else reference_pose.tolist()
            ),
            "query_path_matrix": (
                None if query_pose is None else query_pose.tolist()
            ),
            "final_selected_matrix": (
                None if final_pose is None else final_pose.tolist()
            ),
            "ground_truth_matrix": (
                None if ground_truth_pose is None else ground_truth_pose.tolist()
            ),
        },
        "extra": dict(result.extra),
    }
    _atomic_write_json(pair_json_path, pair_payload)

    pair_csv_path = experiment_root / "pair_results.csv"
    comparison_csv_path = experiment_root / "off_on_comparison.csv"
    comparison_json_path = experiment_root / "off_on_comparison.json"
    run_summary_path = experiment_root / "run_summary.json"
    lock_path = experiment_root / ".important_result_logger.lock"

    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_key": result_key,
        "timestamp_utc": timestamp,
        "experiment_name": result.experiment_name,
        "comparison_group": result.comparison_group,
        "variant": variant,
        "visible_scale_enabled": result.visible_scale_enabled,
        "run_id": result.run_id,
        "pair_id": result.pair_id,
        "dataset": result.dataset,
        "split": result.split,
        "object_id": result.object_id,
        "object_name": result.object_name,
        "reference_scene_id": result.reference_scene_id,
        "reference_image_id": result.reference_image_id,
        "reference_instance_index": result.reference_instance_index,
        "query_scene_id": result.query_scene_id,
        "query_image_id": result.query_image_id,
        "query_instance_index": result.query_instance_index,
        "method": result.method,
        "status": status,
        "rejection_reason": result.rejection_reason,
        "selected_path": result.selected_path,
        "git_commit": result.git_commit,
        "config_hash": config_hash,
        "config_snapshot": _relative_path(
            config_snapshot_path, experiment_root
        ),
        "pose_archive": _relative_path(pose_archive_path, experiment_root),
        "pair_json": _relative_path(pair_json_path, experiment_root),
        **_scale_to_flat("reference", normalized_reference_scale),
        **_scale_to_flat("query", normalized_query_scale),
        "proxy_scale_log_difference": proxy_log_difference,
        "proxy_scale_difference_percent": proxy_percent_difference,
        "path_rotation_difference_deg": (
            result.consistency.rotation_difference_deg
        ),
        "path_translation_difference_cm": (
            result.consistency.translation_difference_cm
        ),
        "normalized_translation_difference": (
            result.consistency.normalized_translation_difference
        ),
        "selection_score": result.consistency.selection_score,
        "top1_top2_margin": result.consistency.top1_top2_margin,
        **_pose_error_to_flat(
            "reference_path", result.reference_path_error
        ),
        **_pose_error_to_flat("query_path", result.query_path_error),
        **_pose_error_to_flat("final", result.final_error),
        "total_time_sec": result.runtime.total_time_sec,
        "visible_scale_time_sec": result.runtime.visible_scale_time_sec,
        "foundationpose_time_sec": result.runtime.foundationpose_time_sec,
    }

    with _FileLock(lock_path):
        pair_rows = _read_csv(pair_csv_path)
        pair_rows = _upsert_row(
            pair_rows,
            row,
            key_field="result_key",
        )
        _atomic_write_csv(pair_csv_path, pair_rows, PAIR_CSV_FIELDS)

        comparison_rows = _build_comparison_rows(pair_rows)
        _atomic_write_csv(
            comparison_csv_path,
            comparison_rows,
            COMPARISON_CSV_FIELDS,
        )
        _atomic_write_json(comparison_json_path, comparison_rows)
        _atomic_write_json(run_summary_path, _build_run_summary(pair_rows))

    return ImportantResultSaveResult(
        pair_json_path=pair_json_path,
        pose_archive_path=pose_archive_path,
        pair_csv_path=pair_csv_path,
        comparison_csv_path=comparison_csv_path,
        comparison_json_path=comparison_json_path,
        run_summary_path=run_summary_path,
        config_snapshot_path=config_snapshot_path,
    )


def rebuild_important_result_indices(
    experiment_root: Path | str,
) -> tuple[Path, Path, Path]:
    """pair_results.csv를 기준으로 비교표와 summary를 다시 생성한다."""

    experiment_root = Path(experiment_root).expanduser().resolve()
    pair_csv_path = experiment_root / "pair_results.csv"
    if not pair_csv_path.is_file():
        raise FileNotFoundError(f"pair_results.csv not found: {pair_csv_path}")

    comparison_csv_path = experiment_root / "off_on_comparison.csv"
    comparison_json_path = experiment_root / "off_on_comparison.json"
    run_summary_path = experiment_root / "run_summary.json"
    lock_path = experiment_root / ".important_result_logger.lock"

    with _FileLock(lock_path):
        pair_rows = _read_csv(pair_csv_path)
        comparison_rows = _build_comparison_rows(pair_rows)
        _atomic_write_csv(
            comparison_csv_path,
            comparison_rows,
            COMPARISON_CSV_FIELDS,
        )
        _atomic_write_json(comparison_json_path, comparison_rows)
        _atomic_write_json(run_summary_path, _build_run_summary(pair_rows))

    return comparison_csv_path, comparison_json_path, run_summary_path


__all__ = [
    "ConsistencyMetrics",
    "ImportantPairResult",
    "ImportantResultSaveResult",
    "PoseErrorMetrics",
    "RuntimeMetrics",
    "ScaleDiagnostics",
    "detect_git_commit",
    "rebuild_important_result_indices",
    "save_important_pair_result",
    "sha256_file",
]
