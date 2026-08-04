from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add BOP-LINEMOD ADD/ADD-S-0.1d metrics to existing "
            "compact result.json files without rerunning inference."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="LINEMOD all-run root or one object output root.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "datasets",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--models-directory",
        type=Path,
        default=None,
        help="Defaults to DATASET_ROOT/models_eval.",
    )
    return parser.parse_args()


def _result_paths_from_summary(
    output_root: Path,
) -> tuple[Path, ...] | None:
    summary_path = output_root / "linemod_all_summary.json"
    if not summary_path.is_file():
        return None

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(
            f"LINEMOD summary has no records list: {summary_path}"
        )

    paths: list[Path] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        object_output_root = record.get("output_root")
        query_image_ids = record.get("query_image_ids")
        object_id = record.get("object_id")
        if (
            object_output_root is None
            or object_id is None
            or not isinstance(query_image_ids, list)
        ):
            continue
        object_output_root_path = Path(object_output_root)
        for query_image_id in query_image_ids:
            result_path = (
                object_output_root_path
                / "results"
                / (
                    f"q{int(object_id):06d}_"
                    f"{int(query_image_id):06d}_i00"
                )
                / "result.json"
            )
            if result_path.is_file():
                paths.append(result_path.resolve())
    return tuple(sorted(set(paths)))


def _result_paths(output_root: Path) -> tuple[Path, ...]:
    summary_paths = _result_paths_from_summary(output_root)
    if summary_paths is not None:
        return summary_paths

    candidates = {
        *output_root.glob("results/*/result.json"),
        *output_root.glob("object_*/results/*/result.json"),
    }
    return tuple(sorted(path.resolve() for path in candidates))


def main() -> int:
    from linemod_all_runner import rebuild_linemod_aggregate_results
    from result_storage import backfill_compact_result_add_metrics

    arguments = _parse_arguments()
    output_root = arguments.output_root.expanduser().resolve()
    dataset_root = arguments.dataset_root.expanduser().resolve()
    paths = _result_paths(output_root)
    if not paths:
        raise FileNotFoundError(
            f"No compact result.json files found under {output_root}"
        )

    evaluated_count = 0
    missing_pose_count = 0
    failures: list[tuple[Path, str]] = []
    for index, result_path in enumerate(paths, start=1):
        try:
            payload = backfill_compact_result_add_metrics(
                result_path=result_path,
                dataset_root=dataset_root,
                split=arguments.split,
                models_directory=arguments.models_directory,
            )
            evaluation = payload["evaluation"]
            evaluated_count += 1
            print(
                f"[ADD backfill {index}/{len(paths)}] "
                f"object={payload['object_id']:02d} "
                f"metric={evaluation['add_metric_used']} "
                f"normalized={evaluation['add_or_adds_normalized']:.6f} "
                f"passed={evaluation['add_or_adds_0_1d_passed']}"
            )
        except ValueError as error:
            if "has no estimated pose" in str(error):
                missing_pose_count += 1
                continue
            failures.append((result_path, str(error)))
        except Exception as error:
            failures.append(
                (
                    result_path,
                    f"{type(error).__name__}: {error}",
                )
            )

    aggregate_summary = output_root / "linemod_all_summary.json"
    if aggregate_summary.is_file():
        json_path, csv_path = rebuild_linemod_aggregate_results(
            output_root
        )
        print(f"[ADD aggregate JSON] {json_path}")
        print(f"[ADD aggregate CSV] {csv_path}")

    print(
        "[ADD backfill complete] "
        f"evaluated={evaluated_count}, "
        f"missing_pose={missing_pose_count}, "
        f"failed={len(failures)}"
    )
    for result_path, message in failures[:20]:
        print(
            f"[ADD backfill warning] {result_path}: {message}",
            file=sys.stderr,
        )
    if len(failures) > 20:
        print(
            "[ADD backfill warning] "
            f"{len(failures) - 20} additional failures omitted.",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
