from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill ADD/ADD-S, export every saved relative pose to "
            "BOP format, and run official eval_bop19_pose.py."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "datasets",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--models-directory", type=Path, default=None)
    parser.add_argument(
        "--renderer-type",
        choices=("vispy", "cpp", "python"),
        default="python",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    return parser.parse_args()


def _result_paths_from_summary(
    output_root: Path,
) -> tuple[Path, ...] | None:
    summary_path = output_root / "linemod_all_summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = summary.get("records")
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
        for query_image_id in query_image_ids:
            result_path = (
                Path(object_output_root)
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


def find_compact_result_paths(output_root: Path) -> tuple[Path, ...]:
    summary_paths = _result_paths_from_summary(output_root)
    if summary_paths is not None:
        return summary_paths
    candidates = {
        *output_root.glob("results/*/result.json"),
        *output_root.glob("object_*/results/*/result.json"),
    }
    return tuple(sorted(path.resolve() for path in candidates))


def _frame_spec_from_payload(
    *,
    payload: dict[str, Any],
    frame_name: str,
    dataset_root: Path,
    split: str,
) -> Any:
    from evaluation.relative_pose_evaluator import BOPFrameGTSpec

    frame = payload.get(frame_name)
    if not isinstance(frame, dict):
        raise ValueError(f"Compact result has no {frame_name} frame.")
    return BOPFrameGTSpec(
        dataset_root=dataset_root,
        split=split,
        scene_id=int(frame["scene_id"]),
        image_id=int(frame["image_id"]),
        object_id=int(payload["object_id"]),
        instance_index=int(frame.get("instance_index", 0)),
    )


def _export_bop19_csv(
    *,
    result_paths: Sequence[Path],
    dataset_root: Path,
    split: str,
    output_directory: Path,
) -> tuple[Path, int, int]:
    from bop_toolkit_lib import inout
    from evaluation.linemod_bop_evaluator import (
        relative_pose_to_query_absolute_mm,
    )

    estimates: list[dict[str, Any]] = []
    missing_pose_count = 0
    for result_path in result_paths:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        pose_values = payload.get("relative_pose_query_from_reference")
        if pose_values is None:
            missing_pose_count += 1
            continue
        reference_frame = _frame_spec_from_payload(
            payload=payload,
            frame_name="reference",
            dataset_root=dataset_root,
            split=split,
        )
        query = payload.get("query")
        if not isinstance(query, dict):
            raise ValueError(f"Compact result has no query: {result_path}")
        query_pose_mm = relative_pose_to_query_absolute_mm(
            predicted_relative_pose=np.asarray(
                pose_values,
                dtype=np.float64,
            ),
            reference_frame=reference_frame,
        )
        estimates.append(
            {
                "scene_id": int(query["scene_id"]),
                "im_id": int(query["image_id"]),
                "obj_id": int(payload["object_id"]),
                "score": 1.0,
                "R": query_pose_mm[:3, :3],
                "t": query_pose_mm[:3, 3:4],
                "time": -1.0,
            }
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "selfmesh_lm-test.csv"
    inout.save_bop_results(csv_path, estimates, version="bop19")
    return csv_path, len(estimates), missing_pose_count


def _target_entries(
    *,
    output_root: Path,
    result_paths: Sequence[Path],
) -> list[dict[str, int]]:
    summary_path = output_root / "linemod_all_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        records = summary.get("records")
        if isinstance(records, list):
            targets = []
            for record in records:
                if not isinstance(record, dict):
                    continue
                query_image_ids = record.get("query_image_ids")
                if not isinstance(query_image_ids, list):
                    continue
                object_id = int(record["object_id"])
                targets.extend(
                    {
                        "scene_id": object_id,
                        "im_id": int(image_id),
                        "obj_id": object_id,
                        "inst_count": 1,
                    }
                    for image_id in query_image_ids
                )
            return targets

    targets = []
    for result_path in result_paths:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        query = payload.get("query")
        if not isinstance(query, dict):
            continue
        targets.append(
            {
                "scene_id": int(query["scene_id"]),
                "im_id": int(query["image_id"]),
                "obj_id": int(payload["object_id"]),
                "inst_count": 1,
            }
        )
    return targets


def _bop_toolkit_provenance() -> dict[str, str | None]:
    import bop_toolkit_lib

    toolkit_root = Path(bop_toolkit_lib.__file__).resolve().parents[1]
    commit: str | None = None
    head_path = toolkit_root / ".git" / "HEAD"
    if head_path.is_file():
        head = head_path.read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            reference_path = toolkit_root / ".git" / head[5:]
            if reference_path.is_file():
                commit = reference_path.read_text(
                    encoding="utf-8"
                ).strip()
        elif head:
            commit = head
    return {
        "bop_toolkit_path": str(toolkit_root),
        "bop_toolkit_git_commit": commit,
    }


def evaluate_output_root(
    *,
    output_root: Path,
    dataset_root: Path,
    split: str = "test",
    models_directory: Path | None = None,
    renderer_type: str = "python",
    num_workers: int = 1,
    rebuild_aggregate: bool = True,
) -> dict[str, Any]:
    from evaluation.linemod_bop_evaluator import (
        create_bop_linemod_dataset_view,
        run_official_bop19_evaluation,
    )
    from result_storage import backfill_compact_result_add_metrics

    resolved_output_root = Path(output_root).expanduser().resolve()
    resolved_dataset_root = Path(dataset_root).expanduser().resolve()
    result_paths = find_compact_result_paths(resolved_output_root)
    targets = _target_entries(
        output_root=resolved_output_root,
        result_paths=result_paths,
    )
    if not targets:
        raise FileNotFoundError(
            "No configured LINEMOD evaluation targets or compact "
            f"result.json files found under {resolved_output_root}"
        )

    add_evaluated_count = 0
    add_failures: list[dict[str, str]] = []
    for result_path in result_paths:
        try:
            backfill_compact_result_add_metrics(
                result_path=result_path,
                dataset_root=resolved_dataset_root,
                split=split,
                models_directory=models_directory,
            )
            add_evaluated_count += 1
        except ValueError as error:
            if "has no estimated pose" in str(error):
                continue
            add_failures.append(
                {"result_path": str(result_path), "error": str(error)}
            )
        except Exception as error:
            add_failures.append(
                {
                    "result_path": str(result_path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    bop_output_directory = resolved_output_root / "evaluation" / "bop19"
    csv_path, exported_pose_count, missing_pose_count = _export_bop19_csv(
        result_paths=result_paths,
        dataset_root=resolved_dataset_root,
        split=split,
        output_directory=bop_output_directory,
    )
    dataset_view_root, targets_path = create_bop_linemod_dataset_view(
        dataset_root=resolved_dataset_root,
        split=split,
        output_directory=bop_output_directory,
        targets=targets,
    )
    provenance = _bop_toolkit_provenance()
    official_result = run_official_bop19_evaluation(
        toolkit_root=Path(str(provenance["bop_toolkit_path"])),
        dataset_view_root=dataset_view_root,
        result_csv_path=csv_path,
        targets_path=targets_path,
        evaluation_directory=bop_output_directory / "official_eval",
        renderer_type=renderer_type,
        num_workers=num_workers,
    )

    aggregate_json_path: Path | None = None
    aggregate_csv_path: Path | None = None
    if (
        rebuild_aggregate
        and (resolved_output_root / "linemod_all_summary.json").is_file()
    ):
        from linemod_all_runner import rebuild_linemod_aggregate_results

        aggregate_json_path, aggregate_csv_path = (
            rebuild_linemod_aggregate_results(resolved_output_root)
        )

    report = {
        "protocol": (
            "Official BOP Toolkit eval_bop19_pose.py; all configured "
            "queries are targets; accepted and rejected saved poses are "
            "exported; missing poses remain absent estimates."
        ),
        **provenance,
        "renderer_type": renderer_type,
        "num_workers": int(num_workers),
        "relative_pose_evaluation": (
            "Query absolute estimate = relative estimate @ reference GT "
            "absolute pose; GT is used only for evaluation."
        ),
        "result_count": len(result_paths),
        "target_count": len(targets),
        "missing_result_count": max(0, len(targets) - len(result_paths)),
        "exported_pose_count": exported_pose_count,
        "missing_pose_count": missing_pose_count,
        "absent_estimate_count": max(
            0,
            len(targets) - exported_pose_count,
        ),
        "add_evaluated_count": add_evaluated_count,
        "add_failure_count": len(add_failures),
        "add_failures": add_failures,
        "official_bop19": official_result.to_dict(),
        "aggregate_json_path": (
            str(aggregate_json_path)
            if aggregate_json_path is not None
            else None
        ),
        "aggregate_csv_path": (
            str(aggregate_csv_path)
            if aggregate_csv_path is not None
            else None
        ),
    }
    report_path = bop_output_directory / "evaluation_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    arguments = _parse_arguments()
    report = evaluate_output_root(
        output_root=arguments.output_root,
        dataset_root=arguments.dataset_root,
        split=arguments.split,
        models_directory=arguments.models_directory,
        renderer_type=arguments.renderer_type,
        num_workers=arguments.num_workers,
    )
    scores = report["official_bop19"]["scores"]
    print(
        "[Official BOP19 evaluation complete] "
        f"targets={report['target_count']}, "
        f"poses={report['exported_pose_count']}, "
        f"AR={scores['bop19_average_recall']:.6f}"
    )
    print(
        "[Official BOP19 scores] "
        f"{report['official_bop19']['scores_path']}"
    )
    print(f"[BOP evaluation report] {report['report_path']}")
    for failure in report["add_failures"][:20]:
        print(
            f"[ADD metric warning] {failure['result_path']}: "
            f"{failure['error']}",
            file=sys.stderr,
        )
    return 1 if report["add_failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
