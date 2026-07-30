from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from main import LINEMOD_OBJECT_METADATA


_CONFLICTING_OPTIONS = (
    "--object-id",
    "--object-ids",
    "--object-name",
    "--sam3-prompt",
    "--reference-scene-id",
    "--reference-image-id",
    "--reference-instance-index",
    "--query-scene-id",
    "--query-image-id",
    "--query-image-ids",
    "--query-instance-index",
)


def _option_is_present(
    argv: Sequence[str],
    option: str,
) -> bool:
    return any(
        argument == option
        or argument.startswith(f"{option}=")
        for argument in argv
    )


def _remove_single_value_option(
    argv: Sequence[str],
    option: str,
) -> list[str]:
    cleaned: list[str] = []
    index = 0

    while index < len(argv):
        argument = argv[index]

        if argument.startswith(f"{option}="):
            index += 1
            continue

        if argument == option:
            if index + 1 >= len(argv):
                raise ValueError(
                    f"{option} requires a value."
                )
            index += 2
            continue

        cleaned.append(argument)
        index += 1

    return cleaned


def load_object_image_ids(
    *,
    dataset_root: Path,
    split: str,
    object_id: int,
) -> tuple[int, ...]:
    scene_gt_path = (
        Path(dataset_root)
        / split
        / f"{object_id:06d}"
        / "scene_gt.json"
    )

    if not scene_gt_path.is_file():
        raise FileNotFoundError(
            f"scene_gt.json not found: {scene_gt_path}"
        )

    with scene_gt_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise TypeError(
            "scene_gt.json root must be an object: "
            f"{scene_gt_path}"
        )

    image_ids: list[int] = []

    for raw_image_id, raw_instances in payload.items():
        if not isinstance(raw_instances, list):
            continue

        if any(
            isinstance(instance, dict)
            and instance.get("obj_id") == object_id
            for instance in raw_instances
        ):
            image_ids.append(int(raw_image_id))

    if not image_ids:
        raise ValueError(
            f"No frames for object {object_id}: "
            f"{scene_gt_path}"
        )

    return tuple(sorted(set(image_ids)))


def build_object_command(
    *,
    python_executable: str,
    entrypoint_path: Path,
    base_argv: Sequence[str],
    object_id: int,
    object_name: str,
    sam3_prompt: str,
    reference_image_id: int,
    query_image_ids: Sequence[int],
    output_root: Path,
) -> list[str]:
    if not query_image_ids:
        raise ValueError(
            "query_image_ids must not be empty."
        )

    return [
        python_executable,
        str(
            Path(entrypoint_path)
            .expanduser()
            .resolve()
        ),
        *base_argv,
        "--object-id",
        str(object_id),
        "--object-name",
        object_name,
        "--sam3-prompt",
        sam3_prompt,
        "--reference-scene-id",
        str(object_id),
        "--reference-image-id",
        str(reference_image_id),
        "--reference-instance-index",
        "0",
        "--query-scene-id",
        str(object_id),
        "--query-instance-index",
        "0",
        "--query-image-ids",
        *(
            str(image_id)
            for image_id in query_image_ids
        ),
        "--output-root",
        str(output_root),
    ]


def _save_summary(
    *,
    output_root: Path,
    pose_path: str,
    records: Sequence[dict[str, Any]],
) -> Path:
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path = (
        output_root / "linemod_all_summary.json"
    )
    failed_count = sum(
        record["status"] != "completed"
        for record in records
    )
    payload = {
        "pose_path": pose_path,
        "execution": (
            "objects sequential; queries sequential "
            "inside each object batch"
        ),
        "completed_objects": (
            len(records) - failed_count
        ),
        "failed_objects": failed_count,
        "records": list(records),
    }

    with summary_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return summary_path


def run_all_linemod_sequence(
    *,
    pose_path: str,
    entrypoint_path: Path,
    argv: Sequence[str],
) -> int:
    conflicts = tuple(
        option
        for option in _CONFLICTING_OPTIONS
        if _option_is_present(argv, option)
    )

    if conflicts:
        raise ValueError(
            "--all-linemod derives object/reference/query "
            "arguments and cannot be combined with "
            + ", ".join(conflicts)
            + "."
        )

    from main import build_config, parse_args

    parsed = parse_args(
        [
            *argv,
            "--pose-path",
            pose_path,
        ]
    )
    config = build_config(parsed)
    configured_output_root = (
        parsed.output_root is not None
    )
    output_root = (
        config.output_root
        if configured_output_root
        else (
            Path(__file__).resolve().parent
            / "outputs"
            / "linemod_all"
            / pose_path
        )
    )
    child_base_argv = _remove_single_value_option(
        argv,
        "--output-root",
    )
    records: list[dict[str, Any]] = []

    print(
        "[LINEMOD all] 15 objects run sequentially. "
        "Each object's reference is reused for all "
        "remaining frames."
    )

    for sequence_index, object_id in enumerate(
        LINEMOD_OBJECT_METADATA,
        start=1,
    ):
        object_name, sam3_prompt = (
            LINEMOD_OBJECT_METADATA[object_id]
        )
        record: dict[str, Any] = {
            "object_id": object_id,
            "object_name": object_name,
            "status": "running",
        }
        records.append(record)

        try:
            image_ids = load_object_image_ids(
                dataset_root=config.dataset_root,
                split=config.split,
                object_id=object_id,
            )
            reference_image_id = (
                config.reference.image_id
                if config.reference.image_id in image_ids
                else image_ids[0]
            )
            query_image_ids = tuple(
                image_id
                for image_id in image_ids
                if image_id != reference_image_id
            )

            if not query_image_ids:
                raise ValueError(
                    f"Object {object_id} has no query frames."
                )

            object_output_root = (
                output_root
                / f"object_{object_id:02d}_{object_name}"
            )
            command = build_object_command(
                python_executable=sys.executable,
                entrypoint_path=entrypoint_path,
                base_argv=child_base_argv,
                object_id=object_id,
                object_name=object_name,
                sam3_prompt=sam3_prompt,
                reference_image_id=(
                    reference_image_id
                ),
                query_image_ids=query_image_ids,
                output_root=object_output_root,
            )

            print(
                "\n[LINEMOD object "
                f"{sequence_index}/15] "
                f"{object_id}: {object_name}, "
                f"reference={reference_image_id}, "
                f"queries={len(query_image_ids)}"
            )
            completed = subprocess.run(
                command,
                check=False,
            )

            if completed.returncode != 0:
                raise RuntimeError(
                    "Object process exited with code "
                    f"{completed.returncode}."
                )

            record.update(
                {
                    "status": "completed",
                    "reference_image_id": (
                        reference_image_id
                    ),
                    "query_count": len(
                        query_image_ids
                    ),
                    "output_root": str(
                        object_output_root
                    ),
                }
            )

        except Exception as error:
            record.update(
                {
                    "status": "failed",
                    "error_type": (
                        type(error).__name__
                    ),
                    "error": str(error),
                }
            )
            print(
                "[LINEMOD object warning] "
                f"object={object_id}: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )

        _save_summary(
            output_root=output_root,
            pose_path=pose_path,
            records=records,
        )

    summary_path = _save_summary(
        output_root=output_root,
        pose_path=pose_path,
        records=records,
    )
    failed_count = sum(
        record["status"] != "completed"
        for record in records
    )
    print(
        "\n[LINEMOD all complete] "
        f"completed={15 - failed_count}, "
        f"failed={failed_count}"
    )
    print(f"[LINEMOD all summary] {summary_path}")

    return 1 if failed_count else 0
