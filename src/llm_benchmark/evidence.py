"""Append-only benchmark evidence storage."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"recorded_at": datetime.now(UTC).isoformat(), **event}
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True))
        output.write("\n")


def read_events(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return ()
    with path.open(encoding="utf-8") as source:
        return tuple(json.loads(line) for line in source if line.strip())


def completed_sample_ids(path: Path) -> set[str]:
    return {
        event["sample_id"]
        for event in read_events(path)
        if event.get("status") == "completed" and isinstance(event.get("sample_id"), str)
    }


def write_transcript(
    transcript_directory: Path,
    transcript_id: str,
    request: Mapping[str, Any],
    response: Mapping[str, Any],
) -> Path:
    transcript_directory.mkdir(parents=True, exist_ok=True)
    output_path = transcript_directory / f"{transcript_id}.json.gz"
    with gzip.open(output_path, "wt", encoding="utf-8") as output:
        json.dump({"request": request, "response": response}, output, sort_keys=True)
    return output_path
