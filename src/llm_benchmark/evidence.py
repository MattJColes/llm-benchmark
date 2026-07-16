"""Append-only benchmark evidence storage."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
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
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return tuple(events)


def completed_sample_ids(path: Path) -> set[str]:
    return {
        event["sample_id"]
        for event in read_events(path)
        if event.get("status") == "completed" and isinstance(event.get("sample_id"), str)
    }


def latest_events(path: Path) -> dict[str, dict[str, Any]]:
    return {
        event["sample_id"]: event
        for event in read_events(path)
        if isinstance(event.get("sample_id"), str)
    }


def write_transcript(
    transcript_directory: Path,
    transcript_id: str,
    request: Mapping[str, Any],
    response: Mapping[str, Any],
) -> Path:
    transcript_directory.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", transcript_id)
    if safe_id != transcript_id:
        digest = hashlib.sha256(transcript_id.encode()).hexdigest()[:12]
        safe_id = f"{safe_id[:80]}-{digest}"
    output_path = transcript_directory / f"{safe_id}.json.gz"
    with gzip.open(output_path, "wt", encoding="utf-8") as output:
        json.dump({"request": request, "response": response}, output, sort_keys=True)
    return output_path
