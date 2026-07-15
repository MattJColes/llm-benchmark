"""Regenerate Markdown reports and time estimates from raw JSONL evidence."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from llm_benchmark.evidence import read_events


def generate_summary(results_root: Path) -> Path:
    runs = sorted((results_root / "runs").glob("*/timings.jsonl"))
    lines = [
        "# Benchmark Summary",
        "",
        "| Run | Prefill tok/s | Decode tok/s |",
        "| --- | ---: | ---: |",
    ]
    for timing_path in runs:
        events = list(read_events(timing_path))
        prefill = _latest(events, "prefill_tokens_per_second")
        decode = _latest(events, "decode_tokens_per_second")
        lines.append(f"| {timing_path.parent.name} | {prefill or '-'} | {decode or '-'} |")
    report_path = results_root / "reports" / "summary.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def generate_run_reports(results_root: Path) -> list[Path]:
    report_paths: list[Path] = []
    for run_directory in sorted((results_root / "runs").glob("*")):
        if not run_directory.is_dir():
            continue
        events = list(read_events(run_directory / "timings.jsonl"))
        findings = list(read_events(run_directory / "findings.jsonl"))
        report_path = results_root / "reports" / "per-run" / f"{run_directory.name}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            "\n".join(
                [
                    f"# {run_directory.name}",
                    "",
                    "## Timings",
                    "",
                    "```json",
                    *[str(event) for event in events],
                    "```",
                    "",
                    "## Findings",
                    "",
                    "```json",
                    *[str(finding) for finding in findings],
                    "```",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        report_paths.append(report_path)
    return report_paths


def generate_unmatched_findings_queue(results_root: Path) -> Path:
    unmatched: list[dict[str, Any]] = []
    for findings_path in (results_root / "runs").glob("*/findings.jsonl"):
        unmatched.extend(
            finding
            for finding in read_events(findings_path)
            if finding.get("verdict") == "unmatched"
        )
    queue_path = results_root / "reports" / "review.md-per-run"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        "# Unmatched Findings\n\n" + "\n".join(f"- {finding}" for finding in unmatched) + "\n",
        encoding="utf-8",
    )
    return queue_path


def generate_track_reports(results_root: Path) -> list[Path]:
    tracks: dict[str, list[dict[str, Any]]] = {}
    for timing_path in (results_root / "runs").glob("*/timings.jsonl"):
        for event in read_events(timing_path):
            track = event.get("model_format", "gguf")
            architecture = event.get("architecture", "dense")
            tracks.setdefault(f"{track}-{architecture}", []).append(event)
    paths: list[Path] = []
    for track, events in tracks.items():
        path = results_root / "reports" / f"{track}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        values = [
            float(event["decode_tokens_per_second"])
            for event in events
            if "decode_tokens_per_second" in event
        ]
        average = sum(values) / len(values) if values else None
        path.write_text(
            f"# {track}\n\nDecode mean: {average if average is not None else '-'}\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def estimate_remaining_seconds(
    *,
    remaining_generated_tokens: int,
    decode_tokens_per_second: float,
) -> float:
    if decode_tokens_per_second <= 0:
        raise ValueError("decode_tokens_per_second must be positive")
    return remaining_generated_tokens / decode_tokens_per_second


def _latest(events: Iterable[dict[str, Any]], key: str) -> Any | None:
    values = [event[key] for event in events if key in event]
    return values[-1] if values else None
