from pathlib import Path

from llm_benchmark.evidence import append_event
from llm_benchmark.reporting import (
    estimate_remaining_seconds,
    generate_run_reports,
    generate_summary,
    generate_track_reports,
    generate_unmatched_findings_queue,
)


def test_generates_summary_from_raw_jsonl(tmp_path: Path) -> None:
    timing_path = tmp_path / "runs" / "run-1" / "timings.jsonl"
    append_event(timing_path, {"prefill_tokens_per_second": 100, "decode_tokens_per_second": 20})

    summary = generate_summary(tmp_path)

    assert "| run-1 | 100 | 20 |" in summary.read_text(encoding="utf-8")


def test_summary_ignores_late_failed_or_non_comparable_observations(tmp_path: Path) -> None:
    timing_path = tmp_path / "runs" / "run-1" / "timings.jsonl"
    append_event(
        timing_path,
        {
            "status": "completed",
            "prefill_tokens_per_second": 100,
            "decode_tokens_per_second": 20,
        },
    )
    append_event(
        timing_path,
        {
            "status": "failed",
            "comparable": False,
            "prefill_tokens_per_second": 999,
            "decode_tokens_per_second": 999,
        },
    )

    summary = generate_summary(tmp_path).read_text(encoding="utf-8")

    assert "| run-1 | 100 | 20 |" in summary


def test_generates_per_run_and_unmatched_finding_reports(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "run-1"
    append_event(run / "timings.jsonl", {"decode_tokens_per_second": 20})
    append_event(run / "findings.jsonl", {"id": "finding-1", "verdict": "unmatched"})

    reports = generate_run_reports(tmp_path)
    queue = generate_unmatched_findings_queue(tmp_path)

    assert reports[0].name == "run-1.md"
    assert "finding-1" in queue.read_text(encoding="utf-8")


def test_separates_gguf_and_mlx_tracks(tmp_path: Path) -> None:
    append_event(
        tmp_path / "runs" / "metal" / "timings.jsonl",
        {"model_format": "gguf", "architecture": "dense", "decode_tokens_per_second": 20},
    )
    append_event(
        tmp_path / "runs" / "mlx" / "timings.jsonl",
        {"model_format": "mlx", "architecture": "dense", "decode_tokens_per_second": 30},
    )

    reports = generate_track_reports(tmp_path)

    assert {path.name for path in reports} == {"gguf-dense.md", "mlx-dense.md"}


def test_excludes_non_comparable_observations_from_track_average(tmp_path: Path) -> None:
    timing_path = tmp_path / "runs" / "cuda" / "timings.jsonl"
    append_event(timing_path, {"decode_tokens_per_second": 20, "comparable": True})
    append_event(timing_path, {"decode_tokens_per_second": 999, "comparable": False})

    report = generate_track_reports(tmp_path)[0].read_text(encoding="utf-8")

    assert "Decode mean: 20.0" in report
    assert "Non-comparable observations: 1" in report


def test_excludes_superseded_performance_samples(tmp_path: Path) -> None:
    timing_path = tmp_path / "runs" / "cuda" / "timings.jsonl"
    append_event(
        timing_path,
        {"sample_id": "before-preflight", "decode_tokens_per_second": 10},
    )
    append_event(
        timing_path,
        {
            "sample_id": "after-preflight",
            "supersedes_sample_id": "before-preflight",
            "decode_tokens_per_second": 20,
        },
    )

    report = generate_track_reports(tmp_path)[0].read_text(encoding="utf-8")

    assert "Decode mean: 20.0" in report


def test_estimates_remaining_time() -> None:
    assert (
        estimate_remaining_seconds(remaining_generated_tokens=600, decode_tokens_per_second=20)
        == 30
    )
    try:
        estimate_remaining_seconds(remaining_generated_tokens=1, decode_tokens_per_second=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("expected non-positive throughput to fail")
