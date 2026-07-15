from __future__ import annotations

from pathlib import Path

from llm_benchmark.config import Box, VersionPins
from llm_benchmark.preflight import (
    CommandResult,
    CommandRunner,
    check_features,
    check_fingerprint,
    check_prerequisites,
    check_tier_zero,
    record_preflight,
    require_fresh_preflight,
)


class FakeCommandRunner(CommandRunner):
    def __init__(self, responses: dict[tuple[str, ...], CommandResult]) -> None:
        self._responses = responses

    def run(self, command: tuple[str, ...], timeout_seconds: float) -> CommandResult:
        return self._responses.get(command, CommandResult(command, 1, "", "not installed"))


def test_rejects_software_vulkan_and_missing_rocm() -> None:
    box = Box.model_validate(
        {"id": "framework", "os": "ubuntu24", "gpu": "strix_halo", "expect": ["rocm", "vulkan"]}
    )
    responses = {
        ("llama-cli", "--version"): CommandResult(("llama-cli", "--version"), 0, "commit abc", ""),
        ("vulkaninfo", "--summary"): CommandResult(
            ("vulkaninfo", "--summary"), 0, "GPU0: llvmpipe", ""
        ),
    }

    result = check_tier_zero(
        box=box,
        pins=VersionPins(llama_cpp_commit="abc"),
        llama_cli="llama-cli",
        load_log="Vulkan0",
        command_runner=FakeCommandRunner(responses),
    )

    assert not result.passed
    assert "expected backend is unavailable: rocm" in result.failures
    assert "vulkaninfo does not report an AMD physical device" in result.failures


def test_records_matching_prerequisite_versions(tmp_path: Path) -> None:
    box = Box.model_validate(
        {
            "id": "spark",
            "os": "ubuntu24",
            "gpu": "gb10",
            "expect": ["cuda"],
            "coding_prereqs": {"python": {"check": "python3 --version", "pin": "3.12"}},
        }
    )
    responses = {
        ("python3", "--version"): CommandResult(("python3", "--version"), 0, "Python 3.12.13", "")
    }

    result = check_prerequisites(box, FakeCommandRunner(responses))
    record_preflight(tmp_path / "timings.jsonl", result)

    assert result.passed
    assert result.observations == {"python": "Python 3.12.13"}
    assert "Python 3.12.13" in (tmp_path / "timings.jsonl").read_text(encoding="utf-8")


def test_rejects_changed_fingerprint_and_perplexity() -> None:
    result = check_fingerprint(
        token_ids=[1, 2, 3],
        expected_hash="wrong",
        perplexity=12.0,
        minimum_perplexity=1.0,
        maximum_perplexity=10.0,
    )

    assert not result.passed
    assert "token fingerprint does not match the configured value" in result.failures
    assert "perplexity is outside the configured tolerance" in result.failures


def test_correctness_gate_marks_mismatched_backend_non_comparable() -> None:
    from llm_benchmark.config import Backend
    from llm_benchmark.preflight import check_correctness_gate, token_fingerprint

    observed = token_fingerprint([5, 6, 7])
    result = check_correctness_gate(
        backend=Backend.CUDA,
        token_ids=[5, 6, 7],
        perplexity=4.2,
        expected_fingerprints={"cuda": observed, "rocm": "deadbeef"},
        perplexity_bounds=(1.0, 10.0),
    )

    assert result.passed
    assert result.comparable is True

    mismatched = check_correctness_gate(
        backend=Backend.CUDA,
        token_ids=[9, 9, 9],
        perplexity=4.2,
        expected_fingerprints={"cuda": observed},
        perplexity_bounds=(1.0, 10.0),
    )

    assert not mismatched.passed
    assert mismatched.comparable is False
    assert "cuda token fingerprint does not match the recorded value" in mismatched.failures

    first_observation = check_correctness_gate(
        backend=Backend.METAL,
        token_ids=[1, 2, 3],
        perplexity=4.2,
        expected_fingerprints={"cuda": observed},
        perplexity_bounds=(1.0, 10.0),
    )

    assert first_observation.passed and first_observation.comparable is True


def test_rejects_failed_vision_feature_and_stale_preflight() -> None:
    result = check_features(
        health_passed=True,
        chat_passed=True,
        grammar_passed=True,
        vision_token_count=8,
        expected_vision_token_count=9,
        context_allocated=False,
    )

    assert "vision image token count does not match the configured value" in result.failures
    try:
        require_fresh_preflight(
            {"llama_cpp_commit": "old", "driver": "one"},
            {"llama_cpp_commit": "new", "driver": "one"},
        )
    except RuntimeError as error:
        assert "llama_cpp_commit changed" in str(error)
    else:
        raise AssertionError("expected stale preflight to fail")
