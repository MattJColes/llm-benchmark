import pytest
from pydantic import ValidationError

from llm_benchmark.code_review import (
    JudgeVerdict,
    ReviewCase,
    ReviewCorpus,
    require_validity_gate,
    review_metrics,
    score_verdicts,
)


def test_accepts_manifest_backed_bug_case() -> None:
    corpus = ReviewCorpus.model_validate(
        {
            "repository": "https://github.com/encode/httpx.git",
            "commit": "b5addb64f0161ff6bfe94c124ef76f6a1fba5254",
            "cases": [
                {
                    "id": "bug-1",
                    "kind": "injected",
                    "branch": "benchmark/bug-1",
                    "files": ["httpx/_client.py"],
                    "category": "missing-await",
                    "line_start": 10,
                    "line_end": 11,
                    "rationale": "coroutine is discarded",
                    "severity": "high",
                }
            ],
        }
    )

    assert corpus.cases[0].kind == "injected"


def test_rejects_bug_fields_on_a_control_case() -> None:
    with pytest.raises(ValidationError, match="control cases"):
        ReviewCase.model_validate(
            {
                "id": "control",
                "kind": "control",
                "branch": "benchmark/control",
                "files": ["x.py"],
                "severity": "low",
            }
        )


def test_calculates_metrics_and_enforces_frontier_gate() -> None:
    metrics = review_metrics(matched=8, findings=10, injected_bugs=10, kloc=2)

    assert metrics == {"precision": 0.8, "recall": 0.8, "false_positives_per_kloc": 1.0}
    require_validity_gate(metrics["recall"])
    with pytest.raises(RuntimeError, match="validity gate"):
        require_validity_gate(0.79)


def test_excludes_allowlisted_unmatched_findings() -> None:
    matched, false_positives = score_verdicts(
        [
            JudgeVerdict(finding_id="a", case_id="bug-1", match="yes"),
            JudgeVerdict(finding_id="b", match="no"),
        ],
        {"b"},
    )

    assert (matched, false_positives) == (1, 0)


def test_builds_lgtmaybe_command_with_security_categories_and_recursion(tmp_path) -> None:
    from llm_benchmark.code_review import ReviewConfig, lgtmaybe_command

    config = ReviewConfig(base_url="http://localhost:8080/v1", model="qwen3")
    command = lgtmaybe_command(config, tmp_path / "case.diff")

    assert "--provider" in command and "openai" in command
    assert "--categories" in command and "correctness,security" in command
    assert "--recursive" in command
    assert "--max-input-tokens" in command and "64000" in command
    assert str(tmp_path / "case.diff") in command


def test_parses_findings_and_assembles_invocation() -> None:
    from llm_benchmark.code_review import (
        ReviewConfig,
        parse_lgtmaybe_findings,
        review_invocation,
    )

    raw = (
        '[{"finding_id": "f1", "file": "httpx/_client.py", '
        '"line_start": 10, "line_end": 12, '
        '"message": "missing await", "category": "missing-await"}]'
    )
    findings = parse_lgtmaybe_findings(raw)
    config = ReviewConfig(base_url="http://localhost:8080/v1", model="qwen3")
    invocation = review_invocation(config, "diff --git ...", findings)

    assert invocation.finding_count == 1
    assert invocation.categories == ("correctness", "security")
    assert invocation.recursive is True
    assert invocation.findings[0].category == "missing-await"


def test_parses_judge_verdicts_and_loads_allowlist(tmp_path) -> None:
    from llm_benchmark.code_review import (
        ReviewCase,
        ReviewFinding,
        judge_match_prompt,
        load_allowlist,
        parse_judge_verdict,
    )

    finding = ReviewFinding(
        finding_id="f1",
        file="httpx/_client.py",
        line_start=10,
        line_end=12,
        message="missing await",
        category="missing-await",
    )
    case = ReviewCase.model_validate(
        {
            "id": "bug-1",
            "kind": "injected",
            "branch": "benchmark/bug-1",
            "files": ["httpx/_client.py"],
            "category": "missing-await",
            "line_start": 10,
            "line_end": 11,
            "rationale": "coroutine discarded",
            "severity": "high",
        }
    )
    prompt = judge_match_prompt(finding, case)
    assert prompt[0]["role"] == "system"
    assert "Does the finding" in prompt[1]["content"]

    assert parse_judge_verdict("Yes", "f1").match == "yes"
    assert parse_judge_verdict("PARTIAL match", "f1").match == "partial"
    assert parse_judge_verdict("nonsense", "f1").match == "no"

    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("f1\n\nf2\n", encoding="utf-8")
    assert load_allowlist(allowlist) == {"f1", "f2"}
    assert load_allowlist(tmp_path / "missing.txt") == set()
