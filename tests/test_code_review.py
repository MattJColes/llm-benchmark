import os
import re
import shutil
import subprocess
from pathlib import Path

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
                    "patch": "httpx/injected/bug-1.patch",
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
    try:
        ReviewCase.model_validate(
            {
                "id": "control",
                "kind": "control",
                "patch": "httpx/control/control.patch",
                "files": ["x.py"],
                "severity": "low",
            }
        )
    except ValueError as error:
        assert "control cases" in str(error)
    else:
        raise AssertionError("expected invalid control case to fail")


def test_calculates_metrics_and_enforces_frontier_gate() -> None:
    metrics = review_metrics(matched=8, findings=10, injected_bugs=10, kloc=2)

    assert metrics == {"precision": 0.8, "recall": 0.8, "false_positives_per_kloc": 1.0}
    require_validity_gate(metrics["recall"])
    try:
        require_validity_gate(0.79)
    except RuntimeError as error:
        assert "validity gate" in str(error)
    else:
        raise AssertionError("expected low recall to fail")


def test_excludes_allowlisted_unmatched_findings() -> None:
    matched, false_positives = score_verdicts(
        [
            JudgeVerdict(finding_id="a", case_id="bug-1", match="yes"),
            JudgeVerdict(finding_id="b", match="no"),
        ],
        {"b"},
    )

    assert matched == 1
    assert false_positives == 0


def test_duplicate_findings_do_not_inflate_case_recall() -> None:
    matched, false_positives = score_verdicts(
        [
            JudgeVerdict(finding_id="first", case_id="bug-1", match="yes"),
            JudgeVerdict(finding_id="duplicate", case_id="bug-1", match="partial"),
        ],
        set(),
    )

    assert matched == 1
    assert false_positives == 1


def test_builds_lgtmaybe_command_with_security_categories_and_recursion(tmp_path) -> None:
    from llm_benchmark.code_review import (
        ReviewConfig,
        lgtmaybe_command,
        write_lgtmaybe_config,
    )

    config = ReviewConfig(base_url="http://localhost:8080/v1", model="qwen3")
    config_path = write_lgtmaybe_config(tmp_path / "lgtmaybe.yaml", config)
    command = lgtmaybe_command(config, config_path)

    assert "--provider" in command and "openai-compatible" in command
    assert "--api-base" in command and config.base_url in command
    assert "--recursive" in command and "--uncommitted" in command
    assert "--no-reflect" in command
    assert "--timeout" in command and "120" in command
    assert "--max-input-tokens" in command and "64000" in command
    assert config_path.read_text(encoding="utf-8") == (
        "categories:\n- correctness\n- security\nreflect: false\n"
    )


def test_builds_hosted_lgtmaybe_command_without_api_base(tmp_path) -> None:
    from llm_benchmark.code_review import ReviewConfig, lgtmaybe_command, write_lgtmaybe_config

    config = ReviewConfig(provider="anthropic", model="claude-sonnet-4-6")
    command = lgtmaybe_command(config, write_lgtmaybe_config(tmp_path / "config.yaml", config))

    assert "anthropic" in command
    assert "--api-base" not in command


def test_parses_findings_and_assembles_invocation() -> None:
    from llm_benchmark.code_review import (
        ReviewConfig,
        parse_lgtmaybe_findings,
        review_invocation,
    )

    raw = (
        "LiteLLM.Info: diagnostic output\n\n"
        '[{"path": "httpx/_client.py", "line": 10, "severity": "high", '
        '"title": "Missing await", "body": "coroutine is discarded", '
        '"category": "correctness"}]'
    )
    findings = parse_lgtmaybe_findings(raw)
    config = ReviewConfig(base_url="http://localhost:8080/v1", model="qwen3")
    invocation = review_invocation(config, "diff --git ...", findings)

    assert invocation.finding_count == 1
    assert list(invocation.categories) == ["correctness", "security"]
    assert invocation.recursive
    assert invocation.findings[0].category == "correctness"


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
            "patch": "httpx/injected/bug-1.patch",
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


def test_rejects_unsafe_patch_and_unordered_range() -> None:
    base = {
        "id": "bug",
        "kind": "injected",
        "files": ["x.py"],
        "category": "correctness",
        "line_start": 2,
        "line_end": 1,
        "rationale": "broken",
        "severity": "high",
    }
    for patch, message in (("../escape.patch", "safe relative"), ("patches/bug.patch", "ordered")):
        try:
            ReviewCase.model_validate({**base, "patch": patch})
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("expected invalid case to fail")


def test_materializes_patch_and_runs_lgtmaybe_in_checkout(tmp_path, monkeypatch) -> None:
    from llm_benchmark.code_review import ReviewConfig, run_review_case

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(("git", "init", "-q", str(upstream)), check=True)
    subprocess.run(
        ("git", "-C", str(upstream), "config", "user.email", "test@example.com"), check=True
    )
    subprocess.run(("git", "-C", str(upstream), "config", "user.name", "Test"), check=True)
    (upstream / "x.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(upstream), "add", "x.py"), check=True)
    subprocess.run(("git", "-C", str(upstream), "commit", "-qm", "base"), check=True)
    commit = subprocess.run(
        ("git", "-C", str(upstream), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (upstream / "x.py").write_text("value = 2\n", encoding="utf-8")
    outside_config = tmp_path / "outside.yml"
    (upstream / ".lgtmaybe-corpus.yml").symlink_to(outside_config)
    subprocess.run(
        ("git", "-C", str(upstream), "add", "--intent-to-add", ".lgtmaybe-corpus.yml"),
        check=True,
    )
    patch = subprocess.run(
        ("git", "-C", str(upstream), "diff"), check=True, capture_output=True, text=True
    ).stdout
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "patches").mkdir(parents=True)
    (corpus_dir / "patches/bug.patch").write_text(patch, encoding="utf-8")
    manifest = corpus_dir / "manifest.yaml"
    manifest.write_text("", encoding="utf-8")
    case = ReviewCase.model_validate(
        {
            "id": "bug",
            "kind": "injected",
            "patch": "patches/bug.patch",
            "files": ["x.py"],
            "category": "correctness",
            "line_start": 1,
            "line_end": 1,
            "rationale": "value changed",
            "severity": "low",
        }
    )
    corpus = ReviewCorpus(repository=str(upstream), commit=commit, cases=[case])
    binary = tmp_path / "bin"
    binary.mkdir()
    executable = binary / "lgtmaybe"
    executable.write_text(
        '#!/bin/sh\nprintf \'[{"path":"x.py","line":1,"title":"Changed",'
        '"body":"value","category":"correctness"}]\'\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary}{os.pathsep}{os.environ['PATH']}")

    invocation = run_review_case(
        corpus, case, manifest, ReviewConfig(base_url="http://localhost/v1", model="test")
    )

    assert "-value = 1" in invocation.reviewed_diff
    assert invocation.finding_count == 1
    assert invocation.version == "0.11.0"
    assert invocation.provider == "openai-compatible"
    assert invocation.model == "test"
    assert invocation.corpus_commit == commit
    assert invocation.patch_sha256 is not None
    assert not outside_config.exists()


def test_frontier_evidence_reducer_rejects_tampered_provenance(tmp_path) -> None:
    from llm_benchmark.code_review import load_review_corpus, validate_frontier_evidence
    from llm_benchmark.evidence import append_event, latest_events

    project = Path(__file__).parents[1]
    manifest = project / "corpus/httpx.yaml"
    source = project / "results/frontier-review"
    evidence = tmp_path / "frontier-review"
    shutil.copytree(source, evidence)
    corpus = load_review_corpus(manifest)

    validate_frontier_evidence(
        corpus,
        manifest,
        evidence / "reviews.jsonl",
        evidence / "verdicts.jsonl",
        evidence / "manual-checks.jsonl",
    )
    patch, event = next(iter(latest_events(evidence / "reviews.jsonl").items()))
    append_event(
        evidence / "reviews.jsonl",
        {**event, "sample_id": patch, "corpus_commit": "0" * 40, "patch_sha256": "0" * 64},
    )

    try:
        validate_frontier_evidence(
            corpus,
            manifest,
            evidence / "reviews.jsonl",
            evidence / "verdicts.jsonl",
            evidence / "manual-checks.jsonl",
        )
    except RuntimeError as error:
        assert "provenance" in str(error)
    else:
        raise AssertionError("expected tampered frontier provenance to fail")

    verdict_evidence = tmp_path / "tampered-verdict"
    shutil.copytree(source, verdict_evidence)
    current_patch = corpus.cases[0].patch
    review = latest_events(verdict_evidence / "reviews.jsonl")[current_patch]
    finding_id = review["findings"][0]["finding_id"]
    verdict_id = f"{current_patch}::{finding_id}"
    verdict = latest_events(verdict_evidence / "verdicts.jsonl")[verdict_id]
    append_event(
        verdict_evidence / "verdicts.jsonl",
        {
            **verdict,
            "sample_id": verdict_id,
            "finding": {**verdict["finding"], "message": "tampered"},
            "judge_model": "not-pinned",
            "temperature": 1,
        },
    )
    try:
        validate_frontier_evidence(
            corpus,
            manifest,
            verdict_evidence / "reviews.jsonl",
            verdict_evidence / "verdicts.jsonl",
            verdict_evidence / "manual-checks.jsonl",
        )
    except RuntimeError as error:
        assert "verdict provenance" in str(error)
    else:
        raise AssertionError("expected tampered verdict provenance to fail")

    manual_evidence = tmp_path / "duplicate-manual"
    shutil.copytree(source, manual_evidence)
    (manual_evidence / "manual-checks.jsonl").unlink()
    linked_verdict = verdict_id
    for index in range(8):
        append_event(
            manual_evidence / "manual-checks.jsonl",
            {
                "sample_id": f"manual-{index}",
                "status": "completed",
                "decision": "confirmed",
                "verdict_sample_id": linked_verdict,
            },
        )
    try:
        validate_frontier_evidence(
            corpus,
            manifest,
            manual_evidence / "reviews.jsonl",
            manual_evidence / "verdicts.jsonl",
            manual_evidence / "manual-checks.jsonl",
        )
    except RuntimeError as error:
        assert "spot-check" in str(error)
    else:
        raise AssertionError("expected duplicate manual links to fail")


def test_httpx_corpus_integrity(tmp_path) -> None:
    from llm_benchmark.code_review import load_review_corpus

    manifest = Path(__file__).parents[1] / "corpus/httpx.yaml"
    corpus = load_review_corpus(manifest)
    cache = Path.home() / ".cache/llm-benchmark/httpx"
    checkout = tmp_path / "httpx"
    if (cache / ".git").exists():
        clone = ("git", "clone", "-q", "--shared", str(cache), str(checkout))
    else:
        clone = (
            "git",
            "clone",
            "-q",
            "--filter=blob:none",
            "--no-checkout",
            corpus.repository,
            str(checkout),
        )
    subprocess.run(clone, check=True, timeout=120)
    subprocess.run(
        ("git", "-C", str(checkout), "checkout", "-q", corpus.commit),
        check=True,
        timeout=120,
    )

    for case in (case for case in corpus.cases if case.kind.value == "historical"):
        assert case.fix_commit is not None
        assert case.parent_commit is not None
        assert case.pr is not None
        fix_commit = case.fix_commit
        parent = subprocess.run(
            ("git", "-C", str(checkout), "rev-parse", f"{fix_commit}^"),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        subject = subprocess.run(
            ("git", "-C", str(checkout), "show", "-s", "--format=%s", fix_commit),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        assert parent == case.parent_commit
        assert f"#{case.pr}" in subject

    unique = {case.patch for case in corpus.cases}
    assert len(unique) == 24
    for patch_name in unique:
        patch = manifest.parent / patch_name
        assert patch.is_file()
        text = patch.read_text(encoding="utf-8")
        declared = {
            file for case in corpus.cases if case.patch == patch_name for file in case.files
        }
        headers = set(
            line.removeprefix("+++ b/") for line in text.splitlines() if line.startswith("+++ b/")
        )
        assert declared == headers
        changed: dict[str, set[int]] = {}
        current_file = ""
        new_line = 0
        for line in text.splitlines():
            if line.startswith("+++ b/"):
                current_file = line.removeprefix("+++ b/")
                changed.setdefault(current_file, set())
            elif line.startswith("@@"):
                match = re.search(r"\+(\d+)", line)
                assert match is not None
                try:
                    new_line = int(match.group(1)) - 1
                except ValueError as error:
                    raise AssertionError("invalid patch hunk") from error
            elif line.startswith("+") and not line.startswith("+++"):
                new_line += 1
                changed[current_file].add(new_line)
            elif line.startswith("-") and not line.startswith("---"):
                changed[current_file].add(new_line + 1)
            elif not line.startswith("\\"):
                new_line += 1
        subprocess.run(
            ("git", "-C", str(checkout), "reset", "--hard", "-q", corpus.commit), check=True
        )
        subprocess.run(("git", "-C", str(checkout), "apply", "--check", str(patch)), check=True)
        subprocess.run(("git", "-C", str(checkout), "apply", str(patch)), check=True)
        for case in (case for case in corpus.cases if case.patch == patch_name):
            if case.line_start is not None and case.line_end is not None:
                line_count = len(
                    (checkout / case.files[0]).read_text(encoding="utf-8").splitlines()
                )
                assert case.line_start <= case.line_end <= line_count
                assert any(
                    case.line_start <= line <= case.line_end
                    for file in case.files
                    for line in changed[file]
                )
