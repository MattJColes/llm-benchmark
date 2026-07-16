"""Manifest contracts and metrics for the owned code-review corpus."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm_benchmark.client import OpenAICompatibleClient
from llm_benchmark.evidence import append_event, latest_events


class ReviewCaseKind(StrEnum):
    INJECTED = "injected"
    CONTROL = "control"
    HISTORICAL = "historical"


class ReviewCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: ReviewCaseKind
    patch: str
    branch: str | None = None
    files: list[str] = Field(min_length=1)
    category: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    rationale: str | None = None
    severity: str | None = None
    fix_commit: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    parent_commit: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    pr: int | None = Field(default=None, gt=0)

    @field_validator("patch", "files")
    @classmethod
    def validates_relative_paths(cls, value: str | list[str]) -> str | list[str]:
        paths = [value] if isinstance(value, str) else value
        if any(
            not path or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
            for path in paths
        ):
            raise ValueError("corpus paths must be safe relative paths")
        return value

    @model_validator(mode="after")
    def validates_case_kind(self) -> ReviewCase:
        bug_fields = (
            self.category,
            self.line_start,
            self.line_end,
            self.rationale,
            self.severity,
        )
        if self.kind is ReviewCaseKind.CONTROL:
            if any(
                value is not None
                for value in (*bug_fields, self.fix_commit, self.parent_commit, self.pr)
            ):
                raise ValueError("control cases cannot declare bug-only fields")
        elif any(value is None for value in bug_fields):
            raise ValueError("bug cases require category, location, rationale, and severity")
        elif (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("bug case line ranges must be ordered")
        if self.kind is ReviewCaseKind.HISTORICAL and any(
            value is None for value in (self.fix_commit, self.parent_commit, self.pr)
        ):
            raise ValueError("historical cases require fix, parent, and PR provenance")
        return self


class ReviewCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository: str
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    cases: list[ReviewCase]


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str
    case_id: str | None = None
    match: str


def score_verdicts(verdicts: list[JudgeVerdict], allowlisted_findings: set[str]) -> tuple[int, int]:
    matched_cases: set[str] = set()
    false_positives = 0
    for verdict in verdicts:
        case_id = verdict.case_id
        if verdict.match in {"yes", "partial"} and case_id is not None:
            if case_id not in matched_cases:
                matched_cases.add(case_id)
            elif verdict.finding_id not in allowlisted_findings:
                false_positives += 1
        elif verdict.finding_id not in allowlisted_findings:
            false_positives += 1
    return len(matched_cases), false_positives


def review_metrics(
    *, matched: int, findings: int, injected_bugs: int, kloc: float
) -> dict[str, float]:
    if injected_bugs <= 0 or findings < 0 or matched < 0 or kloc <= 0:
        raise ValueError(
            "injected_bugs, findings, matched, and kloc must be positive where applicable"
        )
    false_positives = max(findings - matched, 0)
    return {
        "precision": matched / findings if findings else 0.0,
        "recall": matched / injected_bugs,
        "false_positives_per_kloc": false_positives / kloc,
    }


def require_validity_gate(recall: float, threshold: float = 0.8) -> None:
    if recall < threshold:
        raise RuntimeError(
            f"frontier validity gate failed: recall {recall:.2f} is below {threshold:.2f}"
        )


def validate_frontier_evidence(
    corpus: ReviewCorpus,
    manifest_path: Path,
    reviews_path: Path,
    verdicts_path: Path,
    manual_checks_path: Path,
    threshold: float = 0.8,
) -> dict[str, Any]:
    patches = {case.patch for case in corpus.cases}
    reviews = {
        sample_id: event
        for sample_id, event in latest_events(reviews_path).items()
        if sample_id in patches and event.get("status") == "completed"
    }
    if set(reviews) != patches:
        raise RuntimeError("frontier review evidence does not cover the current corpus")
    for patch, event in reviews.items():
        expected_hash = hashlib.sha256((manifest_path.parent / patch).read_bytes()).hexdigest()
        if (
            event.get("corpus_commit") != corpus.commit
            or event.get("patch_sha256") != expected_hash
        ):
            raise RuntimeError("frontier review provenance does not match the current corpus")
    verdicts = {
        sample_id: event
        for sample_id, event in latest_events(verdicts_path).items()
        if event.get("patch") in patches and event.get("status") == "completed"
    }
    expected_findings = {
        f"{patch}::{finding['finding_id']}": finding
        for patch, event in reviews.items()
        for finding in event.get("findings", [])
        if isinstance(finding, dict) and isinstance(finding.get("finding_id"), str)
    }
    expected_verdict_ids = set(expected_findings)
    finding_count = sum(len(event.get("findings", [])) for event in reviews.values())
    if set(verdicts) != expected_verdict_ids or len(expected_verdict_ids) != finding_count:
        raise RuntimeError("frontier verdict evidence does not match the current findings")
    for sample_id, event in verdicts.items():
        finding = event.get("finding")
        if (
            event.get("sample_id") != sample_id
            or not isinstance(finding, dict)
            or sample_id != f"{event.get('patch')}::{finding.get('finding_id')}"
            or finding != expected_findings[sample_id]
            or event.get("judge_model") != "claude-sonnet-4-6"
            or event.get("temperature") != 0
        ):
            raise RuntimeError("frontier verdict provenance is invalid")

    injected_ids = {case.id for case in corpus.cases if case.kind is ReviewCaseKind.INJECTED}
    historical_ids = {case.id for case in corpus.cases if case.kind is ReviewCaseKind.HISTORICAL}
    matched_case_ids = {
        event["case_id"]
        for event in verdicts.values()
        if event.get("match") in {"yes", "partial"} and isinstance(event.get("case_id"), str)
    }
    recall = len(matched_case_ids & injected_ids) / len(injected_ids)
    require_validity_gate(recall, threshold)

    manual_checks = {
        sample_id: event
        for sample_id, event in latest_events(manual_checks_path).items()
        if event.get("status") == "completed"
        and event.get("decision") == "confirmed"
        and event.get("verdict_sample_id") in verdicts
    }
    checked_verdict_ids = {event["verdict_sample_id"] for event in manual_checks.values()}
    required_checks = math.ceil(finding_count * 0.2)
    if len(checked_verdict_ids) < required_checks:
        raise RuntimeError("frontier verdict spot-check is below 20 percent")

    false_positives = sum(event.get("match") == "no" for event in verdicts.values())
    changed_lines = sum(
        line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        for event in reviews.values()
        for line in str(event.get("reviewed_diff", "")).splitlines()
    )
    return {
        "patches": len(reviews),
        "findings": finding_count,
        "precision": (finding_count - false_positives) / finding_count,
        "recall": recall,
        "historical_recall": len(matched_case_ids & historical_ids) / len(historical_ids),
        "false_positives": false_positives,
        "false_positives_per_kloc": false_positives / (changed_lines / 1000),
        "manual_spot_checks": len(checked_verdict_ids),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "gate": "passed",
    }


class ReviewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    provider: str = "openai-compatible"
    base_url: str | None = None
    categories: tuple[str, ...] = ("correctness", "security")
    recursive: bool = True
    reflect: bool = False
    timeout_seconds: int = Field(default=120, gt=0)
    max_input_tokens: int = Field(default=64000, gt=0)
    version: str = "lgtmaybe"

    @model_validator(mode="after")
    def requires_custom_endpoint(self) -> ReviewConfig:
        if self.provider == "openai-compatible" and self.base_url is None:
            raise ValueError("openai-compatible reviews require a base URL")
        return self


class ReviewFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str
    file: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    message: str
    category: str


class ReviewInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    provider: str
    model: str
    categories: tuple[str, ...]
    recursive: bool
    reflect: bool
    timeout_seconds: int
    max_input_tokens: int
    corpus_commit: str | None = None
    patch_sha256: str | None = None
    reviewed_diff: str
    findings: list[ReviewFinding]

    @property
    def finding_count(self) -> int:
        return len(self.findings)


def write_lgtmaybe_config(path: Path, config: ReviewConfig) -> Path:
    path.write_text(
        yaml.safe_dump({"categories": list(config.categories), "reflect": config.reflect}),
        encoding="utf-8",
    )
    return path


def lgtmaybe_command(config: ReviewConfig, config_path: Path) -> tuple[str, ...]:
    """Build the lgtmaybe subprocess invocation against an OpenAI-compatible provider."""
    return (
        "lgtmaybe",
        "review",
        "--provider",
        config.provider,
        *(("--api-base", config.base_url) if config.base_url is not None else ()),
        "--model",
        config.model,
        "--max-input-tokens",
        str(config.max_input_tokens),
        "--timeout",
        str(config.timeout_seconds),
        *(("--recursive",) if config.recursive else ("--no-recursive",)),
        *(("--reflect",) if config.reflect else ("--no-reflect",)),
        "--format",
        "json",
        "--uncommitted",
        "--config",
        str(config_path),
    )


def parse_lgtmaybe_findings(raw: str) -> list[ReviewFinding]:
    """Parse the JSON findings array emitted by ``lgtmaybe review --format json``."""
    candidate = raw.strip()
    if not candidate.startswith("["):
        marker = raw.rfind("\n[")
        candidate = raw[marker + 1 :].strip() if marker >= 0 else candidate
    try:
        decoded = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError("lgtmaybe findings output is invalid JSON") from error
    if not isinstance(decoded, list):
        raise ValueError("lgtmaybe findings output must be a JSON array")
    findings: list[ReviewFinding] = []
    for item in decoded:
        if not isinstance(item, Mapping):
            raise ValueError("lgtmaybe finding must be an object")
        if "path" not in item:
            findings.append(ReviewFinding.model_validate(item))
            continue
        try:
            line = item["line"]
            category = item.get("category") or "unknown"
            title = item["title"]
            finding = ReviewFinding(
                finding_id=f"{item['path']}:{line}:{category}:{title}",
                file=item["path"],
                line_start=line,
                line_end=line,
                message=f"{title}: {item['body']}",
                category=category,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid lgtmaybe finding") from error
        findings.append(finding)
    return findings


def load_review_corpus(path: Path) -> ReviewCorpus:
    return ReviewCorpus.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def run_review_case(
    corpus: ReviewCorpus,
    case: ReviewCase,
    manifest_path: Path,
    config: ReviewConfig,
) -> ReviewInvocation:
    """Materialize one patch at the corpus pin and review its uncommitted diff."""
    patch_path = (manifest_path.parent / case.patch).resolve()
    corpus_root = manifest_path.parent.resolve()
    if not patch_path.is_relative_to(corpus_root) or not patch_path.is_file():
        raise ValueError("patch must resolve to a corpus file")

    with tempfile.TemporaryDirectory(prefix="llm-review-") as directory:
        checkout = Path(directory) / "checkout"
        git_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        subprocess.run(
            ("git", "clone", "--no-checkout", corpus.repository, str(checkout)),
            check=True,
            env=git_env,
            timeout=120,
        )
        subprocess.run(
            ("git", "-C", str(checkout), "fetch", "origin", corpus.commit),
            check=True,
            env=git_env,
            timeout=120,
        )
        subprocess.run(
            ("git", "-C", str(checkout), "checkout", "--detach", corpus.commit),
            check=True,
            env=git_env,
            timeout=120,
        )
        apply = ("git", "-C", str(checkout), "apply", str(patch_path))
        subprocess.run((*apply[:-1], "--check", apply[-1]), check=True, env=git_env, timeout=120)
        subprocess.run(apply, check=True, env=git_env, timeout=120)
        diff = subprocess.run(
            ("git", "-C", str(checkout), "diff", "--binary", "--no-ext-diff", corpus.commit),
            check=True,
            capture_output=True,
            text=True,
            env=git_env,
            timeout=120,
        ).stdout
        config_path = write_lgtmaybe_config(Path(directory) / "lgtmaybe.yml", config)
        completed = subprocess.run(
            lgtmaybe_command(config, config_path),
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds * 4,
        )
        return ReviewInvocation(
            version=version("lgtmaybe"),
            provider=config.provider,
            model=config.model,
            categories=config.categories,
            recursive=config.recursive,
            reflect=config.reflect,
            timeout_seconds=config.timeout_seconds,
            max_input_tokens=config.max_input_tokens,
            corpus_commit=corpus.commit,
            patch_sha256=hashlib.sha256(patch_path.read_bytes()).hexdigest(),
            reviewed_diff=diff,
            findings=parse_lgtmaybe_findings(completed.stdout),
        )


def review_invocation(
    config: ReviewConfig, diff: str, findings: Sequence[ReviewFinding]
) -> ReviewInvocation:
    """Assemble the recorded review invocation from configuration and findings."""
    return ReviewInvocation(
        version=config.version,
        provider=config.provider,
        model=config.model,
        categories=config.categories,
        recursive=config.recursive,
        reflect=config.reflect,
        timeout_seconds=config.timeout_seconds,
        max_input_tokens=config.max_input_tokens,
        reviewed_diff=diff,
        findings=list(findings),
    )


def record_review(path: Path, patch: str, invocation: ReviewInvocation) -> None:
    append_event(
        path,
        {
            "kind": "review",
            "sample_id": patch,
            "status": "completed",
            "patch": patch,
            **invocation.model_dump(mode="json"),
        },
    )


def judge_match_prompt(finding: ReviewFinding, case: ReviewCase) -> list[Mapping[str, str]]:
    """Build the prompt that asks the pinned judge to match a finding to a manifest bug."""
    system = (
        "You are a strict code-review auditor. Decide whether the reported finding "
        "identifies the same defect as the manifested bug. "
        "Respond with exactly one word: yes, no, or partial."
    )
    user = (
        f"Finding: {finding.file}:{finding.line_start}-{finding.line_end} "
        f"({finding.category}): {finding.message}\n"
        f"Manifested bug: {case.files[0]}:{case.line_start}-{case.line_end} "
        f"({case.category}): {case.rationale}\n\n"
        "Does the finding identify the same defect? Reply yes, no, or partial."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_judge_verdict(raw: str, finding_id: str) -> JudgeVerdict:
    """Extract a yes/partial/no verdict from the judge response."""
    lowered = raw.strip().lower()
    for verdict in ("yes", "partial", "no"):
        if verdict in lowered:
            return JudgeVerdict(finding_id=finding_id, match=verdict)
    return JudgeVerdict(finding_id=finding_id, match="no")


def match_finding(
    *,
    client: OpenAICompatibleClient,
    finding: ReviewFinding,
    case: ReviewCase,
    judge_model: str,
    transcript_directory: Path,
) -> JudgeVerdict:
    """Send one finding/bug pair to the pinned judge at temperature 0."""
    completion = client.chat_completion(
        model=judge_model,
        messages=judge_match_prompt(finding, case),
        transcript_directory=transcript_directory,
        transcript_id=finding.finding_id,
        temperature=0,
        max_tokens=16,
    )
    raw = _judge_text(completion)
    verdict = parse_judge_verdict(raw, finding.finding_id)
    return verdict.model_copy(
        update={"case_id": case.id if verdict.match in {"yes", "partial"} else None}
    )


def record_verdict(
    path: Path,
    patch: str,
    finding: ReviewFinding,
    verdict: JudgeVerdict,
    judge_model: str,
) -> None:
    append_event(
        path,
        {
            "kind": "judge_verdict",
            "sample_id": f"{patch}::{verdict.finding_id}",
            "status": "completed",
            "patch": patch,
            "finding": finding.model_dump(mode="json"),
            "judge_model": judge_model,
            "temperature": 0,
            **verdict.model_dump(mode="json"),
        },
    )


def load_allowlist(path: Path) -> set[str]:
    """Load manually-curated false-positive finding ids (one per line)."""
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _judge_text(completion: Mapping[str, Any]) -> str:
    try:
        return str(completion["choices"][0]["message"]["content"])
    except (IndexError, KeyError, TypeError):
        return ""
