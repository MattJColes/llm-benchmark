"""Manifest contracts and metrics for the owned code-review corpus."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_benchmark.client import OpenAICompatibleClient
from llm_benchmark.evidence import append_event


class ReviewCaseKind(StrEnum):
    INJECTED = "injected"
    CONTROL = "control"
    HISTORICAL = "historical"


class ReviewCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: ReviewCaseKind
    branch: str
    files: list[str] = Field(min_length=1)
    category: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    rationale: str | None = None
    severity: str | None = None

    @model_validator(mode="after")
    def validates_case_kind(self) -> ReviewCase:
        if self.kind is ReviewCaseKind.CONTROL:
            if any(
                value is not None
                for value in (self.category, self.line_start, self.rationale, self.severity)
            ):
                raise ValueError("control cases cannot declare an injected bug")
        elif any(
            value is None
            for value in (
                self.category,
                self.line_start,
                self.line_end,
                self.rationale,
                self.severity,
            )
        ):
            raise ValueError("bug cases require category, location, rationale, and severity")
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
    matched = sum(verdict.match in {"yes", "partial"} for verdict in verdicts)
    false_positives = sum(
        verdict.match == "no" and verdict.finding_id not in allowlisted_findings
        for verdict in verdicts
    )
    return matched, false_positives


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



class ReviewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    model: str
    categories: tuple[str, ...] = ("correctness", "security")
    recursive: bool = True
    max_input_tokens: int = Field(default=64000, gt=0)
    version: str = "lgtmaybe"


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
    categories: tuple[str, ...]
    recursive: bool
    max_input_tokens: int
    reviewed_diff: str
    findings: list[ReviewFinding]

    @property
    def finding_count(self) -> int:
        return len(self.findings)


def lgtmaybe_command(config: ReviewConfig, diff_path: Path) -> tuple[str, ...]:
    """Build the lgtmaybe subprocess invocation against an OpenAI-compatible provider."""
    return (
        "lgtmaybe",
        "review",
        "--provider",
        "openai",
        "--base-url",
        config.base_url,
        "--model",
        config.model,
        "--categories",
        ",".join(config.categories),
        "--max-input-tokens",
        str(config.max_input_tokens),
        *(("--recursive",) if config.recursive else ()),
        "--format",
        "json",
        str(diff_path),
    )


def parse_lgtmaybe_findings(raw: str) -> list[ReviewFinding]:
    """Parse the JSON findings array emitted by ``lgtmaybe review --format json``."""
    decoded = json.loads(raw)
    if not isinstance(decoded, list):
        raise ValueError("lgtmaybe findings output must be a JSON array")
    return [ReviewFinding.model_validate(item) for item in decoded]


def review_invocation(
    config: ReviewConfig, diff: str, findings: Sequence[ReviewFinding]
) -> ReviewInvocation:
    """Assemble the recorded review invocation from configuration and findings."""
    return ReviewInvocation(
        version=config.version,
        categories=config.categories,
        recursive=config.recursive,
        max_input_tokens=config.max_input_tokens,
        reviewed_diff=diff,
        findings=list(findings),
    )


def record_review(path: Path, invocation: ReviewInvocation) -> None:
    append_event(path, {"kind": "review", **invocation.model_dump(mode="json")})


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
    return parse_judge_verdict(raw, finding.finding_id)


def record_verdict(path: Path, verdict: JudgeVerdict) -> None:
    append_event(path, {"kind": "judge_verdict", **verdict.model_dump(mode="json")})


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