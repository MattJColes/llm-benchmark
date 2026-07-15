"""Manifest contracts and metrics for the owned code-review corpus."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
