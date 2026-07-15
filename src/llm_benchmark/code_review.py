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
