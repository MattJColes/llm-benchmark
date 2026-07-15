"""Manifest validation and deterministic scoring for VLM evaluation."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ScoringMode(StrEnum):
    EXACT = "exact"
    NUMERIC = "numeric"
    EXACT_SET = "exact_set"


class VisionQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answer: Any
    scoring: ScoringMode
    tolerance: float | None = Field(default=None, ge=0)


class VisionAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z]+_[0-9]{3}$")
    file: str
    generator: str | None = None
    seed: int | None = None
    questions: list[VisionQuestion] = Field(min_length=1)


def score_answer(question: VisionQuestion, response: Any) -> bool:
    if question.scoring is ScoringMode.EXACT:
        return response == question.answer
    if question.scoring is ScoringMode.NUMERIC:
        return abs(float(response) - float(question.answer)) <= question.tolerance
    if question.scoring is ScoringMode.EXACT_SET:
        return set(response) == set(question.answer)
    raise ValueError(f"unsupported scoring mode: {question.scoring}")


def validate_asset_file(asset: VisionAsset, root: Path) -> Path:
    path = root / asset.file
    if not path.is_file():
        raise ValueError(f"manifest asset is missing: {asset.file}")
    return path
