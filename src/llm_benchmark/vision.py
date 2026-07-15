"""Manifest validation and deterministic scoring for VLM evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from llm_benchmark.client import OpenAICompatibleClient


class ScoringMode(StrEnum):
    EXACT = "exact"
    NUMERIC = "numeric"
    EXACT_SET = "exact_set"
    STRUCTURAL = "structural"
    JUDGE = "judge"


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
    """Score a deterministic response. Judge-mode questions use ``judge_answer``."""
    if question.scoring is ScoringMode.EXACT:
        return response == question.answer
    if question.scoring is ScoringMode.NUMERIC:
        return abs(float(response) - float(question.answer)) <= (question.tolerance or 0.0)
    if question.scoring is ScoringMode.EXACT_SET:
        return set(response) == set(question.answer)
    if question.scoring is ScoringMode.STRUCTURAL:
        return structural_match(question.answer, response)
    raise ValueError(f"scoring mode {question.scoring.value} requires judge_answer")


def structural_match(expected: Any, response: Any) -> bool:
    """True when ``expected`` is a structural subset of ``response``.

    Dicts match when every expected key is present with a structurally-matching
    value (extra keys in the response are allowed); lists match element-wise up to
    the expected length; scalars match by equality.
    """
    if isinstance(expected, dict):
        return isinstance(response, dict) and all(
            key in response and structural_match(value, response[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(response, list) and len(response) >= len(expected) and all(
            structural_match(expected_item, response_item)
            for expected_item, response_item in zip(expected, response)
        )
    return expected == response


def constrained_json_request(
    question: VisionQuestion, schema: Mapping[str, Any], image_url: str
) -> dict[str, Any]:
    """Build a chat-completion request that constrains the VLM answer to JSON."""
    instruction = (
        f"{question.question}\n\n"
        f"Respond with a JSON object matching this schema:\n{json.dumps(schema)}"
    )
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": instruction},
                ],
            }
        ],
        "response_format": {"type": "json_object"},
    }


def judge_prompt(question: VisionQuestion, response: Any) -> list[Mapping[str, str]]:
    """Build the prompt asking the judge whether a response is correct."""
    system = (
        "You are a strict evaluator. Decide whether the response answers the question "
        "correctly against the known answer. Respond with exactly one word: yes or no."
    )
    user = (
        f"Question: {question.question}\n"
        f"Expected answer: {question.answer}\n"
        f"Response: {response}\n\nIs the response correct? Reply yes or no."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def judge_answer(
    *,
    client: OpenAICompatibleClient,
    question: VisionQuestion,
    response: Any,
    judge_model: str,
    transcript_directory: Path,
    transcript_id: str,
) -> bool:
    completion = client.chat_completion(
        model=judge_model,
        messages=judge_prompt(question, response),
        transcript_directory=transcript_directory,
        transcript_id=transcript_id,
        temperature=0,
        max_tokens=16,
    )
    return _judge_text(completion).strip().lower().startswith("y")


def strip_exif(source: Path, destination: Path) -> Path:
    """Re-encode an image without EXIF or other embedded metadata."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        fmt = "JPEG" if destination.suffix.lower() in {".jpg", ".jpeg"} else "PNG"
        mode = "RGB" if fmt == "JPEG" else image.mode
        image.convert(mode).save(destination, format=fmt)
    return destination


def has_exif(path: Path) -> bool:
    with Image.open(path) as image:
        return bool(image.getexif())


def image_token_check_passed(observations: Mapping[str, Any], expected: int) -> bool:
    actual = observations.get("vision_token_count")
    return actual is not None and int(actual) == expected


def comparable_vision_runs(
    runs: Iterable[Mapping[str, Any]], expected_image_tokens: int
) -> list[Mapping[str, Any]]:
    """Keep only runs whose preflight recorded a passing image-token-count check."""
    return [
        run
        for run in runs
        if image_token_check_passed(run.get("preflight_observations", {}), expected_image_tokens)
    ]


def validate_asset_file(asset: VisionAsset, root: Path) -> Path:
    path = root / asset.file
    if not path.is_file():
        raise ValueError(f"manifest asset is missing: {asset.file}")
    return path


def _judge_text(completion: Mapping[str, Any]) -> str:
    try:
        return str(completion["choices"][0]["message"]["content"])
    except (IndexError, KeyError, TypeError):
        return ""
