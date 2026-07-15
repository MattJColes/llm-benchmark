"""Tiny-model inference checks shared by llama.cpp and MLX preflight."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from llm_benchmark.config import SmoketestConfig


@dataclass(frozen=True, slots=True)
class SmoketestResult:
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def check_smoketest(
    *,
    completion: dict[str, Any],
    exit_code: int,
    load_log: str,
    config: SmoketestConfig,
) -> SmoketestResult:
    failures: list[str] = []
    content = _completion_content(completion)
    if exit_code != 0:
        failures.append(f"inference exited with status {exit_code}")
    if not content.strip():
        failures.append("inference produced an empty response")
    if "nan" in load_log.lower():
        failures.append("load log contains a NaN warning")
    if _completion_token_count(completion) < config.max_tokens:
        failures.append(f"inference generated fewer than {config.max_tokens} tokens")
    offload_ratio = _offload_ratio(load_log)
    if offload_ratio is None or offload_ratio <= 0.9:
        failures.append("no more than 90 percent of layers were offloaded to GPU")
    return SmoketestResult(tuple(failures))


def _completion_content(completion: dict[str, Any]) -> str:
    try:
        return str(completion["choices"][0]["message"]["content"])
    except (IndexError, KeyError, TypeError):
        return ""


def _completion_token_count(completion: dict[str, Any]) -> int:
    usage = completion.get("usage", {})
    return int(usage.get("completion_tokens", 0))


def _offload_ratio(load_log: str) -> float | None:
    match = re.search(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers", load_log, flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1)) / int(match.group(2))
