"""Owned coding-task contracts and bounded repair-loop accounting."""

from __future__ import annotations

import difflib
import subprocess
from collections.abc import Callable, Sequence
from enum import StrEnum
from math import comb
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

MAX_REPAIRS = 3
REPAIR_BLIND_THRESHOLD = 0.9


class FailureMode(StrEnum):
    DID_NOT_COMPILE = "did-not-compile"
    COMPILED_BUT_WRONG = "compiled-but-wrong"
    TIMEOUT = "timeout"
    REPAIR_BLIND = "repair-blind"


class CodingTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    language: str
    prompt: str
    public_example: str
    hidden_test_command: list[str] = Field(min_length=1)
    image: str = ""


class Attempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = ""
    passed: bool
    output: str
    failure_mode: FailureMode | None = None


def docker_command(*, image: str, workspace: Path, command: list[str]) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--cpus",
        "1",
        "--memory",
        "512m",
        "-v",
        f"{workspace}:/workspace:ro",
        image,
        *command,
    ]


def run_hidden_tests(
    *, image: str, workspace: Path, command: list[str], timeout_seconds: float = 30
) -> Attempt:
    try:
        completed = subprocess.run(
            docker_command(image=image, workspace=workspace, command=command),
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return Attempt(passed=False, output="timeout", failure_mode=FailureMode.TIMEOUT)
    output = f"{completed.stdout}\n{completed.stderr}".strip()
    if completed.returncode == 0:
        return Attempt(passed=True, output=output)
    if _looks_like_compile_failure(output):
        return Attempt(passed=False, output=output, failure_mode=FailureMode.DID_NOT_COMPILE)
    return Attempt(passed=False, output=output, failure_mode=FailureMode.COMPILED_BUT_WRONG)


def build_repair_prompt(task: CodingTask, failure_output: str) -> str:
    """Return a repair prompt that passes verbatim failure output without test source."""
    return (
        f"{task.prompt}\n\n"
        "Your previous attempt produced the failure output below. "
        "The hidden tests are not shown. Fix the issue.\n\n"
        f"{failure_output}"
    )


def run_repair_loop(
    *,
    task: CodingTask,
    generate: Callable[[str], str],
    run_tests: Callable[[str], Attempt],
    max_repairs: int = MAX_REPAIRS,
) -> list[Attempt]:
    """Run one initial generation plus at most ``max_repairs`` repair turns."""
    attempts: list[Attempt] = []
    prompt = task.prompt
    while True:
        code = generate(prompt)
        result = run_tests(code)
        attempts.append(result.model_copy(update={"code": code}))
        if attempts[-1].passed or len(attempts) > max_repairs:
            break
        prompt = build_repair_prompt(task, attempts[-1].output)
    return attempts


def repair_metrics(attempts: list[Attempt]) -> dict[str, float | int]:
    first_pass = int(bool(attempts and attempts[0].passed))
    attempts_to_green = next(
        (index + 1 for index, attempt in enumerate(attempts) if attempt.passed), 0
    )
    return {
        "pass_at_1": first_pass,
        "pass_at_4": int(attempts_to_green > 0),
        "attempts_to_green": attempts_to_green,
    }


def classify_outcome(attempts: Sequence[Attempt]) -> FailureMode | None:
    """Classify a repair chain: ``None`` when it passed, otherwise the failure mode."""
    if not attempts or any(attempt.passed for attempt in attempts):
        return None
    if is_repair_blind(attempts):
        return FailureMode.REPAIR_BLIND
    return attempts[-1].failure_mode or FailureMode.COMPILED_BUT_WRONG


def is_repair_blind(attempts: Sequence[Attempt]) -> bool:
    """True when a repair resubmits code near-identical to a prior failed attempt."""
    for index in range(1, len(attempts)):
        if attempts[index].passed:
            continue
        if _similarity(attempts[index].code, attempts[index - 1].code) >= REPAIR_BLIND_THRESHOLD:
            return True
    return False


def pass_at_k(samples: Sequence[bool], k: int = 4) -> float:
    """Unbiased pass@k: probability at least one of k independent samples passes."""
    n = len(samples)
    if n == 0 or k <= 0:
        raise ValueError("samples and k must be positive")
    if k > n:
        k = n
    correct = sum(samples)
    if n - correct < k:
        return 1.0
    return 1.0 - comb(n - correct, k) / comb(n, k)


_COMPILE_MARKERS = (
    "syntaxerror",
    "indentationerror",
    "error[e",
    "could not compile",
    "failed to compile",
    "parseerror",
    "unexpected token",
)


def _looks_like_compile_failure(output: str) -> bool:
    lowered = output.lower()
    return any(marker in lowered for marker in _COMPILE_MARKERS)


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _normalise(a), _normalise(b)).ratio()


def _normalise(code: str) -> str:
    return " ".join(code.split())
