"""Owned coding-task contracts and bounded repair-loop accounting."""

from __future__ import annotations

import subprocess
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


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


class Attempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    return Attempt(passed=False, output=output, failure_mode=FailureMode.COMPILED_BUT_WRONG)


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
