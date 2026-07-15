"""Fail-fast environment checks that gate benchmark sweeps."""

from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from llm_benchmark.config import Backend, Box, VersionPins
from llm_benchmark.evidence import append_event


@dataclass(frozen=True, slots=True)
class CommandResult:
    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str


class CommandRunner:
    def run(self, command: tuple[str, ...], timeout_seconds: float) -> CommandResult:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
        return CommandResult(command, completed.returncode, completed.stdout, completed.stderr)


@dataclass(frozen=True, slots=True)
class PreflightResult:
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    observations: Mapping[str, str]

    @property
    def passed(self) -> bool:
        return not self.failures


_BACKEND_COMMANDS = {
    Backend.CUDA: ("nvidia-smi", "--query-gpu=name", "--format=csv,noheader"),
    Backend.ROCM: ("rocminfo",),
    Backend.VULKAN: ("vulkaninfo", "--summary"),
}
_LOAD_LOG_MARKERS = {
    Backend.CUDA: "cuda0",
    Backend.ROCM: "rocm0",
    Backend.VULKAN: "vulkan0",
    Backend.METAL: "metal",
}


def check_tier_zero(
    *,
    box: Box,
    pins: VersionPins,
    llama_cli: str,
    load_log: str,
    command_runner: CommandRunner,
) -> PreflightResult:
    failures: list[str] = []
    warnings: list[str] = []

    llama_version = command_runner.run((llama_cli, "--version"), timeout_seconds=10)
    if llama_version.exit_code != 0:
        failures.append(f"{llama_cli} --version failed")
    elif pins.llama_cpp_commit not in llama_version.stdout:
        failures.append("llama.cpp version does not match the configured commit")

    detected = _detect_backends(command_runner)
    for backend in box.expect:
        _validate_backend(backend, box, detected, load_log, failures)
    for backend in detected:
        if backend not in box.expect:
            warnings.append(f"unexpected backend available: {backend.value}")

    if Backend.ROCM in box.expect:
        override = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
        if override is not None and override not in {"11.5.1", "11.5.0"}:
            failures.append("HSA_OVERRIDE_GFX_VERSION must be unset or match gfx1151")

    return PreflightResult(
        tuple(failures),
        tuple(warnings),
        {"llama_cpp_version": llama_version.stdout.strip()},
    )


def check_prerequisites(box: Box, command_runner: CommandRunner) -> PreflightResult:
    failures: list[str] = []
    observations: dict[str, str] = {}
    for name, prerequisite in box.coding_prereqs.items():
        result = command_runner.run(tuple(shlex.split(prerequisite.check)), timeout_seconds=60)
        observations[name] = result.stdout.strip()
        if result.exit_code != 0:
            failures.append(f"{name} prerequisite failed: {prerequisite.check}")
        elif prerequisite.pin != "available" and prerequisite.pin not in result.stdout:
            failures.append(f"{name} version does not match {prerequisite.pin}")
    return PreflightResult(tuple(failures), (), observations)


def record_preflight(path: Path, result: PreflightResult) -> None:
    append_event(
        path,
        {
            "kind": "preflight",
            "status": "passed" if result.passed else "failed",
            "failures": list(result.failures),
            "warnings": list(result.warnings),
            "observations": dict(result.observations),
        },
    )


def check_fingerprint(
    *,
    token_ids: list[int],
    expected_hash: str,
    perplexity: float,
    minimum_perplexity: float,
    maximum_perplexity: float,
) -> PreflightResult:
    fingerprint = token_fingerprint(token_ids)
    failures: list[str] = []
    if fingerprint != expected_hash:
        failures.append("token fingerprint does not match the configured value")
    if not minimum_perplexity <= perplexity <= maximum_perplexity:
        failures.append("perplexity is outside the configured tolerance")
    return PreflightResult(
        tuple(failures),
        (),
        {"token_fingerprint": fingerprint, "perplexity": str(perplexity)},
    )


def token_fingerprint(token_ids: list[int]) -> str:
    return hashlib.sha256(",".join(str(token_id) for token_id in token_ids).encode()).hexdigest()


def check_features(
    *,
    health_passed: bool,
    chat_passed: bool,
    grammar_passed: bool,
    vision_token_count: int | None = None,
    expected_vision_token_count: int | None = None,
    context_allocated: bool | None = None,
) -> PreflightResult:
    failures: list[str] = []
    if not health_passed:
        failures.append("endpoint health check failed")
    if not chat_passed:
        failures.append("OpenAI-compatible chat completion failed")
    if not grammar_passed:
        failures.append("GBNF constrained generation failed")
    if (
        expected_vision_token_count is not None
        and vision_token_count != expected_vision_token_count
    ):
        failures.append("vision image token count does not match the configured value")
    if context_allocated is False:
        failures.append("maximum planned context allocation failed")
    return PreflightResult(tuple(failures), (), {})


def check_freshness(recorded: Mapping[str, str], current: Mapping[str, str]) -> PreflightResult:
    compared_keys = ("llama_cpp_commit", "driver", "os_packages", "mlx_lm_version")
    failures = tuple(
        f"preflight is stale: {key} changed"
        for key in compared_keys
        if key in recorded and recorded.get(key) != current.get(key)
    )
    return PreflightResult(failures, (), {})


def require_fresh_preflight(recorded: Mapping[str, str], current: Mapping[str, str]) -> None:
    result = check_freshness(recorded, current)
    if not result.passed:
        raise RuntimeError("; ".join(result.failures))


def _detect_backends(command_runner: CommandRunner) -> dict[Backend, str]:
    detected: dict[Backend, str] = {}
    for backend, command in _BACKEND_COMMANDS.items():
        result = command_runner.run(command, timeout_seconds=10)
        if result.exit_code == 0:
            detected[backend] = f"{result.stdout}\n{result.stderr}".lower()
    return detected


def _validate_backend(
    backend: Backend,
    box: Box,
    detected: dict[Backend, str],
    load_log: str,
    failures: list[str],
) -> None:
    if backend is Backend.MLX:
        return
    if backend is Backend.METAL:
        if "metal" not in load_log.lower():
            failures.append("llama.cpp load log does not show Metal initialisation")
        return
    output = detected.get(backend)
    if output is None:
        failures.append(f"expected backend is unavailable: {backend.value}")
        return
    if backend is Backend.CUDA and box.gpu.lower() not in output:
        failures.append(f"nvidia-smi does not report expected GPU {box.gpu}")
    if backend is Backend.ROCM and "gfx1151" not in output:
        failures.append("rocminfo does not report gfx1151")
    if backend is Backend.VULKAN and ("llvmpipe" in output or "amd" not in output):
        failures.append("vulkaninfo does not report an AMD physical device")
    marker = _LOAD_LOG_MARKERS[backend]
    if marker not in load_log.lower():
        failures.append(f"llama.cpp load log does not show {marker}")
