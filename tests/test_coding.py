from pathlib import Path

import pytest

from llm_benchmark import coding
from llm_benchmark.coding import (
    Attempt,
    CodingTask,
    FailureMode,
    classify_outcome,
    docker_command,
    pass_at_k,
    repair_metrics,
    run_hidden_tests,
    run_repair_loop,
)


def test_docker_command_disables_network_and_sets_limits(tmp_path: Path) -> None:
    command = docker_command(image="python:3.12", workspace=tmp_path, command=["pytest"])

    assert "--network" in command and "none" in command
    assert "--memory" in command


def test_repair_metrics_records_first_and_eventual_success() -> None:
    metrics = repair_metrics(
        [Attempt(passed=False, output="fail"), Attempt(passed=True, output="pass")]
    )

    assert metrics == {"pass_at_1": 0, "pass_at_4": 1, "attempts_to_green": 2}



def _task() -> CodingTask:
    return CodingTask(
        id="fixture",
        language="python",
        prompt="write solution.py",
        public_example="example",
        hidden_test_command=["python", "-m", "pytest", "-q", "/tests"],
        image="bench/fixture",
    )


def test_repair_loop_passes_on_first_attempt() -> None:
    prompts: list[str] = []

    def generate(prompt: str) -> str:
        prompts.append(prompt)
        return "good"

    def run_tests(code: str) -> Attempt:
        return Attempt(code=code, passed=True, output="ok")

    attempts = run_repair_loop(task=_task(), generate=generate, run_tests=run_tests)

    assert len(attempts) == 1
    assert attempts[0].passed
    assert prompts == ["write solution.py"]


def test_repair_loop_caps_at_one_initial_plus_three_repairs() -> None:
    attempts = run_repair_loop(
        task=_task(),
        generate=lambda prompt: "code",
        run_tests=lambda code: Attempt(
            code=code, passed=False, output="fail", failure_mode=FailureMode.COMPILED_BUT_WRONG
        ),
    )

    assert len(attempts) == 4
    assert all(not attempt.passed for attempt in attempts)


def test_classify_did_not_compile_and_timeout() -> None:
    compile_fail = [
        Attempt(
            code="c",
            passed=False,
            output="SyntaxError: invalid syntax",
            failure_mode=FailureMode.DID_NOT_COMPILE,
        )
    ]
    timeout = [
        Attempt(code="c", passed=False, output="timeout", failure_mode=FailureMode.TIMEOUT)
    ]

    assert classify_outcome(compile_fail) == FailureMode.DID_NOT_COMPILE
    assert classify_outcome(timeout) == FailureMode.TIMEOUT


def test_classify_repair_blind_when_resubmitting_identical_code() -> None:
    repeated = "def fib(n): return n"
    attempts = [
        Attempt(
            code=repeated,
            passed=False,
            output="assertion",
            failure_mode=FailureMode.COMPILED_BUT_WRONG,
        ),
        Attempt(
            code=repeated,
            passed=False,
            output="assertion",
            failure_mode=FailureMode.COMPILED_BUT_WRONG,
        ),
    ]

    assert classify_outcome(attempts) == FailureMode.REPAIR_BLIND


def test_classify_returns_none_when_chain_succeeds() -> None:
    attempts = [
        Attempt(code="c1", passed=False, output="x", failure_mode=FailureMode.COMPILED_BUT_WRONG),
        Attempt(code="c2", passed=True, output="ok"),
    ]

    assert classify_outcome(attempts) is None


def test_pass_at_k_estimates_probability() -> None:
    assert pass_at_k([True, True, False, False], k=4) == 1.0
    assert pass_at_k([False, False, False, False], k=4) == 0.0
    assert pass_at_k([True, False], k=1) == 0.5


def test_run_hidden_tests_detects_compile_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeCompleted:
        returncode = 1
        stdout = ""
        stderr = "SyntaxError: invalid syntax"

    monkeypatch.setattr(coding.subprocess, "run", lambda *args, **kwargs: FakeCompleted())

    attempt = run_hidden_tests(image="bench/x", workspace=tmp_path, command=["pytest"])

    assert attempt.failure_mode == FailureMode.DID_NOT_COMPILE


def test_run_hidden_tests_detects_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _raise(*args: object, **kwargs: object) -> None:
        raise coding.subprocess.TimeoutExpired(cmd="docker", timeout=30)

    monkeypatch.setattr(coding.subprocess, "run", _raise)

    attempt = run_hidden_tests(image="bench/x", workspace=tmp_path, command=["pytest"])

    assert attempt.failure_mode == FailureMode.TIMEOUT
    assert attempt.output == "timeout"