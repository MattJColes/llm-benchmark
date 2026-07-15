from pathlib import Path

from llm_benchmark.coding import Attempt, docker_command, repair_metrics


def test_docker_command_disables_network_and_sets_limits(tmp_path: Path) -> None:
    command = docker_command(image="python:3.12", workspace=tmp_path, command=["pytest"])

    assert "--network" in command and "none" in command
    assert "--memory" in command


def test_repair_metrics_records_first_and_eventual_success() -> None:
    metrics = repair_metrics(
        [Attempt(passed=False, output="fail"), Attempt(passed=True, output="pass")]
    )

    assert metrics == {"pass_at_1": 0, "pass_at_4": 1, "attempts_to_green": 2}
