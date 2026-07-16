from pathlib import Path

from pydantic import BaseModel

from llm_benchmark.structured import (
    ToolCall,
    ToolScenario,
    constrained_delta,
    evaluate_structured_runs,
    score_structured_output,
    score_tool_calls,
)


class Answer(BaseModel):
    value: int


def test_scores_parse_schema_and_value_separately() -> None:
    assert score_structured_output('{"value": 2}', {"value": 2}, Answer).values_correct
    assert not score_structured_output('{"value": "two"}', {"value": 2}, Answer).schema_valid
    assert not score_structured_output("not json", {"value": 2}, Answer).parses


def test_scores_no_call_and_exact_tool_arguments() -> None:
    scenario = ToolScenario(id="no-call", prompt="reply directly", expected_calls=[])

    assert score_tool_calls(scenario, [])
    assert not score_tool_calls(scenario, [ToolCall(name="search", arguments={"query": "x"})])


def test_scores_multi_step_tool_chain() -> None:
    scenario = ToolScenario(
        id="chain",
        prompt="search then read",
        expected_calls=[
            ToolCall(name="search", arguments={"query": "x"}),
            ToolCall(name="read_file", arguments={"path": "x.txt"}),
        ],
    )

    assert score_tool_calls(scenario, scenario.expected_calls)


def test_manifest_covers_required_tool_scenario_shapes() -> None:
    from llm_benchmark.structured import load_tool_manifest

    manifest = load_tool_manifest(Path(__file__).parents[1] / "structured/scenarios.yaml")
    scenarios = {scenario.id: scenario for scenario in manifest.scenarios}

    assert not scenarios["no-call"].expected_calls
    assert len(scenarios["search-then-read-chain"].expected_calls) == 2
    assert scenarios["search-then-read-chain"].tool_results
    assert {"search_files", "search_web"} <= {tool.name for tool in manifest.tools}
    assert all(scenario.prompt for scenario in manifest.scenarios)


def test_executes_chain_with_tool_result_before_second_call() -> None:
    from llm_benchmark.structured import execute_tool_scenario, load_tool_manifest

    manifest = load_tool_manifest(Path(__file__).parents[1] / "structured/scenarios.yaml")
    scenario = next(item for item in manifest.scenarios if item.id == "search-then-read-chain")
    turns = iter(scenario.expected_calls)
    observed_results = []

    def call_model(messages, _tools):
        if len(messages) > 1:
            assert messages[-1]["role"] == "tool"
            observed_results.append(messages[-1]["content"])
        return next(turns, None)

    assert execute_tool_scenario(scenario, manifest.tools, call_model) == scenario.expected_calls
    assert observed_results == scenario.tool_results


def test_chain_rejects_spurious_call_after_tool_results() -> None:
    from llm_benchmark.structured import execute_tool_scenario, load_tool_manifest

    manifest = load_tool_manifest(Path(__file__).parents[1] / "structured/scenarios.yaml")
    scenario = next(item for item in manifest.scenarios if item.id == "search-then-read-chain")
    turns = iter(
        [
            *scenario.expected_calls,
            ToolCall(name="calculator", arguments={"expression": "1 + 1"}),
        ]
    )

    actual = execute_tool_scenario(scenario, manifest.tools, lambda _messages, _tools: next(turns))

    assert not score_tool_calls(scenario, actual)


def test_reports_reliability_stability_and_constraint_delta() -> None:
    freeform = evaluate_structured_runs(['{"value": 2}', "not json"], {"value": 2}, Answer)
    constrained = evaluate_structured_runs(['{"value": 2}', '{"value": 2}'], {"value": 2}, Answer)

    assert not freeform["stable"]
    assert constrained["stable"]
    assert constrained_delta(freeform, constrained) == 0.5
