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
    scenario = ToolScenario(id="no-call", expected_calls=[])

    assert score_tool_calls(scenario, [])
    assert not score_tool_calls(scenario, [ToolCall(name="search", arguments={"query": "x"})])


def test_scores_multi_step_tool_chain() -> None:
    scenario = ToolScenario(
        id="chain",
        expected_calls=[
            ToolCall(name="search", arguments={"query": "x"}),
            ToolCall(name="read_file", arguments={"path": "x.txt"}),
        ],
    )

    assert score_tool_calls(scenario, scenario.expected_calls)


def test_reports_reliability_stability_and_constraint_delta() -> None:
    freeform = evaluate_structured_runs(['{"value": 2}', "not json"], {"value": 2}, Answer)
    constrained = evaluate_structured_runs(['{"value": 2}', '{"value": 2}'], {"value": 2}, Answer)

    assert not freeform["stable"]
    assert constrained["stable"]
    assert constrained_delta(freeform, constrained) == 0.5
