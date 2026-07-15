from pydantic import BaseModel

from llm_benchmark.structured import (
    ToolCall,
    ToolScenario,
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
