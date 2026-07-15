from llm_benchmark.config import SmoketestConfig
from llm_benchmark.smoketest import check_smoketest


def test_accepts_a_greedy_gpu_smoketest() -> None:
    result = check_smoketest(
        completion={
            "choices": [{"message": {"content": "local inference is ready"}}],
            "usage": {"completion_tokens": 32},
        },
        exit_code=0,
        load_log="offloaded 30/32 layers to GPU",
        config=SmoketestConfig(model="tiny", prompt="test"),
    )

    assert result.passed


def test_rejects_empty_cpu_or_nan_smoketest() -> None:
    result = check_smoketest(
        completion={"choices": [{"message": {"content": ""}}], "usage": {"completion_tokens": 2}},
        exit_code=1,
        load_log="NaN detected; offloaded 2/32 layers to GPU",
        config=SmoketestConfig(model="tiny", prompt="test"),
    )

    assert not result.passed
    assert "inference produced an empty response" in result.failures
    assert "load log contains a NaN warning" in result.failures
