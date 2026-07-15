from llm_benchmark.baseline import TokenBaseline


def test_token_baseline_exposes_a_stable_record_shape() -> None:
    baseline = TokenBaseline(token_ids=(1, 2, 3), fingerprint="abc", content="output")

    assert baseline.token_ids == (1, 2, 3)
    assert baseline.fingerprint == "abc"
