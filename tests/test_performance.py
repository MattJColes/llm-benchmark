import json

import pytest

from llm_benchmark.config import Architecture, Backend, KvCacheType, ModelFormat
from llm_benchmark.performance import (
    ContextObservation,
    QuantObservation,
    SustainedSample,
    collect_sustained_samples,
    comparable_quant_observations,
    context_sweep,
    first_divergence,
    measure_context_sweep,
    ngram_repeat_rate,
    parse_llama_bench,
    parse_perplexity,
)


def test_parses_distinct_prefill_and_decode_records() -> None:
    result = parse_llama_bench(
        json.dumps(
            [
                {"n_prompt": 512, "n_gen": 0, "avg_ts": 1000, "n_batch": 2048, "n_ubatch": 512},
                {"n_prompt": 0, "n_gen": 128, "avg_ts": 42, "n_batch": 2048, "n_ubatch": 512},
            ]
        )
    )

    assert result.prefill_tokens_per_second == 1000
    assert result.decode_tokens_per_second == 42


def test_context_sweep_covers_each_fill_and_kv_type() -> None:
    sweep = context_sweep([KvCacheType.F16, KvCacheType.Q8_0])

    assert len(sweep) == 8
    assert (65536, KvCacheType.Q8_0) in sweep
    assert ContextObservation(0, KvCacheType.F16, 20, 100).time_to_first_token_ms == 100
    observations = measure_context_sweep(
        [KvCacheType.F16], lambda context, _kv: (context / 1000 + 1, 10)
    )
    assert observations[-1].decode_tokens_per_second == 132.072


def test_quality_helpers_identify_divergence_repetition_and_gguf_only() -> None:
    observations = [
        QuantObservation("qwen", "Q8_0", Backend.METAL, Architecture.DENSE, ModelFormat.GGUF, 5),
        QuantObservation("qwen", "4bit", Backend.MLX, Architecture.DENSE, ModelFormat.MLX, 5),
    ]

    assert first_divergence([1, 2, 3], [1, 4, 3]) == 1
    assert first_divergence([1], [1, 2]) == 1
    assert ngram_repeat_rate([1, 2, 1, 2, 1, 2], 2) > 0
    assert comparable_quant_observations(observations) == [observations[0]]
    assert SustainedSample(10, 20, 4).tokens_per_joule == 5


def test_rejects_invalid_ngram_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        ngram_repeat_rate([1, 2], 0)


def test_collects_sustained_samples_and_parses_perplexity() -> None:
    times = iter([0, 0, 10, 10])
    samples = collect_sustained_samples(
        lambda: (20, None),
        duration_seconds=10,
        interval_seconds=10,
        monotonic=lambda: next(times),
        sleep=lambda _seconds: None,
    )

    assert samples == [SustainedSample(0, 20, None)]
    assert parse_perplexity("Final estimate: PPL = 4.25") == 4.25
