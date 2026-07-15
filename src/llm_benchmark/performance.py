"""Performance and quantisation-quality measurement helpers."""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from llm_benchmark.config import Architecture, Backend, KvCacheType, ModelFormat


@dataclass(frozen=True, slots=True)
class BenchResult:
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    batch_size: int
    ubatch_size: int


@dataclass(frozen=True, slots=True)
class ContextObservation:
    context_fill: int
    kv_cache_type: KvCacheType
    decode_tokens_per_second: float
    time_to_first_token_ms: float


@dataclass(frozen=True, slots=True)
class SustainedSample:
    elapsed_seconds: float
    decode_tokens_per_second: float
    watts: float | None = None

    @property
    def tokens_per_joule(self) -> float | None:
        if self.watts is None or self.watts <= 0:
            return None
        return self.decode_tokens_per_second / self.watts


@dataclass(frozen=True, slots=True)
class QuantObservation:
    model: str
    quant: str
    backend: Backend
    architecture: Architecture
    model_format: ModelFormat
    perplexity: float


def run_llama_bench(
    *,
    binary: str,
    model_path: Path,
    batch_size: int,
    ubatch_size: int,
    timeout_seconds: float = 900,
) -> BenchResult:
    command = (
        binary,
        "--output",
        "json",
        "--model",
        str(model_path),
        "--n-prompt",
        "512",
        "--n-gen",
        "128",
        "--batch-size",
        str(batch_size),
        "--ubatch-size",
        str(ubatch_size),
    )
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"llama-bench failed: {completed.stderr}")
    return parse_llama_bench(completed.stdout)


def parse_llama_bench(output: str) -> BenchResult:
    records = json.loads(output)
    if not isinstance(records, list):
        raise ValueError("llama-bench JSON output must be a list")
    prefill = next(
        (record for record in records if record["n_prompt"] > 0 and record["n_gen"] == 0), None
    )
    decode = next(
        (record for record in records if record["n_prompt"] == 0 and record["n_gen"] > 0), None
    )
    if prefill is None or decode is None:
        raise ValueError("llama-bench output must contain separate prefill and decode records")
    return BenchResult(
        prefill_tokens_per_second=float(prefill["avg_ts"]),
        decode_tokens_per_second=float(decode["avg_ts"]),
        batch_size=int(prefill["n_batch"]),
        ubatch_size=int(prefill["n_ubatch"]),
    )


def context_sweep(kv_cache_types: list[KvCacheType]) -> tuple[tuple[int, KvCacheType], ...]:
    return tuple(
        (context_fill, kv_cache_type)
        for kv_cache_type in kv_cache_types
        for context_fill in (0, 32768, 65536, 131072)
    )


def measure_context_sweep(
    kv_cache_types: list[KvCacheType],
    measure: Callable[[int, KvCacheType], tuple[float, float]],
) -> list[ContextObservation]:
    return [
        ContextObservation(context_fill, kv_cache_type, *measure(context_fill, kv_cache_type))
        for context_fill, kv_cache_type in context_sweep(kv_cache_types)
    ]


def collect_sustained_samples(
    sample: Callable[[], tuple[float, float | None]],
    *,
    duration_seconds: float = 600,
    interval_seconds: float = 10,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> list[SustainedSample]:
    start = monotonic()
    samples: list[SustainedSample] = []
    while True:
        elapsed = monotonic() - start
        if elapsed >= duration_seconds:
            return samples
        tokens_per_second, watts = sample()
        samples.append(SustainedSample(elapsed, tokens_per_second, watts))
        sleep(min(interval_seconds, duration_seconds - elapsed))


def run_perplexity(
    *,
    binary: str,
    model_path: Path,
    corpus_path: Path,
    timeout_seconds: float = 900,
) -> float:
    completed = subprocess.run(
        (binary, "--model", str(model_path), "--file", str(corpus_path), "--n-gpu-layers", "all"),
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"llama-perplexity failed: {completed.stderr}")
    return parse_perplexity(f"{completed.stdout}\n{completed.stderr}")


def parse_perplexity(output: str) -> float:
    matches = re.findall(
        r"(?:ppl|perplexity)\s*(?:=|:)\s*([0-9]+(?:\.[0-9]+)?)",
        output,
        flags=re.IGNORECASE,
    )
    if not matches:
        raise ValueError("could not locate perplexity in llama-perplexity output")
    return float(matches[-1])


def comparable_quant_observations(observations: list[QuantObservation]) -> list[QuantObservation]:
    return [
        observation for observation in observations if observation.model_format is ModelFormat.GGUF
    ]


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def ngram_repeat_rate(tokens: list[int], ngram_size: int = 3) -> float:
    if ngram_size < 1:
        raise ValueError("ngram_size must be positive")
    ngrams = [
        tuple(tokens[index : index + ngram_size]) for index in range(len(tokens) - ngram_size + 1)
    ]
    if not ngrams:
        return 0.0
    return 1 - len(set(ngrams)) / len(ngrams)


def sustained_duration_seconds(samples: list[SustainedSample]) -> float:
    if not samples:
        return 0
    return samples[-1].elapsed_seconds
