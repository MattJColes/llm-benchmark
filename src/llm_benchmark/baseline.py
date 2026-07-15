"""Collect deterministic local preflight baselines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from llm_benchmark.client import EndpointError, OpenAICompatibleClient
from llm_benchmark.preflight import token_fingerprint
from llm_benchmark.runners import LlamaCppRunner


@dataclass(frozen=True, slots=True)
class TokenBaseline:
    token_ids: tuple[int, ...]
    fingerprint: str
    content: str


def collect_llama_baseline(
    *,
    model_path: Path,
    prompt: str,
    transcript_directory: Path,
    load_log_path: Path,
    port: int = 8088,
    startup_timeout_seconds: float = 180.0,
) -> TokenBaseline:
    runner = LlamaCppRunner(
        binary="llama-server",
        model_path=model_path,
        host="127.0.0.1",
        port=port,
        load_log_path=load_log_path,
        context_size=4096,
    )
    runner.start()
    client = OpenAICompatibleClient(runner.endpoint, timeout_seconds=10)
    deadline = time.monotonic() + startup_timeout_seconds
    try:
        while True:
            try:
                completion = client.completion(
                    prompt=prompt,
                    temperature=0,
                    seed=42,
                    n_predict=32,
                    transcript_directory=transcript_directory,
                    transcript_id="fingerprint",
                )
                token_ids = tuple(int(token_id) for token_id in completion["tokens"])
                return TokenBaseline(
                    token_ids=token_ids,
                    fingerprint=token_fingerprint(list(token_ids)),
                    content=str(completion["content"]),
                )
            except EndpointError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "llama.cpp server did not become ready before the startup deadline"
                    )
                time.sleep(1)
    finally:
        runner.stop()
