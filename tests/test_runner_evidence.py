from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

from llm_benchmark.client import OpenAICompatibleClient
from llm_benchmark.evidence import append_event, completed_sample_ids, write_transcript
from llm_benchmark.runners import LlamaCppRunner, MlxRunner


class _CompletionHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(content_length))
        if self.path == "/completion":
            response = {"content": request["prompt"], "tokens": [1, 2, 3]}
        else:
            response = {"choices": [{"message": {"content": request["messages"][0]["content"]}}]}
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def _fake_server() -> Iterator[tuple[ThreadingHTTPServer, Thread]]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CompletionHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, thread
    server.shutdown()
    thread.join()


def test_runner_commands_target_the_expected_servers(tmp_path: Path) -> None:
    llama = LlamaCppRunner(
        binary="llama-server",
        model_path=Path("model.gguf"),
        host="127.0.0.1",
        port=8080,
        load_log_path=tmp_path / "llama.log",
        context_size=4096,
    )
    mlx = MlxRunner(
        model_path=Path("model"),
        host="127.0.0.1",
        port=8081,
        load_log_path=tmp_path / "mlx.log",
    )

    assert llama.command[:3] == ("llama-server", "-m", "model.gguf")
    assert "--n-gpu-layers" in llama.command
    assert mlx.command == (
        "mlx_lm.server",
        "--model",
        "model",
        "--host",
        "127.0.0.1",
        "--port",
        "8081",
    )


def test_client_retains_a_fake_server_completion(tmp_path: Path) -> None:
    server, _thread = next(_fake_server())
    try:
        client = OpenAICompatibleClient(f"http://127.0.0.1:{server.server_port}")
        completion = client.chat_completion(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            transcript_directory=tmp_path / "transcript",
            transcript_id="sample-1",
        )
    finally:
        server.shutdown()

    assert completion["choices"][0]["message"]["content"] == "hello"
    with gzip.open(
        tmp_path / "transcript" / "sample-1.json.gz", "rt", encoding="utf-8"
    ) as transcript:
        assert json.load(transcript)["request"]["model"] == "tiny"


def test_client_requests_raw_completion_tokens(tmp_path: Path) -> None:
    server, _thread = next(_fake_server())
    try:
        client = OpenAICompatibleClient(f"http://127.0.0.1:{server.server_port}")
        completion = client.completion(
            prompt="fingerprint",
            temperature=0,
            seed=42,
            n_predict=32,
            transcript_directory=tmp_path / "transcript",
            transcript_id="tokens",
        )
    finally:
        server.shutdown()

    assert completion["tokens"] == [1, 2, 3]


def test_jsonl_resume_only_selects_completed_samples(tmp_path: Path) -> None:
    timing_path = tmp_path / "timings.jsonl"
    append_event(timing_path, {"sample_id": "first", "status": "completed"})
    append_event(timing_path, {"sample_id": "second", "status": "failed"})
    write_transcript(tmp_path / "transcript", "first", {"prompt": "x"}, {"output": "y"})

    assert completed_sample_ids(timing_path) == {"first"}
