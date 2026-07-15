"""Local serving runners with a shared OpenAI-compatible endpoint contract."""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunnerMetrics:
    process_id: int
    started_at: datetime


class Runner(ABC):
    """Launches one server and exposes the endpoint suites use."""

    @property
    @abstractmethod
    def endpoint(self) -> str:
        """Return the server's OpenAI-compatible base URL."""

    @abstractmethod
    def start(self) -> RunnerMetrics:
        """Start the server process and return launch metrics."""

    @abstractmethod
    def load_log(self) -> str:
        """Return the captured server load log."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the server process, escalating when it ignores termination."""


class ProcessRunner(Runner):
    def __init__(
        self,
        *,
        command: tuple[str, ...],
        endpoint: str,
        load_log_path: Path,
        stop_timeout_seconds: float = 10.0,
    ) -> None:
        self.command = command
        self._endpoint = endpoint
        self._load_log_path = load_log_path
        self._stop_timeout_seconds = stop_timeout_seconds
        self._process: subprocess.Popen[str] | None = None

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def start(self) -> RunnerMetrics:
        if self._process is not None:
            raise RuntimeError("runner is already started")
        self._load_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = self._load_log_path.open("w", encoding="utf-8")
        try:
            self._process = subprocess.Popen(
                self.command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        finally:
            log_file.close()
        return RunnerMetrics(process_id=self._process.pid, started_at=datetime.now(UTC))

    def load_log(self) -> str:
        if not self._load_log_path.exists():
            return ""
        return self._load_log_path.read_text(encoding="utf-8")

    def stop(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=self._stop_timeout_seconds)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=self._stop_timeout_seconds)


class LlamaCppRunner(ProcessRunner):
    def __init__(
        self,
        *,
        binary: str,
        model_path: Path,
        host: str,
        port: int,
        load_log_path: Path,
        context_size: int,
        gpu_layers: int = 99,
    ) -> None:
        super().__init__(
            command=(
                binary,
                "-m",
                str(model_path),
                "-c",
                str(context_size),
                "--host",
                host,
                "--port",
                str(port),
                "--n-gpu-layers",
                str(gpu_layers),
            ),
            endpoint=f"http://{host}:{port}",
            load_log_path=load_log_path,
        )


class MlxRunner(ProcessRunner):
    def __init__(
        self,
        *,
        model_path: Path,
        host: str,
        port: int,
        load_log_path: Path,
        binary: str = "mlx_lm.server",
    ) -> None:
        super().__init__(
            command=(binary, "--model", str(model_path), "--host", host, "--port", str(port)),
            endpoint=f"http://{host}:{port}",
            load_log_path=load_log_path,
        )
