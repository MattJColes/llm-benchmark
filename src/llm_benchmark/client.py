"""OpenAI-compatible HTTP client with retained request evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from llm_benchmark.evidence import write_transcript


class EndpointError(RuntimeError):
    """An OpenAI-compatible endpoint did not complete a request."""


class OpenAICompatibleClient:
    def __init__(self, base_url: str, timeout_seconds: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[Mapping[str, Any]],
        transcript_directory: Path,
        transcript_id: str,
        **parameters: Any,
    ) -> dict[str, Any]:
        payload = {"model": model, "messages": messages, "stream": False, **parameters}
        completion = self._post_json("/v1/chat/completions", payload)
        write_transcript(transcript_directory, transcript_id, payload, completion)
        return completion

    def completion(
        self,
        *,
        prompt: str,
        transcript_directory: Path,
        transcript_id: str,
        **parameters: Any,
    ) -> dict[str, Any]:
        payload = {"prompt": prompt, "stream": False, "return_tokens": True, **parameters}
        completion = self._post_json("/completion", payload)
        write_transcript(transcript_directory, transcript_id, payload, completion)
        return completion

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = Request(
                f"{self._base_url}{path}",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310
                completion = json.load(response)
        except HTTPError as error:
            raise EndpointError(
                f"endpoint returned HTTP {error.code}: {error.read().decode()}"
            ) from error
        except URLError as error:
            raise EndpointError(f"endpoint request failed: {error.reason}") from error
        return completion
