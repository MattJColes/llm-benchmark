import pytest
from pydantic import ValidationError

from llm_benchmark.code_review import ReviewCase, ReviewCorpus


def test_accepts_manifest_backed_bug_case() -> None:
    corpus = ReviewCorpus.model_validate(
        {
            "repository": "https://github.com/encode/httpx.git",
            "commit": "b5addb64f0161ff6bfe94c124ef76f6a1fba5254",
            "cases": [
                {
                    "id": "bug-1",
                    "kind": "injected",
                    "branch": "benchmark/bug-1",
                    "files": ["httpx/_client.py"],
                    "category": "missing-await",
                    "line_start": 10,
                    "line_end": 11,
                    "rationale": "coroutine is discarded",
                    "severity": "high",
                }
            ],
        }
    )

    assert corpus.cases[0].kind == "injected"


def test_rejects_bug_fields_on_a_control_case() -> None:
    with pytest.raises(ValidationError, match="control cases"):
        ReviewCase.model_validate(
            {
                "id": "control",
                "kind": "control",
                "branch": "benchmark/control",
                "files": ["x.py"],
                "severity": "low",
            }
        )
