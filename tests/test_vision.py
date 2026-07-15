import pytest

from llm_benchmark.vision import (
    ScoringMode,
    VisionAsset,
    VisionQuestion,
    comparable_vision_runs,
    constrained_json_request,
    has_exif,
    score_answer,
    strip_exif,
    structural_match,
    validate_asset_file,
)


def test_scores_exact_numeric_and_set_answers() -> None:
    exact = VisionQuestion(question="sign", answer="EXIT", scoring=ScoringMode.EXACT)

    assert score_answer(exact, "EXIT")
    assert not score_answer(exact, "exit")

    numeric = VisionQuestion(
        question="value", answer=37.5, scoring=ScoringMode.NUMERIC, tolerance=0.5
    )
    labels = VisionQuestion(question="labels", answer=["a", "b"], scoring=ScoringMode.EXACT_SET)

    assert score_answer(numeric, 37.8)
    assert score_answer(labels, ["b", "a"])


def test_validates_stable_asset_name() -> None:
    asset = VisionAsset.model_validate(
        {
            "id": "chart_014",
            "file": "diagrams/charts/chart_014.png",
            "questions": [{"question": "x", "answer": "x", "scoring": "exact"}],
        }
    )

    assert asset.id == "chart_014"


def test_rejects_missing_manifest_asset(tmp_path) -> None:
    asset = VisionAsset(
        id="chart_001",
        file="diagrams/charts/chart_001.png",
        questions=[VisionQuestion(question="x", answer="x", scoring="exact")],
    )

    with pytest.raises(ValueError, match="missing"):
        validate_asset_file(asset, tmp_path)


def test_structural_scoring_allows_extra_keys_but_requires_shape() -> None:
    expected = {"count": 3, "label": "chart"}
    assert structural_match(expected, {"count": 3, "label": "chart", "extra": 1})
    assert not structural_match(expected, {"count": 3})
    assert not structural_match({"items": [1, 2]}, {"items": [1]})

    question = VisionQuestion(question="shape", answer=expected, scoring="structural")
    assert score_answer(question, {"count": 3, "label": "chart", "extra": 1})
    assert not score_answer(question, {"count": 3, "label": "wrong"})


def test_judge_mode_requires_judge_answer() -> None:
    question = VisionQuestion(question="explain", answer="anything", scoring="judge")
    with pytest.raises(ValueError, match="requires judge_answer"):
        score_answer(question, "anything")


def test_constrained_json_request_carries_schema_and_image() -> None:
    question = VisionQuestion(question="count bars", answer=2, scoring="numeric")
    schema = {"type": "object", "properties": {"value": {"type": "number"}}}
    request = constrained_json_request(question, schema, "data:image/png;base64,AAAA")

    content = request["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert request["response_format"] == {"type": "json_object"}
    assert "count bars" in content[1]["text"]


def test_strip_exif_removes_embedded_metadata(tmp_path) -> None:
    from PIL import Image

    source = tmp_path / "raw.jpg"
    image = Image.new("RGB", (8, 8), (12, 34, 56))
    exif = Image.Exif()
    exif[270] = "secret location"
    image.save(source, format="JPEG", exif=exif.tobytes())
    assert has_exif(source)

    destination = strip_exif(source, tmp_path / "clean" / "photo_001.jpg")

    assert destination.is_file()
    assert has_exif(destination) is False


def test_comparable_vision_runs_gates_on_image_token_preflight() -> None:
    runs = [
        {"backend": "cuda", "preflight_observations": {"vision_token_count": 256}},
        {"backend": "vulkan", "preflight_observations": {"vision_token_count": 128}},
        {"backend": "rocm", "preflight_observations": {}},
    ]

    kept = comparable_vision_runs(runs, expected_image_tokens=256)

    assert [run["backend"] for run in kept] == ["cuda"]
