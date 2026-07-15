from llm_benchmark.vision import ScoringMode, VisionAsset, VisionQuestion, score_answer


def test_scores_numeric_and_set_answers() -> None:
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
