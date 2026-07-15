from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_benchmark.config import (
    Backend,
    BenchmarkConfig,
    Model,
    RunSelection,
    load_config,
    run_id,
    write_run_config,
)

CONFIG_PATH = Path("config/benchmark.yaml")
MODEL_HASH = "a" * 64


def test_loads_declared_box_backends() -> None:
    config = load_config(CONFIG_PATH)

    assert config.box("spark").expect == [Backend.CUDA]
    assert config.box("framework").expect == [Backend.ROCM, Backend.VULKAN]
    assert config.box("mac").expect == [Backend.METAL, Backend.MLX]
    assert config.smoketests["gguf"].model == "qwen3-0.6b-q8_0"


def test_rejects_mlx_without_a_pin() -> None:
    with pytest.raises(ValidationError, match="mlx_lm_version"):
        BenchmarkConfig.model_validate(
            {
                "boxes": [{"id": "mac", "os": "macos", "gpu": "apple", "expect": ["mlx"]}],
                "pins": {"llama_cpp_commit": "abc"},
            }
        )


def test_run_id_normalises_model_and_quant() -> None:
    model = Model.model_validate(
        {
            "id": "qwen3-8b",
            "format": "gguf",
            "architecture": "dense",
            "quant": "Q4_K_M",
            "sha256": MODEL_HASH,
            "path": "models/qwen3.gguf",
        }
    )
    selection = RunSelection(
        box="spark",
        backend=Backend.CUDA,
        model=model.id,
        run_date=date(2026, 7, 15),
    )

    assert run_id(selection, model) == "2026-07-15_spark_cuda_qwen38b_q4km"


def test_writes_resolved_run_config_before_results(tmp_path: Path) -> None:
    config = BenchmarkConfig.model_validate(
        {
            **load_config(CONFIG_PATH).model_dump(mode="json"),
            "models": [
                {
                    "id": "qwen3-8b",
                    "format": "gguf",
                    "architecture": "dense",
                    "quant": "Q4_K_M",
                    "sha256": MODEL_HASH,
                    "path": "models/qwen3.gguf",
                }
            ],
            "smoketests": {},
        }
    )
    selection = RunSelection(box="spark", backend=Backend.CUDA, model="qwen3-8b")

    config_path = write_run_config(tmp_path, config, selection)

    assert config_path.name == "config.yaml"
    assert config_path.parent.parent.name == "runs"
    assert "qwen3-8b" in config_path.read_text(encoding="utf-8")
