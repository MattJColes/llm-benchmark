"""Validated benchmark configuration and run provenance."""

from __future__ import annotations

import platform
import re
import sys
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Backend(StrEnum):
    CUDA = "cuda"
    ROCM = "rocm"
    VULKAN = "vulkan"
    METAL = "metal"
    MLX = "mlx"


class ModelFormat(StrEnum):
    GGUF = "gguf"
    MLX = "mlx"


class Architecture(StrEnum):
    DENSE = "dense"
    MOE = "moe"


class KvCacheType(StrEnum):
    F16 = "f16"
    Q8_0 = "q8_0"


class ToolPrerequisite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check: str
    pin: str


class Box(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9-]+$")
    os: str
    gpu: str
    expect: list[Backend] = Field(min_length=1)
    coding_prereqs: dict[str, ToolPrerequisite] = Field(default_factory=dict)

    @model_validator(mode="after")
    def expected_backends_are_unique(self) -> Self:
        if len(self.expect) != len(set(self.expect)):
            raise ValueError("expected backends must be unique")
        return self


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9._-]+$")
    format: ModelFormat
    architecture: Architecture
    quant: str
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    path: Path
    source_repository: str | None = None
    revision: str | None = None
    mmproj_path: Path | None = None
    mmproj_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")

    @model_validator(mode="after")
    def requires_mmproj_hash(self) -> Self:
        if self.mmproj_path is not None and self.mmproj_sha256 is None:
            raise ValueError("mmproj_sha256 is required when mmproj_path is set")
        return self


class VersionPins(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llama_cpp_commit: str
    mlx_lm_version: str | None = None


class SmoketestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: str
    seed: int = 42
    max_tokens: int = Field(default=32, gt=0)


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    boxes: list[Box] = Field(min_length=1)
    models: list[Model] = Field(default_factory=list)
    kv_cache_types: list[KvCacheType] = Field(default_factory=lambda: [KvCacheType.F16])
    pins: VersionPins
    smoketests: dict[ModelFormat, SmoketestConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validates_unique_ids_and_mlx(self) -> Self:
        box_ids = [box.id for box in self.boxes]
        model_ids = [model.id for model in self.models]
        if len(box_ids) != len(set(box_ids)):
            raise ValueError("box ids must be unique")
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("model ids must be unique")
        if (
            any(Backend.MLX in box.expect for box in self.boxes)
            and self.pins.mlx_lm_version is None
        ):
            raise ValueError("mlx_lm_version is required when MLX is configured")
        for model_format, smoketest in self.smoketests.items():
            model = next((model for model in self.models if model.id == smoketest.model), None)
            if model is None:
                raise ValueError(f"smoketest model is not configured: {smoketest.model}")
            if model.format is not model_format:
                raise ValueError(f"smoketest {model_format.value} requires a matching model format")
        return self

    def box(self, box_id: str) -> Box:
        return next(box for box in self.boxes if box.id == box_id)

    def model(self, model_id: str) -> Model:
        return next(model for model in self.models if model.id == model_id)


class RunSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    box: str
    backend: Backend
    model: str
    run_date: date = Field(default_factory=lambda: datetime.now(UTC).date())


def load_config(path: Path) -> BenchmarkConfig:
    with path.open(encoding="utf-8") as config_file:
        return BenchmarkConfig.model_validate(yaml.safe_load(config_file))


def run_id(selection: RunSelection, model: Model) -> str:
    return "_".join(
        (
            selection.run_date.isoformat(),
            selection.box,
            selection.backend.value,
            _slug(model.id),
            _slug(model.quant),
        )
    )


def write_run_config(
    output_root: Path,
    config: BenchmarkConfig,
    selection: RunSelection,
) -> Path:
    box = config.box(selection.box)
    model = config.model(selection.model)
    if selection.backend not in box.expect:
        raise ValueError(f"{selection.backend.value} is not expected for box {box.id}")
    if selection.backend is Backend.MLX and model.format is not ModelFormat.MLX:
        raise ValueError("MLX runs require an MLX-format model")
    if selection.backend is not Backend.MLX and model.format is not ModelFormat.GGUF:
        raise ValueError("llama.cpp backends require a GGUF model")

    run_directory = output_root / "runs" / run_id(selection, model)
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "transcript").mkdir(exist_ok=True)
    config_path = run_directory / "config.yaml"
    resolved_config = {
        "selection": selection.model_dump(mode="json"),
        "box": box.model_dump(mode="json"),
        "model": model.model_dump(mode="json"),
        "pins": config.pins.model_dump(mode="json"),
        "environment": capture_environment(),
    }
    with config_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(resolved_config, config_file, sort_keys=False)
    return config_path


def capture_environment() -> dict[str, str]:
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())
