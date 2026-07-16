# benchmark-configuration Specification

## Purpose
TBD - created by syncing change build-local-llm-benchmark.

## Requirements

### Requirement: Declarative benchmark matrix
The system SHALL load version-controlled configuration that declares the Spark, Framework, and Mac boxes, their expected backends, models, quants, KV modes, and suite-selection tracks.

#### Scenario: Resolving a configured GGUF run
- **WHEN** a user selects a configured box, backend, model, and quant
- **THEN** the system SHALL resolve one run configuration containing those selections and all applicable pins

#### Scenario: Separating MLX configuration
- **WHEN** the selected backend is MLX
- **THEN** the system SHALL resolve an MLX-format model configuration without treating it as a GGUF quant-equivalent run

### Requirement: Backend runner boundary
The system SHALL expose suite access through an OpenAI-compatible endpoint and a runner lifecycle interface that launches, reports metrics for, and stops llama.cpp or MLX servers.

#### Scenario: Running a llama.cpp backend
- **WHEN** a CUDA, ROCm, Vulkan, or Metal configuration is launched
- **THEN** the system SHALL use the llama.cpp runner and provide its endpoint to the selected suite

#### Scenario: Running an MLX backend
- **WHEN** an MLX configuration is launched
- **THEN** the system SHALL use the MLX runner while preserving the same suite endpoint contract

### Requirement: Exact run provenance
The system SHALL write the resolved invocation, model and mmproj hashes where applicable, package/binary pins, and captured environment to each run's `config.yaml` before recording suite output.

#### Scenario: Inspecting a completed run
- **WHEN** a user opens a completed run directory
- **THEN** the directory SHALL contain sufficient configuration and environment data to identify the selected box, backend, model, quant, and binary versions
