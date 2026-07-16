# benchmark-preflight Specification

## Purpose
TBD - created by syncing change build-local-llm-benchmark.

## Requirements

### Requirement: Backend and prerequisite validation
The preflight command SHALL fail a selected box when a configured expected backend is unavailable, when its required physical device evidence is absent, or when a required coding prerequisite does not match its configured version constraint.

#### Scenario: Detecting a software Vulkan fallback
- **WHEN** Vulkan reports llvmpipe or does not report the configured AMD physical device
- **THEN** preflight SHALL fail and SHALL not mark the box eligible for a Vulkan sweep

#### Scenario: Detecting an absent expected backend
- **WHEN** a Framework box lacks ROCm or Vulkan support required by its configuration
- **THEN** preflight SHALL fail and record the missing backend in its result

### Requirement: GPU inference smoketest
The preflight command SHALL run a bundled tiny model for a greedy 32-token generation and fail if the generation is empty, emits a NaN warning, exits unsuccessfully, or offloads no more than 90 percent of layers to the GPU.

#### Scenario: Successful tiny-model inference
- **WHEN** the selected runner produces a non-empty 32-token response with sufficient offload and no warnings
- **THEN** preflight SHALL record the smoketest as passed

### Requirement: Correctness and feature gate
The preflight command SHALL validate a backend-specific fixed-seed token fingerprint and bounded perplexity result, then run only the feature checks required by the planned sweep.

#### Scenario: Fingerprint mismatch after rebuild
- **WHEN** the token fingerprint differs from the configured value
- **THEN** preflight SHALL fail the comparison gate and mark prior results non-comparable

#### Scenario: Vision sweep validation
- **WHEN** a planned sweep includes VLM evaluation
- **THEN** preflight SHALL validate one image request and assert the configured image token count before the sweep can begin

### Requirement: Fresh preflight evidence
The system SHALL reject a sweep unless its preflight pass matches the selected llama.cpp commit, driver versions, and OS package state.

#### Scenario: Stale environment
- **WHEN** a driver or relevant package version differs from the latest passing preflight record
- **THEN** the system SHALL require a new preflight before launching the sweep
