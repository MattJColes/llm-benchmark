# benchmark-quantisation-quality Specification

## Purpose
TBD - created by syncing change build-local-llm-benchmark.

## Requirements

### Requirement: Quantisation quality baseline
The quantisation suite SHALL run perplexity for every planned GGUF model, quant, and backend combination and store the source model, quant, and backend with each result.

#### Scenario: Full cheap matrix run
- **WHEN** a configured quantisation baseline sweep is started
- **THEN** the suite SHALL schedule every configured GGUF quant that passes preflight

### Requirement: Cross-backend numeric comparison
The suite SHALL generate fixed-seed greedy token streams for comparable GGUF configurations and record whether streams match and the first divergent token when they do not.

#### Scenario: Divergent backend output
- **WHEN** two comparable backends first emit different token IDs
- **THEN** the result SHALL identify both configurations and the first divergence position

### Requirement: Degradation measurement
The suite SHALL measure n-gram repetition during 2k to 4k-token greedy decodes across configured quants and high-context conditions, including q8_0 KV where planned.

#### Scenario: Low-quant loop detection
- **WHEN** a low-quant open-ended decode completes
- **THEN** the result SHALL include its configured quant, context/KV setting, generated length, and repetition metric

### Requirement: Comparable-track enforcement
The system SHALL compare quantisation quality only within the GGUF family and SHALL report dense and MoE models separately.

#### Scenario: MLX report generation
- **WHEN** a report includes MLX data
- **THEN** it SHALL present MLX as a separate track and SHALL not use it as a GGUF Q4 comparison column
