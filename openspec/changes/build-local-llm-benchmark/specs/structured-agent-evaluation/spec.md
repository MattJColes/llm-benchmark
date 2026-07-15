## ADDED Requirements

### Requirement: Structured output reliability
The system SHALL execute approximately 20 schema-backed extraction prompts at temperature 0 and 0.7, recording parse validity, schema validity, and ground-truth value correctness across the configured sample count.

#### Scenario: Invalid JSON response
- **WHEN** a structured-output response cannot be parsed
- **THEN** the result SHALL record parse failure separately from schema and value correctness

### Requirement: Constrained and freeform comparison
The structured-output suite SHALL execute each prompt in prompt-nagged freeform and GBNF-constrained modes and report the reliability delta per backend and quant.

#### Scenario: Determinism at zero temperature
- **WHEN** repeated temperature-0 runs receive identical inputs
- **THEN** the suite SHALL record whether their outputs are identical

### Requirement: Typed tool-calling scenarios
The system SHALL evaluate a fixed typed fake-tool set against single-call, chained, argument-extraction, similar-tool, and no-call scenarios, scoring correct selection, arguments, completion, and spurious calls.

#### Scenario: Correct behavior is no tool call
- **WHEN** a scenario specifies that no tool is needed
- **THEN** any emitted tool call SHALL be recorded as a failure

#### Scenario: Multi-step tool chain
- **WHEN** a scenario requires a second call after a supplied tool result
- **THEN** the suite SHALL score success only when the chain completes with correct calls and arguments
