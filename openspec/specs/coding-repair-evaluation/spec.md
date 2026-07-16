# coding-repair-evaluation Specification

## Purpose
TBD - created by syncing change build-local-llm-benchmark.

## Requirements

### Requirement: Owned multi-language coding tasks
The system SHALL provide ten owned task concepts in Rust, Python, and JavaScript, each with a model-visible prompt/public example and a hidden test suite that is not included in model input.

#### Scenario: Prompting a coding task
- **WHEN** the suite starts an attempt
- **THEN** it SHALL provide only the task prompt and public example to the model

### Requirement: Isolated bounded repair loop
The system SHALL execute each attempt in a per-task Docker sandbox with no network, CPU and memory limits, and a 30-second timeout; it SHALL permit one initial generation plus at most three repair turns using verbatim compiler or test output.

#### Scenario: Infinite-loop candidate
- **WHEN** generated code exceeds the 30-second sandbox limit
- **THEN** the attempt SHALL terminate and be recorded as a timed-out failure

#### Scenario: Repair attempt
- **WHEN** an initial attempt fails hidden tests
- **THEN** the next model turn SHALL receive the captured failure output without exposing hidden test source

### Requirement: Coding outcome metrics
The suite SHALL sample each task at configured non-zero temperature and report pass@1, pass@4, mean attempts to green, and failure classifications including did-not-compile, compiled-but-wrong, and repair-blind.

#### Scenario: Repeatedly ignored error
- **WHEN** a repair response is near-identical to a failed prior response while not addressing the supplied failure output
- **THEN** the run SHALL classify the outcome as repair-blind
