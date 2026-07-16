## ADDED Requirements

### Requirement: Owned diff-shaped review corpus
The system SHALL define an owned code-review corpus pinned to an httpx commit with manifest entries for injected bugs, control PRs, historical bug-fix cases, expected file/line ranges, category, rationale, and severity.

#### Scenario: Reviewing a control PR
- **WHEN** the harness selects a manifest-backed control PR
- **THEN** its expected injected-bug set SHALL be empty and unmatched findings SHALL remain eligible for false-positive triage

### Requirement: Local lgtmaybe execution
The review suite SHALL invoke lgtmaybe through its OpenAI-compatible provider against the selected local runner, restricting categories to correctness and security while retaining recursive hunk review.

#### Scenario: Running a review case
- **WHEN** a manifest PR is selected for a local run
- **THEN** the harness SHALL record the lgtmaybe version, category configuration, recursion setting, maximum input tokens, reviewed diff, and findings

### Requirement: Finding scoring and validity gate
The system SHALL match findings to manifest bugs using the pinned `claude-sonnet-4-6` judge at temperature 0, retain judge verdicts, calculate precision, recall, and false positives per KLOC, and require a frontier-model validity run of approximately 80 percent recall before local sweeps.

#### Scenario: Judge-matched finding
- **WHEN** the judge matches a finding to an injected bug
- **THEN** the verdict SHALL record yes, no, or partial matching evidence and feed the corresponding precision and recall calculation

#### Scenario: Failed frontier validity gate
- **WHEN** the validity model does not reach the configured recall threshold
- **THEN** the harness SHALL prevent the local review sweep from starting
