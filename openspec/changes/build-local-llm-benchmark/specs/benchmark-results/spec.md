## ADDED Requirements

### Requirement: Append-only resumable evidence
The system SHALL append timings, findings, verdicts, and run events as JSONL and SHALL preserve compressed full prompts and completions in the run transcript directory.

#### Scenario: Interrupted sweep
- **WHEN** a sweep stops after recording some samples
- **THEN** restarting it SHALL retain completed records and schedule only missing work

### Requirement: Stable run layout
The system SHALL store each run under `results/runs/<date>_<box>_<backend>_<model>_<quant>/` with `config.yaml`, JSONL evidence files, and transcript storage.

#### Scenario: Locating run configuration
- **WHEN** a run directory is created
- **THEN** it SHALL include `config.yaml` before suite observations are appended

### Requirement: Regenerated reports
The reporting command SHALL read raw results and generate Markdown summary and per-run reports containing precision, recall, false positives per KLOC, structured-output validity, tool success, coding pass metrics, and separate prefill/decode data when present.

#### Scenario: Regenerating a report
- **WHEN** raw result records change
- **THEN** rerunning the reporting command SHALL regenerate report content from those records without manually edited aggregate values

### Requirement: Sweep time estimation
The system SHALL estimate remaining matrix time from measured performance-suite throughput and the planned matrix.

#### Scenario: Estimating after baseline performance
- **WHEN** performance measurements exist for a planned model class
- **THEN** the estimate command SHALL project remaining suite time using those observations
