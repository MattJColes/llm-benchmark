## ADDED Requirements

### Requirement: Separate prefill and decode measurement
The performance suite SHALL use `llama-bench` for GGUF headline measurements and record prefill and decode throughput as distinct values for every completed matrix cell.

#### Scenario: Reporting a GGUF benchmark
- **WHEN** a GGUF performance run completes
- **THEN** its result SHALL include separate prefill and decode throughput values rather than one combined tokens-per-second value

### Requirement: Context and KV sweep
The performance suite SHALL measure decode throughput and time to first token at 0, 32k, 64k, and 128k context fill for both f16 and q8_0 KV tracks when the configured model supports the planned context.

#### Scenario: Recording long-context data
- **WHEN** a configured 64k q8_0 KV measurement completes
- **THEN** the result SHALL identify the context fill, KV type, decode throughput, and time to first token

### Requirement: Sustained performance evidence
The performance suite SHALL run decode for at least ten minutes and retain time-series throughput observations; it SHALL record tokens per joule only when a configured power source is available.

#### Scenario: No power meter available
- **WHEN** a run has no configured power measurement source
- **THEN** the system SHALL retain throughput data and omit tokens-per-joule rather than synthesising a value
