## 1. Repository Foundation

- [x] 1.1 Create the Python project metadata, package layout, test configuration, and command entry point.
- [x] 1.2 Add a root `.gitignore` for local models/mmproj files, virtual environments, Python caches, sandbox/build output, editor/OS files, generated archives, and heavyweight uncommitted artifacts while retaining report-friendly result metadata.
- [x] 1.3 Define validated YAML models for boxes, expected backends, model/quant matrix cells, KV modes, prerequisites, feature requirements, and pinned versions/hashes.
- [x] 1.4 Add version-controlled box configurations for Spark, Framework, and Mac with their expected CUDA, ROCm, Vulkan, Metal, and MLX backends.
- [x] 1.5 Implement deterministic run ID creation and write resolved configuration plus environment capture to a run `config.yaml` before output is recorded.
- [x] 1.6 Add unit tests for configuration validation, GGUF/MLX track separation, and run-identity generation.

## 2. Runner And Evidence Contracts

- [x] 2.1 Define the runner lifecycle interface for launch, endpoint access, metrics/load-log access, and cleanup.
- [x] 2.2 Implement the llama.cpp runner for CUDA, ROCm, Vulkan, and Metal `llama-server` configurations with explicit process timeouts and cleanup.
- [x] 2.3 Implement the MLX runner for `mlx_lm.server` with the same OpenAI-compatible endpoint contract.
- [x] 2.4 Implement OpenAI-compatible client calls with explicit timeouts, transcript capture, and structured request/response records.
- [x] 2.5 Implement append-only JSONL writers/readers for run events, timings, findings, verdicts, and resumable completed-work detection.
- [x] 2.6 Add integration tests using a local fake OpenAI-compatible server to verify both runner contracts and JSONL resume behaviour.

## 3. Preflight And Smoketest

- [x] 3.1 Implement Tier 0 binary/backend probes for llama.cpp, CUDA, ROCm, Vulkan physical-device detection, Metal load evidence, and expected/ unexpected backend handling.
- [x] 3.2 Implement configured Rust, Python, Node, and Docker prerequisite checks and persist detected versions in preflight evidence.
- [x] 3.3 Add the bundled tiny GGUF and MLX smoketest configurations and implement 32-token inference, NaN, exit-status, and GPU-offload checks.
- [x] 3.4 Implement backend-specific fixed-seed token fingerprint and bounded perplexity verification with non-comparable mismatch results.
- [x] 3.5 Implement requested-feature checks for endpoint chat completion, GBNF, vision image-token count, and maximum planned context allocation.
- [x] 3.6 Implement preflight freshness comparison against binary commit, driver, and OS package state and block stale sweeps.
- [x] 3.7 Add fixture-driven tests for CPU/software-Vulkan fallback, missing backend, fingerprint mismatch, stale evidence, and vision preflight failures.

## 4. Performance And Quantisation Suites

- [x] 4.1 Implement `llama-bench` execution and parsing for distinct prefill and decode headline measurements.
- [x] 4.2 Implement context-fill and KV-mode sweeps that capture decode throughput and time to first token at configured fills.
- [x] 4.3 Implement ten-minute sustained decode sampling and optional configured power-source collection without synthesising unavailable power values.
- [x] 4.4 Implement full-matrix GGUF perplexity collection with model, quant, backend, architecture, and KV dimensions.
- [x] 4.5 Implement fixed-seed cross-backend token-stream comparison with first-divergence detection.
- [x] 4.6 Implement configured long-generation repetition metrics for quant, context-fill, and q8_0-KV degradation tracks.
- [x] 4.7 Add tests for llama-bench parsing, context observation schema, divergence calculation, repeat-rate calculation, and GGUF-only comparability guards.

## 5. Results And Reporting

- [x] 5.1 Create the stable `results/runs/<date>_<box>_<backend>_<model>_<quant>/` layout and compressed transcript storage.
- [x] 5.2 Implement report generation from JSONL into a Markdown summary, per-run pages, and unmatched-review-finding triage pages.
- [x] 5.3 Generate separate dense/MoE, GGUF/MLX, prefill/decode, and available-power-efficiency report views with distributions for repeated samples.
- [x] 5.4 Implement `bench estimate` using measured performance results and the configured remaining matrix.
- [x] 5.5 Add Matplotlib chart generation from raw results without making charts the source of truth.
- [x] 5.6 Add tests that regenerate reports and estimates from fixture JSONL, including interrupted/resumed runs and MLX-track separation.

## 6. Code Review Evaluation

- [x] 6.1 Pin the httpx source commit and create the review-corpus manifest schema for injected bugs, control PRs, historical cases, categories, locations, rationale, and severity.
- [x] 6.2 Build the human-reviewed diff-shaped corpus with 25-30 injected bugs across 12-15 PRs, control PRs, and 5-10 reverted historical fix cases.
- [x] 6.3 Implement lgtmaybe invocation through its OpenAI-compatible provider with correctness/security categories and recursive hunk review configured and recorded.
- [x] 6.4 Implement finding/manifest matching requests to the pinned judge at temperature 0, verdict persistence, and manual false-positive allowlist support.
- [x] 6.5 Calculate precision, recall, and false positives per KLOC and block local sweeps when the frontier validity run does not meet the configured recall threshold.
- [x] 6.6 Add tests for control PR scoring, partial judge verdicts, allowlisted findings, and validity-gate failures.

## 7. Vision Evaluation

- [x] 7.1 Create the VLM testset layout, manifest models, stable filename convention, pinned dimensions, and image validation that rejects unmanifested files.
- [x] 7.2 Implement deterministic floor-plan, chart, architecture-diagram, and table generators with committed seeds and ground truth.
- [x] 7.3 Add owned photo ingestion that strips EXIF and records hand-labelled OCR, object-count, and spatial-relation questions.
- [x] 7.4 Implement constrained JSON request generation and exact, numeric-tolerance, set, structural, and explicitly configured judge scoring.
- [x] 7.5 Gate cross-backend vision report comparisons on successful image-token-count preflight evidence.
- [x] 7.6 Add generator determinism and scorer tests covering exact, numeric, set, structural, and missing-manifest cases.

## 8. Structured Output And Tool Calling

- [x] 8.1 Define schema-backed extraction prompts and manifests with exact ground truth for chart, table, and code-finding examples.
- [x] 8.2 Implement repeated structured-output runs at temperature 0 and 0.7, recording parse, schema, value correctness, and zero-temperature stability independently.
- [x] 8.3 Implement freeform prompt-nagging and GBNF-constrained modes and report their per-backend/quant reliability delta.
- [x] 8.4 Define the typed fake toolset and scenario manifests for selection, arguments, chains, similar names, and intentional no-call cases.
- [x] 8.5 Implement tool execution simulation and scoring for selection, argument validity/correctness, chain completion, and spurious calls.
- [x] 8.6 Add tests for invalid JSON, schema failure, deterministic-output mismatch, no-call false positives, and multi-step chains.

## 9. Coding Repair Evaluation

- [x] 9.1 Define the ten owned coding task concepts in Rust, Python, and JavaScript with model-visible prompts/public examples and isolated hidden tests.
- [x] 9.2 Create per-task Docker images and execution controls that disable networking, enforce CPU/memory limits, and apply a 30-second timeout.
- [x] 9.3 Implement initial-generation and bounded repair-loop orchestration that passes verbatim failures without exposing hidden test source.
- [x] 9.4 Implement pass@1, pass@4, attempts-to-green, did-not-compile, compiled-but-wrong, timeout, and repair-blind outcome classification.
- [x] 9.5 Add integration tests using safe fixture tasks for a passing attempt, compile failure, hidden-test failure, timeout, and repair attempt limit.

## 10. End-To-End Validation And Operations

- [ ] 10.1 Run the complete foundation path on each available box using the tiny model: configuration resolution, fresh preflight, runner launch, performance/quant observations, and report generation.
- [ ] 10.2 Validate that a changed binary or driver produces a stale-preflight rejection and a fingerprint mismatch produces non-comparable reporting.
- [ ] 10.3 Run the code-review frontier validity gate and manually spot-check at least 20 percent of its judge verdicts before local review sweeps.
- [ ] 10.4 Review and approve the tool-calling scenario set before enabling structured-agent sweeps.
- [ ] 10.5 Confirm Linux smart-plug availability and configure power collection only for boxes with a validated meter.
- [ ] 10.6 Run the pruned accuracy matrix and full cheap matrix with resume testing, then regenerate reports and charts from the raw result tree.
