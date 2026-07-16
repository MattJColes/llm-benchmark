## Why

Local-model benchmarking is usually reduced to a single tokens-per-second figure, which cannot establish whether a box, backend, or quantisation level is useful for real code review, vision extraction, structured output, tool use, or coding repairs. A reproducible, backend-neutral harness is needed now to compare the DGX Spark, Framework Desktop, and Mac while preserving the environmental and correctness evidence required to compare results over time.

## What Changes

- Add a Python benchmark harness that runs backend-neutral suites through OpenAI-compatible llama.cpp and MLX endpoints.
- Add configuration and runner support for CUDA, ROCm, Vulkan, Metal, and the separate MLX track across the three declared boxes.
- Add a fail-fast preflight and smoketest that verifies hardware acceleration, environment pins, deterministic fingerprints, and required features before a sweep starts.
- Add performance and quantisation-quality suites that separately measure prefill and decode, context behaviour, sustained throughput, perplexity, numeric divergence, and degradation.
- Add ground-truth-scored suites for code review, VLM extraction, structured output and tool calling, and coding repair loops, with owned reproducible assets.
- Add append-only run data, resumable execution, transcript retention, generated Markdown reports, and runtime estimation.
- Add repository hygiene, including a root `.gitignore` appropriate for local environments, generated results, and large model/test artifacts.

## Capabilities

### New Capabilities
- `benchmark-configuration`: Declares boxes, backend/model matrices, pinned environments, and the llama.cpp/MLX runner boundary.
- `benchmark-preflight`: Gates sweeps on backend detection, inference, correctness fingerprints, feature checks, and recorded environment state.
- `benchmark-performance`: Measures reproducible prefill, decode, context, sustained-load, and available power-efficiency performance data.
- `benchmark-quantisation-quality`: Measures perplexity, cross-backend divergence, repetition/degradation, and KV-cache interactions within GGUF comparisons.
- `code-review-evaluation`: Runs lgtmaybe against an owned, manifest-backed code-review corpus and scores findings for precision, recall, and noise.
- `vision-evaluation`: Generates and evaluates owned VLM assets with deterministic manifests and exact, tolerance, set, structural, or constrained judge scoring.
- `structured-agent-evaluation`: Measures schema-constrained structured output and typed tool-calling reliability against exact ground truth.
- `coding-repair-evaluation`: Runs owned multi-language coding tasks in isolated repair loops and records pass and failure metrics.
- `benchmark-results`: Writes resumable raw run data and regenerates traceable Markdown reports, charts, and time estimates.
- `benchmark-repository-hygiene`: Keeps generated environments, model files, caches, and uncommitted heavyweight result artifacts out of version control.

### Modified Capabilities

None.

## Impact

- Adds a Python application, configuration, test assets, generators, Docker-based coding sandboxes, and reporting tooling to this currently minimal repository.
- Requires pinned llama.cpp, MLX, Python, and per-box native toolchains; Docker is required only for coding evaluation.
- Uses OpenAI-compatible endpoints as the suite boundary and `claude-sonnet-4-6` only for narrow code-review finding matching at temperature 0.
- Stores raw benchmark evidence as JSONL with optional compressed transcripts; large models and raw artifacts remain excluded from Git or use Git LFS when intentionally retained.
