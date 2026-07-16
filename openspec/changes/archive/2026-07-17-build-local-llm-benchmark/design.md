## Context

This repository currently contains only project scaffolding. The benchmark must compare single-user local inference on a DGX Spark, Framework Desktop, and Mac without coupling suites to a particular serving stack. It must retain enough provenance to distinguish a real model or backend change from a changed binary, driver, preprocessing path, or silent CPU fallback.

The first delivery establishes the harness, configuration, preflight, raw-data contract, reporting pipeline, and the inexpensive performance and quantisation suites. Owned evaluation assets and the accuracy suites follow on the same contracts, then the full cross-box sweeps and blog reporting can run unattended.

## Goals / Non-Goals

**Goals:**
- Run all suites through one OpenAI-compatible endpoint contract while isolating llama.cpp and MLX launch, metrics, and preflight behaviour behind runner implementations.
- Make every result reproducible from pinned configuration, environment capture, append-only JSONL, deterministic asset manifests, and retained transcripts.
- Make invalid benchmark conditions fail before a sweep and make partial sweeps resumable without overwriting prior observations.
- Score task correctness against owned ground truth where possible and limit the frontier judge to code-review finding matching.
- Keep GGUF backend/quant comparisons separate from the non-equivalent MLX track.

**Non-Goals:**
- Establish absolute capability rankings against public benchmarks or cloud models.
- Benchmark concurrent serving, batching throughput, or production deployment characteristics.
- Treat MLX quantisations as directly equivalent to GGUF quantisations.
- Build a dashboard before raw evidence and Markdown reports are useful.

## Decisions

### Backend-neutral suite API

Suites call an OpenAI-compatible endpoint and receive only runner-provided endpoint and lifecycle information. `LlamaCppRunner` launches `llama-server` for CUDA, ROCm, Vulkan, and Metal GGUF runs; `MlxRunner` launches `mlx_lm.server` for the MLX track. Runner-specific probes and measurements remain in the launch/preflight layer.

This uses the compatible APIs already provided by both stacks rather than adding a custom serving protocol. Calling binaries directly from every suite was rejected because it would duplicate request, timing, and cleanup behaviour and make MLX support a special case.

### Declarative configurations and immutable run identity

Versioned YAML defines boxes, expected backends, model/quant matrix cells, toolchain prerequisites, and pinned hashes. A run ID includes date, box, backend, model, and quant. Before a run begins, the harness resolves this configuration and writes an exact `config.yaml` with its environment block to that run directory.

This makes configuration reviewable and allows existing runs to be read without reconstructing command-line state. Database-backed orchestration was rejected because append-only files are simpler, portable across the three boxes, and naturally crash-safe.

### Preflight is a freshness-gated prerequisite

Preflight emits the same JSONL envelope as suites, records binary and environment fingerprints, and marks a box eligible only while the recorded llama.cpp commit, driver versions, OS package state, and required feature checks match the planned run. It fails for missing expected backends, software Vulkan, insufficient GPU offload, failed token fingerprint/perplexity tolerance, or a requested feature that is unavailable. Unexpected backends are warnings.

This prevents silent CPU execution and incomparable historical results. A one-time setup check was rejected because drivers and builds can change without a model or harness change.

### Separate comparable tracks

GGUF suites compare CUDA, ROCm, Vulkan, and Metal at the same model file and quant. MLX runs use their own model/configuration, fingerprint table, and reports. Dense and MoE models, f16 and q8_0 KV, and prefill and decode values remain separate dimensions in stored observations and reports.

Combining these values into a single leaderboard would obscure the constraints that make a result meaningful.

### Evidence-first scoring and reporting

Suites append timestamped timings, findings, verdicts, and compressed transcripts. Reports and charts are regenerated from those files. Evaluation manifests carry exact expected values, tolerances, asset generation seed, and scoring mode; code-review finding matching uses the pinned judge at temperature 0 and stores every verdict.

Hand-authored summaries or a mutable aggregate database were rejected because they cannot be audited when a surprising score needs investigation.

### Isolated coding execution

Each coding attempt is executed in a per-task Docker sandbox without network access, with CPU/memory limits and a 30-second timeout. Hidden tests stay outside the model prompt. Failure output is supplied verbatim for up to three repair turns after the initial generation.

Host execution was rejected because generated code must not access local files or network services, and an unbounded loop must become a data point rather than a hung sweep.

## Risks / Trade-offs

- [Long-running sweeps fail or are interrupted] -> Append every observation independently, identify completed work from raw JSONL, and resume only missing samples.
- [A build silently uses CPU or software Vulkan] -> Require runner load-log evidence, physical-device checks, and GPU-offload thresholds before recording a pass.
- [Numerical changes invalidate comparisons] -> Store backend-specific token fingerprints and perplexity tolerance bands; mark mismatches non-comparable rather than mixing results.
- [Small owned corpora produce wide error bars] -> Report distributions and sample counts, retain task-level evidence, and state the limitation in reports and posts.
- [Large model files and transcripts bloat Git] -> Ignore local models, caches, generated sandboxes, and raw temporary artifacts; use Git LFS or an external tarball only when transcript retention is explicitly enabled.
- [Judge judgments drift] -> Pin the judge model, temperature, prompt, and corpus version, limit judging to binary/partial finding matching, and spot-check the first corpus run.
- [External toolchains differ by box] -> Treat pinned prerequisite checks as preflight data and retain their detected versions in every run configuration.

## Migration Plan

1. Add the repository layout, package metadata, `.gitignore`, configuration schemas, runner interface, and JSONL envelope.
2. Implement and run preflight plus performance/quantisation suites on a small model to prove the complete raw-data-to-report path.
3. Add owned manifests, generators, and sandboxed evaluators without changing the runner or result contracts.
4. Enable the pruned accuracy matrix only after the bug-corpus frontier validity gate passes and the tool scenario set is approved.
5. Rerun preflight after every binary, driver, or OS package change; preserve prior data and mark fingerprints that no longer compare.

Rollback consists of stopping active runners and retaining the append-only data already written. New report versions can be regenerated from the unchanged raw data; no destructive migration is required.

## Open Questions

- Whether one or two smart plugs are available for the Linux boxes before power-efficiency reporting begins.
- Final review of the tool-calling scenario set before the first structured-agent sweep.
- Whether retained full transcripts will use Git LFS or a Git-ignored archival tarball for the initial results publication.
