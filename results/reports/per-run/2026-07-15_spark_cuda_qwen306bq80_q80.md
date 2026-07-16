# 2026-07-15_spark_cuda_qwen306bq80_q80

## Timings

```json
{'decode_tokens_per_second': 279.009449, 'kind': 'performance', 'prefill_tokens_per_second': 24632.26056, 'recorded_at': '2026-07-15T12:04:05.942784+00:00', 'sample_id': 'headline-1', 'status': 'completed'}
{'architecture': 'dense', 'backend': 'cuda', 'comparable': False, 'failures': ['cuda token fingerprint does not match the recorded value'], 'kind': 'correctness', 'model_format': 'gguf', 'recorded_at': '2026-07-15T12:10:01.371518+00:00', 'sample_id': 'fingerprint-mismatch-validation', 'status': 'completed'}
{'architecture': 'dense', 'backend': 'cuda', 'comparable': True, 'kind': 'quantisation', 'kv_cache_type': 'f16', 'model': 'qwen3-0.6b-q8_0', 'model_format': 'gguf', 'perplexity': 2.6821, 'quant': 'Q8_0', 'recorded_at': '2026-07-15T12:14:41.900683+00:00', 'sample_id': 'perplexity-qwen3-0.6b-q8_0-cuda', 'status': 'completed'}
{'architecture': 'dense', 'comparable': True, 'decode_tokens_per_second': 279.629519, 'kind': 'performance', 'model_format': 'gguf', 'prefill_tokens_per_second': 24503.398453, 'preflight_status': 'passed', 'recorded_at': '2026-07-15T13:34:10.460640+00:00', 'sample_id': 'headline-post-preflight', 'status': 'completed', 'supersedes_sample_id': 'headline-1'}
```

## Findings

```json
```
