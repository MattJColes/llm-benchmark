# Code-review frontier validity

- Model: `claude-sonnet-4-6`
- Temperature: `0`
- lgtmaybe: `0.11.0`
- Corpus: 24 patches (26 injected bugs, 5 controls, 7 historical fixes)
- Findings: 38
- Precision: 97.37% (37/38)
- Injected-bug recall: 100% (26/26)
- Historical-case recall: 100% (7/7)
- False positives: 1 (9.01 per KLOC across 111 changed lines)
- Manual spot-check: 8/38 verdicts (21.05%; required 20%)
- Validity threshold: 80% recall
- Gate: **PASSED**

## False-positive triage

`httpx/control/type-annotation.patch`, `httpx/_client.py:185`: lgtmaybe claimed that changing the `EventHook` return annotation from `Any` to `object` could break async hooks. The control intentionally changes only typing metadata; the judge and manual review confirmed no manifest bug.

## Evidence

- `results/frontier-review/reviews.jsonl`
- `results/frontier-review/verdicts.jsonl`
- `results/frontier-review/manual-checks.jsonl`
- `results/frontier-review/metrics.jsonl`
