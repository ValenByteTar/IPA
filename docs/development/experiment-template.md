# Experiment template

```text
experiment_id:
candidate:
capability:
adapter_version:
tool_version:
started_at:
finished_at:
hardware:
input_manifest_hash:
configuration_fingerprint:
```

## Hypothesis

What does this candidate improve and for which workload?

## Controls

- same input manifest;
- same contract version;
- same ground truth;
- same hardware where possible;
- isolated output namespace;
- no production writes.

## Procedure

1. Validate inputs and contracts.
2. Run correctness/coverage.
3. Run incremental/recovery behavior.
4. Inject failure/restart/backpressure when applicable.
5. Measure resources and latency.
6. Measure retrieval/response impact when applicable.

## Results

Record status, quality, throughput, latency p50/p95, memory, recovery,
backpressure and privacy/licensing.

## Decision

`negative | observed | experimental | candidate | preferred | production`.
Explain workloads and trade-offs; do not claim universal superiority.
