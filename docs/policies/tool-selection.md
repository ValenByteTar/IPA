# Tool selection policy

Tools are selected per capability and workload, never as universal defaults.

Every candidate is evaluated for:

- contract compliance;
- correctness and coverage;
- provenance;
- idempotency;
- incremental update;
- crash recovery;
- retry/backpressure;
- resource use;
- p50/p95 latency;
- throughput;
- operational complexity;
- privacy/licensing;
- retrieval/response impact.

Promotion states:

```text
observed -> experimental -> candidate -> preferred -> production
```

A benchmark win for one workload does not make a tool the global default.
