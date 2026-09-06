# Orchestrator improvements

The orchestrator must coordinate independent processes without becoming a second
business-logic layer.

Required properties:

- one active orchestrator per corpus;
- atomic lock and run ID;
- heartbeat and durable phase state;
- graceful shutdown before hard termination;
- independent lexical, vector and enrichment lifecycle states;
- append-only event logs and retained per-run logs;
- explicit safety gates before semantic merge or destructive index operations;
- bounded retries, backoff and recovery.

Implementation belongs in the operations/runtime bounded context; benchmark and
failure-injection evidence belongs in EKS.

## Status (2026-09-06)

Implemented and verified by EXP-005 (E8, 8/8 scenarios): atomic lock and run ID,
heartbeat/state files, graceful shutdown before hard termination, append-only
event logs with per-run logs, durable phase state, bounded retries with
exponential backoff (`run_job_with_retry`, `--retries`/`--backoff`), backpressure
via bounded resource slots (`acquire_slot`/`release_slot`, `--resource`/
`--max-concurrent`), and the per-line idle fix that makes stuck detection
functional at runner level. Independent lifecycle states per job remain
spec-level (one state file per job). Pending for `preferred`: operational
mileage of real orchestrator jobs running with retries and resource slots.
