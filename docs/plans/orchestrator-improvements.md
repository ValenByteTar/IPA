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
