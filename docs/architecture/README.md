# Architecture documentation

This directory contains the current architecture of IPA. It describes ownership,
boundaries and supported flows; it does not replace contracts or experiment
reports.

- `system-overview.md` — current platform and lifecycle.
- `boundaries.md` — ownership between materialization, runtime, Reporter, Tutor and EKS.
- `runtime-map.md` — execution/data-flow map and public entrypoints.
- `agent-runtime.md` — agent core, memory, idle scheduler, cognitive layer, MCP boundary.
- `idle-tiers.md` — Tier 0/1/2 orchestration: leases, gates, signals, tasks, audit, failure model.
- `retrieval.md` — hybrid retrieval pipeline, LanceDB metadata, rerank gate.
- `tutor.md` — Tutor contracts, roadmap gate, focus, unit progress, lessons.
- `reporter.md` — Reporter pipeline and decoupled promotion.
- `dashboard.md` — dashboard state, HTTP API, jobs and orchestration modules.
- `github-surface.md` — what is public versus local/externalized.
- `migration-status.md` — migration state of the orchestration surface.
