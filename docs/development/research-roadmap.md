# Research roadmap

## Stage 0 — Reproducibility

Freeze schemas, synthetic manifests, output namespaces, report format and contract tests.

## Stage 1 — Fast path

Landing Zone, durable manifest, MIME routing, PyMuPDF baseline, deterministic
chunking, DocumentStore, lexical availability and first-queryable metric.

## Stage 2 — Index competitions

Compare FTS5/Tantivy and vector backends under append/update/delete/recovery workloads.

## Stage 3 — Parser, OCR and web competitions

Compare parser/OCR strategies and static/dynamic acquisition without blocking the
fast lexical path.

## Stage 4 — Durable processing

Queues, retry/backoff, backpressure, resource isolation and restart/recovery.

## Stage 5 — Progressive enrichment

Candidate selection, deduplication, batch size, LLM budget and on-demand promotion.

## Stage 6 — End-to-end validation

Isolated A/B indexes, identical retrieval/generation policy, ground truth, citation
review, rollback and promotion.

## Stage 7 — Operational hardening

Privacy/retention, backups, observability, runbooks, upgrade compatibility and
failure injection.
