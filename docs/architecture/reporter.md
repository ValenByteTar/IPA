| Reporter architecture

Reporter is an optional, isolated analytical capability. It receives a source
folder or isolated corpus, normalizes metadata, deduplicates and curates inputs,
discovers emergent topics, renders a report and supports a bounded deep dive.

```text
input
  -> isolated corpus
  -> metadata/representation
  -> curation
  -> embeddings/clustering
  -> emergent topics
  -> report with provenance
  -> deep dive
```

Reporter must not silently alter the main corpus. Its generated labels, decisions,
claims and topic links are derived records with input hashes and review status.
The Reporter corpus boundary is implemented behind `CorpusService` so Reporter
does not become a second canonical ingestion pipeline.

Promotion to the main corpus is now decoupled from Reporter (DEC-003). The
promotion policy in `promotion_policy.py` evaluates provenance from
`DocumentStore.document_sources`:

- `configured_scrape` → auto-promote (no threshold)
- `agent_research` → promote only if `promotion_score >= 0.70`
- unknown → not promoted

The physical copy is performed by `promotion_executor.py` (idempotent) and
runs from the idle enrichment worker after Level 1. The dashboard exposes
pending promotions at `/api/promotion/queue`. The old `promote_report_to_main()`
and `reporter_promotion.py` were removed.
