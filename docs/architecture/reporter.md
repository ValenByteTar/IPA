# Reporter architecture

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
  -> optional human-gated promotion
```

Reporter must not silently alter the main corpus. Its generated labels, decisions,
claims and topic links are derived records with input hashes and review status.
The Reporter corpus boundary is implemented behind `CorpusService` so Reporter
does not become a second canonical ingestion pipeline.
