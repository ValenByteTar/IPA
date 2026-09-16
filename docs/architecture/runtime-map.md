# Runtime map

## Fast path

```text
Landing
  -> LandingZone
  -> MIME/safety
  -> parser
  -> CanonicalDocument
  -> chunker
  -> DocumentStore
  -> lexical index
  -> first_queryable
```

## Slow/derived paths

```text
DocumentStore -> embeddings -> vector index (LanceDB)
DocumentStore/chunks -> enrichment -> derived text/queries/claims
Reporter corpus -> curation -> topics -> report -> deep dive (chat unificado)
```

Note: deep dive runs inside the unified agent chat (`context=deep_dive` in
`/api/agent/chat/stream`). The standalone `deep-dive.html` page and the
`/api/deep-dive*` endpoints are deprecated (kept for compatibility only —
no UI entry point).

## Public entrypoints

The final public CLI entrypoints should be thin wrappers over `ipa` services:

- fast ingestion;
- continuous processing;
- web scrape;
- Reporter;
- validators;
- benchmark runners;
- local dashboard.

Scripts must not become a second business-logic layer.
