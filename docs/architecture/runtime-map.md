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
DocumentStore -> embeddings -> vector index
DocumentStore/chunks -> enrichment -> derived text/queries/claims
Reporter corpus -> curation -> topics -> report/deep-dive
```

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
