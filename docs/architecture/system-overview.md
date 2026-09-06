# IPA system overview

IPA is a local-first platform that acquires, validates, parses, chunks, stores,
indexes and enriches information for agent consumption.

```text
source
  -> acquisition
  -> Landing artifact
  -> MIME and safety validation
  -> parser
  -> CanonicalDocument
  -> deterministic chunks
  -> DocumentStore
  -> derived lexical/vector views
  -> optional enrichment
  -> Reporter, MCP or Agent Runtime
```

`DocumentStore` is the canonical source for documents and chunks. Tantivy, FTS5,
LanceDB, sqlite-vec, embeddings, rerankers and enrichment are derived adapters.
The fast path must make content lexically available without waiting for GPU or
LLM stages.

The platform has separate consumers:

- Reporter: isolated periodic analysis and human-gated promotion.
- Agent Runtime: bounded retrieval, evidence, context and generation.
- Tutor: learning goals, concepts, roadmaps and assessment.
- EKS: development-time engineering memory, never runtime corpus knowledge.

See `contracts/` for authoritative schemas and `AGENTS.md` for operational rules.
