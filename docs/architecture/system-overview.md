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
LanceDB, embeddings, rerankers and enrichment are derived adapters (LanceDB won
the E7 vector competition; sqlite-vec survives only in the memory vector index,
`MemoryVectorIndex` — it is not a corpus vector store).
The fast path must make content lexically available without waiting for GPU or
LLM stages.

The platform has separate consumers:

- Reporter: isolated periodic analysis (promotion is now decoupled — see DEC-003).
- Agent Runtime: bounded retrieval, evidence, context and generation.
- Tutor: learning goals, concepts, roadmaps and assessment.
- EKS: development-time engineering memory, never runtime corpus knowledge.

See `contracts/` for authoritative schemas and `AGENTS.md` for operational rules.
