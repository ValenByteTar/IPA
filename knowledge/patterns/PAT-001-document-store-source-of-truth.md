---
id: PAT-001
category: pattern
status: accepted
created: 2026-09-05
updated: 2026-09-05
author: human
components: [document_store, ingestion, lexical_index, vector_index, provenance]
tags: [source-of-truth, derived-index, canonical-document, provenance]
related: [BM-001, BM-002, BM-003]
supersedes: null
superseded_by: null
---

# PAT-001 — DocumentStore como fuente de verdad

## Problema

Documentos, chunks, embeddings, índices y enriquecimientos pueden divergir si cada componente conserva una copia autoritativa.

## Solución

Mantener `DocumentStore` como autoridad canónica para documentos y chunks. Tantivy, LanceDB, embeddings y enrichment son representaciones derivadas que pueden reconstruirse desde esa fuente.

```text
CanonicalDocument / DocumentChunk
        ↓
DocumentStore
   ├── Tantivy
   ├── LanceDB
   └── enrichment
```

## Trade-offs

Se requiere reconstrucción de derivados y control de fingerprints, pero se evita que un índice o enrichment local se convierta accidentalmente en autoridad.

## Ejemplos

- `src/res023_lab/document_store.py`
- `docs/IPA_SYSTEM_OVERVIEW.md`
- contratos `CanonicalDocument` y `DocumentChunk`.
