---
id: PAT-002
category: pattern
status: accepted
created: 2026-09-05
updated: 2026-09-05
author: human
components: [ingestion, orchestration, vector_index, enrichment]
tags: [fast-path, slow-path, availability, decoupling, backpressure]
related: [BM-001, BM-002, EXP-001]
supersedes: null
superseded_by: null
---

# PAT-002 — Fast path y slow path desacoplados

## Problema

Embeddings, enrichment y parsers costosos pueden bloquear la disponibilidad inicial de nuevos documentos.

## Solución

Publicar primero el camino mínimo consultable:

```text
parse → chunk → DocumentStore → índice lexical
```

Ejecutar como procesos derivados independientes:

```text
DocumentStore → embeddings → LanceDB
chunks → enrichment → re-embedding
```

## Trade-offs

El sistema puede tener disponibilidad parcial y requiere exponer estados `pending`/`complete`, pero reduce tiempo hasta la primera consulta y permite reanudar etapas lentas.

## Ejemplos

- `src/res023_lab/fast_path.py`
- `scripts/run_continuous_pipeline.py`
- `src/res023_lab/document_store.py:embedding_jobs`.
