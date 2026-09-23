---
id: PAT-002
category: pattern
status: accepted
created: 2026-09-05
updated: 2026-09-23
author: human
components: [ingestion, orchestration, vector_index, enrichment]
tags: [fast-path, slow-path, availability, decoupling, backpressure]
related: [BM-001, BM-002, EXP-001, PAT-008, PAT-006]
supersedes: null
superseded_by: null
evidence: ["src/ipa/ingestion/fast_path.py", "tests/test_fast_path.py"]
affects: ["src/ipa/ingestion/**"]
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

- `src/ipa/ingestion/fast_path.py`
- `scripts/operations/run_continuous_pipeline.py`
- `src/ipa/storage/document_store.py:embedding_jobs`.

## Addendum (2026-09-23)

El boundary fast/slow se refinó: las señales derivadas baratas
(`normalized_hash`, título, provenance, dup-flag, novelty hints) se computan
en la misma corrida de ingesta en vez de re-derivarlas por ciclo idle. Ver
PAT-008.
