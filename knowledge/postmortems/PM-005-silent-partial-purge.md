---
id: PM-005
category: postmortem
status: accepted
created: 2026-09-23
updated: 2026-09-23
author: agent
components: [promotion_executor, ingestion, lexical_index, agentic_runtime]
tags: [promotion, purge, bm25, sqlite-locked, desync, retry, defer]
related: [PM-003, PM-004, DEC-007, PAT-007, PM-002]
supersedes: null
superseded_by: null
affects: ["src/ipa/agentic/promotion*", "src/ipa/indexes/bm25_index.py"]
evidence: ["src/ipa/agentic/promotion_executor.py"]
author_model: swe-2
---

# PM-005 — Purga parcial silenciosa del staging durante promoción

## Impacto

El 2026-09-23 se encontraron **18.035 filas FTS vivas** en el
`bm25_index.db` del staging para documentos que ya habían sido tombstoned
del DocumentStore — búsqueda lexical devolviendo chunks de docs "promovidos
y limpiados". Corregido a mano.

## Causa raíz

El paso BM25 de `purge_promoted_from_source` corría un `DELETE`/`UPDATE`
directo sobre el índice del source. Con una ingesta fast-path concurrente
sosteniendo el writer lock, SQLite devolvía `database is locked`, el
`except` lo **logueaba y continuaba** — la purga se reportaba exitosa con el
FTS desincronizado. La promoción es multi-índice (DocumentStore, BM25,
LanceDB): un paso que falla en silencio convierte la idempotencia en desync.

## Corrección

- `_purge_source_bm25`: toma el writer lock por adelantado (`BEGIN
  IMMEDIATE` + `busy_timeout`) y reintenta acotado (4 intentos, 3 s) — la
  contención transitoria de una ingesta se absorbe sin abortar.
- Si el índice nunca cede, el paso se reporta en `incomplete_steps` y
  `promote_documents_to_main` **differe el batch**: la cola queda `pending`,
  el próximo ciclo reintenta los pasos fallidos (todos idempotentes). Nunca
  `done` con el staging desincronizado.
- Mismo criterio aplicado a los pasos DocumentStore y LanceDB.

## Prevención

- `tests/test_promotion_executor.py`: lock de escritura real sobre
  `bm25_index.db` → defer + pending + reconciliación completa al liberar;
  camino feliz sin pasos incompletos.

## Lección reutilizable

En una operación multi-índice, "el paso falló" no es un log — es un estado
del resultado. Cada paso reporta y la operación agrega defiere; la
idempotencia de los pasos individuales es lo que hace seguro el retry. Es la
misma familia de PM-003 (verificar por índice, no asumir) con la causa de
PM-004 (contención con trabajo pesado concurrente).
