---
id: PM-003
category: postmortem
status: accepted
created: 2026-09-11
updated: 2026-09-23
author: human
components: [vector_index, agentic_runtime, ingestion]
tags: [lancedb, duplicates, promotion, idempotency, bm25, sync]
related: [PM-001, PAT-001, DEC-003, EXP-006, PM-005]
supersedes: null
superseded_by: null
evidence: ["src/ipa/agentic/promotion_executor.py", "tests/test_promotion_executor.py"]
affects: ["src/ipa/agentic/promotion_executor.py", "src/ipa/indexes/bm25_index.py"]
---

# PM-003 — Duplicación de vectores LanceDB y desincronización BM25 durante promoción

## Impacto

Después de promover 666 documentos del corpus Reporter al corpus principal, los tres índices del main corpus no coincidían:

- DocumentStore: 40,903 chunks
- BM25: 40,744 chunks (159 faltantes)
- LanceDB: 40,970 vectores (67 duplicados, 40,903 únicos)

Esto significaba que la búsqueda vectorial podía devolver duplicados y que 159 chunks no eran recuperables vía BM25.

## Línea de tiempo

1. La función vieja `promote_report_to_main()` copiaba documentos del Reporter al main corpus. No validaba correctamente si los vectores ya existían en LanceDB del main corpus.
2. La nueva función `promote_documents_to_main()` heredó el mismo patrón: construía `existing_ids` desde `main_lance._table.to_arrow()` pero el check era incompleto.
3. Al promover documentos que ya estaban en el main corpus (promovidos antes vía report approval), los vectores se duplicaron.
4. Separadamente, 159 chunks de un documento original del main corpus (`doc:ad0ae2b7b65ec28d`) nunca se indexaron en BM25 — probablemente un bug preexistente del FastPath.
5. La auditoría de sincronización (DS chunks == BM25 == LanceDB) detectó ambas desincronizaciones.

## Causa raíz

1. **LanceDB duplicados**: el set `existing_ids` se construía desde `to_arrow()` que puede no incluir todos los chunk_ids si la tabla es grande o si hay fragments no compactados. El check `chunk_id not in existing_ids` fallaba silenciosamente para vectores que ya existían.
2. **BM25 faltantes**: el FastPath indexa chunks en BM25 vía `bm25.add_chunks()`, pero si el proceso se interrumpe o el documento se ingiere por una vía que no pasa por `add_chunks()`, los chunks quedan sin indexar.

## Corrección

1. **LanceDB**: deduplicación manual — delete por `chunk_id` duplicado, re-add una sola copia desde el reporter corpus. Resultado: 40,903 rows, 40,903 unique, 0 duplicados.
2. **BM25**: re-indexado de los 159 chunks faltantes vía `bm25.add_chunks()`. Resultado: 40,903 chunks.
3. **promotion_executor.py**: hardened — si no puede leer los IDs existentes de LanceDB, aborta el merge en vez de arriesgar duplicados. Agregado `seen_in_this_batch` para prevenir duplicados dentro del mismo batch.

## Prevención

1. `promote_documents_to_main()` ahora valida `existing_ids` con fallback explícita: si la lectura falla, no procede con el merge.
2. `seen_in_this_batch` previene duplicados dentro del mismo batch de promoción.
3. La auditoría de sincronización (DS == BM25 == LanceDB) debe correr después de cualquier promoción masiva.

## Lección reutilizable

La idempotencia de la promoción debe verificarse por índice, no asumirse. PM-001 ya advertía esto para reprocessing; PM-003 lo confirma para promoción entre corpus. Cada representación derivada (BM25, LanceDB, document_sources) debe tener su propia estrategia de deduplicación.
