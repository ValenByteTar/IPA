---
id: PM-001
category: postmortem
status: accepted
created: 2026-09-05
updated: 2026-09-05
author: human
components: [lexical_index, vector_index, ingestion, idempotency]
tags: [duplicates, restart, reprocessing, tantivy, lancedb]
related: [PAT-001, PAT-002]
supersedes: null
superseded_by: null
---

# PM-001 — Riesgo de duplicación de índices durante reprocessing

## Impacto

La reejecución de un archivo podía duplicar entradas en índices derivados aunque `LandingZone` y `DocumentStore` conservaran IDs estables. Esto degradaba ranking y podía devolver el mismo chunk varias veces.

## Línea de tiempo

1. LandingZone registraba artifacts por hash.
2. DocumentStore hacía upsert por `document_id` y `chunk_id`.
3. Los índices derivados inicialmente agregaban registros sin una garantía equivalente de unicidad.
4. La auditoría de idempotencia identificó el riesgo en Tantivy y LanceDB.
5. Se incorporaron operaciones de eliminación/reemplazo por `chunk_id` antes de agregar derivados.

## Causa raíz

Se asumió que la idempotencia de la fuente canónica se propagaba automáticamente a índices que no imponían primary keys sobre sus registros.

## Corrección

- Tantivy elimina documentos existentes por `chunk_id` antes de agregar.
- LanceDB elimina registros existentes por `chunk_id` antes de insertar.
- Se mantienen hashes e IDs estables en el flujo de ingestión.

## Prevención

Toda nueva representación derivada debe tener estrategia explícita de deduplicación, fingerprint, reanudación y test de reprocessing.

## Lección reutilizable

La idempotencia debe verificarse componente por componente; no alcanza con que el pipeline o el store canónico sean idempotentes.
