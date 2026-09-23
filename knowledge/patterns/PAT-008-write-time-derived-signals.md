---
id: PAT-008
category: pattern
status: accepted
created: 2026-09-23
updated: 2026-09-23
author: agent
components: [ingestion, fast_path, agentic_runtime, reporter, document_store]
tags: [tier0, ingest-signals, derived-metadata, dirty-flag, novelty-hint, idle, incremental]
related: [PAT-002, PAT-003, PM-004, DEC-003, DEC-010]
supersedes: null
superseded_by: null
affects: ["src/ipa/ingestion/**", "src/ipa/agentic/idle_enrichment.py"]
evidence: ["src/ipa/ingestion/ingest_metadata.py"]
author_model: swe-2
---

# PAT-008 — Señales derivadas en tiempo de escritura (Tier 0)

## Problema

Los ciclos idle Tier 1 re-derivaban señales baratas recorriendo el corpus
entero por ciclo: fetchear textos para títulos/hashes, cargar la matriz de
embeddings históricos de main para novelty, backfill de provenance post-hoc.
Con ~128k chunks es un scan O(corpus) cada ciclo aunque nada haya cambiado —
el mismo patrón de desperdicio que PM-004 castigó a nivel de jobs.

## Solución

**Computar lo derivado-barato al escribir; los ciclos idle consumen flags.**

- Cada vía de ingesta (fast_path inicial + watch, `research_executor`,
  `ingest_reviewed_doc`) persiste en la misma corrida:
  `document_metadata` (`normalized_hash` formato `sha256:` de
  `reporter_curation`, `title`, `published_at`, `char_count`, `extra_json`)
  y `document_sources` con provenance real (`configured_scrape` solo para
  artifacts bajo `Landing/web/**`).
- **Dup exacto vs main** al ingerir: `extra.duplicate_of_main` → T1 decide
  DUPLICATE sin gastar scoring ni embeddings.
- **Novelty hint post-drain**: `extra.novelty_hint` = max coseno vs main +
  `nearest_doc_id` + token de snapshot dual (`main_doc_count` +
  `main_latest_stored_at`: el count detecta adds en el mismo segundo y el
  ts detecta tombstone+add a igual count). El segundo factor del gate
  DEC-003 (Jaccard ≥0.85) solo fetchea el texto del doc más cercano.
- **Refresh incremental de hints stale** (T1): una promoción ya no
  invalida todos los hints — el hint se re-verifica solo contra los
  embeddings de los docs agregados a main desde su snapshot
  (`document_embeddings(new_ids)`, O(nuevos) no O(corpus)) y el resultado
  refrescado se persiste. Fallback a la recarga histórica solo si el doc
  no tiene embedding o falta el LanceDB de main.
- **Self-match excluido**: cuando el corpus curado ES main (`topify_main`),
  los docs en curación se excluyen de `main_url_hashes`, de los históricos
  y del refresh — sin esto un doc se marcaba DUPLICATE de sí mismo
  (URL+hash propio, cosine 1.0) y el sweep borraba su archivo de Landing.
- **Gate "corpus changed"**: writers setean `meta dirty:<corpus>` en
  `topic_clusters.db`; `_t_topify` early-exit sin dirty ni drift de
  conteo/cobertura/provenance. El flag se limpia solo tras corrida exitosa —
  un crash deja el dirty puesto y el próximo ciclo reintenta.
- **Filter-first**: `build_document_dicts(doc_ids=…)` filtra ids antes de
  fetchear texto; `LanceDBIndex.document_embeddings(doc_ids)` acota el read.
- **Texto canónico intacto**: el enrichment de chunks vive en
  `metadata.enrichment.enriched_text` (resuelto por `enriched_text()` para
  embed/BM25/LanceDB), nunca en `chunks.text` — `content_hash` queda íntegro
  y se elimina la clase de desync PM-003.

## Trade-offs

- La ingesta paga un poco más por documento (hashes, una query al main
  store); lo amortiza el primer ciclo idle que no re-deriva nada.
- La invaldiación es manual: todo writer nuevo del corpus debe setear el
  dirty flag o sus cambios quedan invisibles para T1 hasta el próximo drift.
- Una fila `document_metadata` puede existir solo con `extra` (hint sin
  hash): `put_doc_meta` mergea extras y hace COALESCE de columnas NULL, y
  `backfill_doc_metadata` cubre filas con `normalized_hash IS NULL`.
- `aux_progress` (marcas ortogonales tipo `gray_reviewed`) es una tabla no
  ordenada aparte — `enrichment_progress` es una progresión
  `clustered < curated` de una fila por doc y pisarla rompe el checkpoint.

## Ejemplos locales

- `src/ipa/ingestion/ingest_metadata.py` (señales + novelty hints),
  `src/ipa/agentic/idle_enrichment.py` (filter-first, zona gris L2),
  `src/ipa/agentic/chunk_enrichment.py` (`enriched_text()`),
  `ipa/dashboard/server.py` (gate dirty de `_t_topify`).
- Diseño completo: `docs/plans/tier0-signals-idle-optimization.md`.
- Tests: `tests/test_ingest_metadata.py`, `test_chunk_enrichment.py`,
  `test_idle_enrichment.py`.
