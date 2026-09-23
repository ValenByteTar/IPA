# Plan — Señales de Tier 0 + optimización de Tiers 1/2

Objetivo: Tier 0 genera en la misma corrida la metadata que hoy T1 re-deriva
costosamente cada ciclo; T1 deja de escanear O(corpus); T2 recupera el win
léxico de EXP-001 y la zona gris deja de ser código muerto.

Invariantes que se respetan: DocumentStore canónico (los metadatos nuevos van
a tablas derivadas), DEC-003 (gate de dos factores se conserva), checkpoints
resumables, nada se borra.

## Bugs confirmados que el plan corrige

- `curate_documents` compara `document["content_hash"]` (= `sha256(text)[:32]`,
  sin normalizar) contra `known_url_hashes` (= `normalized_hash`, formato
  `sha256:<hex>` completo) → la detección de re-descarga idéntica por URL
  **nunca dispara** (`reporter_curation.py:331`).
- `enrich_corpus_level2` curation block es dead code: L1 marca `stage="curated"`
  para todos los docs → `to_curate` siempre vacío en L2.
- `enrich_chunks` hace `UPDATE chunks SET text=...` sin actualizar
  `content_hash` (desync latente, clase PM-003) y no reindexa BM25 — el win
  medido de EXP-001 (Tantivy/BM25 + summary, +14.3% recall@10) no se captura.
- `published_at` hardcodeado `""` en `build_document_dicts` aunque el scraper
  lo extrae (`scrape_report.results[].date`, trafilatura).
- Provenance por backfill post-hoc en cada ciclo T1 pese a que Tier 0 conoce
  el origen al ingerir.

## Fase A — Fundación: tablas derivadas + helpers

**`src/ipa/storage/document_store.py`**
- Nueva tabla `document_metadata` (derivada, rebuildable):
  `document_id PK, normalized_hash TEXT, title TEXT, published_at TEXT,
  char_count INTEGER, extra_json TEXT, computed_at TEXT`.
- `document_sources` gana columna `published_at TEXT` (ALTER guardado,
  mismo patrón que `spans_json`).
- Métodos: `put_doc_meta()`, `get_doc_meta()`, `all_doc_meta()`,
  `url_normalized_hashes()` (join sources+metadata → `main_url_hashes`
  barato), `put_source(..., published_at=None)`.
- `backfill_doc_metadata(store, limit=None)` — rellena filas faltantes
  (hash + title + chars) una vez por corpus.

**`src/ipa/agentic/chunk_enrichment.py`**
- `enriched_text(chunk) -> str`: representación derivada
  `[Summary]/[Questions] + canonical`. Resuelve canonical desde
  `metadata.enrichment.canonical_text` (chunks legacy ya mutados) o `text`
  (chunks nuevos). Punto único de verdad para embed/index.

## Fase B — Señales Tier 0 en la misma corrida

**Nuevo `src/ipa/ingestion/ingest_metadata.py`** — `record_ingest_metadata()`
compartido por fast_path_cli y research_executor. Por doc ingerido:
- `document_metadata`: normalized_hash, title (primera línea), char_count,
  published_at (de `scrape_report.json` por `saved_to`→path, fallback línea
  `Source:`/`Date:` del texto, fallback `stored_at`).
- `document_sources`: si el artifact está bajo `<landing_root>/web/**` →
  `configured_scrape` + url/domain/quality/published_at del reporte o del
  texto (misma heurística que `backfill_from_landing_registry`, en el momento).
- Dup short-circuit: si `normalized_hash` ya existe en el corpus main →
  `extra_json.duplicate_of_main = <doc_id>` (hecho registrado; la decisión
  la toma T1, ver Fase C).

**`src/ipa/ingestion/fast_path_cli.py`**
- Tras `ingest_directory` (corrida inicial + cada pasada watch):
  `record_ingest_metadata()` sobre los docs nuevos.
- Post-drain (cuando `done` y drenado completo, junto a `_compute_centroids`):
  **novelty hints** — para docs del corpus staging sin hint: max coseno vs
  `main.document_embeddings()` + `nearest_doc_id` + `main_doc_count` →
  `extra_json.novelty_hint`. Una sola carga de embeddings de main, matmul
  numpy. Solo cuando el corpus del run no es main y main existe
  (`--main-corpus`, default `outputs/experiments/E12-corpus`).

**`src/ipa/agent/research_executor.py`**
- Tras la ingesta: `record_ingest_metadata()` (mantiene `record_agent_research`
  como está — la metadata extra es complementaria).

## Fase C — T1 filter-first + curación con hints

**`src/ipa/agentic/idle_enrichment.py`**
- `build_document_dicts(..., doc_ids=None)`: filtra antes de fetchear;
  prefiere title/published_at/normalized_hash persistidos.
- `enrich_corpus_level1` reordenado:
  1. sets baratos primero (`all_centroids`, `processed_doc_ids`,
     `is_promotion_pending`, `all_sources`);
  2. `needed = unclustered ∪ uncurated` → build dicts solo de esos;
  3. evaluación de política itera `sources_map` (sin textos);
  4. `main_url_hashes` vía `url_normalized_hashes()` de main (+ backfill
     acotado para docs legacy);
  5. embeddings/textos históricos de main **solo si** algún doc a curar no
     tiene hint válido (`main_doc_count` coincide);
  6. docs con `duplicate_of_main` → decisión DUPLICATE directa.
- Early-exit por dirty-flag + `live_doc_count == last_count` +
  checkpoints cubriendo todo (Fase F).

**`src/ipa/reporter/reporter_curation.py`**
- `curate_documents(..., novelty_hints=None)`: `hints[doc_id] =
  {max_cosine, nearest_doc_id, nearest_text}` — usa el hint en vez de la
  matriz histórica; la confirmación Jaccard del gate de dos factores se hace
  contra `nearest_text` (DEC-003 intacto).
- Fix formato: comparaciones por `normalized_hash` consistente
  (`document.get("normalized_hash") or normalized_hash(text)`) en
  `known_url_hashes` y `by_hash`; `content_hash` queda solo para evidence.
- Fix `_in_period`: fechas naive (`YYYY-MM-DD` de trafilatura) contra bounds
  aware lanzaban TypeError fuera del `except ValueError` → ahora se
  interpretan UTC (bug real que `published_at` haría explotar).

**`src/ipa/indexes/lancedb_index.py`**
- `document_embeddings(doc_ids=None)`: filtro `WHERE document_id IN (...)`
  — el scan de tabla completa por ciclo era el cuello O(corpus) de T1.
- `sync_doc_metadata`: `published_at` prefiere `document_sources.published_at`
  (Tier 0) con `documents.stored_at` como fallback.

## Fase D — enrich_chunks canónico + reindex léxico

**`src/ipa/agentic/chunk_enrichment.py`**
- `write_enrichment` ya NO toca `chunks.text` — guarda
  `enrichment.{summary,questions,enriched_text}` en `metadata_json` →
  `content_hash` queda íntegro, invariante canónico restaurado.
- `scan_chunks` detecta enriquecidos por `metadata.enrichment` (no por
  prefijo de texto) — compatible con filas legacy.
- Reindex: `reembed_batch` embede `enriched_text` (ya lo hace vía el texto
  pasado) + **nuevo**: `bm25.remove_chunk` + `add_chunk` con texto
  enriquecido.
- El drain de embeddings (`_index_lancedb_incremental` + `_embed_drain_loop`
  en fast_path_cli) embede `enriched_text(chunk)` → representación vectorial
  consistente en cualquier camino de indexación.

**`src/ipa/dashboard/server.py`**
- `_t_enrich_chunks` pasa `bm25=BM25Index(MAIN_CORPUS/"bm25_index.db")`.

## Fase E — L2: zona gris en vez de código muerto

**`src/ipa/agentic/idle_enrichment.py` — `enrich_corpus_level2`**
- Reemplaza el bloque `to_curate` muerto por: decisiones `REPORTER_ONLY` con
  `agent_research` + `promotion_score` en banda `[IPA_GRAY_LO=0.5,
  IPA_GRAY_HI=0.70)` → `classify_many` (batch, ≤24/pase, preemptible) →
  persiste decisión `idle-enrichment-llm` → `evaluate_promotion` →
  `mark_promotion_pending` si pasa → marca `gray_reviewed` para no
  re-evaluar en loop.
- **Desviación de implementación**: `enrichment_progress` es una progresión
  ordenada de una fila por doc (`clustered < curated`) — un stage
  `gray_reviewed` ahí pisaría el checkpoint `curated` y `processed_doc_ids`
  lo trataría como nivel 0 (todo doc quedaría "ya revisado"). Se agregó la
  tabla `aux_progress(document_id, stage, processed_at)` no ordenada con
  `mark_stage()`/`stage_doc_ids()` para marcas ortogonales.
- La eligibility se computa antes del early-exit de L2: los candidatos no
  necesitan dicts ni clusters — un corpus quieto puede tener zona gris
  pendiente igual.

## Fase F — Gate "corpus changed"

**`src/ipa/agentic/topic_clusters.py`**
- Tabla `meta(key TEXT PK, value TEXT)` + `set_meta`/`get_meta`.
- Writers setean `dirty:<label>`: fast_path_cli (reporter), research_executor
  (reporter), `ingest_reviewed_doc` (main), `process_promotion_queue` (main).
- `_t_topify`: early-exit si `not dirty` y count inalterado y checkpoints
  cubren todos los docs vivos; al correr consume el flag y guarda count.

## Fase G — Tests + docs

- `tests/test_ingest_metadata.py` (nuevo) + `test_chunk_enrichment.py` y
  `test_idle_enrichment.py` extendidos:
  meta roundtrip, provenance-at-ingest, normalized_hash fix de la
  comparación rota, dup-flag, hint en curate (dup confirmado / no dup /
  stale hint), enrich_chunks canónico (text intacto + bm25 actualizado +
  content_hash estable), gray-zone, dirty-gate early-exit.
- `AGENTS.md` (bloque Tier 0 signals), `CHANGELOG.md`,
  `docs/architecture/agent-runtime.md`.

## Orden de implementación

A → B → C → D → E → F → G. Cada fase compila y tests propios antes de seguir.

## Refinamientos post-implementación (self-review)

Encontrados y corregidos en la auditoría posterior:

- **`web_root` misclassification**: con `--input Landing` (default CLI, no el
  `Landing/web` del pipeline) un archivo manual en `Landing/` raíz hubiera
  recibido provenance `configured_scrape` → auto-promoción sin curación.
  `record_ingest_metadata` ahora exige que el artifact esté bajo `web/**`
  salvo que `web_root` ya sea el dir `web` (`ingest_metadata.py`).
- **Re-encolado eterno de docs ya promovidos**: `pending_eval_ids` filtraba
  solo `is_promotion_pending` (status='pending') → un doc 'promoted' se
  re-evaluaba y re-encolaba cada ciclo (INSERT OR REPLACE → pending →
  executor → done: churn). Nueva `promotion_queue_doc_ids()` (todos los
  estados) + diferencia de conjuntos — además elimina N queries por ciclo.
- **Hint-only rows sin backfill**: `backfill_doc_metadata` seleccionaba
  `m.document_id IS NULL` → una fila creada solo por novelty_hint quedaba
  con `normalized_hash` NULL para siempre. WHERE ampliado a
  `m.normalized_hash IS NULL`.
- **`get_document` antes del skip** en `record_ingest_metadata`: ahora el
  early-skip (meta con hash + source ya registrada) evita el fetch.
- **`published_at` de meta no usaba la fecha del scrape_report** (solo la
  línea `Date:`): la resolución del artifact ahora precede la escritura de
  metadata y el reporte tiene precedencia. `_load_scrape_report` también
  busca en `<dir>/web/` cuando el input es el landing padre.
- **`_title_from_text`** devolvía "Title: X" con prefijo — strippeado.
- **Backfill de main sin límite** dentro del ciclo T1 (~4k docs legacy en
  una pasada) → `IPA_T1_META_BACKFILL_LIMIT` (default 1000/ciclo).
- **Hint staleness → refresh incremental**: el check `main_doc_count`
  invalidaba TODOS los hints ante cualquier promoción (recarga histórica
  completa). El hint ahora lleva `main_latest_stored_at` (token de
  snapshot); en T1 un hint stale se re-verifica solo contra los embeddings
  de los docs agregados a main desde ese snapshot
  (`document_embeddings(new_ids)` filtrado — O(nuevos), no O(corpus)) y el
  hint refrescado se **persiste** (self-heal: el próximo ciclo ya es válido
  sin trabajo). Validez = count AND stored_at (count detecta adds en el
  mismo segundo; ts detecta tombstone+add a igual count). Fallback al path
  histórico solo si el doc no tiene embedding o LanceDB de main falta.
- **Tests nuevos**: manual-Landing-no-scrape, hint-only refill (re-ingesta
  y backfill), `main_latest_stored_at` en el hint, `_refresh_novelty_hint`
  (función pura).

### Segunda pasada (bugs activados/descubiertos)

- **`os` no importado en `idle_enrichment`**: el backfill acotado usaba
  `os.environ` sin import → NameError habría abortado el ciclo L1 entero al
  primer `to_curate` con `main_corpus_path` (ningún test pasaba main).
  Importado a nivel módulo + test de integración L1 con main real.
- **Self-match en `topify_main`** (corpus == main): el fix del formato de
  hash ACTIVÓ un path antes muerto — un doc de main con `document_sources`
  matcheaba su propia URL+hash → DUPLICATE "__main__" → rejection → el
  sweep borraba su archivo de Landing. Mismo hazard por embedding
  (cosine(self)=1.0) en el path histórico y en el refresh incremental
  (doc en `new_ids`). Fix: cuando `main_corpus_path == corpus_path` se
  excluyen los docs en curación de `main_url_hashes`, de los históricos y
  de `new_embs`.
- `document_embeddings(doc_ids)`: variable renombrada (el param se
  reasignaba a los ids de las filas).
