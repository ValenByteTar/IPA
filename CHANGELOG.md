# CHANGELOG — IPA

Formato: [versión] — fecha. Estilo Keep a Changelog (resumido).

## Unreleased — 2026-09-24 (PM-007: race de cargas de modelo + SSE honesto)

- **`MODEL_LOAD_LOCK`** (`src/ipa/model_load_lock.py`, nuevo): RLock global
  que serializa toda construcción pesada de modelos in-process. Motivo:
  `transformers`/`accelerate` parchean `nn.Module.register_parameter` a
  nivel clase durante `from_pretrained` — un loader concurrente deja params
  en device `meta` para siempre ("Cannot copy out of meta tensor"). El
  dashboard los cargaba en warmups paralelos y el retrieval quedaba roto
  todo el uptime. Cubiertos: `BGEM3FlagModel`, `FlagReranker`,
  `easyocr.Reader`, `_load_locked()` de ExL3 (orden: vram.lock →
  MODEL_LOAD_LOCK). Singletons con double-check + tripwire meta que falla
  fuerte en vez de dejar un adapter corrupto.
- **SSE honesto en retrieval**: `api.py` ya no emite `empty` tras
  `error`/`timeout` (decía "no hay datos" cuando la búsqueda ni corrió);
  el ctx volátil informa el fallo técnico. El `empty` genuino lleva
  `auto: true` cuando el auto-research va a disparar → la UI anuncia
  "investigando en la web automáticamente…" (botón queda como fallback).
- Tests: `tests/test_model_load_lock.py` (5 casos: ctor bajo lock,
  single-construction bajo contención, tripwire meta, reranker, OCR).
- Postmortem completo: `knowledge/postmortems/PM-007`.

## Unreleased — 2026-09-23 (regla única de evidencia en EKS)

- **Una sola regla para "¿este record tiene evidencia?"**. Estaba implementada
  **tres veces** con escapes distintos: `validate()` (gate de promoción) y
  `report()` solo salteaban `.lock`, mientras `_missing_artifact_links()`
  también salteaba `outputs/agent/**` y las citas declaradas históricas — dos
  veredictos sobre el mismo hecho. Ahora hay un helper
  (`_body_evidence_citations`) + una regla pública (`record_has_evidence`) que
  consumen los tres.
- **El chequeo de rot nunca miraba `docs/`** (solo `outputs/`). Al unificar
  aparecieron 9 citas: 6 eran ruido de prosa (`docs/s` y `docs/chunks` son
  *unidades de tasa* — "0.69 docs/s" — y `docs/adr` es un path inexistente a
  propósito, DEC-008). Se exige que la cita sea un **archivo** (extensión),
  no un directorio ni una tasa.
- Consecuencia del ajuste: 3 records citaban **directorios** como evidencia
  (DEC-004, EXP-006, PM-002) — ahora llevan `evidence:` explícito a archivos
  reales. Quedan 3 warnings verdaderos, todos citas a docs que **nunca
  existieron en este repo** (git no tiene registro de
  `docs/IPA_SYSTEM_OVERVIEW.md`, `docs/TOOL_DECISION_FRAMEWORK.md`,
  `docs/06-llm-benchmark-summary.md`; el último vive en el repo externo
  `small-model-deliberation`).
- Tests: 6 nuevos en `tests/test_eks.py` — el gate y el reporte comparten la
  regla, `outputs/agent/**` no es evidencia, un directorio no es evidencia,
  prosa/tasa no es cita, cita muerta en `docs/` sí se reporta, y una línea
  histórica no es ni evidencia ni rot. Suite: 1152 passed, 1 skipped.

## Unreleased — 2026-09-23 (research progress + fixes)

- **Progreso granular de research**: `execute_research(on_progress=...)`
  emite `(phase, detail)` en cada transición — search → judge → scrape
  (por URL, con done/total/accepted/rejected) → ingest → embed → retrieval.
  `run_research.py` lo vuelca a `research_progress.json` (`phase` +
  `phase_detail`, limpia `heavy_wait` al avanzar) y el indicador del
  dashboard muestra la fase con conteos en vez de "running" durante
  minutos. El payload inicial marca `phase: "starting"`.
- **`reporter_curation`**: `float('significant')` ya no aborta el batch —
  `quality_score`, `max_cosine` de novelty hints y valores del LLM judge
  pasan por `_bounded_float` (clamp [0,1] + fallback); un score
  no-numérico deja de tirar la curación entera.
- **Troubleshooting de locks**: `docs/operations/troubleshooting.md`
  documenta tier0/heavy/vram/maintenance-lock — formato `pid|owner|ts`,
  TTLs y env overrides, semántica de stale (pid muerto o TTL vencido →
  autolimpieza), cómo diagnosticar un lock "trabado" y mapa síntoma→causa.
- **`list_topics`**: salta centroides/chunks huérfanos (doc tombstoneado o
  eliminado tras indexar) al leer — un índice derivado stale ya no devuelve
  `document_id`s que `get_topic_info` no puede abrir. Cierra el drift que
  dejó el repair del manifiesto de paths (140 refs muertas en clusters).
- Tests: `test_on_progress_emits_phases_in_order` (orden estricto de
  fases + callback que explota no corta el run),
  `test_curation_survives_non_numeric_quality_score`,
  `test_save_article_manifest_uses_basenames_only` (regresión del fix de
  paths absolutos en el manifiesto de documentos linkeados).

## Unreleased — 2026-09-23 (research → staging → promoción, DEC-003b)

- **La research ya no escribe directo al corpus principal**: los documentos
  aceptados por el juez aterrizan en `outputs/agent/research_staging/`
  (staging propio del agente, path fijo — no el puntero móvil del reporter).
  La curación T1 (`topify_research_staging`) + `promotion_policy` deciden qué
  entra a main; `promotion_queue` ya agrupaba por `source_corpus` por entry.
- **`user_provided`**: nueva clase de proveniencia — URLs pegadas por el
  usuario (seeds de una corrida) auto-promueven como `configured_scrape`;
  el resto sigue `agent_research` con gate `promotion_score >= 0.70`.
  Ambas respetan los gates de curación (DUPLICATE / INSUFFICIENT_EVIDENCE).
- `execute_research(staging_corpus_dir=...)`: ingesta + provenance +
  ingest_metadata + dirty flag + embeddings al staging; el retrieval final
  fusiona hits de main (hybrid) + BM25 del staging (`staging_bm25`), así la
  respuesta ve el material fresco aunque aún no esté promovido.
- Backlog residual de embeddings en staging → lanza
  `run_embed_drain --corpus <staging> --background` post heavy-phase (lease
  propio, reanudable, escala a GPU bulk ≥512) — sin él la promoción defería
  indefinidamente por el preflight de cobertura vectorial (PM-004).
- T1: nueva `topify_research_staging` (+ `deep_topify_research_staging` en
  T2) y recurso `corpus_research`; `promotion_queue` lo incluye.
- La cola de review (`ingest_reviewed_doc`) también deja de bypassear:
  los promote del LLM re-review aterrizan en el mismo staging
  (`_research_ingest_corpus()` centraliza el destino).
- Kill switch: `IPA_RESEARCH_STAGING=0` restaura la ingesta directa a main.
- Tests: 5 nuevos — routing a staging (docs/provenance fuera de main),
  seed→`user_provided`, búsqueda→`agent_research`, drain residual lanzado,
  política `user_provided` (auto + gate de duplicados).

## Unreleased — 2026-09-23 (EKS governance + work permits)

- **EKS `affects` file-scoping**: nuevo campo opcional de frontmatter —
  globs repo-relativos que un record gobierna. `eks_governing(paths)`
  (MCP) y `context(paths=)` activan records por *dónde se edita*, no por
  query; incluye rejected/superseded (cementerio anti-reintentos).
  Validador: glob sin archivos que matchear → warning (exento `outputs/`).
- **Work permits dev-time (PAT-009)**: `tools/work_permits.py` +
  `scripts/cli/permit.py` — Permit-to-Work para sesiones Devin paralelas.
  `acquire` rechaza solapes de scope `exclusive`, adjunta los records EKS
  que gobiernan el scope como precautions, marca hot zones (≥4 records);
  `close --eks-draft` enlaza el conocimiento capturado. Estado en
  `outputs/devin/permits/` (gitignored). Enforcement vía
  `.devin/hooks.v1.json` + `scripts/hooks/permit_guard.py`: PreToolUse
  bloquea edits bajo permiso exclusivo ajeno e inyecta governing records;
  SessionStart lista permisos; SessionEnd los cierra; Stop recuerda
  cerrar permisos abiertos; PostCompaction los re-inyecta.
- **Devin config en sinergia**: `.devin/rules/eks-workflow.md`
  (protocolo siempre activo); skills `session-closeout` (cierre con
  cosecha EKS) y `session-launch` (ventanilla previa a sesiones
  paralelas); `experiment-logging` y `adr-proposal` actualizados al
  scaffold + DEC-* + nuevos campos; `engineering-context-builder` gana
  paso 0 (`eks_governing` sobre paths + `permit.py check`).
- **Provenance gate**: campo `evidence` (paths que deben existir) —
  `accepted` creados desde 2026-09-23 sin evidencia verificable = error;
  legados = warning agregado. `author_model`/`trigger` registran qué
  agente produjo el record; convención `trigger: permit:PW-*`.
- **eks_report**: `awaiting_evidence`, `hot_zones`, `author_models`.
- `tests/test_eks.py`: 27 tests — matcher de globs, governing, gate de
  evidencia, ciclo de vida de permisos.

## Unreleased — 2026-09-23 (EKS governance hardening, v0.2.0)

- **`affects` backfill**: 35 records sin `affects` (todas las DEC/PAT/PM/
  BM/EXP/RES fundacionales) ahora declaran los globs que gobiernan, así el
  circuito `eks_governing`/precautions cubre el catálogo entero y no un
  cuarto: editar `src/ipa/agent/**` trae DEC-002/DEC-006, `src/ipa/agentic/**`
  trae DEC-005/EXP-003/PM-003. `evidence` agregado a los 22 `accepted`
  legados que no la tenían (paths reales: tests/runners/módulos);
  `author_model` en DEC-010/EXP-009.
- **`acquire` atómico (PAT-009)**: check-de-conflicto + id + escritura bajo
  un lockfile `O_CREAT|O_EXCL` (`.acquire.lock`, stale 30 s, espera 5 s) —
  dos sesiones ya no pueden ganar el mismo scope exclusivo en simultáneo.
  Test de concurrencia: 8 sesiones → 1 gana.
- **Cierre que no pierde conocimiento**: `SessionEnd` libera el scope pero
  todo permiso cerrado sin `--eks-draft` deja `.unharvested-<session>.json`;
  el siguiente `SessionStart` lo reporta y lo borra. `Stop` ahora bloquea
  **una sola vez** por sesión (marcador + `stop_hook_active`) pidiendo el
  closeout, en vez de un `additionalContext` cuyo soporte en `Stop` no está
  documentado. `SessionStart` poda los `.seen-*.json` > 7 días.
- **`eks_report`**: nueva sección `hot_zones_overlap` — el criterio de
  prefijo que decide las precautions de un permiso, para que el reporte no
  diga "0 hot zones" mientras un `acquire` recibe 11 precautions.
- **`components.json` con `groups`**: `indexes`/`ingestion`/`memory`
  agrupan los componentes específicos; filtrar por el bucket alcanza a los
  records etiquetados con cualquier miembro y viceversa
  (`EKSRepository.component_matches`, usado por `eks_search`/`eks_context`/
  `eks_list`). Alias resueltos en el filtro.
- **Ruido de evidencia**: `_missing_artifact_links` deja de marcar rutas de
  runtime (`outputs/agent/**`: staging, review queue) y citas ya declaradas
  históricas en la propia línea; los bullets "Artefacto:" de BM/EXP
  históricos se marcan como tales. `validate_eks` pasa **sin warnings**.
- **Liveness scan acotado**: `_repo_files()` saltea `outputs/` (163k
  archivos), `Archive/`, `models/`, `local_archive/`, `build/`, `dist/`,
  `exllamav3-dev/` — `validate_eks` local deja de tardar >10 s.
- **`docs/adr` fuera del MCP**: `IPA_EKS_REFERENCE_ROOTS` queda sin default
  (no registra un root muerto) y `validate_eks.py` deja de pasarlo.
- **Skill renombrada**: `engineering-context-builder` (repo) →
  `eks-engineering-brief` — colisionaba por nombre con la skill global de
  Windsurf, que ganaba y devolvía otro contrato de salida.
- **RES-005** promovido a `accepted` con addendum: DEC-006 ya implementó su
  recomendación (planner + task queue; unidad de bound por tarea). El gap
  de benchmark quality-vs-round sigue abierto.
- Tests: `tests/test_permit_guard.py` (15: bloqueo, inyección una vez por
  sesión, marcador unharvested, Stop-once, poda, routing de eventos) +
  `tests/test_eks.py` (5: concurrencia de acquire, grupos/alias de
  componentes, hot zones por solape, skip del liveness). Suite: 1146 passed,
  1 skipped.
- **Skills portadas desde Windsurf**: `self-review`, `refactoring` y
  `rag-component-development` viven ahora en `.devin/skills/`, reescritas con
  el protocolo real (gate de evidencia, `eks_governing`, scaffold vía
  `eks_new.py`, fronteras de `boundaries.md` + PAT-001/003/004/008 y
  PM-001/003). Las 6 skills globales de `~/.codeium/windsurf/skills/` se
  movieron a `skills-backup-20260923/` (fuera del árbol de descubrimiento,
  con README de restauración): no se portaron `engineering-context-builder`
  (duplicada por `eks-engineering-brief`), `documentation` (superada por
  `experiment-logging`) ni `adr-compilance-review` (cubierta por
  `adr-proposal` + `eks_governing`). Las 3 portadas también quedaron fuera de
  servicio como globales, para no reintroducir la colisión de nombres.
  `~/.codeium/windsurf/memories/global_rules.md` intacto.

## Unreleased — 2026-09-23 (research GPU embed escalation)

- `_embed_new_chunks` (research_executor) escala al lote GPU exclusivo cuando
  el backlog de la corrida ≥ `IPA_EMBED_GPU_MIN_BACKLOG` (512): clama el job
  de mantenimiento (`research_embed`), publica estado (chat pausado), toma
  `vram_lock` con espera acotada al budget de ingesta (≤120 s), descarga
  Ollama y embebe en CUDA; al terminar restaura el chat y `release_gpu()`
  deja el adapter en CPU lazy. Fallback inline CPU si no hay lease/VRAM.
- Fix: el embed de research ahora embebe `enriched_text()` (representación
  canónica), alineado con el drain — antes usaba `chunks.text` crudo.
- `EmbeddingAdapter.release_gpu()`: libera la ventana GPU sin recargar el
  modelo (close() dejaba `device="cuda"` → recarga accidental sobre el chat).
- `_start_bulk_gpu` acepta `wait_s` (default `IPA_EMBED_GPU_WAIT_SECONDS`).
- Tests: 2 nuevos en `test_research_executor.py` (escalada sobre umbral,
  no-escalada bajo umbral con job sin clamar).

## Unreleased — 2026-09-23 (LanceDB column projection)

- `table_chunk_id_list(table)` + `LanceDBIndex.chunk_ids()` en
  `lancedb_index.py`: lectura proyectada `search().select(["chunk_id"])`
  en vez de `to_arrow()` (que materializaba los vectores — ~1.3 GB para
  332k chunks de 1024 dims; la proyección lee solo la columna id, ~1.9 s).
  Fallback a `to_arrow()` para tablas/fakes sin query API; `None` marca
  "ilegible" para los call sites que deben distinguirlo de "vacío".
- Migrados los 5 consumidores que solo necesitaban ids: capa física de
  `index_audit` (lista cruda para dup count), preflight `_uncovered_vector_ids`
  y dedupe de copia en `promotion_executor` (None → defer seguro, no
  procede a duplicar), resume-checkpoint y skip de `_embed_drain_loop` en
  `fast_path_cli`, y `_embed_new_chunks` en `research_executor`.
- `document_embeddings`/`_compute_centroids`/copia de vectores en
  promotion quedan con `to_arrow()`: esos SÍ necesitan las columnas.

## Unreleased — 2026-09-23 (get_document tool)

- Nueva system tool `get_document` (`system_tools.py`): abre un documento
  por `doc_id` (de hits de `search_corpus` o de la cola de promoción) y
  devuelve identidad (título, `char_count`, `normalized_hash`, `published_at`),
  provenance real (`document_sources`: url/dominio/clase/quality), lifecycle
  (live/tombstoned, `duplicate_of_main`, novelty hint), estado en
  `promotion_queue`, `curation_decision` completa y texto acotado
  (`max_chars` ≤8000). Busca en main → `research_staging` → stagings del
  reporter — un doc pendiente de promoción no está en main todavía.
  Read-only (sqlite `mode=ro`, patrón `index_audit`); se desbloquea tras
  `search_corpus` en `TOOL_PROGRESSION` y entra al catálogo MCP vía
  `tool_specs()` automáticamente.
- `_doc_corpora()`: resolución de corpus candidatos compartida (main +
  research staging fijo + pointer persistido del reporter + resto de
  `quality-check/*/corpus` por recencia).
- `configs/agent_identity.yaml`: capability `get_document` + skill
  `auditar_origen` (search_corpus → get_document).
- Tests: 5 nuevos en `test_system_tools.py` (registro completo, doc en
  staging, tombstoned, no encontrado, arg faltante).

## Unreleased — 2026-09-23 (index audit idle task)

### `index_audit` — auditoría de salud del corpus (idle T1, read-only)
- Nuevo `ipa/agentic/index_audit.py`: dos capas con cadencias distintas.
  - **Lógica** (barata, cada `IPA_AUDIT_LOGICAL_SECONDS`=900s): consume las
    señales Tier 0 persistidas — docs vacíos vivos (`char_count=0`;
    NULL = no computado, no vacío), `duplicate_of_main` fugados (flag sin
    decisión DUPLICATE en `curation_decisions`), `normalized_hash`
    repetido entre vivos, hints de novelty stale, cola de backfill de
    `document_metadata`, y `configured_scrape` sin `source_url`.
  - **Física** (cara, cada `IPA_AUDIT_PHYSICAL_HOURS`=6h, forzada — el
    drift por kills/locks no setea dirty flags): compara sets de
    `chunk_id` entre store ↔ BM25 meta ↔ FTS ↔ LanceDB (faltantes,
    huérfanos, `chunk_id` duplicados) y chunks spam (`content_hash` en ≥3
    docs = boilerplate same-site; en ≥2 dominios = sindicación, reportado
    aparte para revisión — nunca tombstone automático).
  - La comparación física es por `chunk_id`, nunca por texto: BM25/LanceDB
    guardan la representación `enriched_text()` mientras `chunks.text`
    queda canónico.
- Read-only por diseño: solo escribe `outputs/agent/index_health.json`
  (merge por capa — una corrida lógica preserva el último resultado
  físico). Status `ok`/`warn`/`fail`; expuesto en `/api/state` como
  `index_health` y renderizado como tarjeta "Salud de índices" en el
  dashboard.
- Gating: salta si el lease Tier 0 está activo (no auditar índices en
  escritura parcial); recursos `cluster_store` + `corpus_main` la
  serializan contra topify/promoción.
- Fix en `BM25Index.add_chunks`: el membership check de deletes FTS no
  filtraba `tombstoned` → restores masivos pagaban un scan FTS completo
  por chunk resucitado (~80ms × N). Ahora solo borra ids live.
- `tests/test_index_audit.py` (11 tests): capas, detección de drift real,
  marker-only rows, merge de capas en el JSON.

## Unreleased — 2026-09-23 (Tier 0 signals + idle T1/T2 optimization)

### Señales de ingesta Tier 0 (nuevo `ipa/ingestion/ingest_metadata.py`)
- **`document_metadata`** (tabla derivada nueva en DocumentStore):
  `normalized_hash` (formato `sha256:` de `reporter_curation`), `title`,
  `published_at`, `char_count`, `extra_json` (dup flags, novelty hints).
  `document_sources` gana columna `published_at` (migración aditiva al abrir).
- **Provenance en el momento**: artifacts bajo `<landing>/web/**` registran
  `configured_scrape` con url/domain/quality/fecha del `scrape_report` o de
  las líneas `Source:`/`Date:` del texto — el backfill post-hoc de T1 queda
  como repair path, no como fuente.
- **Dup exacto vs main**: si el `normalized_hash` ya existe en el main
  corpus → `extra.duplicate_of_main` → T1 decide DUPLICATE sin gastar
  scoring ni embeddings.
- **Novelty hints post-drain**: max coseno vs main + `nearest_doc_id` +
  snapshot dual (`main_doc_count` + `main_latest_stored_at`) por doc nuevo.
  T1 los consume: si el hint sigue válido no carga la matriz histórica; si
  quedó stale lo re-verifica solo contra embeddings de docs nuevos en main
  (`document_embeddings(new_ids)` — O(nuevos)) y persiste el resultado. El
  gate de dos factores DEC-003 se conserva (Jaccard ≥0.85 contra el texto
  del doc más cercano, fetcheado solo para ese caso).

### Tier 1/2: filter-first, gate "corpus changed", zona gris
- **`build_document_dicts(doc_ids=…)`** filtra ids antes de fetchear textos;
  `LanceDBIndex.document_embeddings(doc_ids)` acota el read con WHERE.
  Fin del scan O(corpus) por ciclo idle.
- **Gate dirty** (`topic_clusters.meta`): writers (fast_path, research
  ingest, `ingest_reviewed_doc`, `process_promotion_queue`) setean
  `dirty:<corpus>`; `_t_topify` early-exit cuando no hay dirty ni drift de
  conteo/cobertura/provenance. El flag solo se limpia tras una corrida
  exitosa (crash → reintento).
- **Zona gris L2**: reemplaza el bloque de "curación LLM" que era dead code
  (L1 marcaba todo como curated → `to_curate` siempre vacío). Docs
  `agent_research` con decisión `reporter_only` y `promotion_score` en
  `[IPA_GRAY_LO=0.5, IPA_GRAY_HI=0.70)` reciben segunda opinión batched
  (`classify_many`, ≤`IPA_GRAY_LIMIT=24`/pase): la decisión se actualiza con
  scores + fingerprint del modelo, se re-evalúa `evaluate_promotion` y se
  encola si supera el umbral. Duplicados definitivos y otras proveniencias
  no van al LLM. Marca `aux_progress.gray_reviewed` (tabla nueva no
  ordenada — `enrichment_progress` es la progresión clustered<curated).
- **`published_at` real** en `build_document_dicts` (meta → sources → "") —
  el filtro de período y `sync_doc_metadata` dejan de hardcodear/usar solo
  `stored_at`.

### enrich_chunks canónico + índice léxico sincronizado
- `chunks.text` ya NO se muta: la representación enriquecida
  (`[Summary]`/`[Questions]` + canónico) vive en
  `metadata.enrichment.enriched_text` y la resuelve `enriched_text()`
  (compat con filas legacy prefijadas). `content_hash` queda íntegro —
  elimina el desync latente de la clase PM-003.
- `reembed_batch` actualiza también BM25 (`remove_chunk`/`add_chunks` con el
  texto enriquecido) — captura el win medido de EXP-001 (lexical+summary,
  +14.3% recall@10) que antes solo llegaba a LanceDB.
- El drain de embeddings (`_index_lancedb_incremental`, `_embed_drain_loop`)
  embede `enriched_text()` → cualquier camino de indexación usa la misma
  representación derivada.

### Fixes
- `curate_documents`: la comparación URL-duplicado usaba `content_hash`
  (`sha256(text)[:32]`, sin normalizar) contra `normalized_hash`
  (`sha256:<hex>`) → nunca matcheaba. Ahora ambos lados usan el formato
  normalizado persistido por Tier 0.
- `_in_period`: fechas naive (`YYYY-MM-DD` de trafilatura) comparadas con
  bounds aware lanzaban TypeError (no cubierto por `except ValueError`) →
  naive se interpreta UTC.
- Introspección post-implementación: backfill cubre filas `document_metadata`
  hint-only (`normalized_hash IS NULL`); `promotion_queue_doc_ids()` evita
  re-encolar docs ya promovidos; `web_root` exige `web/**` para provenance
  `configured_scrape`; guard anti self-match cuando corpus == main (evita
  DUPLICATE de sí mismo → sweep); `os` faltante en `idle_enrichment`;
  backfill de main acotado (`IPA_T1_META_BACKFILL_LIMIT=1000`).
- **DEC-010** (`knowledge/decisions/`): contrato formal de orquestación
  Tier 0/1/2 — condiciones de activación, leases y exclusión documentadas.

## Unreleased — 2026-09-23 (bulk embedding, FP8 experiment design, PM-004)

### Bulk GPU embeddings: maintenance mode
- **Threshold automático 512** (`IPA_EMBED_GPU_MIN_BACKLOG`): con >=512 chunks
  pendientes, toma un lease exclusivo de VRAM y mantiene BGE-M3 en la RTX 4050
  FP16 hasta que LanceDB alcanza al DocumentStore. Debajo no se toma ese lease;
  el `device=auto` del adapter aún decide por el gate de VRAM y puede elegir CUDA.
  Para fijar CPU se requiere `IPA_EMBED_DEVICE=cpu`. Microbenchmark pareado sobre
  64 chunks reales: CPU FP32 batch 4 = 2.86–2.99 chunks/s; GPU FP16 batch 4/8 =
  124–135 chunks/s.
- **Chat deshabilitado a propósito y señalizado**: status durable en
  `outputs/web_dashboard/embedding_maintenance.json`, HTTP 423 para chat/deep dive,
  banner visible con fase/progreso/ETA, composer y roles deshabilitados. Ollama
  genera bajo un `vram.lock` lease durante cada stream; el bulk espera a que
  termine el request activo antes de descargar el modelo. Restringe nuevas cargas
  Ollama/ExL3 mientras BGE posee la GPU. Un job lock cross-process serializa el
  drain con idle T1/T2 y las rutas manuales de promoción/review/reindex, evitando
  que un promotion purge toque el corpus durante la vectorización.
- **Ciclo de VRAM**: esperar lock (TTL/heartbeat), descargar Ollama y esperar la
  liberación física, cargar BGE en CUDA, drenar el backlog completo, liberar BGE
  y hacer warmup del modelo Ollama que estaba/configurado antes de reabrir chat.
  Si la carga exclusiva CUDA no está disponible, re-warma chat y el adapter cae
  a CPU; `IPA_EMBED_GPU_WAIT_SECONDS` acota espera de otra ocupación.
- **Resumible**: lease de un solo drain, checkpoint por `chunk_id` insertado en
  LanceDB; al reinicio omite lo ya persistido. `run_embed_drain.py --status`
  muestra backlog/ETA; `--max-seconds` pausa entre batches. `IPA_EMBED_GPU_BULK=0`
  y `--cpu-only` solo desactivan el lease bulk: para garantizar CPU, configurar
  `IPA_EMBED_DEVICE=cpu`.
- **Sin consolas parpadeantes en Windows**: liveness usa Win32 process handles
  (sin lanzar `tasklist`); el probe `nvidia-smi` usa `CREATE_NO_WINDOW`.
  `run_embed_drain.py --background` relanza bajo `pythonw.exe` y deja salida en
  `logs/embed_drain.log`.
- **Tuning real**: batch por device 4 CPU/4 GPU (configurable con
  `IPA_EMBED_BATCH_CPU/GPU`); batch 64 reducía GPU real a ~53 chunks/s por el
  padding de chunks variables, frente a ~129 con batch 4/8. El probe fijó 6
  threads CPU; el adapter no los configura y callers alternativos pueden
  sobrescribir batch (`continuous_pipeline` 256, `lancedb_incremental`/Tier 2
  re-embed 192, chunker semántico 16).
- **NVFP8 BGE-M3**: diseño EXP-009 propuesto para Ada/RTX 4050; no se implementó
  backend, no se cambiaron defaults y no se ejecutaron pruebas/benchmarks.
- Tests de regresión cubren lock, provider lease, umbral 511/512, fallback CPU,
  restore, estado stale, reanudación end-to-end, rechazo/chat gate y consola
  Windows oculta. Suite completa: **1023 passed, 1 skipped**. Evidencia del
  incidente y el
  microbenchmark pareado: PM-004.

### Tier 0: lease de ingesta + cierre del pipeline huérfano
- **`ipa/agentic/tier0.py`**: lease cross-process `outputs/agent/tier0.lock`
  (formato `pid|owner|ts`, heartbeat cada `IPA_TIER0_HEARTBEAT`=15s, TTL
  `IPA_TIER0_LOCK_TTL`=300s, staleness por `pid_alive`). `run_fast_path.py`
  lo toma al arrancar y lo mantiene durante toda la ingesta (BM25 + drain de
  embeddings, watch incluido); una segunda instancia detecta el holder vivo
  y sale — la ingesta es idempotente, un duplicado solo contiende.
- **Gate en el idle scheduler**: `_idle()` devuelve False mientras el lease
  Tier 0 esté vivo, incluidos watchers huérfanos de un dashboard anterior
  (no figuran en `JOBS` del proceso actual). Como `_idle()` resetea
  `LAST_ACTIVITY`, la cuenta de idle para Tier 1/Tier 2 arranca recién
  cuando el lease se libera.
- **Sentinel `.scraper_done` lo escribe el propio scraper**: nuevo flag
  `--done-file` en `run_web_scrape.py`/`scrape_cli.py`, escrito en
  `try/finally` (éxito o fallo). Antes lo escribía el thread del pipeline —
  si el dashboard moría/reiniciaba a mitad del scrape, el gate nunca llegaba
  y el watch de fast_path retenía el lease bulk de embeddings para siempre
  (banner 100%, chat bloqueado). El write del pipeline queda como fallback
  para kill duro del scraper.
- **Orphan exit en el watch**: si el padre (pipeline/dashboard) murió y el
  gate no existe, el watcher sale del modo watch en vez de loopear eterno.
- **Stale pipeline progress**: `dashboard_state()` marca `interrupted` un
  `pipeline_progress.json` que diga `running` sin jobs vivos y sin escritura
  en >5 min — la card "PIPELINE EN VIVO" y el indicador del chat dejan de
  mostrar una corrida muerta.
- **`_find_running_processes` sin wmic**: `wmic` fue removido de Windows 11
  y la limpieza de huérfanos (startup + pipeline) era un no-op silencioso;
  ahora consulta `Get-CimInstance` (POSIX: `ps`).
- Tests: `test_tier0.py` (claim/release/steal/heartbeat), `test_scraper_done
  _sentinel.py` (done-file en éxito y fallo, orphan check, parseo CIM).

### Diseño experimental BGE-M3 FP8 (no implementado)
- Añadido `EXP-009` como propuesta para FP8 E4M3/NVIDIA scaling en Ada SM89.
  FlagEmbedding/EmbeddingAdapter no ofrece un switch FP8; FP16 GPU sigue siendo
  el default. No se cambió código/configuración ni se ejecutó prueba o benchmark.
- Documentados los límites reales de CPU: adapter batch 4 por default, threads
  no fijados por código, `--cpu-only`/`IPA_EMBED_GPU_BULK=0` no pinnean CPU y hay
  callers alternativos que sobrescriben el batch.

### Promoción: preflight de cobertura vectorial (PM-004)
- **La purga ya no corre sin vectores verificados**: `promote_documents_to_main`
  comprueba que cada chunk vivo del source tenga vector en main LanceDB antes
  de purgar el staging. Si falta alguno, la promoción se **difiere** (source
  intacto, cola `pending`) y el drain completa los faltantes; el ciclo siguiente
  reintenta idempotentemente. Main LanceDB ilegible → defer (nunca marca done).
- `process_promotion_queue` expone `deferred_docs` y `_t_promotion` lo reporta
  en el log del idle scheduler.
- Escape de emergencia: `IPA_PROMOTION_REQUIRE_VECTORS=0` (default `1`).
- Tests: `tests/test_promotion_executor.py` — defer sin vectores, cobertura
  completa, retry tras backfill (cola end-to-end), opt-out y main ilegible.

### Promoción: la purga del staging ya no deja desync silencioso
- **Causa**: el paso BM25 de `purge_promoted_from_source` solo logueaba un
  `database is locked` y continuaba — una ingesta fast-path concurrente dejó
  18.035 filas FTS vivas para docs ya purgados del DocumentStore (corregido
  a mano el 2026-09-23).
- **Fix** (`promotion_executor.py`): el paso BM25 corre ahora en
  `_purge_source_bm25` con `BEGIN IMMEDIATE` + `busy_timeout` y reintentos
  acotados (4 intentos, 3s). Si sigue lockeado, la purga reporta
  `incomplete_steps` (DocumentStore/BM25/LanceDB) y `promote_documents_to_main`
  **difere el batch** — la cola queda `pending` y el próximo ciclo reintenta
  los pasos que fallaron (todos idempotentes), nunca marca `done` con el
  staging desincronizado.
- Tests: lock real de escritura sobre `bm25_index.db` → defer + pending +
  reconciliación completa al liberar; camino feliz sin pasos incompletos.

### Ingesta: status `no_text` para artefactos sin texto extraíble
- **Causa**: un parse exitoso con 0 chunks (PDF solo-imagen cuyo OCR no
  produjo nada, archivo vacío) igual terminaba `indexed` y guardaba un
  documento de texto vacío que la curación rechazaba después — el artefacto
  figuraba como procesado OK sin haber producido nada consultable (caso real:
  PDF de 669KB de thehackernews, 1 página escaneada, 0 chars).
- **Fix** (`fast_path.py`): `chunks == 0` post-parse → status `no_text`,
  **sin guardar el documento**; `ingest_directory` lo saltea en re-runs.
- **Sweep** (`landing_sweep.py`): `no_text` se clasifica como `delete` en
  Landing y en la re-evaluación de Transit — el archivo se borra como un
  `failed`, no queda "esperando confirmación" eternamente.
- Tests: no_text no entra al store ni se reprocesa; sweep lo borra en
  Landing y en Transit.

### Curación: gate de novelty ya no rechaza por embedding solo (falsos positivos)
- **Causa raíz**: `novelty < 0.05` se calculaba como `1 − max_cosine` del vector
  de documento contra todo el histórico. Un único vector por doc queda dominado
  por el boilerplate del sitio: alertas CISA semanales distintas (CVEs y fechas
  diferentes) medían 0.95+ de coseno y se rechazaban entre sí. Auditoría:
  1.124 rechazados "casi idénticos", ~168 sin cobertura en main eran falsos
  positivos (Jaccard real 0.31–0.77, artículos distintos con mismo template).
- **Fix** (`reporter_curation.py`): la coincidencia por embedding ahora exige
  confirmación léxica — Jaccard ≥0.85 de tokens contra el documento histórico
  matcheado (`argmax` del coseno). Sin confirmación → `REPORTER_ONLY`, nunca
  descarte. Si no hay texto histórico alineado para verificar, también se
  conserva (un match fuzzy no verificable nunca destruye). El fallback léxico
  (sin embeddings) sigue rechazando directo: Jaccard >0.95 ya ES confirmación.
- **Plumbing** (`idle_enrichment.py`): `_load_historical_embeddings` devuelve
  (doc_ids, vectores) alineados y el camino idle pasa `historical_documents`
  con el texto de cada doc de main para la confirmación.
- **Rehabilitación** (`scripts/operations/rehabilitate_rejected_docs.py`):
  dry-run por defecto; selecciona rechazados "casi idénticos" sin cobertura en
  main, un-tombstonea docs/chunks, limpia `embedding_jobs`, restaura BM25,
  marca `review_status=pending` + `rehabilitated` y re-encola con el
  `source_corpus` correcto. Backups SQLite automáticos antes de mutar.
  Ejecutado: 168 docs rehabilitados (133 tombstoned + 35 vivos), todos
  promovidos a main con cobertura vectorial verificada por el preflight.
- Tests: 4 nuevos en `tests/test_reporter.py` — confirmación léxica requerida,
  duplicado verdadero con `duplicate_of`, conservación cuando el match es
  inverificable, y fallback léxico intacto.

### Dashboard: recuperación del corpus Reporter activo
- **Puntero durable**: `run_full_pipeline` persiste el output activo en
  `outputs/web_dashboard/active_reporter_output.json` (escritura atómica); el
  dashboard lo recupera tras reinicio del watchdog. Antes `_ACTIVE_REPORTER_OUTPUT`
  vivía solo en memoria: un restart borraba la referencia.
- **Fallback corregido**: si el puntero falta o apunta a un output borrado,
  seleccionar entre todos los corpora de `quality-check/` por documentos vivos,
  incluso si todavía no tienen `report.json`. Antes ganaba el `report.json` más
  reciente, que podía pertenecer a un corpus vacío y mostrar 0 con datos vivos.
- Test de regresión: restart simulado, puntero stale y newest-empty-vs-populated.
  `tests/test_web_dashboard.py`: 33 passed.

### Research: aislamiento, budgets y prioridad de trabajos pesados
- **Dir de trabajo por corrida** (`_research_run_dir`): `research_topic` scrapea
  a `outputs/agent/research/<stamp>-<slug>/` en vez del `Landing/web`
  compartido, e ingesta **solo ese dir**. Mismo cambio en los otros tres
  consumidores: el Tutor (`execute_approved_research`, default None), el CLI
  (`agent.py research --landing` default None) y los workers de review de docs
  rechazados (idle 60 s + pase T2 batched → `outputs/agent/research/review/`). Antes hacía
  `ingest_directory("Landing/web")` → heredaba los ~600 archivos de una ingesta
  masiva concurrente (violando su propio contrato "accepted material only") y
  competía por el mismo árbol. El handoff al corpus principal no cambia:
  ingesta + `provenance=agent_research` + `web_source` (PAT-003); el dir privado
  queda como traza de auditoría.
- **Budget post-scrape** (`IPA_RESEARCH_INGEST_BUDGET`, default 600 s): el
  `max_seconds` de la tool solo acotaba el loop de scrape; ingesta y embeddings
  no tenían corte. Medido: una research con presupuesto de 120 s llevaba 27+ min.
  `_embed_new_chunks` ahora embebe **solo los documentos de la corrida**
  (antes: todo chunk pendiente del corpus canónico, en una sola llamada),
  batcheado (`IPA_RESEARCH_EMBED_BATCH`, 64) y con deadline entre batches.
  `budget_used.ingest` audita deadline, lock y excedido.
- **Lock de trabajos pesados** (`ipa/agentic/heavy_lock.py`,
  `outputs/agent/heavy.lock`): serializa fases pesadas con prioridad interactiva
  (research) sobre background (pipeline). El drain de embeddings del fast path
  toma el lock por pasada —acotada por `IPA_EMBED_PASS_CHUNKS` (256), lo que
  además hace visible el progreso: antes la pasada 1 recorría el corpus entero y
  el watch reportaba `0 chunks embedded (draining)` durante horas— y **cede**
  (`should_yield`) mientras haya un waiter interactivo registrado. Advisory y
  best-effort: si la research no consigue el lock en `IPA_HEAVY_LOCK_WAIT`
  (300 s) sigue igual (nunca se deadlockea una respuesta al usuario).
- **Visibilidad**: `research_progress.json.heavy_wait` (`blocked_by`, `seconds`)
  y el indicador del dashboard muestran "esperando a <job>"; `run_research.py`
  lo limpia al terminar.
- Tests: `tests/test_heavy_lock.py` (12), embed acotado/budget en
  `tests/test_research_executor.py`, drain con lock y cesión en
  `tests/test_fast_path.py`. Evidencia: `PM-004`.

### Research: barrido temático por facetas (`sub_queries`)
- **`research_topic` acepta `sub_queries`** (máx 8, escritas por el agente):
  cada faceta corre su propia `search_web` y sus resultados entran al pool
  deduplicado por URL. Prefilter, juicio de snippets y juicio de contenido
  evalúan cada candidato contra la query que lo produjo — una faceta con
  vocabulario propio ya no se rechaza por no matchear la query principal.
  `max_urls` de la tool sube de 20 a 50 (sigue acotando ingesta exitosa;
  `max_seconds` ≤600 acota el scrape y `IPA_RESEARCH_INGEST_BUDGET` el resto).
- **Plumbing**: `sub_queries` viaja como argv[4] JSON a `run_research.py`
  (subprocess) y como `--sub-queries` en el CLI (`agent.py research`). Se
  registra en `research_progress.json`, `budget_used.sub_queries` y los
  argumentos del `ToolCall`.
- **Routing de tools (fix de interpretación)**: un pedido de "ingesta masiva
  sobre <tema>" llamaba `run_ingestion`, que es un barrido SIN filtro temático
  de todas las fuentes de `scrape_sites.yaml`. Las descripciones del registry
  ahora lo explicitan: `run_ingestion` = refresco de fuentes configuradas;
  `research_topic` = acumulación por tema (con `sub_queries` para barrido
  amplio).
- Tests: `test_sub_queries_widen_pool_and_judge_per_facet` y
  `test_sub_query_search_failure_does_not_fail_run` en
  `tests/test_research_executor.py`.

## Unreleased — 2026-09-19 (post-v0.1.2)

- **fix(ci)**: `FlagEmbedding`, `langchain-text-splitters` y `tiktoken`
  declarados en extras (`retrieval` + nuevo `chunkers`); `HF_HUB_OFFLINE`
  acotado al load de BGE-M3 (fugaba process-wide y rompía docling en CI).
- **Anti-narración de research (chat general)**: el 9B narraba la
  investigación sin emitir `[TOOL:research_topic]` (bug real: 5 claims,
  0 ejecuciones). El safety net ahora también detecta aceptación por cita
  truncada («…busqu…») u oferta del assistant en el turno previo, y los
  claims sin tool ejecutada se reemplazan por una admisión honesta en vez
  de grabar la mentira en el historial.
- **Derivación general→tutor**: pedidos pedagógicos explícitos ("haceme
  un roadmap de X", "quiero aprender Y") corren por el state machine del
  Tutor aunque lleguen con `role=general` — la investigación ahí es un
  contrato real con gate humano. `tutor_intent()` en `tutor_chat.py`.

## v0.1.2 — 2026-09-19

### Rendimiento y caches (ver `EXP-008`)
- **`num_gpu=30`** para Ollama (`IPA_OLLAMA_NUM_GPU`): el auto-fit dejaba la
  mitad del modelo en CPU. Medido en RTX 4050: 18/34 → 30/34 capas,
  **10.4 → 20 tok/s** de decode. 32 capas colapsa (sin VRAM para compute).
- **Layout del prompt para PT cache**: el contenido volátil (evidencia RAG,
  memoria, catálogo de tools) pasa del system prompt al tail del turno user.
  El runner pasó de **rechazar** el reuse (sim 0.41-0.47, prefill completo) a
  reusar **72-90%** del prefijo por turno (sim 0.74-0.90).
- **`num_keep=2048`** (`IPA_OLLAMA_NUM_KEEP`): el default de llama.cpp es 4, así
  que al llenarse el contexto el system prompt era lo primero en evaporarse.
- **`num_ctx` 8192 → 6144** (los prompts reales miden 2.3-4.1k).
- **`OLLAMA_KV_CACHE_TYPE=q8_0`** + `keep_alive=30m` + métricas por request en
  `outputs/web_dashboard/logs/llm_perf.jsonl` (incluye
  `prompt_eval_cached_count`, la métrica directa del PT cache).
- **Caches de aplicación**: embeddings de query (`IPA_EMBED_CACHE_SIZE`),
  rankings del reranker (`IPA_RERANK_CACHE_SIZE`), resultados de retrieval con
  TTL (`IPA_RETRIEVAL_CACHE_*`), tools read-only (`IPA_TOOL_CACHE_TTL`;
  `get_system_status` excluido por semántica "estado ahora") y respuestas del
  chat opt-in (`IPA_RESPONSE_CACHE=1`, match exacto por sesión).

### ExL3
- **Fix de cache (bug real)**: `cache_tokens = min(context_length,
  mtp_cache_tokens)` dejaba el cache en 4096 con contexto 6144 → todo prompt
  >4096 fallaba con salida vacía (`Job requires N pages`) o divagaba al cruzar
  el límite a mitad de generación. Ahora `max(...)`.
- **PT cache verificado**: el generator de ExLlamaV3 reusa prefijos por hash de
  páginas — medido TTFT 3.7s → 0.34s (10.9x) en la segunda generación.
- **Sweep de batch (ctx 2048, RTX 4050)**: el throughput agregado **satura a
  batch ≥3** (~83 tok/s = 714 tok en 8.6 s, limitado por ancho de banda de
  memoria) — el sweet spot es el batch más chico que satura: **3-4**.
  `create_star_provider` default 6 → 4.
- **MTP y batch**: +14% a batch 1 (interactivo), neutro a 2-4 y **catastrófico
  a batch 6** (17.5 vs 83.2 tok/s sin MTP — realineación del speculative
  decoding, consistente con arXiv 2510.22876 / 2310.18813). El colapso de
  batch 6 atribuido antes a "presión de páginas" era MTP. **Guard**: batch > 2
  desactiva MTP solo (aviso; `IPA_EXL3_FORCE_MTP=1` lo fuerza).
- **Batch en 6 GB**: batch 6 + contexto 6144 no entra (OOM al cargar); con
  contexto 2048 y MTP off, batch 6 rinde igual que batch 4.

### ExL3 activo en el pase Tier 2 profundo
- **`_t2_deep` carga ExL3** (`IPA_T2_ENGINE=exl3`, ctx 2048, batch 4, MTP off
  por el guard) con fallback a Ollama; `unload` + release del lock al terminar.
  Es el único punto del sistema que ya pagaba el costo de cargar un modelo
  desde frío → el switch se amortiza sobre toda la cola Tier 2.
- **Split por longitud de salida**: `cog_principles_llm` (~500 tok) se saltea en
  el pase ExL3; `deep_topify` (labels) y la nueva tarea **`review_batch`**
  (128 tok) corren batched.
- **`batch_llm.generate_many`**: helper que usa `generate_chat_batch` cuando el
  provider lo soporta (ExL3) y cae a serial con Ollama, aislando errores por
  lote. Medido en integración: 4 veredictos en 5.8 s (0.69 docs/s ≈ 88 tok/s
  agregado, 2.6x el serial).
- `IPA_IDLE_DEEP_ENRICHMENT` default 0 → **1** (el path pasa a ser el hogar del
  trabajo batch). Nuevas envs: `IPA_T2_ENGINE`, `IPA_T2_CTX`, `IPA_T2_BATCH`,
  `IPA_T2_REVIEW_LIMIT`, `IPA_EXL3_FORCE_MTP`.
- **Conmutación chat ↔ pase T2 (verificado en vivo)**: el pase profundo marca
  su provider `_t2_owned`; si el usuario manda un chat a mitad del pase,
  `get_deep_dive_provider()` lo descarga y monta el interactivo (prioridad al
  usuario; el pase aborta entre tasks vía `CHAT_BUSY`). Guard por identidad +
  `DEEP_DIVE_LOCK` en el `finally` del worker: solo descarga si la instancia
  sigue siendo la suya — no toca el provider del chat. `deep_done` separado de
  `level2_done` (el warmup cargaba Ollama en el boot y el motor del pase nunca
  llegaba a cargarse). Al terminar, re-warm del interactivo si sigue idle.
  Medido: pase con ExL3 → chat → respuesta en ~29 s (unload ~2 s + reload
  GGUF ~10 s + generación); `vram.lock` liberado; sin OOM ni procesos huérfanos.
- **Tarea `enrich_chunks` (Tier 2, prio 25)**: recupera el trabajo del job
  `enrichment` de la cadena deprecada del Orchestrator (era ExL3 4B cargando
  su propio modelo). Nuevo módulo `ipa.agentic.chunk_enrichment` — summary +
  3 queries sintéticas por chunk + re-embed en LanceDB, sobre el **9B ya
  cargado del pase** (en 6 GB no caben dos modelos). Checkpoints durables:
  `[Summary]` en el texto + `enrichment.embedding_status` (pending→complete
  solo cuando LanceDB acepta) recuperan corridas cortadas; `canonical_text`
  preserva el original. El worker `run_enrichment_exl3.py` queda como runner
  standalone reusando el módulo. `min_chars` por defecto 400 (el 800 original
  no seleccionaba nada: el corpus actual es uniforme ~512 chars; con 400 la
  densidad discrimina ~3% = 4.5k chunks, medido 2026-09). Envs:
  `IPA_T2_ENRICH_LIMIT` (60/pase), `IPA_T2_ENRICH_MIN_CHARS` (400).
- Tests: `tests/test_batch_llm.py` (8 casos) + smoke de integración con el
  modelo real (`scripts/operations/_t2_review_batch_check.py`).

### Orchestrator deprecado
- La cadena de consola (scraper → fast_path → lancedb → hammer → enrichment)
  queda **deprecada**: warning al arrancar, `-StartOrchestrator` avisa, docs
  actualizadas. Los jobs se lanzan desde el dashboard; el trabajo LLM en
  background corre por el idle scheduler (Tiers 1/2). El job `enrichment`
  (ExL3 4B) queda sin trigger activo hasta redefinir su reemplazo.

### Tutor: loop de detección de tema arreglado
- **`awaiting_topic`**: tras "¿Qué querés aprender?" el próximo mensaje que
  no sea comando/pregunta/relleno se toma literalmente como tema. Antes una
  respuesta suelta ("De IA Engineer Senior a CTO en etapas tempranas") no
  matcheaba la regex imperativa → el Tutor repreguntaba en loop.
- **`_detect_topic` ampliado**: "hagamos/armemos un roadmap (de/sobre X)",
  "quiero pasar de X a Y", "transición de X a Y"; recorta colas de
  instrucción ("armame un roadmap de 6 fases") y conteos (unidades/URLs)
  para que nunca queden como tema.
- **Contexto reciente**: "hagamos un roadmap" sin tema busca en los últimos
  episodios del usuario (típicamente la cita a la propuesta anterior).
- **"límite de N URLs"** → `ResearchBudget.max_urls` (persiste entre
  mensajes, como `requested_units`); dominios por defecto ampliados a
  fuentes generales de calidad — el default tech-only del runtime no
  servía para temas como liderazgo/carrera.
- **Cita sola = aceptación**: un mensaje que es solo `[cita: «…»]` /
  `[respondiendo a: «…»]` sobre la propuesta del agente aprueba el gate
  (roadmap e investigación) en vez de repreguntar — regla también
  explícita en `agent_identity.yaml`.

### Tier 2: fixes de convivencia ExL3↔Ollama
- **`deep_topify` se saltea en providers sin batch** (`server.py`): el
  reporter llamaba métodos solo-ExL3 (`reset_generator`,
  `generate_chat_batch`, `.ok`/`.text`) — en Ollama quemaba ~250 tok por
  cluster y los descartaba (14.5 min de GPU en 38 clusters, labels
  perdidos). El pase profundo queda para ExL3.
- **`_unload_ollama_models()` espera la liberación real**: `keep_alive=0`
  es asíncrono y ExL3 medía la VRAM libre al instante → `Insufficient VRAM
  in split`. Ahora pollea `/api/ps` hasta vacío (máx 20s) antes de cargar.

### Operaciones
- **Lock de VRAM ExL3↔Ollama** (`outputs/agent/vram.lock`): ExL3 descarga los
  modelos de Ollama y toma el lock; el chat responde "GPU ocupada por exl3" en
  vez de morir con OOM (medido: `rep_pen.cu` OOM con los dos cargados). Un lock
  de un proceso muerto se roba por TTL.
- **Watchdog**: grace de warmup (`IPA_WATCHDOG_GRACE_SECONDS`, 240s) — sin ella
  reiniciaba el dashboard en loop mientras cargaba BGE-M3 + el modelo (la
  lentitud reportada); claim atómico del slot (`O_EXCL`) — el escaneo de
  procesos fallaba porque `python.exe` del venv es un trampolín; barrido de
  `llama-server.exe` huérfanos (retenían ~4.4 GB de VRAM).
- **SearXNG gestionado** (launcher + watchdog) y dependencias externas
  documentadas en `docs/USAGE.md`.

### Tests
- **Hermeticidad de GPU**: `IPA_EMBED_DEVICE=cpu` en `tests/conftest.py` —
  cargar BGE-M3 en una GPU ocupada por el modelo del chat agotaba la VRAM y
  congelaba la UI de Windows (383 MiB libres). Los caches de proceso se limpian
  entre tests (un test recibía un hit viejo de `list_promotions`).
  Suite: **962 passed, 1 skipped** en ~6:00.

## Unreleased — 2026-09-18

### Retrieval
- **Rerank stage-2 ON por defecto** (`IPA_RERANK=0` para desactivar; acepta
  `false`/`no`/`off`). Eval E10-rerank: recall@1 0.465→0.670 (+20.5pp),
  MRR +0.141, nDCG@10 +0.117; ~+0.65s GPU / ~+0.7s CPU. Ver `EXP-007`.
- **Gate de VRAM corregido**: medía con `torch.cuda.mem_get_info()`, que en
  Windows/WDDM sobreestima la VRAM libre (reportó ~5 GB con 1.6 GB físicos
  libres) → el reranker cargaba en GPU con el LLM cargado. Ahora usa
  `nvidia-smi` (`physical_free_vram_mb()`) con fallback.
- `run_retrieval_eval.py` respeta `IPA_RERANK_DEVICE`.
- **Reranker fijado a CPU** en esta máquina: `IPA_RERANK_DEVICE=cpu` en el
  launcher (`start_ipa_dashboard.ps1`) y como env de usuario. `auto` podía
  cargar el singleton en CUDA mientras el LLM estaba descargado y dejarlo
  residente compitiendo por VRAM al volver el chat; el costo CPU medido es
  ~+0.7 s/query (EXP-007), marginal frente al riesgo de OOM en 6 GB (EXP-008).

### Tutor
- **Scaffold absorbente** (`_shape_units`): propuestas imperfectas del LLM ya no
  producen dead-ends — descarta concept_ids desconocidos, deduplica, completa al
  mínimo (3) y trunca a 7; el prompt dejó de invitar a reutilizar concept_ids
  (contradecía el contrato). Fallos residuales → mensaje amigable (la excepción
  va al log) + nota de transparencia si el corpus no menciona el tema.
- **Foco de roadmap cross-sesión**: click en la card (o activar un roadmap)
  apunta la sesión y persiste el foco (`tutor_focus`); sesiones nuevas/idle lo
  adoptan; rechazar lo limpia. Chip "📍 tema · unidad N/M" en el chat.
- **Tag de roadmap en resúmenes de sesión** (`[roadmap:<id> · tema · unidad N/M]`)
  → recuperable cross-sesión vía `recall_memory`.
- **Lecciones ~2x más largas**: la identidad base escopa "1-5 oraciones" al chat
  general y da excepción al rol tutor; `TUTOR_POLICY` pide explicaciones ricas;
  `max_new_tokens` de lección 768→1536. Medido: ~90→184 palabras.
- Gate del chat: aprobar un roadmap mostraba "Rechazado" (contrato
  frontend/backend: `decide_roadmap` devuelve `active`, no `approved`).
- Diagnóstico sin leak de policy ("Comenzá con un diagnóstico…" ya no se muestra
  al alumno) y sin doble punto; encabezado único en el debate.

### Agente / UI
- **Perilla "Idle T1/T2"** en el sidebar: apaga el enriquecimiento idle completo
  (gate en `_idle()`; aborta pasadas Tier 2 en vuelo). Persistida en
  `idle_enabled.json`; endpoints `GET /api/idle/status`, `POST /api/idle/toggle`.
- **Quote-reply en el chat**: seleccionar texto inserta un puntero compacto
  `[cita: «primeras 3 palabras…»]` en el input (no copia el pasaje completo):
  marca qué sección del chat mirar. El snippet es literal, así que el agente
  lo resuelve con `recall_conversation(query=…)` (skill `responder_a_cita`).
- MCP server: corregido el import (insertaba `src/ipa` en `sys.path` y el
  paquete local `ipa/mcp` sombreaba el SDK `mcp` — el módulo no importaba) y
  `RerankCandidate(id=…)` (campo real: `chunk_id`; el TypeError se tragaba y el
  rerank no se aplicaba). Suite MCP nueva. Docs de tools sincronizadas.

### Research: dedup real + URLs explícitas
- **Dedup para TODOS los llamados** (antes solo safety-net): si una query
  igual o muy parecida ya se investigó dentro de la ventana (10 min), la tool
  no relanza — devuelve el material ya ingerido y pide `search_corpus` /
  `compile_report`. El match exacto no alcanzaba: el modelo reformula la query
  entre turnos ("IA big techs" → "IA tres grandes tecnológicas"). Nuevo
  `find_recent_research` (igualdad normalizada **o** contención de tokens
  ≥ 0.6, stopwords fuera). `force=true` fuerza una corrida nueva.
- **URLs explícitas = seeds**: una URL en la query (o pegada por el usuario en
  su mensaje — la tool la reinyecta, porque el modelo la descarta al
  parafrasear) se scrapea **directo**, salteando el snippet stage (no hay
  snippet que juzgar) pero pasando por scrape → calidad → juicio → ingesta.
  El remanente textual va a la búsqueda complementaria; query solo-URL deriva
  la búsqueda del slug. Con seeds, un fallo del backend de búsqueda ya no
  invalida la corrida (se registra el error y se procesan las fuentes).
- **Identidad**: principio explícito de no relanzar una investigación ya hecha
  ante un "dame lo que investigaste" (skill `investigar_web` actualizado).

### Infra / calidad
- Deprecaciones de LanceDB resueltas (`list_tables()`, `create_index(config=FTS())`);
  warnings de la suite 78 → 18 (los restantes son de terceros).
- `tests/conftest.py`: `IPA_RERANK=0` autouse (suite hermética, sin cargar el
  cross-encoder por el default).
- 881 tests, 1 skipped (red).

## v0.1.0 — 2026-09-16

Primera versión estable consolidada: agente personal local-first, contract-first, funcional de punta a punta.

### Núcleo
- **Hybrid RAG contract-first**: `DocumentStore` canónico; Tantivy/BM25, LanceDB, embeddings, clusters, reportes y memoria como derivados rebuildables y auditables.
- Agent core compartido (DEC-002): CLI y dashboard comparten identidad, sesiones y memoria episódica (`outputs/agent/agent.db`).
- Chat con protocolo de tools acotado (máx. 3 rondas por turno), progressive tool unlocking y safety nets (anti-repetición, auto-research con dedup).

### Tutor
- State machine determinística (`tutor_runtime.py`): diagnóstico → roadmap (LLM propone, humano aprueba) → lección → assessment (JSON estructurado con abstención) → mastery persistido.
- Debate de roadmaps: feedback del alumno → re-propuesta (v+1, supersedes) con el gate humano intacto.
- Research del tutor: 15 fuentes / 300s (chat general: default 5, tope 20).
- Recuperación ante JSON malformado del LLM: retry correctivo + fallback determinístico.
- Gates re-montables en la UI (`mountPendingTutorGates`) tras cada re-render canónico.

### Adquisición
- Scraper multi-engine (requests / Playwright / auto) con OCR, patrones determinísticos por URL y ventanas por fuente.
- FastPath: parse → chunk → BM25 + LanceDB con trazabilidad (E11).
- Reporter como tool del agente (promoción desacoplada, policy por provenance).

### Idle / background
- `idle_scheduler.py`: registro de tareas con tier, prioridad, recursos (locks nombrados) y cooldowns. Tier 1 paralelo (pool de 3), Tier 2 serial preemptible.
- Consolidación automática de sesiones (resumen + hechos → cola de aprobación), con cierre de sesiones huérfanas.
- Review queue de docs rechazados (re-lectura LLM en idle, resumible).
- Enrichment L1 determinístico (topificación, curación heurística, continuidad) y Tier 2 LLM (re-etiquetado, clasificación de grises, principios).

### Infra
- Dashboard `ThreadingHTTPServer` + SSE; stores SQLite thread-bound (sin `check_same_thread=False`).
- Contrato de providers normalizado (`ipa/agent/llm_text.py`): Ollama (str) y ExL3 (`GenerationResult`).
- Provider estrella: Qwen3.5-9B EXL3 3.0bpw + MTP (extensión nativa compilada, sm_89).

### Calidad
- 854 tests, 1 skipped (red). Suite completa en verde.
- Limpieza: propuestas de test fuera de la cola de aprobaciones, scripts temporales eliminados.
