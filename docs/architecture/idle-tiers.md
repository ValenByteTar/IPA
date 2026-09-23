# Idle tiers — orchestración del corpus en tres niveles

IPA corre todo el trabajo pesado del corpus en tres tiers con una única
regla de exclusión: **nada de Tier 1/Tier 2 mientras Tier 0 está activo,
y la cuenta de idle empieza solo cuando Tier 0 se libera**
(DEC-010). Este documento es la referencia consolidada del modelo; el
detalle de implementación de cada pieza vive en los archivos citados.

```
┌─────────────────────────────────────────────────────────────────┐
│ Tier 0 — INGESTA (excluyente, prioridad absoluta)               │
│   parse → chunk → store → BM25 → señales → drain embeddings     │
│   lease: outputs/agent/tier0.lock  ·  interactiva: heavy.lock   │
└─────────────────────────────────────────────────────────────────┘
              ↓ lease liberado → cuenta de idle arranca
┌─────────────────────────────────────────────────────────────────┐
│ _idle() — gate compartido (ver tabla de condiciones)            │
│   + claim_job("idle_scheduler") + ENRICHMENT_LOCK por ciclo     │
└─────────────────────────────────────────────────────────────────┘
        ↓                                    ↓
┌──────────────────────┐          ┌──────────────────────────────┐
│ Tier 1 — determinista│          │ Tier 2 — LLM (serial,        │
│ cada 60 s si idle    │          │ preemptible entre items)     │
│ pool 3 threads,      │          │ pase profundo (ExL3 batch) o │
│ locks por recurso    │          │ modelo ya cargado del chat   │
└──────────────────────┘          └──────────────────────────────┘
```

## Tier 0 — ingesta

Todo lo que escribe documentos/chunks en un corpus. Dos modalidades:

| Modalidad | Entry point | Lease | Notas |
|---|---|---|---|
| Batch | `run_fast_path.py` (inicial + watch + drain final) | `tier0.lock` cross-process: `pid\|owner\|ts`, heartbeat 15 s, TTL 300 s, robo por `pid_alive` | Una segunda instancia ve el holder vivo y sale (ingesta idempotente). El heartbeat es un thread daemon — cubre stalls largos del loop principal |
| Interactiva | `research_ingest`, `ingest_reviewed_doc` | `heavy.lock` | Mismo criterio semántico verificado por `_idle()` vía `heavy_lock.holder()` |

### Señales que Tier 0 persiste en la misma corrida (PAT-008)

En vez de dejar que T1 re-derive todo por ciclo, cada vía de ingesta
(`fast_path` inicial y watch, `research_executor`, `ingest_reviewed_doc`)
escribe en `record_ingest_metadata()`:

- **`document_metadata`**: `normalized_hash` (formato `sha256:` de
  `reporter_curation` — única fuente, el bug del formato truncado hacía
  que el check de re-descarga por URL nunca disparara), `title`,
  `published_at`, `char_count`, `extra_json`.
- **`document_sources`**: `configured_scrape` con url/domain/quality/
  `published_at` real del `scrape_report.json` o de las líneas
  `Source:`/`Date:` del texto — solo para artifacts bajo `web/**`
  (un archivo manual en `Landing/` raíz no es scrape configurado).
- **`extra.duplicate_of_main`**: si el `normalized_hash` ya existe en
  main → hecho registrado; la decisión DUPLICATE la toma T1.
- **`extra.novelty_hint`** (post-drain, en `compute_novelty_hints`):
  `max_cosine` vs main + `nearest_doc_id` + token de snapshot dual
  (`main_doc_count` + `main_latest_stored_at` — el count detecta adds en
  el mismo segundo, el ts detecta tombstone+add a igual count).
- **`dirty:<corpus>`** en `topic_clusters.meta`: seteado por todo writer
  (fast_path, research ingest, review ingest, promotion executor).

Caso terminal nuevo: parse OK con 0 chunks → artifact `no_text` en
landing.db, sin documento en el store; el sweep lo elimina.

## El gate — `_idle()`

Devuelve no-idle si **cualquiera** es cierto (cada chequeo es aislado —
quitar uno no rompe el resto):

| Condición | Cubre |
|---|---|
| Toggle `Idle T1/T2` OFF (`idle_enabled.json`) | kill-switch del sidebar |
| `tier0.active()` | lease Tier 0 vivo, incluye watchers huérfanos de un dashboard anterior (cross-process por PID-liveness, no por árbol de procesos) |
| `heavy_lock.holder()` | ingesta interactiva en curso |
| `embedding_maintenance.job_active()` | drain/lote GPU de índices ajeno |
| `CHAT_BUSY` | usuario conversando |
| Algún `proc` vivo en `JOBS` | hijos del dashboard actual |
| pipeline/reporter `status == "running"` | jobs registrados en estado |

Cada iteración no-idle resetea `LAST_ACTIVITY` → el cronómetro arranca de
cero cuando Tier 0 suelta el lease. Por ciclo se toman además
`claim_job("idle_scheduler")` (serializa con el drain de embeddings) y
`ENRICHMENT_LOCK` (una sola instancia de scheduler por store compartido).

## Tier 1 — enriquecimiento determinista (sin LLM)

Pool de 3 threads; cada tarea declara recursos (locks nombrados tomados
en orden alfabético → sin deadlocks) y cooldown propio. Prioridad menor =
antes.

| # | Task | Recursos | Cooldown | Qué hace |
|---|---|---|---|---|
| 10 | `hygiene_sessions` | `agent_db` | 120 s | Cierra sesiones `active` huérfanas de un restart |
| 20 | `consolidate_sessions` | `agent_db` + `llm` | 90 s | Resume sesiones idle; usa el LLM solo si ya está cargado |
| 30 | `topify_main` | `cluster_store` + `embeddings` + `corpus_main` | 300 s | Provenance backfill (repair path) → clustering → curación heurística → continuidad |
| 31 | `topify_reporter` | `cluster_store` + `embeddings` + `corpus_reporter` | 300 s | Ídem sobre el corpus staging del reporter |
| 32 | `topify_research_staging` | `cluster_store` + `embeddings` + `corpus_research` | 300 s | Ídem sobre `outputs/agent/research_staging/` — el staging propio de la research (DEC-003): cura lo que el juez aceptó y encola lo que supera la política de promoción |
| 40 | `promotion_queue` | `cluster_store` + todos los corpus + `embeddings` | 300 s | Cola de promoción (con preflight de cobertura vectorial y purge reintentable) + sweep de Landing. Agrupa por `source_corpus` de cada entry — no necesita conocer los staging por adelantado |
| 50–53 | `cog_user_model`, `cog_skills`, `cog_principles`, `cog_agenda` | 1 recurso c/u (paralelos) | 300 s | Cognición determinista → propuestas `pending` (gate humano) |
| 60 | `index_audit` | `cluster_store` + `corpus_main` | `IPA_AUDIT_LOGICAL_SECONDS` (900 s) | Ver sección propia abajo |

### Incrementalidad de topify (el cambio grande)

- **Gate "corpus changed"**: sin `dirty:<corpus>` y sin drift de
  conteo/cobertura/provenance → early-exit, cero scan.
- **Filter-first**: set-diffs sobre ids/metadata baratos antes de fetchar
  un solo texto; `build_document_dicts(doc_ids=…)` y
  `LanceDBIndex.document_embeddings(doc_ids)` acotan los reads.
- **Dup short-circuit**: `duplicate_of_main` → decisión DUPLICATE directa.
- **Novelty hints**: si el hint sigue válido (count + stored_at del
  snapshot coinciden) no se carga la matriz histórica; si está stale se
  re-verifica solo contra los docs agregados desde el snapshot
  (O(nuevos), self-heal: el hint refrescado se persiste). El segundo
  factor de DEC-003 (Jaccard ≥0.85) fetchea solo el texto del doc más
  cercano.
- **Self-match excluido**: cuando el corpus curado ES main, los docs en
  curación se excluyen de `main_url_hashes`, históricos y refresh — sin
  esto un doc se marcaba DUPLICATE de sí mismo.
- **Backfill acotado**: `IPA_T1_META_BACKFILL_LIMIT` (1000/ciclo) drena
  `document_metadata` de corpus legacy.

## `index_audit` — salud del corpus (T1, read-only)

`ipa/agentic/index_audit.py`. Detecta automático, repara nunca — las
acciones quedan en scripts ops con dry-run.

- **Capa lógica** (cada corrida, consume señales Tier 0): docs vacíos
  vivos (`char_count=0`; NULL = no computado), `duplicate_of_main`
  fugados (flag sin decisión DUPLICATE), `normalized_hash` repetido entre
  vivos, hints stale, cola de backfill de metadata, `configured_scrape`
  sin URL.
- **Capa física** (forzada cada `IPA_AUDIT_PHYSICAL_HOURS`=6 h — el drift
  por kills/locks no setea dirty flags, así que no puede ser change-
  gated): sets de `chunk_id` entre store ↔ BM25 meta ↔ FTS ↔ LanceDB
  (faltantes, huérfanos, `chunk_id` duplicados) y spam chunks
  (`content_hash` en ≥3 docs = boilerplate same-site; ≥2 dominios =
  sindicación, reportado aparte). Comparación por chunk_id, nunca por
  texto: BM25/LanceDB guardan la representación `enriched_text()`.
- Salida mergeada en `outputs/agent/index_health.json`
  (`status`: ok/warn/fail) → tarjeta "Salud de índices" del dashboard.

## Tier 2 — enriquecimiento con LLM (serial, preemptible)

Dos disparadores con motores distintos (split de EXP-008):

| Disparador | Condición | Motor |
|---|---|---|
| Pase profundo | idle ≥ `IPA_IDLE_DEEP_THRESHOLD_MINUTES` (30) y `IPA_IDLE_DEEP_ENRICHMENT=1` | ExL3 batch propio (`IPA_T2_ENGINE`, ctx `IPA_T2_CTX`=2048, batch `IPA_T2_BATCH`=4); marcado `_t2_owned` — un chat a mitad lo descarga |
| Modelo ya cargado | idle ≥ `IPA_IDLE_LLM_LOADED_THRESHOLD_MINUTES` (5) y `IPA_IDLE_LLM_LOADED_ENRICHMENT`≠0 | Reusa el provider del chat sin cargar nada |

Corre una vez por ventana de idle (`level2_done`/`deep_done` se resetean
al volver a no-idle) y aborta entre items vía `should_abort → _idle()`.

| # | Task | Qué hace |
|---|---|---|
| 10/11/12 | `deep_topify_main/reporter/research_staging` | Labels ricos + grouping. Solo con provider batch (`supports_batch`) — en Ollama serial acapararía el chat horas |
| 15 | `review_batch` | Veredictos de la cola de research review, batched (128 tok/ítem — el caso ideal del motor batch; `IPA_T2_REVIEW_LIMIT`=12) |
| 20 | `cog_principles_llm` | Principios (~500 tok) — se saltea en ExL3 batch (fuera del rango cómodo del motor) |
| 25 | `enrich_chunks` | Summary + queries por chunk (`IPA_T2_ENRICH_LIMIT`=60, `IPA_T2_ENRICH_MIN_CHARS`=400). `chunks.text` canónico — la representación vive en `metadata.enrichment.enriched_text`; re-indexa BM25 **y** LanceDB con ella (win medido EXP-001: +14.3% recall@10) |

### Zona gris (dentro de `deep_topify_*`)

`agent_research` con decisión `reporter_only` y `promotion_score` en
`[IPA_GRAY_LO=0.5, IPA_GRAY_HI=0.70)` recibe segunda opinión batched
(`classify_many`, ≤`IPA_GRAY_LIMIT`=24/pase): la decisión se actualiza,
se re-evalúa `evaluate_promotion` y se encola si supera el umbral.
Marca `aux_progress.gray_reviewed` — tabla no ordenada aparte, porque
`enrichment_progress` es la progresión `clustered < curated` de una fila
por doc y pisarla rompe el checkpoint. Reemplaza el bloque de "curación
LLM" que era dead code (L1 marcaba todo como curated).

## Drain de embeddings (transversal)

No es una tarea del scheduler: corre dentro de Tier 0 (`_embed_drain_loop`)
o standalone (`run_embed_drain.py`). Con backlog ≥ `IPA_EMBED_GPU_MIN_BACKLOG`
(512) toma el lote GPU exclusivo (maintenance mode: descarga Ollama,
chat pausado, banner en el dashboard). Embede `enriched_text()` — la
misma representación en cualquier camino de indexación.

La research (`_embed_new_chunks`) usa la misma escalada: backlog ≥ umbral →
`claim_job("research_embed")` + `_start_bulk_gpu` (espera de VRAM acotada al
budget de ingesta, máx 120 s) → `_finish_bulk_gpu` restaura el chat. Si el
claim o la VRAM fallan, cae al embed inline por CPU sin bloquear la
respuesta. Tras el lote, `release_gpu()` deja el adapter en CPU lazy —
`close()` solo libera VRAM pero deja `device="cuda"` y un embed posterior
recargaría BGE-M3 sobre el chat restaurado.

## Modelo de fallos

| Fallo | Recuperación |
|---|---|
| Kill de Tier 0 | `tier0.lock` queda stale → TTL 300 s o robo por `pid_alive`; la ingesta es idempotente |
| Crash de T1 a mitad | `dirty:<corpus>` queda seteado (se limpia solo tras corrida exitosa) → próximo ciclo reintenta; checkpoints `enrichment_progress`/`aux_progress` resumen |
| Lock de SQLite en purga de promoción | `_purge_source_bm25` reintenta con `BEGIN IMMEDIATE` + busy timeout; si agota, `incomplete_steps` → batch deferred, cola queda pending |
| Kill durante lote GPU | `embedding_maintenance` escribe por batch; reanuda donde quedó (chunk_ids ya vectorizados se precargan) |
| Desync índices (clase PM-003) | `index_audit` capa física lo detecta en ≤6 h y lo reporta en `index_health.json` |

## Referencia de env vars

| Var | Default | Efecto |
|---|---|---|
| `IPA_TIER0_LOCK` / `_TTL` / `_HEARTBEAT` | `outputs/agent/tier0.lock` / 300 / 15 | Lease Tier 0 |
| `IPA_IDLE_DEEP_ENRICHMENT` / `_THRESHOLD_MINUTES` | 1 / 30 | Pase profundo T2 |
| `IPA_IDLE_LLM_LOADED_ENRICHMENT` / `_THRESHOLD_MINUTES` | 1 / 5 | T2 sobre modelo cargado |
| `IPA_T2_ENGINE` / `_CTX` / `_BATCH` | exl3 / 2048 / 4 | Motor del pase profundo |
| `IPA_GRAY_LO` / `_HI` / `_LIMIT` | 0.5 / 0.70 / 24 | Banda de la zona gris |
| `IPA_T2_REVIEW_LIMIT` / `IPA_T2_ENRICH_LIMIT` / `_MIN_CHARS` | 12 / 60 / 400 | Tamaños de pase |
| `IPA_T1_META_BACKFILL_LIMIT` | 1000 | Backfill de `document_metadata` por ciclo |
| `IPA_AUDIT_LOGICAL_SECONDS` / `IPA_AUDIT_PHYSICAL_HOURS` | 900 / 6 | Cadencias del audit |
| `IPA_AUDIT_SPAM_MIN_DOCS` / `IPA_AUDIT_DUP_SAMPLE` | 3 / 10 | Umbrales del audit |
| `IPA_EMBED_GPU_MIN_BACKLOG` | 512 | Umbral del lote GPU del drain |
| `IPA_PROMOTION_REQUIRE_VECTORS` | 1 | Preflight de cobertura vectorial (0 = opt-out de emergencia) |

## Observabilidad

- `outputs/web_dashboard/logs/idle_enrichment.log` —
  `[idle-sched T1/T2] <task>: {counts} (duration)`; líneas sin novedad no
  se loguean.
- `outputs/agent/index_health.json` → tarjeta "Salud de índices".
- `outputs/agent/tier0.lock`, `outputs/agent/heavy.lock`,
  `outputs/agent/vram.lock` — leases activos (archivos de runtime,
  existen solo mientras el holder corre).

## Referencias

- DEC-010 (tres tiers + exclusión de Tier 0), PAT-008 (señales en
  escritura), DEC-003 (gate de dos factores), PM-003/PM-004 (clases de
  desync y starvation que este modelo previene), EXP-001/EXP-008
  (evidencia lexical+summary y split de motores).
- `docs/plans/tier0-signals-idle-optimization.md` — plan y self-review.
- Código: `ipa/agentic/{tier0,idle_scheduler,idle_enrichment,index_audit,
  embedding_maintenance,chunk_enrichment,promotion_executor}.py`,
  `ipa/ingestion/{ingest_metadata,fast_path_cli,landing_sweep}.py`,
  `ipa/dashboard/server.py` (`_idle()`, tasks, drivers T2).
