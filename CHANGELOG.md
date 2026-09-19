# CHANGELOG — IPA

Formato: [versión] — fecha. Estilo Keep a Changelog (resumido).

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
