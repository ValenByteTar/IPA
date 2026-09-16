# AGENTS.md — RES-023 Lab

## Environment

- Python 3.12 required (`>=3.12,<3.13`). On this machine: `py -3.12` (3.12.8).
- venv location: `.venv\` (not committed; created per-machine).
- Core deps: PyYAML, rich, pytest, pytest-cov, PyMuPDF, rank-bm25, numpy.
- Stage 2 deps: tantivy (E6), lancedb + sqlite-vec + sentence-transformers (E7).
- Stage 3 deps: docling (E3 parser), unstructured[pdf] (E3 parser),
  langchain-text-splitters + tiktoken (E5 chunkers),
  trafilatura + easyocr (E4 scraper/OCR),
  playwright (E4 JS-rendered sites).
- Tutor Agent deps: torch (CUDA 12.6), exllamav3 1.4.4, transformers,
  flash-linear-attention. The exllamav3_ext native extension is compiled
  locally in `exllamav3-dev/` (sm_89 / RTX 4050). The provider
  (`src/ipa/providers/exl3_provider.py`) auto-discovers it on import.
- Dev-only deps (installed in venv, not in requirements.txt): jsonschema.

## Common commands

```powershell
# Setup
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install jsonschema  # dev-only

# Environment check
.venv\Scripts\python.exe scripts\operations\check_environment.py

# Build landing manifest (append-safe, timestamped by default)
.venv\Scripts\python.exe scripts\cli\build_landing_manifest.py --input data/sample/input

# Validate a manifest (structure + integrity against source files)
.venv\Scripts\python.exe scripts\validation\validate_contracts.py outputs\manifests\processing.<timestamp>.jsonl --integrity

# Validate an experiment report (jsonschema + integrity)
.venv\Scripts\python.exe scripts\validation\validate_experiment_report.py outputs\experiments/E0/report.json

# Run the fast path pipeline using the project Landing directory
.venv\Scripts\python.exe scripts\cli\run_fast_path.py --output outputs/experiments/E1 --query "ingestion pipeline"

# Run against reproducible synthetic fixtures instead
.venv\Scripts\python.exe scripts\cli\run_fast_path.py --input data/sample/input --output outputs/experiments/E1

# Run index benchmark (E6 lexical + E7 vector)
.venv\Scripts\python.exe scripts\benchmarks\run_index_benchmark.py --store outputs/experiments/E1-corpus/document_store.db --output outputs/experiments/E6 --mode both

# Run only lexical benchmark (FTS5 vs Tantivy)
.venv\Scripts\python.exe scripts\benchmarks\run_index_benchmark.py --store outputs/experiments/E1-corpus/document_store.db --output outputs/experiments/E6 --mode lexical

# Run only vector benchmark (LanceDB vs sqlite-vec)
.venv\Scripts\python.exe scripts\benchmarks\run_index_benchmark.py --store outputs/experiments/E1-corpus/document_store.db --output outputs/experiments/E7 --mode vector

# Run parser benchmark (E3: PyMuPDF vs Docling vs Unstructured)
.venv\Scripts\python.exe scripts\benchmarks\run_parser_benchmark.py --pdfs Landing --output outputs/experiments/E3 --limit 20 --mode parsers

# Run chunker benchmark (E5: fixed-window vs recursive vs token vs semantic)
.venv\Scripts\python.exe scripts\benchmarks\run_parser_benchmark.py --store outputs/experiments/E1-corpus/document_store.db --output outputs/experiments/E5 --limit-docs 50 --mime-type application/pdf --mode chunkers

# Run retrieval evaluation (E10: Tantivy vs LanceDB vs hybrid)
.venv\Scripts\python.exe scripts\benchmarks\run_retrieval_eval.py --store outputs/experiments/E1-corpus/document_store.db --tantivy outputs/experiments/E6-full/tantivy --lancedb outputs/experiments/E6-full/vector/lancedb --output outputs/experiments/E10 --n-queries 200

# Run fast path with E11 traceability enabled
.venv\Scripts\python.exe scripts\cli\run_fast_path.py --input Landing --output outputs/experiments/E11-corpus --trace-db outputs/experiments/E11-corpus/trace.db

# Query trace log (E11 observability)
.venv\Scripts\python.exe scripts\operations\run_trace_query.py --trace-db outputs/experiments/E11-corpus/trace.db --summary
.venv\Scripts\python.exe scripts\operations\run_trace_query.py --trace-db outputs/experiments/E11-corpus/trace.db --artifact sha256:abc123
.venv\Scripts\python.exe scripts\operations\run_trace_query.py --trace-db outputs/experiments/E11-corpus/trace.db --failed

# Run web scraper (E4 OCR + intelligence gathering)
# Scrape all sites from YAML config (deterministic: url_pattern + exclude_paths)
.venv\Scripts\python.exe scripts\cli\run_web_scrape.py --config configs/scrape_sites.yaml --output Landing/web

# Scrape a single URL with explicit pattern (deterministic)
.venv\Scripts\python.exe scripts\cli\run_web_scrape.py --url https://developer.nvidia.com/blog --url-pattern "^/blog/[^/]+/$" --exclude "/blog/category/" "/blog/tag/" "/blog/recent-posts/" --days-back 7

# Scrape a single URL with CSS selector (deterministic)
.venv\Scripts\python.exe scripts\cli\run_web_scrape.py --url https://thehackernews.com/ --selector "article a[href]" --days-back 2

# Scrape a single URL with heuristics (non-deterministic, for unknown sites)
.venv\Scripts\python.exe scripts\cli\run_web_scrape.py --url https://example.com/news --days-back 2 --no-ocr

# Scrape a JS-rendered site with Playwright (e.g. Meta AI)
.venv\Scripts\python.exe scripts\cli\run_web_scrape.py --url https://ai.meta.com/research/ --engine playwright --url-pattern "^/blog/[^/]+/?$" --exclude "/static-resource/" --allowed-domains "research.meta.ai" --days-back 90 --no-ocr

# Auto engine: try requests first, fall back to Playwright if no links found
.venv\Scripts\python.exe scripts\cli\run_web_scrape.py --config configs/scrape_sites.yaml --engine auto --output Landing/web

# Reporter - controlled claim/citation evaluation (benchmark)
.venv\Scripts\python.exe scriptsenchmarks\evaluate_reporter.py

# Reporter - DEPRECATED as standalone pipeline. Now an agent-invoked tool.
# The agent calls compile_report with a set of document_ids to produce a
# fine-grained report. The pipeline no longer runs the Reporter automatically.
# scripts/cli/run_reporter.py was removed (was deprecated).
# Reporter deep dive and review CLIs still work on existing report outputs:
.venv\Scripts\python.exe scripts\cli
eporter_deep_dive.py --corpus outputs
eporter\default6-08\corpus --query "consulta"
.venv\Scripts\python.exe scripts\cli
eporter_review.py --db outputs
eporter\default6-08
eporter.db

# Agent core (Fase 2) — rol Tutor con scope de estado propio
# Loop: diagnóstico (determinístico) → roadmap (LLM propone + humano aprueba) →
#       lección (policy + mastery context) → assessment (LLM JSON + abstención) →
#       mastery update (UserTopicRecord + UserEvidence)
# Estado: outputs/agent/tutor.db (TutorStore: user_topic_records + user_evidence append-only + roadmaps)
# Runtime: ipa/tutor/tutor_runtime.py (TutorSession, TutorStore, DiagnosisResult)
# Roadmap gate: propose_roadmap → approve/reject (humano) → activate (solo approved)
#   Debate: con un roadmap proposed, cualquier mensaje que no sea aprobar/rechazar
#   se trata como feedback → el LLM re-propone (v+1, previous_roadmap_id) y la
#   versión anterior pasa a superseded (supersede_roadmap exige HumanApproval —
#   el debate ES la acción humana). El gate sigue aplicando sobre la revisión.
#   Frontend: gate con botones Aprobar/Rechazar/Debatir; mountPendingTutorGates()
#   re-monta los gates pendientes desde /api/agent/approvals después de cada
#   re-render canónico (los gates no son episodios y el re-render los borraba).
# Dashboard wiring: ipa/tutor/tutor_chat.py (TutorChatDriver) — state machine por
#   sesión, role="tutor" en /api/agent/chat/stream, botones approve/reject en el
#   chat via /api/tutor/roadmap/decision + /api/tutor/research/decision.
#   Sesiones paralelas: estado por session_id; recovery de propuestas pendientes
#   solo aplica a la primera sesión tras boot.
#   Corpus insuficiente (<3 conceptos) → ResearchRequest (gate humano) →
#   executor en background; al completar se graba un episodio en la sesión del
#   usuario y un mensaje sin tema ("dale") retoma el topic guardado.
#   Budget de research del Tutor: 15 fuentes / 300s (research_topic del chat
#   general: default 5, tope 20 — casual vs roadmap). Con una research aprobada
#   en vuelo el driver responde determinísticamente "aguardamos a que llegue la
#   información de la fuente web" (sin LLM, sin proponer pasos).
# Consolidación de sesiones (ipa/agent/session_consolidator.py): resume sesiones
#   cerradas/archivadas idle >= 5 min (worker cada 90s, 3 por ciclo). Cierra
#   sesiones 'active' huérfanas (restart del dashboard) antes de cada ciclo.
#   El indicador "✦ resumida" del panel depende de consolidated_at. Bug
#   histórico: el consolidador esperaba .text y Ollama devuelve str → 0
#   resúmenes en silencio; ahora usa ipa/agent/llm_text.generate_text.
# Deep dive consolidado (Opción C): "Profundizar" en un reporte abre el CHAT
#   del agente con context=deep_dive (corpus/category_id/search en el body).
#   El stream handler corre deep_dive_prepare (retrieval agéntico sobre el
#   corpus del reporte, REPORTER_ROOT-only) e inyecta la evidencia al prompt;
#   max_new_tokens 768 en ese contexto. deep-dive.html + /api/deep-dive*
#   quedan deprecados (sin entry point en la UI; endpoints por compat).
# Progreso por unidad: tabla unit_progress (roadmap_id, unit_order, status)
#   aditiva en TutorStore — pending|current|done, el contrato Roadmap sigue
#   inmutable. Seed: unidad 1 → current al activar. Avance determinístico en
#   lecciones ("siguiente unidad", "ya entendí", "avancemos"… → _ADVANCE_RE).
#   Endpoint GET /api/tutor/roadmaps → unidades + status + títulos (dominio de
#   la fuente, nunca doc_ids) + mastery del tópico; stepper visual en el aside
#   del panel Agente (web/static/app.js renderTutorRoadmaps + CSS .rm-*).
# Memoria agéntica retrievable: ipa/agent/memory_store.py (MemoryStore + FTS5 +
#   MemoryIndexer). Corpus personal/agéntico: resúmenes de sesión, user model,
#   principios estratégicos, mastery del Tutor — items derivados con provenance.
#   Tool: recall_memory (BASE_TOOLS) — sync incremental + recall por scope.
#   Routing: query_gate clasifica "memory" → api.py corre recall_memory
#   server-side e inyecta los items como contexto (sin retrieval del corpus).
#   Recall híbrido: FTS5 + sqlite-vec (MemoryVectorIndex, memory_vectors.db,
#   derivado/rebuildable) fusionados por RRF. El embed usa el adapter warm del
#   dashboard (get_embedding_adapter); sin él → FTS-only (tests, CLI).
#   Resúmenes por unidad: TutorStore.unit_summaries (escrito al avanzar de
#   unidad, _summarize_unit) → indexer los indexa como episodic/lesson_unit.
# Auto-research en gap de corpus (IPA_AUTO_RESEARCH=1 default, =0 desactiva):
#   si el retrieval viene vacío o el reply declara insuficiencia, el prompt
#   instruye emitir [TOOL:research_topic] y el safety-net lo inyecta si el
#   modelo no lo emitió. Dedup por query normalizada (research_recent.json,
#   IPA_AUTO_RESEARCH_DEDUP_MINUTES=10) — las llamadas explícitas no dedupean.
#   Al terminar, el watcher sintetiza una respuesta LLM a la pregunta original
#   con el material ingerido (fallback: resumen de stats).
# Review queue de rechazados (ipa/agent/research_review.py, store en
#   outputs/agent/research_review.db — derivado): docs scrapeados rechazados
#   por quality/date/judge se encolan; el review worker (server.py) los relee
#   con el LLM tras IPA_RESEARCH_REVIEW_IDLE_SECONDS=60 de inactividad,
#   interrumpible entre items y resumable entre restarts; promote → FastPath
#   + provenance agent_research + embed, discard → mark. Solo corre con el
#   provider ya cargado (nunca levanta el modelo para esto).
#   Fix: run_research.py mergea el progress inicial (session_id) — sin eso el
#   watcher perdía el destino del episodio de cierre.

# Fase 3 — Horizontalidad profunda
# Topic clusters: clustering determinístico sobre centroides (threshold-based, emergente)
# Store: outputs/agent/topic_clusters.db (TopicClusterStore, índice derivado rebuildable)
# Multi-hop: TopicNavigator (vertical-first, cobertura = diversidad de documentos, max 2 hops)
# Experimento gate: outputs/experiments/E13-multihop-vs-vertical.json
# Consolidación: MemoryConsolidator + UserModelInference (propuestas pending → aprobación humana)
# Store: outputs/agent/consolidation.db (ConsolidationStore, nunca borra originales)
# Contratos: user_topic_record, user_evidence (validados por validate_agent_contract.py)
# CLI: scripts\cli\agent.py chat --role tutor --llm -m "mensaje"
# Idle enrichment: background topic enrichment while the system is idle.
#   Scheduler (nuevo): ipa/agentic/idle_scheduler.py — reemplaza los workers
#     sueltos por un registro de tareas con tier/prioridad/recursos/cooldown.
#     Recursos = locks nombrados (llm, embeddings, cluster_store, agent_db,
#     user_model, skills, strategic, uncertainty, corpus_main, corpus_reporter):
#     tareas que comparten recurso se serializan solas; recursos disjuntos
#     avanzan en paralelo (pool de 3 threads para Tier 1). Locks en orden
#     alfabético → sin deadlocks. Tier 2 = un pase serial preemptible entre
#     items (should_abort). El scheduler no conoce el dashboard: los cuerpos
#     de las tareas viven en server.py y el contexto llega por CycleContext.
#   Module: ipa/agentic/idle_enrichment.py + ipa/agentic/idle_cognition.py
#   Tier 1 (cada ciclo de 60s, cooldowns por tarea): higiene (cierre de
#     sesiones huérfanas, prio 10) → memoria (consolidación de sesiones,
#     prio 20, usa LLM solo si ya está cargado) → topificación (clustering +
#     curación + continuidad; main y reporter serializados por cluster_store;
#     reusa embeddings de LanceDB — sin VRAM) → promoción (cola + sweep)
#     → cognitivo determinístico (user model, skills, principios, agenda —
#     4 tareas paralelas, todo pending → gate humano).
#   Tier 2 (LLM): corre si (a) idle profundo >= 30 min con
#     IPA_IDLE_DEEP_ENRICHMENT=1 (puede CARGAR el modelo), o (b) el modelo YA
#     está cargado por el chat y el sistema lleva >= 5 min quieto
#     (IPA_IDLE_LLM_LOADED_ENRICHMENT=1 default, IPA_IDLE_LLM_LOADED_THRESHOLD_MINUTES=5)
#     — se aprovecha sin cargar nada y se descarga solo si lo cargamos nosotros.
#     Rich topic labels + LLM classification de gray docs + LLM grouping +
#     reflexión cognitiva. Aborta si el usuario vuelve mid-run.
#   Driver: _idle_enrichment_worker en server.py (único thread idle; los
#     cuerpos de las tareas son closures registradas en el IdleScheduler).
#   Log: outputs/web_dashboard/logs/idle_enrichment.log — formato nuevo
#     "[idle-sched T1/T2] <tarea>: {conteos} (duración)"; líneas sin novedad
#     no se loguean.
#   Contrato de providers normalizado en ipa/agent/llm_text.py (Ollama
#     devuelve str; ExL3 GenerationResult) — usado por el consolidador de
#     sesiones y la reflexión estratégica. Sin esto, un provider str hace
#     que la inferencia muera en silencio (bug real de 0 resúmenes).

# Fase 4 — Capa cognitiva (planificación, memoria estratégica, skills, incertidumbre, user model)
# Punto 2+3: Task planner + persistent task queue
#   Module: ipa/agent/task_planner.py (TaskStore, Planner, TaskExecutor)
#   Store: outputs/agent/task_store.db (persistente, resumible)
#   Planner: 1 generación LLM produce plan JSON → fallback a plantilla determinística
#   Executor: loop sobre sub-tasks, cada uno con bound de 3 tools (loop existente)
#   Budget: max 6 sub-tasks × 3 tools = 18 tools/tarea (vs 3 tools/turno del chat reactivo)
#   Resumibilidad: TaskStore persiste current_subtask → "seguí lo de ayer" resume
#   Tools: plan_task, list_tasks, get_task, resume_task
# Punto 4: Strategic memory (reflexión que extrae principios)
#   Module: ipa/agent/strategic_memory.py (StrategicMemoryStore, StrategicReflector)
#   Store: outputs/agent/strategic_memory.db (principles, append-only proposals)
#   Reflector: detecta tool_patterns, query_noise, response_style (determinístico)
#   + LLM opcional para principios abstractos (idle Level 2)
#   Gate: propuestas pending → humano approve → active → se inyectan en system prompt
# Punto 5: Skill library dinámica (adquisición/composición)
#   Module: ipa/agent/skill_library.py (SkillLibraryStore, SkillDetector)
#   Store: outputs/agent/skill_library.db (skills, append-only proposals)
#   Detector: cuenta secuencias de tool_calls repetidas (>= 3 ocurrencias)
#   Skills approved se inyectan en system prompt junto con las estáticas del YAML
# Punto 6: Uncertainty + active research agenda
#   Module: ipa/agent/uncertainty.py (UncertaintyStore, UncertaintyTracker, ActiveResearchAgenda)
#   Store: outputs/agent/uncertainty.db (topic_confidence, research_proposals)
#   Tracker: EMA sobre scores de search_corpus/compile_report/research_topic
#   Agenda: tópicos con confidence < 0.4 → propone research (pending → gate humano)
#   Tools: list_research_agenda
# Punto 8: User model transversal
#   Module: ipa/agent/user_model.py (UserModelStore, UserModelInferer)
#   Store: outputs/agent/user_model.db (goals, interests, preferences, facts)
#   Inferer: intereses por frecuencia de queries, goals por tareas recurrentes,
#   preferencias de longitud de respuesta (determinístico, idle Level 1)
#   + LLM opcional para goals/estilo abstractos (idle Level 2)
#   Gate: inferencias pending → humano approve → active → se inyectan en system prompt
#   Tools: get_user_profile, set_user_goal, set_user_interest
# System prompt injection: Identity.system_prompt() ahora incluye 4 capas dinámicas:
#   user model, strategic principles, learned skills, uncertainty topics.
#   Cada capa es independiente y renderiza vacío si no hay datos (no-op en fresh install).
#   Ver DEC-006, RES-005, RES-006.

# Tutor Agent — validate a contract record
.venv\Scripts\python.exe scripts\validation\validate_tutor_contract.py LearningGoal path\to\goal.json
.venv\Scripts\python.exe scripts\validation\validate_tutor_contract.py Roadmap path\to\roadmap.json

# Tutor Agent — smoke test del modelo estrella (Qwen3.5-9B EXL3 3.0bpw + MTP)
.venv\Scripts\python.exe scripts\operations\test_exl3_provider.py
.venv\Scripts\python.exe scripts\operations\test_exl3_provider.py --interactive
.venv\Scripts\python.exe scripts\operations\test_exl3_provider.py --all

# Gate pedagogical_v1 (720 generaciones; corre en small-model-deliberation)
# Set-Location C:\Users\Valen\Desktop\Proyectos\small-model-deliberation
# .venv\Scripts\python.exe engine_benchmark\runners\run_engine_benchmark.py --model qwen35-9b-exl3-3.0 --tasks all --cases all --run-id tutor-fase2-qwen35-9b-3.0
# Resultado Fase 2 (2026-09-08): quality 0.9124, diagnosis 0.6628, next_step 0.75, json 0.9028, abstention 0.8125, 0 failures

# Agent core (Fase 0) — identidad + sesiones + memoria episódica (outputs/agent/agent.db)
.venv\Scripts\python.exe scripts\cli\agent.py chat                    # sesión interactiva
.venv\Scripts\python.exe scripts\cli\agent.py chat -m "mensaje"       # turno único
.venv\Scripts\python.exe scripts\cli\agent.py sessions                # listar sesiones
.venv\Scripts\python.exe scripts\cli\agent.py audit --recent 5        # exportar a outputs/agent/exports/

# Agent core (Fase 1) — tools determinísticas + research_topic + dashboard surface
# Dashboard endpoints: /api/agent/sessions, /api/agent/session?session_id=..., /api/agent/chat (POST)
# Agent tools (agent_tools.py, code-level): search_corpus, list_topics, get_topic_info,
#   recall_conversation, research_topic, compile_report — all dispatchable via execute_tool().
# System tools (system_tools.py, chat-visible): unified registry of SystemToolSpec;
#   TOOL_CATALOG is generated from the registry (never hand-edited).
#   Visible: get_system_status, search_corpus, list_topics, list_promotions, get_report,
#   compile_report, research_topic (async), run_ingestion (async), list_sources.
#   Hidden alias: run_pipeline → run_ingestion. Chat tool loop: max 3 rounds/turn,
#   duplicate (name,args) refused. Async tools arm watchers (PIPELINE_WATCH,
#   RESEARCH_WATCH) that deliver a summary episode to the session on completion.
# Contratos: tool_call, tool_result, web_source (validados por validate_agent_contract.py)

# Agent core (Fase 2 bridge) — flujo agéntico con juicio LLM
# Flujo: gap detection → web search → juicio de snippets → scrape → juicio de contenido → ingesta selectiva
# Jueces: HeuristicJudge (determinístico, default) | LLMJudge (modelo estrella vía provider_wiring)
# CLI: scripts\cli\agent.py research "query" [--corpus DIR] [--max-urls N] [--llm]
# Módulos: ipa/agent/judge.py (Judgment, HeuristicJudge, LLMJudge), ipa/agent/provider_wiring.py
# Gap detection: ipa.agent.assess_corpus_coverage (andamiaje determinístico, no LLM)
# Search backends: IPA_SEARXNG_URL (SearXNG local, preferido), cache SQLite local,
# DuckDuckGo fallback controlado. Cache path: IPA_WEB_SEARCH_CACHE.
# SearXNG local: docker compose -f .devin\searxng\docker-compose.yml up -d  (127.0.0.1:8888)
# No dependas de DDG: configura SearXNG local, por ejemplo IPA_SEARXNG_URL=http://127.0.0.1:8888.

# Compile ExLlamaV3 native extension (sm_89 / RTX 4050)
# Only needed after changing ExLlamaV3, Python, PyTorch, CUDA, or GPU.
$env:Path = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;$PWD\.venv\Scripts;$env:Path"
$env:MAX_JOBS = "2"
$env:TORCH_CUDA_ARCH_LIST = "8.9"
Push-Location exllamav3-dev\source
..\..\.venv\Scripts\python.exe setup.py build_ext --inplace
Copy-Item exllamav3_ext*.pyd ..\build\
Pop-Location

# Web dashboard (refresh dinámico, API local y acciones de curación)
.venv\Scripts\python.exe scripts\operations\web_dashboard.py --host 127.0.0.1 --port 8765

# One-click launcher: starts dashboard and opens browser
.\start_ipa_dashboard.bat
# Optional: also start the console orchestrator
.\start_ipa_dashboard.bat -StartOrchestrator

# Run tests
.venv\Scripts\python.exe -m pytest -v
```

## Architecture (Stage 1 + Stage 2 + Stage 3)

Stage 3 additions:
- `adaptive_chunker.py` — deterministic selective merge + recursive fallback for low-density chunks.
- `retrieval_eval.py` — E10 query generation, IR metrics, hybrid fusion.
- `enrichment.py` / `ollama_adapter.py` — optional E9 LLM enrichment; Ollama calls use `think=False`.
- E9 result: summary enrichment best for lexical (+14.3% recall@10), synthetic_queries best for vector (+7.1%); all 3 strategies improve retrieval on NL queries.
- `trace_log.py` — E11 observability: SQLite-backed TraceEvent store; every pipeline stage emits events with artifact_id, stage, status, input/output hash, latency, worker_id, metadata.
- `web_scraper.py` — web scraping: trafilatura + BeautifulSoup; deterministic link extraction (CSS selector, URL pattern, exclude paths, allowed domains); Playwright backend for JS-rendered sites; lazy-load image handling; document download (PDF/DOCX/PPTX/XLSX/TXT/CSV/MD/etc. to Landing zone); arxiv link following (abs/pdf → PDF download); RSS/Atom feed parsing for article discovery; trust_article_dates flag for sites with unreliable date metadata; date filtering.
- `ocr_adapter.py` — EasyOCR wrapper; lazy model loading; GPU support; per-detection confidence.
- `exl3_provider.py` — Tutor Agent LLM engine: ExLlamaV3 provider with MTP speculative decoding, continuous batching, ChatML no-think, VRAM monitoring. Modelo estrella: Qwen3.5-9B EXL3 3.0bpw. Auto-discovers exllamav3-dev/ compiled extension on import. Watchdog thread en `generate_batch` que fuerza `clear_queue()` después del hard timeout (iterate() es bloqueante y el timeout del while loop nunca se evalúa si el prefill cuelga).
- `tutor_contracts.py` — Runtime dataclasses and enums for LearningGoal, Concept, Roadmap, AssessmentResult and ResearchRequest. Authoritative shapes remain in contracts/*.schema.json.
- Reporter modules — `reporter_config.py`, `reporter_metadata.py`, `reporter_representation.py`, `reporter_curation.py`, `reporter_topics.py`, `reporter_claims.py`, `reporter_store.py`, `reporter_report.py`, `reporter_pipeline.py`, `reporter_deep_dive.py` and `reporter_research.py`; isolated periodic reports under `outputs/reporter/`. The legacy `reporter_promotion.py` was removed; promotion is now handled by the agentic layer (see DEC-003). The Reporter is now an agent-invoked tool: the agent calls `compile_report` (via `compile_report_executor.py`) with a set of document_ids to produce a fine-grained report. The pipeline no longer runs the Reporter automatically — `run_full_pipeline()` does scraper + FastPath only. `scripts/cli/run_reporter.py` was removed.
- Reporter curation automation — `reporter_curation.py` computes a weighted `promotion_score` (relevance 0.30, novelty 0.20, source_quality 0.20, impact 0.15, depth 0.10, actionability 0.05) and applies a conservative auto-promotion policy: a document is auto-approved only when it is a `promote` decision, has no duplicate, is in-period, has text, and meets `relevance>=0.80`, `novelty>=0.60`, `source_quality>=0.65`, `impact>=0.60` and `promotion_score>=0.78`. Auto-approved decisions are recorded with `decided_by=auto-promotion-v1`, `review_status=approved` and a note with the score; ambiguous cases remain `review_status=pending` for human review. Curation uses BGE-M3 embeddings from LanceDB (document centroids) for semantic relevance and novelty scoring — no LLM inference needed for classification. Source quality, impact, depth and actionability use improved heuristics (domain trust list, entity density, structure detection, action verb matching).
- Promotion (decoupled from Reporter) — `promotion_policy.py` evaluates promotion based on provenance: `configured_scrape` → auto-promote (no threshold); `agent_research` → promote only if `promotion_score >= 0.70`; unknown → not promoted. `promotion_executor.py` performs the physical copy (documents, chunks, BM25, LanceDB vectors, provenance) and is idempotent, then PURGES the promoted documents from the source staging corpus (`purge_promoted_from_source`: tombstone docs+chunks, drop BM25 FTS entries, delete LanceDB vectors and stale embedding_jobs) — the main corpus is canonical and staging must not accumulate promoted copies. The `promotion_queue` in `TopicClusterStore` tracks pending promotions with `source_corpus`. The idle enrichment worker evaluates promotion after Level 1 and processes the queue. Dashboard endpoint `/api/promotion/queue` exposes pending promotions. See DEC-003, PAT-005, EXP-006.
- Landing/Transit/Archive lifecycle — Landing is intake only; nothing processed stays there. `landing_sweep.py` classifies every registered artifact: live doc in MAIN corpus (`outputs/experiments/E12-corpus`) → `Archive/`; in staging corpus / promotion_queue pending / curation review pending → `Transit/`; `failed` everywhere or human `review_status=rejected` (read from both `topic_clusters.db` and the active `reporter.db`) → deleted; unregistered → stays. Transit is re-scanned every run (by content hash): promoted → Archive, rejected → delete. Operational files (`scrape_history.db`, `scrape_report.json`, `*.db`, hidden, `*.pending_delete`) are never touched. The sweep runs automatically at the end of `run_full_pipeline`, after promotion-queue processing in the idle worker, and after dashboard review actions (`/api/reports/review`, `/api/decisions/review`, `/api/promotions/process`); manual: `scripts/operations/sweep_landing.py [--dry-run]`. Tests: `tests/test_landing_sweep.py`.
- Provenance backfill — scraper docs without a `document_sources` row are discarded by promotion_policy as "unknown provenance" forever. `provenance.backfill_from_landing_registry()` marks artifacts whose source_uri is under `Landing/web/` as `configured_scrape` (agent-research docs self-record `agent_research` at fetch time, so missing provenance implies the scraper); it extracts the real source URL from the `Source:` line embedded in the document text. Wired into `run_full_pipeline` (after fast_path) and into the idle enrichment worker (before policy evaluation).
- Adaptive rechunk artifacts are written under `outputs/experiments/E5-adaptive/`; source stores are never modified by analysis scripts.

```
src/ipa/
  contracts.py       Runtime dataclasses aligned with authoritative JSON Schemas
  ingestion/         Landing, MIME, safety, parsers, chunking and FastPath
  storage/           DocumentStore and canonical persistence
  indexes/           FTS5/Tantivy/LanceDB/sqlite-vec/embedding/reranker adapters
  acquisition/       Web fetch, scraper and OCR adapters
  enrichment/        Derived summaries, queries and claims
  observability/     TraceLog and structured event helpers
  agentic/           QueryIR, evidence, bounded retrieval/context adapters
  reporter/          Reporter metadata, curation, topics, reports and promotion
  tutor/             Learning contracts and future pedagogy capabilities
  providers/         ExLlama/Ollama provider adapters
  mcp/               Runtime MCP boundary
  dashboard/         Dashboard server and orchestration migration target
```

Each bounded context has one responsibility. Retrieval, ranking, context building,
generation and orchestration remain separate. Parsers never import embedding or
enrichment implementations.

## Conventions

- Contracts in `contracts/` are the authority; tools integrate via adapters.
- `contract_vocabulary.json` defines required fields per record.
- `experiment_report.schema.json` defines the machine-readable report format.
- Tutor Agent schemas define LearningGoal, Concept, Roadmap, AssessmentResult and ResearchRequest; `tutor_common.schema.json` holds shared provenance and approval definitions.
- `validation_contract.md` defines the validation discipline.
- All outputs go under `outputs/`; each experiment uses `outputs/experiments/E#/`.
- Manifests are append-safe: each run writes a timestamped file.
- Synthetic fixtures live in `data/sample/input/`; everything under `input/` is
  treated as an artifact.

## Cleanup rules (DO NOT violate)

- **Never delete `Landing/` if it has unprocessed files** — they haven't been
  ingested yet. Only clean files that are already in `Archive/`.
- **Never delete `Archive/`** — those files are already processed and indexed.
- **Never delete `Landing/web/scrape_history.db`** — it prevents re-downloading
  URLs already scraped. Clearing it forces re-download of everything.
  This is the ONLY DB that should exist in `Landing/`. If you see
  `Landing/scrape_history.db` (without `web/`) or `Landing/landing_zone.db`,
  those are residuals from old runs and should be deleted.
- **Scraper output must always be `--output Landing/web`** — never
  `--output Landing` directly. This keeps all scraped content under
  `Landing/web/<site>/` and ensures a single `scrape_history.db` at
  `Landing/web/scrape_history.db`.
- **Only delete `outputs/experiments/E12-corpus/`** when we want to
  re-process from scratch (re-parse, re-chunk, re-index). The source files
  in `Archive/` and `Landing/` stay untouched.
- No ADRs until a capability is implemented and validated.
- Tests must validate real behavior, not just file existence.

## Test count

752 tests across 34 files (1 skipped — network test):
- `test_adaptive_chunker.py` (18) — adaptive merge behavior and invariants
- `test_agent_core.py` (10) — Fase 0: identity, memory, episodes, omnipresence gate
- `test_agent_tools.py` (15) — Fase 1: deterministic tools + contract validation
- `test_agentic_judge.py` (37) — agentic research flow: LLMJudge, HeuristicJudge,
  knowledge-gap detection, selective ingest, judgment audit,
  freshness lenient/strict, scraper auto-engine retry recording,
  retry buffer from snippet pool, scrape failure classification
  (unreachable/blocked/extraction_failed), freshness policy in audit trail
- `test_agentic_runtime.py` (23) — QueryIR, EvidenceSet, ContextPackage and budget contracts
- `test_chunkers_alt.py` (17) — recursive, token and semantic chunkers
- `test_contract_vocabulary.py` (5) — contract authority and identity fields
- `test_corpus_service.py` (1) — Reporter corpus boundary service
- `test_dashboard_agent.py` (3) — Fase 1: omnipresence CLI ↔ dashboard gate
- `test_eks.py` (5) — EKS metadata and validation
- `test_experiment_report_schema.py` (21) — schema and integrity validation
- `test_fast_path.py` (47) — landing, MIME, parsing, chunking, store and BM25
- `test_index_adapters.py` (16) — Tantivy, embeddings, LanceDB and sqlite-vec
- `test_manifest_contract.py` (20) — append-safe manifest and integrity checks
- `test_parsers_alt.py` (9) — Docling, Unstructured and parser consistency
- `test_process_jobs.py` (48) — JobSpec registry, parsers, process runner and state
- `test_public_package.py` (2) — public `ipa` surface imports
- `test_reporter.py` (18) — Reporter metadata, representations, curation, emergent topics,
  continuity, isolated pipeline, contracts and promotion review
- `test_reporter_planner.py` (9) — deterministic query planning
- `test_reporter_retrieval.py` (5) — evidence retrieval adapter
- `test_research_executor.py` (23) — Fase 1: research_topic executor + web_source contracts
- `test_compile_report.py` (8) — compile_report agent tool: report compilation from document IDs
- `test_system_tools.py` (24) — unified system-tool registry, generated catalog,
  deterministic fuzzy marker parsing, executor dispatch, validation paths
- `test_cognitive_layer.py` (67) — cognitive layer: task planner (TaskStore,
  Planner LLM+deterministic fallback, TaskExecutor), strategic memory
  (principle extraction, approval gate), skill library (pattern detection,
  composition), uncertainty tracking (confidence EMA, active research agenda),
  transversal user model (goals, interests, preferences, facts, inference),
  system prompt injection
- `test_retrieval_eval.py` (28) — IR metrics, query generation and hybrid fusion
- `test_trace_log.py` (21) — observability and FastPath trace integration
- `test_memory_consolidation.py` (13) — Fase 3: memory consolidation proposals
  (never deletes, human approval gate), user model inference (no regression)
- `test_topic_clusters.py` (16) — Fase 3: TopicCluster contract + store +
  deterministic clustering over centroids (emergent, threshold-based)
- `test_topic_navigator.py` (10) — Fase 3: bounded multi-hop retrieval
  (vertical-first, document-diversity coverage, neighbor expansion, trace)
- `test_topic_backfill.py` (4) — incremental topic backfill (idle-time worker)
- `test_idle_enrichment.py` (25) — idle enrichment Level 1: build_document_dicts,
  discover_topics full pass, heuristic curation, all-clustered edge case,
  promotion policy (configured_scrape auto, agent_research >= 0.70, below
  threshold, unknown), promotion executor deduplication, source-corpus-aware
  queue processing, document_sources (put/get, all, by_provenance)
- `test_tutor_contracts.py` (35) — Tutor schemas, provenance, approvals,
  integrity invariants and runtime dataclasses
- `test_tutor_runtime.py` (28) — Fase 2: Tutor role runtime (TutorStore state scope,
  deterministic diagnosis, LLM assessment with abstention, mastery update loop,
  roadmap proposal → human approval gate → activation, versioning supersedes,
  ResearchRequest create → approve/reject → execute via Fase 1 executor,
  topic_cluster_id episode wiring)
- `test_user_model_contracts.py` (15) — Fase 2: UserTopicRecord + UserEvidence
  schemas, mastery-requires-evidence invariant, vocabulary registration
- `test_web_dashboard.py` (8) — dashboard paths, sources and URL security
- `test_web_scraper.py` (65) — scraping, downloads, RSS, OCR and Playwright
