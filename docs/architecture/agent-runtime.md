# Agent runtime

Implementation-level notes for the agent core, its memory and the idle
background work. Architectural boundaries live in `boundaries.md`; the Tutor
has its own `tutor.md`; retrieval is `retrieval.md`.

## Core

- Identity from `configs/agent_identity.yaml` (sha256 recorded per episode —
  DEC-002). CLI and dashboard share the same sessions and episodic memory
  (`outputs/agent/agent.db`).
- Tools: `agent_tools.py` (code-level: `search_corpus`, `list_topics`,
  `get_topic_info`, `recall_conversation`, `research_topic`, `compile_report`)
  and `system_tools.py` (chat-visible registry of `SystemToolSpec`; the
  `TOOL_CATALOG` is generated from the registry, never hand-edited).
  `get_document` abre un doc por id (provenance real, lifecycle, decisión de
  curación, texto) buscando en main + stagings — se desbloquea tras
  `search_corpus` en `TOOL_PROGRESSION`.
- Chat tool loop: max 3 rounds per turn; duplicate `(name, args)` refused.
  Async tools arm watchers (`PIPELINE_WATCH`, `RESEARCH_WATCH`) that deliver a
  summary episode to the session on completion.
- Contracts: `tool_call`, `tool_result`, `web_source` (validated by
  `validate_agent_contract.py`).

## Memory

- Retrievable agentic memory: `ipa/agent/memory_store.py` (`MemoryStore` +
  FTS5 + `MemoryIndexer`). Corpus: session summaries, user model, strategic
  principles, Tutor mastery — derived items with provenance.
- Tool `recall_memory` (BASE_TOOLS): incremental sync + recall by scope.
  `query_gate` classifies "memory" → `api.py` runs `recall_memory`
  server-side and injects the items as context (no corpus retrieval).
- Hybrid recall: FTS5 + sqlite-vec (`MemoryVectorIndex`, `memory_vectors.db`,
  derived/rebuildable) fused by RRF. Embeddings use the dashboard's warm
  adapter (`get_embedding_adapter`); without it → FTS-only (tests, CLI).
- Unit summaries: `TutorStore.unit_summaries` (written when advancing a unit)
  are indexed as `episodic/lesson_unit`.

## Session consolidation

`ipa/agent/session_consolidator.py`: summarizes closed/archived sessions idle
≥ 5 min (worker every 90 s, 3 per cycle); closes orphaned `active` sessions
(dashboard restart) before each cycle. The "✦ resumida" indicator depends on
`consolidated_at`. Historical bug: the consolidator expected `.text` while
Ollama returns `str` → 0 summaries silently; it now uses
`ipa/agent/llm_text.generate_text`. Summaries of roadmap sessions carry the
roadmap tag (see `tutor.md`).

## Agentic research flow (Fase 2 bridge)

gap detection → web search → snippet judgment → scrape → content judgment →
selective ingestion. Deterministic scaffolding (`assess_corpus_coverage`) decides
the gap; judges are replaceable: `HeuristicJudge` (default, deterministic) or
`LLMJudge` (star model via `provider_wiring`). Modules: `ipa/agent/judge.py`,
`ipa/agent/provider_wiring.py`. CLI:
`scripts/cli/agent.py research "query" [--corpus DIR] [--max-urls N]
[--sub-queries Q ...] [--llm]`.

Wide topical sweeps (`sub_queries`, max 8): the agent attaches facet queries of
the same topic; each runs its own `search_web` call and its results enter the
URL-deduped candidate pool. Prefilter, snippet judgment and content judgment
score each candidate against the query that produced it — facet-specific
vocabulary is not penalized for lacking the main query's terms. `max_urls`
(tool cap 50) still bounds successful ingestions and `max_seconds` the scrape
loop. `run_ingestion` is the untargeted sweep of all configured sources — no
topic filter; the tool descriptions in the registry make that routing explicit.

Search backends, in order of preference:

- local SearXNG (`IPA_SEARXNG_URL`, default `http://127.0.0.1:8888`) — the
  launcher ensures it at boot and the watchdog keeps it alive
  (`ensure_searxng`: starts Docker Desktop if the daemon is down, then
  `docker compose -f .devin/searxng/docker-compose.yml up -d`; managed only
  for local URLs, disable with `IPA_SEARXNG_MANAGED=0`, check cadence via
  `IPA_SEARXNG_CHECK_INTERVAL`, default 60s);
- local SQLite cache (`IPA_WEB_SEARCH_CACHE`);
- controlled DuckDuckGo fallback.

Do not depend on DDG: the managed local SearXNG is the real backend.

Work dir and heavy-work serialization (PM-004):

- each run scrapes into its own `outputs/agent/research/<run_id>/` and ingests
  **only that dir** — never the shared `Landing/web`, which a concurrent
  pipeline scraper is filling (the run would inherit its contents);
- ingest + embeddings run under `ipa/agentic/heavy_lock.py` with interactive
  priority (`outputs/agent/heavy.lock`, `IPA_HEAVY_LOCK_WAIT` default 300 s) and
  a post-scrape budget (`IPA_RESEARCH_INGEST_BUDGET`, default 600 s);
  `max_seconds` only bounds the scrape loop;
- `_embed_new_chunks` embeds only this run's documents, batched
  (`IPA_RESEARCH_EMBED_BATCH`) and deadline-aware — it used to embed every
  pending chunk of the canonical corpus in one call;
- the fast-path drain takes the same lock as background (per pass, bounded by
  `IPA_EMBED_PASS_CHUNKS`, default 256 chunks) and yields while an interactive
  waiter is registered; the wait is surfaced as
  `research_progress.json.heavy_wait` and shown in the dashboard indicator.

Staging destination (DEC-003, 2026-09-23): research no longer writes the main
corpus directly. Accepted documents land in the dedicated staging corpus
`outputs/agent/research_staging/` (fixed agent-domain path — NOT the
reporter's moving active-run pointer). The run records
`provenance=agent_research` for search-discovered URLs and
`provenance=user_provided` for URLs pasted by the user (seed URLs — a known
source, auto-promotes like `configured_scrape`). T1 `topify_research_staging`
curates the corpus and `promotion_queue` physically promotes per
`promotion_policy` (`agent_research` needs `promotion_score >= 0.70`). The
final answer still sees the fresh material: retrieval merges main hits with
direct BM25 hits over the staging corpus (`staging_bm25` backend). Residual
embedding backlog launches `run_embed_drain --corpus <staging>` post
heavy-phase so the promotion vector-preflight never stalls. Kill switch:
`IPA_RESEARCH_STAGING=0` restores direct-to-main ingestion.

## Auto-research on corpus gap

`IPA_AUTO_RESEARCH=1` (default; `=0` disables): when retrieval comes back empty
or the reply declares insufficiency, the prompt instructs the model to emit
`[TOOL:research_topic]` and the safety net injects it if the model did not.
Dedup by normalized query (`research_recent.json`,
`IPA_AUTO_RESEARCH_DEDUP_MINUTES=10`) — explicit calls do not dedup. On
completion the watcher synthesizes an LLM answer to the original question with
the ingested material (fallback: stats summary).

## Rejected-document review queue

`ipa/agent/research_review.py` (store `outputs/agent/research_review.db`,
derived): scraped docs rejected by quality/date/judge are queued; the review
worker (`server.py`) re-reads them with the LLM after
`IPA_RESEARCH_REVIEW_IDLE_SECONDS=60` of inactivity, interruptible between
items and resumable across restarts. Promote → FastPath + `agent_research`
provenance + embed **into the research staging corpus** (DEC-003b: the LLM
re-review is no longer a bypass to main — T1 curation + the promotion policy
still gate the entry); discard → mark. Only runs with the provider already
loaded (never loads the model for this). `run_research.py` merges the initial
progress (`session_id`) — without it the watcher lost the closing episode
destination.

## Deep dive

"Profundizar" on a report opens the agent chat with `context=deep_dive`
(corpus/category_id/search in the body). The stream handler runs
`deep_dive_prepare` (agentic retrieval over the report corpus,
`REPORTER_ROOT`-only) and injects the evidence into the prompt;
`max_new_tokens` 768 in that context. `deep-dive.html` and `/api/deep-dive*`
are deprecated (no UI entry point; endpoints kept for compatibility).

## Idle enrichment

> Modelo completo de activación: **DEC-010** (`knowledge/decisions/`).

**Activation contract — `_idle()` is false while any of these hold**: the
`Idle T1/T2` sidebar toggle off, `tier0.active()`, `heavy_lock.holder()`,
a foreign `embedding_maintenance` job lease, `CHAT_BUSY`, any live child in
`JOBS`, or pipeline/reporter state `running`. Every non-idle tick resets
`LAST_ACTIVITY`, so idle time is measured from the moment the last writer
released — never from mid-ingestion. Each cycle then takes
`claim_job("idle_scheduler")` + `ENRICHMENT_LOCK` before running anything.

**Tier 0 — ingestion gate.** `run_fast_path` holds a cross-process lease
(`ipa/agentic/tier0.py`, `outputs/agent/tier0.lock`, `pid|owner|ts` +
heartbeat) for its whole run — ingest, watch loop and the final embedding
drain. While the lease is live, `_idle()` reports not-idle: no Tier 1/Tier 2
cycle may start (they read and mutate the same stores), and since a busy
check resets `LAST_ACTIVITY`, the idle countdown only starts once Tier 0 is
released. The lease survives an orphaned watcher (dashboard restart): it is
validated by `pid_alive` + TTL, not by the dashboard's process tree. A second
`run_fast_path` sees the live holder and exits — ingestion is idempotent and
a duplicate only contends. Interactive ingestion counts as Tier 0 too:
`research_ingest` runs under `heavy.lock` (interactive priority) and `_idle()`
also checks `heavy_lock.holder()`, so T1/T2 never evaluate a corpus mid-write.
The scraper writes `.scraper_done` itself via
`--done-file` (try/finally), so the watcher's idle gate no longer depends on
the pipeline thread surviving.

**Tier 0 ingest signals** (`ipa/ingestion/ingest_metadata.py`): every
ingestion path — `run_fast_path` (initial pass + watch cycles),
`research_executor` and `ingest_reviewed_doc` — records derived per-document
signals in the same run instead of letting Tier 1 re-derive them per cycle:
`document_metadata` (`normalized_hash` in `reporter_curation` format, title,
`published_at`, `char_count`, `extra_json`) and `document_sources`
(`configured_scrape` provenance + real `published_at` for artifacts under
`Landing/web/**`). Exact duplicates vs the main corpus are flagged as
`extra.duplicate_of_main`; after the embedding drain, a `extra.novelty_hint`
(max cosine vs main + `nearest_doc_id` + `main_doc_count` +
`main_latest_stored_at`) is persisted so Tier 1 only pays the lexical
confirmation of the two-factor novelty gate (DEC-003). A stale hint is not
discarded: Tier 1 re-verifies it against only the embeddings of docs added
to main since the hint's snapshot (`document_embeddings(new_ids)` — O(new),
not O(corpus)) and persists the refreshed hint, so promotions no longer
invalidate every hint at once. `chunks.text` stays canonical — chunk enrichment lives in
`metadata.enrichment.enriched_text` and `enriched_text()` resolves the
derived representation used for embedding and BM25/LanceDB reindexing.

Scheduler: `ipa/agentic/idle_scheduler.py` — a task registry with
tier/priority/resources/cooldown. Resources are named locks (`llm`,
`embeddings`, `cluster_store`, `agent_db`, `user_model`, `skills`,
`strategic`, `uncertainty`, `corpus_main`, `corpus_reporter`): tasks sharing a
resource serialize, disjoint ones run in parallel (Tier 1 pool of 3 threads).
Locks are taken in alphabetical order → no deadlocks. Tier 2 is a serial
preemptible pass between items (`should_abort`). The scheduler does not know
the dashboard: task bodies live in `server.py` and context arrives via
`CycleContext`.

- **Tier 1** (60 s cycle, per-task cooldowns): hygiene (orphan sessions,
  prio 10) → memory (session consolidation, prio 20, uses the LLM only if
  already loaded) → topification (clustering + curation + continuity; main and
  reporter serialized by `cluster_store`; reuses LanceDB embeddings — no VRAM)
  → promotion (queue + sweep) → deterministic cognition (user model, skills,
  principles, agenda — 4 parallel tasks, all `pending` → human gate).
  Topification is incremental and gated: a `dirty:<corpus>` flag in
  `topic_clusters.meta` is set by every corpus writer (ingest, promotion,
  review ingest); `_t_topify` early-exits when no flag is set and the live
  doc count/coverage/provenance are unchanged — no O(corpus) scan on quiet
  cycles. Inside a run, candidate ids are filtered before any document text
  or embedding is fetched.
  The last T1 task is `index_audit` (prio 60, read-only): a logical layer
  consuming Tier 0 signals (empty docs, leaked `duplicate_of_main` flags,
  repeated `normalized_hash`, stale novelty hints, metadata backfill queue,
  scrape records without URL) every `IPA_AUDIT_LOGICAL_SECONDS` (900 s), and
  a forced physical layer every `IPA_AUDIT_PHYSICAL_HOURS` (6 h) comparing
  chunk_id sets across store ↔ BM25 meta ↔ FTS ↔ LanceDB plus spam-chunk
  detection — crash/lock drift never sets dirty flags, so it cannot be
  change-gated. The LanceDB read is column-projected
  (`table_chunk_id_list`: `select(["chunk_id"])`, `to_arrow()` fallback) —
  same for every id-set consumer (promotion preflight/dedupe, embed-drain
  resume): the vector column is never materialized for membership checks.
  Results merge into `outputs/agent/index_health.json`
  (`status`: ok/warn/fail), surfaced on the dashboard as "Salud de índices".
  Detection only — repair stays gated behind ops scripts with dry-run.
- **Tier 2** (LLM): runs when (a) deep idle ≥ 30 min with
  `IPA_IDLE_DEEP_ENRICHMENT=1` (may load the model), or (b) the model is
  already loaded by the chat and the system has been quiet ≥ 5 min
  (`IPA_IDLE_LLM_LOADED_ENRICHMENT=1` default,
  `IPA_IDLE_LLM_LOADED_THRESHOLD_MINUTES=5`). Rich topic labels + **gray-zone
  second opinions** (batched `classify_many` over `agent_research` docs still
  `reporter_only` with `promotion_score` in `[IPA_GRAY_LO=0.5,
  IPA_GRAY_HI=0.70)` — rescues qualifying docs into the promotion queue;
  tracked via `aux_progress.gray_reviewed`) + LLM grouping + chunk enrichment
  + cognitive reflection. Aborts if the user returns mid-run.
- Driver: `_idle_enrichment_worker` in `server.py` (single idle thread; task
  bodies are closures registered in the scheduler).
- Log: `outputs/web_dashboard/logs/idle_enrichment.log` — format
  `[idle-sched T1/T2] <task>: {counts} (duration)`; lines without news are not
  logged.
- **Sidebar switch**: the `Idle T1/T2` toggle turns the whole enrichment
  off/on. Gate in the worker's `_idle()`: OFF starts no cycle and aborts
  in-flight Tier 2 passes (`should_abort`). Persisted in
  `outputs/web_dashboard/idle_enabled.json` (survives restarts); audited in
  the idle log. Endpoints: `GET /api/idle/status`,
  `POST /api/idle/toggle`.

## Cognitive layer (Fase 4)

- Task planner + persistent queue: `ipa/agent/task_planner.py`
  (`TaskStore`, `Planner`, `TaskExecutor`), store `outputs/agent/task_store.db`.
  1 LLM generation → plan JSON, fallback to a deterministic template; budget
  6 sub-tasks × 3 tools; resumable (`current_subtask`).
  Tools: `plan_task`, `list_tasks`, `get_task`, `resume_task`.
- Strategic memory: `ipa/agent/strategic_memory.py`, store
  `outputs/agent/strategic_memory.db`; deterministic pattern detection +
  optional LLM principles; pending → human approve → active → injected into
  the system prompt.
- Skill library: `ipa/agent/skill_library.py`, store
  `outputs/agent/skill_library.db`; repeated tool sequences (≥ 3) become
  proposals; approved skills join the static YAML ones in the prompt.
- Uncertainty + research agenda: `ipa/agent/uncertainty.py`, store
  `outputs/agent/uncertainty.db`; EMA over retrieval/report scores; topics
  below 0.4 propose research (pending → human gate). Tool:
  `list_research_agenda`.
- User model: `ipa/agent/user_model.py`, store `outputs/agent/user_model.db`;
  deterministic inference + optional LLM goals/style; pending → human approve →
  active → prompt injection. Tools: `get_user_profile`, `set_user_goal`,
  `set_user_interest`.
- `Identity.system_prompt()` composes 4 dynamic layers (user model, strategic
  principles, learned skills, uncertainty). Each renders empty when there is no
  data — a no-op on a fresh install. See DEC-006, RES-005, RES-006.

## Horizontalization (Fase 3)

- Topic clusters: deterministic centroid clustering (threshold-based,
  emergent); store `outputs/agent/topic_clusters.db` (`TopicClusterStore`,
  derived/rebuildable index).
- Multi-hop: `TopicNavigator` (vertical-first, coverage = document diversity,
  max 2 hops); experiment gate
  `outputs/experiments/E13-multihop-vs-vertical.json`.
- Consolidation: `MemoryConsolidator` + `UserModelInference` (pending
  proposals → human approval); store `outputs/agent/consolidation.db` — never
  deletes originals.
- Contracts: `user_topic_record`, `user_evidence` (validated by
  `validate_agent_contract.py`).

## MCP boundary

`ipa/mcp/mcp_server.py` is a **thin proxy** over the dashboard's unified tool
registry (stdio). It owns no models and no pipeline code: at startup it fetches
`GET /api/tools/catalog` and registers one MCP tool per registry spec, so the
MCP surface is generated from the same source of truth the 9B chat uses — it
cannot drift. Every call is `POST /api/tools/execute {name, args}` to the
dashboard (`IPA_PROXY_URL`, default `http://127.0.0.1:8765`); the dashboard
must be running (watchdog-managed). If it is down at startup, the generic
`ipa_tool(name, args)` + `list_ipa_tools()` still work once it comes back.

Read-only Tutor surface for external sessions (Devin/Claude): `tutor_focus`,
`tutor_projects`, `tutor_roadmap_context` — the same read models the Roadmaps
tab renders. `research_topic` is async (same as dashboard chat): it returns
"investigación en curso" and the material lands via `list_promotions` /
`get_report`. The agent itself does **not** consume MCP: its tools are
in-process calls over the registries above. (Pre-2026-09-19 the server
reimplemented retrieval directly — that duplication is gone; see the drift
note in `retrieval.md`.)

EKS (`tools/eks_mcp_server.py`) is a separate, read-only dev-time server.
