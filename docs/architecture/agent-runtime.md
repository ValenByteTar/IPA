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
`scripts/cli/agent.py research "query" [--corpus DIR] [--max-urls N] [--llm]`.

Search backends, in order of preference:

- local SearXNG (`IPA_SEARXNG_URL`, e.g. `http://127.0.0.1:8888`) — start with
  `docker compose -f .devin/searxng/docker-compose.yml up -d`;
- local SQLite cache (`IPA_WEB_SEARCH_CACHE`);
- controlled DuckDuckGo fallback.

Do not depend on DDG: configure local SearXNG.

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
provenance + embed; discard → mark. Only runs with the provider already
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
- **Tier 2** (LLM): runs when (a) deep idle ≥ 30 min with
  `IPA_IDLE_DEEP_ENRICHMENT=1` (may load the model), or (b) the model is
  already loaded by the chat and the system has been quiet ≥ 5 min
  (`IPA_IDLE_LLM_LOADED_ENRICHMENT=1` default,
  `IPA_IDLE_LLM_LOADED_THRESHOLD_MINUTES=5`). Rich topic labels + LLM
  classification of gray docs + LLM grouping + cognitive reflection. Aborts if
  the user returns mid-run.
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

`ipa/mcp/mcp_server.py` exposes the knowledge pipeline to external MCP clients
(stdio): `search_knowledge`, `ingest_url`, `scrape_domain`, `ingest_file`,
`list_sources`, `get_document`. The agent itself does **not** consume MCP:
its tools are in-process calls over the registries above. No client config in
this repo points at this server. It reimplements retrieval instead of reusing
the shared path — see the drift note in `retrieval.md` (its rerank silently did
nothing until the `RerankCandidate` field bug was fixed, 2026-09-18).

EKS (`tools/eks_mcp_server.py`) is a separate, read-only dev-time server.
