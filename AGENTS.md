# AGENTS.md — RES-023 Lab

Operational guide for agents working in this repo. Design detail lives in
`docs/` (architecture, operations, plans) and `knowledge/` (EKS: decisions,
experiments, benchmarks, patterns, postmortems). This file stays under 300
lines on purpose: commands + conventions, not narrative.

## Environment

- Python 3.12 required (`>=3.12,<3.13`). On this machine: `py -3.12` (3.12.8).
- venv: `.venv/` (not committed; created per-machine).
- Core deps: PyYAML, rich, pytest, pytest-cov, PyMuPDF, rank-bm25, numpy.
- Stage 2: tantivy (E6), lancedb + sqlite-vec + sentence-transformers (E7).
- Stage 3: docling + unstructured[pdf] (E3 parsers),
  langchain-text-splitters + tiktoken (E5 chunkers), trafilatura + easyocr
  (E4 scraper/OCR), playwright (E4 JS-rendered sites).
- Tutor Agent: torch (CUDA 12.6), exllamav3 1.4.4, transformers,
  flash-linear-attention. The `exllamav3_ext` native extension is compiled
  locally in `exllamav3-dev/` (sm_89 / RTX 4050); `ipa/providers/exl3_provider.py`
  auto-discovers it on import.
- Dev-only (installed in venv, not in requirements.txt): jsonschema.

## Common commands

```powershell
# Setup
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install jsonschema  # dev-only

# Environment check
.venv/Scripts/python.exe scripts/operations/check_environment.py

# Landing manifest (append-safe, timestamped) + validation
.venv/Scripts/python.exe scripts/cli/build_landing_manifest.py --input data/sample/input
.venv/Scripts/python.exe scripts/validation/validate_contracts.py outputs/manifests/processing.<ts>.jsonl --integrity
.venv/Scripts/python.exe scripts/validation/validate_experiment_report.py outputs/experiments/E0/report.json

# Fast path (Landing) / synthetic fixtures
.venv/Scripts/python.exe scripts/cli/run_fast_path.py --output outputs/experiments/E1 --query "ingestion pipeline"
.venv/Scripts/python.exe scripts/cli/run_fast_path.py --input data/sample/input --output outputs/experiments/E1

# Benchmarks: indexes (E6 lexical / E7 vector), parsers (E3), chunkers (E5)
.venv/Scripts/python.exe scripts/benchmarks/run_index_benchmark.py --store outputs/experiments/E1-corpus/document_store.db --output outputs/experiments/E6 --mode both
.venv/Scripts/python.exe scripts/benchmarks/run_parser_benchmark.py --pdfs Landing --output outputs/experiments/E3 --limit 20 --mode parsers
.venv/Scripts/python.exe scripts/benchmarks/run_parser_benchmark.py --store outputs/experiments/E1-corpus/document_store.db --output outputs/experiments/E5 --limit-docs 50 --mime-type application/pdf --mode chunkers

# Retrieval eval (E10) — Tantivy vs LanceDB vs hybrid; --rerank / --where opt-in
.venv/Scripts/python.exe scripts/benchmarks/run_retrieval_eval.py --store outputs/experiments/E12-corpus/document_store.db --tantivy outputs/experiments/E12-corpus/tantivy_index --lancedb outputs/experiments/E12-corpus/vector/lancedb --output outputs/experiments/E10 --n-queries 200

# Traceability (E11)
.venv/Scripts/python.exe scripts/cli/run_fast_path.py --input Landing --output outputs/experiments/E11-corpus --trace-db outputs/experiments/E11-corpus/trace.db
.venv/Scripts/python.exe scripts/operations/run_trace_query.py --trace-db outputs/experiments/E11-corpus/trace.db --summary

# Web scraper (E4) — output ALWAYS Landing/web
.venv/Scripts/python.exe scripts/cli/run_web_scrape.py --config configs/scrape_sites.yaml --output Landing/web
.venv/Scripts/python.exe scripts/cli/run_web_scrape.py --url https://developer.nvidia.com/blog --url-pattern "^/blog/[^/]+/$" --exclude "/blog/category/" "/blog/tag/" --days-back 7
.venv/Scripts/python.exe scripts/cli/run_web_scrape.py --url https://ai.meta.com/research/ --engine playwright --url-pattern "^/blog/[^/]+/?$" --allowed-domains "research.meta.ai" --days-back 90 --no-ocr
.venv/Scripts/python.exe scripts/cli/run_web_scrape.py --config configs/scrape_sites.yaml --engine auto --output Landing/web

# Landing sweep (Transit/Archive enforcement) — also runs inside the pipeline
.venv/Scripts/python.exe scripts/operations/sweep_landing.py --dry-run

# Reporter — standalone pipeline DEPRECATED; it is an agent-invoked tool
# (compile_report). Claim/citation benchmark + deep dive/review CLIs:
.venv/Scripts/python.exe scripts/benchmarks/evaluate_reporter.py
.venv/Scripts/python.exe scripts/cli/reporter_deep_dive.py --corpus outputs/reporter/<out>/corpus --query "consulta"
.venv/Scripts/python.exe scripts/cli/reporter_review.py --db outputs/reporter/<out>/reporter.db

# LanceDB scalar metadata backfill (source_domain/published_at/provenance/quality_score)
.venv/Scripts/python.exe scripts/operations/sync_index_metadata.py --corpus outputs/experiments/E12-corpus [--all]

# Local SearXNG (preferred web-search backend for research_topic)
# Managed automatically: start_ipa_dashboard.ps1 ensures it at boot and the
# watchdog keeps it alive (starts Docker Desktop + compose up if down).
# Manual: docker compose -f .devin/searxng/docker-compose.yml up -d  # 127.0.0.1:8888
# Env: IPA_SEARXNG_URL defaults to http://127.0.0.1:8888 in watchdog/launcher;
#      IPA_SEARXNG_MANAGED=0 disables watchdog management (e.g. remote instance).

# Caches, VRAM, locks y scheduling — la fuente es EKS, no este archivo:
#   EXP-008 (caches PT/app, num_gpu, batch ExL3, MTP guard), PM-004
#   (starvation, heavy.lock, embed bulk GPU), PM-005 (purga parcial),
#   PAT-007 (leases pid|owner|ts + heartbeat: vram/heavy/tier0/job locks),
#   PAT-008 (señales Tier 0: ingest_metadata, dirty flag, novelty hints,
#   enriched_text canónico). Env vars y detalle operativo:
#   docs/USAGE.md §Caches y convivencia + docs/architecture/agent-runtime.md.
# Reglas que sí son operativas diarias:
#   Reranker pinneado a CPU (IPA_RERANK_DEVICE=cpu). Chat 423 durante bulk
#   embed GPU (>=512 backlog). Locks en outputs/agent/{vram,heavy,tier0}.lock.
#   Investigación: dedup 10 min para todos los llamados, URLs en la query se
#   scrapean como seeds, sub_queries ≤8 facetas (ver agent-runtime.md).

# Agent CLI (sessions, episodic memory, research)
.venv/Scripts/python.exe scripts/cli/agent.py chat -m "mensaje"
.venv/Scripts/python.exe scripts/cli/agent.py sessions
.venv/Scripts/python.exe scripts/cli/agent.py audit --recent 5
.venv/Scripts/python.exe scripts/cli/agent.py research "query" --llm

# Tutor CLI + contract validation + model smoke test
.venv/Scripts/python.exe scripts/cli/agent.py chat --role tutor --llm -m "mensaje"
.venv/Scripts/python.exe scripts/validation/validate_tutor_contract.py Roadmap path/to/roadmap.json
.venv/Scripts/python.exe scripts/operations/test_exl3_provider.py --all

# Dashboard (watchdog + Ollama are handled by the launcher)
.venv/Scripts/python.exe scripts/operations/web_dashboard.py --host 127.0.0.1 --port 8765
.\start_ipa_dashboard.bat            # one-click (el Orchestrator de consola está DEPRECADO)

# Compile the ExLlamaV3 native extension (only after changing ExLlamaV3,
# Python, PyTorch, CUDA or GPU)
$env:Path = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;$PWD\.venv\Scripts;$env:Path"
$env:MAX_JOBS = "2"; $env:TORCH_CUDA_ARCH_LIST = "8.9"
Push-Location exllamav3-dev/source
..\..\.venv\Scripts\python.exe setup.py build_ext --inplace
Copy-Item exllamav3_ext*.pyd ..\build\
Pop-Location

# EKS: validate + hygiene report + new record scaffold
.venv/Scripts/python.exe scripts/validation/validate_eks.py
.venv/Scripts/python.exe scripts/operations/eks_report.py
.venv/Scripts/python.exe scripts/cli/eks_new.py decision --title "..." --status proposed

# Work permits — sesiones paralelas (PAT-009). Un hook PreToolUse ya
# bloquea edits bajo permiso exclusivo ajeno e inyecta los records EKS que
# gobiernan el path; `acquire` es atómico (lockfile O_EXCL); SessionStart
# lista permisos y reporta los cerrados sin cosecha (marcador unharvested);
# SessionEnd cierra los de la sesión; Stop bloquea UNA vez pidiendo el
# closeout (`stop_hook_active` guard); PostCompaction los re-inyecta.
.venv/Scripts/python.exe scripts/cli/permit.py acquire --session <id> \
  --scope "src/ipa/agentic/**" --task "..." --type exclusive
.venv/Scripts/python.exe scripts/cli/permit.py check --scope "src/**"
.venv/Scripts/python.exe scripts/cli/permit.py close --permit PW-... --notes "..."

# Tests
.venv/Scripts/python.exe -m pytest -v
```

## Where things live

| Area | Document |
|---|---|
| Runtime + retrieval + idle + cognitive layer | `docs/architecture/agent-runtime.md` |
| Tier 0/1/2 model (leases, signals, audit) | `docs/architecture/idle-tiers.md` |
| Retrieval pipeline (hybrid, metadata, rerank gate) | `docs/architecture/retrieval.md` |
| Tutor (contracts, gate, focus, progress, lessons) | `docs/architecture/tutor.md` |
| Boundaries, dashboard, reporter, runtime map | `docs/architecture/*.md` |
| Landing / Transit / Archive + sweep rules | `docs/operations/landing-and-archive.md` |
| Roadmap, migration, horizontalization plans | `docs/plans/*.md` |
| Research roadmap (stages 0-7) | `docs/development/research-roadmap.md` |
| Decisions, experiments, benchmarks, patterns | `knowledge/` (EKS) |
| Contracts (authority) | `contracts/*.schema.json` |
| Changes per version | `CHANGELOG.md` |

## Architecture (src layout)

```text
src/ipa/
  contracts.py       Runtime dataclasses aligned with the JSON Schemas
  ingestion/         Landing, MIME, safety, parsers, chunking, FastPath
  storage/           DocumentStore and canonical persistence
  indexes/           FTS5/Tantivy/LanceDB/sqlite-vec/embedding/reranker
  acquisition/       Web fetch, scraper and OCR adapters
  enrichment/        Derived summaries, queries and claims
  observability/     TraceLog and structured event helpers
  agentic/           QueryIR, evidence, bounded retrieval, idle scheduler
  reporter/          Reporter metadata, curation, topics, reports, promotion
  tutor/             Learning contracts and pedagogy runtime
  providers/         ExLlama/Ollama provider adapters
  mcp/               Thin MCP proxy to the dashboard tool registry (external clients only)
  dashboard/         Dashboard server and orchestration
```

Each bounded context has one responsibility. Retrieval, ranking, context
building, generation and orchestration stay separate. Parsers never import
embedding or enrichment implementations. The dashboard is a control room, not
a second domain runtime: state lives in the core (`outputs/agent/*`).

## Conventions

- Contracts in `contracts/` are the authority; tools integrate via adapters.
  `contract_vocabulary.json` defines required fields per record;
  `experiment_report.schema.json` the machine-readable report format.
- Tutor schemas define LearningGoal, Concept, Roadmap, AssessmentResult and
  ResearchRequest; `tutor_common.schema.json` holds shared provenance and
  approval definitions. `validation_contract.md` defines the validation
  discipline.
- All outputs go under `outputs/`; each experiment uses `outputs/experiments/E#/`.
- Manifests are append-safe: each run writes a timestamped file.
- Synthetic fixtures live in `data/sample/input/`; everything under `input/`
  is treated as an artifact.
- DEC-* records in `knowledge/decisions/` are the project's ADR format
  (DEC-008); no separate `docs/adr/`. New knowledge: `scripts/cli/eks_new.py`.
- EKS frontmatter supports `affects` (path globs a record governs — feeds
  `eks_governing`/permits), `evidence` (paths that must exist for
  `accepted` records created since 2026-09-23), `author_model` and
  `trigger` (`permit:PW-*` when produced under a work permit). See
  `knowledge/_schema/metadata.md`.
- `components` uses the controlled vocabulary in
  `knowledge/_schema/components.json`, which also defines `groups`
  (`indexes` → lexical/vector, `ingestion` → fast_path/parsing/chunking/
  landing_zone/acquisition, `memory` → strategic/user_model/skills/
  uncertainty): filtering by a group reaches records tagged with any
  member and vice versa. Tag the specific component when one exists.
- `eks_report` exposes `hot_zones` (exact glob) and `hot_zones_overlap`
  (prefix criterion — the same one that decides a permit's precautions).
- Parallel Devin sessions: acquire a work permit covering your scope
  before editing (`permit.py acquire`); an exclusive overlap means stop
  and tell the user. On close, record what the session learned via
  `eks_new` (draft) and pass it as `--eks-draft` to `permit.py close`.
- Tests must validate real behavior, not just file existence.

## Cleanup rules (DO NOT violate)

- **Never delete `Landing/` while it has unprocessed files** — they are not
  ingested yet. Only clean files already in `Archive/`.
- **Never delete `Archive/` or `Transit/`** — already processed source material.
- **Never delete `Landing/web/scrape_history.db`** — it prevents re-downloading.
  It is the ONLY db allowed under `Landing/`; `Landing/scrape_history.db` and
  `Landing/landing_zone.db` are old-run residuals and should be deleted.
- **Scraper output is always `--output Landing/web`** (never `Landing`), so all
  content lives under `Landing/web/<site>/` with a single history db.
- **Only delete `outputs/experiments/E12-corpus/`** to reprocess from scratch
  (re-parse, re-chunk, re-index); `Archive/` and `Landing/` stay untouched.
- **Promotion is vector-gated** (PM-004): `promote_documents_to_main` purges
  the staging copy only when every live batch chunk already has a vector in
  main LanceDB; otherwise it defers — source intact, `promotion_queue` entry
  stays pending, next cycle retries. Opt-out: `IPA_PROMOTION_REQUIRE_VECTORS=0`.
- Derived stores (`topic_clusters.db`, `memory_vectors.db`,
  `research_review.db`, LanceDB) are rebuildable — deleting them loses no
  authority, only time.
- Full rules: `docs/operations/landing-and-archive.md`.

## Tests

`.venv/Scripts/python.exe -m pytest -v` — currently **1146 passed, 1 skipped**
(network test; run it with `--run-network-tests`). `tests/conftest.py` sets
`IPA_RERANK=0` for the whole suite so the cross-encoder never loads from the
default. Test files are named after the capability they cover; use
`pytest --collect-only` for the current inventory instead of a static list.
