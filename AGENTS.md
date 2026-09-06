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

# Reporter Agent — controlled claim/citation evaluation
.venv\Scripts\python.exe scripts\benchmarks\evaluate_reporter.py

# Reporter Agent — generate a domain-agnostic periodic report in an isolated corpus
.venv\Scripts\python.exe scripts\cli\run_reporter.py --input Landing/web --config configs/reporter.yaml --output outputs/reporter/default/2026-08
.venv\Scripts\python.exe scripts\cli\run_reporter.py --input data/sample/input --config configs/reporter.yaml --output outputs/reporter/smoke/2026-08
.venv\Scripts\python.exe scripts\validation\validate_reporter_contract.py ReporterReport outputs\reporter\default\2026-08\report.json
.venv\Scripts\python.exe scripts\cli\reporter_deep_dive.py --corpus outputs\reporter\default\2026-08\corpus --query "consulta"
.venv\Scripts\python.exe scripts\cli\reporter_review.py --db outputs\reporter\default\2026-08\reporter.db

# Tutor Agent — validate a contract record
.venv\Scripts\python.exe scripts\validation\validate_tutor_contract.py LearningGoal path\to\goal.json
.venv\Scripts\python.exe scripts\validation\validate_tutor_contract.py Roadmap path\to\roadmap.json

# Tutor Agent — smoke test del modelo estrella (Qwen3.5-9B EXL3 3.0bpw + MTP)
.venv\Scripts\python.exe scripts\operations\test_exl3_provider.py
.venv\Scripts\python.exe scripts\operations\test_exl3_provider.py --interactive
.venv\Scripts\python.exe scripts\operations\test_exl3_provider.py --all

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
- Reporter modules — `reporter_config.py`, `reporter_metadata.py`, `reporter_representation.py`, `reporter_curation.py`, `reporter_topics.py`, `reporter_claims.py`, `reporter_store.py`, `reporter_report.py`, `reporter_pipeline.py`, `reporter_deep_dive.py`, `reporter_research.py` and `reporter_promotion.py`; isolated periodic reports under `outputs/reporter/`.
- Reporter curation automation — `reporter_curation.py` computes a weighted `promotion_score` (relevance 0.30, novelty 0.20, source_quality 0.20, impact 0.15, depth 0.10, actionability 0.05) and applies a conservative auto-promotion policy: a document is auto-approved only when it is a `promote` decision, has no duplicate, is in-period, has text, and meets `relevance>=0.80`, `novelty>=0.60`, `source_quality>=0.65`, `impact>=0.60` and `promotion_score>=0.78`. Auto-approved decisions are recorded with `decided_by=auto-promotion-v1`, `review_status=approved` and a note with the score; ambiguous cases remain `review_status=pending` for human review. The physical copy to the main corpus is still gated by the final report approval, so auto-promotion does not bypass that control boundary. Curation uses BGE-M3 embeddings from LanceDB (document centroids) for semantic relevance and novelty scoring — no LLM inference needed for classification. Source quality, impact, depth and actionability use improved heuristics (domain trust list, entity density, structure detection, action verb matching).
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

421 tests across 21 files:
- `test_adaptive_chunker.py` (18) — adaptive merge behavior and invariants
- `test_agentic_runtime.py` (23) — QueryIR, EvidenceSet, ContextPackage and budget contracts
- `test_chunkers_alt.py` (17) — recursive, token and semantic chunkers
- `test_contract_vocabulary.py` (5) — contract authority and identity fields
- `test_corpus_service.py` (1) — Reporter corpus boundary service
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
- `test_retrieval_eval.py` (28) — IR metrics, query generation and hybrid fusion
- `test_trace_log.py` (21) — observability and FastPath trace integration
- `test_tutor_contracts.py` (35) — Tutor schemas, provenance, approvals,
  integrity invariants and runtime dataclasses
- `test_web_dashboard.py` (8) — dashboard paths, sources and URL security
- `test_web_scraper.py` (65) — scraping, downloads, RSS, OCR and Playwright
