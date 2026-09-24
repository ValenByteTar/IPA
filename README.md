# IPA

IPA is a local-first knowledge acquisition and materialization platform for agents.
It turns heterogeneous sources into canonical, provenance-preserving documents and
chunks, then publishes derived lexical, vector, enrichment, and reporting views.

IPA is not a single chatbot and it is not only a vector database. Its core job is
to make knowledge available durably before expensive interpretation is performed.

```text
sources
  -> Landing Zone
  -> MIME routing and safe parsing
  -> CanonicalDocument
  -> deterministic chunks and provenance
  -> DocumentStore (canonical source of truth)
  -> lexical index
  -> optional embeddings/vector index
  -> optional enrichment
  -> Reporter / Agent Runtime / Tutor
```

## Current status

**v0.2.1.** IPA is a functional local-first personal
agent platform: contract-first Hybrid RAG, a shared agent core (CLI + dashboard),
a Tutor role with human approval gates, bounded web research, an idle cognitive
layer, and an idle scheduler with an explicit resource model. It is intentionally
bounded: single user, single machine, bounded tools per turn. See `CHANGELOG.md`
for what ships in this version.

The fast ingestion path, canonical SQLite store, Tantivy/LanceDB adapters, web
acquisition, OCR, Reporter, Tutor contracts, observability, and EKS development
memory are implemented at different maturity levels. The status of any capability
must be verified through its tests and experiment evidence; planned designs are
not presented as production guarantees.

## Hardware: GPU or 100% CPU

IPA runs with or without an NVIDIA GPU:

- **GPU present** — the star model runs on CUDA (Ollama or ExL3), OCR and
  Docling use GPU acceleration.
- **No CUDA-capable GPU** — chat/Tutor fall back to Ollama on CPU, OCR/Docling
  resolve to `cpu`, and `EmbeddingAdapter` resolves `auto` to CPU. Nothing fails
  at boot for lack of a GPU.
- `IPA_FORCE_CPU=1` disables GPU detection in the provider factory; it is not a
  process-wide PyTorch device switch. To pin BGE embeddings/reranker explicitly,
  use `IPA_EMBED_DEVICE=cpu` / `IPA_RERANK_DEVICE=cpu` (see `docs/USAGE.md`).

## Architecture

```text
                    +-----------------------------+
                    |       Developer / Agent     |
                    +---------------+-------------+
                                    |
                 +------------------+------------------+
                 |                                     |
           EKS dev-time                         IPA runtime MCP
        engineering memory                  corpus and jobs interface
                 |                                     |
                 +------------------+------------------+
                                    v
             +-------------------------------------------+
             |      Knowledge materialization plane      |
             | acquisition -> parsing -> store -> views |
             +-------------------+-----------------------+
                                 |
       +-------------------------+-------------------------+
       v                         v                         v
 DocumentStore             Lexical views              Vector views
 canonical data             Tantivy / FTS5             LanceDB
                                 |
                                 v
                 Reporter / Agent Runtime / Tutor
```

The JSON Schemas under `contracts/` are authoritative. Implementations and
external tools must adapt to those contracts rather than redefining them.

## Quickstart

Requirements: Python 3.12.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,retrieval,parsers,web,mcp]"
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\cli\run_fast_path.py --input data/sample/input --output outputs\smoke
```

The core smoke path uses synthetic fixtures and does not require a GPU, a cloud
service, a real corpus, or a local LLM.

Capabilities beyond the core need external dependencies: **Docker Desktop**
(local SearXNG, the real web-search backend — auto-started by the launcher and
watchdog), **Ollama** (chat/Tutor LLM backend), **Playwright browsers** (JS
scraping), and optionally **ExLlamaV3** (GPU star model). See
`docs/USAGE.md` → "Dependencias externas" for install steps.

## Optional profiles

Install only what a workload needs:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[retrieval]"
.venv\Scripts\python.exe -m pip install -e ".[web]"
.venv\Scripts\python.exe -m pip install -e ".[tutor]"
```

The Tutor profile requires a local CUDA-compatible setup and the compiled
ExLlamaV3 extension. Model weights are intentionally not part of this repository.

## Repository layout

```text
src/ipa/       public Python package and bounded contexts
contracts/     authoritative JSON Schemas and contract vocabulary
configs/       reproducible configuration profiles
data/sample/   small synthetic fixtures
tests/         contract, unit, integration and capability tests
scripts/       thin CLI, benchmark, validation and operations entrypoints
tools/         development-only tools, including EKS MCP and work permits
knowledge/     EKS engineering memory (dev-time only)
docs/          architecture, policies, plans and operations
web/           local dashboard assets
.devin/        agent session config: hooks (permit guard), rules, skills, SearXNG compose
```

Local corpus, models, indexes, databases, caches, and generated reports are not
committed. See `public-surface-manifest.json` and `.gitignore`. Pinned
dependencies for reproducible installs live in `requirements.lock`
(`requirements.txt` keeps the curated ranges).

## Dashboard and orchestrator

IPA ships a local web dashboard for corpus inspection, curation, and job
control. The console orchestrator (scraper → fast_path → lancedb → hammer →
enrichment chain) is **deprecated**: jobs are launched from the dashboard and
background LLM work runs through the idle scheduler.

```powershell
# Web dashboard (HTTP + dynamic refresh + curation actions)
.venv\Scripts\python.exe scripts\operations\web_dashboard.py --host 127.0.0.1 --port 8765

# One-click launcher: starts dashboard and opens browser
.\start_ipa_dashboard.bat

# DEPRECATED — console orchestrator (kept for compatibility only)
.venv\Scripts\python.exe scripts\operations\orchestrator.py
```

The orchestrator launches jobs through a shared `JobSpec` / `JobRunner` model
under `src/ipa/dashboard/`, writing process state to
`outputs/experiments/E12-corpus/process_state/` that both the dashboard and the
orchestrator read for health and progress display.

## Engineering knowledge

The Engineering Knowledge System under `knowledge/` stores decisions, experiments,
benchmarks, patterns, postmortems, and research produced while evolving IPA. It
is not the runtime Knowledge System and it is not used to answer corpus queries.
The `ipa-eks` MCP server is read-only and dev-time.

Records are validated (`scripts/validation/validate_eks.py`), carry lifecycle
metadata (`supersedes`/`superseded_by`, evidence paths, author model) and can
declare `affects` globs — the files they govern — so `eks_governing` surfaces
the decisions that apply to a path before it is edited.

Parallel agent sessions coordinate through dev-time work permits
(`scripts/cli/permit.py`, PAT-009): a session acquires an exclusive scope,
receives the governing EKS records as precautions, and a pre-edit hook blocks
foreign sessions from editing inside it. Permit state is ephemeral
(`outputs/devin/permits/`, not committed).

Before a significant change, build engineering context. After a reproducible
experiment, record the result in EKS. Architectural decisions remain subject to
validation and human approval.

## Safety and data policy

- Never place secrets, personal data, or an unauthorized corpus in fixtures.
- Keep source artifacts immutable and preserve provenance.
- Do not delete `Landing/` with unprocessed files.
- Do not delete `Archive/` as routine cleanup.
- Generated indexes and reports are derived and reconstructible.
- External downloads and scraping must use explicit allowlists and bounded jobs.

## Development commands

See `AGENTS.md` for the complete command reference and `docs/USAGE.md` for the
single-page install/usage guide (profiles, dashboard, chat/Tutor, tools,
environment variables, CPU fallback). The principal gates are:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\validation\validate_eks.py knowledge
.venv\Scripts\python.exe scripts\validation\validate_contracts.py <manifest> --integrity
```

## License

MIT. See `LICENSE`.
