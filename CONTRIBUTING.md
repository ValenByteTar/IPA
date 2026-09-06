# Contributing to IPA

## Before changing code

1. Read `AGENTS.md`.
2. Build engineering context from `knowledge/`, contracts and relevant docs.
3. Identify the owning bounded context.
4. Preserve contract authority and provenance.
5. Define a test or benchmark that demonstrates the behavior.

## Development

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev,retrieval,parsers,web,mcp]"
.venv\Scripts\python.exe -m pytest -q
```

Optional dependencies must be installed only for the workload that needs them.
Do not add a service or framework when a local adapter is sufficient.

## Architecture rules

- Retrieval, ranking, context building, generation and orchestration remain separate.
- `DocumentStore` is the canonical source for documents and chunks.
- Indexes and enrichments are derived and rebuildable.
- Contracts under `contracts/` are authoritative.
- EKS is development-time memory, not runtime corpus knowledge.
- New ADRs require implemented and validated evidence.

## Experiments

Every benchmark must identify its corpus/manifest, configuration fingerprint,
hardware, versions, outputs and failure state. Store conclusions in EKS and keep
large generated artifacts outside the public repository.

## Pull requests

A change should explain:

- problem and owning context;
- compatibility impact;
- tests and benchmarks;
- dependency/profile impact;
- data/provenance implications;
- rollback path.
