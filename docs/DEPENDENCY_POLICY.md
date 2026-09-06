# Dependency policy

Core dependencies are intentionally small. Optional tools must be installed
per experiment, pinned in a lock/export file, and reported with their version.

## Required discipline

- Prefer mature releases; do not use floating `latest` in experiment records.
- Keep optional imports lazy.
- Record license and deployment mode.
- Never put credentials in configs or reports.
- Do not add a service dependency when a local implementation is adequate.
- Benchmark a tool before making it preferred.
- Preserve an adapter boundary so the tool remains replaceable.

## Suggested install profiles

```powershell
# core
.venv\Scripts\python.exe -m pip install -r requirements.txt

# PDF/parser experiments
.venv\Scripts\python.exe -m pip install "-e .[embeddings]"

# vector competition
.venv\Scripts\python.exe -m pip install "-e .[vector]"

# observability
.venv\Scripts\python.exe -m pip install "-e .[observability]"
```

Do not install every optional candidate before a competition requires it.
