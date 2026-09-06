# Troubleshooting

## Tests fail during collection

Use Python 3.12 and install the development profile:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Vector/parser/web tests fail because an optional dependency is missing

Install only the profile for the capability being tested. Core ingestion should
not require GPU, web browser, vector database or LLM dependencies at import time.

## EXL3 cannot load

Check the optional Tutor profile, CUDA version, the locally compiled extension,
GPU architecture and model path. Model weights and `exllamav3-dev` are intentionally
externalized from GitHub.

## Pipeline state looks stale

Inspect the run-specific logs, heartbeat/state files and TraceLog before restarting.
Do not delete databases or Archive entries to hide a failure.
