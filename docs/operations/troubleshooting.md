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

## Locks y leases (PAT-007)

All coordination files share the same format — `pid|owner|timestamp` — and
the same stale semantics: a holder is stale when its timestamp exceeds the
TTL **or** its pid is dead. `holder()` self-heals by unlinking the file on
the next read, so a "stuck lock" is almost always a *live* holder doing real
work — check the pid before touching anything.

| Lock | Path | Owner examples | TTL (env override) |
|---|---|---|---|
| Tier 0 | `outputs/agent/tier0.lock` | `fast_path` | 300 s (`IPA_TIER0_LOCK_TTL`), heartbeat 15 s |
| Heavy | `outputs/agent/heavy.lock` | `pipeline`, `research_ingest`, `embed_bulk_gpu` | 1800 s (`IPA_HEAVY_LOCK_TTL`), wait cap 300 s (`IPA_HEAVY_LOCK_WAIT`) |
| VRAM | `outputs/agent/vram.lock` | `embed_bulk_gpu`, `exl3_swap` | pid-alive based (`IPA_VRAM_LOCK`) |
| Maintenance job | `outputs/web_dashboard/embedding_maintenance.lock` | `embedding_drain`, `research_embed` | 6 h (`IPA_EMBED_JOB_LOCK_TTL`), renewed while running |

What each one does:

- **tier0.lock** — held while a Tier-0 ingestion (fast path) is alive in any
  process. The idle scheduler refuses to start T1/T2 cycles while it exists
  (they read/mutate the same stores mid-ingestion). `_idle()` returns False
  and the idle clock resets when it releases.
- **heavy.lock** — serializes heavy writers (`heavy_phase(kind, priority)`).
  Interactive work (`research_ingest`, priority interactive) preempts the
  queue order over background jobs. `heavy.lock.waiting` lists queued
  waiters; `research_progress.json.heavy_wait` exposes "waiting on whom"
  to the dashboard.
- **vram.lock** — exclusive GPU access for model transitions (embedding
  bulk GPU, ExL3 swaps). While held, the embedding maintenance state is
  `waiting_for_vram`/`loading_bge` and chat stays blocked.
- **embedding_maintenance.lock** — the maintenance *job* lease: only one
  corpus/index mutation producer at a time (drain, bulk GPU batch). Long
  TTL (6 h) because drains legitimately run for hours on CPU; the holder
  renews it while alive. State file: `embedding_maintenance.json`
  (`status`, `phase`, `vectorized/total_chunks`, `chat_blocked`).

Diagnosing a "stuck" lock:

```powershell
# Who holds it? (empty output = free; the file is deleted on read if stale)
Get-Content outputs\agent\heavy.lock
# Is that pid alive?
Get-Process -Id <pid> -ErrorAction SilentlyContinue
```

- `pid` alive → the lock is real. Wait, or cancel the holder's job.
- `pid` dead but file present → it cleans itself on the next `holder()`
  call (every gate calls it). If a reader path shows it stuck anyway,
  delete the file manually — safe only after confirming the pid is dead.
- `chat_blocked: true` with no maintenance lock → stale state file, not a
  lock: the drain died after claiming the lease but before releasing it.
  `embedding_maintenance.json` can be reset by rerunning
  `run_embed_drain.py` (it rewrites state) or by editing `chat_blocked`
  to `false` once you confirm no drain process is running.

Trampoline gotcha (Windows): `.venv\Scripts\pythonw.exe` is a *trampoline*
that spawns the real interpreter (`Python312\pythonw.exe`). Process scans
show every pythonw process TWICE — trampoline parent + real child with the
same command line. This is normal, not duplication: killing the "extra"
(system-python) process kills the real worker while the trampoline parent
survives as a zombie. Before killing anything, check `ParentProcessId` —
the trampoline is the parent, the real process owns the port. The same
applies to `dashboard_watchdog.py` (see its header comment).

Restarting the dashboard: prefer `POST /api/restart` or killing the pid
that owns port 8765 — the watchdog respawns it. Killing by name
(`taskkill /IM pythonw.exe`) takes down Ollama runners and unrelated
python processes too.

Symptom → cause map:

- Chat input disabled, banner "Chat pausado" → maintenance job active
  (embedding drain / bulk GPU). Normal; ends when the backlog drains.
- Research "running" for minutes with no progress → check
  `research_progress.json`: `phase` says where it is, `heavy_wait.blocked_by`
  says who holds heavy.lock.
- T1/T2 idle tasks never fire → `tier0.lock` held (ingestion alive) or
  `_idle()` false (recent activity); check the file, not the scheduler.
- Two LanceDB writers / corruption risk → should be impossible: heavy.lock
  and the maintenance job lease are exactly the guards against it. If it
  happens, one of the writers bypassed `heavy_phase`/`claim_job` — that is
  a bug, report it.
