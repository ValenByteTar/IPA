"""Process state read/write helpers for orchestrator jobs.

Provides a common JSON state schema and atomic write semantics so that
all JobSpec/JobRunner jobs produce dashboard-readable state files with
a consistent shape.

State schema:
    {
        "process":   str,          # job name
        "run_id":    str | None,   # IPA_RUN_ID when set
        "status":    str,          # running|idle|stuck|paused|done|error
        "pid":       int,          # os.getpid() of the runner process
        "timestamp": float,        # time.time() at last write
        "metrics":   dict,         # job-specific parsed metrics
        "errors":    list[str],    # recent error lines (last 10)
    }
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

# Default corpus state directory; callers may override per-spec.
DEFAULT_CORPUS = Path("outputs/experiments/E12-corpus")
DEFAULT_STATE_DIR = DEFAULT_CORPUS / "process_state"

# Status values the dashboard understands.
STATUS_RUNNING = "running"
STATUS_IDLE = "idle"
STATUS_STUCK = "stuck"
STATUS_PAUSED = "paused"
STATUS_DONE = "done"
STATUS_ERROR = "error"

_VALID_STATUSES = frozenset({
    STATUS_RUNNING, STATUS_IDLE, STATUS_STUCK,
    STATUS_PAUSED, STATUS_DONE, STATUS_ERROR,
})

MAX_ERRORS = 10


def state_path(state_dir: Path, job_name: str) -> Path:
    """Return the canonical state file path for a job."""
    return Path(state_dir) / f"{job_name}.json"


def build_state(
    job_name: str,
    status: str,
    metrics: dict[str, Any],
    errors: list[str] | None = None,
    run_id: str | None = None,
    pid: int | None = None,
) -> dict[str, Any]:
    """Build a state dict following the common schema."""
    if status not in _VALID_STATUSES:
        raise ValueError(f"Invalid status: {status!r}")
    return {
        "process": job_name,
        "run_id": run_id if run_id is not None else os.environ.get("IPA_RUN_ID"),
        "status": status,
        "pid": pid if pid is not None else os.getpid(),
        "timestamp": time.time(),
        "metrics": metrics,
        "errors": (errors or [])[-MAX_ERRORS:],
    }


def write_state_atomic(
    state_dir: Path,
    job_name: str,
    status: str,
    metrics: dict[str, Any],
    errors: list[str] | None = None,
    run_id: str | None = None,
    pid: int | None = None,
) -> None:
    """Atomically write state JSON using tempfile + os.replace.

    Concurrent readers (dashboard/orchestrator) see either the previous
    complete state or the new complete state, never partial JSON.
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    state = build_state(job_name, status, metrics, errors, run_id, pid)
    target = state_path(state_dir, job_name)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{job_name}-", suffix=".json.tmp", dir=str(state_dir),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(5):
            try:
                os.replace(tmp_name, target)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.25 * (attempt + 1))
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


def write_state_simple(
    state_dir: Path,
    job_name: str,
    status: str,
    metrics: dict[str, Any],
    errors: list[str] | None = None,
    run_id: str | None = None,
    pid: int | None = None,
) -> None:
    """Non-atomic state write (for jobs that don't need atomicity).

    Equivalent to write_state_atomic but without tempfile dance.
    Kept for parity with legacy proc_hammer/proc_rechunk/proc_scraper
    which used direct write_text.
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    state = build_state(job_name, status, metrics, errors, run_id, pid)
    state_path(state_dir, job_name).write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def read_state(state_dir: Path, job_name: str) -> dict[str, Any] | None:
    """Read and parse a state file. Returns None if missing or invalid."""
    path = state_path(state_dir, job_name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
