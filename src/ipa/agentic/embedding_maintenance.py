"""Persisted control-room state for exclusive GPU embedding maintenance.

The embedding worker is a separate process/thread from the dashboard. This
small state file lets the dashboard disable chat, explain why, and recover the
indicator if the dashboard process restarts. LanceDB remains the durable source
of progress; the file is status only, never the embedding checkpoint.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
STATE_PATH = Path(os.environ.get(
    "IPA_EMBED_MAINTENANCE_STATE",
    str(ROOT / "outputs" / "web_dashboard" / "embedding_maintenance.json"),
))
JOB_LOCK_PATH = STATE_PATH.with_suffix(".lock")
JOB_LOCK_TTL_S = float(os.environ.get("IPA_EMBED_JOB_LOCK_TTL", "21600") or 21600)
GPU_MIN_BACKLOG = int(os.environ.get("IPA_EMBED_GPU_MIN_BACKLOG", "512") or 512)
ACTIVE_STATUSES = frozenset({
    "preparing", "waiting_for_vram", "unloading_llm", "loading_bge",
    "embedding", "restoring_chat",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def job_holder() -> dict[str, Any] | None:
    """Current corpus/index maintenance owner, or None/stale."""
    try:
        raw = JOB_LOCK_PATH.read_text(encoding="utf-8").strip()
        parts = raw.split("|")
        if len(parts) != 3:
            raise ValueError("invalid job lock")
        pid, owner, ts = int(parts[0]), parts[1], float(parts[2])
    except (OSError, ValueError):
        try:
            if (JOB_LOCK_PATH.exists()
                    and time.time() - JOB_LOCK_PATH.stat().st_mtime > 1.0):
                JOB_LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    from ipa.providers.vram_lock import pid_alive
    if time.time() - ts > JOB_LOCK_TTL_S or not pid_alive(pid):
        try:
            JOB_LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    return {"pid": pid, "owner": owner, "ts": ts}


def job_active(*, exclude_owner: str | None = None) -> bool:
    holder = job_holder()
    return holder is not None and holder["owner"] != exclude_owner


def claim_job(owner: str = "embedding_drain", *, wait_s: float = 0.0,
              cancel: Any | None = None) -> bool:
    """Atomically claim corpus/index mutation; optionally wait for current owner."""
    JOB_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.0, wait_s)
    while True:
        try:
            fd = os.open(str(JOB_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = job_holder()
            if holder is None:
                continue
            if wait_s <= 0 or (cancel is not None and cancel.is_set()):
                return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.25)
            continue
        try:
            os.write(fd, f"{os.getpid()}|{owner}|{time.time()}".encode("utf-8"))
        finally:
            os.close(fd)
        return True


def renew_job(owner: str) -> bool:
    holder = job_holder()
    if holder is None or holder["pid"] != os.getpid() or holder["owner"] != owner:
        return False
    temp = JOB_LOCK_PATH.with_name(f"{JOB_LOCK_PATH.name}.{os.getpid()}.tmp")
    temp.write_text(f"{os.getpid()}|{owner}|{time.time()}", encoding="utf-8")
    temp.replace(JOB_LOCK_PATH)
    return True


def release_job(owner: str = "embedding_drain") -> None:
    holder = job_holder()
    if (holder is not None and holder["pid"] == os.getpid()
            and holder["owner"] == owner):
        try:
            JOB_LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass


def update_state(**fields: Any) -> dict[str, Any]:
    """Atomically merge a worker update into the persisted status."""
    current: dict[str, Any] = {}
    try:
        current = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    current.update(fields)
    current["pid"] = os.getpid()
    current["updated_at"] = _now()
    _write_json(STATE_PATH, current)
    return current


def start_state(*, corpus: str, total_chunks: int, vectorized: int,
                pending: int, mode: str) -> dict[str, Any]:
    return update_state(
        run_id=f"embed:{os.getpid()}:{time.time_ns()}",
        corpus=corpus,
        status="preparing",
        phase="preparing",
        mode=mode,
        total_chunks=total_chunks,
        vectorized_before=vectorized,
        pending_initial=pending,
        embedded=0,
        vectorized=vectorized,
        pending=pending,
        chunks_per_second=0.0,
        eta_seconds=None,
        blocked_by=None,
        chat_blocked=True,
        error=None,
        warning=None,
        started_at=_now(),
        finished_at=None,
    )


def read_state() -> dict[str, Any]:
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "idle", "chat_blocked": False}
    if not isinstance(state, dict):
        return {"status": "idle", "chat_blocked": False}
    status = str(state.get("status", "idle"))
    blocked = status in ACTIVE_STATUSES
    if blocked:
        from ipa.providers.vram_lock import pid_alive
        pid = int(state.get("pid") or 0)
        if not pid_alive(pid):
            state.update({
                "status": "interrupted",
                "phase": "interrupted",
                "chat_blocked": False,
                "error": "El proceso de embeddings terminó inesperadamente; se puede reanudar desde LanceDB.",
                "finished_at": _now(),
                "updated_at": _now(),
            })
            try:
                _write_json(STATE_PATH, state)
            except OSError:
                pass
            blocked = False
    state["chat_blocked"] = blocked
    return state


def chat_block_reason() -> str | None:
    state = read_state()
    if not state.get("chat_blocked"):
        return None
    return (
        "Chat temporalmente no disponible: IPA está completando una ingesta masiva "
        "de embeddings en la GPU. Se reactivará al terminar."
    )
