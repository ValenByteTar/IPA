"""Lease Tier 0 — ingesta de corpus (fast path).

Tier 0 cubre la ingesta de artefactos al corpus (parse + chunk + BM25 +
drenaje de embeddings). Mientras el lease está vivo, el scheduler idle no
inicia ciclos Tier 1/T2: esos tiers leen y mutan los mismos stores
(document_store.db, bm25_index.db, topic_clusters.db, promotion_queue) y
correrlos contra una ingesta en vuelo produce evaluaciones sobre estados
parciales y contienda de escritura.

El lease es cross-process (archivo ``pid|owner|ts`` + heartbeat): el watcher
de fast_path puede sobrevivir a su orquestador (restart del dashboard), así
que el gate no puede depender del árbol de procesos del dashboard. La cuenta
de idle para T1/T2 arranca recién cuando el lease se libera: mientras está
activo, ``_idle()`` devuelve False y el scheduler resetea LAST_ACTIVITY.

Archivo: outputs/agent/tier0.lock — mismo formato que vram.lock.
"""
from __future__ import annotations

import atexit
import os
import threading
import time
from pathlib import Path
from typing import Any

from ipa.providers.vram_lock import pid_alive

ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = Path(os.environ.get(
    "IPA_TIER0_LOCK", str(ROOT / "outputs" / "agent" / "tier0.lock")))
# TTL holgado: el heartbeat corre cada HEARTBEAT_S; el TTL solo cubre
# stalls largos del proceso (GC, IO) y el kill duro del holder.
TTL_S = float(os.environ.get("IPA_TIER0_LOCK_TTL", "300") or 300)
HEARTBEAT_S = float(os.environ.get("IPA_TIER0_HEARTBEAT", "15") or 15)


def holder() -> dict[str, Any] | None:
    """Dueño vivo del lease Tier 0, o None (un holder stale se limpia)."""
    try:
        parts = LOCK_PATH.read_text(encoding="utf-8").strip().split("|")
        pid, owner, ts = int(parts[0]), parts[1], float(parts[2])
    except Exception:
        return None
    if time.time() - ts > TTL_S or not pid_alive(pid):
        try:
            LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    return {"pid": pid, "owner": owner, "ts": ts}


def active() -> bool:
    """True mientras una ingesta Tier 0 esté viva en cualquier proceso."""
    return holder() is not None


def _write(owner: str) -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LOCK_PATH.with_name(f"{LOCK_PATH.name}.{os.getpid()}.tmp")
    tmp.write_text(f"{os.getpid()}|{owner}|{time.time()}", encoding="utf-8")
    tmp.replace(LOCK_PATH)


def claim(owner: str = "fast_path") -> bool:
    """Toma el lease si está libre o stale. Idempotente para el mismo PID."""
    current = holder()
    if current is not None and current["pid"] != os.getpid():
        return False
    try:
        _write(owner)
    except OSError:
        return False
    atexit.register(_shutdown)
    return True


def renew(owner: str = "fast_path") -> bool:
    """Refresca el lease si seguimos siendo el dueño."""
    current = holder()
    if current is not None and current["pid"] != os.getpid():
        return False
    try:
        _write(owner)
    except OSError:
        return False
    return True


def release() -> None:
    """Libera el lease solo si es nuestro — nunca el de otro proceso."""
    current = holder()
    if current is not None and current["pid"] != os.getpid():
        return
    try:
        LOCK_PATH.unlink(missing_ok=True)
    except OSError:
        pass


_heartbeat_stop: threading.Event | None = None


def start_heartbeat(owner: str = "fast_path") -> None:
    """Daemon que renueva el lease mientras el proceso vive.

    La ingesta puede quedar bloqueada en un solo paso largo (drain final de
    embeddings), así que el renewal no puede depender del loop principal.
    """
    global _heartbeat_stop
    if _heartbeat_stop is not None:
        return
    stop = threading.Event()
    _heartbeat_stop = stop

    def _beat() -> None:
        while not stop.wait(HEARTBEAT_S):
            try:
                renew(owner)
            except Exception:
                pass

    threading.Thread(target=_beat, daemon=True, name="tier0-heartbeat").start()


def stop_heartbeat() -> None:
    global _heartbeat_stop
    if _heartbeat_stop is not None:
        _heartbeat_stop.set()
        _heartbeat_stop = None


def _shutdown() -> None:
    """atexit: corta el heartbeat antes de liberar (evita re-write post-release)."""
    stop_heartbeat()
    release()


__all__ = [
    "LOCK_PATH", "TTL_S", "HEARTBEAT_S", "holder", "active",
    "claim", "renew", "release", "start_heartbeat", "stop_heartbeat",
]
