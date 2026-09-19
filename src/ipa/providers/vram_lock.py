"""Lock de VRAM entre motores (ExL3 ↔ Ollama).

En una GPU chica (6 GB) los dos motores no conviven: si Ollama recarga su
modelo mientras ExL3 corre, ExL3 muere con OOM (visto en rep_pen.cu). El lock
serializa el uso:

  - ExL3.load() adquiere el lock (y antes descarga los modelos de Ollama).
  - OllamaProvider.load() falla con un error claro si otro proceso lo tiene.
  - ExL3.unload() lo libera.
  - Un lock de un proceso muerto o viejo (> TTL) se roba solo: un crash no
    bloquea el sistema para siempre.

Archivo: outputs/agent/vram.lock (pid|owner|timestamp).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = Path(os.environ.get(
    "IPA_VRAM_LOCK", str(ROOT / "outputs" / "agent" / "vram.lock")))
DEFAULT_TTL_S = float(os.environ.get("IPA_VRAM_LOCK_TTL", "1800") or 1800)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import subprocess
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            return str(pid) in out
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def holder() -> dict | None:
    """Devuelve el lock vigente {pid, owner, ts} o None (libre/stale)."""
    try:
        raw = LOCK_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    parts = raw.split("|")
    if len(parts) != 3:
        return None
    try:
        pid, owner, ts = int(parts[0]), parts[1], float(parts[2])
    except ValueError:
        return None
    if time.time() - ts > DEFAULT_TTL_S or not _pid_alive(pid):
        try:
            LOCK_PATH.unlink()
        except Exception:
            pass
        return None
    return {"pid": pid, "owner": owner, "ts": ts}


def acquire(owner: str) -> bool:
    """Toma el lock. False si otro proceso vivo lo tiene."""
    current = holder()
    if current is not None and current["pid"] != os.getpid():
        return False
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(f"{os.getpid()}|{owner}|{time.time()}", encoding="utf-8")
    return True


def release(owner: str) -> None:
    """Libera el lock si somos el dueño."""
    current = holder()
    if current is not None and current["pid"] == os.getpid():
        try:
            LOCK_PATH.unlink()
        except Exception:
            pass


def release_any() -> None:
    """Libera el lock sin chequear dueño (para shutdown/limpieza)."""
    try:
        LOCK_PATH.unlink()
    except Exception:
        pass
