"""Lock exclusivo de VRAM para Ollama, ExL3 y bulk embeddings.

En la RTX 4050 de 6 GB Ollama/ExL3 y BGE-M3 no conviven con margen seguro.
El lock coordina ownership entre procesos:

  - Cada generación Ollama mantiene owner="ollama" durante todo el stream.
  - ExL3 y el bulk embed reclaman owner exclusivo, después de esperar a que
    termine cualquier generación Ollama ya iniciada.
  - El bulk renueva su TTL entre batches y lo libera solo después de cerrar
    BGE y completar el warmup del modelo Ollama.
  - Un lock de un proceso muerto o viejo (> TTL) se recupera; el worker de
    embeddings reanuda desde los chunk_ids ya persistidos en LanceDB.

Archivo: outputs/agent/vram.lock (pid|owner|timestamp).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

_PID_ALIVE_TTL_S = 2.0
_pid_alive_cache: dict[int, tuple[float, bool]] = {}

ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = Path(os.environ.get(
    "IPA_VRAM_LOCK", str(ROOT / "outputs" / "agent" / "vram.lock")))
DEFAULT_TTL_S = float(os.environ.get("IPA_VRAM_LOCK_TTL", "1800") or 1800)


def pid_alive(pid: int) -> bool:
    """True si el PID existe; liveness cacheado para polling Windows barato."""
    if pid <= 0:
        return False
    cached = _pid_alive_cache.get(pid)
    if cached is not None and time.time() - cached[0] < _PID_ALIVE_TTL_S:
        return cached[1]
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetExitCodeProcess.argtypes = (
                wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            kernel32.GetExitCodeProcess.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                alive = ctypes.get_last_error() == 5  # access denied means process exists
            else:
                exit_code = wintypes.DWORD()
                ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                kernel32.CloseHandle(handle)
                alive = bool(ok and exit_code.value == 259)  # STILL_ACTIVE
        except Exception:
            alive = False
    else:
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
    _pid_alive_cache[pid] = (time.time(), alive)
    return alive


def _write_entry(owner: str, ts: float) -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = LOCK_PATH.with_name(f"{LOCK_PATH.name}.{os.getpid()}.tmp")
    temp.write_text(f"{os.getpid()}|{owner}|{ts}", encoding="utf-8")
    temp.replace(LOCK_PATH)


def holder() -> dict | None:
    """Devuelve el lock vigente {pid, owner, ts} o None (libre/stale)."""
    try:
        raw = LOCK_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    parts = raw.split("|")
    try:
        if len(parts) != 3:
            raise ValueError("invalid lock format")
        pid, owner, ts = int(parts[0]), parts[1], float(parts[2])
    except ValueError:
        # Un archivo O_EXCL recién creado puede ser leído antes de que el
        # escritor alcance a llenarlo; no borrar esa adquisición en progreso.
        # Un lock malformado que lleva >1s es residual de un crash y se limpia.
        try:
            if time.time() - LOCK_PATH.stat().st_mtime > 1.0:
                LOCK_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        return None
    if time.time() - ts > DEFAULT_TTL_S or not pid_alive(pid):
        try:
            LOCK_PATH.unlink()
        except Exception:
            pass
        return None
    return {"pid": pid, "owner": owner, "ts": ts}


def acquire(owner: str) -> bool:
    """Toma el lock de forma exclusiva; reentrante solo para el mismo owner/PID."""
    current = holder()
    if current is not None:
        if current["pid"] != os.getpid() or current["owner"] != owner:
            return False
        renew(owner)
        return True

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            current = holder()
            if current is not None:
                if current["pid"] == os.getpid() and current["owner"] == owner:
                    renew(owner)
                    return True
                return False
            continue
        try:
            os.write(fd, f"{os.getpid()}|{owner}|{time.time()}".encode("utf-8"))
        finally:
            os.close(fd)
        return True
    return False


def renew(owner: str) -> bool:
    """Renueva TTL del lock vigente; evita que un lote largo caduque en vuelo."""
    current = holder()
    if current is None or current["pid"] != os.getpid() or current["owner"] != owner:
        return False
    _write_entry(owner, time.time())
    return True


def release(owner: str) -> None:
    """Libera el lock solo si somos el proceso y owner vigentes."""
    current = holder()
    if (current is not None and current["pid"] == os.getpid()
            and current["owner"] == owner):
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
