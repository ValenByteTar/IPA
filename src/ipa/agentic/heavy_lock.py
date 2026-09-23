"""Lock de trabajos pesados (CPU/IO) entre procesos.

En esta máquina (12 threads, una GPU de 6 GB) los trabajos pesados no
conviven: una investigación (ingesta + embeddings BGE-M3) y el drain de
embeddings del fast path compiten por CPU, y con BGE-M3 en CPU (gate de VRAM
por el 9B cargado) un batch de 64 chunks tarda ~40 s. Medido el 2026-09-22: una
research con presupuesto de 120 s llevaba 27 min porque compartía `Landing/web`
con la ingesta masiva (heredó sus ~600 archivos) y quedó detrás del drain de
~70k chunks del pipeline.

El lock serializa las fases pesadas y da prioridad a lo interactivo:

  - El holder escribe ``pid|kind|priority|ts`` en ``outputs/agent/heavy.lock``.
  - Un waiter de mayor prioridad se registra en ``heavy.waiting``.
  - El holder background cede entre unidades de trabajo (``should_yield``)
    para que lo interactivo pase primero.
  - Un lock de un proceso muerto o viejo (> TTL) se roba solo: un crash no
    bloquea el sistema para siempre.

Es advisory y best-effort por diseño (PAT-004): si un trabajo interactivo no
consigue el lock dentro de ``wait_s``, sigue igual — nunca se deadlockea una
respuesta al usuario por un job de background.

Uso:

    with heavy_phase("research", PRIORITY_INTERACTIVE, on_wait=cb) as held:
        ...  # fase pesada; `held` indica si se serializó o se siguió igual

Vive en ``ipa.agentic`` junto al idle scheduler: es disciplina de recursos del
runtime agentivo, no lógica de negocio de ningún dominio.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = Path(os.environ.get(
    "IPA_HEAVY_LOCK", str(ROOT / "outputs" / "agent" / "heavy.lock")))
WAIT_PATH = LOCK_PATH.with_suffix(".waiting")
DEFAULT_TTL_S = float(os.environ.get("IPA_HEAVY_LOCK_TTL", "1800") or 1800)
DEFAULT_WAIT_S = float(os.environ.get("IPA_HEAVY_LOCK_WAIT", "300") or 300)
POLL_S = 0.5

# Lo interactivo (research pedido por el usuario en el chat) gana sobre lo
# background (pipeline, drains, idle). Menor número = mayor prioridad.
PRIORITY_INTERACTIVE = 10
PRIORITY_BACKGROUND = 50


# Cache corto de liveness para no hacer una consulta Win32 al mismo PID en
# cada poll de espera; evita subprocesses y ventanas de consola.
_PID_ALIVE_TTL_S = 2.0
_pid_alive_cache: dict[int, tuple[float, bool]] = {}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    cached = _pid_alive_cache.get(pid)
    if cached is not None and time.time() - cached[0] < _PID_ALIVE_TTL_S:
        return cached[1]
    alive = _pid_alive_uncached(pid)
    _pid_alive_cache[pid] = (time.time(), alive)
    return alive


def _pid_alive_uncached(pid: int) -> bool:
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
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return ctypes.get_last_error() == 5
            exit_code = wintypes.DWORD()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            return bool(ok and exit_code.value == 259)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _parse_holder(raw: str) -> dict[str, Any] | None:
    parts = raw.strip().split("|")
    if len(parts) != 4:
        return None
    try:
        pid, kind, priority, ts = int(parts[0]), parts[1], int(parts[2]), float(parts[3])
    except ValueError:
        return None
    return {"pid": pid, "kind": kind, "priority": priority, "ts": ts}


def holder() -> dict[str, Any] | None:
    """Holder vigente {pid, kind, priority, ts} o None (libre/stale)."""
    try:
        raw = LOCK_PATH.read_text(encoding="utf-8")
    except Exception:
        return None
    current = _parse_holder(raw)
    if current is None:
        return None
    if time.time() - current["ts"] > DEFAULT_TTL_S or not _pid_alive(current["pid"]):
        try:
            LOCK_PATH.unlink()
        except Exception:
            pass
        return None
    return current


def _write_holder(kind: str, priority: int) -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(
        f"{os.getpid()}|{kind}|{priority}|{time.time()}", encoding="utf-8")


def try_acquire(kind: str, priority: int = PRIORITY_BACKGROUND) -> bool:
    """Intenta tomar el lock sin esperar. Re-entrante para el mismo proceso."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        current = holder()  # limpia stale
        if current is None:
            # Se liberó entre el intento y la lectura: reintentar una vez.
            try:
                fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                return False
        elif current["pid"] == os.getpid():
            # Mismo proceso (re-entrada o refresh del heartbeat).
            _write_holder(kind, priority)
            return True
        else:
            return False
    os.write(fd, f"{os.getpid()}|{kind}|{priority}|{time.time()}".encode("utf-8"))
    os.close(fd)
    return True


def release() -> None:
    """Libera el lock si somos el dueño."""
    current = holder()
    if current is not None and current["pid"] == os.getpid():
        try:
            LOCK_PATH.unlink()
        except Exception:
            pass
    unregister_waiter()


def waiters() -> list[dict[str, Any]]:
    """Waiters vivos registrados {pid, kind, priority, ts}."""
    try:
        raw = json.loads(WAIT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    alive = [
        w for w in raw
        if isinstance(w, dict) and w.get("pid") and _pid_alive(int(w["pid"]))
    ]
    if len(alive) != len(raw):
        _write_waiters(alive)
    return alive


def _write_waiters(items: list[dict[str, Any]]) -> None:
    try:
        WAIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        WAIT_PATH.write_text(json.dumps(items), encoding="utf-8")
    except Exception:
        pass


def register_waiter(kind: str, priority: int) -> None:
    """Registra (o refresca) este proceso como waiter de mayor prioridad."""
    items = [w for w in waiters() if int(w.get("pid", 0)) != os.getpid()]
    items.append({"pid": os.getpid(), "kind": kind, "priority": priority, "ts": time.time()})
    _write_waiters(items)


def unregister_waiter() -> None:
    items = [w for w in waiters() if int(w.get("pid", 0)) != os.getpid()]
    if items:
        _write_waiters(items)
    else:
        try:
            WAIT_PATH.unlink(missing_ok=True)
        except Exception:
            pass


def should_yield(priority: int = PRIORITY_BACKGROUND) -> bool:
    """True si hay un waiter vivo de mayor prioridad: el holder background cede."""
    return any(int(w.get("priority", 99)) < priority for w in waiters())


@contextmanager
def heavy_phase(
    kind: str,
    priority: int = PRIORITY_BACKGROUND,
    wait_s: float = 0.0,
    on_wait: Callable[[float, dict[str, Any] | None], None] | None = None,
) -> Iterator[bool]:
    """Context manager: intenta serializar una fase pesada.

    ``wait_s`` acota la espera (0 = no esperar). Si no se consigue, la fase
    corre igual (best-effort, nunca se deadlockea) y el bloque recibe False.
    Mientras espera, el proceso queda registrado como waiter de mayor
    prioridad: el holder background lo ve en ``should_yield`` y cede.
    ``on_wait(elapsed_s, holder)`` se llama en cada poll mientras se espera.
    """
    acquired = try_acquire(kind, priority)
    waited = 0.0
    if not acquired and wait_s > 0:
        register_waiter(kind, priority)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            current = holder()
            if on_wait is not None:
                try:
                    on_wait(waited, current)
                except Exception:
                    pass
            if current is None and try_acquire(kind, priority):
                acquired = True
                break
            time.sleep(POLL_S)
            waited += POLL_S
        unregister_waiter()
    try:
        yield acquired
    finally:
        if acquired:
            release()
