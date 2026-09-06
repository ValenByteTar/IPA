"""Dashboard process/job lifecycle helpers."""
from __future__ import annotations

from . import server as _server

globals().update({name: value for name, value in vars(_server).items() if not name.startswith("__")})

def _find_running_processes(pattern: str) -> list[int]:
    """Find python processes whose command line matches a pattern."""
    try:
        result = subprocess.run(
            ["wmic", "process", "where", "Name='python.exe' or Name='pythonw.exe'", "get", "ProcessId,CommandLine", "/FORMAT:CSV"],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS if os.name == "nt" else 0,
        )
        pids = []
        for line in result.stdout.strip().splitlines():
            if pattern in line:
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        pids.append(int(parts[-1]))
                    except ValueError:
                        continue
        return pids
    except Exception:
        return []


def _job_lock_path(kind: str) -> Path:
    return ROOT / "outputs" / "web_dashboard" / f"{kind}.lock"


def _acquire_job_lock(kind: str, pid: int) -> Path:
    lock_path = _job_lock_path(kind)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_path.write_text(str(pid), encoding="utf-8", errors="strict")
    except OSError:
        pass
    # Atomic create: simultaneous dashboard instances cannot both acquire it.
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(pid))
        return lock_path
    except FileExistsError:
        try:
            owner = int(lock_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            owner = 0
        if owner and _process_exists(owner):
            raise RuntimeError(f"Hay un proceso {kind} corriendo (PID {owner}). Esperá a que termine.")
        try:
            lock_path.unlink()
        except OSError:
            raise RuntimeError(f"No se pudo liberar el lock del proceso {kind}")
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(pid))
        return lock_path


def _process_exists(pid: int) -> bool:
    try:
        no_window = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        return subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, timeout=3, creationflags=no_window).returncode == 0 and str(pid) in subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, timeout=3, creationflags=no_window).stdout
    except Exception:
        return False


def _kill_orphan_processes(pattern: str) -> int:
    """Kill ALL python processes matching pattern. Returns count killed."""
    pids = _find_running_processes(pattern)
    killed = 0
    for pid in pids:
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=5,
                          creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS if os.name == "nt" else 0)
            killed += 1
        except Exception:
            pass
    if killed:
        import time as _t
        _t.sleep(1)  # Give OS time to release resources
    return killed


def spawn_job(kind: str, command: list[str]) -> dict[str, Any]:
    with JOBS_LOCK:
        existing = JOBS.get(kind)
        if existing and existing.poll() is None:
            raise RuntimeError(f"El job {kind} ya está ejecutándose")
        # Kill any orphan processes for this job type before spawning
        orphan_patterns = {
            "scraper": "run_web_scrape.py",
            "pipeline": "run_fast_path.py",
            "lancedb": "run_fast_path.py",  # lancedb reindex uses fast_path
            "reporter": "run_reporter.py",
            "reporter_fast": "run_reporter.py",
        }
        pattern = orphan_patterns.get(kind)
        if pattern:
            killed = _kill_orphan_processes(pattern)
            if killed:
                print(f"  [spawn] killed {killed} orphan(s) for {kind}", flush=True)
        # Clean stale lock
        lock_path = _job_lock_path(kind)
        if lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass
        log_dir = ROOT / "outputs" / "web_dashboard" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = open(log_dir / f"{kind}.log", "a", encoding="utf-8")
        try:
            proc = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": "src", "PYTHONIOENCODING": "utf-8"}, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except Exception:
            log.close()
            lock_path.unlink(missing_ok=True)
            raise
        lock_path.write_text(str(proc.pid), encoding="utf-8")
        JOBS[kind] = proc
        def release_when_done() -> None:
            proc.wait()
            log.close()
            try:
                if lock_path.read_text(encoding="utf-8").strip() == str(proc.pid):
                    lock_path.unlink()
            except OSError:
                pass
        threading.Thread(target=release_when_done, daemon=True).start()
        return {"kind": kind, "pid": proc.pid, "status": "running"}


