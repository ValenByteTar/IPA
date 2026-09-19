"""Watchdog that keeps the IPA web dashboard alive.

Runs as a separate lightweight process. Every few seconds it checks if the
dashboard is responding on its port. If not, it launches a fresh instance.

Usage:
    python scripts/operations/dashboard_watchdog.py [--host 127.0.0.1] [--port 8765] [--interval 5]

Can also be used as a one-shot restarter:
    python scripts/operations/dashboard_watchdog.py --restart
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Use pythonw.exe (no console window) to match the launcher
PYTHONW = str(ROOT / ".venv" / "Scripts" / "pythonw.exe")
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PYTHONW).exists():
    PYTHONW = sys.executable
if not Path(PYTHON).exists():
    PYTHON = sys.executable
DASHBOARD_SCRIPT = str(ROOT / "scripts" / "operations" / "web_dashboard.py")
PID_FILE = ROOT / "outputs" / "web_dashboard" / "dashboard.pid"
WATCHDOG_PID_FILE = ROOT / "outputs" / "web_dashboard" / "watchdog.pid"

SEARXNG_URL_DEFAULT = "http://127.0.0.1:8888"
SEARXNG_COMPOSE = ROOT / ".devin" / "searxng" / "docker-compose.yml"
_DOCKER_DESKTOP_CANDIDATES = [
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Docker" / "Docker" / "Docker Desktop.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Docker" / "Docker Desktop.exe",
]

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _ps_processes(matching: str) -> list[tuple[int, str]]:
    """(pid, command_line) for processes whose cmdline contains `matching`.

    Uses Get-CimInstance — wmic is deprecated and absent on modern Windows,
    which previously made this return [] and let stale dashboards pile up.
    """
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
             "Select-Object ProcessId,CommandLine | "
             "ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }"],
            capture_output=True, text=True, timeout=15,
            creationflags=_NO_WINDOW,
        )
        out = []
        for line in result.stdout.splitlines():
            pid_s, _, cmd = line.strip().partition("|")
            if pid_s.isdigit() and matching in cmd:
                out.append((int(pid_s), cmd))
        return out
    except Exception:
        return []


def is_port_responding(host: str, port: int, timeout: float = 3.0) -> bool:
    # /api/health is cheap — /api/state reads every DB and takes seconds,
    # which made a busy-but-healthy dashboard look dead and triggered
    # spurious restarts.
    try:
        req = urllib.request.Request(f"http://{host}:{port}/api/health", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def find_dashboard_pids() -> list[int]:
    """Find python/pythonw processes running web_dashboard.py."""
    return [pid for pid, _ in _ps_processes("web_dashboard.py")]


# --- SearXNG management ----------------------------------------------------
# The watchdog also keeps the local SearXNG container alive so research_topic
# always has its preferred web-search backend after restarts/resets.
# Disable with IPA_SEARXNG_MANAGED=0. Only manages local URLs — if
# IPA_SEARXNG_URL points elsewhere, that instance is the user's to run.


def _searxng_url() -> str:
    return os.environ.get("IPA_SEARXNG_URL", SEARXNG_URL_DEFAULT).rstrip("/")


def _is_local_url(url: str) -> bool:
    try:
        host = urllib.parse.urlsplit(url).hostname or ""
    except Exception:
        return False
    return host in ("127.0.0.1", "localhost", "::1")


def _searxng_responding(url: str, timeout: float = 3.0) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status < 500
    except Exception:
        return False


def _docker_daemon_up() -> bool:
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, timeout=15, creationflags=_NO_WINDOW,
        )
        return result.returncode == 0
    except Exception:
        return False


def _start_docker_desktop() -> bool:
    for candidate in _DOCKER_DESKTOP_CANDIDATES:
        if candidate.exists():
            subprocess.Popen([str(candidate)], creationflags=_NO_WINDOW)
            return True
    return False


def _compose_up() -> tuple[bool, str]:
    if not SEARXNG_COMPOSE.exists():
        return False, f"compose file missing: {SEARXNG_COMPOSE}"
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", str(SEARXNG_COMPOSE), "up", "-d"],
            capture_output=True, text=True, timeout=120, creationflags=_NO_WINDOW,
        )
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr or result.stdout or "").strip()[:300]
    except Exception as exc:
        return False, str(exc)[:300]


def ensure_searxng(state: dict) -> None:
    """Keep local SearXNG alive: start Docker Desktop if needed, compose up.

    `state` carries rate-limiting/logging fields across loop iterations.
    Called at most once per check interval — cheap when SearXNG is up
    (one refused-or-200 HTTP GET).
    """
    if os.environ.get("IPA_SEARXNG_MANAGED", "1") == "0":
        return
    url = _searxng_url()
    if not _is_local_url(url):
        if not state.get("remote_logged"):
            print(f"[watchdog] IPA_SEARXNG_URL={url} is not local — not managing it", flush=True)
            state["remote_logged"] = True
        return
    if _searxng_responding(url):
        if not state.get("was_up"):
            print(f"[watchdog] SearXNG responding at {url}", flush=True)
        state["was_up"] = True
        return
    state["was_up"] = False
    now = time.time()
    if now - state.get("last_attempt", 0.0) < 300:
        return  # attempts at most every 5 min — Docker Desktop boot is slow
    state["last_attempt"] = now

    if not _docker_daemon_up():
        if _start_docker_desktop():
            print("[watchdog] SearXNG down; launched Docker Desktop (waiting for daemon)", flush=True)
        elif not state.get("no_docker_logged"):
            print("[watchdog] SearXNG down and no Docker Desktop found — install Docker or set IPA_SEARXNG_MANAGED=0", flush=True)
            state["no_docker_logged"] = True
        return
    ok, err = _compose_up()
    if ok:
        print(f"[watchdog] SearXNG down; ran compose up (waiting for {url})", flush=True)
    else:
        print(f"[watchdog] SearXNG compose up failed: {err}", flush=True)


def watchdog_already_running() -> bool:
    """True if the recorded watchdog PID is alive and still a watchdog."""
    try:
        pid = int(WATCHDOG_PID_FILE.read_text().strip())
    except Exception:
        return False
    if pid == os.getpid():
        return False
    return any(p == pid and "dashboard_watchdog" in cmd
               for p, cmd in _ps_processes("dashboard_watchdog"))


def kill_pid(pid: int) -> bool:
    try:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=5, creationflags=_NO_WINDOW)
        return True
    except Exception:
        return False


def launch_dashboard(host: str, port: int) -> subprocess.Popen | None:
    """Launch a new dashboard process."""
    log_dir = ROOT / "outputs" / "web_dashboard" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / "dashboard.log", "a", encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "PYTHONIOENCODING": "utf-8",
        # Default the research web-search backend to the managed local SearXNG
        # (ensure_searxng keeps it alive). Explicit IPA_SEARXNG_URL wins.
        "IPA_SEARXNG_URL": os.environ.get("IPA_SEARXNG_URL", SEARXNG_URL_DEFAULT),
    }
    try:
        proc = subprocess.Popen(
            [PYTHONW, "-u", DASHBOARD_SCRIPT, "--host", host, "--port", str(port)],
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(proc.pid), encoding="utf-8")
        print(f"[watchdog] Launched dashboard PID {proc.pid} on {host}:{port}", flush=True)
        return proc
    except Exception as exc:
        print(f"[watchdog] Failed to launch dashboard: {exc}", flush=True)
        return None


def restart_dashboard(host: str, port: int) -> bool:
    """Kill existing dashboard processes and launch a fresh one."""
    pids = find_dashboard_pids()
    for pid in pids:
        print(f"[watchdog] Killing stale dashboard PID {pid}", flush=True)
        kill_pid(pid)
    time.sleep(1)
    proc = launch_dashboard(host, port)
    if proc is None:
        return False
    # Wait for it to come up
    for _ in range(10):
        time.sleep(1)
        if is_port_responding(host, port):
            print(f"[watchdog] Dashboard is responding on {host}:{port}", flush=True)
            return True
    print(f"[watchdog] Dashboard did not come up within 10s", flush=True)
    return False


def run_watchdog(host: str, port: int, interval: int) -> None:
    """Main watchdog loop: check periodically and relaunch if down."""
    print(f"[watchdog] Monitoring {host}:{port} every {interval}s (Ctrl+C to stop)", flush=True)
    last_launch_time = 0.0
    consecutive_failures = 0
    searxng_state: dict = {}
    searxng_interval = float(os.environ.get("IPA_SEARXNG_CHECK_INTERVAL", "60"))
    last_searxng_check = 0.0
    while True:
        now = time.time()
        if now - last_searxng_check >= searxng_interval:
            try:
                ensure_searxng(searxng_state)
            except Exception as exc:
                print(f"[watchdog] ensure_searxng error: {exc}", flush=True)
            last_searxng_check = now
        if is_port_responding(host, port):
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            # 3 failed checks in a row before restarting — a single slow
            # response under load is not a crash. Min 15s between launches.
            if consecutive_failures >= 3 and now - last_launch_time > 15:
                print(f"[watchdog] Dashboard is down ({consecutive_failures} checks), restarting...", flush=True)
                restart_dashboard(host, port)
                last_launch_time = time.time()
                consecutive_failures = 0
            else:
                print(f"[watchdog] Health check failed ({consecutive_failures}/3)", flush=True)
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=int, default=5, help="Check interval in seconds")
    parser.add_argument("--restart", action="store_true", help="One-shot restart and exit")
    args = parser.parse_args()

    if args.restart:
        ok = restart_dashboard(args.host, args.port)
        sys.exit(0 if ok else 1)

    if watchdog_already_running():
        print("[watchdog] Another watchdog is already running — exiting.", flush=True)
        sys.exit(0)
    WATCHDOG_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    WATCHDOG_PID_FILE.write_text(str(os.getpid()), encoding="utf-8")

    try:
        run_watchdog(args.host, args.port, args.interval)
    except KeyboardInterrupt:
        print("\n[watchdog] Stopped", flush=True)


if __name__ == "__main__":
    main()
