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
    env = {**os.environ, "PYTHONPATH": "src", "PYTHONIOENCODING": "utf-8"}
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
    while True:
        if is_port_responding(host, port):
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            now = time.time()
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
