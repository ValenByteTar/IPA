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
import signal
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


def is_port_responding(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(f"http://{host}:{port}/api/state", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def find_dashboard_pids() -> list[int]:
    """Find python/pythonw processes running web_dashboard.py."""
    try:
        # Use wmic to find all python processes (both python.exe and pythonw.exe)
        result = subprocess.run(
            ["wmic", "process", "where", "Name='python.exe' or Name='pythonw.exe'", "get", "ProcessId,CommandLine", "/FORMAT:CSV"],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        pids = []
        for line in result.stdout.strip().splitlines():
            if "web_dashboard.py" in line and "dashboard_watchdog" not in line:
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        pids.append(int(parts[-1]))
                    except ValueError:
                        continue
        return pids
    except Exception:
        return []


def kill_pid(pid: int) -> bool:
    try:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
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
    while True:
        if not is_port_responding(host, port):
            now = time.time()
            # Don't relaunch more than once every 15 seconds to avoid duplicates
            if now - last_launch_time > 15:
                print(f"[watchdog] Dashboard is down, restarting...", flush=True)
                restart_dashboard(host, port)
                last_launch_time = time.time()
            else:
                print(f"[watchdog] Dashboard still down, waiting before retry...", flush=True)
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
    else:
        try:
            run_watchdog(args.host, args.port, args.interval)
        except KeyboardInterrupt:
            print("\n[watchdog] Stopped", flush=True)


if __name__ == "__main__":
    main()
