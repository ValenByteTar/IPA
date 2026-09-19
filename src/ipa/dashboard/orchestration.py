"""IPA Orchestrator â€” unified launch, monitor, and coordinate all IPA processes.

Runs processes in PARALLEL and shows a unified, readable dashboard:
  - Pipeline (parse + chunk + store + tantivy)
  - LanceDB builder (embed hybrid + index)
  - Enrichment (ExLlamaV3 summaries) â€” auto-launched 300s after LanceDB done

Features:
  - Live dashboard with progress bars, rates, ETAs
  - Health checks: detects stuck processes, lock conflicts
  - GPU scheduling: pauses competing GPU processes
  - Event-driven: enrichment auto-launches when LanceDB done + idle
  - Clean console output: no raw subprocess spam

Usage:
    python scripts/operations/orchestrator.py                          # full system
    python scripts/operations/orchestrator.py --no-scraper --no-hammer # pipeline + lancedb + enrichment
    python scripts/operations/orchestrator.py --chunker semantic       # use semantic chunker
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORPUS = PROJECT_ROOT / "outputs/experiments/E12-corpus"
STATE_DIR = CORPUS / "process_state"
HAMMER_PAUSE_FILE = STATE_DIR / "hammer.pause"
ORCHESTRATOR_LOCK = STATE_DIR / "orchestrator.lock"

PROCESSES = {
    "scraper":    ("scripts/operations/run_job.py", False),
    "pipeline":   ("scripts/operations/run_job.py", False),
    "lancedb":    ("scripts/operations/run_job.py",  True),
    "hammer":     ("scripts/operations/run_job.py",  True),
    "enrichment": ("scripts/operations/run_job.py",  True),
    "rechunk":    ("scripts/operations/run_job.py",  True),
}

# Terminal colors (ANSI)
from .orchestration_ui import C, clear_screen, fmt_time, progress_bar

class Orchestrator:
    """Launches and monitors all IPA processes in parallel."""

    def __init__(self, args):
        self.args = args
        self.run_id = f"{os.getpid()}-{int(time.time())}"
        self._lock_acquired = False
        self.acquire_singleton()
        self.procs: dict[str, subprocess.Popen] = {}
        self.log_handles: dict[str, object] = {}
        self.start_times: dict[str, float] = {}
        self.enrichment_launched = False
        self.lancedb_done_time: float | None = None
        self.idle_start: float | None = None
        self.running = True
        self.events: list[str] = []  # recent event log

    def acquire_singleton(self):
        """Acquire an atomic per-corpus orchestrator lock."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "run_id": self.run_id,
            "started_at": time.time(),
            "command": " ".join(sys.argv),
        }
        try:
            fd = os.open(str(ORCHESTRATOR_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            self._lock_acquired = True
        except FileExistsError:
            try:
                existing = json.loads(ORCHESTRATOR_LOCK.read_text(encoding="utf-8"))
                pid = int(existing.get("pid", 0))
                if pid and subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}"],
                    capture_output=True, text=True, check=False,
                ).stdout.find(str(pid)) >= 0:
                    raise RuntimeError(
                        f"Another orchestrator is already running (pid={pid}, "
                        f"run_id={existing.get('run_id', '?')})"
                    )
            except RuntimeError:
                raise
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            # No live owner was found. Remove only this stale lock and retry
            # atomically; another live orchestrator would have raised above.
            ORCHESTRATOR_LOCK.unlink(missing_ok=True)
            self.acquire_singleton()

    def release_singleton(self):
        """Release the lock only if it still belongs to this process."""
        if not self._lock_acquired:
            return
        try:
            current = json.loads(ORCHESTRATOR_LOCK.read_text(encoding="utf-8"))
            if current.get("pid") == os.getpid() and current.get("run_id") == self.run_id:
                ORCHESTRATOR_LOCK.unlink(missing_ok=True)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        self._lock_acquired = False

    def log_event(self, msg: str):
        """Add an event to the event log."""
        ts = time.strftime("%H:%M:%S")
        self.events.append(f"{C.GRAY}[{ts}]{C.RESET} {msg}")
        if len(self.events) > 8:
            self.events = self.events[-8:]

    def launch_process(self, name: str) -> bool:
        script, _ = PROCESSES[name]
        cmd = [sys.executable, "-u", str(PROJECT_ROOT / script), "--job", name]
        if name == "scraper" and self.args.scraper_config:
            cmd.extend(["--config", self.args.scraper_config])
        if name == "pipeline":
            cmd.extend(["--chunker", self.args.chunker, "--idle-timeout", str(self.args.pipeline_idle_timeout)])

        try:
            # On Windows, create process in a new process group so we can
            # send CTRL_BREAK_EVENT for graceful shutdown (allows finally
            # blocks to run, closing Tantivy writers and flushing LanceDB).
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            log_dir = CORPUS / "logs" / self.run_id
            log_dir.mkdir(parents=True, exist_ok=True)
            log_handle = open(log_dir / f"{name}.log", "a", encoding="utf-8", buffering=1)
            proc = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONPATH": "src", "PYTHONIOENCODING": "utf-8", "IPA_RUN_ID": self.run_id},
                creationflags=creationflags,
            )
            self.procs[name] = proc
            self.log_handles[name] = log_handle
            self.start_times[name] = time.time()
            self.log_event(f"{C.BLUE}â–¶ {name}{C.RESET} launched (pid={proc.pid})")
            return True
        except Exception as e:
            self.log_event(f"{C.RED}âœ— {name}{C.RESET} failed to launch: {e}")
            return False

    def kill_process(self, name: str):
        """Graceful shutdown: CTRL_BREAK â†’ wait 15s â†’ hard kill."""
        if name not in self.procs:
            return
        proc = self.procs[name]
        if proc.poll() is None:
            # Try graceful shutdown first (allows finally/cleanup to run)
            try:
                if os.name == "nt":
                    # CTRL_BREAK_EVENT to the process group â€” Python handles
                    # this as KeyboardInterrupt, running finally blocks
                    proc.send_signal(subprocess.signal.CTRL_BREAK_EVENT)
                else:
                    proc.terminate()
                proc.wait(timeout=15)
                self.log_event(f"{C.YELLOW}â–  {name}{C.RESET} stopped gracefully")
            except subprocess.TimeoutExpired:
                # Graceful shutdown failed â€” hard kill as last resort
                proc.kill()
                proc.wait()
                self.log_event(f"{C.RED}â–  {name}{C.RESET} hard-killed (cleanup timeout)")
            except Exception:
                # send_signal not supported â€” fall back to terminate
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                self.log_event(f"{C.RED}â–  {name}{C.RESET} terminated")
        handle = self.log_handles.pop(name, None)
        if handle is not None:
            handle.close()
        del self.procs[name]

    def pause_hammer(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        HAMMER_PAUSE_FILE.touch()
        self.log_event(f"{C.YELLOW}â¸ hammer{C.RESET} paused (GPU for enrichment)")

    def resume_hammer(self):
        if HAMMER_PAUSE_FILE.exists():
            HAMMER_PAUSE_FILE.unlink()
            self.log_event(f"{C.GREEN}â–¶ hammer{C.RESET} resumed")

    def read_state(self, name: str) -> dict | None:
        state_file = STATE_DIR / f"{name}.json"
        if not state_file.exists():
            return None
        try:
            return json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def check_health(self, name: str) -> str:
        state = self.read_state(name)
        if state is None:
            return "dead"
        state_run_id = state.get("run_id")
        if state_run_id and state_run_id != self.run_id:
            return "stale"
        status = state.get("status", "unknown")
        if status == "done":
            return "done"
        if status == "error":
            return "error"
        if status == "stuck":
            return "stuck"
        if status == "paused":
            return "paused"
        proc = self.procs.get(name)
        if proc and proc.poll() is not None:
            if proc.returncode == 0 or status == "done":
                return "done"
            return "error"
        timestamp = state.get("timestamp", 0)
        if time.time() - timestamp > 300:
            return "stuck"
        return "healthy"

    def maybe_launch_enrichment(self):
        if self.enrichment_launched or self.args.no_enrichment:
            return
        lancedb_health = self.check_health("lancedb")
        if lancedb_health != "done":
            self.idle_start = None
            return
        if self.lancedb_done_time is None:
            self.lancedb_done_time = time.time()
            self.idle_start = time.time()
            self.log_event(
                f"{C.MAGENTA}â—† enrichment{C.RESET} LanceDB done. "
                f"Waiting {self.args.enrichment_idle}s idle..."
            )
        idle_elapsed = time.time() - self.idle_start
        if idle_elapsed >= self.args.enrichment_idle:
            self.log_event(
                f"{C.MAGENTA}â–¶ enrichment{C.RESET} launching after "
                f"{self.args.enrichment_idle}s idle"
            )
            if "hammer" in self.procs:
                self.pause_hammer()
            self.launch_process("enrichment")
            self.enrichment_launched = True

    # ------------------------------------------------------------------
    # Dashboard rendering
    # ------------------------------------------------------------------

    def render_process_row(self, name: str) -> list[str]:
        """Render a single process row in the dashboard."""
        state = self.read_state(name)
        health = self.check_health(name)

        if state is None and name not in self.procs:
            return []

        icon = C.status_icon(health)
        status = state.get("status", "?") if state else "starting"
        elapsed = time.time() - self.start_times.get(name, time.time())
        m = state.get("metrics", {}) if state else {}

        # Build progress line per process type
        if name == "pipeline":
            files = m.get("files_processed", 0)
            chunks = m.get("total_chunks", 0)
            lancedb_rows = m.get("lancedb_rows", 0)
            detail = f"files={C.BOLD}{files}{C.RESET} chunks={C.BOLD}{chunks}{C.RESET}"
            if lancedb_rows > 0:
                detail += f" {C.CYAN}lancedb={lancedb_rows}{C.RESET}"
            if "idle_seconds" in m:
                detail += f" {C.YELLOW}idle {m['idle_seconds']:.0f}s{C.RESET}"

        elif name == "lancedb":
            embedded = m.get("embedded", 0)
            store = m.get("store_count", 0)
            missing = m.get("missing", 0)
            if store > 0:
                bar = progress_bar(embedded, store, 20)
                detail = f"{bar} {C.BOLD}{embedded}{C.RESET}/{store}"
                if missing > 0:
                    detail += f" {C.GRAY}missing={missing}{C.RESET}"
            else:
                detail = f"{C.GRAY}waiting for chunks...{C.RESET}"

        elif name == "enrichment":
            done = m.get("enriched", 0)
            total = m.get("total_to_process", m.get("to_process", 0))
            rate = m.get("rate", 0)
            eta = m.get("eta_seconds", 0)
            reembedded = m.get("reembedded", 0)
            if total > 0:
                bar = progress_bar(done, total, 20)
                detail = f"{bar} {C.BOLD}{done}{C.RESET}/{total}"
                if rate > 0:
                    detail += f" {C.CYAN}{rate:.1f}/s{C.RESET} ETA {fmt_time(eta)}"
                if reembedded > 0:
                    detail += f" {C.MAGENTA}reembed={reembedded}{C.RESET}"
            elif m.get("phase") == "loading_model":
                detail = f"{C.YELLOW}loading ExLlamaV3 + BGE-M3...{C.RESET}"
            else:
                detail = f"{C.GRAY}starting...{C.RESET}"

        elif name == "scraper":
            articles = m.get("articles_saved", 0)
            sites = m.get("sites_processed", 0)
            detail = f"articles={C.BOLD}{articles}{C.RESET} sites={sites}"

        elif name == "hammer":
            rounds = m.get("rounds_completed", 0)
            p50 = m.get("p50_ms", "?")
            detail = f"rounds={rounds} p50={p50}ms"

        elif name == "rechunk":
            done = m.get("completed", 0)
            total = m.get("total_docs", 0)
            if total > 0:
                bar = progress_bar(done, total, 20)
                detail = f"{bar} {done}/{total}"
            else:
                detail = f"{C.GRAY}starting...{C.RESET}"
        else:
            detail = ""

        # Status badge
        status_colors = {
            "running": C.GREEN,
            "done": C.GREEN,
            "error": C.RED,
            "stuck": C.YELLOW,
            "paused": C.YELLOW,
        }
        sc = status_colors.get(health, C.GRAY)
        status_str = f"{sc}{status:8s}{C.RESET}"

        return [
            f"  {icon} {C.BOLD}{name:12s}{C.RESET} {status_str} {fmt_time(elapsed):>8s}",
            f"    {detail}",
        ]

    def render_dashboard(self) -> str:
        """Render the full dashboard."""
        lines = []

        # Header
        lines.append(f"{C.BG_BLUE}{C.BOLD}{' IPA Orchestrator ':=^80}{C.RESET}")
        lines.append(
            f"  {C.GRAY}{time.strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}"
            f"  chunker={C.CYAN}{self.args.chunker}{C.RESET}"
            f"  corpus={C.GRAY}E12-corpus{C.RESET}"
        )
        lines.append(f"  {C.GRAY}{'â”€' * 76}{C.RESET}")

        # Corpus-level availability summary
        pipeline_state = self.read_state("pipeline") or {}
        lancedb_state = self.read_state("lancedb") or {}
        pm = pipeline_state.get("metrics", {})
        lm = lancedb_state.get("metrics", {})
        landing_root = PROJECT_ROOT / "Landing"
        landing_count = sum(1 for p in landing_root.rglob("*") if p.is_file()) if landing_root.exists() else 0
        lines.append(
            f"  {C.BOLD}Corpus:{C.RESET} docs={pm.get('documents', '?')} "
            f"chunks={pm.get('total_chunks', '?')} "
            f"lexical={pipeline_state.get('status', '?')} "
            f"vector={lm.get('embedded', '?')}/{lm.get('store_count', '?')} "
            f"landing={landing_count}"
        )
        lines.append(f"  {C.GRAY}{'â”€' * 76}{C.RESET}")

        # Process rows
        active_names = [n for n in PROCESSES if n in self.procs or self.read_state(n)]
        for name in active_names:
            rows = self.render_process_row(name)
            if rows:
                lines.extend(rows)
                lines.append("")

        # GPU status
        gpu_procs = []
        for name, (_, needs_gpu) in PROCESSES.items():
            if not needs_gpu:
                continue
            h = self.check_health(name)
            if h in ("healthy", "paused"):
                gpu_procs.append(name)
        if gpu_procs:
            gpu_color = C.YELLOW if len(gpu_procs) > 1 else C.GREEN
            lines.append(
                f"  {C.BOLD}GPU:{C.RESET} {gpu_color}{', '.join(gpu_procs)}{C.RESET}"
                + (f" {C.RED}(contention!){C.RESET}" if len(gpu_procs) > 1 else "")
            )

        # Enrichment trigger status
        if not self.enrichment_launched and not self.args.no_enrichment:
            lh = self.check_health("lancedb")
            if lh == "done" and self.idle_start:
                remaining = self.args.enrichment_idle - (time.time() - self.idle_start)
                if remaining > 0:
                    lines.append(
                        f"  {C.MAGENTA}Enrichment trigger:{C.RESET} "
                        f"{fmt_time(remaining)} until launch"
                    )
        lines.append("")

        # Event log
        lines.append(f"  {C.BOLD}Recent events:{C.RESET}")
        for event in self.events[-6:]:
            lines.append(f"    {event}")

        lines.append(f"  {C.GRAY}{'â”€' * 76}{C.RESET}")
        lines.append(f"  {C.GRAY}Press Ctrl+C to stop all processes{C.RESET}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, signum=None, frame=None):
        self.running = False

    def run(self):
        # Register signal handlers
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

        # Launch processes
        if not self.args.no_scraper:
            self.launch_process("scraper")
        if not self.args.no_pipeline:
            self.launch_process("pipeline")
        if not self.args.no_lancedb:
            self.launch_process("lancedb")
        if not self.args.no_hammer:
            self.launch_process("hammer")

        # Main loop
        dashboard_interval = self.args.dashboard_interval
        last_dashboard = 0

        while self.running:
            now = time.time()

            # GPU scheduling
            lancedb_state = self.read_state("lancedb")
            if lancedb_state and lancedb_state.get("status") == "running":
                missing = lancedb_state.get("metrics", {}).get("missing", 0)
                if missing > 100 and "hammer" in self.procs:
                    if not HAMMER_PAUSE_FILE.exists():
                        self.pause_hammer()
                elif missing <= 100:
                    if HAMMER_PAUSE_FILE.exists():
                        self.resume_hammer()
            elif lancedb_state and lancedb_state.get("status") == "done":
                if HAMMER_PAUSE_FILE.exists() and not self.enrichment_launched:
                    self.resume_hammer()

            # Enrichment trigger
            if not self.args.no_enrichment:
                self.maybe_launch_enrichment()

            # Resume hammer after enrichment done
            if self.enrichment_launched:
                eh = self.check_health("enrichment")
                if eh in ("done", "error"):
                    if HAMMER_PAUSE_FILE.exists():
                        self.resume_hammer()

            # Restart stuck scraper
            for name in list(self.procs.keys()):
                if self.check_health(name) == "stuck" and name == "scraper":
                    self.log_event(f"{C.YELLOW}â†» {name}{C.RESET} stuck, restarting...")
                    self.kill_process(name)
                    if not self.args.no_scraper:
                        time.sleep(2)
                        self.launch_process(name)

            # Render dashboard
            if now - last_dashboard > dashboard_interval:
                last_dashboard = now
                clear_screen()
                print(self.render_dashboard())

            time.sleep(1)

        # Shutdown
        clear_screen()
        print(f"\n{C.YELLOW}[orchestrator] Shutting down all processes...{C.RESET}")
        for name in list(self.procs.keys()):
            self.kill_process(name)
        if HAMMER_PAUSE_FILE.exists():
            HAMMER_PAUSE_FILE.unlink()
        self.release_singleton()
        print(f"{C.GREEN}[orchestrator] All processes stopped.{C.RESET}")


def main():
    print(
        f"{C.YELLOW}[DEPRECATED]{C.RESET} El pipeline del Orchestrator "
        "(scraper → fast_path → lancedb → hammer → enrichment) está deprecado: "
        "los jobs se lanzan desde el dashboard (run_ingestion, scraper, "
        "reporter) y el trabajo LLM en background corre por el idle scheduler "
        "(Tiers 1/2). Esta consola sigue funcionando por compatibilidad pero "
        "no recibe nuevas funciones.",
        flush=True,
    )
    parser = argparse.ArgumentParser(description="IPA Orchestrator (DEPRECATED).")
    parser.add_argument("--no-scraper", action="store_true")
    parser.add_argument("--no-pipeline", action="store_true")
    parser.add_argument("--no-lancedb", action="store_true")
    parser.add_argument("--no-hammer", action="store_true")
    parser.add_argument("--no-enrichment", action="store_true")
    parser.add_argument("--rechunk", action="store_true")
    parser.add_argument("--chunker", default="fixed", choices=["fixed", "semantic"])
    parser.add_argument("--scraper-config", default="configs/scrape_sites.yaml")
    parser.add_argument("--enrichment-idle", type=int, default=300)
    parser.add_argument("--pipeline-idle-timeout", type=float, default=None)
    parser.add_argument("--dashboard-interval", type=float, default=3.0)
    args = parser.parse_args()
    if args.pipeline_idle_timeout is None:
        args.pipeline_idle_timeout = 60.0 if args.no_scraper else 600.0

    orch = Orchestrator(args)
    orch.run()


if __name__ == "__main__":
    main()
