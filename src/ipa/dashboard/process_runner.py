"""Common JobRunner — monitors a worker subprocess and writes process state.

Replaces the duplicated subprocess/state/parsing logic from the six
scripts/proc_*.py wrappers with a single reusable runner driven by a
JobSpec.

Lifecycle:
    1. Write initial "running" state.
    2. Spawn worker subprocess (stdout=PIPE, stderr=STDOUT).
    3. For each stdout line:
       a. Skip if skip_line predicate returns True.
       b. Update last_output_time.
       c. Echo line to stdout (for log capture).
       d. Parse line → metrics dict.
       e. Run job-specific postprocessor.
       f. Compute status from idle time + completion.
       g. Write state (atomic or simple per spec).
       h. For hammer: check pause file between lines.
    4. Wait for process exit.
    5. Write final "done" or "error" state.
    6. Return exit code per spec.exit_policy.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .process_specs import JobSpec, POSTPROCESSORS
from .process_state import (
    DEFAULT_STATE_DIR,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_PAUSED,
    STATUS_RUNNING,
    read_state,
    write_state_atomic,
    write_state_simple,
)


class JobRunner:
    """Runs a single JobSpec to completion, writing process state throughout."""

    def __init__(
        self,
        spec: JobSpec,
        project_root: Path | None = None,
        state_dir: Path | None = None,
        run_id: str | None = None,
        command_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.spec = spec
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.state_dir = Path(state_dir) if state_dir else DEFAULT_STATE_DIR
        self.run_id = run_id or os.environ.get("IPA_RUN_ID")
        self.command_overrides = command_overrides or {}
        # Pause file for hammer (same convention as legacy proc_hammer.py)
        self.pause_file = self.state_dir / f"{spec.name}.pause"

    # ------------------------------------------------------------------
    # State writing
    # ------------------------------------------------------------------

    def _write_state(
        self,
        status: str,
        metrics: dict[str, Any],
        errors: list[str] | None = None,
    ) -> None:
        """Write state using the spec's atomic/simple policy."""
        common = dict(
            state_dir=self.state_dir,
            job_name=self.spec.name,
            status=status,
            metrics=metrics,
            errors=errors,
            run_id=self.run_id,
        )
        if self.spec.use_atomic_state:
            write_state_atomic(**common)
        else:
            write_state_simple(**common)

    # ------------------------------------------------------------------
    # Hammer pause/resume
    # ------------------------------------------------------------------

    def _check_pause(self, metrics: dict[str, Any]) -> None:
        """For hammer: block while pause file exists, writing paused state."""
        if self.spec.name != "hammer":
            return
        if not self.pause_file.exists():
            return
        self._write_state(STATUS_PAUSED, {**metrics, "paused_at": time.time()})
        while self.pause_file.exists():
            time.sleep(2)
        self._write_state(STATUS_RUNNING, {**metrics, "resumed_at": time.time()})

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Execute the job. Returns the final exit code."""
        spec = self.spec
        cmd = spec.build_command(self.project_root, self.command_overrides)
        env = spec.build_env()

        # The runner echoes worker lines to its own stdout; when the parent
        # console uses a legacy codepage (cp1252) Unicode output from workers
        # (e.g. "→" in progress lines) would crash print(). Force UTF-8 with
        # replacement so echoing never kills the runner.
        try:
            if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

        # Initial state
        initial_metrics: dict[str, Any] = {"command": " ".join(cmd)}
        if spec.name in ("enrichment", "rechunk"):
            initial_metrics["phase"] = "loading_model"
        self._write_state(STATUS_RUNNING, initial_metrics)

        # Spawn worker
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            creationflags=creationflags,
        )

        metrics: dict[str, Any] = {}
        errors: list[str] = []
        last_output_time = time.time()
        postprocessor = POSTPROCESSORS.get(spec.name)

        # Per-job initial metric seeds (matching legacy wrappers)
        if spec.name == "scraper":
            metrics = {"articles_saved": 0, "sites_processed": 0, "errors": 0}
        elif spec.name == "pipeline":
            metrics = {"files_processed": 0, "total_chunks": 0, "lock_conflicts": 0}
        elif spec.name == "hammer":
            metrics = {"rounds_completed": 0}

        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            # Skip noise lines (e.g. lancedb progress bars)
            if spec.skip_line and spec.skip_line(line):
                continue

            # Compute idle BEFORE refreshing the output timestamp so a line
            # arriving after the idle threshold is classified against the
            # real silence window (previously idle was always ~0 here).
            now = time.time()
            idle = now - last_output_time
            last_output_time = now
            print(line, flush=True)

            # Hammer pause check (between lines, like legacy proc_hammer)
            self._check_pause(metrics)

            # Parse + postprocess
            if spec.parse_line:
                parsed = spec.parse_line(line)
                if parsed:
                    if postprocessor:
                        postprocessor(metrics, parsed, line, errors)
                    else:
                        # lancedb: simple update
                        metrics.update(parsed)
                        if parsed.get("error"):
                            errors.append(line)

            # Compute and write state (idle was measured against the previous
            # output timestamp at the top of the loop).
            status = spec.compute_status(idle, metrics)
            self._write_state(status, {**metrics, "idle_seconds": int(idle)}, errors)

        proc.wait()
        exit_code = proc.returncode

        # Final state
        final_status = spec.final_status(exit_code, metrics)
        self._write_state(final_status, {**metrics, "exit_code": exit_code}, errors)

        return spec.final_exit_code(exit_code, metrics)


def run_job(
    job_name: str,
    project_root: Path | None = None,
    state_dir: Path | None = None,
    run_id: str | None = None,
    command_overrides: dict[str, Any] | None = None,
) -> int:
    """Convenience: look up a spec by name and run it."""
    from .process_specs import get_spec
    spec = get_spec(job_name)
    runner = JobRunner(spec, project_root, state_dir, run_id, command_overrides)
    return runner.run()


def run_job_with_retry(
    job_name: str | None = None,
    project_root: Path | None = None,
    state_dir: Path | None = None,
    run_id: str | None = None,
    command_overrides: dict[str, Any] | None = None,
    *,
    spec: "JobSpec | None" = None,
    max_attempts: int = 3,
    backoff_s: float = 2.0,
    backoff_factor: float = 2.0,
) -> int:
    """Run a job with bounded retries and exponential backoff.

    Accepts either a registered ``job_name`` or an explicit ``spec``.
    Only non-zero outcomes are retried; a successful run returns immediately.
    The attempt count is recorded in the job state metrics. Backoff sleeps
    ``backoff_s * backoff_factor ** (attempt - 1)`` between attempts.
    """
    from .process_specs import get_spec
    from .process_state import read_state, write_state_atomic

    resolved_spec = spec if spec is not None else get_spec(job_name)  # type: ignore[arg-type]
    effective_state_dir = Path(state_dir) if state_dir else DEFAULT_STATE_DIR
    last_code: int | None = None
    last_error: Exception | None = None

    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            runner = JobRunner(resolved_spec, project_root, effective_state_dir, run_id, command_overrides)
            last_code = runner.run()
            last_error = None
        except Exception as exc:  # spawn/IO failures are retryable too
            last_code = None
            last_error = exc
        if last_code == 0:
            state = read_state(effective_state_dir, resolved_spec.name) or {}
            metrics = dict(state.get("metrics", {}))
            metrics["attempts"] = attempt
            write_state_atomic(
                state_dir=effective_state_dir, job_name=resolved_spec.name,
                status=state.get("status", "done"), metrics=metrics,
                errors=state.get("errors"), run_id=run_id,
            )
            return 0
        if attempt < max_attempts:
            time.sleep(backoff_s * (backoff_factor ** (attempt - 1)))

    state = read_state(effective_state_dir, resolved_spec.name) or {}
    metrics = dict(state.get("metrics", {}))
    metrics["attempts"] = max_attempts
    if last_error is not None:
        metrics["retry_exception"] = str(last_error)
    write_state_atomic(
        state_dir=effective_state_dir, job_name=resolved_spec.name,
        status=state.get("status", "error"), metrics=metrics,
        errors=state.get("errors"), run_id=run_id,
    )
    return last_code if last_code is not None else 1
