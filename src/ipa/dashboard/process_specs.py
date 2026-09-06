"""JobSpec definitions for orchestrator-managed jobs.

Each JobSpec captures the worker-specific behavior that was previously
duplicated across scripts/proc_*.py wrappers:

- command builder (what subprocess to launch);
- stdout parser (how to extract metrics from worker output);
- status policy (idle/stuck thresholds, completion predicate);
- state-write policy (atomic vs simple);
- extra environment (e.g. CUDA_PATH for enrichment);
- line-skip predicate (e.g. lancedb skips progress-bar lines).

The orchestrator launches a single generic runner
(scripts/operations/run_job.py --job <name>) which loads the spec
and delegates to process_runner.JobRunner.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .process_state import (
    DEFAULT_STATE_DIR,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_IDLE,
    STATUS_RUNNING,
    STATUS_STUCK,
)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ParseFn = Callable[[str], dict[str, Any]]
LineSkipFn = Callable[[str], bool]
CommandBuilder = Callable[["JobSpec", dict[str, Any]], list[str]]


@dataclass(frozen=True)
class JobSpec:
    """Declarative specification for an orchestrator-managed job."""

    name: str
    """Canonical job name (matches state file name)."""

    worker_script: str
    """Path to the worker script, relative to project root.
    May point into scripts/ or local_archive/scripts/ for archived workers."""

    needs_gpu: bool = False
    """Whether this job competes for GPU resources."""

    idle_threshold: float = 60.0
    """Seconds without output before marking idle/stuck."""

    idle_status: str = STATUS_IDLE
    """Status to set when idle threshold is reached (idle or stuck)."""

    stuck_threshold: float | None = None
    """If set, a separate stuck threshold beyond idle.
    When None, idle_status is used directly after idle_threshold."""

    use_atomic_state: bool = True
    """Whether to use atomic tempfile writes (pipeline/lancedb/enrichment)
    or simple direct writes (scraper/hammer/rechunk)."""

    extra_env: dict[str, str] = field(default_factory=dict)
    """Additional environment variables for the worker subprocess."""

    command_args: list[str] = field(default_factory=list)
    """Static extra arguments appended after the worker script."""

    parse_line: ParseFn | None = None
    """Parser for stdout lines → metrics dict. None means no parsing."""

    skip_line: LineSkipFn | None = None
    """Predicate returning True for lines that should be skipped entirely
    (not printed, not parsed, not counted as output)."""

    complete_predicate: Callable[[dict[str, Any]], bool] | None = None
    """If set, overrides the default 'metrics.get("complete")' check."""

    exit_policy: str = "exit_code"
    """How to determine final exit code:
    - 'exit_code': return the worker's exit code
    - 'complete': return 0 if complete, else worker exit code
    """

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def resolve_worker_path(self, project_root: Path) -> Path:
        """Resolve the worker script path.

        Checks active scripts/ first, then local_archive/scripts/ for
        archived-but-available workers.
        """
        root = Path(project_root)
        candidate = root / self.worker_script
        if candidate.exists():
            return candidate
        # Try archive fallback: scripts/_foo.py → local_archive/scripts/_foo.py
        if self.worker_script.startswith("scripts/"):
            archive_path = root / "local_archive" / self.worker_script
            if archive_path.exists():
                return archive_path
        return candidate  # return original even if missing (for error msg)

    def build_command(self, project_root: Path, overrides: dict[str, Any] | None = None) -> list[str]:
        """Build the subprocess command for this job's worker.

        Overrides can replace values in command_args for keys like
        --chunker, --idle-timeout, --config.
        """
        worker = self.resolve_worker_path(project_root)
        cmd = [sys.executable, "-u", str(worker)]
        args = list(self.command_args)
        if overrides:
            for key, value in overrides.items():
                if value is None:
                    continue
                flag = f"--{key.replace('_', '-')}"
                for i, a in enumerate(args):
                    if a == flag and i + 1 < len(args):
                        args[i + 1] = str(value)
                        break
        cmd.extend(args)
        return cmd

    def build_env(self) -> dict[str, str]:
        """Build the subprocess environment."""
        env = {
            **os.environ,
            "PYTHONPATH": "src",
            "PYTHONIOENCODING": "utf-8",
        }
        env.update(self.extra_env)
        return env

    def compute_status(self, idle_seconds: float, metrics: dict[str, Any]) -> str:
        """Compute current status from idle time and metrics."""
        if self.is_complete(metrics):
            return STATUS_DONE
        if self.stuck_threshold is not None and idle_seconds >= self.stuck_threshold:
            return STATUS_STUCK
        if idle_seconds >= self.idle_threshold:
            return self.idle_status
        return STATUS_RUNNING

    def is_complete(self, metrics: dict[str, Any]) -> bool:
        """Check if the job is complete."""
        if self.complete_predicate is not None:
            return self.complete_predicate(metrics)
        return bool(metrics.get("complete"))

    def final_status(self, exit_code: int, metrics: dict[str, Any]) -> str:
        """Determine final status after process exit."""
        if self.is_complete(metrics):
            return STATUS_DONE
        if exit_code == 0:
            return STATUS_DONE
        return STATUS_ERROR

    def final_exit_code(self, worker_exit_code: int, metrics: dict[str, Any]) -> int:
        """Determine what exit code the runner should return."""
        if self.exit_policy == "complete" and self.is_complete(metrics):
            return 0
        return worker_exit_code


# ---------------------------------------------------------------------------
# Parsers — one per job, migrated from proc_*.py
# ---------------------------------------------------------------------------

def _parse_scraper(line: str) -> dict[str, Any]:
    """Parse scraper stdout. Migrated from proc_scraper.parse_line."""
    metrics: dict[str, Any] = {}
    m = re.search(r'\[(https?://[^\]]+)\] Found: (\d+) articles', line)
    if m:
        metrics["last_site"] = m.group(1)
        metrics["found"] = int(m.group(2))
    m = re.search(r'\[(https?://[^\]]+)\] Scraped: (\d+), Skipped: (\d+)', line)
    if m:
        metrics["last_site"] = m.group(1)
        metrics["scraped"] = int(m.group(2))
        metrics["skipped"] = int(m.group(3))
    m = re.search(r'\[(https?://[^\]]+)\] Saved: (.+?) \((\d+) chars', line)
    if m:
        metrics["last_site"] = m.group(1)
        metrics["last_file"] = m.group(2).strip()
        metrics["last_file_chars"] = int(m.group(3))
    m = re.search(r'^--- (https?://[^ ]+) ---', line)
    if m:
        metrics["current_site"] = m.group(1)
    m = re.search(r'Scrape complete in ([\d.]+)s', line)
    if m:
        metrics["elapsed"] = float(m.group(1))
        metrics["complete"] = True
    m = re.search(r'Errors: (\d+)', line)
    if m:
        metrics["errors"] = int(m.group(1))
    return metrics


def _parse_pipeline(line: str) -> dict[str, Any]:
    """Parse pipeline stdout. Migrated from proc_pipeline.parse_line."""
    metrics: dict[str, Any] = {}
    m = re.search(r'Found (\d+) new file\(s\)', line)
    if m:
        metrics["new_files"] = int(m.group(1))
    m = re.search(r'OK: (\S+) → (\d+) chunks, (\d+) pages', line)
    if m:
        metrics["last_mime"] = m.group(1)
        metrics["last_chunks"] = int(m.group(2))
        metrics["last_pages"] = int(m.group(3))
        metrics["file_processed"] = True
    m = re.search(r'OK \(retry\): (\S+) → (\d+) chunks', line)
    if m:
        metrics["last_mime"] = m.group(1)
        metrics["last_chunks"] = int(m.group(2))
        metrics["file_processed"] = True
    if "Archived to" in line:
        metrics["archived"] = True
    if "ERROR" in line:
        metrics["error"] = line.strip()
    if "Lock conflict" in line:
        metrics["lock_conflict"] = True
    if "Left in Landing" in line:
        metrics["left_in_landing"] = True
    m = re.search(r'Fast path complete: (\d+) files, (\d+) chunks', line)
    if m:
        metrics["files_processed"] = int(m.group(1))
        metrics["total_chunks"] = int(m.group(2))
    if "Pipeline complete" in line:
        metrics["complete"] = True
    m = re.search(r'Total chunks:\s+(\d+)', line)
    if m:
        metrics["total_chunks"] = int(m.group(1))
    m = re.search(r'Idle for ([\d.]+)s', line)
    if m:
        metrics["idle_seconds"] = float(m.group(1))
    m = re.search(r'\[LanceDB incremental\] \+(\d+) chunks → (\d+) total rows', line)
    if m:
        metrics["lancedb_rows"] = int(m.group(2))
    return metrics


def _parse_lancedb(line: str) -> dict[str, Any]:
    """Parse LanceDB builder stdout. Migrated from proc_lancedb.parse_line."""
    metrics: dict[str, Any] = {}
    m = re.search(
        r'\[Round (\d+)\] \+(\d+) chunks → (\d+) total rows.*'
        r'Store: (\d+), Embedded: (\d+), Missing: (\d+)', line,
    )
    if m:
        metrics["round"] = int(m.group(1))
        metrics["batch_chunks"] = int(m.group(2))
        metrics["total_rows"] = int(m.group(3))
        metrics["store_count"] = int(m.group(4))
        metrics["embedded"] = int(m.group(5))
        metrics["missing"] = int(m.group(6))
    if "No new chunks" in line:
        m = re.search(r'Store: (\d+), Embedded: (\d+)', line)
        if m:
            metrics["store_count"] = int(m.group(1))
            metrics["embedded"] = int(m.group(2))
            metrics["missing"] = 0
    if "Incremental LanceDB builder complete" in line:
        metrics["complete"] = True
    m = re.search(r'Total added: (\d+)', line)
    if m:
        metrics["total_added"] = int(m.group(1))
    m = re.search(r'Total rows:\s+(\d+)', line)
    if m:
        metrics["total_rows"] = int(m.group(1))
    if "FTS index" in line and "created" in line.lower():
        metrics["fts_index"] = True
    if "Error" in line or "error" in line.lower():
        if "FTS" not in line:
            metrics["error"] = line.strip()
    return metrics


def _parse_hammer(line: str) -> dict[str, Any]:
    """Parse hammer stdout. Migrated from proc_hammer.parse_line."""
    metrics: dict[str, Any] = {}
    m = re.search(r'\[Round (\d+)\] Docs: (\d+), Chunks: (\d+)', line)
    if m:
        metrics["round"] = int(m.group(1))
        metrics["docs"] = int(m.group(2))
        metrics["chunks"] = int(m.group(3))
    m = re.search(r'Tantivy snapshot: (\d+)ms', line)
    if m:
        metrics["tantivy_snapshot_ms"] = int(m.group(1))
    m = re.search(r'Query latency: p50=([\d.]+)ms p99=([\d.]+)ms \| Hits: (\d+)', line)
    if m:
        metrics["p50_ms"] = float(m.group(1))
        metrics["p99_ms"] = float(m.group(2))
        metrics["hits"] = int(m.group(3))
    m = re.search(r'\[([^\]]+)\] lex\s+([\d.]+)ms \| sem\s+([\d.]+)ms', line)
    if m:
        metrics["last_query"] = m.group(1).strip()
        metrics["last_lex_ms"] = float(m.group(2))
        metrics["last_sem_ms"] = float(m.group(3))
    m = re.search(r'Python processes: (\d+)', line)
    if m:
        metrics["python_processes"] = int(m.group(1))
    return metrics


def _parse_enrichment(line: str) -> dict[str, Any]:
    """Parse enrichment stdout. Migrated from proc_enrichment.parse_line."""
    metrics: dict[str, Any] = {}
    m = re.search(r'Total chunks:\s+(\d+)', line)
    if m:
        metrics["total_chunks"] = int(m.group(1))
    m = re.search(r'Already enriched.*?: (\d+)', line)
    if m:
        metrics["already_enriched"] = int(m.group(1))
    m = re.search(r'To process:\s+(\d+)', line)
    if m:
        metrics["to_process"] = int(m.group(1))
    m = re.search(r'Skipped.*?: (\d+)', line)
    if m:
        metrics["skipped"] = int(m.group(1))
    m = re.search(r'Model loaded in ([\d.]+)s', line)
    if m:
        metrics["model_load_time"] = float(m.group(1))
    m = re.search(
        r'\[(\d+)/(\d+)\] enriched=(\d+) reembedded=(\d+) errors=(\d+)'
        r'.*?([\d.]+)s.*?([\d.]+) chunks/s.*?ETA: ([\d.]+)s', line,
    )
    if m:
        metrics["completed"] = int(m.group(1))
        metrics["total_to_process"] = int(m.group(2))
        metrics["enriched"] = int(m.group(3))
        metrics["reembedded"] = int(m.group(4))
        metrics["errors"] = int(m.group(5))
        metrics["elapsed"] = float(m.group(6))
        metrics["rate"] = float(m.group(7))
        metrics["eta_seconds"] = float(m.group(8))
    if "ExLlamaV3 enrichment complete" in line:
        metrics["complete"] = True
    if "No chunks need enrichment or re-embedding" in line:
        metrics["complete"] = True
        metrics["no_work"] = True
    m = re.search(r'Enriched:\s+(\d+)', line)
    if m:
        metrics["final_enriched"] = int(m.group(1))
    m = re.search(r'Time:\s+([\d.]+)s', line)
    if m:
        metrics["final_time"] = float(m.group(1))
    m = re.search(r'Rate:\s+([\d.]+) chunks/s', line)
    if m:
        metrics["final_rate"] = float(m.group(1))
    if "ERROR" in line:
        metrics["error"] = line.strip()
    return metrics


def _parse_rechunk(line: str) -> dict[str, Any]:
    """Parse rechunk stdout. Migrated from proc_rechunk.parse_line."""
    metrics: dict[str, Any] = {}
    m = re.search(r'Documents to re-chunk: (\d+)', line)
    if m:
        metrics["total_docs"] = int(m.group(1))
    m = re.search(r'Current chunks.*?: (\d+)', line)
    if m:
        metrics["old_chunk_count"] = int(m.group(1))
    m = re.search(r'Documents already semantic-chunked.*?: (\d+)', line)
    if m:
        metrics["already_done"] = int(m.group(1))
    m = re.search(r'Documents remaining: (\d+)', line)
    if m:
        metrics["remaining"] = int(m.group(1))
    m = re.search(r'Device: (.+), Dims: (\d+)', line)
    if m:
        metrics["device"] = m.group(1)
        metrics["dims"] = int(m.group(2))
    m = re.search(
        r'\[(\d+)/(\d+)\] doc .+? → (\d+) chunks.*?total: (\d+)'
        r'.*?([\d.]+)s.*?ETA: ([\d.]+)s', line,
    )
    if m:
        metrics["completed"] = int(m.group(1))
        metrics["total_docs"] = int(m.group(2))
        metrics["last_doc_chunks"] = int(m.group(3))
        metrics["total_new_chunks"] = int(m.group(4))
        metrics["elapsed"] = float(m.group(5))
        metrics["eta_seconds"] = float(m.group(6))
    if "Re-chunking complete" in line:
        metrics["complete"] = True
    m = re.search(r'New chunks:\s+(\d+)', line)
    if m:
        metrics["new_chunk_count"] = int(m.group(1))
    m = re.search(r'Chunk size: min=(\d+), max=(\d+), avg=(\d+)', line)
    if m:
        metrics["min_chunk_size"] = int(m.group(1))
        metrics["max_chunk_size"] = int(m.group(2))
        metrics["avg_chunk_size"] = int(m.group(3))
    if "ERROR" in line:
        metrics["error"] = line.strip()
    return metrics


def _skip_lancedb_noise(line: str) -> bool:
    """Skip progress-bar and pre-tokenize lines for lancedb."""
    if "it/s]" in line and "Inference" in line:
        return True
    if "pre tokenize" in line:
        return True
    return False


# ---------------------------------------------------------------------------
# Metric post-processors — job-specific accumulation logic
# ---------------------------------------------------------------------------

def _scraper_postprocess(metrics: dict[str, Any], parsed: dict[str, Any], line: str, errors: list[str]) -> None:
    """Scraper-specific metric accumulation (from proc_scraper main loop)."""
    metrics.update(parsed)
    if "last_file" in parsed:
        metrics["articles_saved"] = metrics.get("articles_saved", 0) + 1
    if "scraped" in parsed:
        metrics["sites_processed"] = metrics.get("sites_processed", 0) + 1
    if parsed.get("errors", 0) > 0:
        errors.append(line)


def _pipeline_postprocess(metrics: dict[str, Any], parsed: dict[str, Any], line: str, errors: list[str]) -> None:
    """Pipeline-specific metric accumulation (from proc_pipeline main loop)."""
    if parsed.get("file_processed"):
        metrics["files_processed"] = metrics.get("files_processed", 0) + 1
        metrics["total_chunks"] = metrics.get("total_chunks", 0) + parsed.get("last_chunks", 0)
    if "files_processed" in parsed and "total_chunks" in parsed:
        metrics["files_processed"] = parsed["files_processed"]
        metrics["total_chunks"] = parsed["total_chunks"]
    elif "total_chunks" in parsed:
        metrics["total_chunks"] = parsed["total_chunks"]
    if parsed.get("lock_conflict"):
        metrics["lock_conflicts"] = metrics.get("lock_conflicts", 0) + 1
    if parsed.get("error"):
        errors.append(line)
    if parsed.get("left_in_landing"):
        errors.append(f"File left in Landing: {line}")


def _hammer_postprocess(metrics: dict[str, Any], parsed: dict[str, Any], line: str, errors: list[str]) -> None:
    """Hammer-specific metric accumulation."""
    metrics.update(parsed)
    if "round" in parsed:
        metrics["rounds_completed"] = parsed["round"]


def _enrichment_postprocess(metrics: dict[str, Any], parsed: dict[str, Any], line: str, errors: list[str]) -> None:
    """Enrichment-specific metric accumulation."""
    metrics.update(parsed)
    if parsed.get("error"):
        errors.append(line)
    if "completed" in parsed:
        metrics["phase"] = "enriching"
    if parsed.get("complete"):
        metrics["phase"] = "done"


def _rechunk_postprocess(metrics: dict[str, Any], parsed: dict[str, Any], line: str, errors: list[str]) -> None:
    """Rechunk-specific metric accumulation."""
    metrics.update(parsed)
    if parsed.get("error"):
        errors.append(line)
    if "completed" in parsed:
        metrics["phase"] = "rechunking"
    if parsed.get("complete"):
        metrics["phase"] = "done"


# ---------------------------------------------------------------------------
# JobSpec registry
# ---------------------------------------------------------------------------

# Default CUDA path used by enrichment (matches proc_enrichment.py).
_CUDA_PATH = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

SPECS: dict[str, JobSpec] = {
    "scraper": JobSpec(
        name="scraper",
        worker_script="scripts/cli/run_web_scrape.py",
        needs_gpu=False,
        idle_threshold=120.0,
        idle_status=STATUS_STUCK,
        use_atomic_state=False,
        command_args=[
            "--config", "configs/scrape_sites.yaml",
            "--output", "Landing/web",
            "--days-back", "61",
            "--no-ocr",
            "--no-images",
        ],
        parse_line=_parse_scraper,
    ),
    "pipeline": JobSpec(
        name="pipeline",
        worker_script="scripts/operations/run_continuous_pipeline.py",
        needs_gpu=False,
        idle_threshold=60.0,
        idle_status=STATUS_IDLE,
        use_atomic_state=True,
        command_args=[
            "--landing", "Landing",
            "--archive", "Archive",
            "--corpus", str(DEFAULT_STATE_DIR.parent),
            "--poll-interval", "2.0",
            "--idle-timeout", "600.0",
            "--chunker", "fixed",
            "--no-enrichment",
            "--no-lancedb",
        ],
        parse_line=_parse_pipeline,
    ),
    "lancedb": JobSpec(
        name="lancedb",
        worker_script="scripts/operations/workers/lancedb_incremental.py",
        needs_gpu=True,
        idle_threshold=30.0,
        idle_status=STATUS_STUCK,
        use_atomic_state=True,
        parse_line=_parse_lancedb,
        skip_line=_skip_lancedb_noise,
        exit_policy="complete",
    ),
    "hammer": JobSpec(
        name="hammer",
        worker_script="scripts/operations/workers/hammer_queries.py",
        needs_gpu=True,
        idle_threshold=1e9,  # hammer doesn't use idle/stuck; always running
        idle_status=STATUS_RUNNING,
        use_atomic_state=False,
        parse_line=_parse_hammer,
    ),
    "enrichment": JobSpec(
        name="enrichment",
        worker_script="scripts/operations/workers/run_enrichment_exl3.py",
        needs_gpu=True,
        idle_threshold=60.0,
        idle_status=STATUS_STUCK,
        use_atomic_state=True,
        extra_env={"CUDA_PATH": _CUDA_PATH},
        parse_line=_parse_enrichment,
        exit_policy="complete",
    ),
    "rechunk": JobSpec(
        name="rechunk",
        worker_script="scripts/operations/workers/rechunk_semantic.py",
        needs_gpu=True,
        idle_threshold=120.0,
        idle_status=STATUS_STUCK,
        use_atomic_state=False,
        parse_line=_parse_rechunk,
        exit_policy="complete",
    ),
}

# Post-processors keyed by job name. Called after parse_line for each line.
POSTPROCESSORS: dict[str, Callable[[dict, dict, str, list], None]] = {
    "scraper": _scraper_postprocess,
    "pipeline": _pipeline_postprocess,
    "hammer": _hammer_postprocess,
    "enrichment": _enrichment_postprocess,
    "rechunk": _rechunk_postprocess,
    # lancedb uses simple metrics.update (no special accumulation)
}


def get_spec(name: str) -> JobSpec:
    """Look up a JobSpec by name. Raises KeyError if not found."""
    if name not in SPECS:
        raise KeyError(f"Unknown job spec: {name!r}. Available: {list(SPECS)}")
    return SPECS[name]


def spec_names() -> list[str]:
    """Return all registered spec names."""
    return list(SPECS)
