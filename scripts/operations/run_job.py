"""Generic job runner entrypoint.

Replaces the six scripts/proc_*.py wrappers with a single CLI that
delegates to ipa.dashboard.process_runner.JobRunner driven by JobSpec.

Usage:
    python scripts/operations/run_job.py --job scraper
    python scripts/operations/run_job.py --job pipeline --chunker semantic
    python scripts/operations/run_job.py --job lancedb
    python scripts/operations/run_job.py --job hammer
    python scripts/operations/run_job.py --job enrichment
    python scripts/operations/run_job.py --job rechunk

The runner writes process state to:
    outputs/experiments/E12-corpus/process_state/<job>.json

which the orchestrator and dashboard read for health/progress display.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure src/ is on the path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipa.dashboard.process_runner import run_job, run_job_with_retry  # noqa: E402
from ipa.dashboard.process_specs import SPECS, get_spec  # noqa: E402
from ipa.dashboard.process_state import acquire_slot, release_slot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generic IPA job runner (replaces proc_*.py wrappers).",
    )
    parser.add_argument(
        "--job",
        required=True,
        choices=list(SPECS),
        help="Which job to run.",
    )
    # Job-specific overrides (passed through to the worker)
    parser.add_argument("--chunker", default=None, help="Chunker for pipeline (fixed|semantic).")
    parser.add_argument("--idle-timeout", type=float, default=None, help="Pipeline idle timeout.")
    parser.add_argument("--config", default=None, help="Scraper config path.")
    parser.add_argument("--state-dir", default=None, help="Override state directory.")
    # Resilience (orchestrator-improvements): bounded retries + backoff.
    parser.add_argument("--retries", type=int, default=1, help="Max attempts (1 = no retry).")
    parser.add_argument("--backoff", type=float, default=2.0, help="Base backoff seconds between attempts.")
    # Backpressure: bounded concurrency for a shared resource (e.g. gpu).
    parser.add_argument("--resource", default=None, help="Resource name for slot-based backpressure (e.g. gpu).")
    parser.add_argument("--max-concurrent", type=int, default=1, help="Slots for --resource.")
    args = parser.parse_args()

    spec = get_spec(args.job)
    command_overrides: dict = {}

    # Apply overrides to the spec's command_args by rebuilding them
    # We do this by passing overrides to run_job, which passes them to
    # spec.build_command. For now, the simplest approach is to mutate
    # the command_args list based on the override flags.
    if args.job == "pipeline":
        new_args = list(spec.command_args)
        if args.chunker:
            for i, a in enumerate(new_args):
                if a == "--chunker" and i + 1 < len(new_args):
                    new_args[i + 1] = args.chunker
        if args.idle_timeout is not None:
            for i, a in enumerate(new_args):
                if a == "--idle-timeout" and i + 1 < len(new_args):
                    new_args[i + 1] = str(args.idle_timeout)
        # We need to pass these through; since JobSpec is frozen, we use
        # the overrides dict which build_command can consult.
        command_overrides["chunker"] = args.chunker
        command_overrides["idle_timeout"] = args.idle_timeout
    elif args.job == "scraper":
        command_overrides["config"] = args.config

    state_dir = Path(args.state_dir) if args.state_dir else None

    # Backpressure: claim a bounded resource slot before spawning.
    holder = f"{args.job}:{os.getpid()}"
    slot_index: int | None = None
    if args.resource:
        effective_state_dir = state_dir or (PROJECT_ROOT / "outputs" / "experiments" / "E12-corpus" / "process_state")
        slot_index = acquire_slot(
            effective_state_dir, args.resource, max(1, args.max_concurrent), holder,
        )
        if slot_index is None:
            print(
                f"BACKPRESSURE: resource '{args.resource}' at max concurrency "
                f"({args.max_concurrent}); job {args.job} not started.",
                flush=True,
            )
            return 75  # EX_TEMPFAIL: caller may re-queue later

    try:
        if args.retries > 1:
            return run_job_with_retry(
                job_name=args.job,
                project_root=PROJECT_ROOT,
                state_dir=state_dir,
                command_overrides=command_overrides,
                max_attempts=args.retries,
                backoff_s=args.backoff,
            )
        return run_job(
            job_name=args.job,
            project_root=PROJECT_ROOT,
            state_dir=state_dir,
            command_overrides=command_overrides,
        )
    finally:
        if slot_index is not None:
            release_slot(effective_state_dir, args.resource, slot_index, holder)


if __name__ == "__main__":
    sys.exit(main())
