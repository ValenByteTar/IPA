"""E11 — Observability: query and display artifact traces.

Usage:
    python scripts/run_trace_query.py --trace-db outputs/experiments/E11-observability/trace.db
    python scripts/run_trace_query.py --trace-db trace.db --artifact sha256:abc123
    python scripts/run_trace_query.py --trace-db trace.db --summary
    python scripts/run_trace_query.py --trace-db trace.db --failed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ipa.trace_log import TraceLog


def format_event(e, indent: int = 2) -> str:
    """Format a trace event for display."""
    sp = " " * indent
    lines = [
        f"{sp}[{e.timestamp}] {e.stage:12s} {e.status:8s} "
        f"{e.latency_ms:>10.1f}ms  worker={e.worker_id}",
    ]
    if e.input_hash:
        lines.append(f"{sp}  input_hash:  {e.input_hash[:24]}...")
    if e.output_hash:
        lines.append(f"{sp}  output_hash: {e.output_hash[:24]}...")
    if e.error:
        lines.append(f"{sp}  ERROR: {e.error}")
    if e.metadata:
        meta_str = json.dumps(e.metadata, ensure_ascii=False)
        if len(meta_str) > 120:
            meta_str = meta_str[:117] + "..."
        lines.append(f"{sp}  metadata: {meta_str}")
    return "\n".join(lines)


def cmd_summary(trace: TraceLog) -> None:
    s = trace.summary()
    print("=" * 70)
    print("Trace Log Summary")
    print("=" * 70)
    print(f"  Total events:    {s['total_events']}")
    print(f"  Artifacts:       {s['artifacts']}")
    print(f"  Failed events:   {s['failed_events']}")
    print(f"  Avg latency:     {s['avg_latency_ms']:.2f}ms")
    print(f"  Events by stage:")
    for stage, count in sorted(s["stages"].items()):
        print(f"    {stage:14s}  {count}")
    print()


def cmd_artifact(trace: TraceLog, artifact_id: str) -> None:
    events = trace.get_artifact_trace(artifact_id)
    if not events:
        print(f"No events found for artifact: {artifact_id}")
        return
    print("=" * 70)
    print(f"Trace for artifact: {artifact_id}")
    print("=" * 70)
    total_ms = sum(e.latency_ms for e in events if e.stage != "pipeline")
    print(f"  Events: {len(events)}  Total stage latency: {total_ms:.1f}ms")
    print()
    for e in events:
        print(format_event(e))
    print()


def cmd_failed(trace: TraceLog) -> None:
    events = trace.get_failed_events()
    if not events:
        print("No failed events.")
        return
    print("=" * 70)
    print(f"Failed Events: {len(events)}")
    print("=" * 70)
    for e in events:
        print(format_event(e))
    print()


def cmd_all(trace: TraceLog, limit: int) -> None:
    """Show recent events (last N)."""
    all_rows = trace._conn.execute(
        """SELECT event_id, artifact_id, stage, status, input_hash,
                  output_hash, latency_ms, worker_id, error, timestamp,
                  metadata_json
           FROM trace_events
           ORDER BY seq DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    events = [TraceLog._row_to_event(r) for r in all_rows]
    print("=" * 70)
    print(f"Last {len(events)} events (newest first)")
    print("=" * 70)
    for e in reversed(events):
        print(format_event(e))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="E11 trace query tool")
    parser.add_argument("--trace-db", required=True, help="Path to trace.db")
    parser.add_argument("--artifact", help="Show trace for a specific artifact_id")
    parser.add_argument("--summary", action="store_true", help="Show summary")
    parser.add_argument("--failed", action="store_true", help="Show failed events")
    parser.add_argument("--all", action="store_true", help="Show recent events")
    parser.add_argument("--limit", type=int, default=50, help="Limit for --all")
    args = parser.parse_args()

    trace = TraceLog(args.trace_db)
    try:
        if args.artifact:
            cmd_artifact(trace, args.artifact)
        elif args.failed:
            cmd_failed(trace)
        elif args.all:
            cmd_all(trace, args.limit)
        else:
            cmd_summary(trace)
    finally:
        trace.close()


if __name__ == "__main__":
    main()
