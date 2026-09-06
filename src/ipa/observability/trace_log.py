"""E11 â€” Observability: end-to-end traceability by artifact_id.

Every stage of the pipeline emits a TraceEvent with:
  - artifact_id (links to LandingZone)
  - stage name (landing, mime, parsing, chunking, storing, indexing, search)
  - status (running, success, failed, skipped)
  - input_hash (hash of input to the stage)
  - output_hash (hash of output from the stage)
  - latency_ms
  - worker_id (identifier of the process/thread that ran the stage)
  - error (exception message if failed)
  - timestamp (ISO 8601 UTC)
  - metadata (extra stage-specific fields, e.g. parser_id, chunk_count)

TraceLog persists events to SQLite and supports querying the full
lifecycle of any artifact:

    trace = TraceLog("trace.db")
    events = trace.get_artifact_trace("sha256:abc123")
    for e in events:
        print(f"{e.stage:12s} {e.status:8s} {e.latency_ms:>8.1f}ms  {e.error or ''}")

This module is designed to be non-invasive: FastPathRunner accepts an
optional trace_log parameter.  If None, no tracing occurs (zero overhead).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TraceEvent:
    """A single traceability event in the pipeline."""
    event_id: str  # deterministic: sha256(artifact_id + stage + timestamp)
    artifact_id: str
    stage: str  # landing, mime, parsing, chunking, storing, indexing, search
    status: str  # running, success, failed, skipped
    input_hash: str  # sha256 of input content, or "" if not applicable
    output_hash: str  # sha256 of output content, or "" if not applicable
    latency_ms: float
    worker_id: str
    error: str  # empty string if no error
    timestamp: str  # ISO 8601 UTC
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# TraceLog â€” SQLite-backed event store
# ---------------------------------------------------------------------------

class TraceLog:
    """Persistent trace event log backed by SQLite.

    Thread-safe via SQLite's own locking (WAL mode).  Multiple workers
    can emit events concurrently.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trace_events (
                event_id TEXT PRIMARY KEY,
                artifact_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                input_hash TEXT DEFAULT '',
                output_hash TEXT DEFAULT '',
                latency_ms REAL DEFAULT 0,
                worker_id TEXT DEFAULT '',
                error TEXT DEFAULT '',
                timestamp TEXT NOT NULL,
                metadata_json TEXT DEFAULT '{}',
                seq INTEGER
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trace_artifact
            ON trace_events(artifact_id, seq)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trace_stage
            ON trace_events(stage, status)
        """)
        self._conn.commit()
        self._seq = self._next_seq()

    def _next_seq(self) -> int:
        row = self._conn.execute("SELECT COALESCE(MAX(seq), 0) FROM trace_events").fetchone()
        return (row[0] or 0) + 1

    def emit(self, event: TraceEvent) -> None:
        """Persist a trace event."""
        self._conn.execute(
            """INSERT OR REPLACE INTO trace_events
               (event_id, artifact_id, stage, status, input_hash, output_hash,
                latency_ms, worker_id, error, timestamp, metadata_json, seq)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.event_id,
                event.artifact_id,
                event.stage,
                event.status,
                event.input_hash,
                event.output_hash,
                event.latency_ms,
                event.worker_id,
                event.error,
                event.timestamp,
                json.dumps(event.metadata, ensure_ascii=False),
                self._seq,
            ),
        )
        self._seq += 1
        self._conn.commit()

    def get_artifact_trace(self, artifact_id: str) -> list[TraceEvent]:
        """Return all events for an artifact, ordered by sequence."""
        rows = self._conn.execute(
            """SELECT event_id, artifact_id, stage, status, input_hash,
                      output_hash, latency_ms, worker_id, error, timestamp,
                      metadata_json
               FROM trace_events
               WHERE artifact_id = ?
               ORDER BY seq""",
            (artifact_id,),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_stage_events(
        self, stage: str, status: str | None = None
    ) -> list[TraceEvent]:
        """Return all events for a given stage (optionally filtered by status)."""
        if status:
            rows = self._conn.execute(
                """SELECT event_id, artifact_id, stage, status, input_hash,
                          output_hash, latency_ms, worker_id, error, timestamp,
                          metadata_json
                   FROM trace_events
                   WHERE stage = ? AND status = ?
                   ORDER BY seq""",
                (stage, status),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT event_id, artifact_id, stage, status, input_hash,
                          output_hash, latency_ms, worker_id, error, timestamp,
                          metadata_json
                   FROM trace_events
                   WHERE stage = ?
                   ORDER BY seq""",
                (stage,),
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_failed_events(self) -> list[TraceEvent]:
        """Return all events with status='failed'."""
        return self.get_stage_events("", status="failed") if False else [
            self._row_to_event(r)
            for r in self._conn.execute(
                """SELECT event_id, artifact_id, stage, status, input_hash,
                          output_hash, latency_ms, worker_id, error, timestamp,
                          metadata_json
                   FROM trace_events
                   WHERE status = 'failed'
                   ORDER BY seq"""
            ).fetchall()
        ]

    def count(self) -> int:
        """Total number of events."""
        return self._conn.execute("SELECT COUNT(*) FROM trace_events").fetchone()[0]

    def count_by_stage(self) -> dict[str, int]:
        """Count events grouped by stage."""
        rows = self._conn.execute(
            "SELECT stage, COUNT(*) FROM trace_events GROUP BY stage"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def summary(self) -> dict[str, Any]:
        """Return a summary of the trace log."""
        total = self.count()
        if total == 0:
            return {"total_events": 0, "artifacts": 0, "stages": {}}
        artifacts = self._conn.execute(
            "SELECT COUNT(DISTINCT artifact_id) FROM trace_events"
        ).fetchone()[0]
        stages = self.count_by_stage()
        failed = len(self.get_failed_events())
        avg_latency = self._conn.execute(
            "SELECT AVG(latency_ms) FROM trace_events WHERE latency_ms > 0"
        ).fetchone()[0] or 0
        return {
            "total_events": total,
            "artifacts": artifacts,
            "failed_events": failed,
            "avg_latency_ms": round(avg_latency, 2),
            "stages": stages,
        }

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TraceLog":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @staticmethod
    def _row_to_event(row: tuple) -> TraceEvent:
        return TraceEvent(
            event_id=row[0],
            artifact_id=row[1],
            stage=row[2],
            status=row[3],
            input_hash=row[4],
            output_hash=row[5],
            latency_ms=row[6],
            worker_id=row[7],
            error=row[8],
            timestamp=row[9],
            metadata=json.loads(row[10] or "{}"),
        )


# ---------------------------------------------------------------------------
# Helpers for emitting events
# ---------------------------------------------------------------------------

def _worker_id() -> str:
    """Get a unique worker identifier (pid + thread name)."""
    import threading
    return f"pid-{os.getpid()}:tid-{threading.get_ident()}"


def _timestamp() -> str:
    """ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _hash_text(text: str) -> str:
    """SHA-256 hash of text, prefixed with sha256:."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _hash_bytes(data: bytes) -> str:
    """SHA-256 hash of bytes, prefixed with sha256:."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _event_id(artifact_id: str, stage: str, timestamp: str) -> str:
    """Deterministic event ID."""
    raw = f"{artifact_id}:{stage}:{timestamp}"
    return f"sha256:{hashlib.sha256(raw.encode()).hexdigest()}"


def make_event(
    artifact_id: str,
    stage: str,
    status: str,
    latency_ms: float = 0.0,
    input_hash: str = "",
    output_hash: str = "",
    error: str = "",
    metadata: dict[str, Any] | None = None,
) -> TraceEvent:
    """Create a TraceEvent with auto-filled timestamp and worker_id."""
    ts = _timestamp()
    return TraceEvent(
        event_id=_event_id(artifact_id, stage, ts),
        artifact_id=artifact_id,
        stage=stage,
        status=status,
        input_hash=input_hash,
        output_hash=output_hash,
        latency_ms=latency_ms,
        worker_id=_worker_id(),
        error=error,
        timestamp=ts,
        metadata=metadata or {},
    )

