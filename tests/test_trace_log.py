"""Tests for E11 — Observability: TraceLog and end-to-end traceability."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from ipa import (
    FastPathRunner,
    TraceLog,
    TraceEvent,
    make_event,
)


# --- TraceLog basic operations ---

class TestTraceLog:
    def test_emit_and_retrieve_single_event(self, tmp_path):
        with TraceLog(tmp_path / "trace.db") as log:
            ev = make_event(
                "sha256:abc", "parsing", "success",
                latency_ms=42.5,
                output_hash="sha256:out123",
                metadata={"parser_id": "pymupdf", "pages": 3},
            )
            log.emit(ev)
            assert log.count() == 1
            events = log.get_artifact_trace("sha256:abc")
            assert len(events) == 1
            assert events[0].artifact_id == "sha256:abc"
            assert events[0].stage == "parsing"
            assert events[0].status == "success"
            assert events[0].latency_ms == 42.5
            assert events[0].output_hash == "sha256:out123"
            assert events[0].metadata["parser_id"] == "pymupdf"
            assert events[0].metadata["pages"] == 3

    def test_emit_multiple_events_preserve_order(self, tmp_path):
        with TraceLog(tmp_path / "trace.db") as log:
            for stage in ["landing", "mime", "parsing", "chunking", "storing", "indexing"]:
                log.emit(make_event("sha256:xyz", stage, "success", latency_ms=10.0))
            events = log.get_artifact_trace("sha256:xyz")
            assert len(events) == 6
            stages = [e.stage for e in events]
            assert stages == ["landing", "mime", "parsing", "chunking", "storing", "indexing"]

    def test_get_artifact_trace_empty_for_unknown(self, tmp_path):
        with TraceLog(tmp_path / "trace.db") as log:
            log.emit(make_event("sha256:known", "landing", "success"))
            events = log.get_artifact_trace("sha256:unknown")
            assert events == []

    def test_get_stage_events(self, tmp_path):
        with TraceLog(tmp_path / "trace.db") as log:
            log.emit(make_event("sha256:a", "parsing", "success"))
            log.emit(make_event("sha256:b", "parsing", "failed", error="boom"))
            log.emit(make_event("sha256:c", "chunking", "success"))
            parsing_all = log.get_stage_events("parsing")
            assert len(parsing_all) == 2
            parsing_failed = log.get_stage_events("parsing", status="failed")
            assert len(parsing_failed) == 1
            assert parsing_failed[0].error == "boom"

    def test_get_failed_events(self, tmp_path):
        with TraceLog(tmp_path / "trace.db") as log:
            log.emit(make_event("sha256:a", "parsing", "success"))
            log.emit(make_event("sha256:b", "parsing", "failed", error="timeout"))
            log.emit(make_event("sha256:c", "indexing", "failed", error="disk full"))
            failed = log.get_failed_events()
            assert len(failed) == 2
            errors = {e.error for e in failed}
            assert "timeout" in errors
            assert "disk full" in errors

    def test_summary(self, tmp_path):
        with TraceLog(tmp_path / "trace.db") as log:
            log.emit(make_event("sha256:a", "landing", "success", latency_ms=5.0))
            log.emit(make_event("sha256:a", "parsing", "success", latency_ms=20.0))
            log.emit(make_event("sha256:b", "landing", "success", latency_ms=3.0))
            log.emit(make_event("sha256:b", "parsing", "failed", latency_ms=1.0, error="bad"))
            s = log.summary()
            assert s["total_events"] == 4
            assert s["artifacts"] == 2
            assert s["failed_events"] == 1
            assert "landing" in s["stages"]
            assert s["stages"]["landing"] == 2
            assert s["stages"]["parsing"] == 2

    def test_summary_empty_log(self, tmp_path):
        with TraceLog(tmp_path / "trace.db") as log:
            s = log.summary()
            assert s["total_events"] == 0
            assert s["artifacts"] == 0

    def test_persistence_across_reopen(self, tmp_path):
        db = tmp_path / "trace.db"
        with TraceLog(db) as log:
            log.emit(make_event("sha256:persist", "landing", "success", latency_ms=10.0))
        with TraceLog(db) as log2:
            assert log2.count() == 1
            events = log2.get_artifact_trace("sha256:persist")
            assert len(events) == 1
            assert events[0].stage == "landing"

    def test_metadata_with_unicode(self, tmp_path):
        with TraceLog(tmp_path / "trace.db") as log:
            log.emit(make_event(
                "sha256:uni", "parsing", "success",
                metadata={"filename": "café_niño.pdf", "author": "Müller"},
            ))
            events = log.get_artifact_trace("sha256:uni")
            assert events[0].metadata["filename"] == "café_niño.pdf"
            assert events[0].metadata["author"] == "Müller"

    def test_event_id_is_unique(self, tmp_path):
        with TraceLog(tmp_path / "trace.db") as log:
            ev1 = make_event("sha256:a", "parsing", "success")
            time.sleep(0.001)  # ensure different timestamp
            ev2 = make_event("sha256:a", "parsing", "success")
            assert ev1.event_id != ev2.event_id

    def test_count_by_stage(self, tmp_path):
        with TraceLog(tmp_path / "trace.db") as log:
            log.emit(make_event("sha256:a", "landing", "success"))
            log.emit(make_event("sha256:a", "parsing", "success"))
            log.emit(make_event("sha256:b", "landing", "success"))
            counts = log.count_by_stage()
            assert counts["landing"] == 2
            assert counts["parsing"] == 1


# --- make_event helper ---

class TestMakeEvent:
    def test_make_event_fills_fields(self):
        ev = make_event("sha256:test", "landing", "success", latency_ms=5.0)
        assert ev.artifact_id == "sha256:test"
        assert ev.stage == "landing"
        assert ev.status == "success"
        assert ev.latency_ms == 5.0
        assert ev.worker_id.startswith("pid-")
        assert ev.timestamp.endswith("Z")
        assert ev.error == ""
        assert ev.input_hash == ""
        assert ev.output_hash == ""
        assert ev.metadata == {}

    def test_make_event_with_error(self):
        ev = make_event("sha256:err", "parsing", "failed", error="exception text")
        assert ev.status == "failed"
        assert ev.error == "exception text"

    def test_make_event_with_metadata(self):
        ev = make_event("sha256:m", "chunking", "success", metadata={"count": 5})
        assert ev.metadata == {"count": 5}


# --- End-to-end traceability with FastPathRunner ---

class TestFastPathTracing:
    def test_ingest_with_trace_produces_events(self, tmp_path):
        """Ingesting a file with trace_log set produces events for every stage."""
        f = tmp_path / "input" / "test.txt"
        f.parent.mkdir()
        f.write_text("This is a test document for tracing. " * 20, encoding="utf-8")

        trace_db = tmp_path / "trace.db"
        with TraceLog(trace_db) as trace:
            with FastPathRunner(
                landing_db=tmp_path / "landing.db",
                store_db=tmp_path / "store.db",
                index_db=tmp_path / "index.db",
                trace_log=trace,
            ) as runner:
                result = runner.ingest(f)

            assert result.first_queryable
            events = trace.get_artifact_trace(result.artifact_id)
            stages = [e.stage for e in events]
            # All pipeline stages should be present
            assert "landing" in stages
            assert "mime" in stages
            assert "parsing" in stages
            assert "chunking" in stages
            assert "storing" in stages
            assert "indexing" in stages
            assert "pipeline" in stages
            # All events should be success
            for e in events:
                assert e.status == "success", f"Stage {e.stage} was {e.status}"
            # Latencies should be non-negative
            for e in events:
                assert e.latency_ms >= 0

    def test_ingest_without_trace_is_unaffected(self, tmp_path):
        """Ingesting without trace_log produces no trace events and works normally."""
        f = tmp_path / "input" / "test.txt"
        f.parent.mkdir()
        f.write_text("No tracing here. " * 20, encoding="utf-8")

        with FastPathRunner(
            landing_db=tmp_path / "landing.db",
            store_db=tmp_path / "store.db",
            index_db=tmp_path / "index.db",
        ) as runner:
            result = runner.ingest(f)
            assert result.first_queryable
            assert result.chunks_created > 0

    def test_trace_captures_parser_failure(self, tmp_path):
        """If parsing fails, the trace should record a failed parsing event."""
        # Create a file with .pdf extension but invalid content
        f = tmp_path / "input" / "fake.pdf"
        f.parent.mkdir()
        f.write_bytes(b"Not a real PDF file content")

        trace_db = tmp_path / "trace.db"
        with TraceLog(trace_db) as trace:
            with FastPathRunner(
                landing_db=tmp_path / "landing.db",
                store_db=tmp_path / "store.db",
                index_db=tmp_path / "index.db",
                trace_log=trace,
            ) as runner:
                result = runner.ingest(f)

            assert not result.first_queryable
            assert len(result.errors) > 0
            events = trace.get_artifact_trace(result.artifact_id)
            # Should have landing + mime + parsing(failed)
            stages = [e.stage for e in events]
            assert "landing" in stages
            assert "mime" in stages
            assert "parsing" in stages
            parsing_events = [e for e in events if e.stage == "parsing"]
            assert any(e.status == "failed" for e in parsing_events)

    def test_trace_metadata_contains_parser_id(self, tmp_path):
        """Parsing event metadata should contain the parser_id."""
        f = tmp_path / "input" / "test.txt"
        f.parent.mkdir()
        f.write_text("Metadata test for parser id. " * 20, encoding="utf-8")

        trace_db = tmp_path / "trace.db"
        with TraceLog(trace_db) as trace:
            with FastPathRunner(
                landing_db=tmp_path / "landing.db",
                store_db=tmp_path / "store.db",
                index_db=tmp_path / "index.db",
                trace_log=trace,
            ) as runner:
                result = runner.ingest(f)

            events = trace.get_artifact_trace(result.artifact_id)
            parsing_events = [e for e in events if e.stage == "parsing"]
            assert len(parsing_events) == 1
            assert "parser_id" in parsing_events[0].metadata

    def test_trace_metadata_contains_chunk_count(self, tmp_path):
        """Chunking event metadata should contain chunk_count."""
        f = tmp_path / "input" / "test.txt"
        f.parent.mkdir()
        f.write_text("Chunk count metadata test. " * 50, encoding="utf-8")

        trace_db = tmp_path / "trace.db"
        with TraceLog(trace_db) as trace:
            with FastPathRunner(
                landing_db=tmp_path / "landing.db",
                store_db=tmp_path / "store.db",
                index_db=tmp_path / "index.db",
                trace_log=trace,
            ) as runner:
                result = runner.ingest(f)

            events = trace.get_artifact_trace(result.artifact_id)
            chunking_events = [e for e in events if e.stage == "chunking"]
            assert len(chunking_events) == 1
            assert chunking_events[0].metadata["chunk_count"] == result.chunks_created

    def test_trace_output_hash_differs_per_document(self, tmp_path):
        """Two different documents should have different parsing output hashes."""
        f1 = tmp_path / "input" / "a.txt"
        f2 = tmp_path / "input" / "b.txt"
        f1.parent.mkdir()
        f1.write_text("Document A content for hashing. " * 20, encoding="utf-8")
        f2.write_text("Document B content for hashing. " * 20, encoding="utf-8")

        trace_db = tmp_path / "trace.db"
        with TraceLog(trace_db) as trace:
            with FastPathRunner(
                landing_db=tmp_path / "landing.db",
                store_db=tmp_path / "store.db",
                index_db=tmp_path / "index.db",
                trace_log=trace,
            ) as runner:
                r1 = runner.ingest(f1)
                r2 = runner.ingest(f2)

            events1 = trace.get_artifact_trace(r1.artifact_id)
            events2 = trace.get_artifact_trace(r2.artifact_id)
            parse1 = [e for e in events1 if e.stage == "parsing"][0]
            parse2 = [e for e in events2 if e.stage == "parsing"][0]
            assert parse1.output_hash != parse2.output_hash
            assert parse1.output_hash.startswith("sha256:")

    def test_trace_summary_after_multiple_ingests(self, tmp_path):
        """Summary should reflect multiple ingested artifacts."""
        for i in range(3):
            f = tmp_path / "input" / f"doc{i}.txt"
            if i == 0:
                f.parent.mkdir()
            f.write_text(f"Document number {i} for summary test. " * 20, encoding="utf-8")

        trace_db = tmp_path / "trace.db"
        with TraceLog(trace_db) as trace:
            with FastPathRunner(
                landing_db=tmp_path / "landing.db",
                store_db=tmp_path / "store.db",
                index_db=tmp_path / "index.db",
                trace_log=trace,
            ) as runner:
                for i in range(3):
                    runner.ingest(tmp_path / "input" / f"doc{i}.txt")

            s = trace.summary()
            assert s["artifacts"] == 3
            assert s["failed_events"] == 0
            assert s["stages"]["landing"] == 3
            assert s["stages"]["parsing"] == 3
            assert s["stages"]["pipeline"] == 3
