"""Tests for ipa.dashboard.process_state, process_specs, and process_runner.

Validates the JobSpec/JobRunner replacement for scripts/proc_*.py.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from ipa.dashboard import process_specs as specs_mod
from ipa.dashboard import process_state as ps
from ipa.dashboard.process_runner import JobRunner
from ipa.dashboard.process_specs import SPECS, JobSpec, get_spec, spec_names


# ---------------------------------------------------------------------------
# process_state
# ---------------------------------------------------------------------------

class TestProcessState:
    def test_build_state_schema(self, tmp_path):
        state = ps.build_state("scraper", "running", {"x": 1}, ["err1"])
        assert state["process"] == "scraper"
        assert state["status"] == "running"
        assert state["pid"] == os.getpid()
        assert state["timestamp"] > 0
        assert state["metrics"] == {"x": 1}
        assert state["errors"] == ["err1"]
        assert state["run_id"] == os.environ.get("IPA_RUN_ID")

    def test_build_state_invalid_status(self):
        with pytest.raises(ValueError, match="Invalid status"):
            ps.build_state("x", "bogus", {})

    def test_build_state_truncates_errors(self):
        errs = [f"e{i}" for i in range(20)]
        state = ps.build_state("x", "running", {}, errs)
        assert len(state["errors"]) == ps.MAX_ERRORS
        assert state["errors"] == errs[-ps.MAX_ERRORS:]

    def test_write_state_atomic_roundtrip(self, tmp_path):
        ps.write_state_atomic(tmp_path, "scraper", "running", {"a": 1})
        state = ps.read_state(tmp_path, "scraper")
        assert state is not None
        assert state["process"] == "scraper"
        assert state["status"] == "running"
        assert state["metrics"] == {"a": 1}

    def test_write_state_simple_roundtrip(self, tmp_path):
        ps.write_state_simple(tmp_path, "hammer", "done", {"rounds": 5})
        state = ps.read_state(tmp_path, "hammer")
        assert state is not None
        assert state["status"] == "done"
        assert state["metrics"]["rounds"] == 5

    def test_read_state_missing(self, tmp_path):
        assert ps.read_state(tmp_path, "nonexistent") is None

    def test_read_state_corrupt(self, tmp_path):
        (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
        assert ps.read_state(tmp_path, "bad") is None

    def test_state_path(self, tmp_path):
        p = ps.state_path(tmp_path, "scraper")
        assert p == tmp_path / "scraper.json"


# ---------------------------------------------------------------------------
# process_specs
# ---------------------------------------------------------------------------

class TestProcessSpecs:
    def test_all_six_specs_present(self):
        expected = {"scraper", "pipeline", "lancedb", "hammer", "enrichment", "rechunk"}
        assert set(spec_names()) == expected

    def test_get_spec_valid(self):
        spec = get_spec("scraper")
        assert spec.name == "scraper"
        assert spec.parse_line is not None

    def test_get_spec_invalid(self):
        with pytest.raises(KeyError):
            get_spec("bogus")

    def test_gpu_flags(self):
        assert SPECS["scraper"].needs_gpu is False
        assert SPECS["pipeline"].needs_gpu is False
        assert SPECS["lancedb"].needs_gpu is True
        assert SPECS["hammer"].needs_gpu is True
        assert SPECS["enrichment"].needs_gpu is True
        assert SPECS["rechunk"].needs_gpu is True

    def test_atomic_flags(self):
        # scraper, hammer, rechunk use simple writes (legacy parity)
        assert SPECS["scraper"].use_atomic_state is False
        assert SPECS["hammer"].use_atomic_state is False
        assert SPECS["rechunk"].use_atomic_state is False
        # pipeline, lancedb, enrichment use atomic writes
        assert SPECS["pipeline"].use_atomic_state is True
        assert SPECS["lancedb"].use_atomic_state is True
        assert SPECS["enrichment"].use_atomic_state is True

    def test_resolve_worker_path_existing(self, tmp_path):
        spec = SPECS["scraper"]
        # run_web_scrape.py exists in scripts/
        root = Path(__file__).resolve().parents[1]
        path = spec.resolve_worker_path(root)
        assert path.exists()

    def test_resolve_worker_path_archive_fallback(self, tmp_path):
        """lancedb worker is archived; resolve should find it in local_archive."""
        spec = SPECS["lancedb"]
        root = Path(__file__).resolve().parents[1]
        path = spec.resolve_worker_path(root)
        # Either scripts/ or local_archive/scripts/ should have it
        assert path.exists()
        assert "_lancedb_incremental" in path.name

    def test_build_command_basic(self, tmp_path):
        spec = SPECS["scraper"]
        cmd = spec.build_command(tmp_path)
        assert cmd[0]  # python executable
        assert "-u" in cmd
        assert "--config" in cmd

    def test_build_command_overrides(self, tmp_path):
        spec = SPECS["pipeline"]
        cmd = spec.build_command(tmp_path, {"chunker": "semantic"})
        idx = cmd.index("--chunker")
        assert cmd[idx + 1] == "semantic"

    def test_build_env(self):
        spec = SPECS["enrichment"]
        env = spec.build_env()
        assert env["PYTHONPATH"] == "src"
        assert env["PYTHONIOENCODING"] == "utf-8"
        assert "CUDA_PATH" in env

    def test_compute_status_running(self):
        spec = SPECS["scraper"]
        assert spec.compute_status(10, {}) == "running"

    def test_compute_status_stuck(self):
        spec = SPECS["scraper"]
        assert spec.compute_status(200, {}) == "stuck"

    def test_compute_status_done(self):
        spec = SPECS["scraper"]
        assert spec.compute_status(0, {"complete": True}) == "done"

    def test_compute_status_idle(self):
        spec = SPECS["pipeline"]
        assert spec.compute_status(70, {}) == "idle"

    def test_final_status_done_on_complete(self):
        spec = SPECS["lancedb"]
        assert spec.final_status(1, {"complete": True}) == "done"

    def test_final_status_error_on_nonzero(self):
        spec = SPECS["scraper"]
        assert spec.final_status(1, {}) == "error"

    def test_final_exit_code_complete_policy(self):
        spec = SPECS["lancedb"]  # exit_policy="complete"
        assert spec.final_exit_code(1, {"complete": True}) == 0
        assert spec.final_exit_code(1, {}) == 1

    def test_final_exit_code_exit_code_policy(self):
        spec = SPECS["scraper"]  # exit_policy="exit_code"
        assert spec.final_exit_code(0, {}) == 0
        assert spec.final_exit_code(1, {}) == 1


# ---------------------------------------------------------------------------
# Parser parity — each parser matches the legacy proc_*.py behavior
# ---------------------------------------------------------------------------

class TestParserParity:
    def test_scraper_found(self):
        spec = SPECS["scraper"]
        m = spec.parse_line("[https://example.com] Found: 5 articles")
        assert m["last_site"] == "https://example.com"
        assert m["found"] == 5

    def test_scraper_scraped(self):
        spec = SPECS["scraper"]
        m = spec.parse_line("[https://example.com] Scraped: 3, Skipped: 2")
        assert m["scraped"] == 3
        assert m["skipped"] == 2

    def test_scraper_saved(self):
        spec = SPECS["scraper"]
        m = spec.parse_line("[https://example.com] Saved: file.html (1234 chars, 0 imgs)")
        assert m["last_file"] == "file.html"
        assert m["last_file_chars"] == 1234

    def test_scraper_complete(self):
        spec = SPECS["scraper"]
        m = spec.parse_line("Scrape complete in 45.2s")
        assert m["complete"] is True
        assert m["elapsed"] == 45.2

    def test_scraper_errors(self):
        spec = SPECS["scraper"]
        m = spec.parse_line("Errors: 3")
        assert m["errors"] == 3

    def test_pipeline_found_files(self):
        spec = SPECS["pipeline"]
        m = spec.parse_line("Found 3 new file(s)")
        assert m["new_files"] == 3

    def test_pipeline_ok(self):
        spec = SPECS["pipeline"]
        m = spec.parse_line("OK: application/pdf → 12 chunks, 5 pages")
        assert m["last_mime"] == "application/pdf"
        assert m["last_chunks"] == 12
        assert m["last_pages"] == 5
        assert m["file_processed"] is True

    def test_pipeline_complete(self):
        spec = SPECS["pipeline"]
        m = spec.parse_line("Pipeline complete")
        assert m["complete"] is True

    def test_pipeline_lock_conflict(self):
        spec = SPECS["pipeline"]
        m = spec.parse_line("Lock conflict on document_store.db")
        assert m["lock_conflict"] is True

    def test_lancedb_round(self):
        spec = SPECS["lancedb"]
        m = spec.parse_line("[Round 3] +50 chunks → 500 total rows | Store: 500, Embedded: 450, Missing: 50")
        assert m["round"] == 3
        assert m["total_rows"] == 500
        assert m["embedded"] == 450
        assert m["missing"] == 50

    def test_lancedb_complete(self):
        spec = SPECS["lancedb"]
        m = spec.parse_line("Incremental LanceDB builder complete")
        assert m["complete"] is True

    def test_lancedb_skip_noise(self):
        spec = SPECS["lancedb"]
        assert spec.skip_line("Inference: 100%|████████| 50/50 [00:01<00:00, 45.6it/s]") is True
        assert spec.skip_line("pre tokenize batch") is True
        assert spec.skip_line("[Round 1] +10 chunks") is False

    def test_hammer_round(self):
        spec = SPECS["hammer"]
        m = spec.parse_line("[Round 5] Docs: 100, Chunks: 500")
        assert m["round"] == 5
        assert m["docs"] == 100
        assert m["chunks"] == 500

    def test_hammer_latency(self):
        spec = SPECS["hammer"]
        m = spec.parse_line("Query latency: p50=12.5ms p99=45.3ms | Hits: 42")
        assert m["p50_ms"] == 12.5
        assert m["p99_ms"] == 45.3
        assert m["hits"] == 42

    def test_enrichment_progress(self):
        spec = SPECS["enrichment"]
        m = spec.parse_line("[10/50] enriched=8 reembedded=2 errors=0 | 30.5s | 0.33 chunks/s | ETA: 120.5s")
        assert m["completed"] == 10
        assert m["total_to_process"] == 50
        assert m["enriched"] == 8
        assert m["reembedded"] == 2
        assert m["rate"] == 0.33

    def test_enrichment_complete(self):
        spec = SPECS["enrichment"]
        m = spec.parse_line("ExLlamaV3 enrichment complete")
        assert m["complete"] is True

    def test_enrichment_no_work(self):
        spec = SPECS["enrichment"]
        m = spec.parse_line("No chunks need enrichment or re-embedding")
        assert m["complete"] is True
        assert m["no_work"] is True

    def test_rechunk_progress(self):
        spec = SPECS["rechunk"]
        m = spec.parse_line("[5/20] doc abc → 8 chunks (total: 40, 12.5s, ETA: 75.0s)")
        assert m["completed"] == 5
        assert m["total_docs"] == 20
        assert m["last_doc_chunks"] == 8
        assert m["total_new_chunks"] == 40

    def test_rechunk_complete(self):
        spec = SPECS["rechunk"]
        m = spec.parse_line("Re-chunking complete")
        assert m["complete"] is True


# ---------------------------------------------------------------------------
# JobRunner integration (uses a tiny fake worker)
# ---------------------------------------------------------------------------

class TestJobRunnerIntegration:
    def test_runner_writes_done_state(self, tmp_path):
        """JobRunner with a fake worker that exits 0 should write done state."""
        # Create a fake worker script
        worker = tmp_path / "fake_worker.py"
        worker.write_text(
            "import time\n"
            "print('Found 1 new file(s)')\n"
            "print('OK: text/plain → 2 chunks, 1 pages')\n"
            "print('Pipeline complete')\n"
            "time.sleep(0.1)\n",
            encoding="utf-8",
        )
        spec = JobSpec(
            name="testjob",
            worker_script=str(worker),
            needs_gpu=False,
            idle_threshold=60,
            use_atomic_state=True,
            parse_line=specs_mod._parse_pipeline,
        )
        state_dir = tmp_path / "state"
        runner = JobRunner(spec, project_root=tmp_path, state_dir=state_dir, run_id="test-123")
        exit_code = runner.run()
        assert exit_code == 0
        state = ps.read_state(state_dir, "testjob")
        assert state is not None
        assert state["status"] == "done"
        assert state["run_id"] == "test-123"
        assert state["metrics"].get("complete") is True

    def test_runner_writes_error_state(self, tmp_path):
        """JobRunner with a worker that exits 1 should write error state."""
        worker = tmp_path / "fail_worker.py"
        worker.write_text(
            "import sys\n"
            "print('ERROR: something broke')\n"
            "sys.exit(1)\n",
            encoding="utf-8",
        )
        spec = JobSpec(
            name="failjob",
            worker_script=str(worker),
            needs_gpu=False,
            idle_threshold=60,
            use_atomic_state=False,
            parse_line=specs_mod._parse_pipeline,
        )
        state_dir = tmp_path / "state"
        runner = JobRunner(spec, project_root=tmp_path, state_dir=state_dir)
        exit_code = runner.run()
        assert exit_code == 1
        state = ps.read_state(state_dir, "failjob")
        assert state is not None
        assert state["status"] == "error"
        assert state["metrics"]["exit_code"] == 1

    def test_runner_captures_errors(self, tmp_path):
        """JobRunner should capture ERROR lines into the errors list."""
        worker = tmp_path / "err_worker.py"
        worker.write_text(
            "print('ERROR: first error')\n"
            "print('ERROR: second error')\n",
            encoding="utf-8",
        )
        spec = JobSpec(
            name="errjob",
            worker_script=str(worker),
            needs_gpu=False,
            idle_threshold=60,
            use_atomic_state=True,
            parse_line=specs_mod._parse_pipeline,
        )
        state_dir = tmp_path / "state"
        runner = JobRunner(spec, project_root=tmp_path, state_dir=state_dir)
        runner.run()
        state = ps.read_state(state_dir, "errjob")
        assert state is not None
        assert len(state["errors"]) == 2
        assert "first error" in state["errors"][0]
