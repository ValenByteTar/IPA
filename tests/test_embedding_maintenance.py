"""Tests para estado durable y chat gate del lote de embeddings GPU."""
from __future__ import annotations

import json
import os

import pytest

from ipa.agentic import embedding_maintenance as maintenance


@pytest.fixture()
def isolated_maintenance(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance, "STATE_PATH", tmp_path / "embedding.json")
    monkeypatch.setattr(maintenance, "JOB_LOCK_PATH", tmp_path / "embedding.lock")
    return tmp_path


def test_job_claim_is_exclusive_and_released(isolated_maintenance):
    assert maintenance.claim_job() is True
    assert maintenance.claim_job() is False
    maintenance.release_job()
    assert maintenance.claim_job() is True
    maintenance.release_job()


def test_job_owner_serializes_idle_scheduler_and_embedding_drain(isolated_maintenance):
    assert maintenance.claim_job("idle_scheduler") is True
    assert maintenance.job_active() is True
    assert maintenance.job_active(exclude_owner="idle_scheduler") is False
    assert maintenance.claim_job("embedding_drain") is False
    assert maintenance.renew_job("idle_scheduler") is True
    maintenance.release_job("idle_scheduler")
    assert maintenance.job_active() is False
    assert maintenance.claim_job("embedding_drain") is True
    maintenance.release_job("embedding_drain")


def test_job_claim_reclaims_dead_pid(isolated_maintenance, monkeypatch):
    from ipa.providers import vram_lock

    monkeypatch.setattr(vram_lock, "pid_alive", lambda pid: False)
    maintenance.JOB_LOCK_PATH.write_text("987654|0", encoding="utf-8")
    assert maintenance.claim_job() is True
    maintenance.release_job()


def test_state_is_atomic_and_chat_is_blocked_until_terminal(isolated_maintenance):
    maintenance.STATE_PATH.write_text(json.dumps({
        "vectorized": 999, "pending": 0, "chunks_per_second": 7,
        "eta_seconds": 0,
    }), encoding="utf-8")
    maintenance.start_state(
        corpus="corpus", total_chunks=1000, vectorized=100, pending=900,
        mode="bulk_gpu",
    )
    state = maintenance.read_state()
    assert state["status"] == "preparing"
    assert state["chat_blocked"] is True
    assert state["vectorized"] == 100
    assert state["pending"] == 900
    assert state["chunks_per_second"] == 0
    assert state["eta_seconds"] is None
    assert json.loads(maintenance.STATE_PATH.read_text(encoding="utf-8"))["pid"] == os.getpid()
    assert not maintenance.STATE_PATH.with_name(
        f"{maintenance.STATE_PATH.name}.{os.getpid()}.tmp").exists()

    maintenance.update_state(status="completed", phase="complete", chat_blocked=False)
    assert maintenance.read_state()["chat_blocked"] is False


def test_dead_worker_marks_interrupted_and_reenables_chat(
        isolated_maintenance, monkeypatch):
    from ipa.providers import vram_lock

    maintenance.start_state(
        corpus="corpus", total_chunks=1000, vectorized=100, pending=900,
        mode="bulk_gpu",
    )
    monkeypatch.setattr(vram_lock, "pid_alive", lambda pid: False)
    state = maintenance.read_state()
    assert state["status"] == "interrupted"
    assert state["chat_blocked"] is False
    assert "reanudar" in state["error"].lower()
    assert maintenance.chat_block_reason() is None


def test_chat_block_reason_is_clear_and_localized(isolated_maintenance):
    maintenance.start_state(
        corpus="corpus", total_chunks=1000, vectorized=100, pending=900,
        mode="bulk_gpu",
    )
    assert "ingesta masiva" in maintenance.chat_block_reason()
