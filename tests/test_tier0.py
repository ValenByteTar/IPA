"""Tests for the Tier-0 ingestion lease (ipa.agentic.tier0).

Tier 0 gatea los ciclos idle Tier 1/T2: mientras una ingesta fast_path está
viva, el scheduler no puede arrancar — y la cuenta de idle solo empieza
cuando el lease se libera.
"""
from __future__ import annotations

import os
import time

import pytest

from ipa.agentic import tier0


@pytest.fixture
def lock(tmp_path, monkeypatch):
    path = tmp_path / "tier0.lock"
    monkeypatch.setattr(tier0, "LOCK_PATH", path)
    monkeypatch.setattr(tier0, "HEARTBEAT_S", 0.05)
    return path


class TestClaimRelease:
    def test_claim_then_release(self, lock):
        assert tier0.claim("fast_path") is True
        h = tier0.holder()
        assert h["pid"] == os.getpid()
        assert h["owner"] == "fast_path"
        assert tier0.active() is True
        tier0.release()
        assert tier0.active() is False
        assert not lock.exists()

    def test_second_process_cannot_claim(self, lock):
        # Simula otro proceso vivo: el parent del test existe y no somos nosotros.
        lock.write_text(f"{os.getppid()}|fast_path|{time.time()}", encoding="utf-8")
        assert tier0.claim("fast_path") is False
        assert tier0.active() is True

    def test_stale_holder_is_stolen(self, lock):
        # PID muerto (o ts expirado) → el lease se recupera.
        lock.write_text(f"{os.getppid()}|fast_path|{time.time() - tier0.TTL_S - 1}",
                        encoding="utf-8")
        assert tier0.claim("fast_path") is True
        assert tier0.holder()["pid"] == os.getpid()

    def test_release_never_removes_foreign_lock(self, lock):
        lock.write_text(f"{os.getppid()}|fast_path|{time.time()}", encoding="utf-8")
        tier0.release()
        assert lock.exists()  # el lease ajeno queda intacto


class TestHeartbeat:
    def test_heartbeat_renews_timestamp(self, lock):
        tier0.claim("fast_path")
        first = tier0.holder()["ts"]
        tier0.start_heartbeat("fast_path")
        try:
            time.sleep(0.3)
            assert tier0.holder()["ts"] > first
        finally:
            tier0.stop_heartbeat()

    def test_holder_survives_beyond_ttl_with_heartbeat(self, lock, monkeypatch):
        monkeypatch.setattr(tier0, "TTL_S", 0.15)
        tier0.claim("fast_path")
        tier0.start_heartbeat("fast_path")
        try:
            time.sleep(0.4)  # > TTL: solo el heartbeat lo mantiene vivo
            assert tier0.active() is True
        finally:
            tier0.stop_heartbeat()
