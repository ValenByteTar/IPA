"""Tests para el lock cross-process de VRAM (Ollama / ExL3 / bulk embeddings)."""
from __future__ import annotations

import os
import time

import pytest

from ipa.providers import vram_lock


@pytest.fixture()
def isolated_vram_lock(tmp_path, monkeypatch):
    path = tmp_path / "vram.lock"
    monkeypatch.setattr(vram_lock, "LOCK_PATH", path)
    monkeypatch.setattr(vram_lock, "_pid_alive_cache", {})
    return path


def test_pid_liveness_uses_win32_handles_without_subprocess(monkeypatch):
    import subprocess

    if os.name != "nt":
        pytest.skip("Win32 process-handle check")
    monkeypatch.setattr(vram_lock, "_pid_alive_cache", {})
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: pytest.fail("should not spawn tasklist"))
    assert vram_lock.pid_alive(os.getpid()) is True
    assert vram_lock.pid_alive(999_999_999) is False


def test_vram_lock_exclusive_and_owner_scoped(isolated_vram_lock):
    assert vram_lock.acquire("ollama") is True
    holder = vram_lock.holder()
    assert holder is not None
    assert holder["pid"] == os.getpid()
    assert holder["owner"] == "ollama"
    assert vram_lock.acquire("bulk_embedding") is False
    vram_lock.release("bulk_embedding")
    assert vram_lock.holder()["owner"] == "ollama"
    vram_lock.release("ollama")
    assert vram_lock.holder() is None


def test_vram_lock_allows_same_owner_refresh(isolated_vram_lock):
    assert vram_lock.acquire("bulk_embedding") is True
    first = vram_lock.holder()["ts"]
    time.sleep(0.01)
    assert vram_lock.acquire("bulk_embedding") is True
    assert vram_lock.holder()["ts"] >= first
    vram_lock.release("bulk_embedding")


def test_vram_lock_rejects_live_foreign_holder(isolated_vram_lock, monkeypatch):
    monkeypatch.setattr(vram_lock, "pid_alive", lambda pid: True)
    isolated_vram_lock.write_text("987654|exl3|" + str(time.time()), encoding="utf-8")
    assert vram_lock.acquire("bulk_embedding") is False
    assert vram_lock.holder()["owner"] == "exl3"


def test_vram_lock_reclaims_dead_holder(isolated_vram_lock, monkeypatch):
    monkeypatch.setattr(vram_lock, "pid_alive", lambda pid: False)
    isolated_vram_lock.write_text("987654|exl3|" + str(time.time()), encoding="utf-8")
    assert vram_lock.holder() is None
    assert vram_lock.acquire("bulk_embedding") is True
    vram_lock.release("bulk_embedding")


def test_vram_lock_reclaims_expired_holder(isolated_vram_lock, monkeypatch):
    monkeypatch.setattr(vram_lock, "pid_alive", lambda pid: True)
    isolated_vram_lock.write_text(
        f"987654|bulk_embedding|{time.time() - vram_lock.DEFAULT_TTL_S - 1}",
        encoding="utf-8",
    )
    assert vram_lock.holder() is None
    assert vram_lock.acquire("ollama") is True
    vram_lock.release("ollama")


def test_vram_lock_renews_only_its_owner(isolated_vram_lock):
    assert vram_lock.acquire("bulk_embedding") is True
    ts = vram_lock.holder()["ts"]
    time.sleep(0.01)
    assert vram_lock.renew("ollama") is False
    assert vram_lock.renew("bulk_embedding") is True
    assert vram_lock.holder()["ts"] > ts
    vram_lock.release("bulk_embedding")


def test_vram_lock_cleans_old_malformed_file(isolated_vram_lock):
    old = time.time() - 2
    isolated_vram_lock.write_text("not-a-lock", encoding="utf-8")
    os.utime(isolated_vram_lock, (old, old))
    assert vram_lock.holder() is None
    assert not isolated_vram_lock.exists()
    assert vram_lock.acquire("bulk_embedding") is True
    vram_lock.release("bulk_embedding")
