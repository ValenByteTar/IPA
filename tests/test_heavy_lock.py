"""Tests del lock de trabajos pesados (serialización CPU/IO entre procesos).

Cubre: adquisición/liberación, contención entre procesos, robo de locks
stale (proceso muerto o TTL vencido), re-entrada del mismo proceso, registro
de waiters con prioridad (cesión del background) y el context manager
``heavy_phase`` (espera acotada + callback de espera).
"""
from __future__ import annotations

import os
import threading
import time

import pytest

from ipa.agentic import heavy_lock


@pytest.fixture()
def lock_paths(tmp_path, monkeypatch):
    """Aísla el lock y el registry de waiters en tmp_path."""
    lock = tmp_path / "heavy.lock"
    wait = tmp_path / "heavy.waiting"
    monkeypatch.setattr(heavy_lock, "LOCK_PATH", lock)
    monkeypatch.setattr(heavy_lock, "WAIT_PATH", wait)
    return lock, wait


def test_pid_liveness_uses_win32_handles_without_subprocess(monkeypatch):
    import subprocess

    if os.name != "nt":
        pytest.skip("Win32 process-handle check")
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: pytest.fail("should not spawn tasklist"))
    assert heavy_lock._pid_alive_uncached(os.getpid()) is True
    assert heavy_lock._pid_alive_uncached(999_999_999) is False


def _foreign_holder(lock_path, pid=424242, kind="pipeline", priority=50, ts=None):
    """Escribe un holder ajeno (simula otro proceso)."""
    lock_path.write_text(
        f"{pid}|{kind}|{priority}|{ts if ts is not None else time.time()}",
        encoding="utf-8",
    )


def test_acquire_and_release_roundtrip(lock_paths):
    assert heavy_lock.holder() is None
    assert heavy_lock.try_acquire("research", heavy_lock.PRIORITY_INTERACTIVE)
    current = heavy_lock.holder()
    assert current is not None
    assert current["pid"] == os.getpid()
    assert current["kind"] == "research"
    assert current["priority"] == heavy_lock.PRIORITY_INTERACTIVE
    heavy_lock.release()
    assert heavy_lock.holder() is None


def test_second_process_is_blocked(lock_paths, monkeypatch):
    lock, _ = lock_paths
    monkeypatch.setattr(heavy_lock, "_pid_alive", lambda pid: True)
    _foreign_holder(lock)
    assert heavy_lock.try_acquire("research") is False


def test_dead_holder_lock_is_stolen(lock_paths):
    lock, _ = lock_paths
    # pid inexistente: holder() lo considera stale y lo limpia.
    _foreign_holder(lock, pid=999_999_999)
    assert heavy_lock.holder() is None
    assert heavy_lock.try_acquire("research")


def test_expired_ttl_lock_is_stolen(lock_paths, monkeypatch):
    lock, _ = lock_paths
    monkeypatch.setattr(heavy_lock, "_pid_alive", lambda pid: True)
    _foreign_holder(lock, ts=time.time() - heavy_lock.DEFAULT_TTL_S - 10)
    assert heavy_lock.holder() is None
    assert heavy_lock.try_acquire("research")


def test_same_process_is_reentrant(lock_paths):
    assert heavy_lock.try_acquire("research", heavy_lock.PRIORITY_INTERACTIVE)
    assert heavy_lock.try_acquire("research_ingest", heavy_lock.PRIORITY_INTERACTIVE)
    current = heavy_lock.holder()
    assert current is not None and current["kind"] == "research_ingest"


def test_waiter_registry_and_should_yield(lock_paths):
    assert heavy_lock.waiters() == []
    assert heavy_lock.should_yield(heavy_lock.PRIORITY_BACKGROUND) is False

    heavy_lock.register_waiter("research", heavy_lock.PRIORITY_INTERACTIVE)
    waiters = heavy_lock.waiters()
    assert len(waiters) == 1
    assert waiters[0]["kind"] == "research"
    # El background cede ante un waiter interactivo...
    assert heavy_lock.should_yield(heavy_lock.PRIORITY_BACKGROUND) is True
    # ...pero un waiter de igual/menor prioridad no lo hace ceder.
    assert heavy_lock.should_yield(heavy_lock.PRIORITY_INTERACTIVE) is False

    heavy_lock.unregister_waiter()
    assert heavy_lock.waiters() == []
    assert heavy_lock.should_yield(heavy_lock.PRIORITY_BACKGROUND) is False


def test_dead_waiters_are_pruned(lock_paths):
    _, wait = lock_paths
    wait.write_text(
        '[{"pid": 999999999, "kind": "research", "priority": 10, "ts": 0}]',
        encoding="utf-8",
    )
    assert heavy_lock.waiters() == []
    assert heavy_lock.should_yield(heavy_lock.PRIORITY_BACKGROUND) is False


def test_heavy_phase_acquires_when_free(lock_paths):
    with heavy_lock.heavy_phase("research", heavy_lock.PRIORITY_INTERACTIVE) as held:
        assert held is True
        assert heavy_lock.holder() is not None
    assert heavy_lock.holder() is None


def test_heavy_phase_returns_false_without_waiting(lock_paths, monkeypatch):
    lock, _ = lock_paths
    monkeypatch.setattr(heavy_lock, "_pid_alive", lambda pid: True)
    _foreign_holder(lock)
    # wait_s=0: no espera, la fase corre igual (best-effort, nunca deadlockea).
    with heavy_lock.heavy_phase("research", heavy_lock.PRIORITY_INTERACTIVE) as held:
        assert held is False
    # El lock ajeno sigue intacto: no lo pisamos.
    assert heavy_lock.holder() is not None


def test_heavy_phase_waits_and_acquires_when_released(lock_paths, monkeypatch):
    lock, _ = lock_paths
    monkeypatch.setattr(heavy_lock, "_pid_alive", lambda pid: True)
    _foreign_holder(lock)

    def _release_later():
        time.sleep(0.4)
        lock.unlink()

    threading.Thread(target=_release_later, daemon=True).start()
    seen: list[tuple[float, str | None]] = []
    with heavy_lock.heavy_phase(
            "research", heavy_lock.PRIORITY_INTERACTIVE, wait_s=5.0,
            on_wait=lambda elapsed, current: seen.append(
                (elapsed, (current or {}).get("kind")))) as held:
        assert held is True
    assert seen, "el callback de espera debe reportar el holder"
    assert seen[0][1] == "pipeline"


def test_heavy_phase_gives_up_after_wait_s(lock_paths, monkeypatch):
    lock, _ = lock_paths
    monkeypatch.setattr(heavy_lock, "_pid_alive", lambda pid: True)
    _foreign_holder(lock)
    started = time.monotonic()
    with heavy_lock.heavy_phase(
            "research", heavy_lock.PRIORITY_INTERACTIVE, wait_s=0.6) as held:
        assert held is False
    assert 0.5 <= time.monotonic() - started < 5
    # No queda registrado como waiter al salir.
    assert heavy_lock.waiters() == []


def test_background_holder_can_be_yielded(lock_paths):
    """Escenario del incidente: el drain (background) cede al research."""
    assert heavy_lock.try_acquire("fast_path_embed", heavy_lock.PRIORITY_BACKGROUND)
    heavy_lock.register_waiter("research", heavy_lock.PRIORITY_INTERACTIVE)
    assert heavy_lock.should_yield(heavy_lock.PRIORITY_BACKGROUND) is True
    heavy_lock.release()
    assert heavy_lock.holder() is None
    assert heavy_lock.waiters() == []
