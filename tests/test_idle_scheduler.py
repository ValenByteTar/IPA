"""Tests del scheduler idle: tiers, prioridades, recursos, cooldowns, abort."""
from __future__ import annotations

import threading
import time

from ipa.agentic.idle_scheduler import (
    CycleContext, IdleScheduler, IdleTask,
    RES_AGENT_DB, RES_LLM, RES_USER_MODEL,
)


def _task(name, fn, *, tier=1, priority=10, resources=frozenset(),
          cooldown=0.0, min_idle=0.0, needs_llm=False):
    return IdleTask(name, tier, priority, fn, resources=frozenset(resources),
                    cooldown_seconds=cooldown, min_idle_minutes=min_idle,
                    needs_llm=needs_llm)


class _Provider:
    def __init__(self, loaded=True):
        self._loaded = loaded

    def is_loaded(self):
        return self._loaded


def test_tier1_runs_in_priority_order():
    order: list[str] = []
    scheduler = IdleScheduler([
        _task("c", lambda ctx: order.append("c"), priority=30),
        _task("a", lambda ctx: order.append("a"), priority=10),
        _task("b", lambda ctx: order.append("b"), priority=20),
    ], max_tier1_workers=1)
    scheduler.run_tier1(CycleContext())
    assert order == ["a", "b", "c"]


def test_shared_resource_serializes():
    """Dos tareas con el mismo recurso nunca se solapan."""
    active = {"n": 0, "max": 0}
    guard = threading.Lock()

    def body(ctx):
        with guard:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        time.sleep(0.05)
        with guard:
            active["n"] -= 1
        return {}

    scheduler = IdleScheduler([
        _task("x", body, priority=10, resources={RES_AGENT_DB}),
        _task("y", body, priority=11, resources={RES_AGENT_DB}),
    ], max_tier1_workers=2)
    scheduler.run_tier1(CycleContext())
    assert active["max"] == 1


def test_disjoint_resources_run_in_parallel():
    """Tareas con recursos disjuntos se paralelizan (el pool las solapa)."""
    active = {"n": 0, "max": 0}
    guard = threading.Lock()

    def body(ctx):
        with guard:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        time.sleep(0.08)
        with guard:
            active["n"] -= 1
        return {}

    scheduler = IdleScheduler([
        _task("a", body, priority=10, resources={RES_USER_MODEL}),
        _task("b", body, priority=11, resources={"skills"}),
        _task("c", body, priority=12, resources={"strategic"}),
    ], max_tier1_workers=3)
    scheduler.run_tier1(CycleContext())
    assert active["max"] >= 2


def test_cooldown_skips_second_run():
    calls = {"n": 0}
    scheduler = IdleScheduler([
        _task("t", lambda ctx: calls.__setitem__("n", calls["n"] + 1), cooldown=60),
    ])
    scheduler.run_tier1(CycleContext())
    outcomes = scheduler.run_tier1(CycleContext())
    assert calls["n"] == 1
    assert any(o.skipped and "cooldown" in o.skipped for o in outcomes)


def test_min_idle_gate():
    ran = {"v": False}
    scheduler = IdleScheduler([
        _task("t", lambda ctx: ran.__setitem__("v", True), min_idle=5.0),
    ])
    scheduler.run_tier1(CycleContext(idle_minutes=1.0))
    assert ran["v"] is False
    scheduler.run_tier1(CycleContext(idle_minutes=6.0))
    assert ran["v"] is True


def test_needs_llm_gate():
    ran = {"v": False}
    scheduler = IdleScheduler([
        _task("t", lambda ctx: ran.__setitem__("v", True), needs_llm=True),
    ])
    outcomes = scheduler.run_tier1(CycleContext(provider=None))
    assert ran["v"] is False
    assert any(o.skipped == "llm not loaded" for o in outcomes)
    scheduler.run_tier1(CycleContext(provider=_Provider(True)))
    assert ran["v"] is True


def test_tier2_is_serial_and_abortable():
    order: list[str] = []
    scheduler = IdleScheduler([
        _task("deep1", lambda ctx: order.append("deep1"), tier=2, priority=10),
        _task("deep2", lambda ctx: order.append("deep2"), tier=2, priority=20),
    ])
    # Sin abort corre todo, en orden de prioridad
    scheduler.run_tier2(CycleContext(provider=_Provider()))
    assert order == ["deep1", "deep2"]

    order.clear()
    scheduler2 = IdleScheduler([
        _task("deep1", lambda ctx: order.append("deep1"), tier=2, priority=10),
        _task("deep2", lambda ctx: order.append("deep2"), tier=2, priority=20),
    ])
    # Con abort (usuario activo) no corre ninguna
    outcomes = scheduler2.run_tier2(CycleContext(provider=_Provider(), should_abort=lambda: True))
    assert order == []
    assert all(o.skipped and "aborted" in o.skipped for o in outcomes)


def test_failure_isolated():
    """Una tarea que falla no tumba el ciclo ni a las demás."""
    done = {"v": False}

    def boom(ctx):
        raise RuntimeError("explotó")

    scheduler = IdleScheduler([
        _task("bad", boom, priority=10),
        _task("good", lambda ctx: done.__setitem__("v", True), priority=20),
    ], max_tier1_workers=2)
    outcomes = scheduler.run_tier1(CycleContext())
    assert done["v"] is True
    bad = next(o for o in outcomes if o.name == "bad")
    assert bad.ok is False and "explotó" in bad.error


def test_tier1_not_started_when_aborted():
    ran = {"v": False}
    scheduler = IdleScheduler([
        _task("t", lambda ctx: ran.__setitem__("v", True)),
    ])
    scheduler.run_tier1(CycleContext(should_abort=lambda: True))
    assert ran["v"] is False


def test_duplicate_task_names_rejected():
    import pytest
    with pytest.raises(ValueError, match="duplicate"):
        IdleScheduler([
            _task("same", lambda ctx: {}),
            _task("same", lambda ctx: {}),
        ])


def test_invalid_tier_rejected():
    import pytest
    with pytest.raises(ValueError, match="tier"):
        IdleTask("t", 3, 10, lambda ctx: {})
