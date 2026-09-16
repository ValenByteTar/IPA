"""Scheduler de procesos idle — Tier 1 (paralelo, sin VRAM) / Tier 2 (LLM, serial).

Antes cada proceso idle era su propio thread con su propio `while True` +
`sleep`, sin orden ni coordinación: el clustering, la consolidación de
sesiones, el review de research y las inferencias cognitivas competían entre
sí y con el chat sin un modelo explícito de recursos.

Este módulo define el modelo:

  - Cada tarea declara TIER, PRIORIDAD, RECURSOS y COOLDOWN.
  - Los recursos (`llm`, `embeddings`, `cluster_store`, `agent_db`, ...) son
    locks nombrados: dos tareas que comparten recurso se serializan solas.
  - Tier 1 corre en un pool de threads (2-3): las tareas con recursos
    disjuntos avanzan en paralelo; las que comparten store se serializan.
  - Tier 2 corre en un único pase serial (el generator no es thread-safe) y
    es preemptible entre items vía `ctx.should_abort()`.
  - Los locks se toman siempre en orden alfabético → sin deadlocks.

El scheduler NO conoce el dashboard: recibe un `CycleContext` con lo que
las tareas necesiten (provider, episodios pre-cargados, callables de log) y
solo orquesta. Los cuerpos de las tareas viven en server.py.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

# Recursos compartidos (locks nombrados). Dos tareas que declaren el mismo
# recurso nunca corren a la vez.
RES_LLM = "llm"
RES_EMBEDDINGS = "embeddings"
RES_CLUSTER_STORE = "cluster_store"
RES_AGENT_DB = "agent_db"
RES_TUTOR_DB = "tutor_db"
RES_USER_MODEL = "user_model"
RES_SKILLS = "skills"
RES_STRATEGIC = "strategic"
RES_UNCERTAINTY = "uncertainty"
RES_CONSOLIDATION = "consolidation"
RES_CORPUS_MAIN = "corpus_main"
RES_CORPUS_REPORTER = "corpus_reporter"


@dataclass
class CycleContext:
    """Lo que las tareas reciben. Construido una vez por ciclo."""

    idle_minutes: float = 0.0
    provider: Any = None
    episodes: list[dict[str, Any]] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    should_abort: Callable[[], bool] = field(default=lambda: False)
    log: Callable[..., None] = field(default=lambda *a, **k: None)
    extra: dict[str, Any] = field(default_factory=dict)

    def provider_loaded(self) -> bool:
        return self.provider is not None and bool(getattr(self.provider, "is_loaded", lambda: False)())


@dataclass(frozen=True)
class IdleTask:
    """Una unidad de trabajo idle con sus requisitos declarados."""

    name: str
    tier: int  # 1 = sin VRAM (paralelizable) | 2 = LLM (serial)
    priority: int  # menor = antes dentro del tier
    fn: Callable[[CycleContext], dict[str, Any]]
    resources: frozenset[str] = frozenset()
    cooldown_seconds: float = 300.0
    min_idle_minutes: float = 0.0
    needs_llm: bool = False

    def __post_init__(self) -> None:
        if self.tier not in (1, 2):
            raise ValueError("tier must be 1 or 2")


@dataclass
class TaskOutcome:
    name: str
    ok: bool
    duration_s: float
    result: dict[str, Any] | None = None
    error: str | None = None
    skipped: str | None = None


class IdleScheduler:
    """Orquesta las tareas idle por tier, prioridad y recursos."""

    def __init__(self, tasks: list[IdleTask], *, max_tier1_workers: int = 3) -> None:
        names = [t.name for t in tasks]
        if len(set(names)) != len(names):
            raise ValueError("duplicate task names")
        self.tasks = list(tasks)
        self.max_tier1_workers = max(1, max_tier1_workers)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._last_run: dict[str, float] = {}
        self._lock = threading.Lock()

    # ── recursos ──────────────────────────────────────────────────────
    def _resource_lock(self, resource: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(resource)
            if lock is None:
                lock = threading.Lock()
                self._locks[resource] = lock
            return lock

    class _Held:
        """Context manager que toma varios recursos en orden alfabético."""

        def __init__(self, scheduler: "IdleScheduler", resources: frozenset[str]) -> None:
            self._scheduler = scheduler
            self._resources = sorted(resources)
            self._held: list[threading.Lock] = []

        def __enter__(self) -> None:
            for res in self._resources:
                lock = self._scheduler._resource_lock(res)
                lock.acquire()
                self._held.append(lock)
            return None

        def __exit__(self, *exc: object) -> None:
            for lock in reversed(self._held):
                lock.release()
            self._held.clear()

    def _hold(self, resources: frozenset[str]) -> "IdleScheduler._Held":
        return IdleScheduler._Held(self, resources)

    # ── elegibilidad ──────────────────────────────────────────────────
    def _due(self, task: IdleTask, ctx: CycleContext, *, now: float | None = None) -> str | None:
        """None si la tarea puede correr; si no, el motivo del skip."""
        now = time.time() if now is None else now
        if ctx.idle_minutes < task.min_idle_minutes:
            return f"idle {ctx.idle_minutes:.1f} < {task.min_idle_minutes:.1f} min"
        if task.needs_llm and not ctx.provider_loaded():
            return "llm not loaded"
        with self._lock:
            last = self._last_run.get(task.name)
        if last is not None and (now - last) < task.cooldown_seconds:
            return f"cooldown ({now - last:.0f}s < {task.cooldown_seconds:.0f}s)"
        return None

    def _mark_run(self, name: str) -> None:
        with self._lock:
            self._last_run[name] = time.time()

    def _run_one(self, task: IdleTask, ctx: CycleContext) -> TaskOutcome:
        started = time.time()
        self._mark_run(task.name)
        try:
            with self._hold(task.resources):
                result = task.fn(ctx) or {}
            return TaskOutcome(name=task.name, ok=True,
                               duration_s=time.time() - started, result=result)
        except Exception as exc:
            return TaskOutcome(name=task.name, ok=False,
                               duration_s=time.time() - started, error=str(exc)[:300])

    # ── Tier 1 ────────────────────────────────────────────────────────
    def run_tier1(self, ctx: CycleContext) -> list[TaskOutcome]:
        """Corre las tareas de Tier 1 en paralelo, respetando recursos."""
        pending = [t for t in self.tasks if t.tier == 1]
        pending.sort(key=lambda t: (t.priority, t.name))
        ready: list[IdleTask] = []
        outcomes: list[TaskOutcome] = []
        for task in pending:
            reason = self._due(task, ctx)
            if reason is not None:
                outcomes.append(TaskOutcome(name=task.name, ok=True, duration_s=0.0, skipped=reason))
            else:
                ready.append(task)
        if not ready or ctx.should_abort():
            return outcomes
        workers = min(self.max_tier1_workers, len(ready))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="idle-t1") as pool:
            futures = {pool.submit(self._run_one, t, ctx): t for t in ready}
            for fut in as_completed(futures):
                outcomes.append(fut.result())
        return outcomes

    # ── Tier 2 ────────────────────────────────────────────────────────
    def run_tier2(self, ctx: CycleContext) -> list[TaskOutcome]:
        """Pase serial (el generator no es thread-safe). Preemptible entre items."""
        pending = [t for t in self.tasks if t.tier == 2]
        pending.sort(key=lambda t: (t.priority, t.name))
        outcomes: list[TaskOutcome] = []
        for task in pending:
            if ctx.should_abort():
                outcomes.append(TaskOutcome(name=task.name, ok=True, duration_s=0.0,
                                            skipped="aborted (usuario activo)"))
                continue
            reason = self._due(task, ctx)
            if reason is not None:
                outcomes.append(TaskOutcome(name=task.name, ok=True, duration_s=0.0, skipped=reason))
                continue
            outcomes.append(self._run_one(task, ctx))
        return outcomes


__all__ = [
    "CycleContext", "IdleScheduler", "IdleTask", "TaskOutcome",
    "RES_LLM", "RES_EMBEDDINGS", "RES_CLUSTER_STORE", "RES_AGENT_DB",
    "RES_TUTOR_DB", "RES_USER_MODEL", "RES_SKILLS", "RES_STRATEGIC",
    "RES_UNCERTAINTY", "RES_CONSOLIDATION", "RES_CORPUS_MAIN",
    "RES_CORPUS_REPORTER",
]
