"""Punto 2+3: Planner + TaskStore + TaskExecutor.

Planificación con el LLM actual (Qwen3.5-9B 3.0bpw). El bound de 3 tools/turno
es correcto para el chat interactivo, pero castra la autonomía de horizonte
largo. La solución no es subir el bound — es cambiar la unidad de bound: de
"por turno" a "por tarea con budget", donde la tarea es planificada y
persistida FUERA del contexto del LLM (RES-005).

Arquitectura:

    Usuario: "investigá fotónica y hacé un reporte"
      ↓
    Planner (1 generación LLM, sin tools):
      {goal, subtasks: [...], budget: {max_subtasks, max_tools_per_subtask}}
      ↓ (fallback a plantilla determinística si JSON inválido)
    TaskStore (SQLite, persistente):
      task_id, goal, subtasks_json, current_subtask, status, created_at
      ↓
    TaskExecutor (loop sobre subtasks):
      for subtask in subtasks:
        → ejecuta con bound de 3 tools (loop existente del dashboard)
        → guarda resultado resumido en task_store
        → si falla 2 veces, marca failed y continúa
      ↓
    Al terminar: notifica al usuario, resume resultados

El LLM nunca ve el plan completo + 20 tool results acumulados. Ve el
sub-task actual + un resumen de progreso. El 9B no puede sostener un plan
de 20 pasos en su context window — pero sí puede generar un plan corto
(una generación) y ejecutar cada sub-task por separado (contexto corto).

Resumibilidad: TaskStore persiste current_subtask y subtask_results. Si
cerrás el dashboard, "seguí investigando lo de ayer" resume desde el
último sub-task completado. El LLM no necesita "recordar" — el estado
está en SQLite.

Invariante: el planner nunca ejecuta tools. Solo produce el plan. La
ejecución es responsabilidad del TaskExecutor, que reutiliza el loop de
3 tools existente del dashboard (no lo duplica).
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ipa.agent.system_tools import SystemToolResult

DEFAULT_TASK_STORE = Path("outputs/agent/task_store.db")

# Budget por defecto: 6 sub-tasks, 3 tools por sub-task = 18 tools totales
# (vs 3 tools/turno del chat reactivo). Suficiente para investigación real.
DEFAULT_MAX_SUBTASKS = 6
DEFAULT_MAX_TOOLS_PER_SUBTASK = 3


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _compact_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f").lower()


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubTask:
    """Un paso del plan. Ejecutable por el TaskExecutor con el loop de tools."""
    id: int
    action: str  # nombre de la tool: search_corpus, research_topic, compile_report, ...
    args: dict[str, Any]
    why: str  # justificación para el LLM (contexto breve)
    status: str = "pending"  # pending | running | completed | failed | skipped
    result_summary: str | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Task:
    """Una tarea planificada con sub-tasks, persistida y resumible."""
    task_id: str
    goal: str
    subtasks: list[SubTask]
    budget: dict[str, int]
    status: str  # planned | running | completed | failed | paused
    current_subtask: int  # índice del próximo sub-task a ejecutar
    created_at: str
    updated_at: str
    session_id: str | None = None  # sesión de chat que originó la tarea
    resumed_count: int = 0
    final_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def progress(self) -> float:
        if not self.subtasks:
            return 0.0
        done = sum(1 for s in self.subtasks if s.status in ("completed", "skipped"))
        return done / len(self.subtasks)


# ---------------------------------------------------------------------------
# TaskStore — persistencia SQLite (append-only subtask_results, mutable status)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id         TEXT PRIMARY KEY,
    goal            TEXT NOT NULL,
    subtasks_json   TEXT NOT NULL,
    budget_json     TEXT NOT NULL,
    status          TEXT NOT NULL,
    current_subtask INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    session_id      TEXT,
    resumed_count   INTEGER NOT NULL DEFAULT 0,
    final_summary   TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at);
"""


class TaskStore:
    """SQLite-backed persistent task queue. Sobrevive sesiones."""

    def __init__(self, store_path: str | Path | None = None) -> None:
        if store_path is None:
            store_path = DEFAULT_TASK_STORE
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.store_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save_task(self, task: Task) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (task.task_id, task.goal, json.dumps([s.to_dict() for s in task.subtasks], ensure_ascii=False),
             json.dumps(task.budget, ensure_ascii=False), task.status, task.current_subtask,
             task.created_at, task.updated_at, task.session_id, task.resumed_count, task.final_summary),
        )
        self._conn.commit()

    def get_task(self, task_id: str) -> Task | None:
        row = self._conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return self._row_to_task(row) if row else None

    def list_tasks(self, *, status: str | None = None, limit: int = 20) -> list[Task]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def get_active_task(self) -> Task | None:
        """La tarea running o paused más reciente (para resumir)."""
        rows = self._conn.execute(
            "SELECT * FROM tasks WHERE status IN ('running', 'paused', 'planned') "
            "ORDER BY updated_at DESC LIMIT 1"
        ).fetchall()
        return self._row_to_task(rows[0]) if rows else None

    def update_subtask_status(
        self, task_id: str, subtask_index: int, status: str,
        *, result_summary: str | None = None, error: str | None = None,
    ) -> None:
        task = self.get_task(task_id)
        if task is None:
            raise ValueError(f"unknown task: {task_id}")
        subtasks = list(task.subtasks)
        if subtask_index < 0 or subtask_index >= len(subtasks):
            raise ValueError(f"subtask index out of range: {subtask_index}")
        old = subtasks[subtask_index]
        now = _now()
        subtasks[subtask_index] = SubTask(
            id=old.id, action=old.action, args=old.args, why=old.why,
            status=status,
            result_summary=result_summary if result_summary is not None else old.result_summary,
            error=error if error is not None else old.error,
            started_at=old.started_at or (now if status == "running" else None),
            finished_at=now if status in ("completed", "failed", "skipped") else old.finished_at,
        )
        new_current = subtask_index + 1 if status in ("completed", "skipped") else task.current_subtask
        new_status = task.status
        if status in ("completed", "skipped", "failed"):
            # Si era el último sub-task o todos los restantes están done/failed
            remaining = [s for s in subtasks[new_current:] if s.status == "pending"]
            if not remaining:
                any_failed = any(s.status == "failed" for s in subtasks)
                new_status = "failed" if any_failed and not any(s.status == "completed" for s in subtasks) else "completed"
        updated = Task(
            task_id=task.task_id, goal=task.goal, subtasks=subtasks,
            budget=task.budget, status=new_status, current_subtask=new_current,
            created_at=task.created_at, updated_at=now, session_id=task.session_id,
            resumed_count=task.resumed_count, final_summary=task.final_summary,
        )
        self.save_task(updated)

    def set_task_status(self, task_id: str, status: str, *, final_summary: str | None = None) -> None:
        task = self.get_task(task_id)
        if task is None:
            raise ValueError(f"unknown task: {task_id}")
        updated = Task(
            task_id=task.task_id, goal=task.goal, subtasks=task.subtasks,
            budget=task.budget, status=status, current_subtask=task.current_subtask,
            created_at=task.created_at, updated_at=_now(), session_id=task.session_id,
            resumed_count=task.resumed_count + (1 if status == "running" and task.status == "paused" else 0),
            final_summary=final_summary or task.final_summary,
        )
        self.save_task(updated)

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> Task:
        subtasks = [SubTask(**s) for s in json.loads(row["subtasks_json"])]
        return Task(
            task_id=row["task_id"], goal=row["goal"], subtasks=subtasks,
            budget=json.loads(row["budget_json"]), status=row["status"],
            current_subtask=row["current_subtask"], created_at=row["created_at"],
            updated_at=row["updated_at"], session_id=row["session_id"],
            resumed_count=row["resumed_count"], final_summary=row["final_summary"],
        )


# ---------------------------------------------------------------------------
# Planner — genera el plan (1 generación LLM o fallback determinístico)
# ---------------------------------------------------------------------------

_PLANNER_PROMPT = """Sos el módulo planificador de un agente personal. Analizá el pedido del usuario y respondé SOLO con JSON válido con esta forma:

{"goal": "objetivo en una frase", "subtasks": [{"action": "nombre_tool", "args": {"arg": "valor"}, "why": "por qué este paso"}], "budget": {"max_subtasks": 6, "max_tools_per_subtask": 3}}

Herramientas disponibles:
- search_corpus: busca en la base de conocimientos (args: query, limit)
- research_topic: investiga en la web e ingesta al corpus (args: query, max_urls, max_seconds)
- compile_report: compila un reporte fino desde query o document_ids (args: query o document_ids, topic)
- list_topics: lista tópicos del corpus (args: limit)
- list_promotions: cola de promoción (args: {})
- get_report: lee el último reporte (args: {})
- run_ingestion: lanza ingesta de fuentes configuradas (args: days_back)
- list_sources: lista fuentes configuradas (args: {})

Reglas:
- Máximo 6 sub-tasks. Cada sub-task ejecuta UNA tool.
- Si el pedido requiere investigación + reporte: search_corpus → research_topic → search_corpus → compile_report.
- Si el pedido es solo información: search_corpus.
- Si el pedido es reporte sin web: search_corpus → compile_report.
- "why" en una frase, explica por qué ese paso.
- Sin markdown, sin texto fuera del JSON.

Pedido del usuario:"""


# Plantillas determinísticas (fallback si el LLM produce JSON inválido)
_DETERMINISTIC_TEMPLATES = {
    "investig": [
        {"action": "search_corpus", "args": {"query": "{topic}", "limit": 10}, "why": "ver qué hay en el corpus"},
        {"action": "research_topic", "args": {"query": "{topic}", "max_urls": 5, "max_seconds": 120}, "why": "complementar corpus con web"},
        {"action": "search_corpus", "args": {"query": "{topic}", "limit": 10}, "why": "buscar sobre lo nuevo ingestado"},
        {"action": "compile_report", "args": {"query": "{topic}"}, "why": "sintetizar en reporte"},
    ],
    "report": [
        {"action": "search_corpus", "args": {"query": "{topic}", "limit": 15}, "why": "reunir documentos"},
        {"action": "compile_report", "args": {"query": "{topic}"}, "why": "compilar reporte"},
    ],
    "busc": [
        {"action": "search_corpus", "args": {"query": "{topic}", "limit": 10}, "why": "buscar en corpus"},
    ],
    "default": [
        {"action": "search_corpus", "args": {"query": "{topic}", "limit": 8}, "why": "buscar en corpus"},
    ],
}


class Planner:
    """Genera un plan de sub-tasks. LLM primero, fallback determinístico."""

    def __init__(self, store: TaskStore, provider: Any | None = None) -> None:
        self.store = store
        self.provider = provider

    def plan(
        self, goal: str, *, session_id: str | None = None,
        max_subtasks: int = DEFAULT_MAX_SUBTASKS,
        max_tools_per_subtask: int = DEFAULT_MAX_TOOLS_PER_SUBTASK,
    ) -> Task:
        """Produce un Task persistido. Intenta LLM, fallback a plantilla."""
        subtasks = self._plan_with_llm(goal, max_subtasks)
        if subtasks is None:
            subtasks = self._plan_deterministic(goal, max_subtasks)
        now = _now()
        task = Task(
            task_id=f"task:{_compact_stamp()}",
            goal=goal,
            subtasks=subtasks,
            budget={"max_subtasks": max_subtasks, "max_tools_per_subtask": max_tools_per_subtask},
            status="planned",
            current_subtask=0,
            created_at=now, updated_at=now,
            session_id=session_id,
        )
        self.store.save_task(task)
        return task

    def _plan_with_llm(self, goal: str, max_subtasks: int) -> list[SubTask] | None:
        if self.provider is None:
            return None
        try:
            result = self.provider.generate_chat(
                [{"role": "user", "content": f"{_PLANNER_PROMPT}\n{goal}"}],
                max_new_tokens=800,
            )
            text = getattr(result, "text", "") or ""
            if getattr(result, "error", None) or not text.strip():
                return None
            payload = self._parse_json(text)
            if payload is None:
                return None
            return self._validate_plan(payload, max_subtasks)
        except Exception:
            return None

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        cleaned = re.sub(r"<\|im_start\|>|<\|im_end\|>", "", text).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except ValueError:
            return None

    @staticmethod
    def _validate_plan(payload: dict[str, Any], max_subtasks: int) -> list[SubTask] | None:
        from ipa.agent.system_tools import SYSTEM_TOOL_NAMES
        raw_subtasks = payload.get("subtasks", [])
        if not isinstance(raw_subtasks, list) or not raw_subtasks:
            return None
        subtasks: list[SubTask] = []
        for i, st in enumerate(raw_subtasks[:max_subtasks]):
            if not isinstance(st, dict):
                continue
            action = str(st.get("action", "")).strip().lower()
            if action not in SYSTEM_TOOL_NAMES:
                continue  # descarta sub-tasks con tools inexistentes
            args = st.get("args", {})
            if not isinstance(args, dict):
                args = {}
            why = str(st.get("why", "")).strip()[:200]
            subtasks.append(SubTask(id=i, action=action, args=args, why=why))
        return subtasks if subtasks else None

    @staticmethod
    def _plan_deterministic(goal: str, max_subtasks: int) -> list[SubTask]:
        goal_lower = goal.lower()
        # Extraer "topic" del goal: quitar verbos comunes
        topic = goal.strip()
        for prefix in ("investigá ", "investiga ", "hacé un reporte sobre ", "hace un reporte sobre ",
                       "hacé un reporte de ", "hace un reporte de ", "buscá ", "busca ",
                       "investigá sobre ", "investiga sobre "):
            if topic.lower().startswith(prefix):
                topic = topic[len(prefix):]
                break
        template_key = "default"
        for key in _DETERMINISTIC_TEMPLATES:
            if key in goal_lower:
                template_key = key
                break
        # Fallback: matchear sin acentos (investigá → investig, reporté → report)
        if template_key == "default":
            import unicodedata
            normalized = unicodedata.normalize("NFD", goal_lower)
            ascii_only = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
            for key in _DETERMINISTIC_TEMPLATES:
                key_norm = unicodedata.normalize("NFD", key)
                key_ascii = "".join(c for c in key_norm if unicodedata.category(c) != "Mn")
                if key_ascii in ascii_only or key in ascii_only:
                    template_key = key
                    break
        template = _DETERMINISTIC_TEMPLATES[template_key]
        subtasks: list[SubTask] = []
        for i, step in enumerate(template[:max_subtasks]):
            args = {}
            for k, v in step["args"].items():
                if isinstance(v, str) and "{topic}" in v:
                    args[k] = v.replace("{topic}", topic)
                else:
                    args[k] = v
            subtasks.append(SubTask(id=i, action=step["action"], args=args, why=step["why"]))
        return subtasks


# ---------------------------------------------------------------------------
# Async tool support — el TaskExecutor bloquea hasta que async tools terminan
# ---------------------------------------------------------------------------

# Tools que lanzan threads y retornan inmediatamente. El executor debe esperar.
_ASYNC_TOOLS: frozenset[str] = frozenset({"research_topic", "run_ingestion"})

# Timeout por defecto para async tools (segundos). research_topic ya tiene su
# propio max_seconds (30-300); run_ingestion puede tardar minutos.
_DEFAULT_ASYNC_TIMEOUT = 600  # 10 minutos


def _async_timeout(action: str, args: dict[str, Any]) -> int:
    """Calcula el timeout para un async tool basado en sus args."""
    if action == "research_topic":
        try:
            max_seconds = int(args.get("max_seconds", 120))
        except (TypeError, ValueError):
            max_seconds = 120
        # Buffer de 60s sobre el max_seconds declarado (scrape + ingest overhead)
        return min(max_seconds + 60, 400)
    if action == "run_ingestion":
        return _DEFAULT_ASYNC_TIMEOUT
    return _DEFAULT_ASYNC_TIMEOUT


def _wait_for_async_completion(action: str, args: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    """Poll el progress file hasta que el async tool termine.

    Retorna {"ok": bool, "status": str, "summary": str, "data": dict, "error": str|None}.
    Bloquea el thread llamador (el thread de la tarea, no el chat).
    """
    import time as _time
    from ipa.agent.system_tools import _read_progress, _process_state

    poll_interval = 3  # segundos
    elapsed = 0
    progress_key = "research" if action == "research_topic" else "pipeline"

    while elapsed < timeout:
        _time.sleep(poll_interval)
        elapsed += poll_interval

        if action == "research_topic":
            state = _read_progress("research")
        else:  # run_ingestion
            # run_ingestion usa process_state + pipeline_progress
            ps = _process_state("pipeline")
            pp = _read_progress("pipeline")
            state = {**ps, **pp} if pp else ps

        status = str(state.get("status", "")).lower()
        if status in ("done", "completed", "success", "succeeded"):
            # Resumir el resultado final
            if action == "research_topic":
                r = state.get("result", {})
                summary = (
                    f"Research completada: {r.get('search_results', 0)} resultados, "
                    f"{r.get('scraped', 0)} scrapeados, {r.get('ingested', 0)} ingestado."
                )
            else:
                summary = f"Ingesta completada: {state.get('detail', '')}"
            return {"ok": True, "status": "done", "summary": summary, "data": state, "error": None}
        if status in ("failed", "error"):
            error = state.get("error", "async tool falló sin error específico")
            return {"ok": False, "status": "failed", "summary": "", "data": state, "error": str(error)}
        # still running — seguir esperando

    # Timeout
    return {
        "ok": False, "status": "timeout",
        "summary": f"Timeout tras {timeout}s esperando {action}",
        "data": {}, "error": f"{action} no terminó en {timeout}s",
    }


# ---------------------------------------------------------------------------
# TaskExecutor — loop sobre sub-tasks, cada uno con el loop de tools existente
# ---------------------------------------------------------------------------

class TaskExecutor:
    """Ejecuta los sub-tasks de un Task secuencialmente.

    Cada sub-task se ejecuta con el loop de tools del dashboard (bound de 3
    tools por sub-task). El executor NO reimplementa el loop — invoca
    execute_system_tool directamente y persiste el resultado resumido.

    Es async-safe: corre en un thread separado (como research_topic). El
    progreso se persiste en TaskStore después de cada sub-task.
    """

    def __init__(self, store: TaskStore) -> None:
        self.store = store

    def execute(self, task_id: str, *, max_retries: int = 2) -> Task:
        """Ejecuta todos los sub-tasks pendientes. Retorna el task final.

        Async tools (research_topic, run_ingestion) bloquean el thread de la
        tarea hasta terminar — no el chat. Esto asegura que el sub-task
        siguiente (ej: search_corpus sobre lo nuevo ingestado) vea los
        resultados. Ver RES-005, fix del bug de timing 2026-09-09.
        """
        from ipa.agent.system_tools import execute_system_tool
        task = self.store.get_task(task_id)
        if task is None:
            raise ValueError(f"unknown task: {task_id}")
        self.store.set_task_status(task_id, "running")
        task = self.store.get_task(task_id)  # refrescar

        while task.current_subtask < len(task.subtasks):
            subtask = task.subtasks[task.current_subtask]
            if subtask.status in ("completed", "skipped", "failed"):
                task = self.store.get_task(task_id)  # refrescar
                continue
            self.store.update_subtask_status(task_id, task.current_subtask, "running")

            retries = 0
            last_error: str | None = None
            result = None
            while retries <= max_retries:
                try:
                    result = execute_system_tool(subtask.action, dict(subtask.args))
                    if result.ok:
                        # Si es async, esperar a que termine antes de marcar completed
                        if subtask.action in _ASYNC_TOOLS:
                            wait_result = _wait_for_async_completion(
                                subtask.action, subtask.args,
                                timeout=_async_timeout(subtask.action, subtask.args),
                            )
                            if not wait_result["ok"]:
                                last_error = wait_result["error"]
                                retries += 1
                                continue
                            # Actualizar el summary con el resultado real
                            result = SystemToolResult(
                                tool_name=result.tool_name, ok=True,
                                summary=f"{result.summary} {wait_result['summary']}",
                                data={**result.data, "final_status": wait_result["status"], "final_data": wait_result["data"]},
                            )
                        self.store.update_subtask_status(
                            task_id, task.current_subtask, "completed",
                            result_summary=result.summary[:500],
                        )
                        last_error = None
                        break
                    else:
                        last_error = result.error or "tool falló sin error específico"
                        retries += 1
                except Exception as exc:
                    last_error = str(exc)
                    retries += 1

            if last_error is not None:
                self.store.update_subtask_status(
                    task_id, task.current_subtask, "failed",
                    error=last_error[:500],
                )

            task = self.store.get_task(task_id)  # refrescar

        # Resumen final
        task = self.store.get_task(task_id)
        if task is not None:
            completed = sum(1 for s in task.subtasks if s.status == "completed")
            failed = sum(1 for s in task.subtasks if s.status == "failed")
            summary = (
                f"Tarea '{task.goal}': {completed}/{len(task.subtasks)} sub-tasks completados, "
                f"{failed} fallidos."
            )
            self.store.set_task_status(task_id, task.status, final_summary=summary)
        return self.store.get_task(task_id)  # type: ignore[return-value]

    def resume(self, task_id: str) -> Task:
        """Resume una tarea paused/failed desde el último sub-task pendiente."""
        task = self.store.get_task(task_id)
        if task is None:
            raise ValueError(f"unknown task: {task_id}")
        if task.status not in ("paused", "failed", "planned"):
            raise ValueError(f"task {task_id} is {task.status}; only paused/failed/planned can resume")
        return self.execute(task_id)


def plan_and_execute(
    goal: str, *, store: TaskStore | None = None, provider: Any | None = None,
    session_id: str | None = None, async_run: bool = True,
) -> Task:
    """Convenience: planifica y (opcionalmente) ejecuta en background."""
    import threading
    s = store or TaskStore()
    planner = Planner(s, provider)
    task = planner.plan(goal, session_id=session_id)
    executor = TaskExecutor(s)
    if async_run:
        threading.Thread(
            target=executor.execute, args=(task.task_id,),
            daemon=True, name=f"task-{task.task_id}",
        ).start()
    else:
        task = executor.execute(task.task_id)
    return task


__all__ = [
    "Task", "SubTask", "TaskStore", "Planner", "TaskExecutor",
    "plan_and_execute",
    "DEFAULT_TASK_STORE", "DEFAULT_MAX_SUBTASKS", "DEFAULT_MAX_TOOLS_PER_SUBTASK",
]
