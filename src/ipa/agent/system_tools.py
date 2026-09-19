"""System control tools for the agent (bounded, deterministic).

These tools let the agent operate the IPA system itself: launch ingestion,
read process state, search the corpus and fetch report data. They are
deterministic — the LLM only SELECTS a tool; execution and arguments
validation happen here, never in the model.

Single registry: every chat-visible capability is a ``SystemToolSpec`` in
``_SYSTEM_TOOLS`` (bottom of this file). ``TOOL_CATALOG`` is generated from
the specs, so the prompt the LLM sees can never drift from the registry.
Corpus tools wrap the deterministic implementations in agent_tools.py —
same execution path as code-level callers, one source of truth.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VENV_PYTHON = ROOT_PY = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = sys.executable
REPORTER_ROOT = PROJECT_ROOT / "outputs" / "reporter"
PROGRESS_DIR = PROJECT_ROOT / "outputs" / "web_dashboard"
TOPIC_CLUSTER_DB = PROJECT_ROOT / "outputs" / "agent" / "topic_clusters.db"


@dataclass(frozen=True)
class SystemToolResult:
    """Deterministic result of a system tool execution."""
    tool_name: str
    ok: bool
    summary: str
    data: dict[str, Any]
    error: str | None = None

    def to_context_block(self) -> str:
        """Render as a context block for the LLM (compact, factual)."""
        if not self.ok:
            return f"[resultado de {self.tool_name}] ERROR: {self.error}"
        payload = json.dumps(self.data, ensure_ascii=False)[:4000]
        return f"[resultado de {self.tool_name}]\n{self.summary}\n{payload}"


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

def _read_progress(name: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "outputs" / "web_dashboard" / f"{name}_progress.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _process_state(kind: str) -> dict[str, Any]:
    """Read process state from the dashboard state DB (same source as UI)."""
    state_db = PROJECT_ROOT / "outputs" / "web_dashboard" / "dashboard.db"
    if not state_db.exists():
        return {"status": "unknown"}
    try:
        import sqlite3
        with sqlite3.connect(str(state_db)) as conn:
            row = conn.execute(
                "SELECT status, detail, percent, updated_at FROM process_state WHERE name=?",
                (kind,),
            ).fetchone()
        if row is None:
            return {"status": "not_started"}
        return {"status": row[0], "detail": row[1], "percent": row[2], "updated_at": row[3]}
    except Exception:
        return {"status": "unknown"}


def _main_corpus_dir() -> Path | None:
    """Locate the main corpus directory (document_store.db present)."""
    corpus_dir = PROJECT_ROOT / "outputs" / "experiments" / "E12-corpus"
    if (corpus_dir / "document_store.db").exists():
        return corpus_dir
    candidates = sorted(
        (PROJECT_ROOT / "outputs" / "experiments").glob("*-corpus"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "document_store.db").exists():
            return candidate
    return None


def _research_progress() -> dict[str, Any]:
    return _read_progress("research")


def _write_research_progress(payload: dict[str, Any]) -> None:
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    path = PROGRESS_DIR / "research_progress.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def tool_get_system_status(args: dict[str, Any]) -> SystemToolResult:
    """Snapshot of pipeline processes, research jobs and corpus state."""
    scraper = _process_state("scraper")
    pipeline = _process_state("pipeline")
    progress = _read_progress("pipeline")
    research = _research_progress()
    landing = PROJECT_ROOT / "Landing" / "web"
    landing_files = sum(1 for p in landing.rglob("*") if p.is_file() and p.name != "scrape_history.db") if landing.exists() else 0
    data = {
        "scraper": scraper,
        "fastpath": pipeline,
        "research": research or {"status": "idle"},
        "pipeline_progress": progress,
        "landing_files": landing_files,
    }
    running = [k for k, v in [("scraper", scraper), ("fastpath", pipeline)] if v.get("status") == "running"]
    if research.get("status") == "running":
        running.append("research")
    summary = f"Procesos corriendo: {', '.join(running) if running else 'ninguno'}. Archivos en Landing/web: {landing_files}."
    return SystemToolResult(tool_name="get_system_status", ok=True, summary=summary, data=data)


def _latest_report_path() -> Path | None:
    if not REPORTER_ROOT.exists():
        return None
    reports = sorted(REPORTER_ROOT.glob("**/report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0] if reports else None


def tool_get_report(args: dict[str, Any]) -> SystemToolResult:
    """Read the latest compiled report (agent-generated or legacy batch)."""
    path = _latest_report_path()
    if path is None:
        return SystemToolResult(tool_name="get_report", ok=False, summary="", data={}, error="no hay reportes generados todavía")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        return SystemToolResult(tool_name="get_report", ok=False, summary="", data={}, error=f"reporte ilegible: {exc}")
    cats = report.get("categories", [])
    compact = []
    for c in cats[:12]:
        entry = {
            "label": c.get("label"),
            "evolution": c.get("evolution"),
            "document_count": c.get("document_count"),
            "importance": c.get("importance"),
            "description": (c.get("description") or "")[:300],
        }
        subs = c.get("subtopics") or []
        if subs:
            entry["subtopics"] = [
                {"label": s.get("label") if isinstance(s, dict) else str(s), "document_count": s.get("document_count") if isinstance(s, dict) else None}
                for s in subs[:6]
            ]
        compact.append(entry)
    period = report.get("period", {})
    data = {
        "report_id": report.get("report_id"),
        "period": period,
        "status": report.get("status"),
        "generated_at": report.get("generation", {}).get("generated_at"),
        "total_documents": report.get("statistics", {}).get("total_documents"),
        "categories": compact,
        "path": str(path),
    }
    summary = f"Reporte {period.get('label', '?')}: {len(cats)} categorías, estado {report.get('status')}."
    return SystemToolResult(tool_name="get_report", ok=True, summary=summary, data=data)


def tool_run_ingestion(args: dict[str, Any]) -> SystemToolResult:
    """Launch ingestion via the dashboard's own /api/pipeline/run.

    Scraper → FastPath → BM25 + LanceDB indexing. No report — reports are
    compiled on demand with compile_report. The dashboard owns execution
    (locks, progress files, stage tracking): the agent uses the same API
    surface as the UI buttons — one execution path, no duplication.

    The AGENT determines the scrape window per launch — it is not hardcoded:
    - days_back omitted and no date range → no global override; each site
      uses its configured baseline (sources.json / scrape_sites.yaml).
    - days_back N → global window override; N > 30 also clears the scrape
      history to re-discover already-seen articles in the wider window
      (content-hash dedup prevents corpus duplicates).
    - date_from/date_to (YYYY-MM-DD) → explicit range, alternative to
      days_back.

    ``run_pipeline`` resolves to this same implementation (hidden alias):
    there is only one ingestion job.
    """
    has_days = "days_back" in args and args.get("days_back") not in (None, "")
    date_from = str(args.get("date_from", "") or "").strip()
    date_to = str(args.get("date_to", "") or "").strip()
    days_back: int | None = None
    if has_days:
        try:
            days_back = max(0, min(365, int(args["days_back"])))
        except (TypeError, ValueError):
            return SystemToolResult(tool_name="run_ingestion", ok=False, summary="", data={}, error=f"days_back inválido: {args.get('days_back')!r}")
    if date_from or date_to:
        import re as _re
        if not (_re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_from)
                and _re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_to)):
            return SystemToolResult(
                tool_name="run_ingestion", ok=False, summary="", data={},
                error="date_from/date_to deben ser fechas YYYY-MM-DD (ambas requeridas)",
            )
        payload_dict: dict[str, Any] = {
            "period_mode": "range",
            "period_start": f"{date_from}T00:00:00Z",
            "period_end": f"{date_to}T23:59:59Z",
        }
    elif days_back is not None:
        payload_dict = {"period_mode": "days", "days_back": days_back}
    else:
        # Sin ventana explícita: baselines por sitio (sin override global).
        payload_dict = {"period_mode": "days", "days_back": 0}

    # Live status: same progress file the dashboard serves via /api/state.
    # Stale "running" (>10 min sin update) = proceso muerto, no bloquea.
    progress = _read_progress("pipeline")
    if progress.get("status") == "running":
        updated = str(progress.get("updated_at", ""))
        fresh = False
        try:
            from datetime import datetime, timezone
            fresh = (datetime.now(timezone.utc)
                     - datetime.fromisoformat(updated.replace("Z", "+00:00"))
                     ).total_seconds() < 600
        except ValueError:
            pass
        if fresh:
            return SystemToolResult(
                tool_name="run_ingestion", ok=True,
                summary=(
                    f"Ya hay una ingesta corriendo: etapa '{progress.get('stage', '?')}', "
                    f"{progress.get('percent', '?')}% — {progress.get('detail', '')}. "
                    "Pidime el estado en unos minutos."
                ),
                data={"already_running": True, "state": progress},
            )
    import urllib.request
    import urllib.error
    dashboard_url = os.environ.get("IPA_DASHBOARD_URL", "http://127.0.0.1:8765")
    payload = json.dumps(payload_dict).encode("utf-8")
    req = urllib.request.Request(
        f"{dashboard_url}/api/pipeline/run",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        window = (
            f"de {date_from} a {date_to}" if date_from
            else f"de {days_back} días" if days_back is not None
            else "baselines por sitio (sin override global)"
        )
        return SystemToolResult(
            tool_name="run_ingestion", ok=True,
            summary=f"Ingesta iniciada (scraper → FastPath → BM25 + LanceDB) con ventana {window}. Sin reporte — usar compile_report cuando quieras análisis. Te aviso cuando termine.",
            data={**payload_dict, "dashboard_response": result},
        )
    except urllib.error.HTTPError as exc:
        # El dashboard rechaza con 400 {"error": "..."} — leer el motivo real.
        try:
            err_body = json.loads(exc.read().decode("utf-8"))
            detail = err_body.get("error", str(exc))
        except Exception:
            detail = str(exc)
        already = "ejecut" in detail.lower() or "running" in detail.lower()
        return SystemToolResult(
            tool_name="run_ingestion", ok=already, summary=(
                f"Ya hay una ingesta corriendo ({detail}). Pidime el estado en unos minutos."
                if already else ""
            ),
            data={"already_running": already, "detail": detail},
            error=None if already else f"no se pudo iniciar la ingesta: {detail}",
        )
    except Exception as exc:
        return SystemToolResult(tool_name="run_ingestion", ok=False, summary="", data={}, error=f"no se pudo iniciar la ingesta: {exc}")


def _run_pipeline_alias(args: dict[str, Any]) -> SystemToolResult:
    """Hidden alias: run_pipeline → run_ingestion (same ingestion job)."""
    result = tool_run_ingestion(args)
    return SystemToolResult(
        tool_name="run_pipeline", ok=result.ok, summary=result.summary,
        data=result.data, error=result.error,
    )


def tool_compile_report(args: dict[str, Any]) -> SystemToolResult:
    """Compila un reporte fino a partir de documentos del corpus.

    Acepta:
      - query: busca documentos en el corpus via search_corpus y compila el reporte
      - document_ids: lista explícita de IDs de documentos del corpus
      - topic: etiqueta/descripción del reporte (opcional)
      - interests: lista de términos de interés para scoring (opcional)

    El reporte se escribe en outputs/reporter/agent/. No muta el corpus.
    """
    from ipa.agent.agent_tools import ToolContext
    from ipa.agent.agent_memory import AgentMemory
    from ipa.agent.compile_report_executor import execute_compile_report

    query = str(args.get("query", "") or "").strip()
    document_ids = args.get("document_ids")
    topic = str(args.get("topic", "") or "").strip()

    corpus_dir = _main_corpus_dir()
    if corpus_dir is None:
        return SystemToolResult(
            tool_name="compile_report", ok=False, summary="", data={},
            error="no se encontró un corpus con document_store.db en outputs/experiments/",
        )

    # If query provided but no document_ids, search the corpus first
    if query and not document_ids:
        try:
            from ipa.agent.agent_tools import _search_corpus
            memory = AgentMemory()
            ctx = ToolContext(memory=memory, corpus_dir=str(corpus_dir))
            result_dict, _ = _search_corpus({"query": query, "limit": 20}, ctx)
            hits = result_dict.get("hits", [])
            document_ids = list(dict.fromkeys(h["document_id"] for h in hits if h.get("document_id") and h["document_id"] != "unknown"))
            ctx.close()
            memory.close()
            if not document_ids:
                return SystemToolResult(
                    tool_name="compile_report", ok=False, summary="", data={},
                    error=f"la búsqueda '{query}' no encontró documentos en el corpus",
                )
        except Exception as exc:
            return SystemToolResult(
                tool_name="compile_report", ok=False, summary="", data={},
                error=f"error al buscar documentos: {exc}",
            )

    if not document_ids:
        return SystemToolResult(
            tool_name="compile_report", ok=False, summary="", data={},
            error="se requiere 'query' (para buscar) o 'document_ids' (lista explícita)",
        )

    # Execute compile_report
    try:
        memory = AgentMemory()
        ctx = ToolContext(memory=memory, corpus_dir=str(corpus_dir))
        sid = memory.open_session(
            interface="system_tool", role="general",
            identity_hash="system", title=f"compile_report: {topic or query or 'explicit'}",
        )
        ep = memory.record_episode(
            sid, turn_role="user",
            content=f"compile_report({len(document_ids)} docs){': ' + topic if topic else ''}",
            identity_hash="system",
        )
        try:
            call, result, compile_result = execute_compile_report(
                {
                    "document_ids": document_ids,
                    "topic": topic or query or None,
                    "interests": args.get("interests"),
                },
                ctx,
                session_id=sid,
                episode_id=ep.episode_id,
            )
        finally:
            memory.close_session(sid)
            ctx.close()
            memory.close()

        if result.status == "completed" and compile_result.success:
            categories_summary = ", ".join(
                f"{c['label']} ({c['document_count']} docs)"
                for c in compile_result.categories[:5]
            )
            return SystemToolResult(
                tool_name="compile_report", ok=True,
                summary=(
                    f"Reporte compilado: {compile_result.document_count} documentos, "
                    f"{compile_result.category_count} tópicos. "
                    f"Tópicos: {categories_summary or 'ninguno'}. "
                    f"Reporte: {compile_result.report_path}"
                ),
                data={
                    "report_id": compile_result.report_id,
                    "report_path": compile_result.report_path,
                    "output_dir": compile_result.output_dir,
                    "document_count": compile_result.document_count,
                    "category_count": compile_result.category_count,
                    "curation_summary": compile_result.curation_summary,
                    "categories": [
                        {"label": c["label"], "document_count": c["document_count"],
                         "document_ids": c["document_ids"][:10]}
                        for c in compile_result.categories
                    ],
                },
            )
        else:
            return SystemToolResult(
                tool_name="compile_report", ok=False, summary="", data={},
                error=result.error or "compile_report falló sin error específico",
            )
    except Exception as exc:
        return SystemToolResult(
            tool_name="compile_report", ok=False, summary="", data={},
            error=f"error al compilar reporte: {exc}",
        )


def tool_list_sources(args: dict[str, Any]) -> SystemToolResult:
    """List configured scrape sources (effective config)."""
    sources_file = PROJECT_ROOT / "outputs" / "web_dashboard" / "sources.json"
    base_yaml = PROJECT_ROOT / "configs" / "scrape_sites.yaml"
    entries: list[dict[str, Any]] = []
    try:
        import yaml
        if base_yaml.exists():
            cfg = yaml.safe_load(base_yaml.read_text(encoding="utf-8")) or {}
            for s in cfg.get("sites", []):
                entries.append({"url": s.get("url"), "active": True, "days_back": s.get("days_back", 7), "origin": "base"})
    except Exception:
        pass
    if sources_file.exists():
        try:
            overrides = json.loads(sources_file.read_text(encoding="utf-8"))
            entries.extend(overrides.get("added", []))
        except (ValueError, OSError):
            pass
    data = {"total": len(entries), "active": sum(1 for e in entries if e.get("active", True)), "sources": entries[:20]}
    summary = f"{len(entries)} fuentes configuradas ({data['active']} activas)."
    return SystemToolResult(tool_name="list_sources", ok=True, summary=summary, data=data)


def tool_search_corpus(args: dict[str, Any]) -> SystemToolResult:
    """Hybrid search over the main corpus (LanceDB dense + BM25 fallback).

    Returns numbered hits so the model can cite them as [n] in its reply —
    the identity prompt's citation contract. Read-only: never mutates corpus.
    Same execution path as the agent-level search_corpus tool.
    """
    query = str(args.get("query", "") or "").strip()
    if not query:
        return SystemToolResult(
            tool_name="search_corpus", ok=False, summary="", data={},
            error="se requiere 'query' (texto a buscar)",
        )
    try:
        limit = max(1, min(15, int(args.get("limit", 8))))
    except (TypeError, ValueError):
        limit = 8
    corpus_dir = _main_corpus_dir()
    if corpus_dir is None:
        return SystemToolResult(
            tool_name="search_corpus", ok=False, summary="", data={},
            error="no se encontró un corpus con document_store.db en outputs/experiments/",
        )
    from ipa.agent.agent_memory import AgentMemory
    from ipa.agent.agent_tools import ToolContext, _search_corpus

    memory = AgentMemory()
    ctx = ToolContext(memory=memory, corpus_dir=str(corpus_dir))
    try:
        result_dict, _refs = _search_corpus(
            {
                "query": query,
                "limit": limit,
                "date_from": str(args.get("date_from", "") or ""),
                "date_to": str(args.get("date_to", "") or ""),
            },
            ctx,
        )
    except Exception as exc:
        return SystemToolResult(
            tool_name="search_corpus", ok=False, summary="", data={},
            error=f"error al buscar en el corpus: {exc}",
        )
    finally:
        ctx.close()
        memory.close()

    hits = result_dict.get("hits", [])
    evidence = [
        f"[{i}] {hit['document_id']} (score {hit['score']}) — {hit['text_preview']}"
        for i, hit in enumerate(hits, 1)
    ]
    backend = hits[0]["retrieval_backend"] if hits else "sin resultados"
    summary = (
        f"{len(hits)} resultados para '{query}' (backend: {backend}). "
        "Citá los hits como [n] en tu respuesta."
    )
    data = {"query": query, "total": len(hits), "hits": hits, "evidence": evidence}
    return SystemToolResult(tool_name="search_corpus", ok=True, summary=summary, data=data)


def tool_list_topics(args: dict[str, Any]) -> SystemToolResult:
    """List emergent topic clusters (derived index over corpus embeddings)."""
    try:
        limit = max(1, min(50, int(args.get("limit", 20))))
    except (TypeError, ValueError):
        limit = 20
    if not TOPIC_CLUSTER_DB.exists():
        return SystemToolResult(
            tool_name="list_topics", ok=True,
            summary="No hay tópicos todavía — el índice de clusters no existe "
                    "(se construye durante el idle enrichment).",
            data={"topics": [], "total": 0},
        )
    from ipa.agentic.topic_clusters import TopicClusterStore

    store = TopicClusterStore(TOPIC_CLUSTER_DB)
    try:
        clusters = store.list_clusters()[:limit]
        topics = [
            {
                "cluster_id": c.cluster_id,
                "label": c.label,
                "documents": len(c.member_document_ids),
                "coherence": round(c.coherence_score, 3),
                "parent_cluster_id": c.parent_cluster_id,
            }
            for c in clusters
        ]
    finally:
        store.close()
    summary = f"{len(topics)} tópicos (índice derivado sobre embeddings del corpus)."
    return SystemToolResult(
        tool_name="list_topics", ok=True, summary=summary,
        data={"topics": topics, "total": len(topics)},
    )


def tool_list_promotions(args: dict[str, Any]) -> SystemToolResult:
    """Read the promotion queue: pending docs and recently promoted ones.

    Promotion is policy-driven and continuous (idle enrichment evaluates
    provenance + score and process_promotion_queue copies eligible docs).
    This tool only READS the queue — the agent never triggers copies.
    """
    if not TOPIC_CLUSTER_DB.exists():
        return SystemToolResult(
            tool_name="list_promotions", ok=True,
            summary="No hay cola de promoción todavía.",
            data={"pending": [], "recent": []},
        )
    from ipa.agentic.topic_clusters import TopicClusterStore

    store = TopicClusterStore(TOPIC_CLUSTER_DB)
    try:
        pending = store.pending_promotions()
        recent = store.recent_promotions(limit=10)
    finally:
        store.close()
    summary = (
        f"{len(pending)} documentos pendientes de promoción, "
        f"{len(recent)} promovidos recientemente."
    )
    data = {
        "pending": pending[:30],
        "recent": recent,
        "pending_total": len(pending),
    }
    return SystemToolResult(tool_name="list_promotions", ok=True, summary=summary, data=data)


def tool_research_topic(args: dict[str, Any]) -> SystemToolResult:
    """Launch bounded web research in the background (async).

    Searches the web, judges snippets/content with the deterministic
    HeuristicJudge (no VRAM — safe alongside chat), ingests what passes
    into the main corpus with provenance=agent_research. Progress is
    written to outputs/web_dashboard/research_progress.json so the
    dashboard watcher can notify the session when it finishes.
    """
    session_id = args.pop("_session_id", None)
    auto = bool(args.pop("_auto", False))
    user_message = str(args.pop("_user_message", "") or "").strip()
    query = str(args.get("query", "") or "").strip()
    if not query:
        return SystemToolResult(
            tool_name="research_topic", ok=False, summary="", data={},
            error="se requiere 'query' (tema a investigar)",
        )
    # URLs que el usuario pegó en su mensaje: el modelo suele parafrasear la
    # query y descartarlas. Se reinyectan acá — el executor las detecta y las
    # scrapea directo como seeds (fuentes explícitas), además de buscar el
    # remanente textual.
    if user_message:
        from ipa.agent.web_search import extract_urls
        extra = [u for u in extract_urls(user_message) if u not in query]
        if extra:
            query = (query + " " + " ".join(extra)).strip()
    try:
        max_urls = max(1, min(20, int(args.get("max_urls", 5))))
    except (TypeError, ValueError):
        max_urls = 5
    try:
        max_seconds = max(30, min(600, int(args.get("max_seconds", 120))))
    except (TypeError, ValueError):
        max_seconds = 120
    _force_raw = args.get("force")
    force = _force_raw if isinstance(_force_raw, bool) else str(_force_raw or "").strip().lower() in ("1", "true", "yes", "si", "sí")

    state = _research_progress()
    if state.get("status") == "running":
        # Stale check: a "running" state older than 3x max_seconds means the
        # research thread died (e.g. dashboard restart) without writing a
        # final status. Treat it as dead so a new query can proceed.
        try:
            started = state.get("started_at", "")
            stale_limit = max_seconds * 3
            if started:
                from datetime import datetime, timezone
                t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
                elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
                if elapsed > stale_limit:
                    _write_research_progress({
                        "status": "failed", "query": state.get("query", ""),
                        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "error": f"stale: running for {int(elapsed)}s > {stale_limit}s limit",
                    })
                    state = {}
        except Exception:
            pass
    if state.get("status") == "running":
        return SystemToolResult(
            tool_name="research_topic", ok=True,
            summary=f"Ya hay una investigación corriendo: '{state.get('query', '?')}'. Pidime el estado en unos minutos.",
            data={"already_running": True, "state": state},
        )
    # Dedup: aplica a TODOS los llamados (auto y explícitos). Si una query
    # igual o muy parecida ya se investigó dentro de la ventana, no relanzar —
    # el material está ingerido en el corpus. El modelo reformula la query
    # entre turnos ("IA big techs" → "IA tres grandes tecnológicas"), así que
    # el match exacto no alcanza. `force: true` fuerza una corrida nueva.
    if not force:
        from ipa.agent.research_review import find_recent_research
        recent = find_recent_research(query)
        if recent:
            return SystemToolResult(
                tool_name="research_topic", ok=True,
                summary=(
                    f"Ya investigué '{recent['query']}' hace {recent['age_minutes']} min — "
                    "ese material ya está en el corpus: usá search_corpus o compile_report "
                    "sobre lo ingerido en vez de relanzar la búsqueda. Si necesitás una "
                    "corrida nueva igual, volvé a llamarme con force=true."
                ),
                data={"dedup": True, "query": query, "matched_query": recent["query"],
                      "age_minutes": recent["age_minutes"]},
            )
    corpus_dir = _main_corpus_dir()
    if corpus_dir is None:
        return SystemToolResult(
            tool_name="research_topic", ok=False, summary="", data={},
            error="no se encontró un corpus con document_store.db en outputs/experiments/",
        )

    _write_research_progress({
        "status": "running", "query": query,
        "max_urls": max_urls, "max_seconds": max_seconds,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id": session_id,
        "notified": False,
    })
    from ipa.agent.research_review import mark_researched
    mark_researched(query)

    # Launch as a subprocess so the research survives a dashboard restart.
    # The subprocess writes progress to research_progress.json; the dashboard
    # watcher picks it up and notifies the session.
    import subprocess as _sp
    research_script = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "operations" / "run_research.py"
    _sp.Popen(
        [sys.executable, str(research_script), query, str(max_urls), str(max_seconds)],
        stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
        creationflags=_sp.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    return SystemToolResult(
        tool_name="research_topic", ok=True,
        summary=(
            f"Investigación iniciada: '{query}' — busco en la web, filtro con "
            f"juez heurístico e injesto lo relevante al corpus (máx {max_urls} "
            f"fuentes, {max_seconds}s). Te aviso cuando termine."
        ),
        data={"query": query, "max_urls": max_urls, "max_seconds": max_seconds},
    )


def tool_plan_task(args: dict[str, Any]) -> SystemToolResult:
    """Planifica y lanza una tarea multi-paso en background.

    El planner descompone el goal en sub-tasks (cada uno ejecuta una tool
    con bound de 3). El plan se persiste en TaskStore — sobrevive sesiones.
    Lanza la ejecución en un thread separado (async). Ver RES-005.
    """
    from ipa.agent.task_planner import TaskStore, Planner, TaskExecutor, plan_and_execute

    goal = str(args.get("goal", "") or "").strip()
    if not goal:
        return SystemToolResult(
            tool_name="plan_task", ok=False, summary="", data={},
            error="se requiere 'goal' (objetivo de la tarea)",
        )
    try:
        max_subtasks = max(1, min(10, int(args.get("max_subtasks", 6))))
    except (TypeError, ValueError):
        max_subtasks = 6
    try:
        max_tools = max(1, min(5, int(args.get("max_tools_per_subtask", 3))))
    except (TypeError, ValueError):
        max_tools = 3

    store = TaskStore()
    try:
        # Provider opcional: si no hay LLM cargado, fallback a plantilla determinística.
        provider = None
        try:
            from ipa.agent.provider_wiring import get_active_provider
            provider = get_active_provider()
        except Exception:
            pass
        task = plan_and_execute(
            goal, store=store, provider=provider,
            session_id=str(args.get("session_id", "") or None),
            async_run=True,
        )
        subtasks_preview = [
            {"id": s.id, "action": s.action, "why": s.why}
            for s in task.subtasks[:6]
        ]
        return SystemToolResult(
            tool_name="plan_task", ok=True,
            summary=(
                f"Tarea planificada: '{goal}' — {len(task.subtasks)} sub-tasks. "
                f"Ejecutando en background. Pedime el estado con list_tasks."
            ),
            data={
                "task_id": task.task_id, "goal": task.goal,
                "subtasks": subtasks_preview,
                "budget": task.budget, "status": task.status,
            },
        )
    except Exception as exc:
        return SystemToolResult(
            tool_name="plan_task", ok=False, summary="", data={},
            error=f"error al planificar tarea: {exc}",
        )
    finally:
        store.close()


def tool_list_tasks(args: dict[str, Any]) -> SystemToolResult:
    """Lista tareas planificadas/ejecutándose (TaskStore persistente)."""
    from ipa.agent.task_planner import TaskStore

    try:
        limit = max(1, min(50, int(args.get("limit", 10))))
    except (TypeError, ValueError):
        limit = 10
    status_filter = str(args.get("status", "") or "").strip() or None
    store = TaskStore()
    try:
        tasks = store.list_tasks(status=status_filter, limit=limit)
    finally:
        store.close()
    compact = [
        {
            "task_id": t.task_id, "goal": t.goal, "status": t.status,
            "progress": round(t.progress, 2), "current_subtask": t.current_subtask,
            "total_subtasks": len(t.subtasks), "created_at": t.created_at,
            "final_summary": t.final_summary,
        }
        for t in tasks
    ]
    active = sum(1 for t in tasks if t.status in ("running", "planned", "paused"))
    summary = f"{len(tasks)} tareas ({active} activas)."
    return SystemToolResult(
        tool_name="list_tasks", ok=True, summary=summary,
        data={"tasks": compact, "total": len(tasks), "active": active},
    )


def tool_get_task(args: dict[str, Any]) -> SystemToolResult:
    """Detalle de una tarea: sub-tasks, progreso, resultados."""
    from ipa.agent.task_planner import TaskStore

    task_id = str(args.get("task_id", "") or "").strip()
    if not task_id:
        return SystemToolResult(
            tool_name="get_task", ok=False, summary="", data={},
            error="se requiere 'task_id'",
        )
    store = TaskStore()
    try:
        task = store.get_task(task_id)
    finally:
        store.close()
    if task is None:
        return SystemToolResult(
            tool_name="get_task", ok=False, summary="", data={},
            error=f"tarea no encontrada: {task_id}",
        )
    subtasks = [
        {
            "id": s.id, "action": s.action, "args": s.args, "why": s.why,
            "status": s.status, "result_summary": s.result_summary, "error": s.error,
        }
        for s in task.subtasks
    ]
    summary = (
        f"Tarea '{task.goal}': {task.status}, {round(task.progress * 100)}% completado "
        f"({task.current_subtask}/{len(task.subtasks)} sub-tasks)."
    )
    return SystemToolResult(
        tool_name="get_task", ok=True, summary=summary,
        data={
            "task_id": task.task_id, "goal": task.goal, "status": task.status,
            "progress": round(task.progress, 3), "subtasks": subtasks,
            "budget": task.budget, "final_summary": task.final_summary,
            "resumed_count": task.resumed_count,
        },
    )


def tool_resume_task(args: dict[str, Any]) -> SystemToolResult:
    """Resume una tarea paused/failed desde el último sub-task pendiente."""
    from ipa.agent.task_planner import TaskStore, TaskExecutor
    import threading

    task_id = str(args.get("task_id", "") or "").strip()
    if not task_id:
        return SystemToolResult(
            tool_name="resume_task", ok=False, summary="", data={},
            error="se requiere 'task_id'",
        )
    store = TaskStore()
    try:
        task = store.get_task(task_id)
        if task is None:
            return SystemToolResult(
                tool_name="resume_task", ok=False, summary="", data={},
                error=f"tarea no encontrada: {task_id}",
            )
        if task.status not in ("paused", "failed", "planned"):
            return SystemToolResult(
                tool_name="resume_task", ok=True,
                summary=f"La tarea {task_id} ya está {task.status}.",
                data={"task_id": task_id, "status": task.status},
            )
        executor = TaskExecutor(store)
        threading.Thread(
            target=executor.resume, args=(task_id,),
            daemon=True, name=f"task-resume-{task_id}",
        ).start()
        return SystemToolResult(
            tool_name="resume_task", ok=True,
            summary=f"Resumiendo tarea '{task.goal}' desde sub-task {task.current_subtask}.",
            data={"task_id": task_id, "resume_from": task.current_subtask},
        )
    finally:
        store.close()


def tool_get_user_profile(args: dict[str, Any]) -> SystemToolResult:
    """Lee el user model transversal: goals, intereses, preferencias, facts."""
    from ipa.agent.user_model import UserModelStore

    store = UserModelStore()
    try:
        goals = store.active_goals()
        interests = store.top_interests(limit=10)
        prefs = store.list_preferences()
        facts = store.active_facts()
    finally:
        store.close()
    data = {
        "goals": [{"description": g.description, "status": g.status, "source": g.source} for g in goals],
        "interests": [{"topic": i.topic, "score": round(i.score, 2), "count": i.occurrence_count}
                      for i in interests if i.score > 0.05],
        "preferences": [{"key": p.key, "value": p.value, "source": p.source} for p in prefs],
        "facts": facts[:10],
    }
    summary_parts = []
    if goals:
        summary_parts.append(f"{len(goals)} goals activos")
    if interests:
        top = interests[0]
        summary_parts.append(f"interés principal: '{top.topic}' ({top.score:.2f})")
    if prefs:
        summary_parts.append(f"{len(prefs)} preferencias")
    if facts:
        summary_parts.append(f"{len(facts)} facts activos")
    summary = "Perfil del usuario: " + ", ".join(summary_parts) + "." if summary_parts else "Perfil vacío (sin datos todavía)."
    return SystemToolResult(
        tool_name="get_user_profile", ok=True, summary=summary, data=data,
    )


def _warm_embedding_adapter() -> Any | None:
    """Reuse the dashboard's warm BGE-M3 adapter when running in-process.

    Returns None outside the dashboard (tests, CLI) → recall stays FTS-only
    instead of loading a duplicate embedding model.
    """
    try:
        from ipa.dashboard import server as _srv
        adapter = _srv.get_embedding_adapter()
        return adapter if adapter is not None else None
    except Exception:
        return None


def _recall_hybrid(store: Any, query: str, *, scopes: list[str] | None,
                   limit: int) -> list[Any]:
    """FTS5 recall fused with sqlite-vec semantic recall (RRF).

    Embeds items missing from the vector index (bounded per call) when a
    warm embedding adapter exists; otherwise plain FTS recall.
    """
    fts_items = store.recall(query, scopes=scopes, limit=limit)
    adapter = _warm_embedding_adapter()
    if adapter is None:
        return fts_items
    try:
        from ipa.agent.memory_store import MemoryVectorIndex, _rrf_merge
        vec_index = MemoryVectorIndex()
        try:
            # Embed items not yet in the vector index (bounded per call —
            # the rest are picked up by subsequent recalls / idle sync).
            pending = [
                it for it in store.recall("", scopes=scopes, limit=200)
                if it.memory_id not in vec_index.embedded_ids()
            ][:32]
            if pending:
                vectors = adapter.embed_texts([it.text for it in pending])
                for it, vec in zip(pending, vectors):
                    vec_index.upsert(it.memory_id, vec, scope=it.scope)
            query_vec = adapter.embed_query(query)
            vec_hits = vec_index.search(query_vec, limit=max(limit * 2, 10))
            fused_ids = _rrf_merge(
                [it.memory_id for it in fts_items],
                [mid for mid, _ in vec_hits],
            )[:limit]
            fused = store.get_by_ids(fused_ids)
            if scopes:
                allowed = set(scopes)
                fused = [it for it in fused if it.scope in allowed]
            return fused if fused else fts_items
        finally:
            vec_index.close()
    except Exception:
        return fts_items


def tool_recall_memory(args: dict[str, Any]) -> SystemToolResult:
    """Retrieval sobre la memoria agéntica: perfil del usuario, resúmenes
    de sesiones pasadas, principios propios y estado de aprendizaje.

    Mismo patrón que search_corpus pero sobre el corpus personal/agéntico:
    los items son derivados de los stores canónicos (episodios, user model,
    principios, mastery del Tutor) con provenance de vuelta a la fuente.
    """
    from ipa.agent.memory_store import MemoryIndexer, MemoryStore
    from ipa.agent.agent_memory import AgentMemory
    from ipa.agent.user_model import UserModelStore
    from ipa.agent.strategic_memory import StrategicMemoryStore
    from ipa.tutor.tutor_runtime import TutorStore

    query = str(args.get("query") or "").strip()
    scope_arg = str(args.get("scope") or "").strip().lower()
    scopes = [scope_arg] if scope_arg in ("user", "agent", "episodic", "tutor") else None
    limit = args.get("limit", 5)
    try:
        limit = max(1, min(10, int(limit)))
    except (TypeError, ValueError):
        limit = 5

    store = MemoryStore()
    sources = []
    try:
        indexer = MemoryIndexer(store)
        sync_kwargs: dict[str, Any] = {}
        try:
            sync_kwargs["memory"] = AgentMemory()
            sources.append("memory")
        except Exception:
            pass
        try:
            sync_kwargs["user_model"] = UserModelStore()
            sources.append("user_model")
        except Exception:
            pass
        try:
            sync_kwargs["strategic"] = StrategicMemoryStore()
            sources.append("strategic")
        except Exception:
            pass
        try:
            sync_kwargs["tutor"] = TutorStore()
            sources.append("tutor")
        except Exception:
            pass
        try:
            indexer.sync(**sync_kwargs)
        finally:
            for s in sync_kwargs.values():
                close = getattr(s, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
        items = _recall_hybrid(store, query, scopes=scopes, limit=limit)
    except Exception as exc:
        return SystemToolResult(
            tool_name="recall_memory", ok=False, summary="", data={},
            error=f"recall_memory falló: {exc}",
        )
    finally:
        store.close()

    if not items:
        return SystemToolResult(
            tool_name="recall_memory", ok=True,
            summary="Sin recuerdos relevantes en esa memoria.",
            data={"items": [], "scopes": scopes or "all"},
        )
    data = {
        "items": [
            {"scope": i.scope, "kind": i.kind, "text": i.text,
             "source": i.source_ref, "confidence": i.confidence,
             "updated_at": i.updated_at}
            for i in items
        ],
        "scopes": scopes or "all",
    }
    summary = f"{len(items)} recuerdos relevantes: " + "; ".join(
        i.text[:80] for i in items[:3]
    )
    return SystemToolResult(tool_name="recall_memory", ok=True, summary=summary, data=data)


def tool_list_research_agenda(args: dict[str, Any]) -> SystemToolResult:
    """Lista tópicos de baja confianza y propuestas de research activo."""
    from ipa.agent.uncertainty import UncertaintyStore

    store = UncertaintyStore()
    try:
        low_confidence = store.list_low_confidence(limit=10)
        proposals = store.list_proposals(status="pending", limit=10)
    finally:
        store.close()
    data = {
        "low_confidence_topics": [
            {"topic": t.topic, "confidence": round(t.confidence, 2),
             "observations": t.observation_count, "source": t.source}
            for t in low_confidence
        ],
        "pending_proposals": [
            {"proposal_id": p.proposal_id, "topic": p.topic,
             "confidence": round(p.current_confidence, 2), "rationale": p.rationale}
            for p in proposals
        ],
    }
    summary = (
        f"{len(low_confidence)} tópicos de baja confianza, "
        f"{len(proposals)} propuestas de research pendientes."
    )
    return SystemToolResult(
        tool_name="list_research_agenda", ok=True, summary=summary, data=data,
    )


def tool_set_user_goal(args: dict[str, Any]) -> SystemToolResult:
    """Declara o actualiza un goal del usuario (user model transversal)."""
    from ipa.agent.user_model import UserModelStore

    description = str(args.get("description", "") or "").strip()
    if not description:
        return SystemToolResult(
            tool_name="set_user_goal", ok=False, summary="", data={},
            error="se requiere 'description' (el goal a declarar)",
        )
    store = UserModelStore()
    try:
        goal = store.add_goal(description, source="declared", confidence=1.0)
    finally:
        store.close()
    return SystemToolResult(
        tool_name="set_user_goal", ok=True,
        summary=f"Goal declarado: '{description}'. Ahora lo tengo en cuenta en respuestas.",
        data={"goal_id": goal.goal_id, "description": goal.description, "status": goal.status},
    )


def tool_set_user_interest(args: dict[str, Any]) -> SystemToolResult:
    """Declara un interés explícito del usuario."""
    from ipa.agent.user_model import UserModelStore

    topic = str(args.get("topic", "") or "").strip()
    if not topic:
        return SystemToolResult(
            tool_name="set_user_interest", ok=False, summary="", data={},
            error="se requiere 'topic' (el interés a declarar)",
        )
    store = UserModelStore()
    try:
        interest = store.declare_interest(topic)
    finally:
        store.close()
    return SystemToolResult(
        tool_name="set_user_interest", ok=True,
        summary=f"Interés declarado: '{topic}'. Lo pondero alto en futuras respuestas.",
        data={"topic": interest.topic, "score": interest.score},
    )


# ---------------------------------------------------------------------------
# Unified tool registry — single source of truth for the LLM-facing catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemToolSpec:
    """Registry entry: one capability the agent can invoke.

    description/args_doc feed TOOL_CATALOG verbatim — the LLM sees exactly
    what is registered, so catalog and registry cannot drift apart.
    chat_visible=False keeps the name dispatchable while hiding it from the
    catalog (aliases and legacy names the model may still emit).
    """
    name: str
    description: str
    args_doc: str
    fn: Callable[[dict[str, Any]], SystemToolResult]
    chat_visible: bool = True


_SYSTEM_TOOLS: tuple[SystemToolSpec, ...] = (
    SystemToolSpec(
        name="get_system_status",
        description="estado de procesos (ingesta, investigación web) y archivos en Landing.",
        args_doc="{}",
        fn=tool_get_system_status,
    ),
    SystemToolSpec(
        name="search_corpus",
        description="busca en el corpus. Cita hits como [n].",
        args_doc='{"query": "texto a buscar", "limit": <int 1-15, opcional>, "date_from"/"date_to": "YYYY-MM-DD", opcional}',
        fn=tool_search_corpus,
    ),
    SystemToolSpec(
        name="list_topics",
        description="lista tópicos del corpus.",
        args_doc='{"limit": <int 1-50, opcional>}',
        fn=tool_list_topics,
    ),
    SystemToolSpec(
        name="list_promotions",
        description="cola de promoción al corpus.",
        args_doc="{}",
        fn=tool_list_promotions,
    ),
    SystemToolSpec(
        name="get_report",
        description="lee el último reporte.",
        args_doc="{}",
        fn=tool_get_report,
    ),
    SystemToolSpec(
        name="compile_report",
        description="compila un reporte desde el corpus.",
        args_doc='{"query": "tema a buscar"} o {"document_ids": ["doc:1", "doc:2"], "topic": "descripción"}',
        fn=tool_compile_report,
    ),
    SystemToolSpec(
        name="research_topic",
        description=(
            "investiga un tema en la web (async). URLs incluidas en la query se "
            "scrapean directo como fuentes explícitas. Si ya investigué algo muy "
            "parecido hace poco, la tool lo indica y no relanza — pasar force=true "
            "solo si el usuario pide explícitamente investigar de nuevo."
        ),
        args_doc='{"query": "tema o URL a investigar", "max_urls": <int 1-20>, "max_seconds": <int 30-600>, "force": <bool>}',
        fn=tool_research_topic,
    ),
    SystemToolSpec(
        name="run_ingestion",
        description=(
            "lanza ingesta de fuentes (async). VOS determinás la ventana de fechas "
            "según el objetivo: sin args usa los baselines por sitio (incremental); "
            "days_back N amplía la ventana global (N>30 fuerza re-descubrimiento del "
            "historial, más lento); date_from/date_to fija un rango exacto."
        ),
        args_doc='{} o {"days_back": <int 0-365>} o {"date_from": "YYYY-MM-DD", "date_to": "YYYY-MM-DD"}',
        fn=tool_run_ingestion,
    ),
    SystemToolSpec(
        name="list_sources",
        description="lista las URLs configuradas para scraping.",
        args_doc="{}",
        fn=tool_list_sources,
    ),
    SystemToolSpec(
        name="plan_task",
        description="planifica y lanza una tarea multi-paso en background.",
        args_doc='{"goal": "objetivo en una frase", "max_subtasks": <int 1-10, opcional>, "max_tools_per_subtask": <int 1-5, opcional>}',
        fn=tool_plan_task,
    ),
    SystemToolSpec(
        name="list_tasks",
        description="lista tareas y su progreso.",
        args_doc='{"status": "running|planned|completed|failed|paused (opcional)", "limit": <int 1-50, opcional>}',
        fn=tool_list_tasks,
    ),
    SystemToolSpec(
        name="get_task",
        description="detalle de una tarea: sub-tasks, progreso, resultados parciales.",
        args_doc='{"task_id": "task:..."}',
        fn=tool_get_task,
    ),
    SystemToolSpec(
        name="resume_task",
        description="resume una tarea paused/failed desde el último sub-task pendiente.",
        args_doc='{"task_id": "task:..."}',
        fn=tool_resume_task,
    ),
    SystemToolSpec(
        name="get_user_profile",
        description="lee el perfil del usuario.",
        args_doc="{}",
        fn=tool_get_user_profile,
    ),
    SystemToolSpec(
        name="set_user_goal",
        description="declara un goal del usuario.",
        args_doc='{"description": "descripción del goal"}',
        fn=tool_set_user_goal,
    ),
    SystemToolSpec(
        name="set_user_interest",
        description="declara un interés explícito del usuario. Lo pondero alto en futuras respuestas.",
        args_doc='{"topic": "tema de interés"}',
        fn=tool_set_user_interest,
        chat_visible=False,
    ),
    SystemToolSpec(
        name="recall_memory",
        description="busca en mi memoria: perfil del usuario, resúmenes de conversaciones pasadas, principios aprendidos, progreso de aprendizaje.",
        args_doc='{"query": "qué recordar", "scope": "user|agent|episodic|tutor (opcional)", "limit": <int 1-10, opcional>}',
        fn=tool_recall_memory,
    ),
    SystemToolSpec(
        name="list_research_agenda",
        description="lista tópicos de baja confianza (donde sé poco) y propuestas de investigación activa pendientes.",
        args_doc="{}",
        fn=tool_list_research_agenda,
        chat_visible=False,
    ),
    # Aliases ocultos: dispatchables pero no aparecen en el catálogo.
    SystemToolSpec(
        name="run_pipeline",
        description="alias de run_ingestion — mismo job de ingesta.",
        args_doc='{"days_back": <int 0-365>}',
        fn=_run_pipeline_alias,
        chat_visible=False,
    ),
)

SYSTEM_TOOL_NAMES = frozenset(spec.name for spec in _SYSTEM_TOOLS)
_SYSTEM_IMPLEMENTATIONS = {spec.name: spec.fn for spec in _SYSTEM_TOOLS}

# Cache corto de tools read-only entre turnos. Se excluye get_system_status:
# su semántica es "estado AHORA" (procesos corriendo, research en vuelo) y
# servir una foto vieja sería incorrecto. TTL: IPA_TOOL_CACHE_TTL (default
# 30s, 0 desactiva).
_TOOL_CACHEABLE = frozenset({"list_topics", "get_user_profile", "list_promotions"})
_TOOL_CACHE_TTL = float(os.environ.get("IPA_TOOL_CACHE_TTL", "30") or 30)
_TOOL_CACHE: dict[str, tuple[float, SystemToolResult]] = {}


def execute_system_tool(tool_name: str, arguments: dict[str, Any]) -> SystemToolResult:
    """Execute a system tool deterministically. Raises on unknown tool."""
    if tool_name not in SYSTEM_TOOL_NAMES:
        raise ValueError(f"unknown system tool: {tool_name}; valid: {sorted(SYSTEM_TOOL_NAMES)}")
    if tool_name not in _SYSTEM_IMPLEMENTATIONS:
        raise ValueError(f"system tool not implemented: {tool_name}")
    impl = _SYSTEM_IMPLEMENTATIONS[tool_name]
    # Cache corto para tools read-only baratas pero repetidas entre turnos
    # (get_system_status se pedía cada pocos turnos). IPA_TOOL_CACHE_TTL=0 off.
    if tool_name in _TOOL_CACHEABLE and _TOOL_CACHE_TTL > 0:
        key = tool_name + "|" + json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        now = time.time()
        hit = _TOOL_CACHE.get(key)
        if hit is not None and now - hit[0] <= _TOOL_CACHE_TTL:
            return hit[1]
        result = impl(dict(arguments))
        if result.ok:
            _TOOL_CACHE[key] = (now, result)
            if len(_TOOL_CACHE) > 64:
                oldest = min(_TOOL_CACHE, key=lambda k: _TOOL_CACHE[k][0])
                _TOOL_CACHE.pop(oldest, None)
        return result
    return impl(dict(arguments))


def tool_specs() -> list[dict[str, str]]:
    """Registry specs for external frontiers (MCP proxy). Same source of
    truth as the LLM catalog — the surfaces cannot drift apart."""
    return [
        {"name": spec.name, "description": spec.description, "args_doc": spec.args_doc}
        for spec in _SYSTEM_TOOLS
        if spec.chat_visible
    ]


# ---------------------------------------------------------------------------
# Tool catalog — progressive unlocking (reduce cognitive load for the 9B)
# ---------------------------------------------------------------------------

_CATALOG_INTRO = """Herramientas. Si la petición requiere una, respondé SOLO el marcador, sin texto antes ni después:

[TOOL:nombre_tool]{"arg": "valor"}

Ejemplos de conversación completa:
  Usuario: "buscá fotónica"
  Asistente: [TOOL:search_corpus]{"query": "fotónica"}

  Usuario: "investigá RAG retrieval"
  Asistente: [TOOL:research_topic]{"query": "RAG retrieval"}

  Usuario: "estado del sistema"
  Asistente: [TOOL:get_system_status]{}

  Usuario: "qué tareas tengo"
  Asistente: [TOOL:list_tasks]{}

  Usuario: "te acordás qué hablamos ayer?"
  Asistente: [TOOL:recall_memory]{"query": "conversación anterior"}

  Usuario: "qué sabés de mis intereses?"
  Asistente: [TOOL:recall_memory]{"query": "intereses", "scope": "user"}

INCORRECTO — nunca respondas texto cuando se necesita una tool:
  "Voy a investigar X" ← incorrecto, no ejecuta nada
  "La investigación está activa" ← incorrecto, no ejecuta nada
  "Inicié la búsqueda de X" ← incorrecto, no ejecuta nada
  Correcto: [TOOL:research_topic]{"query": "X"}

Si la petición requiere una herramienta, tu ÚNICA respuesta es el marcador. Nada más.

Tools:"""

_CATALOG_OUTRO = """
Máximo 3 tools por turno. Si no requiere herramienta, respondé normalmente. Si pedís algo que no está en la lista, respondé que no tenés esa capacidad aún."""

# Tools base: siempre visibles (entry points para los requests más comunes).
# El agente empieza con 6 tools y desbloquea más a medida que las usa.
BASE_TOOLS: frozenset[str] = frozenset({
    "search_corpus",      # "buscá X"
    "research_topic",     # "investigá X"
    "plan_task",          # "investigá X y hacé un reporte" (multi-paso)
    "run_ingestion",      # "actualizá el corpus" / "corré la ingesta"
    "get_system_status",  # "estado del sistema"
    "set_user_goal",      # "estoy armando un paper sobre X"
    "get_user_profile",   # "qué sabés de mí?"
    "recall_memory",      # "qué te acordás de..." / memoria antes de preguntar
})

# Grafo de progresión: al usar una tool, se desbloquean estas.
# Determinístico — el LLM no decide qué se desbloquea.
TOOL_PROGRESSION: dict[str, frozenset[str]] = {
    "search_corpus": frozenset({"compile_report", "list_topics"}),
    "research_topic": frozenset({"compile_report", "list_promotions"}),
    "plan_task": frozenset({"list_tasks", "get_task", "resume_task"}),
    "compile_report": frozenset({"get_report"}),
    "run_ingestion": frozenset({"list_promotions", "list_sources"}),
}

# Tools que nunca aparecen en el catálogo (solo dispatch internamente).
_ALWAYS_HIDDEN: frozenset[str] = frozenset({
    "run_pipeline", "list_research_agenda", "set_user_interest",
})


def build_tool_catalog(unlocked: set[str] | None = None) -> str:
    """Construye el catálogo dinámico. Si unlocked es None, usa solo BASE_TOOLS."""
    if unlocked is None:
        unlocked = set(BASE_TOOLS)
    lines = [_CATALOG_INTRO]
    for spec in _SYSTEM_TOOLS:
        # Solo mostrar tools que están unlocked, son chat_visible, y no están
        # en la lista de siempre ocultas.
        if spec.name in unlocked and spec.chat_visible and spec.name not in _ALWAYS_HIDDEN:
            lines.append(f"- {spec.name}: {spec.description} Args: {spec.args_doc}")
    return "\n".join(lines) + _CATALOG_OUTRO


# Catálogo legacy (todas las tools visibles) — para compatibilidad con tests.
TOOL_CATALOG = build_tool_catalog(
    set(BASE_TOOLS) | {t for tools in TOOL_PROGRESSION.values() for t in tools}
)


def unlock_after_tool(tool_name: str, current_unlocked: set[str]) -> set[str]:
    """Tras ejecutar una tool, retorna el nuevo set de tools desbloqueadas."""
    new_unlocked = set(current_unlocked)
    if tool_name in TOOL_PROGRESSION:
        new_unlocked |= TOOL_PROGRESSION[tool_name]
    return new_unlocked


def parse_tool_marker(text: str) -> tuple[str, dict[str, Any]] | None:
    """Parse a [TOOL:name]{json} marker from a generation. Returns (name, args) or None.

    Tolerant parsing: el modelo a veces garblea el marker (bug del 2026-09-08:
    '[TRUN_INGESTITION]' en vez de '[TOOL:run_ingestion]'). Acepta:
      - [TOOL:name]{json}  (formato canónico)
      - [TOOL:name] {json} (espacio antes del JSON)
      - [TOOL name]{json}  (sin dos puntos)
      - [TRUN_NAME]{json}  (garbled — fuzzy match contra tool names conocidos)
      - [TOOL:NAME]{json}  (mayúsculas)
    """
    # 1. Formato canónico: [TOOL:name]{json}
    match = re.search(r"\[TOOL:([a-zA-Z_]+)\]\s*(\{.*?\})", text, re.DOTALL)
    if match:
        name = match.group(1).lower()
        if name in SYSTEM_TOOL_NAMES:
            try:
                args = json.loads(match.group(2))
            except ValueError:
                args = {}
            return name, args
    # 2. Formato sin dos puntos: [TOOL name]{json}
    match = re.search(r"\[TOOL\s+([a-zA-Z_]+)\]\s*(\{.*?\})", text, re.DOTALL)
    if match:
        name = match.group(1).lower()
        if name in SYSTEM_TOOL_NAMES:
            try:
                args = json.loads(match.group(2))
            except ValueError:
                args = {}
            return name, args
    # 3. Fuzzy: cualquier [XXX]{json} donde XXX se parece a un tool name conocido.
    # Acepta garbled como [TRUN_INGESTION] → run_ingestion, [TRUN_PIPELINE] → run_pipeline.
    # Determinístico: itera siempre en orden alfabético y resuelve
    # colisiones de key_part (get_report/compile_report → "report") por
    # similitud de nombre completo, nunca por orden del frozenset.
    match = re.search(r"\[([A-Z_]{4,})\]\s*(\{.*?\})", text, re.DOTALL)
    if match:
        raw_name = match.group(1).lower()
        payload = match.group(2)

        def _load_args() -> dict[str, Any]:
            try:
                return json.loads(payload)
            except ValueError:
                return {}

        candidates = sorted(SYSTEM_TOOL_NAMES)
        # a) Contención de nombre completo: "trun_ingestion" ⊃ "run_ingestion"
        for name in candidates:
            if name in raw_name:
                return name, _load_args()
        # b) key_part única: "ingestion" solo pertenece a run_ingestion;
        #    si el key_part es compartido (p.ej. "report"), no decide aquí.
        key_parts = {n: (n.split("_", 1)[-1] if "_" in n else n) for n in candidates}
        key_counts: dict[str, int] = {}
        for kp in key_parts.values():
            key_counts[kp] = key_counts.get(kp, 0) + 1
        for name in candidates:
            kp = key_parts[name]
            if key_counts[kp] == 1 and kp in raw_name:
                return name, _load_args()
        # c) Scoring por nombre completo (empate → gana el alfabéticamente menor)
        best_name = None
        best_score = 0.0
        for name in candidates:
            overlap = sum(1 for c in name if c in raw_name)
            score = overlap / max(len(name), 1)
            if score > best_score:
                best_score = score
                best_name = name
        if best_name and best_score >= 0.5:
            return best_name, _load_args()
    return None


__all__ = [
    "SYSTEM_TOOL_NAMES", "SystemToolResult", "SystemToolSpec",
    "execute_system_tool", "TOOL_CATALOG", "parse_tool_marker",
]
