"""Local web dashboard for IPA ingestion, Reporter and curation workflows."""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sqlite3
import subprocess
import sys
import threading
import urllib.parse
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
VENV_PYTHONW = str(ROOT / ".venv" / "Scripts" / "pythonw.exe")
if not Path(VENV_PYTHON).exists():
    VENV_PYTHON = sys.executable
if not Path(VENV_PYTHONW).exists():
    VENV_PYTHONW = VENV_PYTHON
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
WEB_ROOT = ROOT / "web"
STATIC_ROOT = WEB_ROOT / "static"
SOURCES_DB = ROOT / "outputs" / "web_dashboard" / "sources.json"
STATE_DB = ROOT / "outputs" / "web_dashboard" / "dashboard.db"
REPORTER_ROOT = ROOT / "outputs" / "reporter"
REPORTER_PROGRESS = ROOT / "outputs" / "web_dashboard" / "reporter_progress.json"
PIPELINE_PROGRESS = ROOT / "outputs" / "web_dashboard" / "pipeline_progress.json"
MAIN_CORPUS = ROOT / "outputs" / "experiments" / "E12-corpus"
SCRAPE_CONFIG = ROOT / "configs" / "scrape_sites.yaml"
JOBS: dict[str, subprocess.Popen] = {}
JOBS_LOCK = threading.Lock()
DEEP_DIVE_PROVIDER = None
_CLEANING_LOCK = threading.Lock()
_CLEANING_IN_PROGRESS = False
# Active reporter output directory — set by run_full_pipeline, read by dashboard_state
_ACTIVE_REPORTER_OUTPUT: Path | None = None
DEEP_DIVE_LOCK = threading.Lock()
PIPELINE_THREAD = None
PIPELINE_LOCK = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def safe_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("La URL debe usar http o https y tener hostname")
    if parsed.username or parsed.password:
        raise ValueError("Las URLs con credenciales no están permitidas")
    return value.strip()


def dashboard_connection() -> sqlite3.Connection:
    STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(STATE_DB))
    connection.execute("CREATE TABLE IF NOT EXISTS report_reviews (report_id TEXT PRIMARY KEY, status TEXT NOT NULL, decided_by TEXT NOT NULL, note TEXT, decided_at TEXT NOT NULL)")
    connection.commit()
    return connection


def load_sources() -> dict[str, Any]:
    data = read_json(SOURCES_DB, {"added": [], "disabled": []})
    return {"added": data.get("added", []), "disabled": data.get("disabled", [])}


def base_sources() -> list[dict[str, Any]]:
    try:
        data = yaml.safe_load(SCRAPE_CONFIG.read_text(encoding="utf-8")) or {}
        return [{"url": str(site.get("url")), "days_back": site.get("days_back", 2), "max_articles": site.get("max_articles", 20)} for site in data.get("sites", []) if site.get("url")]
    except (OSError, yaml.YAMLError):
        return []


def save_sources(data: dict[str, Any]) -> None:
    SOURCES_DB.parent.mkdir(parents=True, exist_ok=True)
    temp = SOURCES_DB.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(SOURCES_DB)


def effective_scrape_config() -> Path:
    base = yaml.safe_load(SCRAPE_CONFIG.read_text(encoding="utf-8")) or {}
    sources = load_sources()
    disabled = set(sources["disabled"])
    # Build a map of added overrides (days_back, max_articles) by URL
    added_map = {item.get("url"): item for item in sources["added"] if item.get("url") not in disabled}
    # Start with base sites, applying overrides where they exist
    sites = []
    for site in base.get("sites", []):
        url = site.get("url")
        if url in disabled:
            continue
        if url in added_map:
            # Use override values
            override = added_map.pop(url)
            merged = dict(site)
            merged["days_back"] = override.get("days_back", site.get("days_back", 2))
            if "max_articles" in override:
                merged["max_articles"] = override["max_articles"]
            sites.append(merged)
        else:
            sites.append(site)
    # Add remaining added sources (not in base, not disabled)
    sites.extend(added_map.values())
    path = ROOT / "outputs" / "web_dashboard" / "effective_scrape_sites.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"sites": sites}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _force_rmtree(path: Path, max_retries: int = 3) -> bool:
    """Robust rmtree for Windows: retries on PermissionError (SQLite WAL/SHM locks)."""
    import shutil
    import time as _time
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            return True
        except PermissionError:
            if attempt < max_retries - 1:
                _time.sleep(0.5 * (attempt + 1))
            else:
                # Last resort: mark files for deletion on reboot via os.unlink
                for item in path.rglob("*"):
                    if item.is_file():
                        try:
                            item.unlink()
                        except Exception:
                            pass
                try:
                    shutil.rmtree(path)
                    return True
                except Exception:
                    return False
        except Exception:
            return False
    return False


def active_reporter_output() -> Path:
    """Return the active reporter output directory.

    Priority:
    1. _ACTIVE_REPORTER_OUTPUT (set by run_full_pipeline during execution)
    2. Most recent directory under REPORTER_ROOT/quality-check/ that has a report.json
    3. Fall back to legacy hardcoded path
    """
    global _ACTIVE_REPORTER_OUTPUT
    if _ACTIVE_REPORTER_OUTPUT is not None:
        return _ACTIVE_REPORTER_OUTPUT
    qc_dir = REPORTER_ROOT / "quality-check"
    if qc_dir.exists():
        candidates = []
        for d in qc_dir.iterdir():
            if d.is_dir() and (d / "report.json").exists():
                candidates.append(d)
        if candidates:
            # Most recently modified report.json
            candidates.sort(key=lambda d: (d / "report.json").stat().st_mtime, reverse=True)
            return candidates[0]
    # Legacy fallback
    return REPORTER_ROOT / "quality-check" / "optimized-llm-2026-08"


def db_counts(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    if path.is_dir():
        # Check if it's a LanceDB directory
        try:
            import lancedb as _lancedb
            db = _lancedb.connect(str(path))
            tables = db.table_names() if hasattr(db, "table_names") else [t for t in (db.list_tables().tables if hasattr(db, "list_tables") else [])]
            if "chunks" in tables:
                tbl = db.open_table("chunks")
                return {"exists": True, "path": str(path), "backend": "lancedb", "chunks": tbl.count_rows()}
        except Exception:
            pass
        return {"exists": True, "path": str(path), "backend": "directory", "files": sum(1 for item in path.rglob("*") if item.is_file())}
    try:
        with sqlite3.connect(str(path)) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            counts = {"exists": True, "path": str(path)}
            if "documents" in tables:
                counts["documents"] = conn.execute("SELECT COUNT(*) FROM documents WHERE tombstoned=0").fetchone()[0]
            if "chunks" in tables:
                counts["chunks"] = conn.execute("SELECT COUNT(*) FROM chunks WHERE tombstoned=0").fetchone()[0]
            if "embedding_jobs" in tables:
                counts["embeddings_complete"] = conn.execute("SELECT COUNT(*) FROM embedding_jobs WHERE status='complete'").fetchone()[0]
                counts["embeddings_pending"] = conn.execute("SELECT COUNT(*) FROM embedding_jobs WHERE status!='complete'").fetchone()[0]
            return counts
    except sqlite3.Error as exc:
        return {"exists": True, "path": str(path), "error": str(exc)}


def scrape_counts() -> dict[str, Any]:
    """Count scraped files in Landing/web by site."""
    web_dir = ROOT / "Landing" / "web"
    if not web_dir.exists():
        return {"total_files": 0, "sites": 0, "by_site": {}}
    by_site: dict[str, int] = {}
    total = 0
    for site_dir in web_dir.iterdir():
        if site_dir.is_dir():
            count = sum(1 for f in site_dir.rglob("*") if f.is_file() and f.name != "scrape_history.db")
            if count > 0:
                by_site[site_dir.name] = count
                total += count
    return {"total_files": total, "sites": len(by_site), "by_site": by_site}


def archive_counts() -> dict[str, Any]:
    """Count files in Archive/."""
    archive_dir = ROOT / "Archive"
    if not archive_dir.exists():
        return {"total_files": 0}
    total = sum(1 for f in archive_dir.rglob("*") if f.is_file())
    return {"total_files": total}


def latest_report() -> dict[str, Any] | None:
    reports = sorted(REPORTER_ROOT.glob("**/report.json"), key=lambda p: p.stat().st_mtime, reverse=True) if REPORTER_ROOT.exists() else []
    for path in reports:
        report = read_json(path, {})
        categories = report.get("categories", [])
        if not isinstance(categories, list) or any(not isinstance(category, dict) or not isinstance(category.get("uncertainties", []), list) for category in categories):
            continue
        report["path"] = str(path)
        report["updated_at"] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        return report
    return None


def report_history(limit: int = 20) -> list[dict[str, Any]]:
    results = []
    for path in sorted(REPORTER_ROOT.glob("**/report.json"), key=lambda item: item.stat().st_mtime, reverse=True) if REPORTER_ROOT.exists() else []:
        report = read_json(path, {})
        if not isinstance(report.get("categories"), list):
            continue
        results.append({"report_id": report.get("report_id"), "period": report.get("period"), "status": report.get("status"), "generated_at": report.get("generation", {}).get("generated_at"), "updated_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(), "path": str(path), "category_count": len(report["categories"])})
    return results[:limit]


def load_report_by_path(path_str: str) -> dict[str, Any] | None:
    """Load a specific report by its path, with the same validation as latest_report."""
    if not path_str:
        return None
    candidate = Path(path_str).expanduser().resolve()
    if not candidate.is_file() or candidate.name != "report.json":
        return None
    if REPORTER_ROOT.resolve() not in candidate.parents:
        return None
    report = read_json(candidate, {})
    categories = report.get("categories", [])
    if not isinstance(categories, list) or any(not isinstance(category, dict) or not isinstance(category.get("uncertainties", []), list) for category in categories):
        return None
    report["path"] = str(candidate)
    report["updated_at"] = datetime.fromtimestamp(candidate.stat().st_mtime, timezone.utc).isoformat()
    return report


def topic_details(category_id: str) -> dict[str, Any]:
    report = latest_report()
    if not report:
        raise FileNotFoundError("no hay reporte disponible")
    topic = next((item for item in report["categories"] if item.get("category_id") == category_id), None)
    if topic is None:
        raise FileNotFoundError(category_id)
    db = Path(report["path"]).parent / "reporter.db"
    documents = []
    if db.exists():
        with sqlite3.connect(str(db)) as connection:
            connection.row_factory = sqlite3.Row
            ids = topic.get("document_ids", [])
            if ids:
                marks = ",".join("?" for _ in ids)
                rows = connection.execute(f"SELECT document_id, original_path, source_url, canonical_url, source_domain, title, published_at, published_at_confidence, quality_score FROM document_metadata WHERE document_id IN ({marks})", ids).fetchall()
                documents = [dict(row) for row in rows]
                # Enrich each document with its curation reason from document_decisions
                if documents:
                    doc_id_list = [d["document_id"] for d in documents]
                    dec_marks = ",".join("?" for _ in doc_id_list)
                    dec_rows = connection.execute(f"SELECT document_id, payload_json FROM document_decisions WHERE document_id IN ({dec_marks})", doc_id_list).fetchall()
                    reason_map = {}
                    for dec_id, payload in dec_rows:
                        try:
                            payload_dict = json.loads(payload)
                            reason_map[dec_id] = payload_dict.get("reason")
                        except ValueError:
                            pass
                    for doc in documents:
                        doc["reason"] = reason_map.get(doc["document_id"])
    return {"report": {"report_id": report.get("report_id"), "generated_at": report.get("generation", {}).get("generated_at"), "updated_at": report.get("updated_at")}, "topic": topic, "documents": documents}


def llm_status() -> dict[str, Any]:
    model = ROOT / "models" / "Qwen3.5-9B-exl3-3.0bpw"
    extension = ROOT / "exllamav3-dev" / "exllamav3_ext.cp312-win_amd64.pyd"
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        gpu = torch.cuda.get_device_name(0) if cuda else None
    except (ImportError, RuntimeError):
        cuda, gpu = False, None
    return {"model_present": model.exists(), "extension_present": extension.exists(), "cuda_available": cuda, "gpu": gpu, "ready": model.exists() and extension.exists() and cuda}


def get_deep_dive_provider():
    global DEEP_DIVE_PROVIDER
    with DEEP_DIVE_LOCK:
        if DEEP_DIVE_PROVIDER is not None and DEEP_DIVE_PROVIDER.is_loaded():
            return DEEP_DIVE_PROVIDER
        with JOBS_LOCK:
            reporter_job = JOBS.get("reporter")
            if reporter_job and reporter_job.poll() is None:
                return None
        if not llm_status().get("ready"):
            return None
        from ipa.exl3_provider import create_star_provider
        DEEP_DIVE_PROVIDER = create_star_provider(interactive=True)
        DEEP_DIVE_PROVIDER.load()
        return DEEP_DIVE_PROVIDER


def process_status() -> dict[str, Any]:
    result = {}
    state_dir = MAIN_CORPUS / "process_state"
    # Legacy process state files (from standalone fast path runs)
    for name in ["scraper", "pipeline", "lancedb", "enrichment", "rechunk"]:
        path = state_dir / f"{name}.json"
        result[name] = read_json(path, {"status": "not_started", "path": str(path)})
    # Pipeline thread status (from /api/pipeline/run)
    pipeline_progress = read_json(PIPELINE_PROGRESS, {})
    if pipeline_progress:
        result["rechunk"] = {"status": pipeline_progress.get("status", "running"), "detail": "pipeline completo", "percent": pipeline_progress.get("percent", 0), "stage": pipeline_progress.get("stage", "")}
    # Dashboard-managed jobs (from pipeline runs) — these are the real live ones
    with JOBS_LOCK:
        for name in list(JOBS.keys()):
            proc = JOBS[name]
            if proc.poll() is not None:
                # Process finished — keep status briefly then remove from JOBS
                # so it doesn't show as "error" forever
                status = "done" if proc.returncode == 0 else "error"
                result[f"web_{name}"] = {"status": status, "pid": proc.pid, "returncode": proc.returncode}
                if name == "scraper":
                    result["scraper"] = {"status": status, "pid": proc.pid, "detail": "web scraper"}
                elif name == "pipeline":
                    result["pipeline"] = {"status": status, "pid": proc.pid, "detail": "fast path (BM25 + LanceDB)"}
                elif name == "lancedb":
                    result["lancedb"] = {"status": status, "pid": proc.pid, "detail": "LanceDB re-index"}
                elif name == "reporter":
                    result["enrichment"] = {"status": status, "pid": proc.pid, "detail": "reporter (BGE-M3 + Qwen)"}
                    progress = read_json(REPORTER_PROGRESS, {})
                    if progress:
                        result["web_reporter"].update(progress)
                elif name == "reporter_fast":
                    result["rechunk"] = {"status": status, "pid": proc.pid, "detail": "reporter rápido"}
                # Remove finished jobs from JOBS so they don't persist as "error"
                del JOBS[name]
            else:
                status = "running"
                result[f"web_{name}"] = {"status": status, "pid": proc.pid}
                if name == "scraper":
                    result["scraper"] = {"status": status, "pid": proc.pid, "detail": "web scraper"}
                elif name == "pipeline":
                    result["pipeline"] = {"status": status, "pid": proc.pid, "detail": "fast path (BM25 + LanceDB)"}
                elif name == "lancedb":
                    result["lancedb"] = {"status": status, "pid": proc.pid, "detail": "LanceDB re-index"}
                elif name == "reporter":
                    result["enrichment"] = {"status": status, "pid": proc.pid, "detail": "reporter (BGE-M3 + Qwen)"}
                    progress = read_json(REPORTER_PROGRESS, {})
                    if progress:
                        result["web_reporter"].update(progress)
                elif name == "reporter_fast":
                    result["rechunk"] = {"status": status, "pid": proc.pid, "detail": "reporter rápido"}
    return result


def _log_tail(path: Path, lines: int = 80) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return list(handle.readlines())[-lines:]
    except OSError:
        return []


def process_detail(name: str) -> dict[str, Any]:
    """Return rich detail for a process: state, metrics, errors, log tail."""
    name = str(name).strip()
    if not name or len(name) > 64 or not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("nombre de proceso inválido")
    state_dir = MAIN_CORPUS / "process_state"
    log_dir = ROOT / "outputs" / "web_dashboard" / "logs"
    detail: dict[str, Any] = {"name": name}

    # Map process names to their log files
    log_map = {
        "scraper": "scraper.log",
        "pipeline": "fast_path.log",
        "lancedb": "lancedb.log",
        "enrichment": "reporter.log",
        "rechunk": None,  # pipeline completo — combined log below
    }

    if name.startswith("web_"):
        kind = name.removeprefix("web_")
        with JOBS_LOCK:
            proc = JOBS.get(kind)
            if proc:
                detail["pid"] = proc.pid
                detail["returncode"] = proc.poll()
                detail["status"] = "running" if proc.poll() is None else ("done" if proc.returncode == 0 else "error")
            else:
                detail["status"] = "not_started"
        if kind == "reporter":
            detail["state"] = read_json(REPORTER_PROGRESS, {})
        else:
            state_path = state_dir / f"{kind}.json"
            detail["state"] = read_json(state_path, {})
        log_file = log_map.get(kind, f"{kind}.log")
        if log_file:
            detail["log_path"] = str(log_dir / log_file)
            detail["log"] = _log_tail(log_dir / log_file)
        else:
            detail["log"] = []
    elif name == "rechunk":
        # Pipeline completo — reads from PIPELINE_PROGRESS + combines all logs
        progress = read_json(PIPELINE_PROGRESS, {})
        detail["state"] = progress
        detail["status"] = progress.get("status", "not_started")
        detail["stage"] = progress.get("stage", "")
        detail["percent"] = progress.get("percent")
        detail["timestamp"] = progress.get("updated_at")
        detail["pid"] = None  # runs as thread, not subprocess
        detail["returncode"] = None
        # Combine logs from all sub-processes
        combined_log = []
        for log_name in ["scraper.log", "fast_path.log", "reporter.log"]:
            lines = _log_tail(log_dir / log_name, lines=30)
            if lines:
                combined_log.append(f"--- {log_name} ---")
                combined_log.extend(lines)
        detail["log"] = combined_log[-80:] if combined_log else []
        detail["log_path"] = "combined (scraper + fast_path + reporter)"
    else:
        # Check JOBS first for live status (scraper, pipeline, lancedb, enrichment)
        kind = name
        if name == "enrichment":
            kind = "reporter"
        with JOBS_LOCK:
            proc = JOBS.get(kind)
            if proc:
                detail["pid"] = proc.pid
                detail["returncode"] = proc.poll()
                detail["status"] = "running" if proc.poll() is None else ("done" if proc.returncode == 0 else "error")
            else:
                state_path = state_dir / f"{name}.json"
                state = read_json(state_path, {})
                detail["state"] = state
                detail["status"] = state.get("status", "not_started")
                detail["pid"] = state.get("pid")
                detail["returncode"] = state.get("metrics", {}).get("exit_code")
        # Reporter progress for enrichment
        if name == "enrichment":
            progress = read_json(REPORTER_PROGRESS, {})
            if progress:
                detail["state"] = progress
                detail["stage"] = progress.get("stage", "")
                detail["percent"] = progress.get("percent")
                detail["timestamp"] = progress.get("updated_at")
        log_file = log_map.get(name, f"{name}.log")
        if log_file:
            log_candidate = log_dir / log_file
            if log_candidate.exists():
                detail["log_path"] = str(log_candidate)
                detail["log"] = _log_tail(log_candidate)
            else:
                detail["log"] = []
        else:
            detail["log"] = []
    # Normalize metrics + errors for easy rendering
    state = detail.get("state") or {}
    detail["metrics"] = state.get("metrics", {}) if isinstance(state, dict) else {}
    detail["errors"] = state.get("errors", []) if isinstance(state, dict) else []
    if isinstance(state, dict) and "stage" in state and "stage" not in detail:
        detail["stage"] = state["stage"]
        detail["percent"] = state.get("percent")
    if "timestamp" not in detail or not detail.get("timestamp"):
        detail["timestamp"] = state.get("timestamp") if isinstance(state, dict) else None
    return detail


def report_review(report_id: str | None) -> dict[str, Any] | None:
    if not report_id:
        return None
    with dashboard_connection() as connection:
        row = connection.execute("SELECT report_id, status, decided_by, note, decided_at FROM report_reviews WHERE report_id=?", (report_id,)).fetchone()
    return dict(zip(["report_id", "status", "decided_by", "note", "decided_at"], row)) if row else None


def delete_report(report_path: str) -> dict[str, Any]:
    """Delete a report and all associated data: corpus, indices, vector store, reporter DB.

    Removes the entire report directory which contains:
    - report.json
    - corpus/ (document_store.db, bm25_index.db, tantivy/, vector/lancedb/)
    - reporter.db (curation decisions)
    """
    candidate = Path(report_path).expanduser().resolve()
    report_dir = candidate.parent
    # Security: must be inside REPORTER_ROOT
    if REPORTER_ROOT.resolve() not in report_dir.parents and REPORTER_ROOT.resolve() != report_dir:
        raise ValueError("Report path outside reporter root")
    # Read report_id before deleting (if report.json still exists)
    report_id = ""
    try:
        report = read_json(candidate, {})
        report_id = report.get("report_id", "")
    except Exception:
        pass
    if not report_dir.exists():
        return {"ok": True, "deleted": False, "reason": "already gone"}
    import shutil
    # Delete everything in the report directory (corpus, reporter.db, report.json, etc.)
    cleaned = []
    for item in report_dir.iterdir():
        shutil.rmtree(item, ignore_errors=True)
        cleaned.append(item.name)
    # Also remove any review status
    if report_id:
        with dashboard_connection() as conn:
            conn.execute("DELETE FROM report_reviews WHERE report_id=?", (report_id,))
            conn.commit()
    # Clean pipeline progress files
    for f in [PIPELINE_PROGRESS, REPORTER_PROGRESS]:
        try: f.unlink()
        except Exception: pass
    return {"ok": True, "deleted": True, "path": str(report_dir), "cleaned": cleaned}


def promote_report_to_main(report_path: str) -> dict[str, Any]:
    """Promote a report's documents and indices to the main corpus.

    When a report is approved:
    1. Copy documents from reporter corpus → main corpus (document_store.db)
    2. Copy chunks and index them in main BM25
    3. Copy vector embeddings to main LanceDB
    4. Mark the report as approved in the review table
    """
    candidate = Path(report_path).expanduser().resolve()
    if not candidate.name == "report.json":
        raise ValueError("Path must point to report.json")
    report_dir = candidate.parent
    if REPORTER_ROOT.resolve() not in report_dir.parents and REPORTER_ROOT.resolve() != report_dir:
        raise ValueError("Report path outside reporter root")

    reporter_corpus = report_dir / "corpus"
    reporter_store_db = reporter_corpus / "document_store.db"
    reporter_lancedb = reporter_corpus / "vector" / "lancedb"

    main_store_db = MAIN_CORPUS / "document_store.db"
    main_bm25_db = MAIN_CORPUS / "bm25_index.db"
    main_lancedb = MAIN_CORPUS / "vector" / "lancedb"

    promoted_docs = 0
    promoted_chunks = 0
    promoted_vectors = 0

    # 1. Copy documents and chunks from reporter → main using FastPathRunner
    if reporter_store_db.exists():
        import sqlite3 as sql3
        from ipa import DocumentStore, BM25Index
        # Open both stores
        main_store = DocumentStore(main_store_db)
        main_bm25 = BM25Index(main_bm25_db)
        reporter_conn = sql3.connect(str(reporter_store_db))
        try:
            # Copy documents
            docs = reporter_conn.execute(
                "SELECT document_id, artifact_id, parser_id, mime_type, pages, text, "
                "elements_json, spans_json, stored_at FROM documents WHERE tombstoned=0"
            ).fetchall()
            for doc_row in docs:
                doc_id = doc_row[0]
                # Check if already exists in main
                existing = main_store._conn.execute(
                    "SELECT 1 FROM documents WHERE document_id=?", (doc_id,)
                ).fetchone()
                if existing:
                    continue
                # Insert document
                main_store._conn.execute(
                    "INSERT OR REPLACE INTO documents "
                    "(document_id, artifact_id, parser_id, mime_type, pages, text, "
                    "elements_json, spans_json, stored_at, tombstoned) "
                    "VALUES (?,?,?,?,?,?,?,?,?,0)",
                    doc_row
                )
                promoted_docs += 1

                # Copy chunks for this document
                chunks = reporter_conn.execute(
                    "SELECT chunk_id, document_id, content_hash, text, metadata_json, span_json "
                    "FROM chunks WHERE document_id=? AND tombstoned=0",
                    (doc_id,)
                ).fetchall()
                from ipa import DocumentChunk
                from ipa.contracts import SourceSpan
                chunk_objs = []
                for ch_row in chunks:
                    chunk = DocumentChunk(
                        chunk_id=ch_row[0], document_id=ch_row[1], content_hash=ch_row[2],
                        text=ch_row[3], metadata=json.loads(ch_row[4]),
                        source_span=None,
                    )
                    chunk_objs.append(chunk)
                if chunk_objs:
                    main_store.put_chunks(chunk_objs)
                    main_bm25.add_chunks(chunk_objs)
                    promoted_chunks += len(chunk_objs)
            main_store.commit()
        finally:
            reporter_conn.close()
            main_store.close()
            main_bm25.close()

    # 2. Copy LanceDB vectors from reporter → main
    if reporter_lancedb.exists() and main_lancedb.parent.exists():
        import shutil
        main_lancedb.parent.mkdir(parents=True, exist_ok=True)
        # Merge: open both LanceDB and copy records
        try:
            from ipa.lancedb_index import LanceDBIndex
            main_lance = LanceDBIndex(main_lancedb, vector_dim=1024)
            reporter_lance = LanceDBIndex(reporter_lancedb, vector_dim=1024)
            # Get all records from reporter LanceDB
            if reporter_lance._table is not None:
                existing_ids = set()
                if main_lance._table is not None:
                    try:
                        import pyarrow as pa
                        tbl = main_lance._table.to_arrow()
                        existing_ids = set(tbl.column("chunk_id").to_pylist())
                    except Exception:
                        pass
                # Read all from reporter
                import pyarrow as pa
                reporter_tbl = reporter_lance._table.to_arrow()
                new_rows = []
                for i in range(reporter_tbl.num_rows):
                    chunk_id = reporter_tbl.column("chunk_id")[i].as_py()
                    if chunk_id not in existing_ids:
                        new_rows.append({
                            "chunk_id": chunk_id,
                            "document_id": reporter_tbl.column("document_id")[i].as_py(),
                            "content_hash": reporter_tbl.column("content_hash")[i].as_py(),
                            "text": reporter_tbl.column("text")[i].as_py(),
                            "vector": reporter_tbl.column("vector")[i].as_py(),
                            "span_json": reporter_tbl.column("span_json")[i].as_py(),
                            "sparse_json": reporter_tbl.column("sparse_json")[i].as_py(),
                        })
                if new_rows:
                    # Convert to pyarrow Table and add
                    new_tbl = pa.Table.from_pylist(new_rows, schema=reporter_tbl.schema)
                    # Ensure table exists in main corpus before adding
                    main_lance._ensure_table(new_rows[0]["vector"])
                    main_lance._table.add(new_tbl)
                    promoted_vectors = len(new_rows)
            main_lance.close()
            reporter_lance.close()
        except Exception as exc:
            # LanceDB merge is best-effort but log the error
            print(f"  [promote] LanceDB merge failed: {exc}", flush=True)

    # 3. Move promoted source files from Landing/web to Archive/web
    archived_files = 0
    archive_errors = []
    reporter_db = report_dir / "reporter.db"
    if reporter_db.exists():
        import sqlite3 as sql3_meta
        meta_conn = sql3_meta.connect(str(reporter_db))
        try:
            # Get original_path of all promoted documents (approved decisions)
            meta_rows = meta_conn.execute(
                "SELECT document_id, original_path FROM document_metadata"
            ).fetchall()
            landing_web = ROOT / "Landing" / "web"
            archive_web = ROOT / "Archive" / "web"
            for doc_id, orig_path in meta_rows:
                if not orig_path:
                    continue
                src = Path(orig_path)
                if not src.is_absolute():
                    src = ROOT / src
                # Only move files that are under Landing/web
                try:
                    src_resolved = src.resolve()
                    landing_resolved = landing_web.resolve()
                    if landing_resolved not in src_resolved.parents and src_resolved != landing_resolved:
                        continue
                    if not src.exists():
                        continue
                    # Preserve relative structure: Landing/web/<site>/<file> → Archive/web/<site>/<file>
                    rel = src_resolved.relative_to(landing_resolved)
                    dst = archive_web / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        # Already archived, just remove from Landing
                        src.unlink()
                        archived_files += 1
                    else:
                        src.rename(dst)
                        archived_files += 1
                except (OSError, ValueError) as exc:
                    archive_errors.append(f"{Path(orig_path).name}: {exc}")
        finally:
            meta_conn.close()

    # 4. Mark report as approved
    report = read_json(candidate, {})
    report_id = report.get("report_id", "")
    if report_id:
        with dashboard_connection() as conn:
            conn.execute("INSERT OR REPLACE INTO report_reviews VALUES (?,?,?,?,?)",
                         (report_id, "approved", "web-user", f"promoted to main corpus, {archived_files} files archived", now()))
            conn.commit()

    return {
        "ok": True,
        "promoted_docs": promoted_docs,
        "promoted_chunks": promoted_chunks,
        "promoted_vectors": promoted_vectors,
        "archived_files": archived_files,
        "archive_errors": archive_errors,
    }


def update_latest_topic(body: dict[str, Any]) -> dict[str, Any]:
    """Apply a constrained human edit to the newest valid report and its store."""
    report = latest_report()
    if not report:
        raise FileNotFoundError("no hay reporte disponible")
    category_id = str(body.get("category_id", ""))
    if not category_id or len(category_id) > 200:
        raise ValueError("category_id inválido")
    allowed = {"label", "description"}
    updates = {key: body[key] for key in allowed if key in body}
    if "label" in updates and (not isinstance(updates["label"], str) or not 1 <= len(updates["label"]) <= 200):
        raise ValueError("label inválido")
    if "description" in updates and (not isinstance(updates["description"], str) or not 1 <= len(updates["description"]) <= 2000):
        raise ValueError("description inválida")
    if "status" in body:
        status = str(body["status"])
        if status not in {"draft", "reviewed", "published"}:
            raise ValueError("status inválido")
        report["status"] = status
    if not updates and "status" not in body:
        raise ValueError("no hay campos editables")
    category = next((item for item in report["categories"] if item.get("category_id") == category_id), None)
    if category is None:
        raise FileNotFoundError(category_id)
    category.update(updates)
    report.pop("path", None); report.pop("updated_at", None)
    report_path = Path(latest_report()["path"])
    temp = report_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(report_path)
    db = report_path.parent / "reporter.db"
    if db.exists():
        from ipa.reporter_store import ReporterStore
        with ReporterStore(db) as store:
            if updates and not store.update_topic(category_id, report["report_id"], updates):
                raise FileNotFoundError(category_id)
            store.commit()
    return {"ok": True, "report_id": report["report_id"], "category": category, "status": report["status"]}


def dashboard_state() -> dict[str, Any]:
    reporter = latest_report()
    sources = load_sources()
    base = base_sources()
    # Merge: base sources with overrides from added
    disabled = set(sources["disabled"])
    added_map = {item.get("url"): item for item in sources["added"]}
    effective_sources = []
    for site in base:
        url = site.get("url")
        if url in disabled:
            effective_sources.append({**site, "active": False})
        elif url in added_map:
            merged = dict(site)
            merged["days_back"] = added_map[url].get("days_back", site.get("days_back", 2))
            merged["active"] = True
            effective_sources.append(merged)
        else:
            effective_sources.append({**site, "active": True})
    # Add non-base added sources
    base_urls = {s.get("url") for s in base}
    for item in sources["added"]:
        if item.get("url") not in base_urls and item.get("url") not in disabled:
            effective_sources.append({**item, "active": True})
    # During pipeline, the fast path indexes to the reporter's corpus.
    # Return both: the real main corpus AND the active indexing corpus.
    reporter_output = active_reporter_output()
    reporter_corpus = reporter_output / "corpus"
    # Skip db_counts for reporter corpus while cleaning is in progress (Windows file locks)
    if _CLEANING_IN_PROGRESS:
        staging_ingestion = {"exists": False}
        staging_vector = {"exists": False}
        reporter_ingestion = {"exists": False}
    else:
        staging_ingestion = db_counts(reporter_corpus / "document_store.db")
        staging_vector = db_counts(reporter_corpus / "vector" / "lancedb")
        reporter_ingestion = db_counts(reporter and Path(reporter["path"]).parent / "corpus" / "document_store.db" if reporter else REPORTER_ROOT / "missing.db")
    return {
        "timestamp": now(),
        "llm": llm_status(),
        "main_ingestion": db_counts(MAIN_CORPUS / "document_store.db"),
        "main_vector": db_counts(MAIN_CORPUS / "vector" / "lancedb"),
        "staging_ingestion": staging_ingestion,
        "staging_vector": staging_vector,
        "reporter_ingestion": reporter_ingestion,
        "reporter": {"available": reporter is not None, "report": reporter, "history": report_history(), "review": report_review(reporter.get("report_id") if reporter else None)},
        "processes": process_status(),
        "pipeline": read_json(PIPELINE_PROGRESS, {"stage": "idle", "status": "idle", "percent": 0}),
        "reporter_progress": read_json(REPORTER_PROGRESS, {}),
        "scrape_counts": scrape_counts(),
        "archive_counts": archive_counts(),
        "sources": {**sources, "base": base, "effective": effective_sources},
    }


def list_documents(corpus: str, limit: int = 100) -> list[dict[str, Any]]:
    if corpus == "reporter":
        report = latest_report()
        db = Path(report["path"]).parent / "reporter.db" if report else None
    else:
        db = MAIN_CORPUS / "document_store.db"
    if not db or not db.exists():
        return []
    try:
        with sqlite3.connect(str(db)) as conn:
            if corpus == "reporter":
                rows = conn.execute("SELECT document_id, artifact_id, original_path, source_url, canonical_url, source_domain, title, published_at, published_at_confidence, quality_score, mime_type FROM document_metadata ORDER BY published_at DESC LIMIT ?", (min(limit, 500),)).fetchall()
                keys = ["document_id", "artifact_id", "original_path", "source_url", "canonical_url", "source_domain", "title", "published_at", "published_at_confidence", "quality_score", "mime_type"]
            else:
                rows = conn.execute("SELECT document_id, artifact_id, NULL, NULL, NULL, NULL, document_id, stored_at, NULL, NULL, mime_type FROM documents WHERE tombstoned=0 ORDER BY stored_at DESC LIMIT ?", (min(limit, 500),)).fetchall()
                keys = ["document_id", "artifact_id", "original_path", "source_url", "canonical_url", "source_domain", "title", "published_at", "published_at_confidence", "quality_score", "mime_type"]
                landing = MAIN_CORPUS / "landing.db"
                source_map = {}
                if landing.exists():
                    with sqlite3.connect(str(landing)) as landing_conn:
                        source_map = {row[0]: row[1] for row in landing_conn.execute("SELECT artifact_id, source_uri FROM artifacts")}
                rows = [tuple(source_map.get(row[1]) if index == 2 else row[index] for index in range(len(row))) for row in rows]
            return [dict(zip(keys, row)) for row in rows]
    except sqlite3.Error:
        return []


def read_document(path_value: str) -> tuple[Path, bytes]:
    candidate = Path(path_value).expanduser().resolve()
    allowed_roots = [ROOT / "Landing", ROOT / "Archive", ROOT / "outputs" / "reporter", ROOT / "outputs" / "experiments"]
    if not any(candidate == root.resolve() or root.resolve() in candidate.parents for root in allowed_roots):
        raise PermissionError("La ruta no pertenece a una zona documental permitida")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate, candidate.read_bytes()


def _find_running_processes(pattern: str) -> list[int]:
    """Find python processes whose command line matches a pattern."""
    try:
        result = subprocess.run(
            ["wmic", "process", "where", "Name='python.exe' or Name='pythonw.exe'", "get", "ProcessId,CommandLine", "/FORMAT:CSV"],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS if os.name == "nt" else 0,
        )
        pids = []
        for line in result.stdout.strip().splitlines():
            if pattern in line:
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        pids.append(int(parts[-1]))
                    except ValueError:
                        continue
        return pids
    except Exception:
        return []


def _job_lock_path(kind: str) -> Path:
    return ROOT / "outputs" / "web_dashboard" / f"{kind}.lock"


def _acquire_job_lock(kind: str, pid: int) -> Path:
    lock_path = _job_lock_path(kind)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_path.write_text(str(pid), encoding="utf-8", errors="strict")
    except OSError:
        pass
    # Atomic create: simultaneous dashboard instances cannot both acquire it.
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(pid))
        return lock_path
    except FileExistsError:
        try:
            owner = int(lock_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            owner = 0
        if owner and _process_exists(owner):
            raise RuntimeError(f"Hay un proceso {kind} corriendo (PID {owner}). Esperá a que termine.")
        try:
            lock_path.unlink()
        except OSError:
            raise RuntimeError(f"No se pudo liberar el lock del proceso {kind}")
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(pid))
        return lock_path


def _process_exists(pid: int) -> bool:
    try:
        no_window = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        return subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, timeout=3, creationflags=no_window).returncode == 0 and str(pid) in subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, timeout=3, creationflags=no_window).stdout
    except Exception:
        return False


def _kill_orphan_processes(pattern: str) -> int:
    """Kill ALL python processes matching pattern. Returns count killed."""
    pids = _find_running_processes(pattern)
    killed = 0
    for pid in pids:
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=5,
                          creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS if os.name == "nt" else 0)
            killed += 1
        except Exception:
            pass
    if killed:
        import time as _t
        _t.sleep(1)  # Give OS time to release resources
    return killed


def spawn_job(kind: str, command: list[str]) -> dict[str, Any]:
    with JOBS_LOCK:
        existing = JOBS.get(kind)
        if existing and existing.poll() is None:
            raise RuntimeError(f"El job {kind} ya está ejecutándose")
        # Kill any orphan processes for this job type before spawning
        orphan_patterns = {
            "scraper": "run_web_scrape.py",
            "pipeline": "run_fast_path.py",
            "lancedb": "run_fast_path.py",  # lancedb reindex uses fast_path
            "reporter": "run_reporter.py",
            "reporter_fast": "run_reporter.py",
        }
        pattern = orphan_patterns.get(kind)
        if pattern:
            killed = _kill_orphan_processes(pattern)
            if killed:
                print(f"  [spawn] killed {killed} orphan(s) for {kind}", flush=True)
        # Clean stale lock
        lock_path = _job_lock_path(kind)
        if lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass
        log_dir = ROOT / "outputs" / "web_dashboard" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = open(log_dir / f"{kind}.log", "a", encoding="utf-8")
        try:
            proc = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": "src", "PYTHONIOENCODING": "utf-8"}, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except Exception:
            log.close()
            lock_path.unlink(missing_ok=True)
            raise
        lock_path.write_text(str(proc.pid), encoding="utf-8")
        JOBS[kind] = proc
        def release_when_done() -> None:
            proc.wait()
            log.close()
            try:
                if lock_path.read_text(encoding="utf-8").strip() == str(proc.pid):
                    lock_path.unlink()
            except OSError:
                pass
        threading.Thread(target=release_when_done, daemon=True).start()
        return {"kind": kind, "pid": proc.pid, "status": "running"}


def _write_pipeline_progress(stage: str, status: str, percent: int, detail: str = "") -> None:
    PIPELINE_PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PIPELINE_PROGRESS.write_text(json.dumps({
        "stage": stage, "status": status, "percent": percent,
        "detail": detail, "updated_at": now(),
    }, ensure_ascii=False), encoding="utf-8")


def run_full_pipeline(period_start: str = "", period_end: str = "", period_mode: str = "days", days_back: int = 0) -> None:
    """Run scraper + fast path in parallel, then re-run fast path + both reporters in parallel.

    Stage 1: Scraper + Fast Path (parallel)
      - Scraper downloads files to Landing/web
      - Fast Path indexes whatever is already in Landing/web (BM25 + LanceDB)
        (idempotent — no duplicates on re-run)
    Stage 2: Fast Path (re-run) + Fast Reporter + Full Reporter (parallel)
      - Fast Path picks up new files from the scraper
      - Fast Reporter generates quick report (no embeddings/LLM)
      - Full Reporter generates complete report (BGE-M3 + Qwen)

    period_mode:
      - 'days': use days_back to compute period (today - days_back → today)
      - 'range': use period_start/period_end directly
    """
    try:
        log_dir = ROOT / "outputs" / "web_dashboard" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        no_window = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        env = {**os.environ, "PYTHONPATH": "src", "PYTHONIOENCODING": "utf-8"}

        # Compute period from mode
        if period_mode == "days" and days_back > 0:
            from datetime import datetime, timedelta, timezone
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=days_back)
            period_start = start_dt.strftime("%Y-%m-%dT00:00:00Z")
            period_end = end_dt.strftime("%Y-%m-%dT23:59:59Z")
            print(f"  [pipeline] period from days_back={days_back}: {period_start} → {period_end}", flush=True)
        elif period_mode == "range" and period_start and period_end:
            print(f"  [pipeline] period from range: {period_start} → {period_end}", flush=True)

        # Compute dynamic output directory based on period label
        # e.g. period_end="2026-09-30T23:59:59Z" → "optimized-llm-2026-09"
        period_label = "unspecified"
        if period_end:
            period_label = period_end[:7]  # YYYY-MM
        reporter_output = REPORTER_ROOT / "quality-check" / f"optimized-llm-{period_label}"
        reporter_corpus = reporter_output / "corpus"
        global _ACTIVE_REPORTER_OUTPUT
        _ACTIVE_REPORTER_OUTPUT = reporter_output
        print(f"  [pipeline] reporter output: {reporter_output}", flush=True)

        # Kill any orphan processes from previous runs before starting
        for pattern in ["run_web_scrape.py", "run_fast_path.py", "run_reporter.py"]:
            killed = _kill_orphan_processes(pattern)
            if killed:
                print(f"  [pipeline] killed {killed} orphan(s) matching {pattern}", flush=True)
        # Clean stale locks
        for kind in ["scraper", "pipeline", "lancedb", "reporter", "reporter_fast"]:
            try:
                _job_lock_path(kind).unlink(missing_ok=True)
            except Exception:
                pass

        # --- Stage 1: Scraper + Fast Path (watch mode) IN PARALLEL ---
        _write_pipeline_progress("scraper", "running", 0, "Scraper + indexing continuo en paralelo")

        # Scraper
        config = effective_scrape_config()
        scraper_cmd = [VENV_PYTHONW, "-u", "scripts/run_web_scrape.py", "--config", str(config), "--output", "Landing/web", "--engine", "auto", "--no-images", "--no-ocr"]
        scraper_log = open(log_dir / "scraper.log", "a", encoding="utf-8")
        scraper_proc = subprocess.Popen(scraper_cmd, cwd=str(ROOT), stdout=scraper_log, stderr=subprocess.STDOUT, env=env, creationflags=no_window)

        # Fast Path in watch mode: indexes to the REPORTER's corpus directly
        # The reporter reuses this corpus instead of re-ingesting
        # Only promoted to main corpus when user approves the report
        # reporter_output and reporter_corpus already set above (dynamic)
        fast_path_cmd = [VENV_PYTHONW, "-u", "scripts/run_fast_path.py", "--input", "Landing/web", "--output", str(reporter_corpus), "--watch", "10"]
        fp_log = open(log_dir / "fast_path.log", "a", encoding="utf-8")
        fp_proc = subprocess.Popen(fast_path_cmd, cwd=str(ROOT), stdout=fp_log, stderr=subprocess.STDOUT, env=env, creationflags=no_window)

        with JOBS_LOCK:
            JOBS["scraper"] = scraper_proc
            JOBS["pipeline"] = fp_proc

        # Wait for scraper to finish (fast path keeps running in watch mode)
        scraper_proc.wait()
        scraper_log.close()
        if scraper_proc.returncode != 0:
            _write_pipeline_progress("scraper", "failed", 100, f"Scraper falló (exit {scraper_proc.returncode})")
            fp_proc.terminate()
            fp_log.close()
            return
        _write_pipeline_progress("scraper", "done", 30, "Scraper completado — esperando indexing final (BM25 + LanceDB)")

        # Wait for fast path watch to catch up: poll until BM25 and LanceDB
        # counts match (meaning all files have been ingested AND embedded).
        # This runs AFTER the scraper has finished, so we just need to wait
        # for the fast path to process the last batch of files.
        import time as _time
        _write_pipeline_progress("fast_path", "running", 35, "Esperando indexing final (BM25 + LanceDB)")
        max_wait = 300  # Max 5 minutes — covers slow embedding of large corpora
        waited = 0
        while waited < max_wait:
            _time.sleep(10)
            waited += 10
            # Check if BM25 and LanceDB counts match in the REPORTER corpus
            try:
                import sqlite3 as _sql
                with _sql.connect(str(reporter_corpus / "document_store.db")) as conn:
                    bm25_docs = conn.execute("SELECT COUNT(*) FROM documents WHERE tombstoned=0").fetchone()[0]
                    bm25_chunks = conn.execute("SELECT COUNT(*) FROM chunks WHERE tombstoned=0").fetchone()[0]
                lance_chunks = 0
                try:
                    import lancedb as _ldb
                    ldb = _ldb.connect(str(reporter_corpus / "vector" / "lancedb"))
                    tables = ldb.table_names() if hasattr(ldb, "table_names") else []
                    if "chunks" in tables:
                        lance_chunks = ldb.open_table("chunks").count_rows()
                except Exception:
                    pass
                scraped = scrape_counts()["total_files"]
                _write_pipeline_progress("fast_path", "running", 35,
                    f"Indexing: {bm25_chunks} BM25 / {lance_chunks} LanceDB / {scraped} scraped · {waited}s")
                # Done when LanceDB caught up to BM25 AND BM25 docs >= scraped
                if lance_chunks >= bm25_chunks and bm25_docs >= scraped and lance_chunks > 0:
                    break
            except Exception:
                pass

        # Final wait: even if timeout, give the fast path one more iteration
        # to finish embedding any remaining chunks
        _time.sleep(5)

        fp_proc.terminate()
        try:
            fp_proc.wait(timeout=15)
        except Exception:
            fp_proc.kill()
        fp_log.close()
        _write_pipeline_progress("fast_path", "done", 40, "Indexing completado (BM25 + LanceDB) — lanzando reporter")

        # --- Stage 2: Full Reporter (BGE-M3 LanceDB + Qwen LLM) ---
        _write_pipeline_progress("parallel", "running", 40, "Generando reporte (BGE-M3 + Qwen LLM)")

        full_output = reporter_output  # same as the fast path corpus
        full_cmd = [VENV_PYTHONW, "-u", "scripts/run_reporter.py", "--input", "Landing/web", "--config", "configs/reporter.yaml", "--output", str(full_output), "--embeddings", "--llm"]
        if period_start:
            full_cmd += ["--period-start", period_start]
        if period_end:
            full_cmd += ["--period-end", period_end]
            # Auto-generate label from end date (YYYY-MM)
            try:
                label = period_end[:7]  # YYYY-MM from ISO date
                full_cmd += ["--period-label", label]
            except Exception:
                pass
        full_log = open(log_dir / "reporter.log", "a", encoding="utf-8")
        full_proc = subprocess.Popen(full_cmd, cwd=str(ROOT), stdout=full_log, stderr=subprocess.STDOUT, env=env, creationflags=no_window)

        with JOBS_LOCK:
            JOBS["reporter"] = full_proc

        # Wait for full reporter
        full_proc.wait()
        full_log.close()
        full_ok = full_proc.returncode == 0

        # Report results
        if full_ok:
            _write_pipeline_progress("reporter_full", "done", 100, "Pipeline completado: Reporter OK (BGE-M3 + Qwen)")
        else:
            _write_pipeline_progress("error", "failed", 100, f"Reporter falló (exit {full_proc.returncode})")
    except Exception as exc:
        _write_pipeline_progress("error", "failed", 100, str(exc))


class Handler(BaseHTTPRequestHandler):
    server_version = "IPAWeb/1.0"

    def log_message(self, format: str, *args: object) -> None:
        return

    def send_json(self, value: Any, status: int = 200) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("payload demasiado grande")
        return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.serve_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
        elif parsed.path == "/api/health":
            self.send_json({"ok": True, "timestamp": now()})
        elif parsed.path.startswith("/static/"):
            relative = Path(parsed.path.removeprefix("/static/")).name
            self.serve_file(STATIC_ROOT / relative, mimetypes.guess_type(relative)[0] or "application/octet-stream")
        elif parsed.path == "/api/state":
            self.send_json(dashboard_state())
        elif parsed.path == "/api/topic":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                self.send_json(topic_details(query.get("category_id", [""])[0]))
            except Exception as exc:
                self.send_json({"error": str(exc)}, 404)
        elif parsed.path == "/api/report":
            query = urllib.parse.parse_qs(parsed.query)
            path_str = query.get("path", [""])[0]
            report = load_report_by_path(path_str) if path_str else latest_report()
            if not report:
                self.send_json({"error": "no hay reporte disponible"}, 404)
            else:
                # Enrich with curation summary, doc count, markdown path
                enriched = {
                    "report": report,
                    "curation_summary": report.get("curation_summary", {}),
                    "total_documents": sum((c.get("document_count", 0) for c in report.get("categories", [])), 0),
                    "source_refs_count": len(report.get("source_refs", [])),
                    "uncertainties_count": len(report.get("uncertainties", [])),
                    "recommended_readings_count": len(report.get("recommended_readings", [])),
                    "markdown_path": str(Path(report["path"]).with_suffix(".md")) if Path(report["path"]).with_suffix(".md").exists() else None,
                    "review": report_review(report.get("report_id")),
                }
                self.send_json(enriched)
        elif parsed.path == "/api/deep-dive":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                from ipa.reporter_deep_dive import deep_dive
                corpus = Path(query.get("corpus", [""])[0]).expanduser().resolve()
                if REPORTER_ROOT.resolve() not in corpus.parents:
                    raise PermissionError("deep dive solo puede usar corpus Reporter")
                user_query = query.get("query", [""])[0]
                retrieval_query = query.get("search", [user_query])[0]
                document_ids = query.get("document_id", [])
                agentic = query.get("agentic", [os.environ.get("IPA_AGENTIC_DEEP_DIVE", "0")])[0].lower() in {"1", "true", "yes"}
                category_id = query.get("category_id", [None])[0]
                # Fetch curation reasons for the topic's documents to enrich the LLM context
                doc_reasons = {}
                if category_id:
                    try:
                        topic_data = topic_details(category_id)
                        for doc in topic_data.get("documents", []):
                            if doc.get("reason"):
                                doc_reasons[doc["document_id"]] = doc["reason"]
                    except Exception:
                        pass
                result = deep_dive(corpus, user_query, int(query.get("top_k", [5])[0]), provider=get_deep_dive_provider(), retrieval_query=retrieval_query, document_ids=document_ids, agentic=agentic, report_id=query.get("report_id", [None])[0], category_id=category_id, doc_reasons=doc_reasons or None)
                self.send_json(result)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)
        elif parsed.path == "/api/deep-dive/stream":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                from ipa.reporter_deep_dive import deep_dive_prepare, deep_dive_stream, _clean_token
                corpus = Path(query.get("corpus", [""])[0]).expanduser().resolve()
                if REPORTER_ROOT.resolve() not in corpus.parents:
                    raise PermissionError("deep dive solo puede usar corpus Reporter")
                user_query = query.get("query", [""])[0]
                retrieval_query = query.get("search", [user_query])[0]
                document_ids = query.get("document_id", [])
                category_id = query.get("category_id", [None])[0]
                conversation_history = []
                try:
                    conversation_history = json.loads(query.get("history", ["[]"])[0])
                except (TypeError, ValueError, json.JSONDecodeError):
                    conversation_history = []
                doc_reasons = {}
                if category_id:
                    try:
                        topic_data = topic_details(category_id)
                        for doc in topic_data.get("documents", []):
                            if doc.get("reason"):
                                doc_reasons[doc["document_id"]] = doc["reason"]
                    except Exception:
                        pass
                provider = get_deep_dive_provider()
                # Prepare retrieval (non-streaming) to get evidence + claims
                prep = deep_dive_prepare(corpus, user_query, int(query.get("top_k", [5])[0]), retrieval_query=retrieval_query, document_ids=document_ids, agentic=True, report_id=query.get("report_id", [None])[0], category_id=category_id, doc_reasons=doc_reasons or None, conversation_history=conversation_history)
                # Send SSE headers
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                # Send evidence and metadata first
                import json as _json
                def sse_send(event, data):
                    payload = _json.dumps(data, ensure_ascii=False)
                    self.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                sse_send("evidence", {"evidence": prep["evidence"], "chunks": prep["chunks_info"], "sufficient": prep["sufficient_evidence"]})
                # Stream the answer
                if provider is not None and prep["chunks"]:
                    full_answer = ""
                    for chunk in deep_dive_stream(provider, prep["messages"], max_new_tokens=768):
                        text = _clean_token(chunk.get("text", ""))
                        if text:
                            full_answer += text
                            sse_send("token", {"text": text})
                        if chunk.get("done"):
                            break
                    # Clean the final answer
                    from ipa.reporter_deep_dive import _clean_generated
                    cleaned = _clean_generated(full_answer)
                    if not cleaned:
                        cleaned = prep["fallback"]
                    # Validate claims
                    from ipa.reporter_claims import validate_claims, citation_summary
                    claims = validate_claims(cleaned, prep["evidence_texts"])
                    sse_send("done", {"answer": cleaned, "claims": claims, "citation_summary": citation_summary(claims)})
                else:
                    sse_send("done", {"answer": prep["fallback"], "claims": [], "citation_summary": {}})
            except Exception as exc:
                try:
                    payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
                    self.wfile.write(f"event: error\ndata: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    pass
        elif parsed.path == "/api/process":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                self.send_json(process_detail(query.get("name", [""])[0]))
            except (ValueError, FileNotFoundError) as exc:
                self.send_json({"error": str(exc)}, 400 if isinstance(exc, ValueError) else 404)
        elif parsed.path == "/api/documents":
            query = urllib.parse.parse_qs(parsed.query)
            self.send_json(list_documents(query.get("corpus", ["reporter"])[0], int(query.get("limit", [100])[0])))
        elif parsed.path == "/api/document":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                path, content = read_document(query.get("path", [""])[0])
                self.send_json({"path": str(path), "name": path.name, "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream", "size": len(content), "content_base64": base64.b64encode(content).decode("ascii")})
            except (PermissionError, FileNotFoundError, ValueError) as exc:
                self.send_json({"error": str(exc)}, 403 if isinstance(exc, PermissionError) else 404)
        elif parsed.path == "/api/chunk/view":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                chunk_id = query.get("chunk_id", [""])[0]
                corpus = Path(query.get("corpus", [""])[0]).expanduser().resolve()
                if REPORTER_ROOT.resolve() not in corpus.parents and corpus != REPORTER_ROOT.resolve():
                    raise PermissionError("solo puede acceder al corpus Reporter")
                from ipa import DocumentStore
                with DocumentStore(corpus / "document_store.db") as store:
                    chunk = store.get_chunk(chunk_id)
                    if not chunk:
                        raise FileNotFoundError("chunk no encontrado")
                    doc = store.get_document(chunk.document_id)
                self.send_json({
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "chunk_index": getattr(chunk, "chunk_index", None),
                    "document": {
                        "title": doc.title if doc else None,
                        "original_path": doc.original_path if doc else None,
                        "source_domain": getattr(doc, "source_domain", None) if doc else None,
                        "published_at": getattr(doc, "published_at", None) if doc else None,
                    } if doc else None,
                })
            except (PermissionError, FileNotFoundError, ValueError) as exc:
                self.send_json({"error": str(exc)}, 403 if isinstance(exc, PermissionError) else 404)
        elif parsed.path == "/api/document/raw":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                path, content = read_document(query.get("path", [""])[0])
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                self.send_header("Content-Disposition", f"inline; filename*=UTF-8''{urllib.parse.quote(path.name)}")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except (PermissionError, FileNotFoundError, ValueError) as exc:
                self.send_json({"error": str(exc)}, 403 if isinstance(exc, PermissionError) else 404)
        elif parsed.path == "/api/decisions":
            report = latest_report()
            decisions = []
            if report:
                db = Path(report["path"]).parent / "reporter.db"
                try:
                    with sqlite3.connect(str(db)) as conn:
                        rows = conn.execute("SELECT decision_id, document_id, payload_json, created_at FROM document_decisions ORDER BY created_at DESC LIMIT 500").fetchall()
                        for decision_id, document_id, payload, created_at in rows:
                            item = json.loads(payload)
                            metadata_row = conn.execute("SELECT original_path, title FROM document_metadata WHERE document_id=?", (document_id,)).fetchone()
                            item["decision_id"] = decision_id; item["created_at"] = created_at
                            if metadata_row:
                                item["original_path"], item["title"] = metadata_row
                            scores = item.get("scores", {})
                            item["promotion_score"] = round(
                                0.30 * float(scores.get("relevance", 0)) + 0.20 * float(scores.get("novelty", 0)) +
                                0.20 * float(scores.get("source_quality", 0)) + 0.15 * float(scores.get("impact", 0)) +
                                0.10 * float(scores.get("depth", 0)) + 0.05 * float(scores.get("actionability", 0)), 4)
                            decisions.append(item)
                    decisions.sort(key=lambda item: (-float(item.get("promotion_score", 0)), item.get("created_at", "")))
                except (sqlite3.Error, ValueError):
                    pass
            self.send_json(decisions)
        else:
            self.send_json({"error": "not found"}, 404)

    def serve_file(self, path: Path, content_type: str) -> None:
        try:
            content = path.read_bytes()
        except OSError:
            self.send_json({"error": "not found"}, 404); return
        self.send_response(200); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(content))); self.send_header("Cache-Control", "no-store, max-age=0"); self.end_headers(); self.wfile.write(content)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            body = self.read_body()
            if parsed.path == "/api/topics/edit":
                self.send_json(update_latest_topic(body))
            elif parsed.path == "/api/sources":
                action = body.get("action", "add"); data = load_sources()
                # Only validate URL for actions that need it
                if action in ("add", "disable", "enable", "update_days"):
                    url = safe_url(str(body.get("url", "")))
                else:
                    url = str(body.get("url", ""))
                if action == "add":
                    if not any(item.get("url") == url for item in data["added"]): data["added"].append({"url": url, "days_back": int(body.get("days_back", 7)), "max_articles": int(body.get("max_articles", 20)), "delay_seconds": 2.0})
                    data["disabled"] = [item for item in data["disabled"] if item != url]
                elif action == "disable":
                    if url not in data["disabled"]: data["disabled"].append(url)
                elif action == "enable":
                    data["disabled"] = [item for item in data["disabled"] if item != url]
                elif action == "update_days":
                    # Update days_back for a specific source (added or base override)
                    new_days = int(body.get("days_back", 7))
                    if new_days < 0 or new_days > 365:
                        raise ValueError("days_back debe estar entre 0 y 365")
                    # Check if it's in added sources
                    found = False
                    for item in data["added"]:
                        if item.get("url") == url:
                            item["days_back"] = new_days
                            found = True
                            break
                    if not found:
                        # It's a base source — add an override to added list
                        base = base_sources()
                        base_item = next((b for b in base if b.get("url") == url), None)
                        if base_item:
                            data["added"].append({"url": url, "days_back": new_days, "max_articles": base_item.get("max_articles", 20), "delay_seconds": 2.0})
                elif action == "update_days_all":
                    # Update days_back for ALL sources (added + base overrides)
                    new_days = int(body.get("days_back", 7))
                    if new_days < 0 or new_days > 365:
                        raise ValueError("days_back debe estar entre 0 y 365")
                    for item in data["added"]:
                        item["days_back"] = new_days
                    # Also add overrides for base sources not already in added
                    base = base_sources()
                    existing_urls = {item.get("url") for item in data["added"]}
                    for b in base:
                        if b.get("url") not in existing_urls:
                            data["added"].append({"url": b["url"], "days_back": new_days, "max_articles": b.get("max_articles", 20), "delay_seconds": 2.0})
                else: raise ValueError("action debe ser add, disable, enable, update_days o update_days_all")
                save_sources(data); self.send_json({"ok": True, "sources": data})
            elif parsed.path == "/api/scraper/run":
                config = effective_scrape_config(); command = [VENV_PYTHONW, "-u", "scripts/run_web_scrape.py", "--config", str(config), "--output", "Landing/web", "--engine", "auto"]
                self.send_json(spawn_job("scraper", command), 202)
            elif parsed.path == "/api/fastpath/run":
                reporter_corpus = active_reporter_output() / "corpus"
                command = [VENV_PYTHONW, "-u", "scripts/run_fast_path.py", "--input", "Landing/web", "--output", str(reporter_corpus)]
                self.send_json(spawn_job("pipeline", command), 202)
            elif parsed.path == "/api/lancedb/run":
                # Re-index LanceDB from existing document_store.db
                reporter_corpus = active_reporter_output() / "corpus"
                script = ROOT / "scripts" / "run_lancedb_reindex.py"
                if not script.exists():
                    # Inline: just call index_lancedb from run_fast_path
                    command = [VENV_PYTHONW, "-c", f"import sys; sys.path.insert(0,'src'); from scripts.run_fast_path import index_lancedb; from pathlib import Path; r=index_lancedb(Path('{reporter_corpus}/document_store.db'), Path('{reporter_corpus}/vector/lancedb')); print(r)"]
                else:
                    command = [VENV_PYTHONW, "-u", str(script), "--corpus", str(reporter_corpus)]
                self.send_json(spawn_job("lancedb", command), 202)
            elif parsed.path == "/api/reporter/run":
                period = str(body.get("period", "2026-08")); output = REPORTER_ROOT / "web" / period
                command = [VENV_PYTHONW, "-u", "scripts/run_reporter.py", "--input", "Landing/web", "--config", "configs/reporter.yaml", "--output", str(output)]
                if body.get("embeddings"): command.append("--embeddings")
                if body.get("llm"): command.append("--llm")
                self.send_json(spawn_job("reporter", command), 202)
            elif parsed.path == "/api/pipeline/run":
                with PIPELINE_LOCK:
                    if PIPELINE_THREAD is not None and PIPELINE_THREAD.is_alive():
                        raise RuntimeError("El pipeline ya está ejecutándose")
                    period_mode = str(body.get("period_mode", "days"))
                    period_start = str(body.get("period_start", ""))
                    period_end = str(body.get("period_end", ""))
                    days_back = int(body.get("days_back", 0))
                    thread = threading.Thread(target=run_full_pipeline, args=(period_start, period_end, period_mode, days_back), daemon=True)
                    globals()["PIPELINE_THREAD"] = thread
                    thread.start()
                self.send_json({"ok": True, "status": "running", "message": "Pipeline iniciado: scraper → reporte rápido → reporte completo"}, 202)
            elif parsed.path == "/api/reports/review":
                report_path = str(body.get("path", ""))
                status = str(body.get("status", ""))
                if status not in {"approved", "rejected", "changes_requested"}:
                    raise ValueError("revisión de corpus inválida")
                # Use provided path or fall back to latest
                if report_path:
                    report = load_report_by_path(report_path)
                else:
                    report = latest_report()
                if not report:
                    # No report.json found — still clean up the reporter corpus directory
                    if status == "rejected":
                        import shutil
                        reporter_output = active_reporter_output()
                        cleaned = []
                        if reporter_output.exists():
                            for item in reporter_output.iterdir():
                                if item.name != "report.json":  # already gone
                                    shutil.rmtree(item, ignore_errors=True)
                                    cleaned.append(item.name)
                        # Clean progress files
                        for f in [PIPELINE_PROGRESS, REPORTER_PROGRESS]:
                            try: f.unlink()
                            except Exception: pass
                        self.send_json({"ok": True, "status": "rejected", "deleted": True, "cleaned": cleaned, "note": "report.json not found, cleaned orphan corpus"})
                    else:
                        raise ValueError("reporte no encontrado")
                    return
                if status == "approved":
                    # Promote documents + indices to main corpus
                    result = promote_report_to_main(report["path"])
                    self.send_json({"ok": True, "report_id": report.get("report_id", ""), "status": "approved", **result})
                elif status == "rejected":
                    # Delete the report and all associated data
                    result = delete_report(report["path"])
                    self.send_json({"ok": True, "report_id": report.get("report_id", ""), "status": "rejected", **result})
                else:
                    # changes_requested — just record the review
                    with dashboard_connection() as connection:
                        connection.execute("INSERT OR REPLACE INTO report_reviews VALUES (?,?,?,?,?)", (report.get("report_id", ""), status, str(body.get("decided_by", "web-user")), body.get("note"), now()))
                        connection.commit()
                    self.send_json({"ok": True, "report_id": report.get("report_id", ""), "status": status})
            elif parsed.path == "/api/reports/delete":
                report_path = str(body.get("path", ""))
                if not report_path:
                    raise ValueError("path es requerido")
                result = delete_report(report_path)
                self.send_json({"ok": True, **result})
            elif parsed.path == "/api/decisions/review":
                report = latest_report(); decision_id = str(body.get("decision_id", "")); status = str(body.get("status", ""))
                if not report or status not in {"approved", "rejected", "changes_requested"}: raise ValueError("revisión inválida")
                db = Path(report["path"]).parent / "reporter.db"
                approval = {"decision": status, "decided_at": now(), "decided_by": str(body.get("decided_by", "web-user")), "note": body.get("note")}
                with sqlite3.connect(str(db)) as conn:
                    row = conn.execute("SELECT payload_json FROM document_decisions WHERE decision_id=?", (decision_id,)).fetchone()
                    if not row: raise FileNotFoundError(decision_id)
                    payload = json.loads(row[0]); payload["review_status"] = status; payload["approval"] = approval
                    conn.execute("UPDATE document_decisions SET payload_json=? WHERE decision_id=?", (json.dumps(payload, ensure_ascii=False), decision_id)); conn.commit()
                self.send_json({"ok": True, "decision_id": decision_id, "review_status": status})
            elif parsed.path == "/api/restart":
                # Launch watchdog --restart in a detached process, then exit
                self.send_json({"ok": True, "message": "restarting in 1s..."}, 200)
                def _delayed_restart():
                    import time as _time
                    _time.sleep(1)
                    # Always use venv pythonw to avoid spawning system python dashboards
                    venv_pythonw = str(ROOT / ".venv" / "Scripts" / "pythonw.exe")
                    restart_python = venv_pythonw if Path(venv_pythonw).exists() else sys.executable
                    subprocess.Popen(
                        [restart_python, "-u", str(ROOT / "scripts" / "dashboard_watchdog.py"), "--restart",
                         "--host", "127.0.0.1", "--port", "8765"],
                        cwd=str(ROOT),
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                        close_fds=True,
                    )
                threading.Thread(target=_delayed_restart, daemon=True).start()
            elif parsed.path == "/api/process/stop":
                # Stop all running pipeline processes (scraper, fast path, reporter)
                # NEVER touches the main corpus
                stopped = []
                with JOBS_LOCK:
                    for name, proc in list(JOBS.items()):
                        if proc.poll() is None:
                            proc.terminate()
                            try:
                                proc.wait(timeout=5)
                            except Exception:
                                proc.kill()
                            stopped.append(name)
                        del JOBS[name]
                # Also kill any orphan processes
                for pattern in ["run_web_scrape.py", "run_fast_path.py", "run_reporter.py"]:
                    orphans = _find_running_processes(pattern)
                    for pid in orphans:
                        try:
                            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS if os.name == "nt" else 0)
                            stopped.append(f"orphan:{pid}")
                        except Exception:
                            pass
                # Clear pipeline progress
                for f in [PIPELINE_PROGRESS, REPORTER_PROGRESS]:
                    try:
                        f.unlink()
                    except Exception:
                        pass
                self.send_json({"ok": True, "stopped": stopped})
            elif parsed.path == "/api/landing/clean":
                # Clean Landing/web scraped content + reporter corpus DB + scrape history
                web_dir = ROOT / "Landing" / "web"
                cleaned = []
                # Remove all site directories (scraped content)
                if web_dir.exists():
                    for item in web_dir.iterdir():
                        if item.is_dir():
                            if _force_rmtree(item):
                                cleaned.append(item.name)
                    # Remove scrape_report.json (regenerated on next scrape)
                    report_file = web_dir / "scrape_report.json"
                    if report_file.exists():
                        try:
                            report_file.unlink()
                            cleaned.append("scrape_report.json")
                        except Exception:
                            pass
                    # Also remove scrape_history.db so scraper re-downloads everything
                    history_file = web_dir / "scrape_history.db"
                    if history_file.exists():
                        try:
                            history_file.unlink()
                            cleaned.append("scrape_history.db")
                        except Exception:
                            pass
                # Also clean the reporter corpus (fast path DB indices)
                reporter_output = active_reporter_output()
                reporter_corpus = reporter_output / "corpus"
                if reporter_corpus.exists():
                    if _force_rmtree(reporter_corpus):
                        cleaned.append("reporter_corpus")
                # Remove pipeline progress files
                for f in [PIPELINE_PROGRESS, REPORTER_PROGRESS]:
                    try:
                        f.unlink()
                    except Exception:
                        pass
                self.send_json({"ok": True, "cleaned": cleaned})
            elif parsed.path == "/api/corpus/clean":
                # Clean the REPORTER corpus only — NEVER the main corpus
                # This removes all indices from the current run
                global _CLEANING_IN_PROGRESS
                import time as _ctime
                cleaned = []
                errors = []
                with _CLEANING_LOCK:
                    _CLEANING_IN_PROGRESS = True
                    # Wait for any in-flight db_counts() to finish (they hold SQLite handles)
                    _ctime.sleep(1.0)
                    # Remove ALL reporter runs (history) — this includes corpus + output
                    if REPORTER_ROOT.exists():
                        if _force_rmtree(REPORTER_ROOT):
                            cleaned.append("reporter_root (corpus + output + history)")
                        else:
                            errors.append("reporter_root (locked)")
                    # Remove pipeline progress
                    for f in [PIPELINE_PROGRESS, REPORTER_PROGRESS]:
                        try:
                            f.unlink()
                        except Exception:
                            pass
                    # Remove locks
                    for f in (ROOT / "outputs" / "web_dashboard").glob("*.lock"):
                        try:
                            f.unlink()
                        except Exception:
                            pass
                    _CLEANING_IN_PROGRESS = False
                self.send_json({"ok": len(errors) == 0, "cleaned": cleaned, "errors": errors})
            elif parsed.path == "/api/process/kill":
                pid = int(body.get("pid", 0))
                name = str(body.get("name", ""))
                if pid <= 0:
                    raise ValueError("pid inválido")
                # Kill by PID
                try:
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS if os.name == "nt" else 0)
                    # Also remove from JOBS if present
                    with JOBS_LOCK:
                        for k, proc in list(JOBS.items()):
                            if proc.pid == pid:
                                proc.kill()
                                del JOBS[k]
                                break
                    self.send_json({"ok": True, "pid": pid, "message": f"Proceso {pid} terminado"})
                except Exception as exc:
                    raise RuntimeError(f"No se pudo terminar el proceso {pid}: {exc}")
            else:
                self.send_json({"error": "not found"}, 404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    # Kill any orphan pipeline processes from previous runs
    for pattern in ["run_web_scrape.py", "run_fast_path.py", "run_reporter.py"]:
        killed = _kill_orphan_processes(pattern)
        if killed:
            print(f"  [startup] killed {killed} orphan(s) matching {pattern}", flush=True)
    # Clean stale locks
    for kind in ["scraper", "pipeline", "lancedb", "reporter", "reporter_fast"]:
        try:
            _job_lock_path(kind).unlink(missing_ok=True)
        except Exception:
            pass
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"IPA web dashboard: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if DEEP_DIVE_PROVIDER is not None:
            DEEP_DIVE_PROVIDER.unload()


if __name__ == "__main__":
    main()
