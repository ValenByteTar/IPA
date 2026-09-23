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

import time

import urllib.parse

from datetime import datetime, timezone

from http import HTTPStatus

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pathlib import Path

from typing import Any



import yaml



from .state import (

    ROOT,

    SOURCES_DB,

    STATE_DB,

    SCRAPE_CONFIG,

    base_sources,

    dashboard_connection,

    effective_scrape_config,

    load_sources,

    now,

    read_json,

    safe_url,

    save_sources,

)



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

REPORTER_ROOT = ROOT / "outputs" / "reporter"

REPORTER_PROGRESS = ROOT / "outputs" / "web_dashboard" / "reporter_progress.json"

AGENT_RESEARCH_PROGRESS = ROOT / "outputs" / "web_dashboard" / "research_progress.json"

PIPELINE_PROGRESS = ROOT / "outputs" / "web_dashboard" / "pipeline_progress.json"

MAIN_CORPUS = ROOT / "outputs" / "experiments" / "E12-corpus"

JOBS: dict[str, subprocess.Popen] = {}

JOBS_LOCK = threading.Lock()

DEEP_DIVE_PROVIDER = None

# Singleton embedding adapter for auto-retrieval (CPU, preloaded at startup).
_EMBED_ADAPTER = None

def get_embedding_adapter():
    """Lazy singleton for BGE-M3 on CPU — avoids reloading per request."""
    global _EMBED_ADAPTER
    if _EMBED_ADAPTER is None:
        from ipa.indexes.embedding_adapter import EmbeddingAdapter
        _EMBED_ADAPTER = EmbeddingAdapter(device="cpu", show_progress=False)
    return _EMBED_ADAPTER

_CLEANING_LOCK = threading.Lock()

_CLEANING_IN_PROGRESS = False

# Active reporter output directory — set by run_full_pipeline, read by dashboard_state.
# Persistido a disco: un restart del dashboard (watchdog) perdía el puntero en
# memoria y las métricas caían al fallback "report.json más reciente", que puede
# ser un corpus vacío — el corpus en uso no tiene report.json hasta que se
# genera (bug real medido 2026-09-22: mostraba 0 documentos con 127k chunks vivos).
_ACTIVE_REPORTER_OUTPUT: Path | None = None
ACTIVE_REPORTER_OUTPUT_PATH = ROOT / "outputs" / "web_dashboard" / "active_reporter_output.json"
_CORPUS_DOC_COUNT_TTL_S = 30.0
_corpus_doc_count_cache: dict[str, tuple[float, int]] = {}


def set_active_reporter_output(path: Path | None) -> None:
    """Fija y persiste el output activo del reporter (sobrevive al restart)."""
    global _ACTIVE_REPORTER_OUTPUT
    _ACTIVE_REPORTER_OUTPUT = Path(path).resolve() if path is not None else None
    try:
        ACTIVE_REPORTER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        if _ACTIVE_REPORTER_OUTPUT is None:
            ACTIVE_REPORTER_OUTPUT_PATH.unlink(missing_ok=True)
        else:
            temp = ACTIVE_REPORTER_OUTPUT_PATH.with_suffix(".tmp")
            temp.write_text(json.dumps({"path": str(_ACTIVE_REPORTER_OUTPUT)}),
                            encoding="utf-8")
            temp.replace(ACTIVE_REPORTER_OUTPUT_PATH)
    except Exception:
        pass  # el puntero persistido es una conveniencia, no estado canónico


def _load_persisted_active_reporter_output() -> Path | None:
    """Recupera el puntero persistido si el directorio sigue existiendo."""
    try:
        data = json.loads(ACTIVE_REPORTER_OUTPUT_PATH.read_text(encoding="utf-8"))
        raw_path = str(data.get("path") or "")
        if raw_path:
            path = Path(raw_path).resolve()
            if (REPORTER_ROOT.resolve() in path.parents
                    and path.is_dir() and (path / "corpus").is_dir()):
                return path
    except Exception:
        pass
    try:
        ACTIVE_REPORTER_OUTPUT_PATH.unlink(missing_ok=True)
    except Exception:
        pass
    return None


def _corpus_doc_count(output_dir: Path) -> int:
    """Docs vivos del corpus de un output (cache 30s: /api/state lo consulta seguido)."""
    key = str(output_dir)
    cached = _corpus_doc_count_cache.get(key)
    if cached is not None and time.time() - cached[0] < _CORPUS_DOC_COUNT_TTL_S:
        return cached[1]
    count = 0
    store = output_dir / "corpus" / "document_store.db"
    if store.exists():
        try:
            with sqlite3.connect(f"file:{store}?mode=ro", uri=True, timeout=5) as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM documents WHERE tombstoned=0").fetchone()[0]
        except Exception:
            count = 0
    _corpus_doc_count_cache[key] = (time.time(), count)
    return count

DEEP_DIVE_LOCK = threading.Lock()

PIPELINE_THREAD = None

PIPELINE_LOCK = threading.Lock()

# True mientras un chat stream está generando: el consolidador de sesiones
# lo respeta para no usar el generator concurrentemente (no thread-safe).
CHAT_BUSY = {"flag": False}

# Sesión que pidió un pipeline por chat: cuando el pipeline termina, el
# agente genera el resumen y lo escribe como episodio en esa sesión.
PIPELINE_WATCH = {"session_id": None, "saw_running": False}

# Sesión que pidió una investigación web por chat (research_topic): el
# watcher lee outputs/web_dashboard/research_progress.json y, cuando el
# job termina, escribe el resumen como episodio en esa sesión.
RESEARCH_WATCH = {"session_id": None, "saw_running": False}

# Idle enrichment: timestamp of last user activity (chat, job, pipeline).
# Level 1 runs every 5 min of idle; Level 2 (LLM) after 30 min continuous idle.
LAST_ACTIVITY = {"ts": time.time()}
IDLE_DEEP_THRESHOLD = int(os.environ.get("IPA_IDLE_DEEP_THRESHOLD_MINUTES", "30"))
IDLE_DEEP_ENABLED = os.environ.get("IPA_IDLE_DEEP_ENRICHMENT", "1") == "1"
# Tier 2 "liviano": si el modelo YA está cargado por el chat y el sistema
# está quieto, se aprovecha para enriquecer sin cargar nada (interrumpible
# al primer mensaje). No levanta el modelo por sí solo.
IDLE_LLM_LOADED_ENABLED = os.environ.get("IPA_IDLE_LLM_LOADED_ENRICHMENT", "1") != "0"
IDLE_LLM_LOADED_THRESHOLD = int(os.environ.get("IPA_IDLE_LLM_LOADED_THRESHOLD_MINUTES", "5"))

# Re-lectura LLM de docs rechazados por el research executor: corre tras
# ~60s sin input del usuario (configurable). Cola SQLite resumable; el
# worker corta entre items si vuelve la actividad.
RESEARCH_REVIEW_IDLE = int(os.environ.get("IPA_RESEARCH_REVIEW_IDLE_SECONDS", "60"))

# Lock to prevent concurrent enrichment runs (e.g. two dashboard instances).
ENRICHMENT_LOCK = threading.Lock()

# Perilla del sidebar: apaga TODO el enriquecimiento idle (Tier 1 y Tier 2).
# El worker consulta este dict en cada ciclo vía _idle(); OFF también aborta
# pasadas Tier 2 en vuelo (should_abort → _idle()). Persistido en disco para
# que un restart del dashboard respete la decisión del usuario.
IDLE_ENABLED = {"enabled": True}
IDLE_ENABLED_PATH = ROOT / "outputs" / "web_dashboard" / "idle_enabled.json"


def idle_enabled_state() -> bool:
    return bool(IDLE_ENABLED["enabled"])


def idle_enabled_state() -> bool:
    return bool(IDLE_ENABLED["enabled"])


def set_idle_enabled(enabled: bool) -> None:
    IDLE_ENABLED["enabled"] = bool(enabled)
    try:
        IDLE_ENABLED_PATH.parent.mkdir(parents=True, exist_ok=True)
        IDLE_ENABLED_PATH.write_text(
            json.dumps({"enabled": bool(enabled)}), encoding="utf-8")
    except OSError:
        pass
    # Auditoría en el log del worker (mismo archivo que los ciclos idle).
    try:
        _log_path = ROOT / "outputs" / "web_dashboard" / "logs" / "idle_enrichment.log"
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [idle-sched] "
                    f"enrichment {'ENABLED' if enabled else 'DISABLED'} by user\n")
    except OSError:
        pass


def _read_idle_enabled_at_boot() -> None:
    try:
        if IDLE_ENABLED_PATH.exists():
            IDLE_ENABLED["enabled"] = json.loads(
                IDLE_ENABLED_PATH.read_text(encoding="utf-8")).get("enabled", True)
    except Exception:
        pass  # archivo ausente/corrupto → default ON


_read_idle_enabled_at_boot()



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

    2. El mismo puntero persistido en disco (sobrevive al restart del dashboard)

    3. Most populated corpus under REPORTER_ROOT/quality-check/ (works before
       report.json exists and avoids choosing a newer empty run)

    4. Fall back to legacy hardcoded path

    """

    global _ACTIVE_REPORTER_OUTPUT

    if _ACTIVE_REPORTER_OUTPUT is None:

        _ACTIVE_REPORTER_OUTPUT = _load_persisted_active_reporter_output()

    if (_ACTIVE_REPORTER_OUTPUT is not None
            and _ACTIVE_REPORTER_OUTPUT.is_dir()
            and (_ACTIVE_REPORTER_OUTPUT / "corpus").is_dir()):

        return _ACTIVE_REPORTER_OUTPUT

    if _ACTIVE_REPORTER_OUTPUT is not None:

        set_active_reporter_output(None)

    qc_dir = REPORTER_ROOT / "quality-check"

    if qc_dir.exists():

        candidates = [d for d in qc_dir.iterdir()
                      if d.is_dir() and (d / "corpus").is_dir()]

        if candidates:

            # Priorizar corpus poblado (aunque aún no tenga report.json); entre
            # corpus equivalentes, el de actividad más reciente. Un reporte
            # reciente vacío nunca debe ocultar el corpus realmente indexado.
            candidates.sort(
                key=lambda d: (_corpus_doc_count(d), d.stat().st_mtime),
                reverse=True,
            )

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

            _resp = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
            tables = list(_resp.tables if hasattr(_resp, "tables") else _resp)

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




def transit_counts() -> dict[str, Any]:

    """Count files in Transit/ (processed, awaiting promotion confirmation)."""

    transit_dir = ROOT / "Transit"

    if not transit_dir.exists():

        return {"total_files": 0}

    total = sum(1 for f in transit_dir.rglob("*") if f.is_file())

    return {"total_files": total}




def run_landing_sweep() -> dict[str, Any]:

    """Run the Landing transit-policy sweep with the standard dashboard roots.

    Safe to call from any lifecycle point (pipeline end, promotion queue,
    human review) — it only moves/deletes files whose artifacts are already
    in a terminal or pending-confirmation state.
    """

    try:
        from ipa.ingestion.landing_sweep import sweep_landing
        reporter_output = REPORTER_ROOT / "quality-check" / active_reporter_output().name
        return sweep_landing(
            ROOT / "Landing",
            ROOT / "Archive",
            [MAIN_CORPUS, reporter_output / "corpus"],
            main_corpus=MAIN_CORPUS,
            transit_root=ROOT / "Transit",
            cluster_db=ROOT / "outputs" / "agent" / "topic_clusters.db",
            reporter_db=reporter_output / "reporter.db",
        )
    except Exception as exc:
        return {"archived": 0, "transit": 0, "deleted": 0, "skipped": 0,
                "pending_cleaned": 0, "errors": [str(exc)]}





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

    # The compiled extension lives in exllamav3-dev/build/ (or beside the

    # checkout); the provider auto-discovers it on import.

    extension_candidates = [

        ROOT / "exllamav3-dev" / "exllamav3_ext.cp312-win_amd64.pyd",

        ROOT / "exllamav3-dev" / "build" / "exllamav3_ext.cp312-win_amd64.pyd",

    ]

    extension = next((p for p in extension_candidates if p.exists()), None)

    try:

        import torch

        cuda = bool(torch.cuda.is_available())

        gpu = torch.cuda.get_device_name(0) if cuda else None

    except (ImportError, RuntimeError):

        cuda, gpu = False, None

    # Active provider/model — what the agent actually uses at runtime.
    # Determined by IPA_LLM_PROVIDER (default: ollama). Si el provider
    # configurado es ExL3 y no hay GPU, la fábrica cae a Ollama (CPU) y el
    # status refleja eso. The EXL3 fields above remain for diagnostics.
    from ipa.providers.device import has_gpu
    provider_type = os.environ.get("IPA_LLM_PROVIDER", "ollama").strip().lower()
    if provider_type != "ollama" and not has_gpu():
        provider_type = "ollama"
    if provider_type == "ollama":
        active_model = os.environ.get("IPA_OLLAMA_MODEL", "qwen3.5:9b-q4_K_M")
        # Ollama readiness: probe the daemon. Cheap GET /api/tags.
        ollama_ready = False
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2).read()
            ollama_ready = True
        except Exception:
            ollama_ready = False
        return {
            "provider": "Ollama",
            "model_name": active_model,
            "model_present": True,
            "extension_present": True,
            "cuda_available": cuda,
            "gpu": gpu,
            "ready": ollama_ready,
        }
    # EXL3 path
    return {
        "provider": "ExLlamaV3",
        "model_name": "Qwen3.5-9B EXL3 3.0bpw",
        "model_present": model.exists(),
        "extension_present": extension is not None,
        "cuda_available": cuda,
        "gpu": gpu,
        "ready": model.exists() and extension is not None and cuda,
    }





def get_deep_dive_provider():

    global DEEP_DIVE_PROVIDER

    with DEEP_DIVE_LOCK:

        if DEEP_DIVE_PROVIDER is not None and DEEP_DIVE_PROVIDER.is_loaded():

            if not getattr(DEEP_DIVE_PROVIDER, "_t2_owned", False):

                return DEEP_DIVE_PROVIDER

            # El pase Tier 2 cargó un motor batch (ExL3 ctx 2048): no sirve
            # para el chat (prompts de 2.3-4.1k) y el usuario tiene prioridad.
            # Se descarga y se cae al camino normal (factory → Ollama).
            # unload() libera el lock de VRAM en el mismo proceso, así la
            # carga siguiente lo encuentra libre. El worker del Tier 2 aborta
            # vía should_abort (CHAT_BUSY) y su finally solo descarga si la
            # instancia sigue siendo la suya (guard por identidad).
            try:

                DEEP_DIVE_PROVIDER.unload()

            except Exception:

                pass

            DEEP_DIVE_PROVIDER = None

        with JOBS_LOCK:

            reporter_job = JOBS.get("reporter")

            if reporter_job and reporter_job.poll() is None:

                return None

        # Provider selection: IPA_LLM_PROVIDER=ollama uses Ollama (GGUF),
        # otherwise ExL3 (EXL3). Con la fábrica, si no hay GPU el ExL3 cae
        # automáticamente a Ollama (CPU) — el sistema nunca queda sin LLM
        # por falta de GPU. El guard de readiness de ExL3 queda acá porque
        # depende de archivos locales del dashboard.
        if os.environ.get("IPA_LLM_PROVIDER", "ollama").strip().lower() != "ollama":
            if not llm_status().get("ready"):
                return None
        from ipa.providers.factory import create_star_provider
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

    pipeline_progress = _pipeline_progress_view() if PIPELINE_PROGRESS.exists() else {}

    if pipeline_progress:

        result["rechunk"] = {"status": pipeline_progress.get("status", "running"), "detail": "pipeline completo", "percent": pipeline_progress.get("percent", 0), "stage": pipeline_progress.get("stage", "")}

    # Dashboard-managed jobs (from pipeline runs) â€” these are the real live ones

    with JOBS_LOCK:

        for name in list(JOBS.keys()):

            proc = JOBS[name]

            if proc.poll() is not None:

                # Process finished â€” keep status briefly then remove from JOBS

                # so it doesn't show as "error" forever

                status = "done" if proc.returncode == 0 else "error"

                result[f"web_{name}"] = {"status": status, "pid": proc.pid, "returncode": proc.returncode}

                if name == "scraper":

                    result["scraper"] = {"status": status, "pid": proc.pid, "detail": "web scraper"}

                elif name == "pipeline":

                    result["pipeline"] = {"status": status, "pid": proc.pid, "detail": "fast path (BM25 + LanceDB)"}

                    # El fast path incluye el embedding LanceDB: reflejarlo
                    # en la card de LanceDB (mismo subprocess, misma suerte).
                    result["lancedb"] = {"status": status, "pid": proc.pid, "detail": "embeddings (dentro de FastPath)"}

                elif name == "lancedb":

                    result["lancedb"] = {"status": status, "pid": proc.pid, "detail": "LanceDB re-index"}

                elif name == "reporter":

                    result["enrichment"] = {"status": status, "pid": proc.pid, "detail": "reporter (BGE-M3 + Qwen)"}

                    progress = read_json(REPORTER_PROGRESS, {})

                    if progress:

                        result["web_reporter"].update(progress)

                elif name == "reporter_fast":

                    result["rechunk"] = {"status": status, "pid": proc.pid, "detail": "reporter rÃ¡pido"}

                # Remove finished jobs from JOBS so they don't persist as "error"

                del JOBS[name]

            else:

                status = "running"

                result[f"web_{name}"] = {"status": status, "pid": proc.pid}

                if name == "scraper":

                    result["scraper"] = {"status": status, "pid": proc.pid, "detail": "web scraper"}

                elif name == "pipeline":

                    result["pipeline"] = {"status": status, "pid": proc.pid, "detail": "fast path (BM25 + LanceDB)"}

                    # El fast path también hace los embeddings: reflejar en LanceDB
                    result["lancedb"] = {"status": status, "pid": proc.pid, "detail": "embeddings (dentro de FastPath)"}

                elif name == "lancedb":

                    result["lancedb"] = {"status": status, "pid": proc.pid, "detail": "LanceDB re-index"}

                elif name == "reporter":

                    result["enrichment"] = {"status": status, "pid": proc.pid, "detail": "reporter (BGE-M3 + Qwen)"}

                    progress = read_json(REPORTER_PROGRESS, {})

                    if progress:

                        result["web_reporter"].update(progress)

                elif name == "reporter_fast":

                    result["rechunk"] = {"status": status, "pid": proc.pid, "detail": "reporter rÃ¡pido"}

                # Remove finished jobs from JOBS so they don't persist as "error"

                del JOBS[name]

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

        raise ValueError("nombre de proceso invÃ¡lido")

    state_dir = MAIN_CORPUS / "process_state"

    log_dir = ROOT / "outputs" / "web_dashboard" / "logs"

    detail: dict[str, Any] = {"name": name}



    # Map process names to their log files

    log_map = {

        "scraper": "scraper.log",

        "pipeline": "fast_path.log",

        "lancedb": "lancedb.log",

        "enrichment": "reporter.log",

        "rechunk": None,  # pipeline completo â€” combined log below

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

        # Pipeline completo — reads from PIPELINE_PROGRESS + combines logs.
        # Prioriza el log de la fase activa (stage) para que el watch loop del
        # fast_path no inunde el panel y el usuario vea el progreso real.
        progress = read_json(PIPELINE_PROGRESS, {})

        detail["state"] = progress

        detail["status"] = progress.get("status", "not_started")

        detail["stage"] = progress.get("stage", "")

        detail["percent"] = progress.get("percent")

        detail["timestamp"] = progress.get("updated_at")

        detail["pid"] = None  # runs as thread, not subprocess

        detail["returncode"] = None

        # Mapear stage → log principal para priorizarlo en el panel
        stage = progress.get("stage", "")
        stage_log_map = {
            "scraper": "scraper.log",
            "fast_path": "fast_path.log",
            "ingestion": "fast_path.log",
            "reporter_fast": "reporter.log",
            "reporter": "reporter.log",
            "parallel": "reporter.log",
        }
        primary_log = stage_log_map.get(stage, "scraper.log")
        # Orden: primary primero, luego los otros dos con menos líneas
        log_order = [primary_log] + [l for l in ["scraper.log", "fast_path.log", "reporter.log"] if l != primary_log]

        combined_log = []

        for log_name in log_order:

            # El log principal: 50 líneas; los otros: 15 (menos ruido)
            n = 50 if log_name == primary_log else 15

            lines = _log_tail(log_dir / log_name, lines=n)

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





# promote_report_to_main() was removed — promotion is now handled by
# ipa.agentic.promotion_executor.promote_documents_to_main() which works
# with individual document IDs, independent of the report.
# The /api/reports/review endpoint now calls promote_documents_to_main directly.

def update_latest_topic(body: dict[str, Any]) -> dict[str, Any]:

    """Apply a constrained human edit to the newest valid report and its store."""

    report = latest_report()

    if not report:

        raise FileNotFoundError("no hay reporte disponible")

    category_id = str(body.get("category_id", ""))

    if not category_id or len(category_id) > 200:

        raise ValueError("category_id invÃ¡lido")

    allowed = {"label", "description"}

    updates = {key: body[key] for key in allowed if key in body}

    if "label" in updates and (not isinstance(updates["label"], str) or not 1 <= len(updates["label"]) <= 200):

        raise ValueError("label invÃ¡lido")

    if "description" in updates and (not isinstance(updates["description"], str) or not 1 <= len(updates["description"]) <= 2000):

        raise ValueError("description invÃ¡lida")

    if "status" in body:

        status = str(body["status"])

        if status not in {"draft", "reviewed", "published"}:

            raise ValueError("status invÃ¡lido")

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

        from ipa.reporter.reporter_store import ReporterStore

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

    from ipa.agentic.embedding_maintenance import read_state as embedding_maintenance_state

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

        "pipeline": _pipeline_progress_view(),

        "reporter_progress": read_json(REPORTER_PROGRESS, {}),

        "agent_research": read_json(AGENT_RESEARCH_PROGRESS, {}),

        "embedding_maintenance": embedding_maintenance_state(),

        "index_health": read_json(
            ROOT / "outputs" / "agent" / "index_health.json", {}),

        "scrape_counts": scrape_counts(),

        "archive_counts": archive_counts(),

        "transit_counts": transit_counts(),

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





from .jobs import _acquire_job_lock, _find_running_processes, _job_lock_path, _kill_orphan_processes, _process_exists, spawn_job



def _write_pipeline_progress(stage: str, status: str, percent: int, detail: str = "") -> None:

    PIPELINE_PROGRESS.parent.mkdir(parents=True, exist_ok=True)

    PIPELINE_PROGRESS.write_text(json.dumps({

        "stage": stage, "status": status, "percent": percent,

        "detail": detail, "updated_at": now(),

    }, ensure_ascii=False), encoding="utf-8")




def _pipeline_progress_view() -> dict[str, Any]:

    """pipeline_progress.json with stale-run detection.

    The pipeline thread lives inside the dashboard process; if the process
    dies (restart, crash) the file keeps saying "running" forever and the UI
    shows a frozen live card — the run can no longer advance because the
    orchestrator is gone even if child processes survived. Once no spawned
    job is alive and the last write is old, mark it interrupted (same
    self-healing discipline as embedding_maintenance's pid liveness check).
    """

    progress = read_json(PIPELINE_PROGRESS, {"stage": "idle", "status": "idle", "percent": 0})

    if progress.get("status") != "running":
        return progress

    try:
        age_s = time.time() - PIPELINE_PROGRESS.stat().st_mtime
    except OSError:
        return progress

    with JOBS_LOCK:
        jobs_alive = any(proc.poll() is None for proc in JOBS.values())

    if jobs_alive or age_s < 300:
        return progress

    detail = str(progress.get("detail") or "")

    _write_pipeline_progress(
        str(progress.get("stage") or "pipeline"), "interrupted",
        int(progress.get("percent") or 0),
        f"{detail} — interrumpido: el proceso orquestador ya no está vivo",
    )

    return read_json(PIPELINE_PROGRESS, progress)





def run_full_pipeline(period_start: str = "", period_end: str = "", period_mode: str = "days", days_back: int = 0) -> None:

    """Run scraper + fast path in parallel, then re-run fast path + both reporters in parallel.



    Stage 1: Scraper + Fast Path (parallel)

      - Scraper downloads files to Landing/web

      - Fast Path indexes whatever is already in Landing/web (BM25 + LanceDB)

        (idempotent â€” no duplicates on re-run)

    Stage 2: Fast Path (re-run) + Fast Reporter + Full Reporter (parallel)

      - Fast Path picks up new files from the scraper

      - Fast Reporter generates quick report (no embeddings/LLM)

      - Full Reporter generates complete report (BGE-M3 + Qwen)



    period_mode:

      - 'days': use days_back to compute period (today - days_back â†’ today)

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

            print(f"  [pipeline] period from days_back={days_back}: {period_start} â†’ {period_end}", flush=True)

        elif period_mode == "range" and period_start and period_end:

            print(f"  [pipeline] period from range: {period_start} â†’ {period_end}", flush=True)



        # Compute dynamic output directory based on period label

        # e.g. period_end="2026-09-30T23:59:59Z" â†’ "optimized-llm-2026-09"

        period_label = "unspecified"

        if period_end:

            period_label = period_end[:7]  # YYYY-MM

        reporter_output = REPORTER_ROOT / "quality-check" / f"optimized-llm-{period_label}"

        reporter_corpus = reporter_output / "corpus"

        set_active_reporter_output(reporter_output)

        print(f"  [pipeline] reporter output: {reporter_output}", flush=True)



        # Kill any orphan processes from previous runs before starting

        for pattern in ["run_web_scrape.py", "run_fast_path.py"]:

            killed = _kill_orphan_processes(pattern)

            if killed:

                print(f"  [pipeline] killed {killed} orphan(s) matching {pattern}", flush=True)

        # Clean stale locks

        for kind in ["scraper", "pipeline", "lancedb", "reporter", "reporter_fast"]:

            try:

                _job_lock_path(kind).unlink(missing_ok=True)

            except Exception:

                pass

        # CRÍTICO: liberar la VRAM del modelo estrella antes de que el
        # reporter cargue BGE-M3 + Qwen en GPU. Con el LLM residente
        # (~5.4 GB de 6 GB) el reporter hace OOM de CUDA y el pipeline
        # falla (caída del 2026-09-08). El chat lo recarga lazy después.
        global DEEP_DIVE_PROVIDER
        try:
            if DEEP_DIVE_PROVIDER is not None and DEEP_DIVE_PROVIDER.is_loaded():
                print("  [pipeline] unload del LLM estrella (liberar VRAM para el reporter)", flush=True)
                DEEP_DIVE_PROVIDER.unload()
                DEEP_DIVE_PROVIDER = None
        except Exception as exc:
            print(f"  [pipeline] aviso: no se pudo descargar el LLM: {exc}", flush=True)



        # --- Stage 1: Scraper + Fast Path (watch mode) IN PARALLEL ---

        _write_pipeline_progress("scraper", "running", 0, "Scraper + indexing continuo en paralelo")



        # The scraper-done sentinel tells the fast path watch it can stop

        # counting idle iterations. The scraper itself writes it via

        # --done-file on exit (success or failure), so the gate fires even if

        # this pipeline thread dies; the write below stays as a fallback for

        # hard-killed scrapers. Remove any stale sentinel first.

        scraper_done_flag = ROOT / "Landing" / "web" / ".scraper_done"

        try:

            scraper_done_flag.unlink(missing_ok=True)

        except Exception:

            pass



        # Scraper

        config = effective_scrape_config()

        scraper_cmd = [VENV_PYTHONW, "-u", "scripts/cli/run_web_scrape.py", "--config", str(config), "--output", "Landing/web", "--engine", "auto", "--no-images", "--no-ocr", "--done-file", str(scraper_done_flag)]

        # Si el agente pidió days_back específico, sobreescribir el config
        # por sitio (bug del 2026-09-08: el days_back del agente no llegaba
        # al scraper, que usaba el days_back del sources.json que era 5).
        if days_back and days_back > 0:

            scraper_cmd += ["--days-back", str(days_back)]

        # Si days_back es grande (>30), limpiar el historial de scrape para
        # permitir re-descubrir artículos viejos que ya fueron scrapeados
        # en runs anteriores con days_back chico.
        if days_back and days_back > 30:

            scraper_cmd += ["--clear-history"]

            print(f"  [pipeline] days_back={days_back} > 30: clearing scrape history", flush=True)

        scraper_log = open(log_dir / "scraper.log", "a", encoding="utf-8")

        scraper_proc = subprocess.Popen(scraper_cmd, cwd=str(ROOT), stdout=scraper_log, stderr=subprocess.STDOUT, env=env, creationflags=no_window)



        # Fast Path in watch mode: indexes to the REPORTER's corpus directly

        # The reporter reuses this corpus instead of re-ingesting

        # Only promoted to main corpus when user approves the report

        # reporter_output and reporter_corpus already set above (dynamic)
        # (scraper_done_flag already defined and cleared above)

        fast_path_cmd = [VENV_PYTHONW, "-u", "scripts/cli/run_fast_path.py", "--input", "Landing/web", "--output", str(reporter_corpus), "--watch", "10", "--idle-exit", "3", "--idle-gate", str(scraper_done_flag)]

        fp_log = open(log_dir / "fast_path.log", "a", encoding="utf-8")

        fp_proc = subprocess.Popen(fast_path_cmd, cwd=str(ROOT), stdout=fp_log, stderr=subprocess.STDOUT, env=env, creationflags=no_window)



        with JOBS_LOCK:

            JOBS["scraper"] = scraper_proc

            JOBS["pipeline"] = fp_proc



        # Wait for scraper to finish — with LIVE progress feedback.
        # Antes: scraper_proc.wait() dejaba la card en "0%" durante minutos
        # (parecía clavada). Ahora poll de counts reales cada 5s.
        import time as _time

        baseline_files = scrape_counts()["total_files"]

        while scraper_proc.poll() is None:

            _time.sleep(5)

            try:

                live = scrape_counts()

                files = live.get("total_files", 0)

                sites = live.get("sites", 0)

                # Progreso honesto: sube con los archivos nuevos (tope 28%,
                # el resto del rango lo completa el indexing + reporter)
                gained = max(0, files - baseline_files)

                pct = min(28, int(files * 1.2))

                _write_pipeline_progress("scraper", "running", pct,

                    f"Scraping: {files} documentos · {live.get('sites', 0)} sitios")

            except Exception:

                pass

        scraper_log.close()

        if scraper_proc.returncode != 0:

            _write_pipeline_progress("scraper", "failed", 100, f"Scraper fallÃ³ (exit {scraper_proc.returncode})")

            fp_proc.terminate()

            fp_log.close()

            return

        _write_pipeline_progress("scraper", "done", 30, "Scraper completado — esperando indexing final (BM25 + LanceDB)")



        # Signal the fast path watch that the scraper is done: from now on it

        # counts idle iterations and exits on its own after --idle-exit rounds

        # with no new work. The pipeline waits for the process to finish

        # NATURALLY instead of killing it on a fixed timeout.

        try:

            scraper_done_flag.write_text("done", encoding="utf-8")

        except Exception:

            pass

        import time as _time

        _write_pipeline_progress("fast_path", "running", 35, "Esperando indexing final (BM25 + LanceDB)")

        max_wait = 6 * 3600  # safety net only; normal exit is via --idle-exit

        waited = 0

        while fp_proc.poll() is None and waited < max_wait:

            _time.sleep(10)

            waited += 10

            try:

                import sqlite3 as _sql

                with _sql.connect(str(reporter_corpus / "document_store.db")) as conn:

                    bm25_docs = conn.execute("SELECT COUNT(*) FROM documents WHERE tombstoned=0").fetchone()[0]

                    bm25_chunks = conn.execute("SELECT COUNT(*) FROM chunks WHERE tombstoned=0").fetchone()[0]

                lance_chunks = 0

                try:

                    import lancedb as _ldb

                    ldb = _ldb.connect(str(reporter_corpus / "vector" / "lancedb"))

                    _resp = ldb.list_tables() if hasattr(ldb, "list_tables") else ldb.table_names()
                    tables = list(_resp.tables if hasattr(_resp, "tables") else _resp)

                    if "chunks" in tables:

                        lance_chunks = ldb.open_table("chunks").count_rows()

                except Exception:

                    pass

                scraped = scrape_counts()["total_files"]

                catchup = (lance_chunks / bm25_chunks) if bm25_chunks > 0 else 0

                pct = 30 + int(min(1.0, catchup) * 10)

                _write_pipeline_progress("fast_path", "running", pct,

                    f"Indexing: {bm25_chunks} BM25 / {lance_chunks} LanceDB / {scraped} scraped · {waited}s")

            except Exception:

                pass

        if fp_proc.poll() is None:

            print(f"  [pipeline] fast path exceeded {max_wait}s — terminating", flush=True)

            fp_proc.terminate()

            try:

                fp_proc.wait(timeout=15)

            except Exception:

                fp_proc.kill()

        fp_log.close()

        # Provenance: docs scraped from configured sites need a
        # document_sources row or promotion_policy discards them as
        # "unknown provenance". scrape_report.json first, landing registry
        # as fallback (works even if the report is empty or was swept).
        try:
            from ipa.ingestion.provenance import (
                backfill_from_scrape_report, backfill_from_landing_registry,
                load_configured_urls,
            )
            from ipa.storage.document_store import DocumentStore
            if (reporter_corpus / "document_store.db").exists():
                _prov_store = DocumentStore(reporter_corpus / "document_store.db")
                try:
                    _n = backfill_from_scrape_report(
                        _prov_store,
                        ROOT / "Landing" / "web" / "scrape_report.json",
                        reporter_corpus / "landing.db",
                        load_configured_urls(ROOT / "configs" / "scrape_sites.yaml"),
                    )
                    _n += backfill_from_landing_registry(
                        _prov_store,
                        reporter_corpus / "landing.db",
                        ROOT / "Landing",
                    )
                    if _n:
                        print(f"  [pipeline] provenance backfill: {_n} docs", flush=True)
                finally:
                    _prov_store.close()
        except Exception as _prov_exc:
            print(f"  [pipeline] provenance backfill error: {_prov_exc}", flush=True)

        # Ingesta pura: el pipeline sin agente termina acá — scraper + indexing.
        # El Reporter ya no es parte del pipeline automático; el agente lo
        # invoca via la tool compile_report cuando necesita un reporte fino.
        #
        # Landing es zona de tránsito (docs/operations/landing-and-archive.md):
        # aprobado (en main corpus) → Archive/, pendiente de confirmación →
        # Transit/, rechazado → eliminar. Nada procesado queda en Landing.
        scraped = scrape_counts()["total_files"]
        try:
            from ipa.ingestion.landing_sweep import sweep_landing
            _sweep = sweep_landing(
                ROOT / "Landing",
                ROOT / "Archive",
                [MAIN_CORPUS, reporter_corpus],
                main_corpus=MAIN_CORPUS,
                transit_root=ROOT / "Transit",
                cluster_db=ROOT / "outputs" / "agent" / "topic_clusters.db",
                reporter_db=reporter_output / "reporter.db",
            )
        except Exception as _sweep_exc:
            _sweep = {"archived": 0, "transit": 0, "deleted": 0, "skipped": 0,
                      "errors": [str(_sweep_exc)]}

        _write_pipeline_progress("ingestion", "done", 100,

            f"Ingesta completada: {scraped} documentos procesados (BM25 + LanceDB). "
            f"Landing barrido: {_sweep['archived']} a Archive, {_sweep['transit']} a Transit "
            f"(pendientes de confirmación), {_sweep['deleted']} eliminados, "
            f"{_sweep['skipped']} sin procesar. Reporter desacoplado — usar agente para reportes.")

        return

    except Exception as exc:

        _write_pipeline_progress("error", "failed", 100, str(exc))





from .api import Handler



def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--host", default="127.0.0.1")

    parser.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()

    # Single-instance guard: si ya hay un dashboard respondiendo en el
    # puerto, esta instancia se niega a arrancar. Dos instancias comparten
    # las mismas DBs (locks de SQLite) y duplican la carga de VRAM.
    import urllib.request as _ureq
    try:
        probe = urllib.request.urlopen(f"http://{args.host}:{args.port}/api/health", timeout=2)
        probe.close()
        print(f"[dashboard] ya hay una instancia activa en http://{args.host}:{args.port} — no se lanza otra.", flush=True)
        return
    except Exception:
        pass  # puerto libre: continuar

    # Kill any orphan pipeline processes from previous runs

    for pattern in ["run_web_scrape.py", "run_fast_path.py"]:

        killed = _kill_orphan_processes(pattern)

        if killed:

            print(f"  [startup] killed {killed} orphan(s) matching {pattern}", flush=True)

    # Clean stale locks

    for kind in ["scraper", "pipeline", "lancedb", "reporter", "reporter_fast"]:

        try:

            _job_lock_path(kind).unlink(missing_ok=True)

        except Exception:

            pass

    class _SingleBindHTTPServer(ThreadingHTTPServer):
        # SO_REUSEADDR on Windows lets a second process bind an already
        # listening port — that let runaway watchdog respawns stack N
        # dashboard instances on 8765. With this off, a duplicate fails
        # the bind and exits instead of piling up.
        allow_reuse_address = False

    server = _SingleBindHTTPServer((args.host, args.port), Handler)

    print(f"IPA web dashboard: http://{args.host}:{args.port}", flush=True)



    # Eager warmup: load the star model into VRAM in background so the

    # first chat message doesn't pay the ~10s load penalty. The thread

    # is non-blocking — the server serves requests immediately.

    def _warmup_provider():

        try:

            p = get_deep_dive_provider()

            if p is not None:

                print("  [warmup] modelo estrella cargado en VRAM y listo", flush=True)

            else:

                print("  [warmup] modelo no disponible (LLM no ready)", flush=True)

        except Exception as exc:

            print(f"  [warmup] error: {exc}", flush=True)

    # Preload BGE-M3 embedding adapter on CPU for auto-retrieval (singleton).

    def _warmup_embeddings():

        try:

            adapter = get_embedding_adapter()

            print("  [warmup] BGE-M3 embedding adapter listo (CPU)", flush=True)

            # Warmup REAL del pipeline de retrieval: el primer embed en CPU
            # paga ~10s de warmup de torch/tokenizer y la primera search
            # híbrida construye el índice FTS (~26s sobre 129k chunks).
            # Ejecutar ambos acá mueve ese costo al startup, no al primer
            # mensaje del usuario.
            try:
                from ipa.agent.system_tools import _main_corpus_dir
                _corpus = _main_corpus_dir()
                if _corpus:
                    from ipa.indexes.lancedb_index import LanceDBIndex
                    from ipa.storage.document_store import DocumentStore
                    _lance = LanceDBIndex(Path(_corpus / "vector" / "lancedb"), vector_dim=1024)
                    if _lance.is_queryable():
                        _dense, _sparse = adapter.embed_query_hybrid("warmup")
                        _lance.create_fts_index()
                        list(_lance.search_hybrid("warmup", _dense, limit=3, query_sparse=_sparse))
                        # Backfill scalar metadata columns (source_domain,
                        # published_at, provenance, quality_score) from the
                        # canonical store so pre-filtered hybrid search works
                        # on indexes that predate the metadata schema.
                        try:
                            _meta_store = DocumentStore(Path(_corpus / "document_store.db"))
                            _synced = _lance.sync_doc_metadata(_meta_store, only_missing=True)
                            _meta_store.close()
                            if _synced:
                                print(f"  [warmup] LanceDB metadata sync: {_synced} docs", flush=True)
                        except Exception as _me:
                            print(f"  [warmup] metadata sync skip: {_me!r}"[:200], flush=True)
                        print("  [warmup] retrieval pipeline listo (embed + FTS + hybrid)", flush=True)
                    _lance.close()
            except Exception as _exc:
                print(f"  [warmup] retrieval pipeline skip: {_exc!r}"[:200], flush=True)

        except Exception as exc:

            print(f"  [warmup] embedding adapter error: {exc}", flush=True)

    import threading as _threading

    _warmup_thread = _threading.Thread(target=_warmup_provider, daemon=True)

    _warmup_embed_thread = _threading.Thread(target=_warmup_embeddings, daemon=True)
    _warmup_embed_thread.start()

    _warmup_thread.start()

    # La consolidación de sesiones dejó de ser un worker propio: es la tarea
    # `consolidate_sessions` del scheduler idle (Tier 1, prioridad 20).
    # El cierre de sesiones huérfanas es la tarea `hygiene_sessions`.

    # Pipeline watcher: cuando el agente lanzó un pipeline por chat y este
    # termina, el agente escribe el resumen del informe como episodio en la
    # sesión que lo pidió (proactivo, sin request del usuario).
    def _pipeline_watch_worker():
        import time as _time
        from ipa.agent import AgentMemory
        from ipa.agent.system_tools import execute_system_tool

        def _deliver_episode(session_id: str, content: str, tag: str) -> None:
            memory = AgentMemory()
            try:
                memory.reopen_session(session_id)
                session = memory.get_session(session_id)
                memory.record_episode(
                    session_id, turn_role="assistant", content=content,
                    identity_hash=session.identity_hash if session else "unknown",
                )
                memory.close_session(session_id)
            finally:
                memory.close()
            print(f"  [{tag}] resumen entregado a sesion ...{session_id[-12:]}", flush=True)

        def _research_completion_answer(query: str) -> str | None:
            """Sintetiza una respuesta a la pregunta original con el material
            recién ingerido (BM25 sobre el corpus ya actualizado). None → el
            watcher cae al resumen de stats."""
            try:
                provider = DEEP_DIVE_PROVIDER
                if provider is None or not provider.is_loaded():
                    return None
                from ipa.agent.system_tools import _main_corpus_dir
                corpus = _main_corpus_dir() or MAIN_CORPUS
                if not (corpus / "bm25_index.db").exists():
                    return None
                from ipa.indexes.bm25_index import BM25Index
                from ipa.storage.document_store import DocumentStore
                bm25 = BM25Index(corpus / "bm25_index.db")
                store = DocumentStore(corpus / "document_store.db")
                try:
                    hits = bm25.search(query, limit=6)
                    texts = []
                    for h in hits:
                        ch = store.get_chunk(h.chunk_id)
                        if ch and ch.text:
                            texts.append(ch.text[:900])
                finally:
                    bm25.close()
                    store.close()
                if not texts:
                    return None
                ctx = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts, 1))
                messages = [
                    {"role": "system", "content": (
                        "Sos RA. La investigación web terminó y este material "
                        "quedó indexado en el corpus. Respondé la pregunta "
                        "original del usuario SOLO con este material, "
                        "concreto y sin marcadores [n] en el texto. Si el "
                        "material no alcanza, decí qué falta."
                    )},
                    {"role": "user", "content": (
                        f"Pregunta original: {query}\n\nMaterial encontrado:\n{ctx}"
                    )},
                ]
                CHAT_BUSY["flag"] = True  # el generator no es thread-safe
                try:
                    result = provider.generate_chat(
                        messages, max_new_tokens=384, temperature=0.3)
                finally:
                    CHAT_BUSY["flag"] = False
                text = result.text if hasattr(result, "text") else str(result or "")
                if getattr(result, "error", None):
                    return None
                return text.strip() or None
            except Exception:
                CHAT_BUSY["flag"] = False
                return None

        while True:
            _time.sleep(30)
            # --- Watch 1: pipeline/ingesta iniciada por chat ---
            try:
                watch = PIPELINE_WATCH
                if watch.get("session_id"):
                    state = _process_state("pipeline")
                    status = state.get("status", "unknown")
                    if status == "running":
                        watch["saw_running"] = True
                    elif watch.get("saw_running") and not CHAT_BUSY["flag"]:
                        provider = DEEP_DIVE_PROVIDER
                        if provider is not None and provider.is_loaded():
                            session_id = watch["session_id"]
                            watch_mode = watch.get("mode", "run_ingestion")
                            watch["session_id"] = None
                            watch["saw_running"] = False
                            live = scrape_counts()
                            content = (
                                f"✅ **{('Ingesta' if watch_mode == 'run_ingestion' else 'Pipeline')} completada** (estado: {status}).\n\n"
                                f"- Documentos en Landing/web: **{live.get('total_files', '?')}**\n"
                                f"- Sitios scrapeados: **{live.get('sites', '?')}**\n\n"
                                f"El contenido ya está indexado (BM25 + LanceDB) y consultable. "
                                "Tu base de conocimientos quedó enriquecida — preguntame lo que quieras sobre lo nuevo, "
                                "o pedime compilar un reporte con compile_report."
                            )
                            _deliver_episode(session_id, content, "pipeline-watch")
            except Exception:
                pass
            # --- Watch 2: investigación web iniciada por chat (research_topic) ---
            try:
                progress_path = ROOT / "outputs" / "web_dashboard" / "research_progress.json"
                progress = read_json(progress_path, {}) if progress_path.exists() else {}
                rstatus = progress.get("status", "idle")
                session_id = progress.get("session_id")
                notified = progress.get("notified", False)
                _watch_state = (rstatus, session_id, notified)
                if _watch_state != RESEARCH_WATCH.get("last_logged"):
                    RESEARCH_WATCH["last_logged"] = _watch_state
                    print(f"[research-watch] status={rstatus} session_id={session_id} notified={notified} busy={CHAT_BUSY['flag']}", flush=True)
                if rstatus in ("done", "failed") and session_id and not notified and not CHAT_BUSY["flag"]:
                    r = progress.get("result") or {}
                    if rstatus == "done":
                        # Si la investigación ingirió material, responder la
                        # pregunta original con ese contenido (el usuario pidió
                        # "responderme con lo que encontró"). Fallback: stats.
                        answer = None
                        if r.get("ingested"):
                            answer = _research_completion_answer(
                                progress.get("query", ""))
                        if answer:
                            content = (
                                f"Investigación completada: \"{progress.get('query', '?')}\"\n\n"
                                f"{answer}\n\n"
                                f"— {r.get('ingested', '?')} chunks quedaron indexados "
                                "en el corpus principal."
                            )
                        else:
                            content = (
                                f"Investigación completada: \"{progress.get('query', '?')}\"\n\n"
                                f"- Resultados de búsqueda: {r.get('search_results', '?')}\n"
                                f"- URLs scrapeadas: {r.get('scraped', '?')}\n"
                                f"- Rechazadas por el juez: {r.get('rejected', '?')}\n"
                                f"- Chunks ingeridos al corpus: {r.get('ingested', '?')}\n\n"
                                "Lo nuevo ya está indexado y consultable. Preguntame sobre el tema "
                                "o pedime compilar un reporte."
                            )
                    else:
                        content = (
                            f"La investigación falló: \"{progress.get('query', '?')}\"\n\n"
                            f"Error: {progress.get('error', 'desconocido')}"
                        )
                    _deliver_episode(session_id, content, "research-watch")
                    progress["notified"] = True
                    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
                    print(f"[research-watch] delivered to {session_id}", flush=True)
            except Exception as exc:
                print(f"[research-watch] error: {exc}", flush=True)

    _pipeline_watch_thread = _threading.Thread(target=_pipeline_watch_worker, daemon=True)

    _pipeline_watch_thread.start()

    # Research review worker: re-lectura LLM de docs scrapeados que el
    # research executor rechazó (quality/date/judge). Corre cuando el
    # usuario lleva ~RESEARCH_REVIEW_IDLE segundos sin interactuar;
    # interrumpible (corta entre items si vuelve actividad o el chat se
    # ocupa) y resumable (la cola SQLite sobrevive restarts — los pendientes
    # se retoman en el próximo idle).
    def _research_review_worker():
        import time as _time
        from ipa.agent.research_review import (
            ResearchReviewStore,
            ingest_reviewed_doc,
            review_doc_with_llm,
        )
        from ipa.agent.system_tools import _research_ingest_corpus

        # Dir propio del review (PM-004): ingerir Landing/web compartido hacía
        # que el worker recorriera el material de un pipeline concurrente.
        landing = ROOT / "outputs" / "agent" / "research" / "review"
        while True:
            _time.sleep(15)
            try:
                if _time.time() - LAST_ACTIVITY["ts"] < RESEARCH_REVIEW_IDLE:
                    continue
                if CHAT_BUSY["flag"]:
                    continue
                store = ResearchReviewStore()
                try:
                    items = store.pending(limit=3)
                    if not items:
                        continue
                    # Solo relee si el LLM ya está cargado — el review no
                    # justifica levantar el modelo (eso es Level 2).
                    provider = DEEP_DIVE_PROVIDER
                    if provider is None or not provider.is_loaded():
                        continue
                    # DEC-003b: material de research → staging, la curación T1
                    # decide la entrada a main (no bypass vía review queue).
                    corpus = _research_ingest_corpus() or MAIN_CORPUS
                    for item in items:
                        if (_time.time() - LAST_ACTIVITY["ts"] < RESEARCH_REVIEW_IDLE
                                or CHAT_BUSY["flag"]):
                            break
                        CHAT_BUSY["flag"] = True  # generator no thread-safe
                        try:
                            verdict = review_doc_with_llm(provider, item)
                        finally:
                            CHAT_BUSY["flag"] = False
                        if verdict.get("error"):
                            store.mark(item["review_id"], "error",
                                       verdict["error"])
                            continue
                        if verdict.get("promote"):
                            try:
                                doc_id = ingest_reviewed_doc(
                                    corpus, landing, item,
                                    embedding_adapter=get_embedding_adapter(),
                                )
                                store.mark(item["review_id"], "promoted",
                                           verdict.get("reason", ""),
                                           document_id=doc_id)
                                # Backlog residual → drain (PM-004: la
                                # promoción defiere sin vectores).
                                from ipa.agent.system_tools import _spawn_embed_drain
                                _spawn_embed_drain(corpus)
                                print(f"[research-review] promoted "
                                      f"{item['url']}: {verdict.get('reason','')}",
                                      flush=True)
                            except Exception as exc:
                                store.mark(item["review_id"], "error",
                                           str(exc)[:200])
                        else:
                            store.mark(item["review_id"], "discarded",
                                       verdict.get("reason", ""))
                            print(f"[research-review] discarded "
                                  f"{item['url']}: {verdict.get('reason','')}",
                                  flush=True)
                finally:
                    store.close()
            except Exception:
                pass

    _review_thread = _threading.Thread(target=_research_review_worker, daemon=True)
    _review_thread.start()

    # Idle enrichment worker: two levels of background topic enrichment.
    #
    # Level 1 (every 5 min idle): full discover_topics() on all unclustered
    #   docs + heuristic curation + topic continuity + deterministic grouping.
    #   100% deterministic — reuses BGE-M3 embeddings, no LLM, no VRAM.
    #
    # Level 2 (after 30 min continuous idle, if IPA_IDLE_DEEP_ENRICHMENT=1):
    #   loads the LLM for rich topic labels + LLM classification of gray docs
    #   + LLM topic grouping. Unloads the model when done so VRAM returns
    #   to the chat path. Aborts gracefully if the user returns mid-run.
    #
    # Processes the main corpus and the active reporter corpus against the
    # same TopicClusterStore derived (topics cross the document space).
    def _process_state(name: str) -> dict[str, Any]:
        """Read a single process state from the process_state directory."""
        state_dir = MAIN_CORPUS / "process_state"
        path = state_dir / f"{name}.json"
        return read_json(path, {"status": "not_started"})

    def _load_deep_engine(ilog) -> Any:
        """Motor del pase Tier 2 profundo.

        ExL3 (IPA_T2_ENGINE=exl3, default): batch 3-4, ctx 2048, MTP off (el
        guard del provider lo fuerza para batch > 2). Es el único punto del
        sistema que YA pagaba el costo de cargar un modelo desde frío — el
        switch se amortiza sobre toda la cola Tier 2, y el throughput agregado
        es ~2.4x (EXP-008 §8). Fallback a Ollama (factory) si ExL3 no carga.
        """
        engine = os.environ.get("IPA_T2_ENGINE", "exl3").strip().lower()
        if engine == "exl3":
            try:
                from ipa.providers.exl3_provider import ExL3Provider
                _ctx = int(os.environ.get("IPA_T2_CTX", "2048") or 2048)
                _bs = int(os.environ.get("IPA_T2_BATCH", "4") or 4)
                _root = ROOT / "models" / "Qwen3.5-9B-exl3-3.0bpw"
                provider = ExL3Provider(
                    model_path=str(_root),
                    model_id="Qwen3.5-9B-EXL3-3.0bpw",
                    quantization="EXL3-3.0bpw",
                    context_length=_ctx,
                    max_output_tokens=512,
                    temperature=0.1,
                    no_think=True,
                    batch_size=_bs,
                    use_mtp=True,  # el guard lo apaga para batch > 2
                    mtp_draft_tokens=2,
                    mtp_cache_tokens=_ctx,
                    cache_k_bits=8,
                    cache_v_bits=8,
                    suppress_cjk=True,
                    rep_p=1.15,
                )
                provider.load()
                # Marca de propiedad: si el chat llega con este motor cargado,
                # get_deep_dive_provider() lo descarga y monta el interactivo
                # (el usuario tiene prioridad sobre el trabajo de fondo).
                provider._t2_owned = True
                ilog(f"  [idle-sched T2] ExL3 listo (ctx {_ctx}, batch {_bs}, "
                     f"MTP {'on' if provider.use_mtp else 'off'})", flush=True)
                return provider
            except Exception as exc:
                ilog(f"  [idle-sched T2] ExL3 no disponible ({exc}) — fallback Ollama",
                     flush=True)
        from ipa.providers.factory import create_star_provider
        provider = create_star_provider(interactive=True)
        provider.load()
        # El fallback también es del pase: mismo marcador de prioridad al chat.
        provider._t2_owned = True
        return provider

    def _idle_enrichment_worker():
        import time as _time
        # El provider global se lee y se (des)carga desde este worker.
        global DEEP_DIVE_PROVIDER
        from ipa.agentic.topic_clusters import TopicClusterStore
        from ipa.agentic.idle_enrichment import (
            enrich_corpus_level1, enrich_corpus_level2,
        )

        # The dashboard runs under pythonw (no stdout) — worker logs would be
        # lost. Tee every message to a file so idle cycles are auditable.
        _ilog_path = ROOT / "outputs" / "web_dashboard" / "logs" / "idle_enrichment.log"

        def _ilog(msg: str, *_, **__) -> None:
            print(msg, flush=True)
            try:
                _ilog_path.parent.mkdir(parents=True, exist_ok=True)
                with open(_ilog_path, "a", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
            except OSError:
                pass

        def _idle() -> bool:
            # Perilla del sidebar: OFF corta TODO el enriquecimiento idle
            # (Tier 1 no arranca y las pasadas Tier 2 en vuelo abortan entre
            # items vía should_abort → _idle()).
            if not IDLE_ENABLED["enabled"]:
                return False
            # Tier 0 (fast_path) muta los corpora que T1/T2 leen y promueven.
            # El lease es cross-process: cubre también watchers huérfanos de
            # un dashboard anterior, que no figuran en JOBS. Mientras esté
            # activo el reset de LAST_ACTIVITY hace que la cuenta de idle
            # arranque recién cuando Tier 0 se libera.
            try:
                from ipa.agentic import tier0
                if tier0.active():
                    return False
            except Exception:
                pass
            # La ingesta interactiva (research_ingest) corre bajo heavy.lock:
            # mismo criterio — escribe el corpus canónico y T1/T2 no deben
            # evaluar estados parciales mientras dura.
            try:
                from ipa.agentic import heavy_lock
                if heavy_lock.holder() is not None:
                    return False
            except Exception:
                pass
            # La promoción idle escribe/purga los mismos corpora que el drain.
            # El lease cross-process evita que comiencen ciclos mientras un
            # embedding drain está activo (y viceversa).
            try:
                from ipa.agentic.embedding_maintenance import job_active
                if job_active(exclude_owner="idle_scheduler"):
                    return False
            except Exception:
                pass
            if CHAT_BUSY["flag"]:
                return False
            with JOBS_LOCK:
                if any(proc.poll() is None for proc in JOBS.values()):
                    return False
            if _process_state("pipeline").get("status") == "running":
                return False
            if _process_state("reporter").get("status") == "running":
                return False
            return True

        # ── Registro de tareas del scheduler idle ───────────────────────
        # Cada tarea declara tier, prioridad, recursos y cooldown. El
        # scheduler ordena, paraleliza lo que no comparte recursos y
        # serializa lo que sí (locks nombrados).
        from ipa.agentic.idle_scheduler import (
            CycleContext, IdleScheduler, IdleTask,
            RES_AGENT_DB, RES_CLUSTER_STORE, RES_CORPUS_MAIN,
            RES_CORPUS_REPORTER, RES_CORPUS_RESEARCH, RES_EMBEDDINGS,
            RES_LLM, RES_SKILLS, RES_STRATEGIC, RES_UNCERTAINTY,
            RES_USER_MODEL,
        )
        from ipa.agentic.idle_cognition import (
            detect_skills, infer_user_model, load_episode_dicts, load_task_dicts,
            reflect_principles, scan_research_agenda,
        )

        def _reporter_corpus() -> Path:
            return REPORTER_ROOT / "quality-check" / active_reporter_output().name / "corpus"

        def _research_staging_corpus() -> Path:
            # Staging propio de la research (DEC-003): path fijo del dominio
            # agente — no el puntero del reporter, que se mueve entre corridas
            # y el cleanup puede borrar.
            from ipa.agent.system_tools import _research_staging_dir
            return _research_staging_dir()

        # --- Higiene / memoria ---
        def _t_hygiene(ctx):
            """Cerrar sesiones 'active' huérfanas (restart del dashboard)."""
            from ipa.agent import AgentMemory as _AM
            _m = _AM()
            try:
                return {"sessions_closed": _m.close_stale_active_sessions(idle_minutes=30)}
            finally:
                _m.close()

        def _t_consolidate(ctx):
            """Resumir sesiones idle. Usa el LLM solo si ya está cargado."""
            from ipa.agent.session_consolidator import run_idle_consolidation
            results = run_idle_consolidation(ctx.provider, idle_minutes=5, max_sessions=3)
            for r in results:
                extra = f" · {len(r['fact_proposals'])} hechos propuestos" if r.get("fact_proposals") else ""
                _ilog(f"  [consolidacion] sesion ...{r['session_id'][-12:]} resumida{extra}", flush=True)
            return {"sessions_summarized": len(results)}

        # --- Topificación / promoción ---
        def _t_topify(corpus_getter, label):
            """Provenance backfill + clustering + curación + continuidad."""
            def _run(ctx):
                if ctx.should_abort():
                    return {"skipped": "aborted"}
                corpus = corpus_getter()
                if not corpus or not (corpus / "document_store.db").exists():
                    return {"skipped": "no corpus"}
                cluster_store = TopicClusterStore()
                try:
                    # Gate "corpus changed": sin dirty flag, con conteo
                    # estable, cobertura completa y sin gaps de provenance
                    # → no hay trabajo nuevo; se salta el scan O(corpus)
                    # (docs fetichados + sha256 + embeddings de main).
                    dirty_key = f"dirty:{corpus.resolve()}"
                    count_key = f"last_count:{corpus.resolve()}"
                    if cluster_store.get_meta(dirty_key) != "1":
                        from ipa.storage.document_store import DocumentStore as _DS
                        _ds = _DS(corpus / "document_store.db")
                        try:
                            live_count = _ds.count_documents()
                            live_ids = set(_ds.all_centroids())
                            source_ids = set(_ds.all_sources())
                        finally:
                            _ds.close()
                        covered = (
                            live_ids
                            <= cluster_store.processed_doc_ids(stage="clustered")
                            and live_ids
                            <= cluster_store.processed_doc_ids(stage="curated"))
                        stable = str(live_count) == (
                            cluster_store.get_meta(count_key) or "")
                        provenance_gap = bool(live_ids - source_ids)
                        if covered and stable and not provenance_gap:
                            return {"skipped": "corpus unchanged",
                                    "corpus": corpus.name}
                    _ilog(f"  [idle-sched T1] topify {label}: starting {corpus.name}...",
                          flush=True)
                    # Provenance self-heal before policy evaluation: scraper docs
                    # without document_sources would be "unknown provenance"
                    # forever. Tier 0 ya registra provenance al ingerir — esto
                    # queda como repair path para corpus legacy/incompletos.
                    try:
                        from ipa.ingestion.provenance import backfill_from_landing_registry
                        from ipa.storage.document_store import DocumentStore as _DS2
                        _ps = _DS2(corpus / "document_store.db")
                        try:
                            _pn = backfill_from_landing_registry(
                                _ps, corpus / "landing.db", ROOT / "Landing")
                            if _pn:
                                _ilog(f"  [idle-sched T1] provenance backfill: {_pn} docs", flush=True)
                        finally:
                            _ps.close()
                    except Exception:
                        pass
                    result = enrich_corpus_level1(corpus, cluster_store, main_corpus_path=MAIN_CORPUS)
                    # Corrida completada → limpiar el gate. Si crashea a mitad,
                    # el dirty flag queda y el próximo ciclo reintenta.
                    try:
                        from ipa.storage.document_store import DocumentStore as _DS3
                        _ds3 = _DS3(corpus / "document_store.db")
                        try:
                            cluster_store.set_meta(count_key, str(_ds3.count_documents()))
                        finally:
                            _ds3.close()
                        cluster_store.set_meta(dirty_key, "0")
                    except Exception:
                        pass
                    return {
                        "corpus": corpus.name,
                        "topics_new": result.get("topics_new", 0),
                        "parents_new": result.get("parents_new", 0),
                        "curated": result.get("curated", 0),
                        "promoted": result.get("promoted", 0),
                        "continuity": result.get("continuity_links", 0),
                    }
                finally:
                    cluster_store.close()
            return _run

        def _t_promotion(ctx):
            """Procesar la cola de promoción + sweep de Landing."""
            from ipa.agentic.promotion_executor import process_promotion_queue
            cluster_store = TopicClusterStore()
            try:
                promo = process_promotion_queue(cluster_store, _reporter_corpus(), MAIN_CORPUS)
            finally:
                cluster_store.close()
            if promo.get("processed", 0) <= 0:
                # Nothing promoted — surface deferrals (PM-004 vector
                # preflight) so the idle log explains the pending queue.
                deferred = promo.get("deferred_docs", 0)
                return {"deferred_docs": deferred} if deferred else {}
            _sw = run_landing_sweep()
            return {
                "processed": promo["processed"],
                "promoted_docs": promo.get("promoted_docs", 0),
                "promoted_chunks": promo.get("promoted_chunks", 0),
                "deferred_docs": promo.get("deferred_docs", 0),
                "archived": _sw.get("archived", 0),
                "deleted": _sw.get("deleted", 0),
            }

        # --- Cognitivo Tier 1: una tarea por store → corren en paralelo ---
        def _t_cog_user_model(ctx):
            return infer_user_model(ctx.episodes, ctx.tasks)

        def _t_cog_skills(ctx):
            return detect_skills(ctx.episodes)

        def _t_cog_principles(ctx):
            return reflect_principles(ctx.episodes)

        def _t_cog_agenda(ctx):
            return scan_research_agenda()

        # --- Tier 2 (LLM): serializado, preemptible entre items ---
        def _t_deep_topify(corpus_getter, label):
            def _run(ctx):
                if ctx.should_abort():
                    return {"skipped": "aborted"}
                # Trabajo batch por diseño: en Ollama serial cada doc cuesta
                # ~25 s y acapara el motor del chat durante horas sobre el
                # corpus completo. Solo corre con un provider que implemente
                # generate_chat_batch (ExL3) — el split de motores de EXP-008.
                from ipa.agentic.batch_llm import supports_batch
                if not supports_batch(ctx.provider):
                    return {"skipped": "requiere provider con batch (ExL3)"}
                cluster_store = TopicClusterStore()
                try:
                    result = enrich_corpus_level2(
                        corpus_getter(), cluster_store, ctx.provider,
                        is_busy=ctx.should_abort,
                    )
                finally:
                    cluster_store.close()
                return {
                    "corpus": label,
                    "labeled": result.get("labeled", 0),
                    "classified": result.get("classified", 0),
                    "gray_reviewed": result.get("gray_reviewed", 0),
                    "gray_rescued": result.get("gray_rescued", 0),
                    "aborted": result.get("aborted"),
                }
            return _run

        def _t_cog_principles_llm(ctx):
            # Split por longitud de salida: los principios usan ~500 tok
            # (max_new_tokens=500 en strategic_memory) — fuera del rango
            # cómodo del motor batch ExL3 (~300-400 tok con contexto largo,
            # EXP-008 §6). Se saltea en el pase ExL3 y queda para Ollama.
            engine = str(getattr(ctx.provider, "engine", "") or "")
            if engine == "exllamav3":
                return {"skipped": "salida larga (~500 tok): no apta para motor batch"}
            return reflect_principles(ctx.episodes, provider=ctx.provider)

        def _t_review_batch(ctx):
            """Veredictos de la cola de review en un pase batched (128 tok/ítem).

            Con ExL3 batch 3-4 el throughput agregado es ~2.4x el serial
            (EXP-008 §8). Reemplaza al worker de 60s de idle: el switch de
            motor no se amortiza por pocos docs, sí dentro del pase Tier 2.
            """
            from ipa.agent.research_review import (
                ResearchReviewStore, ingest_reviewed_doc, review_docs_with_llm,
            )
            from ipa.agent.system_tools import _research_ingest_corpus

            limit = int(os.environ.get("IPA_T2_REVIEW_LIMIT", "12") or 12)
            store = ResearchReviewStore()
            try:
                items = store.pending(limit=limit)
                if not items:
                    return {"pending": 0}
                if ctx.should_abort():
                    return {"skipped": "aborted"}
                # DEC-003b: material de research → staging, la curación T1
                # decide la entrada a main (no bypass vía review queue).
                corpus = _research_ingest_corpus() or MAIN_CORPUS
                # Dir propio del review (PM-004), no Landing/web compartido.
                landing = ROOT / "outputs" / "agent" / "research" / "review"
                CHAT_BUSY["flag"] = True  # el generator no es thread-safe
                try:
                    verdicts = review_docs_with_llm(ctx.provider, items)
                finally:
                    CHAT_BUSY["flag"] = False
                promoted = discarded = errors = 0
                for item, verdict in zip(items, verdicts):
                    if verdict.get("error"):
                        store.mark(item["review_id"], "error", verdict["error"])
                        errors += 1
                        continue
                    if verdict.get("promote"):
                        try:
                            doc_id = ingest_reviewed_doc(
                                corpus, landing, item,
                                embedding_adapter=get_embedding_adapter(),
                            )
                            store.mark(item["review_id"], "promoted",
                                       verdict.get("reason", ""), document_id=doc_id)
                            # Backlog residual → drain (PM-004: la promoción
                            # defiere sin vectores).
                            from ipa.agent.system_tools import _spawn_embed_drain
                            _spawn_embed_drain(corpus)
                            promoted += 1
                        except Exception as exc:
                            store.mark(item["review_id"], "error", str(exc)[:200])
                            errors += 1
                    else:
                        store.mark(item["review_id"], "discarded",
                                   verdict.get("reason", ""))
                        discarded += 1
                return {"promoted": promoted, "discarded": discarded, "errors": errors}
            finally:
                store.close()

        def _t_enrich_chunks(ctx):
            """Summary + queries sintéticas por chunk + re-embed en LanceDB.

            Recupera el trabajo del worker deprecado del Orchestrator
            (run_enrichment_exl3.py, era 4B): ahora corre sobre el 9B ya
            cargado del pase — el switch de motor ya está pagado. Incremental:
            los chunks ya enriquecidos ([Summary]) se saltean y el checkpoint
            embedding_status recupera corridas cortadas.
            """
            from ipa.agentic.chunk_enrichment import count_pending, enrich_chunks
            from ipa.indexes.lancedb_index import LanceDBIndex

            store_db = MAIN_CORPUS / "document_store.db"
            if not store_db.exists():
                return {"skipped": "sin document_store"}
            limit = int(os.environ.get("IPA_T2_ENRICH_LIMIT", "60") or 60)
            # El worker original usaba min_chars=800 (corpus de chunks
            # grandes). El corpus actual es uniforme ~512 chars — con 800 el
            # filtro no selecciona nada. Con ~400 la densidad discrimina
            # (~3% del corpus, medido 2026-09).
            min_chars = int(os.environ.get("IPA_T2_ENRICH_MIN_CHARS", "400") or 400)
            if count_pending(store_db, min_chars=min_chars) <= 0:
                return {"pending": 0}
            if ctx.should_abort():
                return {"skipped": "aborted"}
            embedding = get_embedding_adapter()  # CPU — no compite con ExL3
            lancedb = LanceDBIndex(MAIN_CORPUS / "vector" / "lancedb",
                                   vector_dim=embedding.dimension)
            bm25 = None
            _bm25_db = MAIN_CORPUS / "bm25_index.db"
            if _bm25_db.exists():
                from ipa.indexes.bm25_index import BM25Index
                bm25 = BM25Index(_bm25_db)  # win léxico de EXP-001 (+14.3%)
            try:
                return enrich_chunks(
                    ctx.provider, store_db,
                    lancedb=lancedb, embedding=embedding, bm25=bm25,
                    limit=limit, min_chars=min_chars,
                    should_abort=ctx.should_abort, log=_ilog)
            finally:
                lancedb.close()
                if bm25 is not None:
                    bm25.close()

        def _t_index_audit(ctx):
            """Auditoría de salud del corpus (read-only) → index_health.json.

            Capa lógica cada corrida (consume señales Tier 0, barata); capa
            física solo si su checked_at supera el cooldown largo — el drift
            por kills/locks no setea dirty flags, así que es forzada, no
            gated por cambios.
            """
            from ipa.agentic import tier0
            if tier0.active():
                return {"skipped": "tier0 lease activo — índices parciales"}
            from ipa.agentic.index_audit import (
                DEFAULT_OUTPUT, run_index_audit)
            import json as _json
            phys_hours = float(os.environ.get(
                "IPA_AUDIT_PHYSICAL_HOURS", "6") or 6)
            run_physical = True
            try:
                prev = _json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
                last = (prev.get("physical") or {}).get("checked_at")
                if last:
                    from datetime import datetime, timezone
                    age = (datetime.now(timezone.utc) - datetime.fromisoformat(
                        last.replace("Z", "+00:00"))).total_seconds()
                    run_physical = age >= phys_hours * 3600
            except Exception:
                run_physical = True  # sin previo o JSON roto → forzar física
            result = run_index_audit(
                MAIN_CORPUS, layer="both" if run_physical else "logical",
                cluster_db=ROOT / "outputs" / "agent" / "topic_clusters.db")
            logical = result.get("logical") or {}
            physical = result.get("physical") or {}
            return {
                "status": result.get("status"),
                "physical_run": run_physical,
                "empty_docs": logical.get("empty_docs_count", 0),
                "dup_hashes": logical.get("dup_hashes_count", 0),
                "dup_flag_leaks": logical.get("dup_flag_leaks_count", 0),
                "missing_meta": logical.get("missing_meta_count", 0),
                "bm25_missing": (physical.get("bm25") or {}).get("meta_missing_count", 0),
                "lance_missing": (physical.get("lance") or {}).get("missing_count", 0),
                "spam_hashes": (physical.get("spam_chunks") or {}).get("same_site_hashes", 0),
            }

        _scheduler = IdleScheduler([
            # Tier 1 — CPU/IO, sin VRAM. Prioridad: higiene → memoria →
            # topificación → promoción → cognitivo → auditoría.
            IdleTask("hygiene_sessions", 1, 10, _t_hygiene,
                     resources=frozenset({RES_AGENT_DB}), cooldown_seconds=120),
            IdleTask("consolidate_sessions", 1, 20, _t_consolidate,
                     resources=frozenset({RES_AGENT_DB, RES_LLM}),
                     cooldown_seconds=90, needs_llm=True),
            IdleTask("topify_main", 1, 30, _t_topify(lambda: MAIN_CORPUS, "main"),
                     resources=frozenset({RES_CLUSTER_STORE, RES_EMBEDDINGS, RES_CORPUS_MAIN}),
                     cooldown_seconds=300),
            IdleTask("topify_reporter", 1, 31, _t_topify(_reporter_corpus, "reporter"),
                     resources=frozenset({RES_CLUSTER_STORE, RES_EMBEDDINGS, RES_CORPUS_REPORTER}),
                     cooldown_seconds=300),
            IdleTask("topify_research_staging", 1, 32,
                     _t_topify(_research_staging_corpus, "research_staging"),
                     resources=frozenset({RES_CLUSTER_STORE, RES_EMBEDDINGS, RES_CORPUS_RESEARCH}),
                     cooldown_seconds=300),
            IdleTask("promotion_queue", 1, 40, _t_promotion,
                     resources=frozenset({RES_CLUSTER_STORE, RES_CORPUS_MAIN,
                                          RES_CORPUS_REPORTER, RES_CORPUS_RESEARCH,
                                          RES_EMBEDDINGS}),
                     cooldown_seconds=300),
            IdleTask("cog_user_model", 1, 50, _t_cog_user_model,
                     resources=frozenset({RES_USER_MODEL}), cooldown_seconds=300),
            IdleTask("cog_skills", 1, 51, _t_cog_skills,
                     resources=frozenset({RES_SKILLS}), cooldown_seconds=300),
            IdleTask("cog_principles", 1, 52, _t_cog_principles,
                     resources=frozenset({RES_STRATEGIC}), cooldown_seconds=300),
            IdleTask("cog_agenda", 1, 53, _t_cog_agenda,
                     resources=frozenset({RES_UNCERTAINTY}), cooldown_seconds=300),
            # Auditoría read-only: capa lógica barata por ciclo, física cada
            # IPA_AUDIT_PHYSICAL_HOURS (default 6h). Última: usa los stores
            # que topify/promoción escriben — mejor que vea estado estable.
            IdleTask("index_audit", 1, 60, _t_index_audit,
                     resources=frozenset({RES_CLUSTER_STORE, RES_CORPUS_MAIN}),
                     cooldown_seconds=int(os.environ.get(
                         "IPA_AUDIT_LOGICAL_SECONDS", "900") or 900)),
            # Tier 2 — LLM (un solo pase serial)
            IdleTask("deep_topify_main", 2, 10, _t_deep_topify(lambda: MAIN_CORPUS, "main"),
                     resources=frozenset({RES_LLM, RES_CLUSTER_STORE, RES_CORPUS_MAIN}),
                     cooldown_seconds=300, needs_llm=True),
            IdleTask("deep_topify_reporter", 2, 11, _t_deep_topify(_reporter_corpus, "reporter"),
                     resources=frozenset({RES_LLM, RES_CLUSTER_STORE, RES_CORPUS_REPORTER}),
                     cooldown_seconds=300, needs_llm=True),
            IdleTask("deep_topify_research_staging", 2, 12,
                     _t_deep_topify(_research_staging_corpus, "research_staging"),
                     resources=frozenset({RES_LLM, RES_CLUSTER_STORE, RES_CORPUS_RESEARCH}),
                     cooldown_seconds=300, needs_llm=True),
            # Review batched: veredictos de 128 tok — el caso de uso ideal del
            # motor batch. Prioridad 15: después de los labels, antes de los
            # principios (que se saltean en el pase ExL3).
            IdleTask("review_batch", 2, 15, _t_review_batch,
                     resources=frozenset({RES_LLM, RES_AGENT_DB}),
                     cooldown_seconds=300, needs_llm=True),
            IdleTask("cog_principles_llm", 2, 20, _t_cog_principles_llm,
                     resources=frozenset({RES_LLM, RES_STRATEGIC}),
                     cooldown_seconds=300, needs_llm=True),
            # Enrichment por chunk (ex-job "enrichment" del Orchestrator):
            # summary + queries sintéticas + re-embed. Última del pase: es la
            # de mayor volumen — las rápidas (labels, veredictos) se completan
            # aunque el usuario vuelva a mitad del pase. RES_EMBEDDINGS porque
            # re-embede con BGE-M3 (CPU, no compite con ExL3 en VRAM).
            IdleTask("enrich_chunks", 2, 25, _t_enrich_chunks,
                     resources=frozenset({RES_LLM, RES_EMBEDDINGS, RES_CORPUS_MAIN}),
                     cooldown_seconds=300, needs_llm=True),
        ], max_tier1_workers=3)

        def _log_outcomes(outcomes, tier):
            for o in outcomes:
                if o.skipped:
                    continue
                if not o.ok:
                    _ilog(f"  [idle-sched {tier}] {o.name} FALLÓ: {o.error}", flush=True)
                    continue
                interesting = {k: v for k, v in (o.result or {}).items() if v}
                if interesting:
                    _ilog(f"  [idle-sched {tier}] {o.name}: {interesting} ({o.duration_s:.1f}s)", flush=True)

        level2_done = False  # prevent Tier 2 from re-running every cycle
        deep_done = False    # el pase profundo (motor ExL3) corre una vez por ventana

        while True:
            _time.sleep(60)
            try:
                if not _idle():
                    LAST_ACTIVITY["ts"] = time.time()
                    level2_done = False  # reset so Tier 2 can run next idle
                    deep_done = False    # idem para el pase profundo
                    continue

                idle_mins = (time.time() - LAST_ACTIVITY["ts"]) / 60.0
                # Lease cross-process: el drain de embeddings y este ciclo
                # mutan los mismos stores/indexes y deben ir serializados.
                from ipa.agentic.embedding_maintenance import claim_job, release_job
                if not claim_job("idle_scheduler"):
                    _ilog("  [idle-sched] embedding/index maintenance busy — skipping", flush=True)
                    continue
                # Lock de instancia: dos dashboards compartiendo store no
                # pueden enriquecer a la vez.
                if not ENRICHMENT_LOCK.acquire(blocking=False):
                    release_job("idle_scheduler")
                    _ilog(f"  [idle-sched] lock busy — skipping", flush=True)
                    continue
                try:
                    _ilog(f"  [idle-sched] idle {idle_mins:.1f} min — ciclo Tier 1", flush=True)
                    ctx = CycleContext(
                        idle_minutes=idle_mins,
                        provider=DEEP_DIVE_PROVIDER,
                        episodes=load_episode_dicts(),
                        tasks=load_task_dicts(),
                        should_abort=lambda: not _idle(),
                        log=_ilog,
                    )
                    _log_outcomes(_scheduler.run_tier1(ctx), "T1")

                    # --- Tier 2 (LLM) ---
                    # Dos disparadores con motores distintos:
                    #  (a) pase profundo (idle >= IPA_IDLE_DEEP_THRESHOLD): usa
                    #      SU PROPIO motor — ExL3 batch (IPA_T2_ENGINE=exl3).
                    #      Si el chat dejó Ollama warm, se descarga: en 6 GB
                    #      no conviven y el pase batch rinde ~2.4x (EXP-008).
                    #      Marcado _t2_owned: si el usuario vuelve, el chat lo
                    #      descarga y monta el interactivo (prioridad al chat).
                    #  (b) modelo ya cargado (>= IPA_IDLE_LLM_LOADED_THRESHOLD):
                    #      aprovecha el provider del chat sin cargar nada.
                    _loaded = (DEEP_DIVE_PROVIDER is not None
                               and DEEP_DIVE_PROVIDER.is_loaded())
                    _ours_loaded = (_loaded and bool(
                        getattr(DEEP_DIVE_PROVIDER, "_t2_owned", False)))
                    _t2_deep_due = (IDLE_DEEP_ENABLED and not deep_done
                                    and idle_mins >= IDLE_DEEP_THRESHOLD
                                    and not _ours_loaded)
                    _t2_loaded_due = (IDLE_LLM_LOADED_ENABLED and not level2_done
                                      and _loaded and not _ours_loaded
                                      and idle_mins >= IDLE_LLM_LOADED_THRESHOLD)
                    if _t2_deep_due or _t2_loaded_due:
                        _ilog(
                            f"  [idle-sched] idle {idle_mins:.1f} min — Tier 2 "
                            f"(LLM, {'profundo' if _t2_deep_due else 'modelo ya cargado'})",
                            flush=True,
                        )
                        we_loaded = False
                        _our_provider = None
                        if _t2_deep_due:
                            # El pase profundo corre con su motor: si hay otro
                            # provider cargado (Ollama warm del chat), se baja —
                            # ExL3 necesita la VRAM y el lock es de un solo dueño.
                            if _loaded:
                                try:
                                    DEEP_DIVE_PROVIDER.unload()
                                except Exception:
                                    pass
                                DEEP_DIVE_PROVIDER = None
                            _ilog(f"  [idle-sched T2] loading LLM provider...", flush=True)
                            try:
                                _our_provider = _load_deep_engine(_ilog)
                                DEEP_DIVE_PROVIDER = _our_provider
                                we_loaded = True
                            except Exception as exc:
                                _ilog(f"  [idle-sched T2] LLM load failed: {exc}", flush=True)
                                deep_done = True
                                level2_done = True
                                continue
                        try:
                            ctx2 = CycleContext(
                                idle_minutes=idle_mins,
                                provider=DEEP_DIVE_PROVIDER,
                                episodes=ctx.episodes,
                                tasks=ctx.tasks,
                                should_abort=lambda: not _idle(),
                                log=_ilog,
                            )
                            _log_outcomes(_scheduler.run_tier2(ctx2), "T2")
                        finally:
                            # Descargar solo si lo cargamos nosotros Y la
                            # instancia sigue siendo la nuestra: si el chat
                            # llegó a mitad del pase ya descargó el motor
                            # batch (_t2_owned) y montó el suyo — no hay
                            # que tocar el provider del chat.
                            if we_loaded and DEEP_DIVE_PROVIDER is _our_provider:
                                try:
                                    with DEEP_DIVE_LOCK:
                                        if DEEP_DIVE_PROVIDER is _our_provider:
                                            DEEP_DIVE_PROVIDER.unload()
                                            DEEP_DIVE_PROVIDER = None
                                except Exception:
                                    pass
                            level2_done = True
                            if _t2_deep_due:
                                deep_done = True
                            # Re-warm del motor interactivo: si el sistema sigue
                            # idle, recargar Ollama para que el próximo chat no
                            # pague el reload del GGUF (~10 s de TTFT extra).
                            if we_loaded and _idle():
                                try:
                                    get_deep_dive_provider()
                                    _ilog("  [idle-sched T2] motor interactivo re-cargado (chat listo)", flush=True)
                                except Exception as _rw_exc:
                                    _ilog(f"  [idle-sched T2] re-warm falló: {_rw_exc}", flush=True)
                finally:
                    ENRICHMENT_LOCK.release()
                    release_job("idle_scheduler")
            except Exception as exc:
                # Don't let the worker die on errors
                _ilog(f"  [idle-sched] error: {exc}", flush=True)
                continue

    _idle_enrichment_thread = _threading.Thread(target=_idle_enrichment_worker, daemon=True)

    _idle_enrichment_thread.start()

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

