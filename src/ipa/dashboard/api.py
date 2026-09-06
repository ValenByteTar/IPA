"""HTTP API handler for the IPA dashboard.

The handler is loaded by :mod:`ipa.dashboard.server` after its shared dashboard
context has been initialized.
"""
from __future__ import annotations

from . import server as _server

# The handler historically referenced dashboard context by module-global name.
# Copying the initialized context keeps the HTTP surface isolated while the
# remaining state/services are migrated incrementally.
globals().update({name: value for name, value in vars(_server).items() if not name.startswith("__")})

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


