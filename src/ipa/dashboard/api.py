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

# Retrieval infrastructure: a single persistent worker thread runs corpus
# searches so the LanceDBIndex handles can be cached and reused across
# requests. DocumentStore (sqlite) is NOT cached: sqlite connections are
# thread-bound and the tutor path calls retrieval from the request thread,
# not the pool — a cached connection would cross threads. It is opened and
# closed per call instead (~1ms, negligible vs. embedding+search).
import concurrent.futures as _cf
import os
import re as _re_module
import time as _time_module

_RETRIEVAL_POOL = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="retrieval")
_RETRIEVAL_STORES: dict[str, object] = {}


class _TTLCache:
    """Cache TTL+LSU mínimo para resultados de retrieval (query → hits).

    Repetir la misma pregunta (o re-consultar en turnos seguidos) pagaba
    LanceDB + rerank completos. TTL corto (IPA_RETRIEVAL_CACHE_TTL, default
    300s) acota la staleness frente a documentos nuevos; el embedding ya
    tiene su propio cache. IPA_RETRIEVAL_CACHE_SIZE=0 lo desactiva.
    """

    def __init__(self, maxsize: int, ttl_s: float) -> None:
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._data: dict[str, tuple[float, object]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        if self.maxsize <= 0:
            return None
        item = self._data.get(key)
        if item is None:
            self.misses += 1
            return None
        ts, value = item
        if _time_module.time() - ts > self.ttl_s:
            self._data.pop(key, None)
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, key: str, value) -> None:
        if self.maxsize <= 0:
            return
        self._data[key] = (_time_module.time(), value)
        while len(self._data) > self.maxsize:
            oldest = min(self._data, key=lambda k: self._data[k][0])
            self._data.pop(oldest, None)

    def stats(self) -> dict[str, float]:
        return {"size": len(self._data), "maxsize": self.maxsize,
                "ttl_s": self.ttl_s, "hits": self.hits, "misses": self.misses}


_RETRIEVAL_CACHE = _TTLCache(
    maxsize=int(os.environ.get("IPA_RETRIEVAL_CACHE_SIZE", "64") or 64),
    ttl_s=float(os.environ.get("IPA_RETRIEVAL_CACHE_TTL", "300") or 300),
)

# Cache de respuestas del chat: opt-in (IPA_RESPONSE_CACHE=1, default OFF).
# Match exacto normalizado por (mensaje, rol, sesión) — dedup de repetidos,
# no similitud semántica (servir una respuesta "parecida" es incorrecto).
_RESPONSE_CACHE_ON = os.environ.get("IPA_RESPONSE_CACHE", "0").strip().lower() not in (
    "0", "false", "no", "off", "")
_RESPONSE_CACHE = _TTLCache(
    maxsize=int(os.environ.get("IPA_RESPONSE_CACHE_SIZE", "32") or 32),
    ttl_s=float(os.environ.get("IPA_RESPONSE_CACHE_TTL", "120") or 120),
)


def _response_cache_key(message: str, role: str, session_id: str) -> str:
    norm = " ".join(message.lower().split())
    return f"{role}\x00{session_id}\x00{norm}"


def _retrieval_lance(corpus):
    """LanceDBIndex cacheado por corpus (thread-safe para lecturas).

    Nunca cachear DocumentStore aquí: sqlite3.Connection es thread-bound.
    """
    key = str(corpus)
    lance = _RETRIEVAL_STORES.get(key)
    if lance is None:
        from ipa.indexes.lancedb_index import LanceDBIndex
        lance = LanceDBIndex(corpus / "vector" / "lancedb")
        _RETRIEVAL_STORES[key] = lance
    return lance

# ── Respuesta del modelo: limpieza determinística ─────────────────────────
# Citas [n]: no son clicables en el chat, así que se remueven del texto
# visible (las fuentes viven en el panel "Fuentes").
_CITE_PATTERN = _re_module.compile(r"\s*\[\d{1,2}\]")
# Filtro de emojis y símbolos decorativos: el 9B degradado genera secuencias
# de emojis aunque el prompt lo prohíba.
_EMOJI_PATTERN = _re_module.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U0001F7E0-\U0001F7FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F"
    "\U00002B00-\U00002BFF"
    "\U0000200D"
    "\U00002300-\U000023FF"
    "\U00002A00-\U00002AFF"
    "]+",
    flags=_re_module.UNICODE,
)
_NON_LATIN_PATTERN = _re_module.compile(
    "[\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF"
    "\u0600-\u06FF\u0900-\u097F]",
    flags=_re_module.UNICODE,
)
# Prefijos de relleno que el 9B emite al arrancar una respuesta (material
# extraído de AgenticRAG answer_postprocessor). Se recortan del texto visible.
_FILLER_PREFIXES = (
    "Entendido. ", "Entendido, ", "Entendido — ", "Entendido —",
    "Entendido: ", "Entendido:",
    "Claro, ", "Claro. ",
    "Por supuesto, ", "Por supuesto. ", "Desde luego, ",
    "Análisis: ", "Respuesta final: ",
    "Basándome en los documentos: ", "Según los documentos: ",
)


def _strip_emojis(text: str) -> str:
    text = _EMOJI_PATTERN.sub("", text)
    text = _NON_LATIN_PATTERN.sub("", text)
    text = _re_module.sub(r"  +", " ", text)
    return text


def _strip_filler(text: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _FILLER_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip()
                changed = True
    return text


def _clean_model_output(text: str) -> str:
    for stop in ("<|im_end|>", "</s>", "<|im_start|>"):
        if stop in text:
            text = text.split(stop)[0]
    # Some models emit a stray closing think tag; drop it and anything
    # after (the model sometimes repeats its answer).
    if "</think>" in text:
        text = text.split("</think>")[0]
    return text.strip()


def _truncate_at_sentence(text: str, limit: int) -> str:
    """Cut text at the last sentence boundary before `limit` chars."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in (". ", "! ", "? ", ".\n", "\n\n"):
        idx = cut.rfind(sep)
        if idx > limit // 2:
            return cut[: idx + 1].rstrip()
    return cut.rstrip() + "…"


def tutor_roadmaps_payload() -> dict[str, Any]:
    """Roadmaps with per-unit progress for the dashboard stepper.

    Unit status lives in the additive unit_progress table (pending/current/
    done); titles resolve to source domains via the main corpus DocumentStore
    — raw concept/doc ids never reach the UI.
    """
    from ipa.tutor.tutor_runtime import TutorStore
    from ipa.tutor.tutor_contracts import RoadmapStatus

    visible = {
        RoadmapStatus.PROPOSED, RoadmapStatus.APPROVED,
        RoadmapStatus.ACTIVE, RoadmapStatus.COMPLETED,
        RoadmapStatus.REJECTED,
    }
    doc_store = None
    try:
        from ipa.agent.system_tools import _main_corpus_dir
        from ipa.storage.document_store import DocumentStore
        corpus = _main_corpus_dir()
        if corpus and (corpus / "document_store.db").exists():
            doc_store = DocumentStore(corpus / "document_store.db")
    except Exception:
        doc_store = None

    def _unit_title(concept_id: str) -> str:
        if doc_store is not None:
            try:
                src = doc_store.get_source(concept_id) or {}
                title = src.get("source_domain") or src.get("title")
                if title:
                    return str(title)
            except Exception:
                pass
        return "Documento del corpus"

    roadmaps: list[dict[str, Any]] = []
    store = TutorStore()
    try:
        for rm in store.list_roadmaps():
            if rm.status not in visible:
                continue
            topic_id = rm.goal_id.removeprefix("goal:")
            statuses = store.unit_statuses(rm.roadmap_id)
            # Roadmaps activados antes de unit_progress: mostrar la unidad 1
            # como actual sin escribir (GET no muta; el driver seedea en el
            # próximo turno de lección).
            if not statuses and rm.status in (RoadmapStatus.ACTIVE, RoadmapStatus.COMPLETED):
                statuses = {1: "current"}
            record = store.get_topic_record(topic_id)
            mastery = None
            if record is not None:
                mastery = {
                    "status": record.mastery_status.value,
                    "score": record.mastery_score,
                    "attempts": record.attempts,
                    "evidence_count": len(record.evidence_ids),
                }
            roadmaps.append({
                "roadmap_id": rm.roadmap_id,
                "topic": topic_id.replace("-", " "),
                "status": rm.status.value,
                "version": rm.version,
                "created_at": rm.created_at,
                "mastery": mastery,
                "units": [
                    {
                        "order": u.order,
                        "title": _unit_title(u.concept_id),
                        "reason": u.reason,
                        "minutes": u.estimated_effort_minutes,
                        "assessment_types": [
                            at.value if hasattr(at, "value") else str(at)
                            for at in u.assessment_types
                        ],
                        "status": statuses.get(u.order, "pending"),
                    }
                    for u in rm.units
                ],
            })
    finally:
        store.close()
    return {"roadmaps": roadmaps}


def _goal_payload(goal: Any) -> dict[str, Any]:
    """LearningGoal → JSON for the Roadmaps tab (the 'project' section)."""
    return {
        "goal_id": goal.goal_id,
        "title": goal.title,
        "description": goal.description,
        "status": goal.status.value,
        "success_criteria": list(goal.success_criteria),
        "constraints": list(goal.constraints),
        "created_at": goal.created_at,
        "updated_at": goal.updated_at,
        "approved": bool(goal.approval and goal.approval.approved),
    }


def tutor_projects_payload() -> dict[str, Any]:
    """Projects (LearningGoals) with their roadmaps — the Roadmaps tab
    selector. Legacy roadmaps without a goal row get one synthesized on read
    (goal_for_roadmap), so every roadmap belongs to a project."""
    from ipa.tutor.tutor_runtime import TutorStore, goal_for_roadmap

    store = TutorStore()
    try:
        projects: dict[str, dict[str, Any]] = {}
        for rm in store.list_roadmaps(include_archived=True):
            goal = goal_for_roadmap(store, rm)
            proj = projects.setdefault(rm.goal_id, {
                "goal": _goal_payload(goal), "roadmaps": [],
            })
            proj["roadmaps"].append({
                "roadmap_id": rm.roadmap_id,
                "version": rm.version,
                "status": rm.status.value,
                "created_at": rm.created_at,
                "n_units": len(rm.units),
            })
        return {"projects": sorted(
            projects.values(),
            key=lambda p: p["goal"]["updated_at"], reverse=True,
        )}
    finally:
        store.close()


def tool_catalog_payload() -> dict[str, Any]:
    """Registry specs for the MCP proxy — one source of truth, no drift."""
    from ipa.agent.system_tools import tool_specs
    return {"tools": tool_specs()}


def execute_tool_payload(name: str, args: dict[str, Any] | None) -> dict[str, Any]:
    """Unified tool frontier for external agents (MCP proxy): dispatches to
    the same registry the dashboard chat uses. Never loads models here —
    the tools open their own stores; LLM-backed tools (research) run in
    the dashboard's background executors."""
    from ipa.agent.system_tools import execute_system_tool, SYSTEM_TOOL_NAMES

    if not name:
        return {"ok": False, "error": "tool name required"}
    if name not in SYSTEM_TOOL_NAMES:
        return {"ok": False,
                "error": f"unknown tool: {name}; valid: {sorted(SYSTEM_TOOL_NAMES)}"}
    if args is not None and not isinstance(args, dict):
        return {"ok": False, "error": "args must be a JSON object"}
    try:
        result = execute_system_tool(name, args or {})
    except Exception as exc:
        return {"ok": False, "tool": name, "error": str(exc)}
    return {"ok": bool(result.ok), "tool": result.tool_name,
            "summary": result.summary, "data": result.data,
            **({"error": result.error} if result.error else {})}


def tutor_roadmap_context(roadmap_id: str) -> dict[str, Any]:
    """Full context for the Roadmaps tab: goal (objective), rationale,
    progress, re-derived concepts, focus state.

    Concepts re-derive from the canonical DocumentStore by concept_id
    (= document_id): source domain/url + a text excerpt. Honest labels —
    until concept curation improves these are source-material titles, not
    polished pedagogical names (surfaced, not hidden).
    """
    from ipa.tutor.tutor_contracts import RoadmapStatus
    from ipa.tutor.tutor_runtime import TutorStore, goal_for_roadmap

    doc_store = None
    try:
        from ipa.agent.system_tools import _main_corpus_dir
        from ipa.storage.document_store import DocumentStore
        corpus = _main_corpus_dir()
        if corpus and (corpus / "document_store.db").exists():
            doc_store = DocumentStore(corpus / "document_store.db")
    except Exception:
        doc_store = None

    def _concept(concept_id: str) -> dict[str, Any]:
        info = {"concept_id": concept_id, "title": "Documento del corpus",
                "excerpt": "", "source_url": None, "source_domain": None}
        if doc_store is not None:
            try:
                src = doc_store.get_source(concept_id) or {}
                info["source_domain"] = src.get("source_domain")
                info["source_url"] = src.get("source_url")
                if src.get("source_domain"):
                    info["title"] = str(src["source_domain"])
                doc = doc_store.get_document(concept_id)
                text = (getattr(doc, "text", "") or "").strip().replace("\n", " ")
                info["excerpt"] = text[:400]
            except Exception:
                pass
        return info

    store = TutorStore()
    try:
        rm = store.get_roadmap(roadmap_id)
        if rm is None:
            return {"ok": False, "error": f"unknown roadmap: {roadmap_id}"}
        goal = goal_for_roadmap(store, rm)
        statuses = store.unit_statuses(rm.roadmap_id)
        # Roadmaps activados antes de unit_progress: unidad 1 como actual
        # (GET no muta — mismo criterio que tutor_roadmaps_payload).
        if not statuses and rm.status in (RoadmapStatus.ACTIVE, RoadmapStatus.COMPLETED):
            statuses = {1: "current"}
        record = store.get_topic_record(rm.goal_id.removeprefix("goal:"))
        mastery = None
        if record is not None:
            mastery = {
                "status": record.mastery_status.value,
                "score": record.mastery_score,
                "attempts": record.attempts,
                "evidence_count": len(record.evidence_ids),
            }
        units = [
            {
                "order": u.order,
                "concept_id": u.concept_id,
                "reason": u.reason,
                "minutes": u.estimated_effort_minutes,
                "assessment_types": [
                    at.value if hasattr(at, "value") else str(at)
                    for at in u.assessment_types
                ],
                "status": statuses.get(u.order, "pending"),
                "concept": _concept(u.concept_id),
            }
            for u in rm.units
        ]
        done = sum(1 for s in statuses.values() if s == "done")
        current = next(
            (o for o, s in sorted(statuses.items()) if s == "current"), None
        )
        return {
            "ok": True,
            "roadmap_id": rm.roadmap_id,
            "goal_id": rm.goal_id,
            "version": rm.version,
            "status": rm.status.value,
            "created_at": rm.created_at,
            "is_focus": store.get_focus() == rm.roadmap_id,
            "goal": _goal_payload(goal),
            "rationale": {
                "assumptions": list(rm.assumptions),
                "uncertainties": list(rm.uncertainties),
                "change_reason": rm.change_reason,
            },
            "progress": {"done": done, "current": current, "total": len(rm.units)},
            "units": units,
            "mastery": mastery,
        }
    finally:
        store.close()


def parse_deep_dive_context(body: dict[str, Any]) -> dict[str, Any] | None:
    """Validate the deep-dive chat context ("Profundizar" from a report).

    Returns the context dict, None when absent, raises on invalid corpus.
    """
    ctx = str(body.get("context") or "").strip().lower()
    if ctx != "deep_dive":
        return None
    corpus = Path(str(body.get("corpus") or "")).expanduser().resolve()
    if REPORTER_ROOT.resolve() not in corpus.parents:
        raise PermissionError("deep dive solo puede usar corpus Reporter")
    return {
        "corpus": corpus,
        "category_id": body.get("category_id") or None,
        "search": str(body.get("search") or "").strip(),
    }


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

    def _sse_write(self, data: dict[str, Any]) -> None:
        """Write a Server-Sent Event to the response stream.

        Silently ignores ConnectionAbortedError — the client may close
        the connection (e.g. browser abort) while the server is still
        writing. The stream is dead anyway; no point crashing.
        """
        try:
            payload = json.dumps(data, ensure_ascii=False)
            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

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
                from ipa.reporter.reporter_deep_dive import deep_dive
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
                from ipa.reporter.reporter_deep_dive import deep_dive_prepare, deep_dive_stream, _clean_token
                corpus = Path(query.get("corpus", [""])[0]).expanduser().resolve()
                if REPORTER_ROOT.resolve() not in corpus.parents:
                    raise PermissionError("deep dive solo puede usar corpus Reporter")
                user_query = query.get("query", [""])[0]
                retrieval_query = query.get("search", [user_query])[0]
                document_ids = query.get("document_id", [])
                category_id = query.get("category_id", [None])[0]
                # Agent session (DEC-002): the deep dive runs inside an agent
                # session so the conversation lives in the shared store — the
                # dashboard never owns agent state. An existing session can be
                # continued via ?session_id=...; otherwise one is created.
                from ipa.agent import AgentMemory, load_identity
                agent_session_id = query.get("session_id", [None])[0]
                memory = AgentMemory()
                identity = load_identity()
                if agent_session_id and memory.get_session(agent_session_id) is not None:
                    session = memory.get_session(agent_session_id)
                else:
                    session = memory.open_session(
                        interface="dashboard", role="general",
                        identity_hash=identity.identity_hash,
                        title=f"deep dive: {user_query[:50]}",
                    )
                    agent_session_id = session.session_id
                memory.record_episode(
                    agent_session_id, turn_role="user", content=user_query,
                    identity_hash=identity.identity_hash,
                )
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
                sse_send("evidence", {"evidence": prep["evidence"], "chunks": prep["chunks_info"], "sufficient": prep["sufficient_evidence"], "session_id": agent_session_id})
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
                    from ipa.reporter.reporter_deep_dive import _clean_generated
                    cleaned = _clean_generated(full_answer)
                    if not cleaned:
                        cleaned = prep["fallback"]
                    # Validate claims
                    from ipa.reporter.reporter_claims import validate_claims, citation_summary
                    claims = validate_claims(cleaned, prep["evidence_texts"])
                    # Record the assistant turn in the shared agent memory
                    memory.record_episode(
                        agent_session_id, turn_role="assistant", content=cleaned,
                        identity_hash=identity.identity_hash,
                    )
                    sse_send("done", {"answer": cleaned, "claims": claims, "citation_summary": citation_summary(claims), "session_id": agent_session_id})
                else:
                    memory.record_episode(
                        agent_session_id, turn_role="assistant",
                        content=prep["fallback"], identity_hash=identity.identity_hash,
                    )
                    sse_send("done", {"answer": prep["fallback"], "claims": [], "citation_summary": {}, "session_id": agent_session_id})
                memory.close()
            except Exception as exc:
                if "memory" in locals():
                    try:
                        memory.close()
                    except Exception:
                        pass
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
        elif parsed.path == "/api/promotions":
            # Promotion queue — documents pending promotion to the main corpus.
            from ipa.agentic.topic_clusters import TopicClusterStore
            cluster_store = TopicClusterStore()
            try:
                pending = cluster_store.pending_promotions()
            finally:
                cluster_store.close()
            self.send_json({"pending": pending, "count": len(pending)})
        elif parsed.path == "/api/agent/sessions":
            # Agent surface: list sessions from the shared agent store (DEC-002).
            from ipa.agent import AgentMemory
            with AgentMemory() as memory:
                sessions = memory.list_sessions(limit=50)
                self.send_json({"sessions": [s.__dict__ for s in sessions]})
        elif parsed.path == "/api/idle/status":
            # Perilla del sidebar: estado del enriquecimiento idle (T1/T2).
            from . import server as _server_mod
            self.send_json({"enabled": _server_mod.idle_enabled_state()})
        elif parsed.path == "/api/tutor/focus":
            # Indicador del chat: sobre qué roadmap está el agente en esta
            # sesión (o el foco global si la sesión aún no adoptó ninguno).
            parsed_query = urllib.parse.parse_qs(parsed.query)
            _fsid = parsed_query.get("session_id", [""])[0]
            try:
                from ipa.tutor.tutor_chat import get_tutor_driver
                self.send_json(get_tutor_driver().get_focus(_fsid))
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 500)
        elif parsed.path == "/api/tools/catalog":
            # Frontera MCP: specs del registry para que el proxy genere tools.
            self.send_json(tool_catalog_payload())
        elif parsed.path == "/api/agent/session":
            query = urllib.parse.parse_qs(parsed.query)
            session_id = query.get("session_id", [""])[0]
            if not session_id:
                self.send_json({"error": "session_id required"}, 400)
            else:
                from ipa.agent import AgentMemory
                with AgentMemory() as memory:
                    session = memory.get_session(session_id)
                    if session is None:
                        self.send_json({"error": "session not found"}, 404)
                    else:
                        episodes = memory.get_episodes(session_id)
                        self.send_json({
                            "session": session.__dict__,
                            "episodes": [e.__dict__ for e in episodes],
                        })
        elif parsed.path == "/api/agent/approvals":
            # Unified human-in-the-loop queue (Fase 3): consolidation proposals,
            # mastery inferences, roadmap proposals, research requests.
            parsed_query = urllib.parse.parse_qs(parsed.query)
            status_filter = parsed_query.get("status", ["pending"])[0]
            approvals = []
            # Consolidations + mastery inferences
            try:
                from ipa.agentic.memory_consolidation import ConsolidationStore
                consolidation_store = ConsolidationStore()
                for p in consolidation_store.list_proposals(status=status_filter if status_filter != "all" else None):
                    approvals.append({
                        "id": p.proposal_id, "kind": p.kind, "topic_id": p.topic_id,
                        "summary": p.summary, "status": p.status,
                        "proposed_at": p.proposed_at, "source": "consolidation",
                    })
                consolidation_store.close()
            except Exception:
                pass
            # Roadmap proposals + research requests from TutorStore
            try:
                from ipa.tutor.tutor_runtime import TutorStore
                tutor_store = TutorStore()
                for rm in tutor_store.list_roadmaps():
                    if status_filter == "all" or rm.status.value == "proposed":
                        approvals.append({
                            "id": rm.roadmap_id, "kind": "roadmap",
                            "topic_id": rm.goal_id,
                            "summary": f"Roadmap v{rm.version} con {len(rm.units)} unidades",
                            "status": rm.status.value, "proposed_at": rm.created_at,
                            "units": [{"order": u.order, "concept_id": u.concept_id, "reason": u.reason} for u in rm.units],
                        })
                for r in tutor_store.list_research_requests():
                    if status_filter == "all" or r.status.value == "pending_approval":
                        approvals.append({
                            "id": r.request_id, "kind": "research_request",
                            "topic_id": r.concept_id, "summary": r.question,
                            "status": r.status.value, "proposed_at": r.created_at,
                            "budget": {"max_urls": r.budget.max_urls, "max_seconds": r.budget.max_seconds},
                        })
                tutor_store.close()
            except Exception:
                pass  # TutorStore may not exist yet (no tutor usage)
            self.send_json({"approvals": approvals})
        elif parsed.path == "/api/agent/tutor/state":
            # Tutor state: mastery records + evidence counts
            from ipa.tutor.tutor_runtime import TutorStore
            tutor_store = TutorStore()
            records = tutor_store.list_topic_records()
            self.send_json({
                "records": [
                    {
                        "topic_id": r.topic_id,
                        "mastery_status": r.mastery_status.value,
                        "mastery_score": r.mastery_score,
                        "attempts": r.attempts,
                        "evidence_count": len(r.evidence_ids),
                        "updated_at": r.updated_at,
                    }
                    for r in records
                ]
            })
            tutor_store.close()
        elif parsed.path == "/api/tutor/roadmaps":
            self.send_json(tutor_roadmaps_payload())
        elif parsed.path == "/api/tutor/projects":
            # Selector de la pestaña Roadmaps: proyectos (goals) + sus roadmaps.
            self.send_json(tutor_projects_payload())
        elif parsed.path == "/api/tutor/roadmap/context":
            # Contexto completo de un roadmap para la pestaña (objetivo,
            # porqué, avance, conceptos re-derivados, foco).
            _rid = urllib.parse.parse_qs(parsed.query).get("roadmap_id", [""])[0]
            if not _rid:
                self.send_json({"ok": False, "error": "roadmap_id required"}, 400)
            else:
                self.send_json(tutor_roadmap_context(_rid))
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
                        # It's a base source â€” add an override to added list
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
                config = effective_scrape_config(); command = [VENV_PYTHONW, "-u", "scripts/cli/run_web_scrape.py", "--config", str(config), "--output", "Landing/web", "--engine", "auto", "--no-images", "--no-ocr"]
                self.send_json(spawn_job("scraper", command), 202)
            elif parsed.path == "/api/fastpath/run":
                reporter_corpus = active_reporter_output() / "corpus"
                command = [VENV_PYTHONW, "-u", "scripts/cli/run_fast_path.py", "--input", "Landing/web", "--output", str(reporter_corpus)]
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
                # Reporter desacoplado del pipeline — el agente lo invoca via
                # la tool compile_report. Este endpoint queda deshabilitado.
                self.send_json({"ok": False, "error": "Reporter desacoplado del pipeline. Usar el agente via compile_report tool."}, 410)
            elif parsed.path == "/api/pipeline/run":
                with PIPELINE_LOCK:
                    if PIPELINE_THREAD is not None and PIPELINE_THREAD.is_alive():
                        progress = read_json(PIPELINE_PROGRESS, {})
                        self.send_json({
                            "error": "El pipeline ya está ejecutándose",
                            "already_running": True,
                            "stage": progress.get("stage"),
                            "percent": progress.get("percent"),
                            "detail": progress.get("detail"),
                        }, 409)
                        return
                    period_mode = str(body.get("period_mode", "days"))
                    period_start = str(body.get("period_start", ""))
                    period_end = str(body.get("period_end", ""))
                    days_back = int(body.get("days_back", 0))
                    thread = threading.Thread(target=run_full_pipeline, args=(period_start, period_end, period_mode, days_back), daemon=True)
                    globals()["PIPELINE_THREAD"] = thread
                    thread.start()
                self.send_json({"ok": True, "status": "running", "message": "Ingesta iniciada: scraper â†’ FastPath (indexing BM25 + LanceDB). Reporter desacoplado — usar agente para reportes."}, 202)
            elif parsed.path == "/api/reports/review":
                report_path = str(body.get("path", ""))
                status = str(body.get("status", ""))
                if status not in {"approved", "rejected", "changes_requested"}:
                    raise ValueError("revisiÃ³n de corpus invÃ¡lida")
                # Use provided path or fall back to latest
                if report_path:
                    report = load_report_by_path(report_path)
                else:
                    report = latest_report()
                if not report:
                    # No report.json found â€” still clean up the reporter corpus directory
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
                    # Promote approved documents from the reporter corpus to the main corpus.
                    # Uses the new promote_documents_to_main() which works with
                    # individual document IDs, independent of the report.
                    report_dir = Path(report["path"]).parent
                    reporter_corpus = report_dir / "corpus"
                    reporter_db = report_dir / "reporter.db"
                    doc_ids_to_promote: list[str] = []
                    if reporter_db.exists():
                        import sqlite3 as _sql3
                        conn = _sql3.connect(str(reporter_db))
                        try:
                            # Get all document_ids with approved decisions
                            rows = conn.execute(
                                "SELECT document_id FROM document_decisions WHERE payload_json LIKE '%\"review_status\": \"approved\"%'"
                            ).fetchall()
                            doc_ids_to_promote = [row[0] for row in rows]
                            # If no approved decisions, promote all non-duplicate docs
                            if not doc_ids_to_promote:
                                rows = conn.execute(
                                    "SELECT document_id FROM document_metadata"
                                ).fetchall()
                                doc_ids_to_promote = [row[0] for row in rows]
                        finally:
                            conn.close()

                    result = {"promoted_docs": 0, "promoted_chunks": 0, "promoted_vectors": 0}
                    if doc_ids_to_promote and reporter_corpus.exists():
                        from ipa.agentic.promotion_executor import promote_documents_to_main
                        result = promote_documents_to_main(
                            doc_ids_to_promote, reporter_corpus, MAIN_CORPUS,
                        )

                    # Mark the report as approved in the review table
                    report_id = report.get("report_id", "")
                    if report_id:
                        with dashboard_connection() as connection:
                            connection.execute("INSERT OR REPLACE INTO report_reviews VALUES (?,?,?,?,?)",
                                (report_id, "approved", str(body.get("decided_by", "web-user")),
                                 f"promoted {result.get('promoted_docs', 0)} docs to main corpus", now()))
                            connection.commit()

                    _sweep = run_landing_sweep()
                    self.send_json({"ok": True, "report_id": report_id, "status": "approved",
                                    "sweep": _sweep, **result})
                elif status == "rejected":
                    # Delete the report and all associated data
                    result = delete_report(report["path"])
                    _sweep = run_landing_sweep()
                    self.send_json({"ok": True, "report_id": report.get("report_id", ""),
                                    "status": "rejected", "sweep": _sweep, **result})
                else:
                    # changes_requested â€” just record the review
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
            elif parsed.path == "/api/promotions/process":
                # Manually process the promotion queue — physically promote
                # pending documents from the reporter corpus to the main corpus.
                from ipa.agentic.topic_clusters import TopicClusterStore
                from ipa.agentic.promotion_executor import process_promotion_queue
                cluster_store = TopicClusterStore()
                try:
                    rep_corpus = REPORTER_ROOT / "quality-check" / active_reporter_output().name / "corpus"
                    result = process_promotion_queue(cluster_store, rep_corpus, MAIN_CORPUS)
                finally:
                    cluster_store.close()
                _sweep = run_landing_sweep() if result.get("processed", 0) > 0 else None
                self.send_json({"ok": True, "sweep": _sweep, **result})
            elif parsed.path == "/api/decisions/review":
                report = latest_report(); decision_id = str(body.get("decision_id", "")); status = str(body.get("status", ""))
                if not report or status not in {"approved", "rejected", "changes_requested"}: raise ValueError("revisiÃ³n invÃ¡lida")
                db = Path(report["path"]).parent / "reporter.db"
                approval = {"decision": status, "decided_at": now(), "decided_by": str(body.get("decided_by", "web-user")), "note": body.get("note")}
                with sqlite3.connect(str(db)) as conn:
                    row = conn.execute("SELECT payload_json FROM document_decisions WHERE decision_id=?", (decision_id,)).fetchone()
                    if not row: raise FileNotFoundError(decision_id)
                    payload = json.loads(row[0]); payload["review_status"] = status; payload["approval"] = approval
                    conn.execute("UPDATE document_decisions SET payload_json=? WHERE decision_id=?", (json.dumps(payload, ensure_ascii=False), decision_id)); conn.commit()
                _sweep = run_landing_sweep()
                self.send_json({"ok": True, "decision_id": decision_id, "review_status": status, "sweep": _sweep})
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
                        [restart_python, "-u", str(ROOT / "scripts" / "operations" / "dashboard_watchdog.py"), "--restart",
                         "--host", "127.0.0.1", "--port", "8765"],
                        cwd=str(ROOT),
                        creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0,
                        close_fds=True,
                    )
                threading.Thread(target=_delayed_restart, daemon=True).start()
            elif parsed.path == "/api/idle/toggle":
                # Perilla del sidebar: ON/OFF del enriquecimiento idle
                # (Tier 1 y Tier 2). Persistido — sobrevive restarts.
                try:
                    from . import server as _server_mod
                    enabled = bool(body.get("enabled", True))
                    _server_mod.set_idle_enabled(enabled)
                    self.send_json({"ok": True, "enabled": enabled}, 200)
                except Exception as exc:
                    self.send_json({"ok": False, "error": str(exc)}, 500)
            elif parsed.path == "/api/tutor/roadmap/focus":
                # El usuario indica sobre qué roadmap trabajar (click en la
                # card del stepper). La sesión actual lo adopta y el foco
                # persiste globalmente — otras sesiones lo heredan.
                try:
                    from ipa.tutor.tutor_chat import get_tutor_driver
                    rid = body.get("roadmap_id", "")
                    sid = body.get("session_id") or ""
                    if not rid:
                        self.send_json({"ok": False, "error": "roadmap_id required"}, 400)
                    else:
                        self.send_json(get_tutor_driver().focus_roadmap(sid, rid), 200)
                except Exception as exc:
                    self.send_json({"ok": False, "error": str(exc)}, 500)
            elif parsed.path == "/api/tutor/roadmap/unfocus":
                # Despinear: la sesión suelta el roadmap y el foco global se
                # limpia si apuntaba a él (botón × del chip del chat).
                try:
                    from ipa.tutor.tutor_chat import get_tutor_driver
                    rid = body.get("roadmap_id", "")
                    sid = body.get("session_id") or ""
                    if not rid:
                        self.send_json({"ok": False, "error": "roadmap_id required"}, 400)
                    else:
                        self.send_json(get_tutor_driver().unfocus_roadmap(sid, rid), 200)
                except Exception as exc:
                    self.send_json({"ok": False, "error": str(exc)}, 500)
            elif parsed.path == "/api/tools/execute":
                # Frontera unificada de tools para agentes externos (MCP
                # proxy): mismo registry que el chat del dashboard.
                self.send_json(
                    execute_tool_payload(body.get("name", ""), body.get("args")), 200)
            elif parsed.path == "/api/tutor/roadmap/decision":
                # Human gate on a proposed roadmap (button click).
                try:
                    from ipa.tutor.tutor_chat import get_tutor_driver
                    rid = body.get("roadmap_id", "")
                    decision = body.get("decision", "")
                    sid = body.get("session_id") or ""
                    if not rid or decision not in ("approve", "reject", "proposed"):
                        self.send_json({"ok": False, "error": "roadmap_id + decision(approve|reject|proposed) required"}, 400)
                    else:
                        self.send_json(get_tutor_driver().decide_roadmap(sid, rid, decision), 200)
                except Exception as exc:
                    self.send_json({"ok": False, "error": str(exc)}, 500)
            elif parsed.path == "/api/tutor/roadmap/archive":
                # Archive/unarchive a roadmap card — operational flag, the
                # Roadmap contract and unit progress stay untouched.
                try:
                    from ipa.tutor.tutor_runtime import TutorStore
                    rid = body.get("roadmap_id", "")
                    archived = bool(body.get("archived", True))
                    if not rid:
                        self.send_json({"ok": False, "error": "roadmap_id required"}, 400)
                        return
                    _st = TutorStore()
                    try:
                        _st.set_roadmap_archived(rid, archived)
                    finally:
                        _st.close()
                    self.send_json({"ok": True, "roadmap_id": rid, "archived": archived}, 200)
                except Exception as exc:
                    self.send_json({"ok": False, "error": str(exc)}, 500)
            elif parsed.path == "/api/tutor/research/decision":
                # Human gate on a Tutor ResearchRequest (button click).
                # On approve, the approved request executes in background via
                # the Fase-1 executor (bounded, audited).
                try:
                    from ipa.tutor.tutor_chat import get_tutor_driver
                    rid = body.get("request_id", "")
                    decision = body.get("decision", "")
                    sid = body.get("session_id") or ""
                    if not rid or decision not in ("approve", "reject"):
                        self.send_json({"ok": False, "error": "request_id + decision(approve|reject) required"}, 400)
                        return
                    drv = get_tutor_driver()
                    res = drv.decide_research(sid, rid, decision)
                    if res.get("ok") and res.get("status") == "approved":
                        def _exec_tutor_research():
                            try:
                                from ipa.agent.agent_tools import ToolContext
                                from ipa.agent.system_tools import _main_corpus_dir
                                from ipa.agent.agent_core import AgentCore
                                from ipa.tutor.tutor_runtime import TutorSession, TutorStore
                                _core = AgentCore(interface="dashboard", role="tutor")
                                _ctx = ToolContext(memory=_core.memory, corpus_dir=_main_corpus_dir())
                                _store = TutorStore()
                                try:
                                    _sess = TutorSession(core=_core, store=_store, provider=None)
                                    outcome = _sess.execute_approved_research(rid, _ctx)
                                finally:
                                    _store.close()
                                # Notificar a la sesión del usuario (el poller
                                # del chat recoge el episodio nuevo).
                                ok = outcome["request"].status.value == "completed"
                                n = len(outcome["request"].result_source_refs or [])
                                try:
                                    _core.memory.record_episode(
                                        sid, turn_role="assistant",
                                        content=(
                                            f"Investigación completada — {n} fuentes quedaron "
                                            "indexadas en el corpus. Decime 'dale' y armo el roadmap."
                                            if ok else
                                            "La investigación falló. Podés reintentarla o ajustar el tema."
                                        ),
                                        identity_hash=_core.identity.identity_hash,
                                    )
                                except Exception:
                                    pass
                                print(f"[tutor] research {rid} completed", flush=True)
                            except Exception as exc:
                                print(f"[tutor] research {rid} failed: {exc}", flush=True)
                        import threading as _t
                        _t.Thread(target=_exec_tutor_research, daemon=True).start()
                    self.send_json(res, 200)
                except Exception as exc:
                    self.send_json({"ok": False, "error": str(exc)}, 500)
            elif parsed.path == "/api/tutor/state":
                # Tutor state for the session (active roadmap, topic, phase).
                try:
                    from ipa.tutor.tutor_chat import get_tutor_driver
                    sid = body.get("session_id") or ""
                    st = get_tutor_driver().state(sid)
                    self.send_json({
                        "ok": True, "phase": st.phase, "topic": st.topic,
                        "topic_id": st.topic_id, "roadmap_id": st.roadmap_id,
                    }, 200)
                except Exception as exc:
                    self.send_json({"ok": False, "error": str(exc)}, 500)
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
                # Clean the REPORTER corpus only â€” NEVER the main corpus
                # This removes all indices from the current run
                global _CLEANING_IN_PROGRESS
                import time as _ctime
                cleaned = []
                errors = []
                with _CLEANING_LOCK:
                    _CLEANING_IN_PROGRESS = True
                    # Wait for any in-flight db_counts() to finish (they hold SQLite handles)
                    _ctime.sleep(1.0)
                    # Remove ALL reporter runs (history) â€” this includes corpus + output
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
                    raise ValueError("pid invÃ¡lido")
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
            elif parsed.path == "/api/agent/chat":
                # Agent surface: submit a message from the dashboard (DEC-002).
                # Uses the shared star model provider when available (same
                # lazy-load pattern as Deep Dive); without a provider the
                # fallback reply records the turn.
                from ipa.agent import AgentCore
                from ipa.agent.provider_wiring import build_responder
                message = body.get("message", "").strip()
                session_id = body.get("session_id")
                if not message:
                    self.send_json({"error": "message required"}, 400)
                else:
                    provider = get_deep_dive_provider()  # shared lazy LLM
                    responder = build_responder(provider) if provider is not None else None
                    core = AgentCore(interface="dashboard", role="general")
                    try:
                        if session_id:
                            core.session_id = session_id
                        result = core.submit(message, responder=responder)
                        self.send_json({
                            "session_id": result["session_id"],
                            "reply": result["reply"],
                            "episode_count": core.get_session().episode_count if core.get_session() else 0,
                        })
                    finally:
                        core.close_session()
                        core.memory.close()
            elif parsed.path == "/api/agent/chat/stream":
                # Streaming chat via SSE (Server-Sent Events).
                # Bounded tool protocol: the model may emit [TOOL:name]{args}
                # as its FIRST output; the handler executes the tool
                # deterministically and regenerates with the result. Max one
                # tool round per turn — no ReAct loops.
                from ipa.agent import AgentCore
                from ipa.agent.system_tools import TOOL_CATALOG, execute_system_tool, parse_tool_marker
                message = body.get("message", "").strip()
                session_id = body.get("session_id")
                if not message:
                    self.send_json({"error": "message required"}, 400)
                    return
                provider = get_deep_dive_provider()
                if provider is None or not provider.is_loaded():
                    self.send_json({"error": "model not loaded"}, 503)
                    return
                # Rol del turno: general (reactivo) o tutor (state machine).
                _role = str(body.get("role") or "general").strip().lower()
                if _role not in ("general", "tutor"):
                    _role = "general"
                # Contexto opcional del turno: deep_dive ("Profundizar" desde
                # un reporte) corre el retrieval agéntico sobre el corpus del
                # reporte dentro del MISMO chat (superficie unificada).
                _dd = None
                _dd_error = None
                if _role == "general":
                    try:
                        _dd = parse_deep_dive_context(body)
                    except Exception as exc:
                        _dd, _dd_error = None, str(exc)
                # Build messages and record user turn BEFORE streaming.
                core = AgentCore(interface="dashboard", role=_role)
                if session_id:
                    core.session_id = session_id
                core.ensure_session()

                # ── ROL TUTOR: state machine drives, no reactive loop ────
                # Diagnóstico → roadmap (LLM propone) → gate humano (botón) →
                # lecciones con policy pedagógica. El LLM solo renderiza y
                # propone; el estado es determinístico (tutor_runtime).
                if _role == "tutor":
                    from . import server as _server_mod
                    from ipa.tutor.tutor_chat import get_tutor_driver
                    from ipa.agent.system_tools import _main_corpus_dir

                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    self._sse_write({"type": "session", "session_id": core.session_id})
                    _server_mod.CHAT_BUSY["flag"] = True
                    _server_mod.LAST_ACTIVITY["ts"] = time.time()

                    # Retrieval del corpus para el tutor (mismo pipeline que
                    # el chat general — conceptos para el roadmap).
                    def _tutor_retrieve(query: str):
                        _corpus = _main_corpus_dir()
                        if not _corpus:
                            return []
                        _lance = _retrieval_lance(_corpus)
                        _tbl = getattr(_lance, "_table", None)
                        _checkout = getattr(_tbl, "checkout_latest", None)
                        if callable(_checkout):
                            try:
                                _checkout()
                            except Exception:
                                pass
                        if not _lance.is_queryable():
                            return []
                        from ipa.storage.document_store import DocumentStore
                        _embed = _server_mod.get_embedding_adapter()
                        _dense, _sparse = _embed.embed_query_hybrid(query)
                        from ipa.indexes.reranker_adapter import maybe_rerank, rerank_enabled
                        _out: list[dict[str, Any]] = []
                        _seen: set[str] = set()
                        _store = DocumentStore(_corpus / "document_store.db")
                        _fetch = 24 if rerank_enabled() else 10
                        try:
                            for h in _lance.search_hybrid(query, _dense, limit=_fetch, query_sparse=_sparse):
                                _ch = _store.get_chunk(h.chunk_id)
                                if not _ch or _ch.document_id in _seen:
                                    continue
                                _seen.add(_ch.document_id)
                                _src = _store.get_source(_ch.document_id) or {}
                                _out.append({
                                    "text": ( _ch.text or "")[:400],
                                    "document_id": _ch.document_id,
                                    "source_domain": _src.get("source_domain"),
                                })
                            _out = maybe_rerank(query, _out, 10)
                        finally:
                            _store.close()
                        return _out

                    try:
                        # Streaming real en modo lección: el driver emite cada
                        # chunk del LLM via on_token mientras genera. Los turnos
                        # de gate/roadmap no streamean (texto renderizado, no
                        # generación de chat) — se emiten como bloques abajo.
                        _streamed = {"n": 0}

                        def _on_token(_t: str) -> None:
                            _streamed["n"] += len(_t)
                            self._sse_write({"type": "token", "text": _t})

                        result = get_tutor_driver().handle(
                            core, core.session_id, message, provider,
                            retrieve=_tutor_retrieve, on_token=_on_token,
                        )
                        reply = result.get("reply", "")
                        reply = _strip_filler(_strip_emojis(_clean_model_output(reply)))
                        if not _streamed["n"]:
                            for _i in range(0, len(reply), 400):
                                self._sse_write({"type": "token", "text": reply[_i:_i + 400]})
                        if result.get("roadmap_proposal"):
                            _prop = result["roadmap_proposal"]
                            self._sse_write({
                                "type": "roadmap_proposal",
                                "roadmap_id": _prop["roadmap_id"],
                                "title": f"Roadmap: {result.get('topic') or 'aprendizaje'}",
                                "items": _prop.get("items") or [
                                    f"{u['order']}. {u['reason']} (~{u['minutes']} min)"
                                    for u in _prop.get("units", [])
                                ],
                            })
                        if result.get("research_proposal"):
                            _rp = result["research_proposal"]
                            self._sse_write({
                                "type": "research_proposal",
                                "request_id": _rp["request_id"],
                                "query": _rp.get("question", ""),
                            })
                        self._sse_write({"type": "done", "reply": reply})
                    except Exception as exc:
                        self._sse_write({"type": "done", "reply": f"[tutor error] {exc}"})
                    finally:
                        _server_mod.CHAT_BUSY["flag"] = False
                        core.memory.close()
                    return

                messages = core.build_messages(message, history_limit=6)
                # Prefix-cache friendly: el contexto volátil de este turno
                # (evidencia RAG/memoria, notas de research) se acumula acá y
                # va al TAIL del prompt — no al system — para que el prefijo
                # [system estable + historia append-only] se reutilice entre
                # turnos en el KV del runner de Ollama (longest-prefix reuse).
                _volatile_ctx: list[str] = []
                # Progressive tool unlocking: el agente empieza con 7 tools
                # base y desbloquea más a medida que las usa. Esto reduce la
                # carga cognitiva del 9B (7 tools vs 17 en el catálogo).
                from . import server as _server_mod
                from ipa.agent.system_tools import (
                    BASE_TOOLS, build_tool_catalog, unlock_after_tool,
                )
                # Track unlocked tools per session (in-memory)
                if not hasattr(_server_mod, "_SESSION_TOOLS"):
                    _server_mod._SESSION_TOOLS = {}
                session_key = core.session_id
                used_tools = _server_mod._SESSION_TOOLS.get(session_key, set())
                unlocked = set(BASE_TOOLS)
                for used in used_tools:
                    unlocked = unlock_after_tool(used, unlocked)
                _server_mod._SESSION_TOOLS[session_key] = unlocked | used_tools
                catalog = build_tool_catalog(unlocked)
                # Tool protocol en el TAIL, no en el system: el catálogo cambia
                # cuando una tool se desbloquea, y cualquier mutación temprana
                # del prompt mata el longest-prefix-reuse del runner (todo lo
                # que sigue al diff —incluida la historia estable— se re-evalúa).
                _volatile_ctx.append(catalog)
                # Investigación en vuelo: si el pedido depende de los datos que
                # están llegando, el agente aguarda en vez de improvisar pasos
                # que requieren material que todavía no existe.
                try:
                    from ipa.agent.system_tools import _research_progress
                    _rp = _research_progress()
                    if _rp.get("status") == "running":
                        _volatile_ctx.append(
                            "\n\nHay una investigación web en curso sobre "
                            f"'{_rp.get('query', '?')}'. Si el pedido del usuario "
                            "depende de esos datos, respondé en una oración que "
                            "aguardamos a que llegue la información de la fuente "
                            "web — no propongas pasos que requieran ese material "
                            "ni pidas feedback que no podés usar todavía."
                        )
                except Exception:
                    pass

                core.memory.record_episode(
                    core.session_id,
                    turn_role="user",
                    content=message,
                    identity_hash=core.identity.identity_hash,
                )
                # SSE headers — MUST go before any _sse_write, otherwise the
                # event payload is flushed before the HTTP status line and the
                # browser aborts the malformed response (HTTP 502).
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                self._sse_write({"type": "session", "session_id": core.session_id})
                if _dd_error:
                    self._sse_write({"type": "error", "error": _dd_error[:200]})
                    self._sse_write({"type": "done", "reply": ""})
                    _server_mod.CHAT_BUSY["flag"] = False
                    core.memory.close()
                    return
                # Block the idle-session consolidator while generating
                _server_mod.CHAT_BUSY["flag"] = True
                _server_mod.LAST_ACTIVITY["ts"] = time.time()

                from ipa.agent.query_gate import classify_message, is_imperative
                _msg_kind = classify_message(message)
                _is_order = is_imperative(message)
                # Auto-research: ante un gap del corpus (retrieval vacío o
                # respuesta "no hay datos"), el agente dispara research_topic
                # solo — corre en background con budgets y dedup. IPA_AUTO_RESEARCH=0
                # restaura el comportamiento anterior (solo ofrece).
                _auto_research = os.environ.get("IPA_AUTO_RESEARCH", "1") != "0"
                _auto_hits: list[dict[str, Any]] = []
                # Cache de respuestas (opt-in IPA_RESPONSE_CACHE=1, default OFF):
                # repetir la MISMA pregunta en la MISMA sesión (doble envío,
                # retry tras timeout) re-emite la respuesta sin retrieval ni
                # generación. Match exacto normalizado — no similitud difusa:
                # servir una respuesta parecida-pero-no-igual es incorrecto.
                if _RESPONSE_CACHE_ON and _dd is None:
                    _rck = _response_cache_key(message, _role, core.session_id)
                    _rchit = _RESPONSE_CACHE.get(_rck)
                    if _rchit is not None:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "keep-alive")
                        self.end_headers()
                        self._sse_write({"type": "session", "session_id": core.session_id})
                        self._sse_write({"type": "cached", "hit": True})
                        for _i in range(0, len(_rchit), 400):
                            self._sse_write({"type": "token", "text": _rchit[_i:_i + 400]})
                        self._sse_write({"type": "done", "reply": _rchit})
                        _server_mod.CHAT_BUSY["flag"] = False
                        core.memory.close()
                        return
                if _dd is not None:
                    # Deep dive: retrieval agéntico sobre el corpus del
                    # reporte (planner + Tantivy + LanceDB híbrido 3-vías),
                    # evidencia inyectada al prompt del MISMO chat.
                    self._sse_write({"type": "retrieval", "stage": "start", "query": message})

                    def _do_dd():
                        from ipa.reporter.reporter_deep_dive import deep_dive_prepare
                        doc_reasons = {}
                        if _dd["category_id"]:
                            try:
                                td = topic_details(_dd["category_id"])
                                doc_reasons = {
                                    d["document_id"]: d["reason"]
                                    for d in td.get("documents", []) if d.get("reason")
                                }
                            except Exception:
                                pass
                        return deep_dive_prepare(
                            _dd["corpus"], message, 5,
                            retrieval_query=_dd["search"] or message,
                            agentic=True, category_id=_dd["category_id"],
                            doc_reasons=doc_reasons or None,
                        )

                    _dd_prep = None
                    try:
                        _dd_timeout = float(os.environ.get("IPA_RETRIEVAL_TIMEOUT_SECONDS", "60"))
                        _dd_prep = _RETRIEVAL_POOL.submit(_do_dd).result(timeout=_dd_timeout)
                    except _cf.TimeoutError:
                        self._sse_write({"type": "retrieval", "stage": "timeout"})
                    except Exception as exc:
                        self._sse_write({"type": "retrieval", "stage": "error", "error": str(exc)[:100]})
                    if _dd_prep is not None:
                        _dd_chunks = _dd_prep.get("chunks") or []
                        if _dd_chunks:
                            _dd_lines = "\n".join(
                                f"[{i}] {chunk.text[:1200]}"
                                for i, chunk in enumerate(_dd_chunks, 1)
                            )
                            _volatile_ctx.append(
                                "\n\nEvidencia del corpus del reporte — respondé SOLO "
                                "con estos datos, sin escribir marcadores [n] en la "
                                "respuesta:\n" + _dd_lines
                            )
                            self._sse_write({
                                "type": "retrieval", "stage": "found",
                                "count": len(_dd_chunks), "sources": [
                                    {"n": n,
                                     "document_id": ev.get("document_id"),
                                     "source_url": (ev.get("source") or {}).get("source_url")
                                        or (ev.get("source") or {}).get("canonical_url"),
                                     "source_domain": (ev.get("source") or {}).get("source_domain")}
                                    for n, ev in enumerate(_dd_prep.get("evidence") or [], 1)
                                ],
                            })
                        else:
                            self._sse_write({"type": "retrieval", "stage": "empty", "query": message})
                            _volatile_ctx.append(
                                "\n\nEl corpus del reporte no devolvió evidencia para "
                                "esta consulta. Decilo y ofrecé ampliar la búsqueda."
                            )
                elif _msg_kind == "memory":
                    # Memoria agéntica: la pregunta es sobre el usuario o
                    # sesiones pasadas → recall_memory resuelve sobre el
                    # corpus personal, NO el corpus de documentos. El 9B no
                    # elige tools confiablemente, así que corre server-side.
                    self._sse_write({"type": "retrieval", "stage": "start", "query": message})
                    try:
                        _mem = execute_system_tool("recall_memory", {"query": message, "limit": 5})
                        _items = (_mem.data or {}).get("items", [])
                        if _items:
                            _mem_lines = "\n".join(
                                f"- [{i['scope']}/{i['kind']}] {i['text']}"
                                for i in _items
                            )
                            _volatile_ctx.append(
                                "\n\nMEMORIA RECUPERADA — esto ES tu memoria real "
                                "(episodios y perfil de tu usuario, que es Valen; "
                                "vos sos RA):\n" + _mem_lines +
                                "\nRespondé la pregunta citando estos recuerdos "
                                "con naturalidad. NUNCA digas que no tenés acceso "
                                "a datos del usuario: esto que ves ES tu memoria."
                            )
                            self._sse_write({
                                "type": "retrieval", "stage": "found",
                                "count": len(_items), "sources": [
                                    {"n": n, "document_id": i.get("source") or "memoria",
                                     "source_domain": f"memoria/{i['scope']}"}
                                    for n, i in enumerate(_items, 1)
                                ],
                            })
                        else:
                            self._sse_write({"type": "retrieval", "stage": "empty", "query": message})
                            _volatile_ctx.append(
                                "\n\nTu memoria no tiene nada registrado sobre esto "
                                "todavía. Decilo honestamente y ofrecé recordarlo "
                                "si el usuario lo comparte ahora."
                            )
                    except Exception as exc:
                        self._sse_write({"type": "retrieval", "stage": "error", "error": str(exc)[:100]})
                elif _msg_kind == "knowledge":
                    self._sse_write({"type": "retrieval", "stage": "start", "query": message})
                    print("[retrieval] start", flush=True)

                    def _do_retrieval():
                        from ipa.agent.system_tools import _main_corpus_dir
                        _corpus = _main_corpus_dir()
                        if not _corpus:
                            return []
                        _lance = _retrieval_lance(_corpus)
                        # Pick up chunks ingested after the table was opened.
                        _tbl = getattr(_lance, "_table", None)
                        _checkout = getattr(_tbl, "checkout_latest", None)
                        if callable(_checkout):
                            try:
                                _checkout()
                            except Exception:
                                pass
                        _hits = []
                        if _lance.is_queryable():
                            print("[retrieval] embedding query", flush=True)
                            from ipa.storage.document_store import DocumentStore
                            from ipa.indexes.reranker_adapter import maybe_rerank, rerank_enabled
                            _embed = _server_mod.get_embedding_adapter()
                            dense_vec, sparse_weights = _embed.embed_query_hybrid(message)
                            print("[retrieval] searching hybrid", flush=True)
                            # Dedup por documento: preferir cobertura de fuentes
                            # distintas sobre múltiples chunks del mismo doc.
                            # Con rerank activo (default; IPA_RERANK=0 lo
                            # desactiva) se traen más candidatos y el
                            # cross-encoder elige el top-8.
                            _rerank = rerank_enabled()
                            _fetch = 24 if _rerank else 10
                            _seen_docs: set[str] = set()
                            _store = DocumentStore(_corpus / "document_store.db")
                            try:
                                for h in _lance.search_hybrid(
                                    message, dense_vec,
                                    limit=_fetch, query_sparse=sparse_weights,
                                ):
                                    chunk = _store.get_chunk(h.chunk_id)
                                    if not chunk or chunk.document_id in _seen_docs:
                                        continue
                                    _seen_docs.add(chunk.document_id)
                                    _src = _store.get_source(chunk.document_id) or {}
                                    _hits.append({
                                        "text": _truncate_at_sentence(chunk.text, 900),
                                        "chunk_id": h.chunk_id,
                                        "document_id": chunk.document_id,
                                        "source_url": _src.get("source_url"),
                                        "source_domain": _src.get("source_domain"),
                                    })
                                    if len(_hits) >= (_fetch if _rerank else 8):
                                        break
                                _hits = maybe_rerank(message, _hits, 8)
                            finally:
                                _store.close()
                        print(f"[retrieval] found {len(_hits)} hits", flush=True)
                        return _hits

                    try:
                        # Timeout configurable: el embed BGE-M3 en CPU (~10s)
                        # + la primera search que construye el FTS index
                        # (~26s sobre 129k chunks) superan el timeout viejo
                        # de 15s en máquinas con la GPU ocupada por el LLM.
                        _retrieval_timeout = float(
                            os.environ.get("IPA_RETRIEVAL_TIMEOUT_SECONDS", "60")
                        )
                        from ipa.agent.system_tools import _main_corpus_dir as _mcd
                        from ipa.indexes.reranker_adapter import rerank_enabled as _re_on
                        _rkey = f"{message}\x00{_mcd()}\x00{_re_on()}"
                        _cached_hits = _RETRIEVAL_CACHE.get(_rkey)
                        if _cached_hits is not None:
                            _auto_hits = _cached_hits
                        else:
                            _auto_hits = _RETRIEVAL_POOL.submit(_do_retrieval).result(timeout=_retrieval_timeout)
                            _RETRIEVAL_CACHE.put(_rkey, _auto_hits)
                    except _cf.TimeoutError:
                        self._sse_write({"type": "retrieval", "stage": "timeout"})
                    except Exception as exc:
                        self._sse_write({"type": "retrieval", "stage": "error", "error": str(exc)[:100]})
                    if _auto_hits:
                        _sources = [
                            {
                                "n": i,
                                "document_id": h["document_id"],
                                "source_url": h.get("source_url"),
                                "source_domain": h.get("source_domain"),
                            }
                            for i, h in enumerate(_auto_hits, 1)
                        ]
                        self._sse_write({
                            "type": "retrieval", "stage": "found",
                            "count": len(_auto_hits), "sources": _sources,
                        })
                        # El retrieval ya corrió este turno: NO se muta el
                        # catálogo a mitad de turno (rompería el prefijo KV
                        # compartido). Si el modelo emite search_corpus igual,
                        # el dedup por call_key lo rechaza.
                        context_lines = "\n".join(
                            f"[{i}] ({h['source_domain'] or h['document_id']}) {h['text']}"
                            for i, h in enumerate(_auto_hits, 1)
                        )
                        _volatile_ctx.append(
                            "\n\nContexto relevante del corpus — la búsqueda ya "
                            "se ejecutó este turno: respondé SOLO con "
                            "estos datos, sin escribir marcadores [n] en la "
                            "respuesta (el usuario no puede abrir esas fuentes):\n" + context_lines
                        )
                        if _is_order:
                            _volatile_ctx.append(
                                "\n\nEl usuario te dio una ORDEN directa de profundizar. "
                                "Ejecutá ya con el contexto de arriba: respondé con el "
                                "análisis profundo que pidieron. Si el contexto no "
                                "alcanza, emití [TOOL:search_corpus]{...} o "
                                "[TOOL:research_topic]{...} en esta misma respuesta — "
                                "NO preguntes si querés que empiece."
                            )
                        elif _auto_research and _msg_kind == "knowledge":
                            _volatile_ctx.append(
                                "\n\nSi este contexto NO responde la pregunta del "
                                "usuario, decí qué falta y emití "
                                "[TOOL:research_topic]{\"query\": \"<tema faltante>\"} "
                                "en esta misma respuesta — la búsqueda web corre en "
                                "background y avisa sola al terminar."
                            )
                    else:
                        self._sse_write({"type": "retrieval", "stage": "empty", "query": message})
                        if _is_order:
                            _volatile_ctx.append(
                                "\n\nEl corpus NO devolvió resultados y el usuario dio una "
                                "ORDEN directa. Ejecutá: emití [TOOL:research_topic]{...} "
                                "con la query en esta misma respuesta — no pidas permiso."
                            )
                        elif _auto_research and _msg_kind == "knowledge":
                            _volatile_ctx.append(
                                "\n\nEl corpus NO devolvió resultados para esta consulta. "
                                "Decí que no hay datos en el corpus y emití "
                                "[TOOL:research_topic]{\"query\": \"<tema>\"} en esta "
                                "misma respuesta — la búsqueda web corre en background "
                                "y avisa sola al terminar. No respondas desde tu "
                                "conocimiento interno."
                            )
                        else:
                            _volatile_ctx.append(
                                "\n\nEl corpus NO devolvió resultados para esta consulta. "
                                "Decí que no hay datos y ofrecé lanzar una investigación "
                                "web con la tool research_topic. No respondas desde tu "
                                "conocimiento interno."
                            )

                # Merge: contexto volátil al tail — dentro del turno user,
                # antes de la pregunta (evidencia primero, pregunta última =
                # más cerca de la generación). El episodio grabado guarda el
                # mensaje crudo, así la historia sigue append-only/estable.
                if _volatile_ctx:
                    messages[-1]["content"] = (
                        "\n\n".join(_volatile_ctx) + "\n\n" + messages[-1]["content"]
                    )

                try:
                    def _clean(text: str) -> str:
                        return _clean_model_output(text)

                    def _stream_generation(msgs: list[dict[str, str]], *, holdback: bool = False):
                        """Stream tokens; with holdback, buffer until we know the
                        output is not a [TOOL:...] call. Returns (emitted, tool_detected, buffer)."""
                        emitted = ""
                        buffer = ""
                        is_tool_call = False
                        # Safety net: detect degenerate repetition loops (bug
                        # del 2026-09-08: "¿Qué necesitas? 😊" x40+). Si una
                        # frase corta se repite 8+ veces, cortar la generación.
                        _rep_check_window = 200  # chars to check for repetition
                        _rep_threshold = 8  # max repeats of same short phrase
                        # max_new_tokens: 384 — margen para explicaciones sin
                        # cortar a media oración; el guard anti-repetición
                        # sigue cortando loops degenerados antes. Deep dive
                        # pide 768 (explicación larga con evidencia).
                        _max_tokens = 768 if _dd is not None else 384
                        for chunk in provider.generate_chat_stream(msgs, max_new_tokens=_max_tokens):
                            if chunk.get("error"):
                                self._sse_write({"type": "error", "error": chunk["error"]})
                                break
                            if chunk.get("text"):
                                piece = chunk["text"]
                                if is_tool_call:
                                    buffer += piece  # accumulate the full tool marker
                                elif holdback:
                                    buffer += piece
                                    if len(buffer) >= 6:
                                        stripped = buffer.lstrip()
                                        # Aceptar [TOOL:...] canónico Y garbled
                                        # ([TRUN_...], [TOOL ...], etc.) como
                                        # posible tool call — el parser fuzzy
                                        # lo resuelve después.
                                        if stripped.startswith("[TOOL:") or (
                                            stripped.startswith("[T")
                                            and not stripped.startswith("[Tiene")
                                            and not stripped.startswith("[Toda")
                                            and not stripped.startswith("[Tú")
                                            and not stripped.startswith("[Tu ")
                                            and not stripped.startswith("[Tamb")
                                            and not stripped.startswith("[Todo")
                                            and not stripped.startswith("[Tien")
                                            and not stripped.startswith("[Tan ")
                                            and not stripped.startswith("[Tal ")
                                            and not stripped.startswith("[Tras")
                                            and not stripped.startswith("[Tec")
                                            and not stripped.startswith("[Tra")
                                            and not stripped.startswith("[Tus ")
                                            and not stripped.startswith("[Te ")
                                            and not stripped.startswith("[Ter")
                                            and not stripped.startswith("[Tar")
                                        ):
                                            is_tool_call = True
                                        else:
                                            emitted += buffer
                                            self._sse_write({"type": "token", "text": _strip_emojis(buffer)})
                                            buffer = ""
                                            holdback = False
                                else:
                                    emitted += piece
                                    self._sse_write({"type": "token", "text": _strip_emojis(piece)})
                                # Degenerate repetition detection — NO correr
                                # si el texto parece un tool marker (empieza con [).
                                if not is_tool_call and len(emitted) > _rep_check_window:
                                    if emitted.lstrip().startswith("["):
                                        # Podría ser tool marker garbled: no
                                        # cortar con el detector de repetición.
                                        pass
                                    else:
                                        tail = emitted[-_rep_check_window:]
                                        # a) Repetición de frases cortas (bug original)
                                        cut = False
                                        for phrase_len in range(10, 60):
                                            phrase = tail[-phrase_len:]
                                            count = tail.count(phrase)
                                            if count >= _rep_threshold:
                                                cut_at = len(emitted) - phrase_len * (count - 1)
                                                emitted = emitted[:cut_at].rstrip()
                                                self._sse_write({"type": "token", "text": " […]"})
                                                cut = True
                                                break
                                        if cut:
                                            break
                                        # b) Detección de basura: alta densidad
                                        # de emojis/caracteres no alfanuméricos en
                                        # el tail. El 9B degradado produce secuencias
                                        # de emojis y símbolos sin texto coherente.
                                        non_alpha = sum(1 for c in tail if not c.isalnum() and c not in " .,;:!?¿¡\n-—'\"()")
                                        if non_alpha > len(tail) * 0.4 and len(tail) > 150:
                                            # Cortar en el último punto/oración coherente
                                            # antes de la basura
                                            for i in range(len(emitted) - 50, max(0, len(emitted) - 300), -1):
                                                if emitted[i] in ".!?":
                                                    emitted = emitted[:i + 1].rstrip()
                                                    break
                                            else:
                                                emitted = emitted[:max(0, len(emitted) - 100)].rstrip()
                                            self._sse_write({"type": "token", "text": " […]"})
                                            break
                            if chunk.get("done"):
                                break
                        if not is_tool_call and buffer:
                            emitted += buffer
                            self._sse_write({"type": "token", "text": _strip_emojis(buffer)})
                            buffer = ""
                        raw_out = _clean(_strip_emojis(emitted))
                        print(f"[chat] raw_output: {raw_out[:200]}", flush=True)
                        return raw_out, is_tool_call, buffer
                    # Each round executes deterministically, feeds the result
                    # back, and regenerates — up to MAX_TOOL_ROUNDS calls per
                    # turn. A repeated (name, args) call is refused. This is
                    # a bounded ReAct loop, not an unbounded one.
                    MAX_TOOL_ROUNDS = 3
                    executed_calls: set[str] = set()
                    # Tools ejecutadas en el turno → se persisten como
                    # tool_calls del episodio (SkillDetector y la reflexión
                    # estratégica infieren patrones desde ahí).
                    _turn_tools: list[str] = []
                    tool_round = 0

                    def _strip_tool_marker(text: str) -> str:
                        """Cut text at the first tool-marker-like span."""
                        m = re.search(r"\[TOOL[: ]\s*[a-zA-Z_]+\]|\[[A-Z_]{4,}\]", text)
                        return text[:m.start()].strip() if m else text.strip()

                    def _run_tool(tool_name: str, tool_args: dict[str, Any]) -> str:
                        """Execute one tool round: SSE events + async watchers."""
                        self._sse_write({"type": "tool_start", "tool": tool_name, "args": tool_args})
                        _turn_tools.append(tool_name)
                        result = None
                        try:
                            if tool_name == "search_corpus" and _auto_hits:
                                # El auto-retrieval ya ejecutó esta búsqueda en
                                # este turno: responder con el cache en vez de
                                # repetir embedding + hybrid search (~10s).
                                _ev = "\n".join(
                                    f"[{i}] {h['document_id']} — {h['text'][:200]}"
                                    for i, h in enumerate(_auto_hits, 1)
                                )
                                tool_block = (
                                    f"[Resultado de search_corpus]\n"
                                    f"{len(_auto_hits)} resultados (ya recuperados "
                                    f"en este turno). Citá los hits como [n].\n{_ev}"
                                )
                                self._sse_write({"type": "tool_result", "tool": tool_name, "ok": True})
                                return tool_block
                            result = execute_system_tool(tool_name, tool_args)
                            if result.ok:
                                # Compact, natural-language format for the 9B:
                                # summary first, then a SHORT data excerpt (not
                                # raw JSON — the 9B degrades with large JSON).
                                data_str = json.dumps(result.data, ensure_ascii=False)
                                if len(data_str) > 800:
                                    data_str = data_str[:800] + "…"
                                tool_block = (
                                    f"[Resultado de {tool_name}]\n{result.summary}\n"
                                    f"Datos: {data_str}"
                                )
                            else:
                                tool_block = f"[La herramienta {tool_name} falló: {result.error}]"
                        except Exception as exc:
                            tool_block = f"[La herramienta {tool_name} falló: {exc}]"
                        self._sse_write({"type": "tool_result", "tool": tool_name, "ok": bool(result is not None and result.ok)})
                        # Watchers: las tools async (ingesta, investigación web)
                        # avisan al terminar con un resumen del agente en la
                        # misma sesión (proactivo, sin request del usuario).
                        if result is not None and result.ok and not result.data.get("already_running"):
                            if tool_name in ("run_pipeline", "run_ingestion"):
                                _server_mod.PIPELINE_WATCH["session_id"] = core.session_id
                                _server_mod.PIPELINE_WATCH["saw_running"] = False
                                _server_mod.PIPELINE_WATCH["mode"] = tool_name
                            elif tool_name == "research_topic":
                                _server_mod.RESEARCH_WATCH["session_id"] = core.session_id
                                _server_mod.RESEARCH_WATCH["saw_running"] = False
                        return tool_block

                    full_reply = ""
                    while True:
                        reply, tool_detected, tool_buffer = _stream_generation(messages, holdback=True)
                        # Si la respuesta llegó al límite de tokens quedó
                        # cortada a mitad de frase: marcarla para que el
                        # modelo no la "continúe" en el próximo turno.
                        if reply and not reply.rstrip().endswith((".", "!", "?", ":", ")")):
                            reply = reply.rstrip() + " […]"
                        parsed_tool = None
                        if tool_detected:
                            parsed_tool = parse_tool_marker(_clean(tool_buffer))
                            if parsed_tool is None:
                                # Falso positivo: el buffer era texto normal.
                                flushed = _strip_emojis(_clean(tool_buffer))
                                if flushed:
                                    self._sse_write({"type": "token", "text": flushed})
                                    reply = (reply + " " + flushed).strip() if reply else flushed
                                tool_detected = False
                        if parsed_tool is None and reply:
                            # Post-scan: el modelo a veces escribe prosa y emite
                            # el marcador al FINAL del mensaje.
                            tail = parse_tool_marker(reply)
                            if tail is not None:
                                parsed_tool = tail
                                reply = _strip_tool_marker(reply)
                        if parsed_tool is None:
                            # Safety net: el modelo dice que va a investigar
                            # pero no emitió el marcador. Solo dispara si el
                            # usuario pidió investigar Y el modelo habla de
                            # investigar sin emitir el marcador. NO dispara en
                            # follow-ups (después de que ya se ejecutó una tool).
                            import re as _re
                            user_asked_research = _re.search(
                                r"investig|buscá.*web|buscar.*web|research|buscá.*internet",
                                message.lower()
                            )
                            model_claimed_research = _re.search(
                                r"investigaci[oó]n.*activa|está ejecutándose|en curso|"
                                r"voy a investigar|inicié la búsqueda|busqueda.*activa",
                                reply.lower()
                            )
                            if (
                                user_asked_research
                                and model_claimed_research
                                and tool_round == 0  # solo en la primera ronda
                            ):
                                parsed_tool = ("research_topic", {"query": message, "_session_id": core.session_id})

                            # Safety net: corpus search. Si el usuario pide
                            # información sobre un tema específico y el modelo
                            # dice "no tengo datos" sin haber buscado, ejecuta
                            # search_corpus con el query extraído del mensaje.
                            # Con auto-research, la insuficiencia dispara
                            # research_topic directamente (el retrieval de este
                            # turno ya corrió — re-buscar no aporta).
                            model_claimed_no_data = _re.search(
                                r"no hay informaci[oó]n|no tengo datos|"
                                r"no existe evidencia|no encuentro|sin resultados|"
                                r"no hay datos|no dispongo|no detalla|"
                                r"no menciona|no especifica|no incluye|no cubre|"
                                r"sin datos|no tengo informaci[oó]n",
                                reply.lower()
                            )
                            if (
                                parsed_tool is None
                                and model_claimed_no_data
                                and tool_round == 0
                                and not user_asked_research
                            ):
                                if _auto_research and _msg_kind == "knowledge":
                                    from ipa.agent.research_review import recently_researched
                                    if not recently_researched(message):
                                        parsed_tool = ("research_topic", {
                                            "query": message,
                                            "_session_id": core.session_id,
                                            "_auto": True,
                                        })
                                if parsed_tool is None:
                                    # Extraer el tema: "sobre X", "de X", "para X", "X"
                                    topic_match = _re.search(
                                        r"(?:sobre|de|para|acerca de|respecto a)\s+(.+?)(?:\?|$|\.)",
                                        message, _re.IGNORECASE
                                    )
                                    query = topic_match.group(1).strip() if topic_match else message.strip()
                                    parsed_tool = ("search_corpus", {"query": query, "limit": 5})
                            if parsed_tool is None:
                                full_reply = reply.strip()
                                break

                        tool_name, tool_args = parsed_tool
                        if tool_name == "research_topic":
                            tool_args["_session_id"] = core.session_id
                            # Mensaje crudo del usuario: la tool rescata de ahí
                            # las URLs que el modelo descartó al parafrasear la
                            # query (se scrapean directo como fuentes).
                            tool_args["_user_message"] = message
                        call_key = tool_name + "|" + json.dumps(tool_args, sort_keys=True, ensure_ascii=False)
                        tool_round += 1
                        refused = None
                        if call_key in executed_calls:
                            refused = "ya ejecutaste esta herramienta con los mismos argumentos en este turno"
                        elif tool_round > MAX_TOOL_ROUNDS:
                            refused = f"límite de {MAX_TOOL_ROUNDS} herramientas por turno alcanzado"

                        if not tool_detected:
                            # Marker al final de una respuesta: cerrar la
                            # burbuja de texto y grabarla como episodio.
                            self._sse_write({"type": "done", "reply": reply})
                            if reply:
                                core.memory.record_episode(
                                    core.session_id, turn_role="assistant", content=reply,
                                    identity_hash=core.identity.identity_hash,
                                )
                            self._sse_write({"type": "new_message"})

                        messages.append({"role": "assistant", "content": f"[TOOL:{tool_name}]{json.dumps(tool_args)}"})
                        if refused is not None:
                            self._sse_write({"type": "tool_start", "tool": tool_name, "args": tool_args})
                            self._sse_write({"type": "tool_result", "tool": tool_name, "ok": False})
                            tool_block = f"[La herramienta {tool_name} no se ejecutó: {refused}]"
                            messages.append({"role": "user", "content": tool_block + "\n\nRespondé al usuario con lo que tenés, sin más herramientas."})
                            self._sse_write({"type": "new_message"})
                            reply2, _, _ = _stream_generation(messages, holdback=False)
                            full_reply = _strip_tool_marker(reply2).strip() or "No pude ejecutar más acciones en este turno."
                            break
                        executed_calls.add(call_key)
                        tool_block = _run_tool(tool_name, tool_args)
                        # Progressive unlocking: registrar la tool usada y
                        # desbloquear las relacionadas para la próxima ronda.
                        session_key = core.session_id
                        if not hasattr(_server_mod, "_SESSION_TOOLS"):
                            _server_mod._SESSION_TOOLS = {}
                        if session_key not in _server_mod._SESSION_TOOLS:
                            _server_mod._SESSION_TOOLS[session_key] = set(BASE_TOOLS)
                        _server_mod._SESSION_TOOLS[session_key].add(tool_name)
                        new_unlocked = unlock_after_tool(
                            tool_name, _server_mod._SESSION_TOOLS[session_key]
                        )
                        _server_mod._SESSION_TOOLS[session_key] = new_unlocked
                        # NO reescribir el catálogo en messages[0]: mutar el
                        # system a mitad de turno rompe el prefijo KV y toda la
                        # historia se re-evalúa. Las tools recién desbloqueadas
                        # se anuncian en el tail (junto al resultado).
                        newly = sorted(new_unlocked - unlocked)
                        unlock_note = (
                            f"\n\nHerramientas desbloqueadas: {', '.join(newly)} — podés emitirlas si hacen falta."
                            if newly else ""
                        )
                        messages.append({"role": "user", "content": tool_block + unlock_note + "\n\nRespondé al usuario en 1-3 oraciones. Si necesitás otra herramienta, emití el marcador."})
                        self._sse_write({"type": "new_message"})
                        # loop → la próxima generación puede ser otro marcador
                        # o la respuesta final al usuario.

                    if not full_reply:
                        full_reply = "La herramienta se ejecutó."
                    # Filtrar emojis/basura ANTES de grabar el episodio.
                    # Si no, el historial se contamina y el modelo imita la
                    # basura en futuras respuestas (loop de feedback).
                    full_reply = _strip_filler(
                        _CITE_PATTERN.sub("", _strip_emojis(full_reply))
                    ).strip()
                    if not full_reply:
                        full_reply = "La herramienta se ejecutó."
                    core.memory.record_episode(
                        core.session_id,
                        turn_role="assistant",
                        content=full_reply,
                        identity_hash=core.identity.identity_hash,
                        tool_calls=_turn_tools,
                    )
                    self._sse_write({"type": "done", "reply": full_reply})
                    if _RESPONSE_CACHE_ON and _dd is None and full_reply:
                        _RESPONSE_CACHE.put(
                            _response_cache_key(message, _role, core.session_id),
                            full_reply,
                        )
                    # Cerrar la conexión: HTTP/1.1 keep-alive dejaría el socket
                    # abierto y el reader del browser nunca terminaría.
                    self.close_connection = True
                finally:
                    _server_mod.CHAT_BUSY["flag"] = False
                    _server_mod.LAST_ACTIVITY["ts"] = time.time()
                core.close_session()
                core.memory.close()
            elif parsed.path == "/api/agent/sessions/manage":
                # Session management: new | rename | archive (DEC-002 surface).
                from ipa.agent import AgentMemory
                action = body.get("action", "")
                memory = AgentMemory()
                try:
                    if action == "new":
                        from ipa.agent.agent_identity import load_identity
                        identity = load_identity()
                        sid = memory.open_session(
                            interface="dashboard", role=body.get("role", "general"),
                            identity_hash=identity.identity_hash,
                            title=body.get("title") or "Nueva conversación",
                        )
                        self.send_json({"ok": True, "session_id": sid})
                    elif action == "rename":
                        sid = body.get("session_id", "")
                        title = str(body.get("title", "")).strip()
                        if not sid or not title:
                            self.send_json({"error": "session_id y title requeridos"}, 400)
                        else:
                            memory.rename_session(sid, title)
                            self.send_json({"ok": True})
                    elif action == "archive":
                        sid = body.get("session_id", "")
                        if not sid:
                            self.send_json({"error": "session_id requerido"}, 400)
                        else:
                            memory.archive_session(sid)
                            self.send_json({"ok": True})
                    else:
                        self.send_json({"error": "action inválida (new|rename|archive)"}, 400)
                finally:
                    memory.close()
            elif parsed.path == "/api/agent/approvals/decide":
                # Unified approval gate (Fase 3): approve/reject any pending
                # proposal — consolidation, mastery inference, roadmap,
                # research request. One endpoint, one queue.
                from ipa.agentic.memory_consolidation import (
                    ConsolidationStore, approve_proposal, reject_proposal,
                    apply_approved_memory_consolidation,
                    apply_approved_mastery_inference,
                )
                from ipa.tutor.tutor_runtime import TutorStore, TutorSession
                from ipa.agent import AgentCore
                proposal_id = body.get("id", "")
                decision = body.get("decision", "approved")
                decided_by = body.get("decided_by", "web-user")
                note = body.get("note")
                if not proposal_id:
                    self.send_json({"error": "id required"}, 400)
                    return
                # 1. Consolidation proposals (memory + mastery inference)
                consolidation_store = ConsolidationStore()
                try:
                    proposal = consolidation_store.get_proposal(proposal_id)
                    if proposal is not None:
                        if decision == "approved":
                            approved = approve_proposal(consolidation_store, proposal_id, decided_by=decided_by)
                            if approved.kind == "memory_consolidation":
                                apply_approved_memory_consolidation(consolidation_store, proposal_id)
                            elif approved.kind == "mastery_inference":
                                # Mismo gate: la inferencia materializa en el
                                # UserTopicRecord del TutorStore.
                                _mts = TutorStore()
                                try:
                                    apply_approved_mastery_inference(
                                        consolidation_store, proposal_id, _mts)
                                finally:
                                    _mts.close()
                            self.send_json({"ok": True, "status": "approved"})
                        else:
                            reject_proposal(consolidation_store, proposal_id, decided_by=decided_by, note=note)
                            self.send_json({"ok": True, "status": "rejected"})
                        return
                finally:
                    consolidation_store.close()
                # 2. Roadmap proposals and research requests (TutorStore)
                tutor_store = TutorStore()
                try:
                    core = AgentCore(interface="dashboard", role="tutor")
                    tutor_session = TutorSession(core, tutor_store)
                    try:
                        if tutor_store.get_roadmap(proposal_id) is not None:
                            if decision == "approved":
                                tutor_session.approve_roadmap(proposal_id, decided_by=decided_by, note=note)
                            else:
                                tutor_session.reject_roadmap(proposal_id, decided_by=decided_by, note=note)
                            self.send_json({"ok": True, "status": decision})
                        elif tutor_store.get_research_request(proposal_id) is not None:
                            if decision == "approved":
                                tutor_session.approve_research_request(proposal_id, decided_by=decided_by, note=note)
                            else:
                                tutor_session.reject_research_request(proposal_id, decided_by=decided_by, note=note)
                            self.send_json({"ok": True, "status": decision})
                        else:
                            self.send_json({"error": "unknown proposal"}, 404)
                    except ValueError as exc:
                        self.send_json({"error": str(exc)}, 400)
                finally:
                    tutor_store.close()
                    core.memory.close()
            elif parsed.path == "/api/curation/judge":
                # LLM judge para la zona gris de la cascada de promoción.
                # Recibe {"docs": [{document_id, title, text, source_domain,
                # promotion_score}, ...]} y devuelve {"verdicts": [...]}.
                # Usa el provider del dashboard con think_mode temporal.
                from . import server as _server_mod
                from ipa.reporter.curation_judge import judge_gray_batch
                docs = body.get("docs", [])
                if not docs:
                    self.send_json({"verdicts": []}, 200)
                elif _server_mod.CHAT_BUSY["flag"]:
                    # El chat está generando — no usar el provider concurrentemente
                    self.send_json({
                        "verdicts": [
                            {"document_id": d["document_id"], "verdict": "defer",
                             "confidence": 0.0, "reason": "Chat activo, juez pospuesto"}
                            for d in docs
                        ]
                    }, 200)
                else:
                    provider = get_deep_dive_provider()
                    if provider is None or not provider.is_loaded():
                        # Provider no disponible → todo defer
                        self.send_json({
                            "verdicts": [
                                {"document_id": d["document_id"], "verdict": "defer",
                                 "confidence": 0.0, "reason": "Provider no cargado"}
                                for d in docs
                            ]
                        }, 200)
                    else:
                        # think_mode: desactivar no_think temporalmente para
                        # que el modelo razone sobre la zona gris.
                        original_no_think = provider.no_think
                        try:
                            provider.no_think = False
                            verdicts = judge_gray_batch(docs, provider)
                        finally:
                            provider.no_think = original_no_think
                        self.send_json({
                            "verdicts": [
                                {"document_id": v.document_id, "verdict": v.verdict,
                                 "confidence": v.confidence, "reason": v.reason}
                                for v in verdicts
                            ]
                        }, 200)
            else:
                self.send_json({"error": "not found"}, 404)
        except Exception as exc:
            import traceback
            print(f"[chat] EXCEPTION in do_POST: {exc}", flush=True)
            traceback.print_exc()
            try:
                self.send_json({"error": str(exc)}, 400)
            except Exception:
                pass


