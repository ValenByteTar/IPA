"""Tutor chat driver: wires the dashboard chat to the TutorSession.

The dashboard chat stays role="general" by default; when the user picks
the Tutor role (or accepts the derivation from a general turn), the
session switches here. The LLM only renders lesson text or proposes
structured artifacts — the state machine (topic detection, diagnosis,
roadmap gate, lesson turns) is deterministic.

Parallel roadmaps: state is keyed (session_id, topic); the TutorStore
already supports multiple roadmaps. Each session has one focused topic
at a time, but different sessions/goals can run in parallel.

Phases per (session):
  idle              → no topic: detect from message or ask
  roadmap_proposed  → a Roadmap waits on the human gate (button/message)
  active            → approved roadmap: lesson turns with tutor policy
  research_pending  → a ResearchRequest waits on the human gate
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ipa.tutor.tutor_runtime import TutorSession, TutorStore
from ipa.tutor.tutor_contracts import RoadmapStatus

DEFAULT_TUTOR_STORE = Path("outputs") / "agent" / "tutor.db"

# ── deterministic matchers ────────────────────────────────────────────────

# Topic detection: "quiero aprender X", "enseñame X", "un roadmap de X",
# "estudiar X". The topic is the trailing phrase.
_TOPIC_RE = re.compile(
    r"(?:quiero\s+(?:aprender|estudiar)|quisiera\s+(?:aprender|estudiar)|"
    r"me\s+gustar[ií]a\s+(?:aprender|estudiar)|"
    r"ense[ñn]ame|enseñame|aprendizaje\s+de|aprender\s+(?:sobre\s+)?|"
    r"estudiar\s+(?:sobre\s+)?|roadmap\s+(?:de|para|sobre)\s+|"
    r"mapa\s+de\s+estudio\s+(?:de|sobre)\s+)"
    r"(.+)",
    re.IGNORECASE,
)

_APPROVE_RE = re.compile(r"^\s*(aprobad|aprueb|si|sí|dale|adelante|ok|acepto|activa)[\w\s,.¡!]*$", re.IGNORECASE)
_REJECT_RE = re.compile(r"^\s*(rechaz|no\b|cancela|descarta|anula)[\w\s,.¡!]*$", re.IGNORECASE)

# Unit advance intent during an active roadmap: "siguiente", "avancemos",
# "ya entendí", "terminé esta unidad". Deterministic — the LLM doesn't decide.
_ADVANCE_RE = re.compile(
    r"\b(siguiente\s+unidad|siguiente\s+tema|avancemos|avanzar|avanzemos|"
    r"ya\s+entend[ií]|termin[eé]\s+(esta|la)\s+(unidad|tema|parte)|"
    r"continuemos|seguimos|pr[oó]xima\s+unidad)\b",
    re.IGNORECASE,
)


def _slug(text: str) -> str:
    """topic text → stable topic/goal slug."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:60] or "tema"


@dataclass
class _ChatResult:
    text: str = ""
    error: str | None = None


class _ProviderAdapter:
    """Adapt generate_chat → str providers (Ollama) to the tutor runtime
    contract: generate_chat → result with .text/.error."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.model_id = getattr(inner, "model_id", None) or getattr(inner, "model", "provider")

    def generate_chat(self, messages: Any, *, max_new_tokens: Any = None,
                      temperature: Any = None, **kw: Any) -> Any:
        try:
            result = self._inner.generate_chat(
                messages, max_new_tokens=max_new_tokens, temperature=temperature,
            )
        except Exception as exc:
            return _ChatResult(text="", error=str(exc))
        if hasattr(result, "text"):
            return result
        return _ChatResult(text=str(result or ""), error=None)

    def generate_chat_stream(self, messages: Any, *, max_new_tokens: Any = None,
                             temperature: Any = None, **kw: Any) -> Any:
        """Streaming passthrough when the inner provider supports it."""
        inner_stream = getattr(self._inner, "generate_chat_stream", None)
        if not callable(inner_stream):
            result = self.generate_chat(
                messages, max_new_tokens=max_new_tokens, temperature=temperature)
            yield {"text": "", "done": True, "error": result.error} if result.error \
                else {"text": result.text, "done": True}
            return
        yield from inner_stream(
            messages, max_new_tokens=max_new_tokens, temperature=temperature)


def _tutor_provider(provider: Any) -> Any:
    """Return a provider satisfying the tutor contract (.text/.error)."""
    if provider is None:
        return None
    if isinstance(provider, _ProviderAdapter):
        return provider
    return _ProviderAdapter(provider)


@dataclass
class TutorChatState:
    topic: str | None = None
    topic_id: str | None = None
    goal_id: str | None = None
    roadmap_id: str | None = None
    phase: str = "idle"  # idle | roadmap_proposed | active | research_pending
    pending_request_id: str | None = None


class TutorChatDriver:
    """Per-session tutor state machine for the dashboard chat.

    Holds no DB connection: each operation opens a fresh TutorStore bound
    to the calling thread (the dashboard serves requests on worker
    threads; sqlite connections are not thread-safe by default).
    """

    def __init__(self, store: TutorStore | None = None,
                 store_path: str | Path = DEFAULT_TUTOR_STORE) -> None:
        # Injected store for tests (same-thread); production uses path.
        self._injected = store
        self._store_path = Path(store_path)
        self._states: dict[str, TutorChatState] = {}

    class _open_store:
        """Context manager yielding a thread-bound TutorStore."""

        def __init__(self, driver: "TutorChatDriver") -> None:
            self._driver = driver
            self._tmp: TutorStore | None = None

        def __enter__(self) -> TutorStore:
            if self._driver._injected is not None:
                return self._driver._injected
            self._tmp = TutorStore(self._driver._store_path)
            return self._tmp

        def __exit__(self, *exc: object) -> None:
            if self._tmp is not None:
                self._tmp.close()

    def _store(self) -> "TutorChatDriver._open_store":
        return TutorChatDriver._open_store(self)

    def state(self, session_id: str) -> TutorChatState:
        st = self._states.get(session_id)
        if st is None:
            st = TutorChatState()
            # Recover an in-flight proposal after a restart — only for the
            # first session seen since boot, so a brand-new parallel session
            # doesn't inherit another session's pending gate.
            if not self._states:
                with self._store() as store:
                    try:
                        for rm in store.list_roadmaps():
                            if rm.status == RoadmapStatus.PROPOSED:
                                st.roadmap_id = rm.roadmap_id
                                st.goal_id = rm.goal_id
                                st.topic_id = rm.goal_id.removeprefix("goal:")
                                st.topic = st.topic_id.replace("-", " ")
                                st.phase = "roadmap_proposed"
                                break
                            if rm.status == RoadmapStatus.ACTIVE:
                                st.roadmap_id = rm.roadmap_id
                                st.goal_id = rm.goal_id
                                st.topic_id = rm.goal_id.removeprefix("goal:")
                                st.topic = st.topic_id.replace("-", " ")
                                st.phase = "active"
                                break
                    except Exception:
                        pass
            self._states[session_id] = st
        return st

    def _detect_topic(self, message: str) -> str | None:
        m = _TOPIC_RE.search(message.strip())
        if not m:
            return None
        topic = m.group(m.lastindex).strip().rstrip("?.!").strip()
        # "punto de partida N" / "desde cero" es contexto del alumno, no tema.
        topic = re.sub(
            r"[,\s]+(?:punto\s+de\s+partida\s+\S+|desde\s+cero|empezando\s+de\s+cero)\s*$",
            "", topic, flags=re.IGNORECASE,
        ).strip()
        topic = re.sub(r"^(?:sobre|de|acerca\s+de|el|la|los|las)\s+", "", topic, flags=re.IGNORECASE)
        return topic[:120] or None

    def _concepts_from_hits(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Corpus hits → concept candidates for roadmap proposal."""
        seen: set[str] = set()
        concepts: list[dict[str, Any]] = []
        for h in hits:
            doc_id = h.get("document_id") or h.get("chunk_id") or ""
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            # Titles the learner sees come from the source domain; the raw
            # doc id is internal only (never rendered).
            title = h.get("source_domain") or h.get("title") or "Documento del corpus"
            text = (h.get("text") or "").strip().replace("\n", " ")
            concepts.append({
                "concept_id": doc_id,
                "title": title,
                "definition": text[:150],
            })
        return concepts

    def _render_roadmap(self, roadmap: Any, concepts: list[dict[str, Any]]) -> str:
        by_id = {c["concept_id"]: c for c in concepts}
        lines = ["Roadmap propuesto:"]
        for u in roadmap.units:
            title = by_id.get(u.concept_id, {}).get("title", u.concept_id)
            lines.append(f"{u.order}. {title} — {u.reason} (~{u.estimated_effort_minutes} min)")
        return "\n".join(lines)

    # ── per-unit progress (unit_progress table; roadmap stays immutable) ──

    def _ensure_progress(self, store: Any, roadmap_id: str) -> None:
        """Seed unit 1 as 'current' when a roadmap activates."""
        if not store.unit_statuses(roadmap_id):
            store.set_unit_status(roadmap_id, 1, "current")

    def _advance_unit(self, store: Any, roadmap_id: str) -> dict[str, int] | None:
        """Mark the current unit done and the next one current."""
        roadmap = store.get_roadmap(roadmap_id)
        if roadmap is None or not roadmap.units:
            return None
        statuses = store.unit_statuses(roadmap_id)
        orders = [u.order for u in roadmap.units]
        current = next(
            (o for o in sorted(orders) if statuses.get(o) == "current"),
            next((o for o in sorted(orders) if statuses.get(o) != "done"), None),
        )
        if current is None:
            return None
        store.set_unit_status(roadmap_id, current, "done")
        nxt = next((o for o in sorted(orders) if o > current), None)
        if nxt is not None:
            store.set_unit_status(roadmap_id, nxt, "current")
            return {"done": current, "current": nxt}
        return {"done": current, "current": 0}

    def _progress_note(self, store: Any, roadmap_id: str) -> str | None:
        """Unit progress rendered for the lesson system prompt — so the
        tutor knows what was already taught and doesn't repeat it."""
        roadmap = store.get_roadmap(roadmap_id)
        if roadmap is None or not roadmap.units:
            return None
        statuses = store.unit_statuses(roadmap_id)
        done = [u.order for u in roadmap.units if statuses.get(u.order) == "done"]
        current = next(
            (u.order for u in roadmap.units if statuses.get(u.order) == "current"),
            None,
        )
        total = len(roadmap.units)
        if current is None and not done:
            return None
        parts = [f"Progreso del roadmap: unidad {current or '—'} de {total} en curso"]
        if done:
            parts.append(f"ya enseñadas: {', '.join(str(o) for o in sorted(done))}")
        return (
            ". ".join(parts) + ". "
            "No repitas unidades ya enseñadas salvo que el alumno lo pida "
            "explícitamente; enlazá con ellas si el tema lo requiere."
        )

    def _summarize_unit(self, store: Any, provider: Any, roadmap_id: str,
                        unit_order: int, lesson_result: dict[str, Any]) -> None:
        """Summarize what a completed unit taught (bounded LLM call, derived
        data — failure skips; the session summary still covers it)."""
        try:
            adapted = _tutor_provider(provider) if provider is not None else None
            if adapted is None:
                return
            roadmap = store.get_roadmap(roadmap_id)
            unit = next((u for u in (roadmap.units if roadmap else []) if u.order == unit_order), None)
            topic = (roadmap.goal_id.removeprefix("goal:").replace("-", " ")
                     if roadmap else "el tema")
            transcript = "\n".join(
                f"{m['role']}: {m['content'][:400]}"
                for m in (lesson_result.get("messages") or [])[-6:]
                if isinstance(m, dict) and m.get("content")
            )
            prompt = (
                f"Unidad {unit_order} de '{topic}'"
                + (f" (objetivo: {unit.reason[:200]})" if unit is not None else "")
                + ". Conversación de la lección:\n" + transcript[:3000]
                + "\n\nResumé en 2-3 oraciones qué se enseñó y qué entendió el "
                "alumno. Respondé solo el resumen."
            )
            res = adapted.generate_chat(
                [{"role": "user", "content": prompt}], max_new_tokens=160, temperature=0.0,
            )
            text = str(getattr(res, "text", "") or "").strip()
            if text:
                store.save_unit_summary(roadmap_id, unit_order, text)
        except Exception:
            pass

    def handle(
        self,
        core: Any,
        session_id: str,
        message: str,
        provider: Any,
        *,
        retrieve: Any | None = None,
        on_token: Any | None = None,
    ) -> dict[str, Any]:
        """Advance the tutor state machine. Returns:
        {"reply": str, "roadmap_proposal": {...}|None, "research_proposal": {...}|None}
        """
        st = self.state(session_id)
        out: dict[str, Any] = {"reply": "", "roadmap_proposal": None, "research_proposal": None}

        with self._store() as store:
            session = TutorSession(core=core, store=store, provider=_tutor_provider(provider))

            # ── Gate on a pending research request ───────────────────────
            if st.phase == "research_pending" and st.pending_request_id:
                # Self-heal if decided via the approvals panel.
                try:
                    req = store.get_research_request(st.pending_request_id)
                    rstatus = getattr(req.status, "value", req.status) if req else None
                    if rstatus == "approved":
                        st.phase = "idle"
                    elif rstatus in ("cancelled", "rejected", "completed"):
                        st.phase = "idle"
                        st.pending_request_id = None
                except Exception:
                    pass
            if st.phase == "research_pending":
                if st.pending_request_id and _APPROVE_RE.match(message):
                    res = self.decide_research(session_id, st.pending_request_id, "approve")
                    out["reply"] = (
                        "Investigación aprobada — corre en segundo plano. "
                        "Cuando llegue el material armo el roadmap."
                        if res.get("ok") else f"No pude aprobarla: {res.get('error')}"
                    )
                    self._record(core, session_id, message, out["reply"])
                    return out
                if st.pending_request_id and _REJECT_RE.match(message):
                    res = self.decide_research(session_id, st.pending_request_id, "reject")
                    out["reply"] = (
                        "Investigación cancelada. Decime cómo ajustar el enfoque."
                        if res.get("ok") else f"No pude cancelarla: {res.get('error')}"
                    )
                    self._record(core, session_id, message, out["reply"])
                    return out
                out["reply"] = (
                    "Hay una investigación pendiente de aprobación. "
                    "Usá el botón de la propuesta anterior o decime 'aprobado'."
                )
                self._record(core, session_id, message, out["reply"])
                return out

            # ── Gate on a pending roadmap proposal ──────────────────────
            if st.phase == "roadmap_proposed" and st.roadmap_id:
                # Self-heal: the gate may have been decided via the approvals
                # panel (unified endpoint) — sync the phase with store truth.
                try:
                    rm = store.get_roadmap(st.roadmap_id)
                    if rm is not None and rm.status == RoadmapStatus.APPROVED:
                        session.activate_roadmap(st.roadmap_id)
                        st.phase = "active"
                        self._ensure_progress(store, st.roadmap_id)
                    elif rm is not None and rm.status == RoadmapStatus.ACTIVE:
                        st.phase = "active"
                        self._ensure_progress(store, st.roadmap_id)
                    elif rm is not None and rm.status in (RoadmapStatus.REJECTED, RoadmapStatus.SUPERSEDED):
                        st.phase = "idle"
                        st.roadmap_id = None
                except Exception:
                    pass
            if st.phase == "roadmap_proposed" and st.roadmap_id:
                if _APPROVE_RE.match(message):
                    try:
                        session.approve_roadmap(st.roadmap_id, decided_by="dashboard")
                        session.activate_roadmap(st.roadmap_id)
                        st.phase = "active"
                        self._ensure_progress(store, st.roadmap_id)
                        roadmap = store.get_roadmap(st.roadmap_id)
                        first = roadmap.units[0] if roadmap and roadmap.units else None
                        out["reply"] = (
                            f"Roadmap aprobado y activo. Empezamos por la unidad 1"
                            + (f" ({first.concept_id})" if first else "")
                            + ". ¿Qué sabés ya de este tema? Así ajusto el punto de partida."
                        )
                        self._record(core, session_id, message, out["reply"])
                    except Exception as exc:
                        out["reply"] = f"Error al aprobar el roadmap: {exc}"
                    return out
                if _REJECT_RE.match(message):
                    try:
                        session.reject_roadmap(st.roadmap_id, decided_by="dashboard")
                        st.phase = "idle"
                        st.roadmap_id = None
                        out["reply"] = "Roadmap rechazado. Contame qué ajustar (enfoque, nivel, alcance) y propongo otro."
                        self._record(core, session_id, message, out["reply"])
                    except Exception as exc:
                        out["reply"] = f"Error al rechazar el roadmap: {exc}"
                    return out
                # ── Debatir: cualquier otro mensaje es feedback ──────────
                # El LLM re-propongo con el feedback incorporado (v+1,
                # supersedes); el gate humano sigue aplicando.
                try:
                    old = store.get_roadmap(st.roadmap_id)
                    hits = retrieve(st.topic) if (callable(retrieve) and st.topic) else []
                    concepts = self._concepts_from_hits(hits)
                    if len(concepts) < 3:
                        out["reply"] = (
                            "Hay un roadmap esperando tu aprobación. Para debatirlo "
                            "necesito material en el corpus — usá los botones o "
                            "decime 'aprobado' / 'rechazado'."
                        )
                        self._record(core, session_id, message, out["reply"])
                        return out
                except Exception as exc:
                    out["reply"] = f"No pude preparar el debate del roadmap: {exc}"
                    self._record(core, session_id, message, out["reply"])
                    return out
                try:
                    revised = session.propose_roadmap(
                        st.goal_id, concepts,
                        version=(old.version + 1) if old is not None else 1,
                        previous_roadmap_id=st.roadmap_id if old is not None else None,
                        change_reason=f"Debate del alumno: {message[:300]}",
                        feedback=message,
                    )
                    if old is not None:
                        try:
                            session.supersede_roadmap(st.roadmap_id, decided_by="dashboard")
                        except Exception:
                            pass
                    st.roadmap_id = revised.roadmap_id
                    rendered = self._render_roadmap(revised, concepts)
                    out["reply"] = (
                        f"Roadmap revisado (v{revised.version}) con tu feedback:\n\n{rendered}\n\n"
                        "Aprobá, rechazá, o seguí debatiendo."
                    )
                    out["roadmap_proposal"] = {
                        "roadmap_id": revised.roadmap_id,
                        "units": [
                            {"order": u.order, "concept_id": u.concept_id, "reason": u.reason,
                             "minutes": u.estimated_effort_minutes}
                            for u in revised.units
                        ],
                    }
                    self._record(core, session_id, message, out["reply"])
                except Exception as exc:
                    out["reply"] = f"No pude revisar el roadmap con ese feedback: {exc}"
                    self._record(core, session_id, message, out["reply"])
                return out

            # ── Lesson mode (active roadmap) ────────────────────────────
            if st.phase == "active" and st.topic_id:
                progress_note = None
                if st.roadmap_id:
                    self._ensure_progress(store, st.roadmap_id)
                    progress_note = self._progress_note(store, st.roadmap_id)
                from ipa.agent.provider_wiring import (
                    build_responder, build_streaming_responder,
                )
                # 768: una lección puede necesitar explicar conceptos con
                # detalle; 256/512 cortaban explicaciones a mitad.
                # Streaming: los tokens de la lección llegan a la UI en vivo
                # via on_token (los gates/roadmaps no streamean — son texto
                # renderizado o JSON estructurado, no generación de chat).
                if provider is not None and on_token is not None:
                    responder = build_streaming_responder(
                        _tutor_provider(provider), max_new_tokens=768,
                        on_token=on_token,
                    )
                else:
                    responder = build_responder(
                        _tutor_provider(provider), max_new_tokens=768,
                    ) if provider is not None else None
                result = session.lesson(
                    st.topic_id, message, responder=responder,
                    progress_note=progress_note,
                )
                out["reply"] = result["reply"]
                # Advance intent: "siguiente", "ya entendí", "avancemos"…
                # marks the current unit done and the next one current.
                if st.roadmap_id and _ADVANCE_RE.search(message):
                    adv = self._advance_unit(store, st.roadmap_id)
                    if adv is not None:
                        if adv["current"]:
                            out["reply"] += (
                                f"\n\n— Unidad {adv['done']} marcada como completada; "
                                f"seguimos con la unidad {adv['current']}."
                            )
                        else:
                            out["reply"] += (
                                f"\n\n— Unidad {adv['done']} completada: "
                                "llegaste al final del roadmap."
                            )
                        # Per-unit lesson summary → memory (granularidad
                        # pedagógica que el resumen de sesión comprime).
                        self._summarize_unit(
                            store, provider, st.roadmap_id, adv["done"], result,
                        )
                return out

            # ── Investigación aprobada en vuelo ────────────────────────
            # No hay nada accionable hasta que llegue el material: el
            # tutor responde de forma determinística (sin LLM, sin
            # proponer pasos que dependen de datos que todavía no están).
            if st.topic_id:
                try:
                    inflight = any(
                        r.concept_id == st.topic_id
                        and getattr(r.status, "value", r.status) in ("approved", "running")
                        for r in store.list_research_requests()
                    )
                except Exception:
                    inflight = False
                if inflight:
                    out["reply"] = (
                        "Aguardamos a que llegue la información de la fuente web. "
                        "Te aviso apenas termine la investigación y armo el roadmap."
                    )
                    self._record(core, session_id, message, out["reply"])
                    return out

            # ── Idle: detect topic or ask ───────────────────────────────
            topic = self._detect_topic(message)
            if topic is None and st.topic_id:
                # Resume pendiente: si una investigación aprobada para este
                # tema ya completó, el próximo mensaje ("dale", "seguí")
                # retoma el flujo con el tema guardado en vez de preguntar
                # de nuevo.
                try:
                    reqs = store.list_research_requests()
                    if any(
                        r.concept_id == st.topic_id
                        and getattr(r.status, "value", r.status) == "completed"
                        for r in reqs
                    ):
                        topic = st.topic
                except Exception:
                    pass
            if topic is None:
                out["reply"] = "¿Qué querés aprender? Decime el tema y armo el diagnóstico."
                self._record(core, session_id, message, out["reply"])
                return out

            st.topic = topic
            st.topic_id = _slug(topic)
            st.goal_id = f"goal:{st.topic_id}"

            diagnosis = session.diagnose(st.topic_id)

            # Concepts from the corpus (the roadmap grounds on real documents).
            hits = retrieve(topic) if callable(retrieve) else []
            concepts = self._concepts_from_hits(hits)

            if len(concepts) < 3:
                # Corpus can't support a roadmap — research gate (the Tutor may
                # call research_topic; the human approves before it executes).
                try:
                    req = session.create_research_request(
                        st.topic_id,
                        f"Material de estudio sobre {topic}",
                        goal_id=st.goal_id,
                    )
                    st.phase = "research_pending"
                    st.pending_request_id = req.request_id
                    out["reply"] = (
                        f"Diagnóstico: tema nuevo ({diagnosis.summary}). "
                        f"El corpus tiene poco material sobre {topic} ({len(concepts)} fuentes). "
                        "Propongo una investigación web primero — aprobala y en cuanto "
                        "llegue el material armo el roadmap."
                    )
                    out["research_proposal"] = {
                        "request_id": req.request_id,
                        "question": req.question,
                    }
                    self._record(core, session_id, message, out["reply"])
                except Exception as exc:
                    out["reply"] = f"Diagnóstico hecho ({diagnosis.summary}), pero falló la propuesta de investigación: {exc}"
                return out

            # Roadmap proposal (LLM picks/orders; human approves via gate).
            try:
                roadmap = session.propose_roadmap(st.goal_id, concepts)
                st.phase = "roadmap_proposed"
                st.roadmap_id = roadmap.roadmap_id
                rendered = self._render_roadmap(roadmap, concepts)
                out["reply"] = (
                    f"Diagnóstico: {diagnosis.summary}\n\n{rendered}\n\n"
                    "Aprobá o rechazá el roadmap (botones abajo o 'aprobado'/'rechazado')."
                )
                out["roadmap_proposal"] = {
                    "roadmap_id": roadmap.roadmap_id,
                    "units": [
                        {"order": u.order, "concept_id": u.concept_id, "reason": u.reason,
                         "minutes": u.estimated_effort_minutes}
                        for u in roadmap.units
                    ],
                    "items": rendered.split("\n")[1:],  # rendered lines, titles not ids
                }
                out["topic"] = topic
                self._record(core, session_id, message, out["reply"])
            except Exception as exc:
                out["reply"] = (
                    f"Diagnóstico: {diagnosis.summary}. "
                    f"No pude proponer el roadmap ({exc}). "
                    "Probá de nuevo o decime si querés que investigue más el tema."
                )
                self._record(core, session_id, message, out["reply"])
            return out

    def _record(self, core: Any, session_id: str, user_msg: str, reply: str) -> None:
        """Record a turn into the shared episodic memory (lesson() handles
        its own; proposal/gate turns record here)."""
        try:
            core.ensure_session()
            core.memory.record_episode(
                session_id, turn_role="user", content=user_msg,
                identity_hash=core.identity.identity_hash,
            )
            core.memory.record_episode(
                session_id, turn_role="assistant", content=reply,
                identity_hash=core.identity.identity_hash,
            )
        except Exception:
            pass

    # ── Gate endpoints (called by the approve/reject buttons) ────────────

    def _fresh_session(self, store: TutorStore) -> TutorSession:
        """A tutor-role session bound to the given store (gate decisions)."""
        from ipa.agent.agent_core import AgentCore
        return TutorSession(
            core=AgentCore(interface="dashboard", role="tutor"),
            store=store, provider=None,
        )

    def decide_roadmap(self, session_id: str, roadmap_id: str, decision: str) -> dict[str, Any]:
        """Button gate: approve+activate or reject a proposed roadmap."""
        st = self.state(session_id)
        with self._store() as store:
            session = self._fresh_session(store)
            try:
                if decision == "approve":
                    session.approve_roadmap(roadmap_id, decided_by="dashboard")
                    session.activate_roadmap(roadmap_id)
                    self._ensure_progress(store, roadmap_id)
                    if st.roadmap_id == roadmap_id:
                        st.phase = "active"
                    return {"ok": True, "status": "active"}
                session.reject_roadmap(roadmap_id, decided_by="dashboard")
                if st.roadmap_id == roadmap_id:
                    st.phase = "idle"
                    st.roadmap_id = None
                return {"ok": True, "status": "rejected"}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            finally:
                try:
                    session.core.memory.close()
                except Exception:
                    pass

    def decide_research(self, session_id: str, request_id: str, decision: str) -> dict[str, Any]:
        st = self.state(session_id)
        with self._store() as store:
            session = self._fresh_session(store)
            try:
                if decision == "approve":
                    session.approve_research_request(request_id, decided_by="dashboard")
                    if st.pending_request_id == request_id:
                        st.phase = "idle"  # research runs in background; topic known
                    return {"ok": True, "status": "approved", "request_id": request_id}
                session.reject_research_request(request_id, decided_by="dashboard")
                if st.pending_request_id == request_id:
                    st.phase = "idle"
                    st.pending_request_id = None
                return {"ok": True, "status": "cancelled"}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            finally:
                try:
                    session.core.memory.close()
                except Exception:
                    pass


_DRIVER: TutorChatDriver | None = None


def get_tutor_driver() -> TutorChatDriver:
    """Process-level singleton (single-user dashboard)."""
    global _DRIVER
    if _DRIVER is None:
        _DRIVER = TutorChatDriver()
    return _DRIVER


__all__ = ["TutorChatDriver", "get_tutor_driver"]
