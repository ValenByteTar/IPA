from __future__ import annotations

import json

import pytest

from ipa.dashboard import state
from ipa.dashboard import server as dashboard


def test_safe_url_accepts_http_and_https():
    assert state.safe_url("https://example.com/news") == "https://example.com/news"
    assert state.safe_url("http://localhost:8080") == "http://localhost:8080"


def test_safe_url_rejects_credentials_and_invalid_scheme():
    with pytest.raises(ValueError):
        state.safe_url("ftp://example.com/file")
    with pytest.raises(ValueError):
        state.safe_url("https://user:secret@example.com/news")


def test_read_document_rejects_paths_outside_allowed_roots(tmp_path):
    path = tmp_path / "secret.txt"
    path.write_text("secret", encoding="utf-8")
    with pytest.raises(PermissionError):
        dashboard.read_document(str(path))


def test_sources_round_trip_uses_atomic_json(tmp_path, monkeypatch):
    sources_path = tmp_path / "sources.json"
    monkeypatch.setattr(state, "SOURCES_DB", sources_path)
    payload = {"added": [{"url": "https://example.com/news"}], "disabled": []}
    state.save_sources(payload)
    assert state.load_sources() == payload
    assert not sources_path.with_suffix(".tmp").exists()


def test_effective_scrape_config_applies_additions_and_disables(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    base.write_text("sites:\n  - url: https://base.example/news\n", encoding="utf-8")
    monkeypatch.setattr(state, "SCRAPE_CONFIG", base)
    monkeypatch.setattr(state, "SOURCES_DB", tmp_path / "sources.json")
    monkeypatch.setattr(state, "ROOT", tmp_path)
    state.save_sources({
        "added": [{"url": "https://added.example/news", "days_back": 7}],
        "disabled": ["https://base.example/news"],
    })
    result = state.effective_scrape_config()
    data = json.loads(json.dumps(__import__("yaml").safe_load(result.read_text(encoding="utf-8"))))
    assert [site["url"] for site in data["sites"]] == ["https://added.example/news"]


def test_topic_edit_updates_report_and_reporter_store(tmp_path, monkeypatch):
    report_dir = tmp_path / "reporter" / "period"
    report_dir.mkdir(parents=True)
    report_path = report_dir / "report.json"
    report_path.write_text(json.dumps({
        "report_id": "r1", "status": "draft", "categories": [{"category_id": "topic-1", "label": "Old", "description": "Old description"}],
    }), encoding="utf-8")
    monkeypatch.setattr(dashboard, "REPORTER_ROOT", tmp_path / "reporter")
    from ipa.reporter.reporter_store import ReporterStore
    with ReporterStore(report_dir / "reporter.db") as store:
        store.put_topic("topic-1", "r1", {"category_id": "topic-1", "label": "Old", "description": "Old description"})
        store.commit()
    result = dashboard.update_latest_topic({"category_id": "topic-1", "label": "New", "description": "New description", "status": "reviewed"})
    assert result["status"] == "reviewed"
    assert json.loads(report_path.read_text(encoding="utf-8"))["categories"][0]["label"] == "New"
    from ipa.reporter.reporter_store import ReporterStore as RS
    with RS(report_dir / "reporter.db") as store:
        row = store._conn.execute("SELECT payload_json FROM topic_clusters WHERE category_id='topic-1'").fetchone()
        assert json.loads(row[0])["description"] == "New description"


def test_topic_edit_rejects_invalid_status(tmp_path, monkeypatch):
    report_dir = tmp_path / "reporter"
    report_dir.mkdir()
    (report_dir / "report.json").write_text(json.dumps({"report_id": "r1", "status": "draft", "categories": [{"category_id": "t", "label": "x", "description": "y"}]}), encoding="utf-8")
    monkeypatch.setattr(dashboard, "REPORTER_ROOT", tmp_path / "reporter")
    with pytest.raises(ValueError):
        dashboard.update_latest_topic({"category_id": "t", "status": "invalid"})


def test_directory_backend_is_reported_without_sqlite_error(tmp_path):
    directory = tmp_path / "lancedb"
    directory.mkdir()
    (directory / "data.bin").write_bytes(b"data")
    result = dashboard.db_counts(directory)
    assert result["backend"] == "directory"
    assert result["files"] == 1


# ── Perilla idle enrichment (sidebar): ON/OFF Tier 1 y Tier 2 ───────────────

def test_idle_switch_persists_across_boot(tmp_path, monkeypatch):
    """set_idle_enabled persiste la decisión; el boot-read la respeta — un
    restart del dashboard no re-enciende el enrichment silenciosamente."""
    monkeypatch.setattr(dashboard, "IDLE_ENABLED_PATH", tmp_path / "idle_enabled.json")
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    dashboard.IDLE_ENABLED["enabled"] = True  # reset estado del proceso
    dashboard._read_idle_enabled_at_boot()
    assert dashboard.idle_enabled_state() is True  # default ON

    dashboard.set_idle_enabled(False)
    assert dashboard.idle_enabled_state() is False
    persisted = json.loads((tmp_path / "idle_enabled.json").read_text(encoding="utf-8"))
    assert persisted == {"enabled": False}
    # Auditoría en el log del worker
    log = tmp_path / "outputs" / "web_dashboard" / "logs" / "idle_enrichment.log"
    assert "DISABLED by user" in log.read_text(encoding="utf-8")

    # Restart simulado: el boot-read respeta la decisión persistida.
    dashboard.IDLE_ENABLED["enabled"] = True
    dashboard._read_idle_enabled_at_boot()
    assert dashboard.idle_enabled_state() is False

    dashboard.set_idle_enabled(True)
    assert dashboard.idle_enabled_state() is True


# ── Deep dive consolidado en el chat (context=deep_dive) ────────────────────

def test_deep_dive_context_absent_returns_none():
    from ipa.dashboard.api import parse_deep_dive_context
    assert parse_deep_dive_context({}) is None
    assert parse_deep_dive_context({"context": "chat"}) is None


def test_deep_dive_context_valid_reporter_corpus():
    from ipa.dashboard.api import parse_deep_dive_context
    corpus = dashboard.REPORTER_ROOT / "some-report" / "corpus"
    ctx = parse_deep_dive_context({
        "context": "deep_dive", "corpus": str(corpus),
        "category_id": "cat:1", "search": "rag agentes",
    })
    assert ctx is not None
    assert ctx["corpus"] == corpus.resolve()
    assert ctx["category_id"] == "cat:1"
    assert ctx["search"] == "rag agentes"


def test_deep_dive_context_rejects_corpus_outside_reporter_root():
    from ipa.dashboard.api import parse_deep_dive_context
    with pytest.raises(PermissionError):
        parse_deep_dive_context({
            "context": "deep_dive", "corpus": str(dashboard.ROOT / "Landing"),
        })
    with pytest.raises(PermissionError):
        parse_deep_dive_context({"context": "deep_dive", "corpus": "C:/Windows"})


def test_retrieval_cache_holds_only_thread_safe_handles(tmp_path):
    """Regression: caching DocumentStore (sqlite) in _RETRIEVAL_STORES caused
    cross-thread errors when the tutor path used it from a request thread."""
    import sqlite3
    import threading

    from ipa.dashboard import api as api_mod
    from ipa.indexes.lancedb_index import LanceDBIndex

    corpus = tmp_path / "corpus"
    lance = api_mod._retrieval_lance(corpus)
    assert isinstance(lance, LanceDBIndex)
    assert api_mod._retrieval_lance(corpus) is lance
    for cached in api_mod._RETRIEVAL_STORES.values():
        assert not isinstance(cached, (tuple, list)), (
            "cached tuples carried a thread-bound DocumentStore"
        )
        assert not any(
            isinstance(v, sqlite3.Connection)
            for v in vars(cached).values()
        ), "cached object holds a sqlite connection"


def test_sqlite_connection_cross_thread_raises(tmp_path):
    """Documents the failure mode the fix avoids: a sqlite3.Connection created
    in one thread cannot be used in another."""
    import sqlite3
    import threading

    db = tmp_path / "t.db"
    conn_holder: list[sqlite3.Connection] = []
    done = threading.Event()

    def _owner():
        conn = sqlite3.connect(str(db))
        conn_holder.append(conn)
        done.wait(timeout=5)
        conn.close()

    t = threading.Thread(target=_owner)
    t.start()
    while not conn_holder:
        t.join(timeout=0.05)
    conn = conn_holder[0]
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
    # close() is also thread-bound — the owning thread must close it.
    done.set()
    t.join()


# ---------------------------------------------------------------------------
# Roadmaps tab read model (Fase: pestaña de proyectos del Tutor)
# ---------------------------------------------------------------------------

def _seed_tutor_store(tmp_path):
    """Roadmap + LearningGoal sembrados en el store default (cwd-patcheado)."""
    from datetime import datetime, timezone
    from ipa.tutor.tutor_runtime import TutorStore
    from ipa.tutor.tutor_contracts import (
        AssessmentType, GenerationProvenance, HumanApproval,
        HumanApprovalDecision, LearningGoal, LearningGoalStatus,
        Roadmap, RoadmapStatus, RoadmapUnit, SourceRef, SourceType,
    )
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    gen = GenerationProvenance(
        generator="t", generated_at=now,
        input_hash="sha256:" + "a" * 64, model_fingerprint="m",
    )
    store = TutorStore(tmp_path / "outputs/agent/tutor.db")
    goal = LearningGoal(
        goal_id="goal:demo", title="Proyecto Demo",
        description="Dominar el tema demo",
        status=LearningGoalStatus.ACTIVE,
        success_criteria=["Explicar demo", "Aplicar demo"],
        constraints=["Nivel avanzado"],
        created_at=now, updated_at=now,
        approval=HumanApproval(
            decision=HumanApprovalDecision.APPROVED,
            decided_at=now, decided_by="dashboard",
        ),
        field_origins={
            "title": "user", "description": "user",
            "success_criteria": "generated", "constraints": "user",
            "status": "system",
        },
        generation=gen,
    )
    store.save_goal(goal)
    units = [
        RoadmapUnit(
            unit_id=f"roadmap_unit:u{i}", order=i, concept_id=f"doc:c{i}",
            reason=f"razón {i}", estimated_effort_minutes=30,
            source_refs=[SourceRef(source_id=f"doc:c{i}", source_type=SourceType.CHUNK)],
            assessment_types=[AssessmentType.EXPLANATION],
        )
        for i in (1, 2, 3)
    ]
    rm = Roadmap(
        roadmap_id="roadmap:demo1", goal_id="goal:demo", version=1,
        status=RoadmapStatus.ACTIVE, units=units,
        assumptions=["asunción 1"], uncertainties=["duda 1"],
        change_reason=None, previous_roadmap_id=None, created_at=now,
        approval=HumanApproval(
            decision=HumanApprovalDecision.APPROVED,
            decided_at=now, decided_by="dashboard",
        ),
        generation=gen,
        field_origins={
            "goal_id": "user", "units": "generated",
            "assumptions": "generated", "uncertainties": "generated",
            "change_reason": "user_or_generated",
        },
    )
    store.save_roadmap(rm)
    store.set_unit_status("roadmap:demo1", 1, "done")
    store.set_unit_status("roadmap:demo1", 2, "current")
    store.set_unit_status("roadmap:demo1", 3, "pending")
    store.set_focus("roadmap:demo1")
    return store, rm, goal


def test_tutor_projects_lists_goals_with_roadmaps(tmp_path, monkeypatch):
    """GET /api/tutor/projects: proyectos (goals) con sus roadmaps."""
    monkeypatch.chdir(tmp_path)
    from ipa.dashboard.api import tutor_projects_payload
    store, rm, goal = _seed_tutor_store(tmp_path)
    store.close()
    payload = tutor_projects_payload()
    assert len(payload["projects"]) == 1
    proj = payload["projects"][0]
    assert proj["goal"]["goal_id"] == "goal:demo"
    assert proj["goal"]["title"] == "Proyecto Demo"
    assert proj["goal"]["status"] == "active"
    assert proj["goal"]["success_criteria"] == ["Explicar demo", "Aplicar demo"]
    assert proj["goal"]["approved"] is True
    assert proj["roadmaps"][0]["roadmap_id"] == rm.roadmap_id


def test_tutor_roadmap_context_full_read_model(tmp_path, monkeypatch):
    """GET /api/tutor/roadmap/context: objetivo + porqué + avance + conceptos
    + foco — todo lo que la pestaña renderiza."""
    monkeypatch.chdir(tmp_path)
    from ipa.dashboard.api import tutor_roadmap_context
    store, rm, goal = _seed_tutor_store(tmp_path)
    store.close()
    ctx = tutor_roadmap_context(rm.roadmap_id)
    assert ctx["ok"] is True
    assert ctx["status"] == "active" and ctx["is_focus"] is True
    # Objetivo (LearningGoal persistido, no sintético)
    assert ctx["goal"]["title"] == "Proyecto Demo"
    assert ctx["goal"]["constraints"] == ["Nivel avanzado"]
    # Porqué del roadmap (racionalidad antes invisible)
    assert ctx["rationale"]["assumptions"] == ["asunción 1"]
    assert ctx["rationale"]["uncertainties"] == ["duda 1"]
    # Avance por unidad (la barra de progreso)
    assert ctx["progress"] == {"done": 1, "current": 2, "total": 3}
    statuses = {u["order"]: u["status"] for u in ctx["units"]}
    assert statuses == {1: "done", 2: "current", 3: "pending"}
    # Conceptos re-derivados (sin corpus en tmp → label honesto)
    assert ctx["units"][0]["concept"]["concept_id"] == "doc:c1"
    assert ctx["units"][0]["concept"]["title"]


def test_tutor_roadmap_context_synthesizes_legacy_goal(tmp_path, monkeypatch):
    """Un roadmap previo a learning_goals recibe goal sintetizado al leerse:
    active + approval reutilizado del roadmap."""
    monkeypatch.chdir(tmp_path)
    from ipa.dashboard.api import tutor_roadmap_context
    store, rm, _ = _seed_tutor_store(tmp_path)
    store._conn.execute("DELETE FROM learning_goals")
    store._conn.commit()
    store.close()
    ctx = tutor_roadmap_context(rm.roadmap_id)
    assert ctx["ok"] is True
    assert ctx["goal"]["status"] == "active"
    assert ctx["goal"]["approved"] is True
    assert ctx["goal"]["title"] == "demo"  # slug → título


def test_tutor_roadmap_context_unknown_returns_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from ipa.dashboard.api import tutor_roadmap_context
    ctx = tutor_roadmap_context("roadmap:inexistente")
    assert ctx["ok"] is False
    assert "unknown roadmap" in ctx["error"]
