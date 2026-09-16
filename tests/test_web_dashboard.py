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
