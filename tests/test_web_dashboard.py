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
