"""Landing sweep lifecycle — synthetic fixtures, no real data touched.

Covers the transit-zone policy enforced by
``ipa.ingestion.landing_sweep.sweep_landing``:

- approved (live doc in main corpus)  -> Archive/
- pending (staging-only / review pending) -> Transit/
- rejected (failed everywhere or review_status=rejected) -> deleted
- unregistered -> stays in Landing
- Transit re-scan: promoted -> Archive, rejected -> delete
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ipa.ingestion.landing_sweep import sweep_landing
from ipa.ingestion.landing_zone import _sha256


def _mk_corpus(root: Path) -> Path:
    corpus = root
    corpus.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(corpus / "landing.db"))
    conn.execute("CREATE TABLE artifacts (artifact_id TEXT, status TEXT, source_uri TEXT)")
    conn.commit()
    conn.close()
    conn = sqlite3.connect(str(corpus / "document_store.db"))
    conn.execute("CREATE TABLE documents (artifact_id TEXT, document_id TEXT, tombstoned INT)")
    conn.commit()
    conn.close()
    return corpus


def _register(corpus: Path, artifact_id: str, status: str, source_uri: str) -> None:
    conn = sqlite3.connect(str(corpus / "landing.db"))
    conn.execute("INSERT INTO artifacts VALUES (?,?,?)", (artifact_id, status, source_uri))
    conn.commit()
    conn.close()


def _store_doc(corpus: Path, artifact_id: str, document_id: str, tombstoned: int = 0) -> None:
    conn = sqlite3.connect(str(corpus / "document_store.db"))
    conn.execute("INSERT INTO documents VALUES (?,?,?)", (artifact_id, document_id, tombstoned))
    conn.commit()
    conn.close()


@pytest.fixture()
def env(tmp_path: Path):
    landing = tmp_path / "Landing"
    landing.mkdir()
    return {
        "landing": landing,
        "archive": tmp_path / "Archive",
        "transit": tmp_path / "Transit",
        "staging": _mk_corpus(tmp_path / "staging"),
        "main": _mk_corpus(tmp_path / "main"),
    }


def _sweep(env, **kwargs) -> dict:
    return sweep_landing(
        env["landing"], env["archive"], [env["staging"], env["main"]],
        main_corpus=env["main"], transit_root=env["transit"], **kwargs,
    )


def _artifact(landing: Path, rel: str, content: str) -> tuple[Path, str]:
    path = landing / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path, "sha256:" + _sha256(path)


def test_staged_indexed_goes_to_transit(env):
    path, aid = _artifact(env["landing"], "web/site-a/docA.md", "A")
    _register(env["staging"], aid, "indexed", str(path))
    _store_doc(env["staging"], aid, "doc-A")

    stats = _sweep(env)

    assert stats["transit"] == 1 and stats["errors"] == []
    assert (env["transit"] / "web/site-a/docA.md").exists()
    assert not path.exists()


def test_failed_artifact_is_deleted(env):
    path, aid = _artifact(env["landing"], "web/site-b/docB.md", "B")
    _register(env["staging"], aid, "failed", str(path))

    stats = _sweep(env)

    assert stats["deleted"] == 1
    assert not path.exists()


def test_no_text_artifact_is_deleted(env):
    """Parse OK pero sin texto utilizable → igual que failed: delete."""
    path, aid = _artifact(env["landing"], "web/site-x/blank.pdf", "BLANK")
    _register(env["staging"], aid, "no_text", str(path))

    stats = _sweep(env)

    assert stats["deleted"] == 1
    assert not path.exists()


def test_no_text_artifact_deleted_from_transit(env):
    """Un no_text que ya había pasado a Transit se borra en la re-evaluación."""
    path, aid = _artifact(env["landing"], "web/site-x/blank.pdf", "BLANK")
    _register(env["staging"], aid, "indexed", str(path))
    _store_doc(env["staging"], aid, "doc-blank")
    _sweep(env)
    transit_file = env["transit"] / "web/site-x/blank.pdf"
    assert transit_file.exists()

    # La curación/corrección posterior lo reclasifica como no_text.
    _register(env["staging"], aid, "no_text", str(path))

    stats = _sweep(env)

    assert stats["deleted"] == 1
    assert not transit_file.exists()


def test_unregistered_file_stays(env):
    _artifact(env["landing"], "web/site-c/docC.md", "unknown-content")

    stats = _sweep(env)

    assert stats["skipped"] == 1
    assert (env["landing"] / "web/site-c/docC.md").exists()


def test_main_corpus_membership_archives_even_from_transit(env):
    path, aid = _artifact(env["landing"], "web/site-a/docA.md", "A")
    _register(env["staging"], aid, "indexed", str(path))
    _store_doc(env["staging"], aid, "doc-A")
    _sweep(env)
    transit_file = env["transit"] / "web/site-a/docA.md"
    assert transit_file.exists()

    # Promotion: the document now lives in the main corpus.
    _store_doc(env["main"], aid, "doc-A")

    stats = _sweep(env)

    assert stats["archived"] == 1
    assert (env["archive"] / "web/site-a/docA.md").exists()
    assert not transit_file.exists()


def test_human_rejection_deletes(env):
    path, aid = _artifact(env["landing"], "web/site-d/docD.md", "D")
    _register(env["staging"], aid, "indexed", str(path))
    _store_doc(env["staging"], aid, "doc-D")

    cluster = env["staging"].parent / "cluster.db"
    conn = sqlite3.connect(str(cluster))
    conn.execute("CREATE TABLE curation_decisions (payload_json TEXT, created_at TEXT)")
    conn.execute(
        "INSERT INTO curation_decisions VALUES (?,?)",
        (json.dumps({"document_id": "doc-D", "review_status": "rejected"}), "2026-09-13"),
    )
    conn.commit()
    conn.close()

    stats = _sweep(env, cluster_db=cluster)

    assert stats["deleted"] == 1
    assert not path.exists()
    assert not (env["transit"] / "web/site-d/docD.md").exists()


def test_operational_files_never_touched(env):
    history = env["landing"] / "web" / "scrape_history.db"
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_bytes(b"sqlite")
    report = env["landing"] / "web" / "scrape_report.json"
    report.write_text("{}", encoding="utf-8")

    _sweep(env)

    assert history.exists() and report.exists()


def test_duplicate_content_dedup_on_move(env):
    path, aid = _artifact(env["landing"], "web/site-a/docA.md", "A")
    _register(env["staging"], aid, "indexed", str(path))
    _store_doc(env["staging"], aid, "doc-A")
    _sweep(env)
    first = env["transit"] / "web/site-a/docA.md"

    # Same content re-downloaded under a new name in Landing.
    path2, _ = _artifact(env["landing"], "web/site-a/docA_copy.md", "A")
    _register(env["staging"], aid, "indexed", str(path2))
    _sweep(env)

    assert first.exists()
    assert not path2.exists()
    assert not (env["transit"] / "web/site-a/docA_copy.md").exists()
