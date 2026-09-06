"""SQLite persistence for Reporter metadata, decisions and topics."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ipa.reporter.reporter_contracts import ReporterDocumentDecision, TopicLink
from ipa.reporter.reporter_metadata import ReporterDocumentMetadata

_SCHEMA = """
CREATE TABLE IF NOT EXISTS document_metadata (
 document_id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL, original_path TEXT NOT NULL,
 source_url TEXT, canonical_url TEXT, source_domain TEXT, title TEXT NOT NULL,
 published_at TEXT, published_at_confidence TEXT NOT NULL, scraped_at TEXT,
 content_hash TEXT NOT NULL, quality_score REAL, mime_type TEXT, metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS document_decisions (
 decision_id TEXT PRIMARY KEY, report_id TEXT NOT NULL, document_id TEXT NOT NULL,
 payload_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS topic_clusters (
 category_id TEXT PRIMARY KEY, report_id TEXT NOT NULL, payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS topic_documents (
 category_id TEXT NOT NULL, document_id TEXT NOT NULL, membership_score REAL NOT NULL,
 membership_reason TEXT, PRIMARY KEY(category_id, document_id)
);
CREATE TABLE IF NOT EXISTS topic_links (
 current_category_id TEXT NOT NULL, previous_category_id TEXT, payload_json TEXT NOT NULL,
 PRIMARY KEY(current_category_id, previous_category_id)
);
CREATE TABLE IF NOT EXISTS report_runs (
 report_id TEXT PRIMARY KEY, corpus_id TEXT NOT NULL, period_start TEXT NOT NULL,
 period_end TEXT NOT NULL, status TEXT NOT NULL, config_fingerprint TEXT NOT NULL,
 corpus_fingerprint TEXT NOT NULL, report_json_path TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS promotion_requests (
 promotion_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, decision_id TEXT NOT NULL,
 status TEXT NOT NULL, approval_json TEXT, target_corpus TEXT, created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL
);
"""


class ReporterStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ReporterStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def put_run(self, report_id: str, corpus_id: str, period_start: str, period_end: str, status: str, config_fingerprint: str, corpus_fingerprint: str, report_json_path: str, created_at: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO report_runs VALUES (?,?,?,?,?,?,?,?,?)",
            (report_id, corpus_id, period_start, period_end, status, config_fingerprint, corpus_fingerprint, report_json_path, created_at),
        )

    def put_metadata(self, metadata: ReporterDocumentMetadata) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO document_metadata VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (metadata.document_id, metadata.artifact_id, metadata.original_path,
             metadata.source_url, metadata.canonical_url, metadata.source_domain,
             metadata.title, metadata.published_at, metadata.published_at_confidence,
             metadata.scraped_at, metadata.content_hash, metadata.quality_score,
             metadata.mime_type, json.dumps(metadata.metadata, ensure_ascii=False)),
        )

    def all_metadata(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM document_metadata ORDER BY document_id").fetchall()
        columns = [column[1] for column in self._conn.execute("PRAGMA table_info(document_metadata)")]
        result = []
        for row in rows:
            item = dict(zip(columns, row))
            item["metadata"] = json.loads(item.pop("metadata_json"))
            result.append(item)
        return result

    def put_decision(self, decision: ReporterDocumentDecision, created_at: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO document_decisions VALUES (?,?,?, ?,?)",
            (decision.decision_id, decision.report_id, decision.document_id,
             json.dumps(decision.to_dict(), ensure_ascii=False), created_at),
        )

    def put_topic(self, category_id: str, report_id: str, payload: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO topic_clusters VALUES (?,?,?)",
            (category_id, report_id, json.dumps(payload, ensure_ascii=False)),
        )

    def update_topic(self, category_id: str, report_id: str, updates: dict[str, Any]) -> bool:
        """Update human-editable topic fields without changing its membership."""
        row = self._conn.execute(
            "SELECT payload_json FROM topic_clusters WHERE category_id=? AND report_id=?",
            (category_id, report_id),
        ).fetchone()
        if not row:
            return False
        payload = json.loads(row[0])
        payload.update(updates)
        self._conn.execute(
            "UPDATE topic_clusters SET payload_json=? WHERE category_id=? AND report_id=?",
            (json.dumps(payload, ensure_ascii=False), category_id, report_id),
        )
        return True

    def put_topic_document(self, category_id: str, document_id: str, score: float, reason: str = "") -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO topic_documents VALUES (?,?,?,?)",
            (category_id, document_id, score, reason),
        )

    def put_topic_link(self, link: TopicLink) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO topic_links VALUES (?,?,?)",
            (link.current_category_id, link.previous_category_id,
             json.dumps(link.to_dict(), ensure_ascii=False)),
        )

    def create_promotion(self, promotion_id: str, document_id: str, decision_id: str, target_corpus: str, created_at: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO promotion_requests VALUES (?,?,?,?,?,?,?,?)",
            (promotion_id, document_id, decision_id, "pending", None, target_corpus, created_at, created_at),
        )

    def approve_promotion(self, promotion_id: str, approval: dict[str, Any], updated_at: str) -> None:
        if approval.get("decision") != "approved":
            raise ValueError("promotion approval must be approved")
        self._conn.execute(
            "UPDATE promotion_requests SET status='approved', approval_json=?, updated_at=? WHERE promotion_id=?",
            (json.dumps(approval, ensure_ascii=False), updated_at, promotion_id),
        )

    def list_promotions(self, status: str = "pending") -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT promotion_id, document_id, decision_id, status, approval_json, target_corpus, created_at, updated_at FROM promotion_requests WHERE status=? ORDER BY created_at",
            (status,),
        ).fetchall()
        keys = ["promotion_id", "document_id", "decision_id", "status", "approval", "target_corpus", "created_at", "updated_at"]
        result = []
        for row in rows:
            item = dict(zip(keys, row))
            item["approval"] = json.loads(item["approval"]) if item["approval"] else None
            result.append(item)
        return result

    def commit(self) -> None:
        self._conn.commit()

