"""Landing Zone â€” durable registry of received artifacts.

Stores artifact metadata in SQLite without modifying or copying the original
files.  Supports idempotent re-ingestion: an artifact with the same
content_hash is not re-registered.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Iterator

from ipa.contracts import ArtifactRef

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id      TEXT PRIMARY KEY,
    content_hash     TEXT NOT NULL,
    source_uri       TEXT NOT NULL,
    mime_type        TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    byte_size        INTEGER NOT NULL,
    received_at      TEXT NOT NULL,
    registered_at    TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'received',
    attempts         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS stage_status (
    artifact_id  TEXT NOT NULL,
    stage        TEXT NOT NULL,
    status       TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (artifact_id, stage),
    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id)
);
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LandingZone:
    """Durable artifact registry backed by SQLite."""

    def __init__(self, db_path: str | Path, root: str | Path | None = None) -> None:
        self.db_path = Path(db_path)
        self.root = Path(root).resolve() if root is not None else None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        self._conn.execute("PRAGMA busy_timeout = 30000")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA temp_store = MEMORY")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "LandingZone":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def iter_files(self) -> Iterator[Path]:
        """Yield files from the configured Landing directory in stable order."""
        if self.root is None:
            raise RuntimeError("LandingZone root is not configured")
        if not self.root.exists():
            return
        yield from sorted(
            path for path in self.root.rglob("*")
            if path.is_file() and not path.name.startswith(".")
        )

    def register_directory(self, root: str | Path | None = None) -> list[ArtifactRef]:
        """Register every file in the configured or supplied Landing directory."""
        if root is not None:
            root_path = Path(root).resolve()
            files = sorted(
                path for path in root_path.rglob("*")
                if path.is_file() and not path.name.startswith(".")
            )
        else:
            files = list(self.iter_files())
        return [self.register(path) for path in files]

    def register(self, path: Path) -> ArtifactRef:
        """Register a single file.  Idempotent: same hash returns existing ref."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        digest = _sha256(path)
        artifact_id = f"sha256:{digest}"
        stat = path.stat()
        received_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime))
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        ref = ArtifactRef(
            artifact_id=artifact_id,
            content_hash=artifact_id,
            source_uri=str(path.resolve()),
            mime_type="",
            original_filename=path.name,
            byte_size=stat.st_size,
            received_at=received_at,
        )

        for attempt in range(4):
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO artifacts "
                    "(artifact_id, content_hash, source_uri, mime_type, original_filename, "
                    "byte_size, received_at, registered_at, status, attempts) "
                    "VALUES (?,?,?,?,?,?,?,?, 'received', 0)",
                    (ref.artifact_id, ref.content_hash, ref.source_uri, "",
                     ref.original_filename, ref.byte_size, ref.received_at, now),
                )
                self._conn.commit()
                return ref
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 3:
                    raise
                self._conn.rollback()
                time.sleep(0.5 * (attempt + 1))
        return ref

    def set_mime_type(self, artifact_id: str, mime_type: str) -> None:
        self._conn.execute(
            "UPDATE artifacts SET mime_type = ? WHERE artifact_id = ?",
            (mime_type, artifact_id),
        )

    def set_status(self, artifact_id: str, status: str) -> None:
        """Set the document lifecycle status."""
        self._conn.execute(
            "UPDATE artifacts SET status = ? WHERE artifact_id = ?",
            (status, artifact_id),
        )

    def get_status(self, artifact_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT status FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        return row[0] if row else None

    def set_stage(self, artifact_id: str, stage: str, status: str) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._conn.execute(
            "INSERT OR REPLACE INTO stage_status (artifact_id, stage, status, updated_at) "
            "VALUES (?,?,?,?)",
            (artifact_id, stage, status, now),
        )

    def commit(self) -> None:
        """Flush pending writes to disk. Call after a batch of updates."""
        self._conn.commit()

    def get_stage(self, artifact_id: str, stage: str) -> str | None:
        row = self._conn.execute(
            "SELECT status FROM stage_status WHERE artifact_id = ? AND stage = ?",
            (artifact_id, stage),
        ).fetchone()
        return row[0] if row else None

    def list_artifacts(self) -> Iterator[ArtifactRef]:
        rows = self._conn.execute(
            "SELECT artifact_id, content_hash, source_uri, mime_type, "
            "original_filename, byte_size, received_at FROM artifacts ORDER BY artifact_id"
        ).fetchall()
        for row in rows:
            yield ArtifactRef(*row)

    def get_artifact(self, artifact_id: str) -> ArtifactRef | None:
        row = self._conn.execute(
            "SELECT artifact_id, content_hash, source_uri, mime_type, "
            "original_filename, byte_size, received_at FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if not row:
            return None
        return ArtifactRef(*row)

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()
        return row[0]

    def _lifecycle_for(self, artifact_id: str) -> tuple[str, int]:
        row = self._conn.execute(
            "SELECT status, attempts FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        return (row[0], row[1]) if row else ("received", 0)

    def export_manifest(self, output_path: str | Path) -> str:
        """Export all artifacts as a JSONL manifest. Returns the manifest hash."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for ref in self.list_artifacts():
            lifecycle, attempts = self._lifecycle_for(ref.artifact_id)
            record = {
                "artifact_id": ref.artifact_id,
                "content_hash": ref.content_hash,
                "source_uri": ref.source_uri,
                "original_filename": ref.original_filename,
                "mime_type": ref.mime_type,
                "byte_size": ref.byte_size,
                "received_at": ref.received_at,
                "status": lifecycle,
                "attempts": attempts,
                "stages": self._stages_for(ref.artifact_id),
            }
            lines.append(json.dumps(record, ensure_ascii=False))
        content = "".join(line + "\n" for line in lines)
        output_path.write_text(content, encoding="utf-8")
        return "sha256:" + hashlib.sha256(output_path.read_bytes()).hexdigest()

    def _stages_for(self, artifact_id: str) -> dict[str, str]:
        defaults = {
            "acquisition": "success", "parsing": "pending",
            "chunking": "pending", "bm25": "pending",
            "embedding": "pending", "enrichment": "pending",
        }
        rows = self._conn.execute(
            "SELECT stage, status FROM stage_status WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchall()
        for stage, status in rows:
            defaults[stage] = status
        return defaults

