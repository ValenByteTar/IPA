"""Durable episodic memory for the personal agent.

Canonical flat store (SQLite, append-only) at ``outputs/agent/agent.db``:
sessions and episodes. Any vector index over episodes is a derived
representation rebuilt from this store (Fase 3, DEC-002). Episodes are personal
conversations: this store lives outside any corpus and never leaves the machine.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MEMORY_PATH = Path(os.environ.get("IPA_AGENT_STORE", "outputs/agent/agent.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id      TEXT PRIMARY KEY,
    interface       TEXT NOT NULL,
    role            TEXT NOT NULL,
    status          TEXT NOT NULL,
    title           TEXT,
    started_at      TEXT NOT NULL,
    last_active_at  TEXT NOT NULL,
    identity_hash   TEXT NOT NULL,
    episode_count   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS agent_episodes (
    episode_id      TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES agent_sessions(session_id),
    interface       TEXT NOT NULL,
    role            TEXT NOT NULL,
    turn_role       TEXT NOT NULL,
    content         TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    identity_hash   TEXT NOT NULL,
    topic_cluster_id TEXT,
    tool_calls      TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON agent_episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_created ON agent_episodes(created_at);
"""

# Migraciones aditivas (columnas derivadas de consolidación de sesiones).
_SESSION_MIGRATIONS = (
    ("summary", "ALTER TABLE agent_sessions ADD COLUMN summary TEXT"),
    ("consolidated_at", "ALTER TABLE agent_sessions ADD COLUMN consolidated_at TEXT"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _compact_stamp() -> str:
    """Lowercase compact UTC stamp, safe for the identifier pattern."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f").lower()


def content_hash(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Episode:
    episode_id: str
    session_id: str
    interface: str
    role: str
    turn_role: str
    content: str
    content_hash: str
    identity_hash: str
    topic_cluster_id: str | None
    tool_calls: list[str]
    created_at: str

    def to_contract(self) -> dict[str, Any]:
        """AgentEpisode contract record (contracts/agent_episode.schema.json)."""
        return {
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "interface": self.interface,
            "role": self.role,
            "turn_role": self.turn_role,
            "content": self.content,
            "content_hash": self.content_hash,
            "identity_hash": self.identity_hash,
            "topic_cluster_id": self.topic_cluster_id,
            "tool_calls": list(self.tool_calls),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class Session:
    session_id: str
    interface: str
    role: str
    status: str
    title: str | None
    started_at: str
    last_active_at: str
    identity_hash: str
    episode_count: int
    summary: str | None = None
    consolidated_at: str | None = None

    def to_contract(self) -> dict[str, Any]:
        """AgentSession contract record (contracts/agent_session.schema.json)."""
        return {
            "session_id": self.session_id,
            "interface": self.interface,
            "role": self.role,
            "status": self.status,
            "title": self.title,
            "started_at": self.started_at,
            "last_active_at": self.last_active_at,
            "identity_hash": self.identity_hash,
            "episode_count": self.episode_count,
        }


class AgentMemory:
    """SQLite-backed canonical store for agent sessions and episodes."""

    def __init__(self, store_path: str | Path | None = None) -> None:
        # Resolve IPA_AGENT_STORE at construction time, not import time: CLI,
        # dashboard, and tests may configure their store before opening a
        # connection in the same process.
        self.store_path = Path(store_path) if store_path else Path(
            os.environ.get("IPA_AGENT_STORE", str(DEFAULT_MEMORY_PATH))
        )
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.store_path))
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        for column, ddl in _SESSION_MIGRATIONS:
            columns = {row[1] for row in self._connection.execute("PRAGMA table_info(agent_sessions)")}
            if column not in columns:
                try:
                    self._connection.execute(ddl)
                except sqlite3.OperationalError:
                    pass  # ya existe (carrera benigna)
        self._connection.commit()

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def open_session(
        self,
        *,
        interface: str,
        role: str,
        identity_hash: str,
        title: str | None = None,
        session_id: str | None = None,
    ) -> str:
        session_id = session_id or f"agent_session:{_compact_stamp()}"
        now = _now()
        self._connection.execute(
            "INSERT INTO agent_sessions (session_id, interface, role, status, title, started_at, last_active_at, identity_hash, episode_count) "
            "VALUES (?, ?, ?, 'active', ?, ?, ?, ?, 0)",
            (session_id, interface, role, title, now, now, identity_hash),
        )
        self._connection.commit()
        return session_id

    def close_session(self, session_id: str) -> None:
        self._connection.execute(
            "UPDATE agent_sessions SET status = 'closed', last_active_at = ? WHERE session_id = ?",
            (_now(), session_id),
        )
        self._connection.commit()

    def reopen_session(self, session_id: str) -> None:
        """Resume a closed session (dashboard/CLI return visits)."""
        self._connection.execute(
            "UPDATE agent_sessions SET status = 'active', last_active_at = ? WHERE session_id = ?",
            (_now(), session_id),
        )
        self._connection.commit()

    def rename_session(self, session_id: str, title: str) -> None:
        title = title.strip()[:120]
        if not title:
            raise ValueError("title must be non-empty")
        self._connection.execute(
            "UPDATE agent_sessions SET title = ? WHERE session_id = ?", (title, session_id),
        )
        self._connection.commit()

    def archive_session(self, session_id: str) -> None:
        """Archive a session: out of the active list, episodes preserved."""
        self._connection.execute(
            "UPDATE agent_sessions SET status = 'archived', last_active_at = ? WHERE session_id = ?",
            (_now(), session_id),
        )
        self._connection.commit()

    def update_session_summary(self, session_id: str, summary: str, *, title: str | None = None) -> None:
        """Store the derived session summary (consolidation output)."""
        if title:
            self._connection.execute(
                "UPDATE agent_sessions SET summary = ?, consolidated_at = ?, title = ? WHERE session_id = ?",
                (summary, _now(), title.strip()[:120], session_id),
            )
        else:
            self._connection.execute(
                "UPDATE agent_sessions SET summary = ?, consolidated_at = ? WHERE session_id = ?",
                (summary, _now(), session_id),
            )
        self._connection.commit()

    def find_idle_sessions(self, *, idle_minutes: float = 5, limit: int = 5) -> list[Session]:
        """Closed (or archived), unconsolidated sessions idle >= idle_minutes."""
        cutoff = (datetime.now(timezone.utc) - __import__("datetime").timedelta(minutes=idle_minutes)).isoformat().replace("+00:00", "Z")
        rows = self._connection.execute(
            "SELECT * FROM agent_sessions "
            "WHERE status IN ('closed', 'archived') AND consolidated_at IS NULL "
            "AND episode_count >= 2 AND last_active_at < ? "
            "ORDER BY last_active_at ASC LIMIT ?",
            (cutoff, limit),
        ).fetchall()
        return [self._row_to_session(row) for row in rows]

    def close_stale_active_sessions(self, *, idle_minutes: float = 30) -> int:
        """Close 'active' sessions idle too long (dashboard restarts leave
        them open forever — they would never become consolidation-eligible)."""
        cutoff = (datetime.now(timezone.utc) - __import__("datetime").timedelta(minutes=idle_minutes)).isoformat().replace("+00:00", "Z")
        cur = self._connection.execute(
            "UPDATE agent_sessions SET status = 'closed' "
            "WHERE status = 'active' AND last_active_at < ?",
            (cutoff,),
        )
        self._connection.commit()
        return cur.rowcount

    def get_session(self, session_id: str) -> Session | None:
        row = self._connection.execute(
            "SELECT * FROM agent_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return self._row_to_session(row) if row is not None else None

    def list_sessions(self, limit: int = 20, *, include_archived: bool = False) -> list[Session]:
        if include_archived:
            rows = self._connection.execute(
                "SELECT * FROM agent_sessions ORDER BY last_active_at DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM agent_sessions WHERE status != 'archived' ORDER BY last_active_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_session(row) for row in rows]

    # ------------------------------------------------------------------
    # Episodes (append-only: insert only, never update)
    # ------------------------------------------------------------------

    def record_episode(
        self,
        session_id: str,
        *,
        turn_role: str,
        content: str,
        identity_hash: str,
        episode_id: str | None = None,
        topic_cluster_id: str | None = None,
        tool_calls: list[str] | None = None,
    ) -> Episode:
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"unknown session: {session_id}")
        if session.status != "active":
            raise ValueError(f"session {session_id} is {session.status}; episodes require an active session")
        episode_id = episode_id or f"agent_episode:{_compact_stamp()}"
        created_at = _now()
        self._connection.execute(
            "INSERT INTO agent_episodes (episode_id, session_id, interface, role, turn_role, content, content_hash, identity_hash, topic_cluster_id, tool_calls, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                episode_id, session_id, session.interface, session.role, turn_role,
                content, content_hash(content), identity_hash, topic_cluster_id,
                json.dumps(tool_calls or []), created_at,
            ),
        )
        self._connection.execute(
            "UPDATE agent_sessions SET last_active_at = ?, episode_count = episode_count + 1 WHERE session_id = ?",
            (_now(), session_id),
        )
        self._connection.commit()
        return Episode(
            episode_id=episode_id, session_id=session_id, interface=session.interface,
            role=session.role, turn_role=turn_role, content=content,
            content_hash=content_hash(content), identity_hash=identity_hash,
            topic_cluster_id=topic_cluster_id, tool_calls=list(tool_calls or []),
            created_at=created_at,
        )

    def get_episodes(self, session_id: str, limit: int = 100) -> list[Episode]:
        rows = self._connection.execute(
            "SELECT * FROM agent_episodes WHERE session_id = ? ORDER BY created_at ASC, episode_id ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def recent_episodes(self, limit: int = 10) -> list[Episode]:
        rows = self._connection.execute(
            "SELECT * FROM agent_episodes ORDER BY created_at DESC, episode_id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_episode(row) for row in reversed(rows)]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> Session:
        keys = row.keys()
        return Session(
            session_id=row["session_id"], interface=row["interface"], role=row["role"],
            status=row["status"], title=row["title"], started_at=row["started_at"],
            last_active_at=row["last_active_at"], identity_hash=row["identity_hash"],
            episode_count=row["episode_count"],
            summary=row["summary"] if "summary" in keys else None,
            consolidated_at=row["consolidated_at"] if "consolidated_at" in keys else None,
        )

    @staticmethod
    def _row_to_episode(row: sqlite3.Row) -> Episode:
        return Episode(
            episode_id=row["episode_id"], session_id=row["session_id"],
            interface=row["interface"], role=row["role"], turn_role=row["turn_role"],
            content=row["content"], content_hash=row["content_hash"],
            identity_hash=row["identity_hash"], topic_cluster_id=row["topic_cluster_id"],
            tool_calls=json.loads(row["tool_calls"]), created_at=row["created_at"],
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "AgentMemory":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


__all__ = ["AgentMemory", "Episode", "Session", "content_hash"]
