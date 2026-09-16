"""Retrievable agentic memory — the "personal/agentic corpus".

The chat agent retrieves knowledge from the document corpus via hybrid
search. This module gives it the same *kind* of retrieval over its own
memory: session summaries, user profile facts, self-knowledge, tutor
state — indexed as memory items with scope + provenance.

Architecture (mirrors the Hybrid RAG contract):

- Canonical truth stays in the source stores: episodes in agent.db,
  user model in user_model.db, principles in strategic_memory.db,
  mastery in tutor.db. This store holds DERIVED memory items — each
  carries a source_ref back to its origin and can be rebuilt by
  re-running the indexer.
- The indexer is deterministic: it reads the source stores and upserts
  items keyed by stable ids (mem:<kind>:<source_id>). The LLM only
  produces the upstream artifacts (session summaries, inferred facts);
  it never writes here directly.
- Retrieval is FTS5 (lexical) + optional sqlite-vec (semantic) merged
  via reciprocal rank fusion. The vector side is a separate rebuildable
  file (memory_vectors.db) fed by the shared BGE-M3 adapter when a warm
  instance is available; without it recall degrades to FTS-only.

Scopes:
    user      — goals, interests, preferences, approved facts about the user
    agent     — self-knowledge: principles, capabilities, decisions
    episodic  — session summaries + per-unit lesson summaries
    tutor     — topic mastery records and evidence-backed learner state
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_MEMORY_STORE = Path("outputs") / "agent" / "memory_store.db"
DEFAULT_MEMORY_VECTORS = Path("outputs") / "agent" / "memory_vectors.db"

SCOPES = ("user", "agent", "episodic", "tutor")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    memory_id   TEXT PRIMARY KEY,
    scope       TEXT NOT NULL,
    kind        TEXT NOT NULL,
    text        TEXT NOT NULL,
    source_ref  TEXT,
    confidence  REAL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_scope ON memory_items(scope, updated_at DESC);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    text, content='memory_items', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS memory_items_ai AFTER INSERT ON memory_items BEGIN
    INSERT INTO memory_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS memory_items_ad AFTER DELETE ON memory_items BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS memory_items_au AFTER UPDATE ON memory_items BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO memory_fts(rowid, text) VALUES (new.rowid, new.text);
END;
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _item_id(kind: str, source_id: str) -> str:
    digest = hashlib.sha256(f"{kind}|{source_id}".encode()).hexdigest()[:16]
    return f"mem:{kind}:{digest}"


@dataclass
class MemoryItem:
    memory_id: str
    scope: str
    kind: str
    text: str
    source_ref: str | None = None
    confidence: float | None = None
    created_at: str = ""
    updated_at: str = ""


def _rrf_merge(*ranked_id_lists: list[str], k: int = 60) -> list[str]:
    """Reciprocal rank fusion over ranked memory-id lists (higher = better)."""
    scores: dict[str, float] = {}
    for ids in ranked_id_lists:
        for rank, mid in enumerate(ids):
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank + 1)
    return [mid for mid, _ in sorted(scores.items(), key=lambda x: -x[1])]


class MemoryVectorIndex:
    """Semantic side of memory recall: sqlite-vec over memory items.

    Derived and rebuildable — clear() + re-embed rebuilds it from
    memory_items. Lives in its own file so the FTS store stays
    dependency-free (sqlite_vec is an optional extension).
    """

    def __init__(self, db_path: str | Path = DEFAULT_MEMORY_VECTORS,
                 vector_dim: int = 1024) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_dim = vector_dim
        self._conn = sqlite3.connect(str(self.db_path))
        try:
            import sqlite_vec
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
        except Exception:
            self._conn.close()
            raise
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_memory USING vec0("
            f"memory_id TEXT PRIMARY KEY, "
            f"embedding float[{vector_dim}] distance_metric=cosine)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vec_memory_meta ("
            "memory_id TEXT PRIMARY KEY, scope TEXT, embedded_at TEXT)"
        )
        self._conn.commit()

    @staticmethod
    def _blob(vec: list[float]) -> bytes:
        import struct
        return struct.pack(f"{len(vec)}f", *vec)

    def upsert(self, memory_id: str, vec: list[float], scope: str = "") -> None:
        import time as _t
        self._conn.execute(
            "INSERT OR REPLACE INTO vec_memory (memory_id, embedding) VALUES (?, ?)",
            (memory_id, self._blob(vec)),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO vec_memory_meta (memory_id, scope, embedded_at) "
            "VALUES (?, ?, ?)",
            (memory_id, scope, _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime())),
        )
        self._conn.commit()

    def embedded_ids(self) -> set[str]:
        return {
            r[0] for r in self._conn.execute("SELECT memory_id FROM vec_memory_meta")
        }

    def search(self, query_vec: list[float], limit: int = 10) -> list[tuple[str, float]]:
        blob = self._blob(query_vec)
        rows = self._conn.execute(
            "SELECT v.memory_id, v.distance FROM vec_memory v "
            "WHERE v.embedding MATCH ? AND k = ? "
            "ORDER BY v.distance ASC",
            (blob, limit),
        ).fetchall()
        # sqlite-vec cosine distance: 0 = identical → negate (higher = better).
        return [(r[0], -float(r[1])) for r in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM vec_memory_meta").fetchone()[0]

    def clear(self) -> None:
        self._conn.execute("DROP TABLE IF EXISTS vec_memory")
        self._conn.execute("DROP TABLE IF EXISTS vec_memory_meta")
        self._conn.execute(
            f"CREATE VIRTUAL TABLE vec_memory USING vec0("
            f"memory_id TEXT PRIMARY KEY, "
            f"embedding float[{self.vector_dim}] distance_metric=cosine)"
        )
        self._conn.execute(
            "CREATE TABLE vec_memory_meta ("
            "memory_id TEXT PRIMARY KEY, scope TEXT, embedded_at TEXT)"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class MemoryStore:
    """Canonical store for derived memory items + FTS5 retrieval."""

    def __init__(self, store_path: str | Path = DEFAULT_MEMORY_STORE) -> None:
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.store_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def upsert_item(self, item: MemoryItem) -> None:
        if item.scope not in SCOPES:
            raise ValueError(f"invalid scope {item.scope!r}; valid: {SCOPES}")
        now = _now()
        self._conn.execute(
            "INSERT INTO memory_items(memory_id, scope, kind, text, source_ref,"
            " confidence, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(memory_id) DO UPDATE SET text=excluded.text,"
            " source_ref=excluded.source_ref, confidence=excluded.confidence,"
            " updated_at=excluded.updated_at",
            (item.memory_id, item.scope, item.kind, item.text,
             item.source_ref, item.confidence,
             item.created_at or now, now),
        )
        self._conn.commit()

    def get_item(self, memory_id: str) -> MemoryItem | None:
        row = self._conn.execute(
            "SELECT * FROM memory_items WHERE memory_id=?", (memory_id,)
        ).fetchone()
        return self._row_to_item(row) if row else None

    def get_by_ids(self, memory_ids: list[str]) -> list[MemoryItem]:
        """Ordered fetch for fused retrieval (RRF rank order preserved)."""
        if not memory_ids:
            return []
        placeholders = ",".join("?" * len(memory_ids))
        rows = self._conn.execute(
            f"SELECT * FROM memory_items WHERE memory_id IN ({placeholders})",
            tuple(memory_ids),
        ).fetchall()
        by_id = {r[0]: self._row_to_item(r) for r in rows}
        return [by_id[mid] for mid in memory_ids if mid in by_id]

    def delete_item(self, memory_id: str) -> None:
        self._conn.execute("DELETE FROM memory_items WHERE memory_id=?", (memory_id,))
        self._conn.commit()

    def count(self, scope: str | None = None) -> int:
        if scope:
            return self._conn.execute(
                "SELECT COUNT(*) FROM memory_items WHERE scope=?", (scope,)
            ).fetchone()[0]
        return self._conn.execute("SELECT COUNT(*) FROM memory_items").fetchone()[0]

    def recall(
        self,
        query: str,
        *,
        scopes: Iterable[str] | None = None,
        limit: int = 5,
    ) -> list[MemoryItem]:
        """FTS5 retrieval over memory items, scope-filtered, rank-ordered."""
        limit = max(1, min(20, int(limit)))
        scope_list = [s for s in (scopes or SCOPES) if s in SCOPES]
        if not scope_list:
            return []
        placeholders = ",".join("?" * len(scope_list))
        terms = re.findall(r"[\wáéíóúñü]+", query.lower())
        if not terms:
            # No searchable terms → most recent items in scope.
            rows = self._conn.execute(
                f"SELECT m.* FROM memory_items m WHERE m.scope IN ({placeholders})"
                " ORDER BY m.updated_at DESC LIMIT ?",
                (*scope_list, limit),
            ).fetchall()
            return [self._row_to_item(r) for r in rows]
        match = " OR ".join(f'"{t}"' for t in terms[:8])
        rows = self._conn.execute(
            f"SELECT m.*, bm25(memory_fts) AS rank FROM memory_items m"
            f" JOIN memory_fts f ON f.rowid = m.rowid"
            f" WHERE memory_fts MATCH ? AND m.scope IN ({placeholders})"
            " ORDER BY rank LIMIT ?",
            (match, *scope_list, limit),
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def _row_to_item(self, row: sqlite3.Row | tuple) -> MemoryItem:
        return MemoryItem(
            memory_id=row[0], scope=row[1], kind=row[2], text=row[3],
            source_ref=row[4], confidence=row[5],
            created_at=row[6], updated_at=row[7],
        )

    def close(self) -> None:
        self._conn.close()


class MemoryIndexer:
    """Deterministic sync from canonical stores → retrievable memory items.

    Every item carries a stable memory_id derived from its source, so sync
    is idempotent and safe to run on every recall (cheap on small stores).
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def sync(
        self,
        *,
        memory: Any | None = None,
        user_model: Any | None = None,
        strategic: Any | None = None,
        tutor: Any | None = None,
    ) -> int:
        n = 0
        if memory is not None:
            n += self._sync_sessions(memory)
        if user_model is not None:
            n += self._sync_user_model(user_model)
        if strategic is not None:
            n += self._sync_strategic(strategic)
        if tutor is not None:
            n += self._sync_tutor(tutor)
        return n

    def _sync_sessions(self, memory: Any) -> int:
        n = 0
        try:
            sessions = memory.list_sessions(limit=100, include_archived=True)
        except Exception:
            return 0
        for s in sessions:
            summary = getattr(s, "summary", None)
            if not summary or "(sesión trivial" in summary:
                continue
            title = getattr(s, "title", None) or ""
            text = f"{title}. {summary}".strip(". ").strip()
            self.store.upsert_item(MemoryItem(
                memory_id=_item_id("session_summary", s.session_id),
                scope="episodic", kind="session_summary",
                text=text[:2000],
                source_ref=s.session_id,
                confidence=None,
            ))
            n += 1
        return n

    def _sync_user_model(self, um: Any) -> int:
        n = 0
        try:
            for g in um.list_goals(status="active", limit=50):
                self.store.upsert_item(MemoryItem(
                    memory_id=_item_id("user_goal", g.goal_id),
                    scope="user", kind="goal",
                    text=f"Objetivo del usuario: {g.description}",
                    source_ref=g.goal_id, confidence=1.0,
                ))
                n += 1
            for i in um.list_interests(limit=50):
                self.store.upsert_item(MemoryItem(
                    memory_id=_item_id("user_interest", i.topic),
                    scope="user", kind="interest",
                    text=f"Interés del usuario: {i.topic}",
                    source_ref=i.topic,
                    confidence=i.score,
                ))
                n += 1
            for p in um.list_preferences():
                self.store.upsert_item(MemoryItem(
                    memory_id=_item_id("user_pref", p.key),
                    scope="user", kind="preference",
                    text=f"Preferencia del usuario — {p.key}: {p.value}",
                    source_ref=p.key, confidence=1.0,
                ))
                n += 1
            for f in um.list_facts(status="active", limit=100):
                self.store.upsert_item(MemoryItem(
                    memory_id=_item_id("user_fact", f["fact_id"]),
                    scope="user", kind="fact",
                    text=f"Hecho sobre el usuario: {f['fact']}",
                    source_ref=f["fact_id"], confidence=1.0,
                ))
                n += 1
        except Exception:
            pass
        return n

    def _sync_strategic(self, sm: Any) -> int:
        n = 0
        try:
            for p in sm.list_principles(status="active", limit=50):
                self.store.upsert_item(MemoryItem(
                    memory_id=_item_id("principle", p.principle_id),
                    scope="agent", kind="principle",
                    text=f"Principio aprendido: {p.principle}",
                    source_ref=p.principle_id, confidence=p.confidence,
                ))
                n += 1
        except Exception:
            pass
        return n

    def _sync_tutor(self, ts: Any) -> int:
        n = 0
        try:
            for rec in ts.list_topic_records():
                status = getattr(rec.mastery_status, "value", rec.mastery_status)
                score = rec.mastery_score
                score_txt = f"{score:.2f}" if isinstance(score, (int, float)) else "sin evaluar"
                self.store.upsert_item(MemoryItem(
                    memory_id=_item_id("tutor_topic", rec.topic_id),
                    scope="tutor", kind="mastery",
                    text=(f"Aprendizaje del usuario en {rec.topic_id}: "
                          f"estado {status}, mastery {score_txt}, "
                          f"{rec.attempts} intentos."),
                    source_ref=rec.topic_id,
                    confidence=score if isinstance(score, (int, float)) else None,
                ))
                n += 1
        except Exception:
            pass
        # Resúmenes por unidad: granularidad pedagógica que el resumen de
        # sesión comprime. topic → roadmap → "Lección de {topic}, unidad N".
        try:
            goal_topics: dict[str, str] = {}
            for rm in ts.list_roadmaps():
                goal_topics[rm.roadmap_id] = rm.goal_id.removeprefix("goal:").replace("-", " ")
            for us in ts.list_unit_summaries():
                topic = goal_topics.get(us["roadmap_id"], "el tema")
                self.store.upsert_item(MemoryItem(
                    memory_id=_item_id(
                        "lesson_unit", f"{us['roadmap_id']}:{us['unit_order']}"),
                    scope="episodic", kind="lesson_unit",
                    text=(f"Lección de {topic}, unidad {us['unit_order']}: "
                          f"{us['summary']}"),
                    source_ref=f"{us['roadmap_id']}:{us['unit_order']}",
                    confidence=None,
                ))
                n += 1
        except Exception:
            pass
        return n


__all__ = ["MemoryItem", "MemoryIndexer", "MemoryStore", "SCOPES", "DEFAULT_MEMORY_STORE"]
