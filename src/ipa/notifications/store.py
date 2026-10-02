"""Cola persistente de notificaciones + slot único de dispositivo (DEC-011).

Estado: ``outputs/notifications/push.db`` — derivado y reconstruible (borrarlo
solo pierde el historial de avisos). Modelo de acceso aprobado por el usuario:
exactamente UN dispositivo emparejado (fila ``devices.id='slot'``); tomar la
plaza exige ``notifications.py pair`` ejecutado en el PC.

Fanout perezoso: las entregas se materializan en ``next_batch`` para toda
notificación con ``id > last_ack_id`` sin fila de entrega — así un aviso
encolado antes del emparejamiento se entrega igual al emparejar.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE = ROOT / "outputs" / "notifications"


def default_db() -> Path:
    """Ruta de la cola. IPA_NOTIFICATIONS_DIR redirige base+TLS (tests)."""
    override = os.environ.get("IPA_NOTIFICATIONS_DIR")
    base = Path(override) if override else DEFAULT_BASE
    return base / "push.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
  id TEXT PRIMARY KEY CHECK (id = 'slot'),
  cert_fingerprint TEXT NOT NULL,
  name TEXT,
  paired_at TEXT,
  last_seen_at TEXT,
  last_ack_id INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS notifications (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_type TEXT NOT NULL,
  title TEXT NOT NULL,
  body TEXT NOT NULL DEFAULT '',
  payload_json TEXT NOT NULL DEFAULT '{}',
  collapse_key TEXT,
  priority INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS deliveries (
  notification_id INTEGER NOT NULL,
  device_id TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('pending','sent','acked','superseded','failed')),
  attempts INTEGER NOT NULL DEFAULT 0,
  sent_at TEXT,
  acked_at TEXT,
  last_error TEXT,
  PRIMARY KEY (notification_id, device_id)
);
CREATE INDEX IF NOT EXISTS idx_deliveries_state
  ON deliveries (device_id, state, notification_id);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Store:
    """Cola de notificaciones + registro del slot. Conexión por operación:
    volumen trivial y hilos del transport + watcher sin estado compartido."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else default_db()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── Slot único ────────────────────────────────────────────────────
    def slot(self) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM devices WHERE id='slot'").fetchone()
        return dict(row) if row else None

    def bind_slot(self, cert_fingerprint: str, name: str = "") -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO devices (id, cert_fingerprint, name, paired_at,"
                " last_seen_at, last_ack_id) VALUES ('slot', ?, ?, ?, ?, 0)"
                " ON CONFLICT(id) DO UPDATE SET cert_fingerprint=excluded"
                ".cert_fingerprint, name=excluded.name, paired_at=excluded"
                ".paired_at, last_ack_id=0",
                (cert_fingerprint, name, _utcnow(), _utcnow()),
            )

    def revoke_slot(self) -> str | None:
        """Libera la plaza; devuelve el fingerprint revocado (o None)."""
        with self._conn() as c:
            row = c.execute(
                "SELECT cert_fingerprint FROM devices WHERE id='slot'"
            ).fetchone()
            c.execute("DELETE FROM devices WHERE id='slot'")
        return row["cert_fingerprint"] if row else None

    def touch(self, last_ack_id: int | None = None) -> None:
        with self._conn() as c:
            if last_ack_id is None:
                c.execute(
                    "UPDATE devices SET last_seen_at=? WHERE id='slot'",
                    (_utcnow(),),
                )
            else:
                c.execute(
                    "UPDATE devices SET last_seen_at=?, last_ack_id=MAX"
                    "(last_ack_id, ?) WHERE id='slot'",
                    (_utcnow(), int(last_ack_id)),
                )

    # ── Cola ─────────────────────────────────────────────────────────
    def _fanout(self, conn: sqlite3.Connection) -> None:
        """Materializa entregas faltantes del slot para notificaciones con
        id > last_ack_id. COALESCE FUERA de la subconsulta: sin fila slot,
        el escalar es NULL y `n.id > NULL` no insertaría nada."""
        conn.execute(
            "INSERT OR IGNORE INTO deliveries (notification_id, device_id,"
            " state) SELECT n.id, 'slot', 'pending' FROM notifications n"
            " WHERE n.id > COALESCE((SELECT last_ack_id FROM devices WHERE"
            " id='slot'), 0) AND NOT EXISTS (SELECT 1 FROM deliveries d"
            " WHERE d.notification_id=n.id AND d.device_id='slot')"
        )

    def enqueue(
        self,
        event_type: str,
        title: str,
        body: str = "",
        payload: dict[str, Any] | None = None,
        *,
        priority: int = 1,
        collapse_key: str | None = None,
    ) -> int:
        """Inserta la notificación y colapsa pendientes del mismo
        (event_type, collapse_key) — anti-spam (solo viaja la última)."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO notifications (event_type, title, body,"
                " payload_json, collapse_key, priority, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    event_type,
                    title,
                    body,
                    json.dumps(payload or {}, ensure_ascii=False),
                    collapse_key,
                    int(priority),
                    _utcnow(),
                ),
            )
            nid = int(cur.lastrowid)
            self._fanout(c)
            if collapse_key:
                c.execute(
                    "UPDATE deliveries SET state='superseded'"
                    " WHERE state IN ('pending','sent') AND device_id='slot'"
                    " AND notification_id IN (SELECT id FROM notifications"
                    " WHERE event_type=? AND collapse_key=? AND id<?)",
                    (event_type, collapse_key, nid),
                )
        return nid

    def next_batch(self, since_id: int, limit: int = 50) -> list[dict[str, Any]]:
        """Eventos con id > since_id para el slot, materializando el fanout
        perezoso (avisos encolados antes del emparejamiento incluidos)."""
        with self._conn() as c:
            self._fanout(c)
            rows = c.execute(
                "SELECT n.id, n.event_type, n.title, n.body, n.payload_json,"
                " n.priority, n.created_at FROM notifications n JOIN"
                " deliveries d ON d.notification_id=n.id AND d.device_id="
                "'slot' WHERE n.id > ? AND d.state IN ('pending','sent')"
                " ORDER BY n.id LIMIT ?",
                (int(since_id), int(limit)),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "event_type": r["event_type"],
                "title": r["title"],
                "body": r["body"],
                "payload": json.loads(r["payload_json"] or "{}"),
                "priority": r["priority"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def mark_sent(self, ids: list[int]) -> None:
        if not ids:
            return
        marks = ",".join("?" * len(ids))
        with self._conn() as c:
            c.execute(
                f"UPDATE deliveries SET state='sent', attempts=attempts+1,"
                f" sent_at=COALESCE(sent_at, ?) WHERE device_id='slot' AND"
                f" notification_id IN ({marks})",
                (_utcnow(), *ids),
            )

    def ack(self, ids: list[int]) -> int:
        if not ids:
            return 0
        marks = ",".join("?" * len(ids))
        with self._conn() as c:
            cur = c.execute(
                f"UPDATE deliveries SET state='acked', acked_at=? WHERE"
                f" device_id='slot' AND notification_id IN ({marks})"
                f" AND state IN ('pending','sent')",
                (_utcnow(), *ids),
            )
            row = c.execute(
                "SELECT COALESCE(MAX(notification_id), 0) AS top FROM"
                " deliveries WHERE device_id='slot' AND state='acked'"
            ).fetchone()
            c.execute(
                "UPDATE devices SET last_ack_id=MAX(last_ack_id, ?) WHERE"
                " id='slot'",
                (int(row["top"]),),
            )
        return cur.rowcount

    def mark_failed(self, ids: list[int], error: str, max_attempts: int = 5) -> None:
        if not ids:
            return
        marks = ",".join("?" * len(ids))
        with self._conn() as c:
            self._fanout(c)
            c.execute(
                f"UPDATE deliveries SET attempts=attempts+1, last_error=?"
                f" WHERE device_id='slot' AND notification_id IN ({marks})",
                (error, *ids),
            )
            c.execute(
                f"UPDATE deliveries SET state='failed' WHERE device_id="
                f"'slot' AND attempts >= ? AND notification_id IN ({marks})",
                (max_attempts, *ids),
            )

    def prune(self, retention_days: int = 7) -> int:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=int(retention_days))
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._conn() as c:
            cur = c.execute(
                "DELETE FROM deliveries WHERE state IN ('acked','superseded')"
                " AND COALESCE(acked_at, sent_at) < ?",
                (cutoff,),
            )
            c.execute(
                "DELETE FROM notifications WHERE created_at < ? AND NOT"
                " EXISTS (SELECT 1 FROM deliveries d WHERE d.notification_id"
                "= notifications.id AND d.state IN ('pending','sent'))",
                (cutoff,),
            )
        return cur.rowcount

    def stats(self) -> dict[str, Any]:
        with self._conn() as c:
            self._fanout(c)
            by_state = {
                r["state"]: r["n"]
                for r in c.execute(
                    "SELECT state, COUNT(*) AS n FROM deliveries"
                    " GROUP BY state"
                ).fetchall()
            }
            total = c.execute(
                "SELECT COUNT(*) AS n FROM notifications"
            ).fetchone()["n"]
        return {"slot": self.slot(), "deliveries": by_state, "total": total}
