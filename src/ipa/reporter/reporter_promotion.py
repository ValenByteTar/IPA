"""Human-reviewed promotion queue for Reporter documents."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from ipa.reporter.reporter_store import ReporterStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def queue_promotion(store: ReporterStore, document_id: str, decision_id: str, target_corpus: str = "main") -> str:
    promotion_id = "promotion:" + hashlib.sha256((document_id + decision_id).encode()).hexdigest()[:32]
    now = _now()
    store.create_promotion(promotion_id, document_id, decision_id, target_corpus, now)
    store.commit()
    return promotion_id


def approve_promotion(store: ReporterStore, promotion_id: str, decided_by: str, note: str = "") -> None:
    if not decided_by.strip():
        raise ValueError("decided_by is required")
    store.approve_promotion(promotion_id, {
        "decision": "approved", "decided_at": _now(), "decided_by": decided_by, "note": note or None,
    }, _now())
    store.commit()


def pending_promotions(store: ReporterStore) -> list[dict[str, Any]]:
    return store.list_promotions("pending")

