"""Normalize scraper metadata without changing canonical source text."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class ReporterDocumentMetadata:
    document_id: str
    artifact_id: str
    original_path: str
    source_url: str | None
    canonical_url: str | None
    source_domain: str | None
    title: str
    published_at: str | None
    published_at_confidence: str
    scraped_at: str | None
    content_hash: str
    quality_score: float | None
    mime_type: str | None
    metadata: dict[str, str]

    def to_dict(self) -> dict:
        return asdict(self)


def _hash(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _parse_date(value: str | None) -> tuple[str | None, str]:
    if not value:
        return None, "unknown"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"), "structured"
    except ValueError:
        return None, "unknown"


def _headers(text: str) -> tuple[str, str | None, str | None]:
    lines = text.splitlines()
    title = ""
    source = None
    date = None
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
    for line in lines[:20]:
        if line.lower().startswith("source:"):
            source = line.split(":", 1)[1].strip()
        elif line.lower().startswith("date:"):
            date = line.split(":", 1)[1].strip()
    return title, source, date


def normalize_article(path: str | Path, scrape_record: dict | None = None) -> ReporterDocumentMetadata:
    path = Path(path)
    is_binary = path.suffix.lower() == ".pdf"
    text = "" if is_binary else path.read_text(encoding="utf-8", errors="replace")
    record = scrape_record or {}
    title, header_url, header_date = _headers(text)
    source_url = record.get("url") or header_url
    canonical_url = record.get("canonical_url") or source_url
    parsed = urlparse(canonical_url or source_url or "")
    published_at, confidence = _parse_date(record.get("date") or header_date)
    if record.get("date") and published_at:
        confidence = "structured"
    content_hash = record.get("content_hash")
    if not content_hash:
        payload = path.read_bytes() if is_binary else text
        content_hash = _hash(payload)
    if not str(content_hash).startswith("sha256:"):
        content_hash = "sha256:" + str(content_hash)
    artifact_id = content_hash
    document_id = "doc:" + hashlib.sha256((canonical_url or str(path)).encode()).hexdigest()[:32]
    return ReporterDocumentMetadata(
        document_id=document_id,
        artifact_id=artifact_id,
        original_path=str(path),
        source_url=source_url,
        canonical_url=canonical_url,
        source_domain=parsed.netloc.lower() or str(record.get("source_domain") or path.parent.name or "unknown"),
        title=record.get("title") or title or path.stem,
        published_at=published_at,
        published_at_confidence=confidence,
        scraped_at=record.get("scraped_at"),
        content_hash=content_hash,
        quality_score=record.get("quality_score"),
        mime_type=record.get("mime_type"),
        metadata={str(k): str(v) for k, v in (record.get("metadata") or {}).items()},
    )


def load_scrape_records(path: str | Path) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data.get("results", data if isinstance(data, list) else [])
    return {str(item.get("saved_to")): item for item in records if item.get("saved_to")}

