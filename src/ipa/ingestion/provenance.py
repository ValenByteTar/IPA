"""Provenance tracking — records where each document came from.

Populates DocumentStore.document_sources from:
  - scrape_report.json (configured scrape sites → provenance="configured_scrape")
  - WebSource records (agent research → provenance="agent_research")

This is a derived index: it can be rebuilt at any time from the source data.
The DocumentStore remains the canonical authority for documents and chunks.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ipa.storage.document_store import DocumentStore


def _domain_from_url(url: str) -> str:
    """Extract the domain from a URL, or return empty string."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


def _build_path_to_doc_id_map(store: DocumentStore, landing_db_path: Path) -> dict[str, str]:
    """Build a mapping from file path → document_id using the LandingZone.

    The LandingZone stores source_uri (absolute path) and artifact_id.
    The DocumentStore stores document_id and artifact_id.
    We join them to get path → document_id.
    """
    if not landing_db_path.exists():
        return {}

    landing_conn = sqlite3.connect(str(landing_db_path))
    try:
        landing_rows = landing_conn.execute(
            "SELECT source_uri, artifact_id FROM artifacts"
        ).fetchall()
    finally:
        landing_conn.close()

    # Build artifact_id → document_id map from DocumentStore
    doc_rows = store._conn.execute(
        "SELECT document_id, artifact_id FROM documents"
    ).fetchall()
    artifact_to_doc = {row[1]: row[0] for row in doc_rows}

    # Build path → document_id map
    path_to_doc: dict[str, str] = {}
    for source_uri, artifact_id in landing_rows:
        doc_id = artifact_to_doc.get(artifact_id)
        if doc_id and source_uri:
            # Normalize to resolved path for matching
            path_to_doc[str(Path(source_uri).resolve())] = doc_id
            # Also store the raw source_uri in case paths don't resolve
            path_to_doc[source_uri] = doc_id
    return path_to_doc


def backfill_from_scrape_report(
    store: DocumentStore,
    scrape_report_path: str | Path,
    landing_db_path: str | Path,
    configured_urls: set[str] | None = None,
) -> int:
    """Backfill document_sources from a scrape_report.json file.

    Matches by file path: scrape_report.saved_to → LandingZone.source_uri →
    DocumentStore.document_id.

    Args:
        store: DocumentStore to write provenance into.
        scrape_report_path: Path to scrape_report.json.
        landing_db_path: Path to the landing.db in the same corpus.
        configured_urls: Set of base URLs from configs/scrape_sites.yaml.
            Documents whose source URL matches a configured site get
            provenance="configured_scrape"; others get "agent_research".

    Returns:
        Number of documents with provenance recorded.
    """
    configured_urls = configured_urls or set()
    report_path = Path(scrape_report_path)
    if not report_path.exists():
        return 0

    # Build path → document_id map
    path_to_doc = _build_path_to_doc_id_map(store, Path(landing_db_path))
    if not path_to_doc:
        return 0

    data = json.loads(report_path.read_text(encoding="utf-8"))
    results = data.get("results", data if isinstance(data, list) else [])
    count = 0

    for record in results:
        saved_to = record.get("saved_to")
        if not saved_to:
            continue

        # Try to match by resolved path
        try:
            resolved = str(Path(saved_to).resolve())
        except Exception:
            resolved = saved_to

        document_id = path_to_doc.get(resolved) or path_to_doc.get(saved_to)
        if not document_id:
            continue

        source_url = record.get("url") or record.get("canonical_url") or ""
        source_domain = record.get("source_domain") or _domain_from_url(source_url)
        quality_score = float(record.get("quality_score") or 0.0)

        # Determine provenance
        provenance = "agent_research"
        if configured_urls:
            for configured in configured_urls:
                if source_url and configured in source_url:
                    provenance = "configured_scrape"
                    break
        else:
            # Without configured_urls, assume scrape_report = configured sources
            provenance = "configured_scrape"

        store.put_source(document_id, source_url, source_domain, provenance, quality_score)
        count += 1

    store.commit()
    return count


def _source_url_from_text(text: str) -> str:
    """Extract the ``Source: <url>`` line trafilatura embeds in scraped text."""
    for line in (text or "")[:2000].splitlines():
        line = line.strip()
        if line.lower().startswith("source:"):
            url = line[7:].strip()
            if url.startswith("http"):
                return url
    return ""


def backfill_from_landing_registry(
    store: DocumentStore,
    landing_db_path: str | Path,
    landing_root: str | Path,
) -> int:
    """Mark scraper-ingested docs as configured_scrape when provenance is missing.

    Fallback for runs where ``scrape_report.json`` is absent or incomplete.
    Artifacts whose ``source_uri`` sits under ``<landing_root>/web/`` came from
    the configured scraper — agent-research downloads record their own
    ``agent_research`` provenance at fetch time, so a missing row implies the
    scraper. The ``Source:`` line embedded in the document text supplies the
    real source URL/domain when available.

    Returns:
        Number of documents with provenance recorded.
    """
    web_root = (Path(landing_root) / "web").resolve()
    ldb = Path(landing_db_path)
    if not ldb.exists():
        return 0

    landing_conn = sqlite3.connect(str(ldb))
    try:
        rows = landing_conn.execute(
            "SELECT artifact_id, source_uri FROM artifacts"
        ).fetchall()
    finally:
        landing_conn.close()

    artifact_to_doc = {
        r[1]: r[0] for r in store._conn.execute(
            "SELECT document_id, artifact_id FROM documents WHERE tombstoned = 0").fetchall()
    }
    try:
        recorded = {
            r[0] for r in store._conn.execute(
                "SELECT document_id FROM document_sources").fetchall()
        }
    except sqlite3.Error:
        recorded = set()

    count = 0
    for artifact_id, source_uri in rows:
        doc_id = artifact_to_doc.get(artifact_id)
        if not doc_id or doc_id in recorded or not source_uri:
            continue
        try:
            Path(source_uri).resolve().relative_to(web_root)
        except (ValueError, OSError):
            continue  # not a scraper artifact
        text_row = store._conn.execute(
            "SELECT text FROM documents WHERE document_id=?", (doc_id,)).fetchone()
        source_url = _source_url_from_text(text_row[0] if text_row else "")
        store.put_source(
            doc_id, source_url, _domain_from_url(source_url),
            "configured_scrape", 0.0)
        count += 1
    store.commit()
    return count


def record_agent_research(
    store: DocumentStore,
    document_id: str,
    source_url: str,
    source_domain: str = "",
    quality_score: float = 0.0,
    provenance: str = "agent_research",
) -> None:
    """Record that a document was acquired by the agent's research executor.

    Args:
        store: DocumentStore to write provenance into.
        document_id: The document ID in the DocumentStore.
        source_url: The URL the document was fetched from.
        source_domain: The domain of the source URL.
        quality_score: Quality score from the fetch (if available).
        provenance: "agent_research" (search-discovered, promotion needs
            score >= 0.70) or "user_provided" (URL pasted by the user —
            auto-promote like configured_scrape; DEC-003).
    """
    if not source_domain:
        source_domain = _domain_from_url(source_url)
    store.put_source(document_id, source_url, source_domain, provenance, quality_score)
    store.commit()


def load_configured_urls(yaml_path: str | Path) -> set[str]:
    """Load the set of base URLs from configs/scrape_sites.yaml.

    Returns a set of URL prefixes that can be matched against source URLs.
    """
    import yaml

    path = Path(yaml_path)
    if not path.exists():
        return set()

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    sites = data.get("sites", data if isinstance(data, list) else [])
    urls = set()
    for site in sites:
        url = site.get("url", "")
        if url:
            urls.add(url)
    return urls
