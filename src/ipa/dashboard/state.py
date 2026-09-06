"""Dashboard state/config helpers extracted from the HTTP server.

Owns the dashboard path constants; ``server.py`` imports them from here
(state must never import server — that would be circular).
"""
from __future__ import annotations

import json
import sqlite3
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
SOURCES_DB = ROOT / "outputs" / "web_dashboard" / "sources.json"
STATE_DB = ROOT / "outputs" / "web_dashboard" / "dashboard.db"
SCRAPE_CONFIG = ROOT / "configs" / "scrape_sites.yaml"

def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def safe_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("La URL debe usar http o https y tener hostname")
    if parsed.username or parsed.password:
        raise ValueError("Las URLs con credenciales no están permitidas")
    return value.strip()


def dashboard_connection() -> sqlite3.Connection:
    STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(STATE_DB))
    connection.execute("CREATE TABLE IF NOT EXISTS report_reviews (report_id TEXT PRIMARY KEY, status TEXT NOT NULL, decided_by TEXT NOT NULL, note TEXT, decided_at TEXT NOT NULL)")
    connection.commit()
    return connection


def load_sources() -> dict[str, Any]:
    data = read_json(SOURCES_DB, {"added": [], "disabled": []})
    return {"added": data.get("added", []), "disabled": data.get("disabled", [])}


def base_sources() -> list[dict[str, Any]]:
    try:
        data = yaml.safe_load(SCRAPE_CONFIG.read_text(encoding="utf-8")) or {}
        return [{"url": str(site.get("url")), "days_back": site.get("days_back", 2), "max_articles": site.get("max_articles", 20)} for site in data.get("sites", []) if site.get("url")]
    except (OSError, yaml.YAMLError):
        return []


def save_sources(data: dict[str, Any]) -> None:
    SOURCES_DB.parent.mkdir(parents=True, exist_ok=True)
    temp = SOURCES_DB.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(SOURCES_DB)


def effective_scrape_config() -> Path:
    base = yaml.safe_load(SCRAPE_CONFIG.read_text(encoding="utf-8")) or {}
    sources = load_sources()
    disabled = set(sources["disabled"])
    # Build a map of added overrides (days_back, max_articles) by URL
    added_map = {item.get("url"): item for item in sources["added"] if item.get("url") not in disabled}
    # Start with base sites, applying overrides where they exist
    sites = []
    for site in base.get("sites", []):
        url = site.get("url")
        if url in disabled:
            continue
        if url in added_map:
            # Use override values
            override = added_map.pop(url)
            merged = dict(site)
            merged["days_back"] = override.get("days_back", site.get("days_back", 2))
            if "max_articles" in override:
                merged["max_articles"] = override["max_articles"]
            sites.append(merged)
        else:
            sites.append(site)
    # Add remaining added sources (not in base, not disabled)
    sites.extend(added_map.values())
    path = ROOT / "outputs" / "web_dashboard" / "effective_scrape_sites.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"sites": sites}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


