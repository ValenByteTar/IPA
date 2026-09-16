"""Local-first web search adapters for agentic research.

Search order:
  1. local SearXNG (``IPA_SEARXNG_URL``; no cloud key)
  2. cached results from the local SQLite cache
  3. DuckDuckGo HTML as a best-effort fallback

DDG is not treated as an authority or an availability dependency. A blocked
backend is recorded as an error; the caller never sees an empty successful
result that could be confused with "no results".
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlparse, urlsplit

import requests


@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str
    snippet: str
    domain: str


@dataclass
class SearchSummary:
    query: str
    results: list[SearchResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str | None = None
    backend: str | None = None
    from_cache: bool = False

    @property
    def success(self) -> bool:
        return self.error is None and len(self.results) > 0


_DDG_HTML_URL = "https://html.duckduckgo.com/html/"
_USER_AGENT = "IPA-local-research/1.0"
_RESULT_LINK_RE = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.I | re.S)
_RESULT_SNIPPET_RE = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean_html(text: str) -> str:
    text = _TAG_RE.sub("", text)
    return (text.replace("&amp;", "&").replace("&lt;", "<")
            .replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")
            .strip())


def _extract_ddg_redirect(href: str) -> str:
    if "uddg=" in href:
        values = parse_qs(urlsplit(href).query).get("uddg")
        if values:
            return values[0]
    return href


def _cache_path() -> Path:
    return Path(os.environ.get("IPA_WEB_SEARCH_CACHE", "outputs/agent/web_search_cache.db"))


def _cache_get(query: str, max_results: int, ttl_seconds: int) -> SearchSummary | None:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        with sqlite3.connect(path) as conn:
            row = conn.execute(
                "SELECT created_at, payload FROM web_search_cache WHERE query_hash=?",
                (hashlib.sha256(query.encode()).hexdigest(),),
            ).fetchone()
        if row and time.time() - row[0] <= ttl_seconds:
            results = [SearchResult(**item) for item in json.loads(row[1])[:max_results]]
            return SearchSummary(query, results, backend="cache", from_cache=True)
    except (sqlite3.Error, OSError, ValueError, TypeError):
        return None
    return None


def _cache_put(query: str, results: list[SearchResult]) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS web_search_cache (query_hash TEXT PRIMARY KEY, created_at REAL NOT NULL, payload TEXT NOT NULL)")
            conn.execute(
                "INSERT OR REPLACE INTO web_search_cache VALUES (?, ?, ?)",
                (hashlib.sha256(query.encode()).hexdigest(), time.time(), json.dumps([r.__dict__ for r in results])),
            )
            conn.commit()
    except (sqlite3.Error, OSError):
        pass


def _search_searxng(query: str, max_results: int, timeout: int) -> SearchSummary:
    base = os.environ.get("IPA_SEARXNG_URL", "").rstrip("/")
    if not base:
        return SearchSummary(query, error="local SearXNG is not configured", backend="searxng")
    t0 = time.monotonic()
    try:
        response = requests.get(
            f"{base}/search", params={"q": query, "format": "json", "categories": "general"},
            headers={"User-Agent": _USER_AGENT}, timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results = []
        for item in payload.get("results", [])[:max_results]:
            url = item.get("url", "")
            if url.startswith("http"):
                results.append(SearchResult(url, item.get("title", ""), item.get("content", ""), urlparse(url).netloc))
        if not results:
            return SearchSummary(query, error="local SearXNG returned no results", backend="searxng", elapsed_seconds=time.monotonic() - t0)
        return SearchSummary(query, results, time.monotonic() - t0, backend="searxng")
    except Exception as exc:
        return SearchSummary(query, error=f"local SearXNG failed: {exc}", backend="searxng", elapsed_seconds=time.monotonic() - t0)


def _search_ddg(query: str, max_results: int, timeout: int) -> SearchSummary:
    t0 = time.monotonic()
    try:
        response = requests.post(_DDG_HTML_URL, data={"q": query, "b": ""}, headers={"User-Agent": _USER_AGENT}, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:
        return SearchSummary(query, error=f"DuckDuckGo request failed: {exc}", backend="duckduckgo", elapsed_seconds=time.monotonic() - t0)
    html = response.text
    if response.status_code != 200 or "anomaly" in html.lower():
        return SearchSummary(query, error=f"DuckDuckGo blocked request (status {response.status_code})", backend="duckduckgo", elapsed_seconds=time.monotonic() - t0)
    links = list(_RESULT_LINK_RE.finditer(html))
    snippets = list(_RESULT_SNIPPET_RE.finditer(html))
    results = []
    for i, match in enumerate(links[:max_results]):
        url = _extract_ddg_redirect(match.group(1))
        if url.startswith("http") and urlparse(url).netloc:
            results.append(SearchResult(url, _clean_html(match.group(2)), _clean_html(snippets[i].group(1)) if i < len(snippets) else "", urlparse(url).netloc))
    if not results:
        return SearchSummary(query, error="DuckDuckGo returned no results", backend="duckduckgo", elapsed_seconds=time.monotonic() - t0)
    return SearchSummary(query, results, time.monotonic() - t0, backend="duckduckgo")


def search_web(query: str, *, max_results: int = 10, timeout: int = 15, cache_ttl_seconds: int = 86400) -> SearchSummary:
    """Search local-first with cache and controlled external fallback.

    Configure a local SearXNG instance with ``IPA_SEARXNG_URL``. If it is not
    available, cached results may still satisfy the request. DDG is only a
    fallback and is never retried aggressively after a block.
    """
    query = query.strip()
    max_results = max(1, min(max_results, 50))
    cached = _cache_get(query, max_results, cache_ttl_seconds)
    if cached:
        return cached

    searx = _search_searxng(query, max_results, timeout)
    if searx.success:
        _cache_put(query, searx.results)
        return searx

    ddg = _search_ddg(query, max_results, timeout)
    if ddg.success:
        _cache_put(query, ddg.results)
        return ddg

    return SearchSummary(
        query=query,
        error=f"no search backend available: SearXNG={searx.error}; DDG={ddg.error}",
        backend="chain",
        elapsed_seconds=searx.elapsed_seconds + ddg.elapsed_seconds,
    )


__all__ = ["SearchResult", "SearchSummary", "search_web"]
