"""Web scraper â€” fetch articles from whitelisted sites, extract text + images.

This module is designed for intelligence gathering: scrape recent articles
from a curated list of sites, extract clean text (trafilatura) and images
(BeautifulSoup), and save them as artifacts for the ingestion pipeline.

Usage:
    from ipa.acquisition.web_scraper import WebScraper, ScrapeSite

    scraper = WebScraper(output_dir="Landing/web")
    sites = [
        ScrapeSite(url="https://example.com/news", days_back=2),
    ]
    results = scraper.scrape_sites(sites)

Each scraped article is saved as:
    Landing/web/<domain>/<article_slug>.txt   â€” article text
    Landing/web/<domain>/<article_slug>_img<N>.png  â€” downloaded images

The .txt files are ready for the fast path pipeline.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse
from threading import Lock


def normalize_url(url: str, base_url: str | None = None) -> str:
    """Return a stable HTTP(S) URL for deduplication and navigation."""
    value = urljoin(base_url or "", url.strip())
    parsed = urlparse(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    path = parsed.path or "/"
    # Fragments never identify different server content; normalize host/case.
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path,
                       "", parsed.query, ""))

import requests
import trafilatura
from bs4 import BeautifulSoup

from ipa.ingestion.content_safety import (
    DownloadValidationConfig,
    DownloadValidationError,
    validate_download_stream,
    detect_file_type,
)


# Document extensions that the scraper will download to the Landing zone.
# These are processed by the ingestion pipeline (mime_router â†’ parsers).
# NOTE: .md is excluded â€” markdown files linked from blog posts are usually
# repo READMEs/docs (e.g. GitHub), not research papers or reports.
DOCUMENT_EXTENSIONS: set[str] = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".txt", ".csv", ".rst", ".rtf", ".odt", ".epub",
}

# Domains that host source code / repos, not research documents.
# Links to these domains are skipped during document download.
REPO_DOMAINS: set[str] = {
    "github.com", "raw.githubusercontent.com", "gitlab.com",
    "bitbucket.org", "codeberg.org",
}


# ---------------------------------------------------------------------------
# Scrape history â€” persistent deduplication of scraped URLs
# ---------------------------------------------------------------------------

class ScrapeHistory:
    """SQLite-backed record of URLs that have been scraped.

    Prevents re-scraping the same article across runs.  The history DB
    is stored alongside the output (default: <output_dir>/scrape_history.db).

    Usage:
        history = ScrapeHistory("Landing/scrape_history.db")
        if history.is_scraped(url):
            skip()
        else:
            result = scrape(url)
            history.record(url, title=result.title, status="ok")
    """

    def __init__(self, db_path: str | Path) -> None:
        import sqlite3
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=30)
        # Enable WAL mode for concurrent access from multiple threads
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS scraped_urls (
                url TEXT PRIMARY KEY, title TEXT, status TEXT,
                scraped_at TEXT, site_url TEXT, claimed_at TEXT
            )
        """)
        # Durable job state allows interrupted workers to be inspected/retried.
        self._conn.execute("""CREATE TABLE IF NOT EXISTS scrape_jobs (
            url TEXT PRIMARY KEY, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            claimed_at TEXT, completed_at TEXT, error TEXT
        )""")
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(scraped_urls)")}
        if "claimed_at" not in columns:
            self._conn.execute("ALTER TABLE scraped_urls ADD COLUMN claimed_at TEXT")
        self._conn.commit()

    def is_scraped(self, url: str) -> bool:
        url = normalize_url(url)
        return bool(url and self._conn.execute(
            "SELECT 1 FROM scraped_urls WHERE url = ? AND status = 'ok'", (url,)
        ).fetchone())

    def claim(self, url: str, stale_after: int = 3600) -> bool:
        """Atomically claim a URL; safe when multiple scrapers share the DB."""
        url = normalize_url(url)
        if not url:
            return False
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute("""INSERT INTO scrape_jobs(url,status,attempts,claimed_at)
            VALUES (?, 'running', 1, ?) ON CONFLICT(url) DO UPDATE SET
            status='running', attempts=attempts+1, claimed_at=excluded.claimed_at
            WHERE status != 'running' OR claimed_at < datetime('now', ?)
        """, (url, now, f'-{stale_after} seconds'))
        self._conn.commit()
        return cur.rowcount == 1

    def record(self, url: str, title: str = "", status: str = "ok",
               site_url: str = "") -> None:
        """Record a URL as scraped."""
        from datetime import datetime, timezone
        url = normalize_url(url)
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO scraped_urls (url, title, status, scraped_at, site_url, claimed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)", (url, title, status, now, site_url, now))
        self._conn.execute("UPDATE scrape_jobs SET status=?, completed_at=?, error=? WHERE url=?",
                           ("complete" if status == "ok" else "failed", now,
                            None if status == "ok" else title, url))
        self._conn.commit()

    def count(self) -> int:
        """Total number of scraped URLs."""
        cur = self._conn.execute("SELECT COUNT(*) FROM scraped_urls")
        return cur.fetchone()[0]

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScrapeSite:
    """A site to scrape â€” deterministic configuration.

    The link extraction strategy is explicit and prioritized:

    1. article_selector (CSS): if set, only links matching this selector
       are collected.  This is the most deterministic mode.
    2. url_pattern (regex): if set, only URLs whose path matches this
       pattern are kept.  Works with or without article_selector.
    3. exclude_paths (list[str]): URL substrings to always exclude
       (e.g. ["/tag/", "/category/", "/page/"]).
    4. If none of the above are set, falls back to generic heuristics
       (<article> tags, <main> area, date URL patterns, sub-path scan).

    Example â€” NVIDIA blog (no <article> tags, articles under /blog/<slug>/):
        ScrapeSite(
            url="https://developer.nvidia.com/blog",
            url_pattern=r"^/blog/[^/]+/$",
            exclude_paths=["/blog/category/", "/blog/tag/", "/blog/recent-posts/"],
            days_back=7,
        )

    Example â€” The Hacker News (articles in <article> tags):
        ScrapeSite(
            url="https://thehackernews.com/",
            article_selector="article a[href]",
            days_back=2,
        )
    """
    url: str
    days_back: int = 2
    # CSS selector for article links on the listing page.
    article_selector: str | None = None
    # Regex pattern that article URL paths must match (applied after selector).
    url_pattern: str | None = None
    # URL substrings to exclude (applied after selector + url_pattern).
    exclude_paths: list[str] = field(default_factory=list)
    # Additional domains to allow (besides the base URL's domain).
    # Useful when a listing page links to articles on a sister domain
    # (e.g. ai.meta.com links to research.meta.ai).
    allowed_domains: list[str] = field(default_factory=list)
    # RSS/Atom feed URL for article discovery.  When set, the scraper
    # parses the feed for article URLs + dates instead of (or before)
    # crawling the listing page HTML.
    rss_feed: str | None = None
    # Sitemap URL for article discovery (sitemap index or sitemap.xml).
    # When set, the scraper parses the sitemap (following child sitemaps
    # if it's an index) and filters URLs by url_pattern + exclude_paths.
    # Used when listing pages are JS-rendered and no RSS feed exists.
    sitemap_url: str | None = None
    # JSON API URL for article discovery (for SPAs that load content via API).
    # When set, the scraper fetches the JSON endpoint and extracts article URLs.
    # Requires json_api_url + json_api_id_field + json_api_url_template.
    json_api_url: str | None = None
    # Field name in the JSON response array items that contains the article ID.
    json_api_id_field: str = "id"
    # URL template for article pages. {id} is replaced with the ID value.
    # e.g. "https://qwen.ai/blog?id={id}"
    json_api_url_template: str | None = None
    # Whether to trust trafilatura's date extraction for date filtering.
    # Some sites (e.g. Anthropic) don't expose date metadata, causing
    # trafilatura to return wrong dates from footer/copyright text.
    # Set to False to skip date filtering for articles from this site.
    trust_article_dates: bool = True
    # Max articles to scrape per site.
    max_articles: int = 20
    # Request delay in seconds (politeness).
    delay_seconds: float = 1.0
    # Per-site engine override: 'requests', 'playwright', 'auto', or None
    # (use the WebScraper's global engine setting).
    engine: str | None = None
    # Pagination: if set, the scraper follows additional pages.
    # Tries: paginate_url_template with {n}, then /page/N/, then ?page=N
    # up to max_pages. Only used when no RSS/feed/sitemap is configured.
    paginate: bool = False
    max_pages: int = 50
    # Custom pagination URL template. {n} is replaced with the page number.
    # e.g. "https://www.offsec.com/blog/{n}/" â†’ /blog/2/, /blog/3/, ...
    # If not set, defaults to /page/N/ then ?page=N
    paginate_url_template: str | None = None


@dataclass
class ScrapeResult:
    """Result of scraping a single article."""
    url: str
    title: str
    text: str
    date: str | None = None  # ISO 8601 if extracted
    image_paths: list[str] = field(default_factory=list)
    document_paths: list[str] = field(default_factory=list)
    ocr_texts: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed_seconds: float = 0.0
    canonical_url: str = ""
    content_hash: str = ""
    quality_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.text)


@dataclass
class ScrapeSummary:
    """Summary of a scrape run."""
    site_url: str
    total_articles_found: int
    articles_scraped: int
    articles_skipped: int
    images_downloaded: int
    documents_downloaded: int = 0
    errors: list[str] = field(default_factory=list)
    results: list[ScrapeResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# PlaywrightBackend â€” headless browser for JS-rendered sites
# ---------------------------------------------------------------------------

class PlaywrightBackend:
    """Headless browser backend using Playwright (Chromium).

    Used for sites that render content with JavaScript (React, Vue, etc.)
    or that block non-browser requests (Meta, Facebook, etc.).

    Lazy-loaded: the browser is only launched on first use.
    """

    def __init__(
        self,
        headless: bool = True,
        timeout: int = 30,
        wait_for: str | None = None,
        wait_timeout: int = 10000,
    ) -> None:
        self.headless = headless
        self.timeout = timeout
        self.wait_for = wait_for  # CSS selector to wait for before extracting HTML
        self.wait_timeout = wait_timeout
        self._playwright = None
        self._browser = None

    def _ensure_browser(self) -> None:
        """Launch Playwright + Chromium (lazy)."""
        if self._browser is not None:
            return
        from playwright.sync_api import sync_playwright
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)

    def fetch_page(self, url: str) -> str | None:
        """Fetch fully-rendered HTML after JavaScript execution."""
        try:
            self._ensure_browser()
            page = self._browser.new_page()
            page.set_default_timeout(self.timeout * 1000)
            page.goto(url, wait_until="networkidle")
            # Optional: wait for a specific element to appear
            if self.wait_for:
                try:
                    page.wait_for_selector(self.wait_for, timeout=self.wait_timeout)
                except Exception:
                    pass  # Continue even if selector doesn't appear
            return page.content()
        except Exception:
            return None
        finally:
            try:
                page.close()
            except Exception:
                pass

    def fetch_page_with_links(self, url: str) -> tuple[str | None, list[str]]:
        """Fetch rendered HTML and extract all <a> hrefs from the DOM.

        Returns (html, hrefs).  hrefs are extracted via JavaScript evaluation
        to capture dynamically-created links that might not be in page.content().
        """
        try:
            self._ensure_browser()
            page = self._browser.new_page()
            page.set_default_timeout(self.timeout * 1000)
            page.goto(url, wait_until="networkidle")
            if self.wait_for:
                try:
                    page.wait_for_selector(self.wait_for, timeout=self.wait_timeout)
                except Exception:
                    pass
            html = page.content()
            # Extract hrefs directly from the DOM
            hrefs = page.eval_on_selector_all(
                "a[href]",
                "els => els.map(e => e.href)",
            )
            return html, hrefs
        except Exception:
            return None, []
        finally:
            try:
                page.close()
            except Exception:
                pass

    def close(self) -> None:
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None

    def __enter__(self) -> "PlaywrightBackend":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# WebScraper
# ---------------------------------------------------------------------------

class WebScraper:
    """Scrape articles from whitelisted sites.

    Supports two engines:
      - 'requests': fast, server-rendered HTML (default)
      - 'playwright': headless browser, JS-rendered sites (Meta, SPAs)
      - 'auto': try requests first, fall back to Playwright if no links found
    """

    def __init__(
        self,
        output_dir: str | Path = "Landing/web",
        timeout: int = 30,
        user_agent: str = "RES023-Research-Bot/1.0",
        download_images: bool = True,
        engine: str = "requests",
        playwright_headless: bool = True,
        playwright_wait_for: str | None = None,
        history_db: str | Path | None = None,
        safety_config: DownloadValidationConfig | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.user_agent = user_agent
        self.download_images = download_images
        self.engine = engine
        self.safety_config = safety_config or DownloadValidationConfig(
            max_size_mb=200,
            max_redirects=3,
            timeout_seconds=timeout,
        )
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        self._session.max_redirects = self.safety_config.max_redirects
        self._playwright: PlaywrightBackend | None = None
        self._domain_lock = Lock()
        self._domain_last_request: dict[str, float] = {}
        self._pw_headless = playwright_headless
        self._pw_wait_for = playwright_wait_for
        # Scrape history for deduplication
        if history_db is None:
            history_db = self.output_dir / "scrape_history.db"
        self.history = ScrapeHistory(history_db)

    def _rate_limit(self, url: str, delay: float = 0.0) -> None:
        """Apply a minimum delay independently for each hostname."""
        if delay <= 0:
            return
        domain = urlparse(url).netloc.lower()
        with self._domain_lock:
            wait = delay - (time.monotonic() - self._domain_last_request.get(domain, 0.0))
            if wait > 0:
                time.sleep(wait)
            self._domain_last_request[domain] = time.monotonic()

    def _get_playwright(self) -> PlaywrightBackend:
        """Lazy-init Playwright backend."""
        if self._playwright is None:
            self._playwright = PlaywrightBackend(
                headless=self._pw_headless,
                timeout=self.timeout,
                wait_for=self._pw_wait_for,
            )
        return self._playwright

    def fetch_page(self, url: str, use_engine: str | None = None) -> str | None:
        """Fetch HTML content of a page.

        Args:
            url: URL to fetch.
            use_engine: 'requests', 'playwright', or None (use self.engine).
        """
        engine = use_engine or self.engine
        if engine == "playwright":
            return self._get_playwright().fetch_page(url)
        else:
            try:
                resp = self._session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                return resp.text
            except Exception:
                return None

    def extract_article_links(
        self,
        html: str,
        base_url: str,
        selector: str | None = None,
        url_pattern: str | None = None,
        exclude_paths: list[str] | None = None,
        allowed_domains: list[str] | None = None,
    ) -> list[str]:
        """Extract article links from a listing page.

        Deterministic mode (when selector or url_pattern is set):
            1. If selector is set, collect links from CSS-selected elements.
               Otherwise, collect all <a> tags.
            2. If url_pattern is set, keep only URLs whose path matches.
            3. If exclude_paths is set, drop any URL containing those substrings.
            4. Filter to same domain as base_url.

        Heuristic mode (when neither selector nor url_pattern is set):
            Falls back to 4 generic heuristics. Less predictable but works
            as a starting point for unknown sites.

        Returns links in HTML document order (deduplicated).
        """
        soup = BeautifulSoup(html, "lxml")
        base_domain = urlparse(base_url).netloc.lower()
        base_path = urlparse(base_url).path.rstrip("/")

        # Compile pattern if provided
        path_re = re.compile(url_pattern) if url_pattern else None
        excludes = exclude_paths or []

        # --- Step 1: Collect candidate links ---
        candidates: list[str] = []  # (href, in-order)

        if selector:
            # Deterministic: CSS selector
            elements = soup.select(selector)
            for el in elements:
                a = el if el.name == "a" else el.find("a")
                if a and a.get("href"):
                    candidates.append(a["href"])
        elif path_re or excludes:
            # Deterministic: URL pattern filtering on all links
            for a in soup.find_all("a", href=True):
                candidates.append(a["href"])
        else:
            # Heuristic mode: collect from multiple sources
            candidates = self._heuristic_collect(soup, base_url, base_path)

        # --- Step 2: Resolve to absolute URLs ---
        absolute = [urljoin(base_url, href) for href in candidates]

        # --- Step 3: Filter by domain ---
        allowed = {base_domain}
        if allowed_domains:
            allowed.update(allowed_domains)
        same_domain = []
        for candidate in absolute:
            normalized = normalize_url(candidate)
            if normalized and urlparse(normalized).netloc in {d.lower() for d in allowed}:
                same_domain.append(normalized)

        # --- Step 4: Apply url_pattern filter ---
        if path_re:
            same_domain = [
                url for url in same_domain
                if path_re.match(urlparse(url).path)
            ]

        # --- Step 5: Apply exclude_paths filter ---
        if excludes:
            same_domain = [
                url for url in same_domain
                if not any(ex in urlparse(url).path for ex in excludes)
            ]

        # --- Step 6: Exclude the listing page itself ---
        same_domain = [
            url for url in same_domain
            if urlparse(url).path.rstrip("/") != base_path
        ]

        # --- Step 7: Deduplicate preserving order ---
        seen: set[str] = set()
        ordered: list[str] = []
        for url in same_domain:
            if url not in seen:
                seen.add(url)
                ordered.append(url)
        return ordered

    def _heuristic_collect(
        self, soup: BeautifulSoup, base_url: str, base_path: str
    ) -> list[str]:
        """Heuristic link collection â€” used when no explicit config is set.

        Tries 4 strategies in order and merges results:
        1. <article> tags
        2. <main> or content-area divs
        3. URLs with date patterns (/2026/01/15/)
        4. Fallback: all sub-path links (if < 3 found above)
        """
        candidates: list[str] = []
        seen: set[str] = set()

        def add(href: str) -> None:
            if href and href not in seen:
                seen.add(href)
                candidates.append(href)

        # Heuristic 1: <article> tags
        for article in soup.find_all("article"):
            a = article.find("a", href=True)
            if a:
                add(a["href"])

        # Heuristic 2: <main> area links
        main = soup.find("main") or soup.find("div", class_=re.compile(
            r"content|articles|news|posts|feed", re.I
        ))
        if main:
            for a in main.find_all("a", href=True):
                if self._looks_like_article(a["href"], base_url):
                    add(a["href"])

        # Heuristic 3: links with date patterns in URL
        for a in soup.find_all("a", href=True):
            if re.search(r"/(20\d{2})[/-]\d{2}[/-]\d{2}/", a["href"]):
                add(a["href"])

        # Heuristic 4 (fallback): scan all sub-path links
        if len(candidates) < 3:
            base_domain = urlparse(base_url).netloc
            reserved = {"category", "tag", "author", "page",
                        "search", "feed", "rss", "recent-posts",
                        "about", "contact", "privacy", "terms"}
            for a in soup.find_all("a", href=True):
                href = a["href"]
                full = urljoin(base_url, href)
                parsed = urlparse(full)
                if parsed.netloc != base_domain:
                    continue
                if not parsed.path.startswith(base_path + "/"):
                    continue
                path_parts = parsed.path.strip("/").split("/")
                if any(r in parsed.path.lower() for r in reserved):
                    continue
                if parsed.path.rstrip("/") == base_path:
                    continue
                if self._looks_like_article(href, base_url):
                    add(full)

        return candidates

    @staticmethod
    def _looks_like_article(href: str, base_url: str) -> bool:
        """Heuristic: does this href look like an article link?"""
        if not href or href.startswith("#") or href.startswith("javascript:"):
            return False
        # Skip social, mailto, tel
        if any(href.startswith(s) for s in ("mailto:", "tel:", "whatsapp:", "telegram:")):
            return False
        # Skip common non-article paths
        skip_patterns = [
            "/tag/", "/category/", "/author/", "/page/", "/feed",
            "/rss", "/sitemap", "/search", "/login", "/register",
            "/about", "/contact", "/privacy", "/terms",
            ".jpg", ".png", ".gif", ".css", ".js",
            "facebook.com", "twitter.com", "x.com", "linkedin.com",
            "youtube.com", "instagram.com",
        ]
        href_lower = href.lower()
        if any(p in href_lower for p in skip_patterns):
            return False
        # Must have some path depth (not just /)
        path = urlparse(urljoin(base_url, href)).path
        if len(path.strip("/").split("/")) < 1:
            return False
        return True

    def extract_article(self, url: str, days_back: int = 0) -> ScrapeResult:
        """Fetch and extract a single article.

        Uses the configured engine.  In 'auto' mode, if requests fails or
        trafilatura extracts no text, retries with Playwright.
        When days_back > 0, skips image/document downloads for articles
        older than the cutoff to avoid wasting bandwidth on filtered content.
        """
        start = time.monotonic()
        html = self.fetch_page(url)
        if html is None and self.engine in ("auto", "playwright"):
            html = self.fetch_page(url, use_engine="playwright")
        if html is None:
            return ScrapeResult(
                url=url, title="", text="", date=None,
                error="Failed to fetch page", elapsed_seconds=time.monotonic() - start,
            )

        # trafilatura for clean text extraction
        extracted = trafilatura.extract(
            html,
            with_metadata=True,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )

        # Auto fallback: if trafilatura got no text and we're in auto mode,
        # retry with Playwright-rendered HTML
        if not extracted and self.engine == "auto":
            pw_html = self.fetch_page(url, use_engine="playwright")
            if pw_html:
                extracted = trafilatura.extract(
                    pw_html,
                    with_metadata=True,
                    include_comments=False,
                    include_tables=True,
                    favor_recall=True,
                )
                if extracted:
                    html = pw_html  # Use rendered HTML for image extraction too

        if not extracted:
            return ScrapeResult(
                url=url, title="", text="", date=None,
                error="trafilatura extracted no text", elapsed_seconds=time.monotonic() - start,
            )

        quality_score, quality_reason = self._quality_gate(extracted, html)
        if quality_score < 0.25:
            return ScrapeResult(
                url=url, title="", text="", date=None,
                error=f"quality gate rejected: {quality_reason}",
                quality_score=quality_score,
                elapsed_seconds=time.monotonic() - start,
            )

        # Parse trafilatura output (it returns a dict-like with metadata)
        metadata = trafilatura.extract(
            html, output_format="json", with_metadata=True,
            include_comments=False,
        )

        title = ""
        date_str = None
        if metadata:
            try:
                meta_dict = json.loads(metadata)
                title = meta_dict.get("title", "")
                date_str = meta_dict.get("date", None)
            except (json.JSONDecodeError, TypeError):
                pass

        # Prefer structured metadata when available; it is more reliable than
        # heuristic footer/copyright dates.
        structured = self._structured_metadata(html)
        title = structured.get("title") or title
        date_str = structured.get("datePublished") or date_str

        # Fallback: extract title from <title> tag
        if not title:
            soup = BeautifulSoup(html, "lxml")
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)

        # Date filter: skip image/document downloads for old articles
        skip_downloads = False
        if days_back > 0 and date_str:
            try:
                article_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                if article_date.tzinfo is None:
                    article_date = article_date.replace(tzinfo=timezone.utc)
                cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
                if article_date < cutoff:
                    skip_downloads = True
            except (ValueError, TypeError):
                pass

        # Download images
        image_paths: list[str] = []
        if self.download_images and not skip_downloads:
            image_paths = self._download_images(html, url)

        # Download linked documents (PDF, DOCX, PPTX, etc.)
        document_paths: list[str] = []
        if not skip_downloads:
            document_paths = self._download_documents(html, url)

        canonical_url = self._canonical_url(html, url)
        quality_score = self._quality_score(extracted, title)
        content_hash = hashlib.sha256(extracted.encode("utf-8")).hexdigest()
        return ScrapeResult(
            url=url,
            title=title,
            text=extracted,
            date=date_str,
            image_paths=image_paths,
            document_paths=document_paths,
            elapsed_seconds=time.monotonic() - start,
            canonical_url=canonical_url,
            content_hash=content_hash,
            quality_score=quality_score,
            metadata={"word_count": str(len(extracted.split()))},
        )

    @staticmethod
    def _structured_metadata(html: str) -> dict[str, str]:
        """Extract JSON-LD and OpenGraph metadata with safe fallbacks."""
        soup = BeautifulSoup(html, "lxml")
        result: dict[str, str] = {}
        for node in soup.select('script[type="application/ld+json"]'):
            try:
                payload = json.loads(node.string or node.get_text())
                values = payload if isinstance(payload, list) else [payload]
                for item in values:
                    if not isinstance(item, dict):
                        continue
                    if "@graph" in item and isinstance(item["@graph"], list):
                        values.extend(x for x in item["@graph"] if isinstance(x, dict))
                    types = item.get("@type", [])
                    if isinstance(types, str):
                        types = [types]
                    if any(t in {"Article", "NewsArticle", "TechArticle", "Report"} for t in types):
                        for key in ("headline", "name", "datePublished", "url"):
                            if item.get(key) and key not in result:
                                result[key if key != "headline" else "title"] = str(item[key])
                        return result
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue
        for prop, key in (("og:title", "title"), ("article:published_time", "datePublished"), ("og:url", "url")):
            tag = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"name": prop})
            if tag and tag.get("content"):
                result.setdefault(key, tag["content"].strip())
        return result

    @staticmethod
    def _canonical_url(html: str, fallback: str):
        soup = BeautifulSoup(html, "lxml")
        link = soup.find("link", rel=lambda value: value and "canonical" in value)
        return normalize_url(link.get("href")) if link and link.get("href") else normalize_url(fallback)

    @staticmethod
    def _quality_gate(text: str, html: str) -> tuple[float, str]:
        """Reject login/cookie walls and low-value boilerplate pages."""
        lowered = text.lower()
        markers = ("sign in", "log in", "accept cookies", "cookie policy", "enable javascript")
        marker_hits = sum(lowered.count(marker) for marker in markers)
        words = text.split()
        if len(words) < 40:
            return 0.10, "too little article text"
        if marker_hits >= 3 and len(words) < 250:
            return 0.10, "cookie/login/javascript wall"
        boilerplate = sum(lowered.count(x) for x in ("subscribe", "privacy policy", "all rights reserved"))
        score = 0.5 + min(len(words) / 2000, 0.35) - min(boilerplate / 10, 0.25)
        return max(0.0, min(score, 1.0)), "accepted"

    @staticmethod
    def _quality_score(text: str, title: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        score = 0.35
        if title:
            score += 0.15
        if len(words) >= 100:
            score += 0.25
        if len(words) >= 500:
            score += 0.15
        if len(set(w.lower() for w in words)) / len(words) > 0.35:
            score += 0.10
        return min(score, 1.0)

    def _download_images(self, html: str, base_url: str) -> list[str]:
        """Download content images from article HTML. Returns local file paths.

        Handles lazy-loaded images (data-src), filters out avatars, logos,
        icons, and tracking pixels.
        """
        soup = BeautifulSoup(html, "lxml")
        img_tags = soup.find_all("img")
        domain = urlparse(base_url).netloc
        site_dir = self.output_dir / self._slugify(domain)
        site_dir.mkdir(parents=True, exist_ok=True)

        # Classes that indicate non-content images (lazyload is NOT here â€”
        # it just means the image is lazy-loaded, handled via data-src)
        skip_classes = {"avatar", "logo", "icon", "emoji"}

        downloaded: list[str] = []
        for i, img in enumerate(img_tags):
            # Resolve src: handle lazy loading (data-src)
            src = img.get("src", "")
            if src.startswith("data:"):
                src = img.get("data-src", "")
            if not src:
                continue

            # Skip data URIs, tracking pixels, tiny icons
            if src.startswith("data:"):
                continue
            width = img.get("width", "")
            if width and width.isdigit() and int(width) < 100:
                continue

            # Skip avatars, logos, icons by class
            classes = img.get("class", [])
            if any(cls in skip_classes for cls in classes):
                continue

            # Skip social/platform images
            if any(s in src.lower() for s in ("gravatar.com", "avatar", "logo")):
                continue

            img_url = urljoin(base_url, src)
            try:
                # Send Referer header â€” many CDNs require it to serve images
                resp = self._session.get(
                    img_url, timeout=self.timeout, stream=True,
                    headers={"Referer": base_url},
                )
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                if "image" not in content_type:
                    continue
                # Determine extension
                ext = ".png"
                if "jpeg" in content_type or "jpg" in content_type:
                    ext = ".jpg"
                elif "webp" in content_type:
                    ext = ".webp"
                elif "gif" in content_type:
                    ext = ".gif"

                # Layer 1: validate download (magic bytes, size)
                from ipa.ingestion.content_safety import _EXT_TO_TYPE
                expected_type = _EXT_TO_TYPE.get(ext)
                content = validate_download_stream(
                    resp, expected_type=expected_type, config=self.safety_config,
                )

                # Hash the URL for a stable filename
                url_hash = hashlib.sha256(img_url.encode()).hexdigest()[:12]
                filename = f"img_{url_hash}{ext}"
                filepath = site_dir / filename
                if not filepath.exists():
                    filepath.write_bytes(content)
                downloaded.append(str(filepath))
            except DownloadValidationError:
                continue  # Silently skip invalid images
            except requests.RequestException:
                continue
        return downloaded

    def _download_documents(self, html: str, base_url: str) -> list[str]:
        """Download linked documents (PDF, DOCX, PPTX, etc.) + arxiv papers.

        Scans all <a> tags for:
          1. hrefs pointing to document files (.pdf, .docx, etc.)
          2. hrefs pointing to arxiv.org/abs/<id> or arxiv.org/pdf/<id>
             â€” these are followed and the PDF is downloaded

        Returns list of local file paths.
        """
        soup = BeautifulSoup(html, "lxml")
        domain = urlparse(base_url).netloc
        site_dir = self.output_dir / self._slugify(domain)
        site_dir.mkdir(parents=True, exist_ok=True)

        downloaded: list[str] = []
        seen_urls: set[str] = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Resolve to absolute URL
            doc_url = urljoin(base_url, href)
            parsed = urlparse(doc_url)

            # --- Check for arxiv links ---
            arxiv_id = self._extract_arxiv_id(doc_url)
            if arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                if pdf_url in seen_urls:
                    continue
                seen_urls.add(pdf_url)
                try:
                    resp = self._session.get(
                        pdf_url, timeout=self.timeout, stream=True,
                        headers={"Referer": doc_url},
                    )
                    resp.raise_for_status()
                    # Layer 1: validate download (magic bytes, size, content-type)
                    content = validate_download_stream(
                        resp, expected_type="pdf", config=self.safety_config,
                    )
                    filename = f"arxiv_{arxiv_id}.pdf"
                    filepath = site_dir / filename
                    if not filepath.exists():
                        filepath.write_bytes(content)
                    downloaded.append(str(filepath))
                except DownloadValidationError as e:
                    print(f"    [safety] Rejected arxiv {arxiv_id}: {e}")
                    continue
                except Exception:
                    continue
                continue

            # --- Check for document extensions ---
            ext = Path(parsed.path).suffix.lower()
            if ext not in DOCUMENT_EXTENSIONS:
                continue

            # Skip repo/source-code domains (GitHub, GitLab, etc.)
            if parsed.netloc in REPO_DOMAINS:
                continue

            # Deduplicate
            if doc_url in seen_urls:
                continue
            seen_urls.add(doc_url)

            try:
                resp = self._session.get(
                    doc_url, timeout=self.timeout, stream=True,
                    headers={"Referer": base_url},
                )
                resp.raise_for_status()

                # Verify content-type is not HTML (some sites redirect to
                # error pages with 200 + text/html for missing docs)
                ct = resp.headers.get("content-type", "")
                if "text/html" in ct and ext in {".pdf", ".docx", ".pptx",
                                                  ".xlsx", ".doc", ".ppt",
                                                  ".xls", ".odt", ".epub"}:
                    continue

                # Layer 1: validate download (magic bytes, size)
                # Map extension to expected type for magic bytes check
                from ipa.ingestion.content_safety import _EXT_TO_TYPE
                expected_type = _EXT_TO_TYPE.get(ext)
                # Some document hosts return generic bytes/content types;
                # keep strict checks for PDFs/images and avoid false rejects.
                if ext not in {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp"}:
                    expected_type = None
                content = validate_download_stream(
                    resp, expected_type=expected_type, config=self.safety_config,
                )

                # Build a safe filename: original name + hash for uniqueness
                original_name = Path(parsed.path).name
                safe_name = self._slugify(original_name)
                url_hash = hashlib.sha256(doc_url.encode()).hexdigest()[:12]
                filename = f"{safe_name}_{url_hash}{ext}"
                filepath = site_dir / filename
                if not filepath.exists():
                    filepath.write_bytes(content)
                downloaded.append(str(filepath))
            except DownloadValidationError as e:
                print(f"    [safety] Rejected {doc_url}: {e}")
                continue
            except Exception:
                continue
        return downloaded

    @staticmethod
    def _extract_arxiv_id(url: str) -> str | None:
        """Extract arxiv paper ID from a URL.

        Handles:
          - https://arxiv.org/abs/2608.07592
          - https://arxiv.org/pdf/2608.07592
          - https://arxiv.org/pdf/2608.07592.pdf
          - https://arxiv.org/abs/2608.07592v1
        """
        import re
        # Match arxiv.org/abs/<id> or arxiv.org/pdf/<id>
        match = re.search(
            r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
            url,
        )
        if match:
            arxiv_id = match.group(1)
            # Strip version suffix for the PDF URL (arxiv serves latest)
            return arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
        return None

    def _parse_json_api(
        self,
        api_url: str,
        id_field: str,
        url_template: str,
        days_back: int,
        url_pattern: str | None = None,
        exclude_paths: list[str] | None = None,
    ) -> list[str]:
        """Fetch a JSON API endpoint and extract article URLs.

        The API should return a JSON array of objects, each with an `id_field`
        containing the article ID. The URL is constructed from `url_template`
        by replacing `{id}` with the ID value.

        If days_back > 0, filters by the `date` field in each item.
        """
        from datetime import datetime, timedelta, timezone

        try:
            resp = self._session.get(api_url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException:
            return []

        try:
            data = resp.json()
        except (ValueError, TypeError):
            return []

        if not isinstance(data, list):
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back) if days_back > 0 else None
        path_re = re.compile(url_pattern) if url_pattern else None
        excludes = exclude_paths or []

        links: list[str] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            article_id = item.get(id_field)
            if not article_id:
                continue

            # Date filter (if date field exists and days_back > 0)
            if cutoff and item.get("date"):
                try:
                    date_str = str(item["date"]).replace("Z", "+00:00")
                    article_date = datetime.fromisoformat(date_str)
                    if article_date.tzinfo is None:
                        article_date = article_date.replace(tzinfo=timezone.utc)
                    if article_date < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass  # keep if date can't be parsed

            url = url_template.replace("{id}", str(article_id))

            # Apply url_pattern filter
            if path_re:
                path = urlparse(url).path
                if not path_re.match(path):
                    continue

            # Apply exclude_paths filter
            if any(ex in url for ex in excludes):
                continue

            links.append(url)

        return links

    def _parse_rss_feed(self, feed_url: str, days_back: int,
                        url_pattern: str | None = None,
                        exclude_paths: list[str] | None = None,
                        ) -> list[str]:
        """Parse an RSS/Atom feed and return article URLs within days_back.

        Extracts <link> from each <item> (RSS) or <entry> (Atom), filters by
        date (pubDate / updated) and optionally by url_pattern + exclude_paths.
        """
        from xml.etree import ElementTree as ET
        from datetime import datetime, timedelta, timezone

        try:
            resp = self._session.get(feed_url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException:
            return []

        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError:
            return []

        # Determine cutoff date
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

        # RSS 2.0: <item><link>, <item><pubDate>
        # Atom: <entry><link href="...">, <entry><updated>
        items = root.findall(".//item")
        is_atom = False
        if not items:
            items = root.findall(".//{http://www.w3.org/2005/Atom}entry")
            is_atom = bool(items)

        links: list[str] = []
        for item in items:
            # Extract link
            link = None
            if is_atom:
                link_el = item.find("{http://www.w3.org/2005/Atom}link")
                if link_el is not None:
                    link = link_el.get("href")
            else:
                link = item.findtext("link")

            if not link:
                continue

            # Extract date
            date_str = None
            if is_atom:
                date_str = (item.findtext("{http://www.w3.org/2005/Atom}published")
                           or item.findtext("{http://www.w3.org/2005/Atom}updated"))
            else:
                date_str = item.findtext("pubDate")

            # Parse date and filter by days_back
            if date_str:
                dt = None
                for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z",
                           "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"]:
                    try:
                        dt = datetime.strptime(date_str.strip(), fmt)
                        break
                    except ValueError:
                        continue
                if dt:
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt < cutoff:
                        continue

            # Apply url_pattern
            if url_pattern:
                parsed = urlparse(link)
                if not re.search(url_pattern, parsed.path):
                    continue

            # Apply exclude_paths
            if exclude_paths:
                if any(ex in link for ex in exclude_paths):
                    continue

            links.append(link)

        return links

    def _parse_sitemap(self, sitemap_url: str,
                       url_pattern: str | None = None,
                       exclude_paths: list[str] | None = None,
                       max_urls: int = 500,
                       ) -> list[str]:
        """Parse a sitemap (or sitemap index) and return article URLs.

        Handles both sitemapindex (references child sitemaps) and urlset
        (direct URLs).  Follows child sitemaps recursively up to max_urls.
        Filters by url_pattern and exclude_paths.
        """
        from xml.etree import ElementTree as ET

        try:
            resp = self._session.get(sitemap_url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException:
            return []

        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError:
            return []

        # Namespace handling â€” sitemaps use xmlns
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        urls: list[str] = []

        # Check if this is a sitemap index (<sitemapindex>)
        sitemap_tags = root.findall(f"{ns}sitemap")
        if sitemap_tags:
            # It's a sitemap index â€” follow child sitemaps
            for sm in sitemap_tags:
                if len(urls) >= max_urls:
                    break
                child_url = sm.findtext(f"{ns}loc")
                if child_url:
                    child_urls = self._parse_sitemap(
                        child_url, url_pattern, exclude_paths,
                        max_urls - len(urls),
                    )
                    urls.extend(child_urls)
            return urls

        # It's a urlset â€” extract <url><loc>
        for url_el in root.findall(f"{ns}url"):
            if len(urls) >= max_urls:
                break
            loc = url_el.findtext(f"{ns}loc")
            if not loc:
                continue

            # Apply url_pattern
            if url_pattern:
                parsed = urlparse(loc)
                if not re.search(url_pattern, parsed.path):
                    continue

            # Apply exclude_paths
            if exclude_paths:
                if any(ex in loc for ex in exclude_paths):
                    continue

            urls.append(loc)

        return urls

    def scrape_site(self, site: ScrapeSite) -> ScrapeSummary:
        """Scrape a single site: find articles, extract each.

        Engine behavior:
          - 'requests': fetch with requests, extract links from static HTML.
          - 'playwright': fetch with headless browser, extract links from
            rendered DOM.
          - 'auto': try requests first; if 0 links found, retry with Playwright.

        Discovery priority:
          1. json_api_url â€” fetch JSON API for article URLs (SPAs like Qwen)
          2. rss_feed â€” parse RSS/Atom feed for URLs with dates
          3. sitemap_url â€” parse sitemap (index) for URLs
          4. HTML listing crawl (with selector/pattern/excludes)
        """
        start = time.monotonic()

        # --- JSON API discovery (for SPAs that load content via API) ---
        if site.json_api_url and site.json_api_url_template:
            article_links = self._parse_json_api(
                site.json_api_url,
                site.json_api_id_field,
                site.json_api_url_template,
                site.days_back,
                url_pattern=site.url_pattern,
                exclude_paths=site.exclude_paths,
            )
            if article_links:
                if site.max_articles > 0:
                    article_links = article_links[:site.max_articles]
                results: list[ScrapeResult] = []
                images_downloaded = 0
                documents_downloaded = 0
                scraped = 0
                skipped = 0
                errors: list[str] = []

                for link in article_links:
                    if self.history.is_scraped(link) or not self.history.claim(link):
                        skipped += 1
                        continue
                    result = self.extract_article(link, days_back=site.days_back)
                    if result is None or result.error:
                        err = result.error if result else "extract returned None"
                        errors.append(err)
                        self.history.record(link, status="error", site_url=site.url)
                        self._rate_limit(link, site.delay_seconds)
                        continue
                    if result.date and site.days_back > 0:
                        try:
                            date_str = result.date.replace("Z", "+00:00")
                            article_date = datetime.fromisoformat(date_str)
                            if article_date.tzinfo is None:
                                article_date = article_date.replace(tzinfo=timezone.utc)
                            cutoff = datetime.now(timezone.utc) - timedelta(days=site.days_back)
                            if article_date < cutoff:
                                skipped += 1
                                continue
                        except (ValueError, TypeError):
                            pass
                    results.append(result)
                    scraped += 1
                    images_downloaded += len(result.image_paths)
                    documents_downloaded += len(result.document_paths)
                    self.history.record(link, title=result.title, status="ok", site_url=site.url)
                    self._rate_limit(link, site.delay_seconds)

                return ScrapeSummary(
                    site_url=site.url,
                    total_articles_found=len(article_links),
                    articles_scraped=scraped,
                    articles_skipped=skipped,
                    images_downloaded=images_downloaded,
                    documents_downloaded=documents_downloaded,
                    errors=errors,
                    results=results,
                    elapsed_seconds=time.monotonic() - start,
                )

        # --- RSS feed discovery (preferred when configured) ---
        if site.rss_feed:
            article_links = self._parse_rss_feed(
                site.rss_feed, site.days_back,
                url_pattern=site.url_pattern,
                exclude_paths=site.exclude_paths,
            )
            # If RSS gave us links, skip the HTML listing crawl
            if article_links:
                # Limit
                if site.max_articles > 0:
                    article_links = article_links[:site.max_articles]

                results: list[ScrapeResult] = []
                images_downloaded = 0
                documents_downloaded = 0
                errors: list[str] = []
                scraped = 0
                skipped = 0

                for link in article_links:
                    # Skip already-scraped URLs (deduplication)
                    if self.history.is_scraped(link) or not self.history.claim(link):
                        skipped += 1
                        continue
                    result = self.extract_article(link, days_back=site.days_back)
                    if result.error:
                        errors.append(f"{link}: {result.error}")
                        skipped += 1
                    else:
                        results.append(result)
                        scraped += 1
                        images_downloaded += len(result.image_paths)
                        documents_downloaded += len(result.document_paths)
                        self.history.record(link, title=result.title, status="ok", site_url=site.url)
                    self._rate_limit(link, site.delay_seconds)

                return ScrapeSummary(
                    site_url=site.url,
                    total_articles_found=len(article_links),
                    articles_scraped=scraped,
                    articles_skipped=skipped,
                    images_downloaded=images_downloaded,
                    documents_downloaded=documents_downloaded,
                    errors=errors,
                    results=results,
                    elapsed_seconds=time.monotonic() - start,
                )
            # RSS returned nothing â€” fall through to sitemap or HTML crawl

        # --- Sitemap discovery (second priority) ---
        if site.sitemap_url:
            article_links = self._parse_sitemap(
                site.sitemap_url,
                url_pattern=site.url_pattern,
                exclude_paths=site.exclude_paths,
                max_urls=site.max_articles * 3 if site.max_articles > 0 else 10000,
            )
            if article_links:
                if site.max_articles > 0:
                    article_links = article_links[:site.max_articles]

                results: list[ScrapeResult] = []
                images_downloaded = 0
                documents_downloaded = 0
                errors: list[str] = []
                scraped = 0
                skipped = 0

                for link in article_links:
                    # Skip already-scraped URLs (deduplication)
                    if self.history.is_scraped(link) or not self.history.claim(link):
                        skipped += 1
                        continue
                    result = self.extract_article(link, days_back=site.days_back)
                    if result.error:
                        errors.append(f"{link}: {result.error}")
                        skipped += 1
                    elif result.date and site.days_back > 0:
                        try:
                            date_str = result.date.replace("Z", "+00:00")
                            article_date = datetime.fromisoformat(date_str)
                            if article_date.tzinfo is None:
                                article_date = article_date.replace(tzinfo=timezone.utc)
                            cutoff = datetime.now(timezone.utc) - timedelta(days=site.days_back)
                            if article_date < cutoff:
                                skipped += 1
                                continue
                        except (ValueError, TypeError):
                            pass
                    results.append(result)
                    scraped += 1
                    images_downloaded += len(result.image_paths)
                    documents_downloaded += len(result.document_paths)
                    self.history.record(link, title=result.title, status="ok", site_url=site.url)
                    self._rate_limit(link, site.delay_seconds)

                return ScrapeSummary(
                    site_url=site.url,
                    total_articles_found=len(article_links),
                    articles_scraped=scraped,
                    articles_skipped=skipped,
                    images_downloaded=images_downloaded,
                    documents_downloaded=documents_downloaded,
                    errors=errors,
                    results=results,
                    elapsed_seconds=time.monotonic() - start,
                )
            # Sitemap returned nothing â€” fall through to HTML crawl

        # --- Fetch listing page ---
        # Use per-site engine override if set, otherwise use global engine
        fetch_engine = site.engine or self.engine
        html = self.fetch_page(site.url, use_engine=fetch_engine)
        if html is None and fetch_engine in ("auto", "playwright"):
            # requests failed â€” try Playwright
            html = self.fetch_page(site.url, use_engine="playwright")

        if html is None:
            return ScrapeSummary(
                site_url=site.url, total_articles_found=0,
                articles_scraped=0, articles_skipped=0,
                images_downloaded=0,
                errors=[f"Failed to fetch listing page: {site.url}"],
                elapsed_seconds=time.monotonic() - start,
            )

        # --- Extract article links ---
        article_links = self.extract_article_links(
            html, site.url,
            selector=site.article_selector,
            url_pattern=site.url_pattern,
            exclude_paths=site.exclude_paths,
            allowed_domains=site.allowed_domains,
        )

        # --- Auto fallback: if no links found with requests, try Playwright ---
        if not article_links and self.engine == "auto":
            pw = self._get_playwright()
            pw_html, pw_hrefs = pw.fetch_page_with_links(site.url)
            if pw_html:
                # Build a synthetic HTML from the DOM-extracted hrefs
                # so extract_article_links can process them
                soup = BeautifulSoup(pw_html, "lxml")
                # Also try DOM-extracted hrefs (more reliable for JS-rendered)
                if pw_hrefs:
                    # Inject any hrefs that aren't in the HTML as <a> tags
                    existing_hrefs = {a.get("href", "") for a in soup.find_all("a", href=True)}
                    body = soup.find("body") or soup.find("html")
                    if body:
                        for href in pw_hrefs:
                            if href not in existing_hrefs:
                                new_a = soup.new_tag("a", href=href)
                                body.append(new_a)
                article_links = self.extract_article_links(
                    str(soup), site.url,
                    selector=site.article_selector,
                    url_pattern=site.url_pattern,
                    exclude_paths=site.exclude_paths,
                    allowed_domains=site.allowed_domains,
                )

        # --- Pagination: follow additional pages ---
        if site.paginate and article_links:
            base_url = site.url.rstrip("/") + "/"
            seen = set(article_links)
            
            # Special: Blogger-style pagination (?updated-max=<timestamp>)
            if site.paginate_url_template == "blogger":
                current_html = html  # already fetched the first page
                for page_num in range(2, site.max_pages + 1):
                    # Extract the "next page" link from the current page
                    next_url = None
                    soup = BeautifulSoup(current_html, "lxml")
                    for a in soup.find_all("a", href=True):
                        text = a.get_text().lower().strip()
                        if "next page" in text or text == "Â»" or text == "â†’":
                            next_url = a["href"]
                            break
                    if not next_url:
                        break  # no more pages
                    current_html = self.fetch_page(next_url, use_engine=fetch_engine)
                    if current_html is None:
                        break
                    page_links = self.extract_article_links(
                        current_html, next_url,
                        selector=site.article_selector,
                        url_pattern=site.url_pattern,
                        exclude_paths=site.exclude_paths,
                        allowed_domains=site.allowed_domains,
                    )
                    new_links = [l for l in page_links if l not in seen]
                    if not new_links:
                        break
                    seen.update(new_links)
                    article_links.extend(new_links)
                    self._rate_limit(next_url, site.delay_seconds)
            else:
                for page_num in range(2, site.max_pages + 1):
                    page_html = None
                    # 1. Try custom URL template
                    if site.paginate_url_template:
                        page_url = site.paginate_url_template.replace("{n}", str(page_num))
                        page_html = self.fetch_page(page_url, use_engine=fetch_engine)
                    # 2. Try path-based pagination (/page/N/)
                    if page_html is None:
                        page_url = f"{base_url}page/{page_num}/"
                        page_html = self.fetch_page(page_url, use_engine=fetch_engine)
                    # 3. Fall back to query-param pagination (?page=N)
                    if page_html is None:
                        sep = "&" if "?" in site.url else "?"
                        page_url = f"{site.url}{sep}page={page_num}"
                        page_html = self.fetch_page(page_url, use_engine=fetch_engine)
                    if page_html is None:
                        break
                    page_links = self.extract_article_links(
                        page_html, page_url,
                        selector=site.article_selector,
                        url_pattern=site.url_pattern,
                        exclude_paths=site.exclude_paths,
                        allowed_domains=site.allowed_domains,
                    )
                    new_links = [l for l in page_links if l not in seen]
                    if not new_links:
                        break  # no new articles â€” stop paginating
                    seen.update(new_links)
                    article_links.extend(new_links)
                    self._rate_limit(page_url, site.delay_seconds)

        # Limit
        if site.max_articles > 0:
            article_links = article_links[:site.max_articles]

        results: list[ScrapeResult] = []
        images_downloaded = 0
        documents_downloaded = 0
        errors: list[str] = []
        scraped = 0
        skipped = 0

        # Date filter
        cutoff = datetime.now(timezone.utc) - timedelta(days=site.days_back)

        for link in article_links:
            # Skip already-scraped URLs (deduplication)
            if self.history.is_scraped(link):
                skipped += 1
                continue
            result = self.extract_article(link, days_back=site.days_back)
            if result.error:
                errors.append(f"{link}: {result.error}")
                skipped += 1
            elif result.date and site.days_back > 0:
                try:
                    # trafilatura returns ISO dates (may be date-only or
                    # datetime with/without timezone).
                    date_str = result.date.replace("Z", "+00:00")
                    article_date = datetime.fromisoformat(date_str)
                    # Make naive datetimes UTC-aware for comparison
                    if article_date.tzinfo is None:
                        article_date = article_date.replace(tzinfo=timezone.utc)
                    if article_date < cutoff:
                        skipped += 1
                        continue
                except (ValueError, TypeError):
                    pass  # If we can't parse date, include it

            results.append(result)
            scraped += 1
            images_downloaded += len(result.image_paths)
            documents_downloaded += len(result.document_paths)
            self.history.record(link, title=result.title, status="ok", site_url=site.url)

            # Politeness delay
            self._rate_limit(link, site.delay_seconds)

        return ScrapeSummary(
            site_url=site.url,
            total_articles_found=len(article_links),
            articles_scraped=scraped,
            articles_skipped=skipped,
            images_downloaded=images_downloaded,
            documents_downloaded=documents_downloaded,
            errors=errors,
            results=results,
            elapsed_seconds=time.monotonic() - start,
        )

    def scrape_sites(self, sites: list[ScrapeSite]) -> list[ScrapeSummary]:
        """Scrape multiple sites."""
        return [self.scrape_site(site) for site in sites]

    def save_article(self, result: ScrapeResult) -> Path:
        """Save a scraped article as a .txt file in the output directory."""
        domain = urlparse(result.url).netloc
        site_dir = self.output_dir / self._slugify(domain)
        site_dir.mkdir(parents=True, exist_ok=True)

        # Create a slug from the URL path
        path = urlparse(result.url).path
        slug = self._slugify(path) or "article"
        # Limit slug length
        slug = slug[:80]

        # Add a short hash for uniqueness
        url_hash = hashlib.sha256(result.url.encode()).hexdigest()[:8]
        filename = f"{slug}_{url_hash}.txt"
        filepath = site_dir / filename

        # Build content: title + URL + date + text
        lines = []
        if result.title:
            lines.append(f"# {result.title}")
            lines.append("")
        lines.append(f"Source: {result.url}")
        if result.date:
            lines.append(f"Date: {result.date}")
        lines.append("")
        lines.append(result.text)

        # Append OCR text if available
        if result.ocr_texts:
            lines.append("")
            lines.append("--- OCR extracted text from images ---")
            for i, ocr_text in enumerate(result.ocr_texts):
                if ocr_text.strip():
                    lines.append(f"[Image {i+1}]")
                    lines.append(ocr_text)
                    lines.append("")

        # Append document references if any were downloaded
        if result.document_paths:
            lines.append("")
            lines.append("--- Linked documents downloaded to Landing zone ---")
            for i, doc_path in enumerate(result.document_paths):
                lines.append(f"[Document {i+1}] {doc_path}")
            lines.append("")

        filepath.write_text("\n".join(lines), encoding="utf-8")
        return filepath

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to a filesystem-safe slug."""
        # Remove protocol
        text = re.sub(r"https?://", "", text)
        # Replace common separators (dots, slashes, colons) with hyphens
        text = re.sub(r"[./:]+", "-", text)
        # Remove non-alphanumeric (except spaces and hyphens)
        text = re.sub(r"[^\w\s-]", "", text)
        # Collapse whitespace and hyphens
        text = re.sub(r"[\s_-]+", "-", text).strip("-")
        return text.lower()

    def close(self) -> None:
        self._session.close()
        if self._playwright:
            self._playwright.close()
            self._playwright = None
        self.history.close()

    def __enter__(self) -> "WebScraper":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

