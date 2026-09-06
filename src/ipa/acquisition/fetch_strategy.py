"""Fetch strategy â€” pluggable HTTP fetching with automatic fallback.

Three strategies:
  - RequestsFetchStrategy: fast, for server-rendered HTML (no JS)
  - PlaywrightFetchStrategy: headless browser, for JS-rendered sites
  - AutoFetchStrategy: try requests first, fall back to Playwright

All strategies integrate with Layer 1 safety (content_safety.py):
  - Magic bytes validation
  - Size limit enforcement
  - Redirect limit
  - Timeout

Usage:
  from ipa.acquisition.fetch_strategy import AutoFetchStrategy

  fetcher = AutoFetchStrategy()
  result = fetcher.fetch("https://example.com/article")
  print(result.text)       # clean HTML
  print(result.status)     # 200
  print(result.engine)     # "requests" or "playwright"

  # With content filter for clean text:
  from ipa.ingestion.content_filter import ContentFilter
  cf = ContentFilter()
  text = cf.extract(result.html)
"""
from __future__ import annotations

import ipaddress
import os
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urlunparse


def normalize_url(url: str) -> str:
    p = urlparse(url.strip())
    if p.scheme.lower() not in {"http", "https"} or not p.netloc:
        return ""
    return urlunparse((p.scheme.lower(), p.netloc.lower(), p.path or "/", "", p.query, ""))

import requests

from ipa.ingestion.content_safety import (
    DownloadValidationConfig,
    DownloadValidationError,
    validate_download_stream,
    detect_file_type,
    validate_html_size,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    """Result of fetching a URL."""
    url: str
    status: int
    html: str
    text: str = ""  # Extracted text (filled by caller with ContentFilter)
    engine: str = "unknown"  # "requests" or "playwright"
    elapsed_seconds: float = 0.0
    content_type: str = ""
    final_url: str = ""  # After redirects
    error: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None and 200 <= self.status < 400


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class FetchStrategy(ABC):
    """Abstract base class for URL fetching strategies."""

    @abstractmethod
    def fetch(
        self,
        url: str,
        timeout: int = 30,
        safety_config: DownloadValidationConfig | None = None,
    ) -> FetchResult:
        """Fetch a URL and return the result.

        Args:
            url: URL to fetch.
            timeout: Request timeout in seconds.
            safety_config: Download validation config (Layer 1 safety).

        Returns:
            FetchResult with HTML content and metadata.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources (browser, sessions, etc.)."""
        ...

    def __enter__(self) -> "FetchStrategy":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# RequestsFetchStrategy â€” fast, for server-rendered HTML
# ---------------------------------------------------------------------------

def _validate_final_host(original_url: str, final_url: str) -> None:
    """Reject unsafe or policy-violating redirect destinations."""
    if not final_url:
        return
    original = urlparse(original_url).hostname
    final = urlparse(final_url).hostname
    if not final or urlparse(final_url).scheme not in {"http", "https"}:
        raise DownloadValidationError("redirected to an invalid URL scheme or host")
    allowed = {d.strip().lower().lstrip(".") for d in os.environ.get("IPA_ALLOWED_DOMAINS", "").split(",") if d.strip()}
    if allowed and not any(final.lower() == d or final.lower().endswith("." + d) for d in allowed):
        raise DownloadValidationError(f"redirected outside allowed domains: {final}")
    if original and final.lower() != original.lower() and not allowed:
        try:
            for address in socket.getaddrinfo(final, None):
                ip = ipaddress.ip_address(address[4][0])
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                    raise DownloadValidationError("redirected to a private or reserved host")
        except socket.gaierror as exc:
            raise DownloadValidationError("redirect destination cannot be resolved") from exc


class RequestsFetchStrategy(FetchStrategy):
    """Fast HTTP fetching using requests library.

    Best for: server-rendered HTML, APIs, static sites.
    Not suitable for: JavaScript-rendered SPAs (React, Vue, etc.).
    """

    def __init__(
        self,
        user_agent: str = "IPA-Research-Bot/1.0",
        safety_config: DownloadValidationConfig | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.safety_config = safety_config or DownloadValidationConfig(
            max_size_mb=200,
            max_redirects=3,
            timeout_seconds=30,
        )
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self._session.max_redirects = self.safety_config.max_redirects

    def fetch(
        self,
        url: str,
        timeout: int = 30,
        safety_config: DownloadValidationConfig | None = None,
    ) -> FetchResult:
        start = time.monotonic()
        cfg = safety_config or self.safety_config

        url = normalize_url(url) or url
        try:
            # Retry only transient transport/server failures; never retry 4xx or safety errors.
            resp = None
            for attempt in range(3):
                try:
                    resp = self._session.get(url, timeout=timeout, stream=True)
                    if resp.status_code not in {408, 425, 429} and resp.status_code < 500:
                        break
                except (requests.Timeout, requests.ConnectionError):
                    if attempt == 2:
                        raise
                if attempt < 2:
                    retry_after = resp.headers.get("Retry-After") if resp is not None else None
                    try: wait = min(float(retry_after), 30.0) if retry_after else 2 ** attempt
                    except ValueError: wait = 2 ** attempt
                    time.sleep(wait)
            assert resp is not None

            # Determine expected type from URL extension
            parsed = urlparse(url)
            ext = ""
            path = parsed.path.lower()
            if path.endswith(".pdf"):
                ext = "pdf"
            elif path.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                ext = "image"

            # For HTML pages, we read directly (not through safety stream)
            # because HTML is text and we need the full content for parsing
            content_type = resp.headers.get("content-type", "")
            _validate_final_host(url, resp.url)

            if "text/html" in content_type or "application/xhtml" in content_type:
                # HTML â€” read with the same bounded safety policy as binaries
                content = validate_download_stream(resp, expected_type=None, config=cfg)
                validate_html_size(content, cfg)
                html = content.decode(resp.encoding or "utf-8", errors="replace")
                return FetchResult(
                    url=url,
                    status=resp.status_code,
                    html=html,
                    engine="requests",
                    elapsed_seconds=time.monotonic() - start,
                    content_type=content_type,
                    final_url=resp.url,
                )

            # Binary content â€” validate through safety layer
            if ext:
                content = validate_download_stream(resp, expected_type=ext, config=cfg)
            else:
                content = validate_download_stream(resp, config=cfg)

            # For binary, return empty html (caller handles differently)
            return FetchResult(
                url=url,
                status=resp.status_code,
                html="",
                engine="requests",
                elapsed_seconds=time.monotonic() - start,
                content_type=content_type,
                final_url=resp.url,
                metadata={"binary_size": str(len(content))},
            )

        except DownloadValidationError as e:
            return FetchResult(
                url=url, status=0, html="", engine="requests",
                elapsed_seconds=time.monotonic() - start,
                error=f"safety: {e}",
            )
        except requests.RequestException as e:
            return FetchResult(
                url=url, status=0, html="", engine="requests",
                elapsed_seconds=time.monotonic() - start,
                error=str(e),
            )

    def close(self) -> None:
        self._session.close()


# ---------------------------------------------------------------------------
# PlaywrightFetchStrategy â€” headless browser, for JS-rendered sites
# ---------------------------------------------------------------------------

class PlaywrightFetchStrategy(FetchStrategy):
    """Headless browser fetching using Playwright.

    Best for: JavaScript-rendered SPAs, dynamic content, sites that
    require JS execution to display content.

    Requires: playwright installed + browser binaries (`playwright install chromium`).
    """

    def __init__(
        self,
        user_agent: str = "IPA-Research-Bot/1.0",
        headless: bool = True,
        wait_for: str | None = None,
        safety_config: DownloadValidationConfig | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.headless = headless
        self.wait_for = wait_for
        self.safety_config = safety_config or DownloadValidationConfig(
            max_size_mb=200,
            max_redirects=3,
            timeout_seconds=30,
        )
        self._playwright = None
        self._browser = None
        self._context = None

    def _ensure_browser(self) -> None:
        """Lazy-init Playwright browser."""
        if self._browser is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise ImportError(
                "Playwright is not installed. Install with: "
                "pip install playwright && playwright install chromium"
            ) from e

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(
            user_agent=self.user_agent,
            viewport={"width": 1280, "height": 720},
        )

    def fetch(
        self,
        url: str,
        timeout: int = 30,
        safety_config: DownloadValidationConfig | None = None,
    ) -> FetchResult:
        start = time.monotonic()
        cfg = safety_config or self.safety_config

        try:
            self._ensure_browser()
            assert self._context is not None

            page = self._context.new_page()
            page.set_default_timeout(timeout * 1000)

            # Navigate
            response = page.goto(url, wait_until="domcontentloaded")

            # Wait for specific selector if configured
            if self.wait_for:
                try:
                    page.wait_for_selector(self.wait_for, timeout=timeout * 1000)
                except Exception:
                    pass  # Selector not found â€” continue with what we have

            # Give JS a moment to render
            page.wait_for_timeout(500)

            # Get the rendered HTML
            html = page.content()
            final_url = page.url
            _validate_final_host(url, final_url)
            status = response.status if response else 200
            content_type = response.headers.get("content-type", "") if response else ""

            # Validate size (Layer 1)
            if len(html.encode("utf-8")) > cfg.max_size_mb * 1024 * 1024:
                return FetchResult(
                    url=url, status=status, html="", engine="playwright",
                    elapsed_seconds=time.monotonic() - start,
                    error=f"safety: content too large ({len(html)} chars)",
                )

            return FetchResult(
                url=url,
                status=status,
                html=html,
                engine="playwright",
                elapsed_seconds=time.monotonic() - start,
                content_type=content_type,
                final_url=final_url,
            )

        except ImportError as e:
            return FetchResult(
                url=url, status=0, html="", engine="playwright",
                elapsed_seconds=time.monotonic() - start,
                error=str(e),
            )
        except Exception as e:
            return FetchResult(
                url=url, status=0, html="", engine="playwright",
                elapsed_seconds=time.monotonic() - start,
                error=str(e),
            )
        finally:
            try:
                if 'page' in locals():
                    page.close()
            except Exception:
                pass

    def close(self) -> None:
        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                pass
            self._context = None
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None


# ---------------------------------------------------------------------------
# AutoFetchStrategy â€” try requests, fall back to Playwright
# ---------------------------------------------------------------------------

class AutoFetchStrategy(FetchStrategy):
    """Automatic strategy selection: requests first, Playwright fallback.

    Decision logic:
    1. Try RequestsFetchStrategy (fast, no browser overhead)
    2. If the response looks like a JS-rendered SPA (empty body, very
       little text, or contains JS app markers), retry with Playwright
    3. If requests fails entirely (timeout, connection error), try Playwright

    This gives the speed of requests for static sites and the robustness
    of Playwright for dynamic sites, without the caller needing to know
    which one to use.
    """

    # Markers that indicate the page needs JS rendering
    _SPA_MARKERS = [
        '<div id="root"></div>',
        '<div id="app"></div>',
        '<div id="__next"></div>',
        '<div id="__nuxt"></div>',
        'data-reactroot',
        'window.__INITIAL_STATE__',
        'ng-app',
        'vue-app',
    ]

    # Minimum text length to consider the page "rendered"
    _MIN_RENDERED_TEXT = 200

    def __init__(
        self,
        user_agent: str = "IPA-Research-Bot/1.0",
        playwright_headless: bool = True,
        playwright_wait_for: str | None = None,
        safety_config: DownloadValidationConfig | None = None,
    ) -> None:
        self._requests_strategy = RequestsFetchStrategy(
            user_agent=user_agent,
            safety_config=safety_config,
        )
        self._playwright_strategy: PlaywrightFetchStrategy | None = None
        self._pw_headless = playwright_headless
        self._pw_wait_for = playwright_wait_for
        self._pw_config = playwright_headless, playwright_wait_for
        self._safety_config = safety_config

    def _get_playwright(self) -> PlaywrightFetchStrategy:
        """Lazy-init Playwright strategy (only when needed)."""
        if self._playwright_strategy is None:
            self._playwright_strategy = PlaywrightFetchStrategy(
                user_agent=self._requests_strategy.user_agent,
                headless=self._pw_headless,
                wait_for=self._pw_wait_for,
                safety_config=self._safety_config,
            )
        return self._playwright_strategy

    def _needs_playwright(self, result: FetchResult) -> bool:
        """Check if a requests result needs Playwright fallback.

        Returns True if:
        - The HTML is very short (likely JS-rendered shell)
        - The HTML contains SPA markers
        - The request failed with a connection error
        """
        if not result.success:
            # Connection errors, timeouts â†’ try Playwright
            if result.error and any(
                s in result.error.lower()
                for s in ("timeout", "connection", "refused", "reset")
            ):
                return True
            return False

        html = result.html
        if len(html) < self._MIN_RENDERED_TEXT:
            return True

        # Check for SPA markers
        html_lower = html.lower()
        for marker in self._SPA_MARKERS:
            if marker.lower() in html_lower:
                return True

        # Check text-to-HTML ratio (very low = mostly JS)
        from bs4 import BeautifulSoup
        try:
            soup = BeautifulSoup(html, "lxml")
            text = soup.get_text(strip=True)
            if len(text) < self._MIN_RENDERED_TEXT:
                return True
        except Exception:
            pass

        return False

    def fetch(
        self,
        url: str,
        timeout: int = 30,
        safety_config: DownloadValidationConfig | None = None,
    ) -> FetchResult:
        """Fetch with automatic strategy selection.

        Tries requests first. If the result looks like a JS-rendered
        shell or the request fails, falls back to Playwright.
        """
        # Step 1: Try requests
        result = self._requests_strategy.fetch(url, timeout, safety_config)

        if result.success and not self._needs_playwright(result):
            return result

        # Step 2: Fall back to Playwright
        try:
            pw = self._get_playwright()
            pw_result = pw.fetch(url, timeout, safety_config)

            # If Playwright also failed, return the original requests error
            # (more informative than Playwright's generic errors)
            if not pw_result.success and result.success:
                # Requests got something, Playwright failed â€” keep requests
                return result

            return pw_result
        except ImportError:
            # Playwright not installed â€” return requests result
            if not result.success:
                result.metadata["playwright_unavailable"] = "true"
            return result

    def close(self) -> None:
        self._requests_strategy.close()
        if self._playwright_strategy is not None:
            self._playwright_strategy.close()
            self._playwright_strategy = None

