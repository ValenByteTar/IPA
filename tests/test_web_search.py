"""Tests for web_search URL helpers (seed detection for research_topic)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ipa.agent.web_search import extract_urls, query_from_url, strip_urls  # noqa: E402


def test_extract_urls_single():
    assert extract_urls("mira https://example.com/a?b=1") == ["https://example.com/a?b=1"]


def test_extract_urls_strips_trailing_punctuation():
    txt = "Como que no, mira https://www.pagina12.com.ar/2026/09/12/nota/."
    assert extract_urls(txt) == ["https://www.pagina12.com.ar/2026/09/12/nota/"]
    assert extract_urls("(ver https://example.com/x), luego") == ["https://example.com/x"]


def test_extract_urls_dedupes_preserving_order():
    txt = "https://a.com/1 y de nuevo https://a.com/1 y https://b.com/2"
    assert extract_urls(txt) == ["https://a.com/1", "https://b.com/2"]


def test_extract_urls_ignores_non_http():
    assert extract_urls("ftp://x.com/a y www.example.com") == []


def test_extract_urls_empty():
    assert extract_urls("") == []
    assert extract_urls("sin links acá") == []


def test_strip_urls_keeps_text():
    txt = "investigá esto https://example.com/a sobre IA"
    assert strip_urls(txt) == "investigá esto sobre IA"


def test_strip_urls_url_only_is_empty():
    assert strip_urls("https://example.com/a") == ""


def test_query_from_url_uses_slug():
    url = "https://www.pagina12.com.ar/2026/09/12/las-big-tech-de-la-ia-dicen-estar-de-acuerdo-en-frenar-su-desarrollo/"
    assert query_from_url(url) == (
        "las big tech de la ia dicen estar de acuerdo en frenar su desarrollo"
    )


def test_query_from_url_strips_extension_and_underscores():
    assert query_from_url("https://x.com/blog/como_usar_rag.html") == "como usar rag"


def test_query_from_url_empty_for_bare_domain():
    assert query_from_url("https://example.com/") == ""
    assert query_from_url("https://example.com") == ""


def test_query_from_url_empty_for_numeric_slug():
    assert query_from_url("https://example.com/123/456") == ""
