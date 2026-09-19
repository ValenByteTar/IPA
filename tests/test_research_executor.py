"""Fase 1 tests: research_topic executor + web_source contract validation."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "validation"))

from ipa.agent import AgentMemory, ToolContext, WebSource, execute_research, load_identity  # noqa: E402
from ipa.agent.research_executor import _trust_label_for_domain, _snippet_relevance, _content_quality, _article_is_too_old  # noqa: E402
from ipa.agent.web_search import SearchResult, SearchSummary  # noqa: E402
from validate_agent_contract import validate  # noqa: E402


NOW = "2026-09-06T12:00:00Z"


@pytest.fixture()
def memory(tmp_path):
    with AgentMemory(store_path=tmp_path / "agent.db") as store:
        yield store


# ---------------------------------------------------------------------------
# WebSource contract validation
# ---------------------------------------------------------------------------

def test_web_source_contract_validates():
    ws = WebSource(
        web_source_id="web_source:testws001",
        source_url="https://arxiv.org/abs/2401.12345",
        fetched_at=NOW,
        content_hash="sha256:" + "a" * 64,
        trust_label="high",
        fetch_method="requests",
        content_type="text/html",
        byte_size=12345,
        title="Test Paper",
    )
    errors = validate("WebSource", ws.to_contract())
    assert errors == [], errors


def test_web_source_rejects_invalid_trust_label():
    ws = WebSource(
        web_source_id="web_source:testws002",
        source_url="https://example.com",
        fetched_at=NOW,
        content_hash="sha256:" + "a" * 64,
        trust_label="super_trustworthy",  # invalid
        fetch_method="requests",
        content_type="text/html",
        byte_size=100,
    )
    errors = validate("WebSource", ws.to_contract())
    assert any("trust_label" in e for e in errors)


def test_web_source_rejects_invalid_fetch_method():
    ws = WebSource(
        web_source_id="web_source:testws003",
        source_url="https://example.com",
        fetched_at=NOW,
        content_hash="sha256:" + "a" * 64,
        trust_label="medium",
        fetch_method="telepathy",  # invalid
        content_type="text/html",
        byte_size=100,
    )
    errors = validate("WebSource", ws.to_contract())
    assert any("fetch_method" in e for e in errors)


# ---------------------------------------------------------------------------
# Trust label classification
# ---------------------------------------------------------------------------

def test_trust_label_arxiv_is_high():
    assert _trust_label_for_domain("arxiv.org") == "high"


def test_trust_label_github_is_high():
    assert _trust_label_for_domain("github.com") == "high"


def test_trust_label_wikipedia_is_medium():
    assert _trust_label_for_domain("en.wikipedia.org") == "medium"


def test_trust_label_random_blog_is_low():
    assert _trust_label_for_domain("random-blog.example.com") == "low"


# ---------------------------------------------------------------------------
# Pre-scrape relevance filter (snippet-based)
# ---------------------------------------------------------------------------

def test_snippet_relevance_high_for_matching_terms():
    score = _snippet_relevance("python asyncio tutorial", "Python AsyncIO Tutorial", "Learn async programming in Python")
    assert score > 0.5


def test_snippet_relevance_low_for_unrelated_result():
    score = _snippet_relevance("python asyncio tutorial", "Best Pizza Recipes", "How to make pizza dough at home")
    assert score < 0.2


def test_snippet_relevance_zero_for_no_overlap():
    score = _snippet_relevance("python asyncio", "Cooking Guide", "How to bake a cake")
    assert score == 0.0


def test_snippet_relevance_title_bonus():
    """Title matches should boost the score over snippet-only matches."""
    score_title = _snippet_relevance("python asyncio", "Python asyncio guide", "some random text here")
    score_snippet_only = _snippet_relevance("python asyncio", "unrelated title here", "python asyncio tutorial")
    # Both have full term overlap, but title match adds a bonus
    assert score_title >= score_snippet_only


# ---------------------------------------------------------------------------
# Post-scrape quality filter (content-based)
# ---------------------------------------------------------------------------

def test_content_quality_rejects_short_content():
    quality, reason = _content_quality("too short", "python asyncio")
    assert quality == 0.0
    assert "short" in reason


def test_content_quality_rejects_no_query_overlap():
    long_text = "a" * 500 + " this is about cooking and recipes " + "b" * 500
    quality, reason = _content_quality(long_text, "python asyncio")
    assert quality == 0.0
    assert "no query terms" in reason


def test_content_quality_rejects_cookie_wall():
    text = "please enable javascript to view this page. python asyncio tutorial. " + "x" * 500
    quality, reason = _content_quality(text, "python asyncio")
    assert quality == 0.0
    assert "blocked" in reason


def test_content_quality_accepts_good_content():
    text = (
        "Python asyncio is a library for writing concurrent code. "
        "This tutorial covers coroutines, event loops, and async/await syntax. "
        + "Here is a detailed explanation of how asyncio works. " * 20
    )
    quality, reason = _content_quality(text, "python asyncio tutorial")
    assert quality > 0.2
    assert reason == "passed"


# ---------------------------------------------------------------------------
# Date filter (post-scrape age check)
# ---------------------------------------------------------------------------

def test_article_too_old_rejects_old_article():
    from datetime import datetime, timezone, timedelta
    old_date = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    assert _article_is_too_old(old_date, max_age_days=365) is True


def test_article_too_old_accepts_recent_article():
    from datetime import datetime, timezone, timedelta
    recent_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    assert _article_is_too_old(recent_date, max_age_days=365) is False


def test_article_too_old_returns_false_for_missing_date():
    assert _article_is_too_old(None, max_age_days=365) is False


def test_article_too_old_returns_false_for_unparseable_date():
    assert _article_is_too_old("not-a-date", max_age_days=365) is False


def test_article_too_old_disabled_when_max_age_zero():
    from datetime import datetime, timezone, timedelta
    old_date = (datetime.now(timezone.utc) - timedelta(days=9999)).isoformat()
    assert _article_is_too_old(old_date, max_age_days=0) is False


# ---------------------------------------------------------------------------
# Research executor — error handling without network
# ---------------------------------------------------------------------------

def test_research_topic_fails_gracefully_without_network(memory, tmp_path, monkeypatch):
    """When web search fails (no network), the tool returns a failed result, not a crash."""
    from ipa.agent import research_executor as re_module

    def mock_search_web(query, **kwargs):
        return SearchSummary(query=query, error="network unavailable")

    monkeypatch.setattr(re_module, "search_web", mock_search_web)

    ctx = ToolContext(memory=memory)
    identity = load_identity()
    sid = memory.open_session(interface="cli", role="general", identity_hash=identity.identity_hash)
    ep = memory.record_episode(sid, turn_role="user", content="research test", identity_hash=identity.identity_hash)

    call, result, research = execute_research(
        "test query", ctx,
        session_id=sid, episode_id=ep.episode_id,
        max_urls=2, max_seconds=10,
        landing_dir=tmp_path / "landing",
    )

    assert call.status == "failed"
    assert result.status == "failed"
    assert "web search failed" in (result.error or "")
    assert validate("ToolCall", call.to_contract()) == []
    assert validate("ToolResult", result.to_contract()) == []


def test_research_topic_fails_when_no_results(memory, tmp_path, monkeypatch):
    """When web search returns no results, the tool fails with a clear message."""
    from ipa.agent import research_executor as re_module

    def mock_search_web(query, **kwargs):
        return SearchSummary(query=query, results=[])

    monkeypatch.setattr(re_module, "search_web", mock_search_web)

    ctx = ToolContext(memory=memory)
    identity = load_identity()
    sid = memory.open_session(interface="cli", role="general", identity_hash=identity.identity_hash)
    ep = memory.record_episode(sid, turn_role="user", content="no results test", identity_hash=identity.identity_hash)

    call, result, research = execute_research(
        "obscure query with no results", ctx,
        session_id=sid, episode_id=ep.episode_id,
        max_urls=3, max_seconds=10,
        landing_dir=tmp_path / "landing",
    )

    assert call.status == "failed"
    assert result.status == "failed"
    assert "no search results" in (result.error or "")


# ---------------------------------------------------------------------------
# Research executor — URLs explícitas en la query (seeds)
# ---------------------------------------------------------------------------

PAGINA12 = ("https://www.pagina12.com.ar/2026/09/12/"
            "las-big-tech-de-la-ia-dicen-estar-de-acuerdo-en-frenar-su-desarrollo/")
_ARTICLE = "Las big tech de la IA dicen estar de acuerdo en frenar su desarrollo. " * 40


def _fake_scraper(monkeypatch, seen: dict):
    """WebScraper fake: registra las URLs scrapeadas y devuelve un artículo."""
    from ipa.acquisition import web_scraper as ws_module

    class _Scrape:
        success = True
        error = None
        title = "Las big tech y la IA"
        date = None
        canonical_url = None
        content_hash = ""
        metadata = {"engine": "requests"}

        def __init__(self):
            self.text = _ARTICLE

    class _Scraper:
        def __init__(self, **kwargs):
            pass

        def extract_article(self, url, days_back=0):
            seen.setdefault("urls", []).append(url)
            return _Scrape()

        def save_article(self, scrape_result):
            pass

    monkeypatch.setattr(ws_module, "WebScraper", _Scraper)


def _run(memory, tmp_path, query, **kw):
    ctx = ToolContext(memory=memory)
    identity = load_identity()
    sid = memory.open_session(interface="cli", role="general",
                              identity_hash=identity.identity_hash)
    ep = memory.record_episode(sid, turn_role="user", content=query,
                               identity_hash=identity.identity_hash)
    return execute_research(
        query, ctx, session_id=sid, episode_id=ep.episode_id,
        max_urls=kw.pop("max_urls", 3), max_seconds=kw.pop("max_seconds", 10),
        landing_dir=tmp_path / "landing", **kw,
    )


def test_seed_url_is_scraped_directly_without_search_results(memory, tmp_path, monkeypatch):
    """Una URL en la query se scrapea directo (fuente explícita) aunque la
    búsqueda no devuelva nada: saltea el snippet stage, no el juicio."""
    from ipa.agent import research_executor as re_module

    seen: dict = {}

    def mock_search_web(query, **kwargs):
        seen["query"] = query
        return SearchSummary(query=query, results=[])

    monkeypatch.setattr(re_module, "search_web", mock_search_web)
    _fake_scraper(monkeypatch, seen)

    call, result, research = _run(
        memory, tmp_path,
        f"las big tech de la ia acuerdan frenar su desarrollo {PAGINA12}",
    )

    assert call.status == "completed", result.error
    assert seen["urls"] == [PAGINA12]
    assert PAGINA12 not in seen["query"]  # la búsqueda fue sobre el remanente
    stages = {j["stage"] for j in result.result["judgments"]}
    assert "seed" in stages
    assert len(research.web_sources) == 1
    assert research.web_sources[0].source_url == PAGINA12
    assert result.result["budget_used"]["seed_urls"] == 1


def test_seed_url_survives_search_failure(memory, tmp_path, monkeypatch):
    """Con seeds, un fallo del backend de búsqueda no invalida la corrida."""
    from ipa.agent import research_executor as re_module

    seen: dict = {}

    def mock_search_web(query, **kwargs):
        return SearchSummary(query=query, error="no search backend available")

    monkeypatch.setattr(re_module, "search_web", mock_search_web)
    _fake_scraper(monkeypatch, seen)

    call, result, research = _run(memory, tmp_path, f"nota sobre IA {PAGINA12}")

    assert call.status == "completed", result.error
    assert seen["urls"] == [PAGINA12]
    search_errors = [j for j in result.result["judgments"]
                     if j["stage"] == "search" and j["verdict"] == "error"]
    assert search_errors and "no search backend" in search_errors[0]["reason"]


def test_search_failure_without_seeds_still_fails(memory, tmp_path, monkeypatch):
    from ipa.agent import research_executor as re_module

    monkeypatch.setattr(
        re_module, "search_web",
        lambda query, **kw: SearchSummary(query=query, error="no search backend available"),
    )
    call, result, _ = _run(memory, tmp_path, "consulta sin urls")
    assert call.status == "failed"
    assert "web search failed" in (result.error or "")


def test_url_only_query_derives_search_from_slug(memory, tmp_path, monkeypatch):
    """Query solo-URL: la búsqueda complementaria usa el slug de la URL."""
    from ipa.agent import research_executor as re_module

    seen: dict = {}

    def mock_search_web(query, **kwargs):
        seen["query"] = query
        return SearchSummary(query=query, results=[])

    monkeypatch.setattr(re_module, "search_web", mock_search_web)
    _fake_scraper(monkeypatch, seen)

    call, result, _ = _run(memory, tmp_path, PAGINA12)

    assert call.status == "completed", result.error
    assert seen["query"] == (
        "las big tech de la ia dicen estar de acuerdo en frenar su desarrollo"
    )


def test_seed_urls_respect_budget(memory, tmp_path, monkeypatch):
    """Las seeds consumen el presupuesto de max_urls como cualquier candidato."""
    from ipa.agent import research_executor as re_module

    seen: dict = {}
    monkeypatch.setattr(
        re_module, "search_web",
        lambda query, **kw: SearchSummary(query=query, results=[]),
    )
    _fake_scraper(monkeypatch, seen)

    urls = " ".join(f"https://example.com/nota-{i}-sobre-ia" for i in range(5))
    call, result, research = _run(memory, tmp_path, f"ia {urls}", max_urls=2)

    assert call.status == "completed", result.error
    assert len(seen["urls"]) == 2
    assert result.result["budget_used"]["seed_urls"] == 2


# ---------------------------------------------------------------------------
# Research executor — end-to-end with real network (skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="requires network access; run manually with --run-network-tests")
def test_research_topic_end_to_end_with_network(memory, tmp_path):
    """Full end-to-end: web search → scrape → web_source records with provenance.

    This test hits DuckDuckGo and fetches a real page. It is skipped by
    default to avoid network dependencies in CI. Run manually:

        pytest tests/test_research_executor.py::test_research_topic_end_to_end_with_network --run-network-tests
    """
    ctx = ToolContext(memory=memory)
    identity = load_identity()
    sid = memory.open_session(interface="cli", role="general", identity_hash=identity.identity_hash)
    ep = memory.record_episode(sid, turn_role="user", content="research topic", identity_hash=identity.identity_hash)

    call, result, research = execute_research(
        "python asyncio tutorial", ctx,
        session_id=sid, episode_id=ep.episode_id,
        max_urls=2, max_seconds=30,
        landing_dir=tmp_path / "landing",
    )

    assert call.status == "completed", result.error
    assert result.status == "completed"
    assert len(research.web_sources) > 0

    # Every web_source must pass contract validation
    for ws in research.web_sources:
        errors = validate("WebSource", ws.to_contract())
        assert errors == [], f"web_source {ws.web_source_id} failed: {errors}"

    # ToolCall and ToolResult must pass contract validation
    assert validate("ToolCall", call.to_contract()) == []
    assert validate("ToolResult", result.to_contract()) == []

    # Budget must be respected
    assert research.budget_used["elapsed_seconds"] <= 30
    assert research.budget_used["urls_scraped"] <= 2

    # Every web_source must have a trust_label (PAT-003 boundary)
    for ws in research.web_sources:
        assert ws.trust_label in ("high", "medium", "low", "unverified")
        assert ws.content_hash.startswith("sha256:")
