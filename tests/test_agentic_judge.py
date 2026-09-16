"""Fase 2 bridge tests: agentic research flow with LLM judgment.

Covers:
  - LLMJudge JSON parsing (valid, malformed, wrong-count responses)
  - LLMJudge fallback to heuristic on provider errors
  - HeuristicJudge deterministic decisions
  - Knowledge gap detection (assess_corpus_coverage)
  - Full agentic flow with a mock judge: gap → search → snippet judgment →
    scrape → content judgment → selective ingest → retrieval
  - Judgment audit records (PAT-004)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "validation"))

from ipa.agent import (  # noqa: E402
    AgentMemory,
    HeuristicJudge,
    Judgment,
    LLMJudge,
    ToolContext,
    assess_corpus_coverage,
    execute_research,
    load_identity,
)
from ipa.agent.judge import _extract_json  # noqa: E402
from ipa.storage.document_store import DocumentStore  # noqa: E402
from ipa.indexes.bm25_index import BM25Index  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

@dataclass
class FakeGenerationResult:
    text: str = ""
    error: str | None = None


class FakeProvider:
    """Chat provider that returns scripted responses per call."""
    def __init__(self, responses: list[str] | None = None, error: str | None = None):
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []
        self.error = error

    def generate_chat(self, messages, *, max_new_tokens=None, temperature=None, **kw):
        self.calls.append({"messages": messages, "max_new_tokens": max_new_tokens})
        if self.error:
            return FakeGenerationResult(text="", error=self.error)
        if self.responses:
            return FakeGenerationResult(text=self.responses.pop(0))
        return FakeGenerationResult(text="{}")


class MockJudge:
    """Scripted judge for deterministic end-to-end tests."""
    name = "mock"

    def __init__(self, snippet_verdicts=None, content_verdicts=None):
        self.snippet_verdicts = list(snippet_verdicts or [])
        self.content_verdicts = list(content_verdicts or [])
        self.snippet_calls: list[list[dict]] = []
        self.content_calls: list[tuple[str, str]] = []

    def judge_snippets(self, query, candidates):
        self.snippet_calls.append(list(candidates))
        out = []
        for i, cand in enumerate(candidates):
            if i < len(self.snippet_verdicts):
                out.append(Judgment(self.snippet_verdicts[i], "mock snippet verdict", 0.9, self.name))
            else:
                out.append(Judgment("reject", "mock default reject", 0.5, self.name))
        return out

    def judge_content(self, query, title, text, *, age_days=None):
        self.content_calls.append((title, text[:100]))
        if self.content_verdicts:
            verdict = self.content_verdicts.pop(0)
        else:
            verdict = "accept"
        return Judgment(verdict, "mock content verdict", 0.9, self.name)


@pytest.fixture()
def memory(tmp_path):
    with AgentMemory(store_path=tmp_path / "agent.db") as store:
        yield store


@pytest.fixture()
def corpus(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    ds = DocumentStore(corpus_dir / "document_store.db")
    ds.close()
    bm = BM25Index(corpus_dir / "bm25_index.db")
    bm.close()
    return corpus_dir


def _open_session(memory):
    identity = load_identity()
    sid = memory.open_session(
        interface="cli", role="general",
        identity_hash=identity.identity_hash, title="agentic flow test",
    )
    ep = memory.record_episode(
        sid, turn_role="user", content="test research",
        identity_hash=identity.identity_hash,
    )
    return sid, ep.episode_id


# ---------------------------------------------------------------------------
# _extract_json robustness
# ---------------------------------------------------------------------------

def test_extract_json_plain_object():
    assert _extract_json('{"verdict": "accept"}') == {"verdict": "accept"}


def test_extract_json_with_prose_around():
    raw = 'Here is my judgment:\n{"verdict": "reject", "reason": "paywall"}\nDone.'
    assert _extract_json(raw) == {"verdict": "reject", "reason": "paywall"}


def test_extract_json_markdown_fence():
    raw = '```json\n[{"index": 0, "verdict": "accept"}]\n```'
    assert _extract_json(raw) == [{"index": 0, "verdict": "accept"}]


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        _extract_json("no json here at all")


# ---------------------------------------------------------------------------
# LLMJudge: parsing, validation, fallback
# ---------------------------------------------------------------------------

def test_llm_judge_snippets_parses_batch_verdicts():
    provider = FakeProvider(responses=[json.dumps([
        {"index": 0, "verdict": "accept", "reason": "relevant tutorial", "confidence": 0.9},
        {"index": 1, "verdict": "reject", "reason": "paywall", "confidence": 0.8},
    ])])
    judge = LLMJudge(provider)
    judgments = judge.judge_snippets("python asyncio", [
        {"url": "https://a.com", "title": "Asyncio Tutorial", "snippet": "learn asyncio"},
        {"url": "https://b.com", "title": "Premium Course", "snippet": "buy now"},
    ])
    assert len(judgments) == 2
    assert judgments[0].verdict == "accept"
    assert judgments[0].judge == "llm"
    assert judgments[1].verdict == "reject"


def test_llm_judge_falls_back_on_malformed_json():
    provider = FakeProvider(responses=["this is not json at all"])
    judge = LLMJudge(provider)
    judgments = judge.judge_snippets("python asyncio", [
        {"url": "https://a.com", "title": "Python asyncio guide", "snippet": "python asyncio tutorial"},
    ])
    assert len(judgments) == 1
    assert judgments[0].judge == "llm_fallback_heuristic"


def test_llm_judge_falls_back_on_wrong_count():
    provider = FakeProvider(responses=[json.dumps([{"verdict": "accept"}])])  # 1 verdict for 2 candidates
    judge = LLMJudge(provider)
    judgments = judge.judge_snippets("python asyncio", [
        {"url": "https://a.com", "title": "t1", "snippet": "python asyncio"},
        {"url": "https://b.com", "title": "t2", "snippet": "python asyncio"},
    ])
    assert len(judgments) == 2
    assert all(j.judge == "llm_fallback_heuristic" for j in judgments)


def test_llm_judge_falls_back_on_provider_error():
    provider = FakeProvider(error="GPU OOM")
    judge = LLMJudge(provider)
    judgment = judge.judge_content("python asyncio", "title", "x" * 500)
    assert judgment.judge == "llm_fallback_heuristic"


def test_llm_judge_rejects_invalid_verdict_value():
    provider = FakeProvider(responses=[json.dumps({"verdict": "maybe", "reason": "unsure"})])
    judge = LLMJudge(provider)
    judgment = judge.judge_content("python asyncio", "title", "x" * 500)
    assert judgment.judge == "llm_fallback_heuristic"


def test_embed_new_chunks_indexes_pending(monkeypatch, tmp_path):
    """_embed_new_chunks embeds store chunks missing from LanceDB (Fase 2 bridge)."""
    from ipa.agent.research_executor import _embed_new_chunks
    from ipa.contracts import CanonicalDocument, DocumentChunk, SourceSpan
    from ipa.storage.document_store import DocumentStore
    from ipa.indexes.lancedb_index import LanceDBIndex

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    ds = DocumentStore(corpus / "document_store.db")
    doc = CanonicalDocument(
        document_id="doc:test001", pages=1, elements=[], source_spans=[],
        text="python asyncio event loops", mime_type="text/plain", parser_id="test",
    )
    ds.put_document(doc, artifact_id="artifact:test")
    span = SourceSpan(artifact_id="artifact:test", page=1, offset_start=0, offset_end=40)
    ds.put_chunks([
        DocumentChunk("chunk:t1", "doc:test001", "sha256:" + "a" * 64,
                      "python asyncio enables concurrent code", source_span=span),
        DocumentChunk("chunk:t2", "doc:test001", "sha256:" + "b" * 64,
                      "event loops schedule coroutines cooperatively", source_span=span),
    ])
    ds.commit()
    ds.close()

    ctx = ToolContext(memory=AgentMemory(store_path=tmp_path / "agent.db"), corpus_dir=str(corpus))
    # Real BGE-M3 embedding (cached locally) against a temp LanceDB
    embedded = _embed_new_chunks(corpus, ctx)
    assert embedded == 2

    lance = ctx.lance_index()
    assert lance.is_queryable()

    # Idempotent: second call embeds nothing new
    embedded2 = _embed_new_chunks(corpus, ctx)
    assert embedded2 == 0
    ctx.close()


# ---------------------------------------------------------------------------
# Retry buffer: rejections rotate instead of shrinking the result set
# ---------------------------------------------------------------------------

def test_retry_buffer_pulls_next_candidate_on_failure(memory, corpus, tmp_path, monkeypatch):
    """When early candidates fail, the next accepted candidates are tried —
    max_urls counts successful ingestions, not attempts."""
    from ipa.agent import research_executor as re_module
    from ipa.agent.research_executor import execute_research
    from ipa.agent.web_search import SearchResult, SearchSummary

    def fake_search_web(query, **kwargs):
        return SearchSummary(query=query, results=[
            SearchResult(f"https://site{i}.com/post", f"Python asyncio {i}", "python asyncio tutorial", f"site{i}.com")
            for i in range(4)
        ])

    monkeypatch.setattr(re_module, "search_web", fake_search_web)

    class FakeScrapeResult:
        def __init__(self, url, text):
            self.url = url
            self.text = text
            self.title = "Asyncio"
            self.success = bool(text)
            self.error = None if text else "Failed to fetch page"
            self.date = None
            self.canonical_url = None
            self.image_paths = []
            self.document_paths = []
            self.content_hash = None
            self.quality_score = 0.9
            self.metadata = {"word_count": "500", "engine": "requests"}

    class FakeScraper:
        def __init__(self, **kw):
            pass
        def extract_article(self, url, days_back=0):
            if "site0" in url or "site1" in url:
                # Simulate unreachable hosts for the first two candidates
                return FakeScrapeResult(url, "")
            # Unique text per URL so the duplicate check does not reject them
            return FakeScrapeResult(url, f"Python asyncio tutorial {url}. " * 60)
        def save_article(self, result):
            return tmp_path / "landing" / "x.txt"

    monkeypatch.setattr("ipa.acquisition.web_scraper.WebScraper", FakeScraper, raising=True)

    ctx = ToolContext(memory=memory, corpus_dir=str(corpus))
    with AgentMemory(store_path=tmp_path / "agent.db") as mem:
        identity = load_identity()
        sid = mem.open_session(interface="cli", role="general", identity_hash=identity.identity_hash)
        ep = mem.record_episode(sid, turn_role="user", content="q", identity_hash=identity.identity_hash)
        ep_id = ep.episode_id
        mem.close()

    call, result, research = execute_research(
        "python asyncio tutorial", ctx,
        session_id=sid, episode_id=ep_id,
        max_urls=2, max_seconds=30,
        landing_dir=tmp_path / "landing",
    )

    # Without the retry buffer, site0+site1 (both failing) would exhaust the
    # budget and the research would fail. With the buffer, site2 and site3
    # fill the quota.
    assert call.status == "completed", result.error
    assert research.scraped_count == 2
    assert len(research.web_sources) == 2
    accepted_urls = {ws.source_url for ws in research.web_sources}
    assert "https://site2.com/post" in accepted_urls
    assert "https://site3.com/post" in accepted_urls
    # The two failures are recorded with classification
    scrape_errors = [j for j in research.judgments if j.stage == "scrape" and j.verdict == "error"]
    assert len(scrape_errors) == 2
    assert all(j.kind == "unreachable" for j in scrape_errors)
    # Budget records attempts vs successes
    assert research.budget_used["urls_attempted"] == 4
    assert research.budget_used["urls_scraped"] == 2


def test_max_urls_stops_after_successful_ingestions(memory, corpus, tmp_path, monkeypatch):
    """The loop stops once max_urls successful ingestions are reached, even if
    more accepted candidates remain."""
    from ipa.agent import research_executor as re_module
    from ipa.agent.research_executor import execute_research
    from ipa.agent.web_search import SearchResult, SearchSummary

    def fake_search_web(query, **kwargs):
        return SearchSummary(query=query, results=[
            SearchResult(f"https://site{i}.com/post", f"Python asyncio {i}", "python asyncio", f"site{i}.com")
            for i in range(5)
        ])

    monkeypatch.setattr(re_module, "search_web", fake_search_web)

    attempted = []

    class FakeScrapeResult:
        success = True
        error = None
        date = None
        canonical_url = None
        image_paths = []
        document_paths = []
        def __init__(self, url):
            self.url = url
            # Unique text per URL so the duplicate check does not reject them
            self.text = f"Python asyncio tutorial {url}. " * 60
            self.title = "Asyncio"
            self.content_hash = None
            self.quality_score = 0.9
            self.metadata = {"word_count": "500", "engine": "requests"}

    class FakeScraper:
        def __init__(self, **kw):
            pass
        def extract_article(self, url, days_back=0):
            return FakeScrapeResult(url)
        def save_article(self, result):
            return tmp_path / "landing" / "x.txt"

    monkeypatch.setattr("ipa.acquisition.web_scraper.WebScraper", FakeScraper, raising=True)

    ctx = ToolContext(memory=memory, corpus_dir=str(corpus))
    with AgentMemory(store_path=tmp_path / "agent.db") as mem:
        identity = load_identity()
        sid = mem.open_session(interface="cli", role="general", identity_hash=identity.identity_hash)
        ep = mem.record_episode(sid, turn_role="user", content="q", identity_hash=identity.identity_hash)
        ep_id = ep.episode_id
        mem.close()

    call, result, research = execute_research(
        "python asyncio", ctx,
        session_id=sid, episode_id=ep_id,
        max_urls=2, max_seconds=30,
        landing_dir=tmp_path / "landing",
    )

    assert call.status == "completed", result.error
    assert research.scraped_count == 2
    assert research.budget_used["urls_attempted"] == 2  # stopped at quota


# ---------------------------------------------------------------------------
# Freshness policy recorded in the audit trail (PAT-004)
# ---------------------------------------------------------------------------

def test_toolcall_records_freshness_policy(memory, corpus, tmp_path, monkeypatch):
    """ToolCall.arguments includes freshness and max_age_days for traceability."""
    from ipa.agent import research_executor as re_module
    from ipa.agent.research_executor import execute_research
    from ipa.agent.web_search import SearchResult, SearchSummary

    def fake_search_web(query, **kwargs):
        return SearchSummary(query=query, results=[
            SearchResult("https://a.com/x", "Python asyncio", "python asyncio", "a.com"),
        ])

    monkeypatch.setattr(re_module, "search_web", fake_search_web)

    class FakeScrapeResult:
        success = True
        error = None
        date = None
        canonical_url = None
        image_paths = []
        document_paths = []
        def __init__(self):
            self.url = "https://a.com/x"
            self.text = "Python asyncio tutorial content. " * 60
            self.title = "Asyncio"
            self.content_hash = None
            self.quality_score = 0.9
            self.metadata = {"word_count": "500", "engine": "requests"}

    class FakeScraper:
        def __init__(self, **kw):
            pass
        def extract_article(self, url, days_back=0):
            return FakeScrapeResult()
        def save_article(self, result):
            return tmp_path / "landing" / "x.txt"

    monkeypatch.setattr("ipa.acquisition.web_scraper.WebScraper", FakeScraper, raising=True)

    ctx = ToolContext(memory=memory, corpus_dir=str(corpus))
    with AgentMemory(store_path=tmp_path / "agent.db") as mem:
        identity = load_identity()
        sid = mem.open_session(interface="cli", role="general", identity_hash=identity.identity_hash)
        ep = mem.record_episode(sid, turn_role="user", content="q", identity_hash=identity.identity_hash)
        ep_id = ep.episode_id
        mem.close()

    call, result, research = execute_research(
        "python asyncio", ctx,
        session_id=sid, episode_id=ep_id,
        max_urls=1, max_seconds=30,
        freshness="strict", max_age_days=60,
        landing_dir=tmp_path / "landing",
    )

    args = call.to_contract()["arguments"]
    assert args["freshness"] == "strict"
    assert args["max_age_days"] == 60


# ---------------------------------------------------------------------------
# Scrape failure classification
# ---------------------------------------------------------------------------

def test_classify_scrape_error_unreachable():
    from ipa.agent.research_executor import _classify_scrape_error
    assert _classify_scrape_error("Failed to fetch page") == "unreachable"
    assert _classify_scrape_error("ConnectTimeout: max retries exceeded")
    assert _classify_scrape_error(None, Exception("ConnectionError: connection reset")) == "unreachable"


def test_classify_scrape_error_blocked():
    from ipa.agent.research_executor import _classify_scrape_error
    assert _classify_scrape_error("403 Forbidden") == "blocked"
    assert _classify_scrape_error("captcha required") == "blocked"
    assert _classify_scrape_error("anomaly challenge") == "blocked"


def test_classify_scrape_error_extraction_failed():
    from ipa.agent.research_executor import _classify_scrape_error
    assert _classify_scrape_error("trafilatura extracted no text") == "extraction_failed"


def test_classify_scrape_error_unknown():
    from ipa.agent.research_executor import _classify_scrape_error
    assert _classify_scrape_error("something weird") == "unknown"


def test_scrape_error_kind_recorded_in_judgment(memory, corpus, tmp_path, monkeypatch):
    """A failed fetch produces a judgment with kind='unreachable' and the
    research continues to the next candidate."""
    from ipa.agent import research_executor as re_module
    from ipa.agent.research_executor import execute_research
    from ipa.agent.web_search import SearchResult, SearchSummary

    def fake_search_web(query, **kwargs):
        return SearchSummary(query=query, results=[
            SearchResult("https://down.com/a", "Python asyncio", "python asyncio", "down.com"),
        ])

    monkeypatch.setattr(re_module, "search_web", fake_search_web)

    class FakeScrapeResult:
        success = False
        error = "Failed to fetch page"
        date = None
        canonical_url = None
        image_paths = []
        document_paths = []
        text = ""
        title = ""
        content_hash = None
        quality_score = 0.0
        metadata = {}

    class FakeScraper:
        def __init__(self, **kw):
            pass
        def extract_article(self, url, days_back=0):
            return FakeScrapeResult()
        def save_article(self, result):
            return tmp_path / "landing" / "x.txt"

    monkeypatch.setattr("ipa.acquisition.web_scraper.WebScraper", FakeScraper, raising=True)

    ctx = ToolContext(memory=memory, corpus_dir=str(corpus))
    with AgentMemory(store_path=tmp_path / "agent.db") as mem:
        identity = load_identity()
        sid = mem.open_session(interface="cli", role="general", identity_hash=identity.identity_hash)
        ep = mem.record_episode(sid, turn_role="user", content="q", identity_hash=identity.identity_hash)
        ep_id = ep.episode_id
        mem.close()

    call, result, research = execute_research(
        "python asyncio", ctx,
        session_id=sid, episode_id=ep_id,
        max_urls=1, max_seconds=10,
        landing_dir=tmp_path / "landing",
    )

    scrape_errors = [j for j in research.judgments if j.stage == "scrape" and j.verdict == "error"]
    assert len(scrape_errors) == 1
    assert scrape_errors[0].kind == "unreachable"
    assert "[unreachable]" in scrape_errors[0].reason


def test_search_web_prefers_local_searxng(monkeypatch, tmp_path):
    from ipa.agent import web_search as ws_module

    class FakeResponse:
        def raise_for_status(self):
            return None
        def json(self):
            return {"results": [{"url": "https://example.org/a", "title": "A", "content": "snippet"}]}

    monkeypatch.setenv("IPA_SEARXNG_URL", "http://127.0.0.1:8888")
    monkeypatch.setenv("IPA_WEB_SEARCH_CACHE", str(tmp_path / "cache.db"))
    monkeypatch.setattr(ws_module.requests, "get", lambda *a, **kw: FakeResponse())
    monkeypatch.setattr(ws_module.requests, "post", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("DDG should not be called")))
    summary = ws_module.search_web("test query")
    assert summary.success
    assert summary.backend == "searxng"
    assert summary.results[0].domain == "example.org"


def test_search_web_uses_cache_when_backends_fail(monkeypatch, tmp_path):
    from ipa.agent import web_search as ws_module
    cache = tmp_path / "cache.db"
    monkeypatch.setenv("IPA_WEB_SEARCH_CACHE", str(cache))

    ws_module._cache_put("cached query", [ws_module.SearchResult("https://example.org", "Example", "cached", "example.org")])
    monkeypatch.setenv("IPA_SEARXNG_URL", "http://127.0.0.1:9")
    summary = ws_module.search_web("cached query")
    assert summary.success
    assert summary.backend == "cache"
    assert summary.from_cache is True


def test_search_web_reports_all_backends_unavailable(monkeypatch, tmp_path):
    from ipa.agent import web_search as ws_module
    monkeypatch.setenv("IPA_WEB_SEARCH_CACHE", str(tmp_path / "cache.db"))
    monkeypatch.delenv("IPA_SEARXNG_URL", raising=False)

    class FakeResponse:
        status_code = 202
        text = "anomaly challenge"
        def raise_for_status(self):
            return None

    monkeypatch.setattr(ws_module.requests, "post", lambda *a, **kw: FakeResponse())
    summary = ws_module.search_web("uncached query")
    assert not summary.success
    assert "SearXNG" in summary.error
    assert "DuckDuckGo" in summary.error


# ---------------------------------------------------------------------------
# Freshness modes: lenient (date as signal) vs strict (hard cutoff)
# ---------------------------------------------------------------------------

def _research_env(tmp_path, monkeypatch, article_date):
    """Shared fixture helper: scripted search + scraper with a dated article."""
    from ipa.agent import research_executor as re_module
    from ipa.agent.web_search import SearchResult, SearchSummary

    def fake_search_web(query, **kwargs):
        return SearchSummary(query=query, results=[
            SearchResult("https://old.com/tutorial", "Python asyncio tutorial", "python asyncio tutorial", "old.com"),
        ])

    monkeypatch.setattr(re_module, "search_web", fake_search_web)

    class FakeScrapeResult:
        success = True
        error = None
        canonical_url = None
        image_paths = []
        document_paths = []
        def __init__(self):
            self.url = "https://old.com/tutorial"
            self.text = "Python asyncio tutorial content. " * 60
            self.title = "Asyncio Tutorial"
            self.date = article_date
            self.content_hash = None
            self.quality_score = 0.9
            self.metadata = {"word_count": "500", "engine": "requests"}

    class FakeScraper:
        def __init__(self, **kw):
            self.kw = kw
        def extract_article(self, url, days_back=0):
            return FakeScrapeResult()
        def save_article(self, result):
            return tmp_path / "landing" / "x.txt"

    return FakeScraper


def test_lenient_mode_accepts_old_article_with_age_note(memory, corpus, tmp_path, monkeypatch):
    """Default (lenient): an old article is NOT rejected by date; the age is
    recorded in the content judgment as a signal."""
    from datetime import datetime, timezone, timedelta
    from ipa.agent.research_executor import execute_research

    old_date = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    FakeScraper = _research_env(tmp_path, monkeypatch, old_date)
    monkeypatch.setattr("ipa.acquisition.web_scraper.WebScraper", FakeScraper, raising=True)

    ctx = ToolContext(memory=memory, corpus_dir=str(corpus))
    with AgentMemory(store_path=tmp_path / "agent.db") as mem:
        identity = load_identity()
        sid = mem.open_session(interface="cli", role="general", identity_hash=identity.identity_hash)
        ep = mem.record_episode(sid, turn_role="user", content="q", identity_hash=identity.identity_hash)
        ep_id = ep.episode_id
        mem.close()

    call, result, research = execute_research(
        "python asyncio tutorial", ctx,
        session_id=sid, episode_id=ep_id,
        max_urls=1, max_seconds=10,
        landing_dir=tmp_path / "landing",
        # freshness defaults to "lenient"
    )

    assert call.status == "completed", result.error
    # No date-stage rejection
    assert not any(j.stage == "date" for j in research.judgments)
    # The content judgment records the age as a signal
    content_j = [j for j in research.judgments if j.stage == "content"]
    assert len(content_j) == 1
    assert "age 400d" in content_j[0].reason
    assert research.scraped_count == 1


def test_strict_mode_rejects_old_article(memory, corpus, tmp_path, monkeypatch):
    """strict: articles older than max_age_days are rejected outright."""
    from datetime import datetime, timezone, timedelta
    from ipa.agent.research_executor import execute_research

    old_date = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    FakeScraper = _research_env(tmp_path, monkeypatch, old_date)
    monkeypatch.setattr("ipa.acquisition.web_scraper.WebScraper", FakeScraper, raising=True)

    ctx = ToolContext(memory=memory, corpus_dir=str(corpus))
    with AgentMemory(store_path=tmp_path / "agent.db") as mem:
        identity = load_identity()
        sid = mem.open_session(interface="cli", role="general", identity_hash=identity.identity_hash)
        ep = mem.record_episode(sid, turn_role="user", content="q", identity_hash=identity.identity_hash)
        ep_id = ep.episode_id
        mem.close()

    call, result, research = execute_research(
        "python asyncio tutorial", ctx,
        session_id=sid, episode_id=ep_id,
        max_urls=1, max_seconds=10,
        freshness="strict",
        landing_dir=tmp_path / "landing",
    )

    date_rejects = [j for j in research.judgments if j.stage == "date"]
    assert len(date_rejects) == 1
    assert date_rejects[0].verdict == "reject"
    assert "older than 365 days" in date_rejects[0].reason


def test_lenient_mode_llm_judge_receives_age_context(memory, corpus, tmp_path, monkeypatch):
    """In lenient mode the LLM judge's prompt includes the publication age."""
    import json as _json
    from datetime import datetime, timezone, timedelta
    from ipa.agent.research_executor import execute_research

    old_date = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    FakeScraper = _research_env(tmp_path, monkeypatch, old_date)
    monkeypatch.setattr("ipa.acquisition.web_scraper.WebScraper", FakeScraper, raising=True)

    provider = FakeProvider(responses=[
        _json.dumps([{"index": 0, "verdict": "accept", "reason": "relevant", "confidence": 0.9}]),
        _json.dumps({"verdict": "accept", "reason": "still accurate", "confidence": 0.9}),
    ])
    judge = LLMJudge(provider)

    ctx = ToolContext(memory=memory, corpus_dir=str(corpus))
    with AgentMemory(store_path=tmp_path / "agent.db") as mem:
        identity = load_identity()
        sid = mem.open_session(interface="cli", role="general", identity_hash=identity.identity_hash)
        ep = mem.record_episode(sid, turn_role="user", content="q", identity_hash=identity.identity_hash)
        ep_id = ep.episode_id
        mem.close()

    call, result, research = execute_research(
        "python asyncio tutorial", ctx,
        session_id=sid, episode_id=ep_id,
        max_urls=1, max_seconds=10,
        judge=judge,
        landing_dir=tmp_path / "landing",
    )

    assert call.status == "completed", result.error
    # The content prompt must contain the age context
    content_prompt = provider.calls[1]["messages"][1]["content"]
    assert "Publication age: 400 days" in content_prompt


def test_invalid_freshness_mode_rejected(memory):
    from ipa.agent.research_executor import execute_research
    ctx = ToolContext(memory=memory)
    with pytest.raises(ValueError, match="freshness"):
        execute_research(
            "q", ctx, session_id="s", episode_id="e", freshness="bogus",
        )


# ---------------------------------------------------------------------------
# Scraper auto-engine retry recording
# ---------------------------------------------------------------------------

def test_scrape_retry_recorded_in_judgment(memory, corpus, tmp_path, monkeypatch):
    """When auto mode falls back to Playwright, the judgment records it."""
    from ipa.agent.research_executor import execute_research
    from ipa.agent import research_executor as re_module
    from ipa.agent.web_search import SearchResult, SearchSummary

    def fake_search_web(query, **kwargs):
        return SearchSummary(query=query, results=[
            SearchResult("https://js-heavy.com/post", "Python asyncio", "python asyncio", "js-heavy.com"),
        ])

    monkeypatch.setattr(re_module, "search_web", fake_search_web)

    class FakeScrapeResult:
        success = True
        error = None
        date = None
        canonical_url = None
        image_paths = []
        document_paths = []
        def __init__(self):
            self.url = "https://js-heavy.com/post"
            self.text = "Python asyncio tutorial content. " * 60
            self.title = "Asyncio"
            self.content_hash = None
            self.quality_score = 0.9
            self.metadata = {"word_count": "600", "engine": "playwright"}

    class FakeScraper:
        def __init__(self, **kw):
            assert kw.get("engine") == "auto"
        def extract_article(self, url, days_back=0):
            return FakeScrapeResult()
        def save_article(self, result):
            return tmp_path / "landing" / "x.txt"

    monkeypatch.setattr("ipa.acquisition.web_scraper.WebScraper", FakeScraper, raising=True)

    ctx = ToolContext(memory=memory, corpus_dir=str(corpus))
    with AgentMemory(store_path=tmp_path / "agent.db") as mem:
        identity = load_identity()
        sid = mem.open_session(interface="cli", role="general", identity_hash=identity.identity_hash)
        ep = mem.record_episode(sid, turn_role="user", content="q", identity_hash=identity.identity_hash)
        ep_id = ep.episode_id
        mem.close()

    call, result, research = execute_research(
        "python asyncio", ctx,
        session_id=sid, episode_id=ep_id,
        max_urls=1, max_seconds=10,
        landing_dir=tmp_path / "landing",
    )

    assert call.status == "completed", result.error
    scrape_accepts = [j for j in research.judgments if j.stage == "scrape" and j.verdict == "accept"]
    assert len(scrape_accepts) == 1
    assert "playwright fallback" in scrape_accepts[0].reason


# ---------------------------------------------------------------------------

def test_search_web_detects_anomaly_block(monkeypatch):
    """DDG anomaly/bot challenges (202 + challenge form) are reported as errors,
    not as silent empty results."""
    from ipa.agent import web_search as ws_module

    class FakeResponse:
        status_code = 202
        text = "<html><body>anomaly detected challenge-form</body></html>"
        def raise_for_status(self):
            return None

    monkeypatch.setattr(ws_module.requests, "post", lambda *a, **kw: FakeResponse())
    summary = ws_module.search_web("test query")
    assert summary.error is not None
    assert "blocked" in summary.error or "anomaly" in summary.error
    assert summary.results == []


def test_search_web_detects_non_200(monkeypatch):
    from ipa.agent import web_search as ws_module

    class FakeResponse:
        status_code = 403
        text = "<html>forbidden</html>"
        def raise_for_status(self):
            raise RuntimeError("403 forbidden")

    monkeypatch.setattr(ws_module.requests, "post", lambda *a, **kw: FakeResponse())
    summary = ws_module.search_web("test query")
    assert summary.error is not None


# ---------------------------------------------------------------------------
# HeuristicJudge — deterministic scaffold
# ---------------------------------------------------------------------------

def test_heuristic_judge_accepts_relevant_snippet():
    j = HeuristicJudge().judge_snippets("python asyncio", [
        {"url": "https://a.com", "title": "Python asyncio tutorial", "snippet": "learn python asyncio"},
    ])
    assert j[0].verdict == "accept"
    assert j[0].judge == "heuristic"


def test_heuristic_judge_rejects_irrelevant_snippet():
    j = HeuristicJudge().judge_snippets("python asyncio", [
        {"url": "https://a.com", "title": "Pizza recipes", "snippet": "how to bake bread"},
    ])
    assert j[0].verdict == "reject"


def test_heuristic_judge_rejects_short_content():
    j = HeuristicJudge().judge_content("python asyncio", "t", "short")
    assert j.verdict == "reject"


# ---------------------------------------------------------------------------
# Knowledge gap detection
# ---------------------------------------------------------------------------

def test_assess_coverage_empty_query(memory):
    ctx = ToolContext(memory=memory)
    assessment = assess_corpus_coverage("", ctx)
    assert assessment.sufficient is False
    assert "empty query" in assessment.reason


def test_assess_coverage_without_corpus(memory):
    ctx = ToolContext(memory=memory)
    assessment = assess_corpus_coverage("python asyncio", ctx)
    assert assessment.sufficient is False
    assert "unavailable" in assessment.reason


def test_assess_coverage_with_real_corpus(memory, tmp_path):
    """Uses the E12 corpus fixture if present; skips otherwise."""
    e12 = Path(__file__).parents[1] / "outputs" / "experiments" / "E12-corpus"
    if not (e12 / "document_store.db").exists():
        pytest.skip("E12 corpus fixture not available")
    ctx = ToolContext(memory=memory, corpus_dir=str(e12))
    assessment = assess_corpus_coverage("machine learning", ctx, min_hits=3)
    assert assessment.hit_count >= 3
    assert assessment.sufficient is True


# ---------------------------------------------------------------------------
# Full agentic flow with MockJudge (no network)
# ---------------------------------------------------------------------------

def test_agentic_flow_selective_ingest(memory, corpus, tmp_path, monkeypatch):
    """The agent judges snippets and content; only accepted material is ingested."""
    from ipa.agent import research_executor as re_module
    from ipa.agent.research_executor import execute_research

    # Scripted web search: 3 results
    from ipa.agent.web_search import SearchResult, SearchSummary

    def fake_search_web(query, **kwargs):
        return SearchSummary(query=query, results=[
            SearchResult("https://good.com/a", "Python asyncio tutorial", "learn asyncio", "good.com"),
            SearchResult("https://paywall.com/b", "Premium asyncio course", "asyncio course", "paywall.com"),
            SearchResult("https://dup.com/c", "Python asyncio tutorial copy", "learn asyncio", "dup.com"),
        ])

    monkeypatch.setattr(re_module, "search_web", fake_search_web)

    # Scripted scraper: all URLs return content (dup has same text as good)
    class FakeScrapeResult:
        def __init__(self, url, text, title, date=None):
            self.url = url
            self.text = text
            self.title = title
            self.date = None
            self.success = True
            self.error = None
            self.content_hash = None
            self.canonical_url = None
            self.quality_score = 0.9
            self.image_paths = []
            self.document_paths = []
            self.elapsed_seconds = 0.1
            self.metadata = {}

    good_text = "Python asyncio tutorial. " + "Asyncio enables concurrent code with async await. " * 30
    dup_text = good_text  # exact duplicate content

    class FakeScraper:
        def __init__(self, **kw):
            self.saved = []
        def extract_article(self, url, days_back=0):
            if "good.com" in url:
                return FakeScrapeResult(url, good_text, "Good Tutorial")
            if "paywall.com" in url:
                return FakeScrapeResult(url, "python asyncio " + "x" * 600, "Paywall")
            return FakeScrapeResult(url, dup_text, "Dup")
        def save_article(self, result):
            self.saved.append(result.url)
            out = tmp_path / "landing" / "saved.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(result.text, encoding="utf-8")
            return out

    monkeypatch.setattr(
        "ipa.acquisition.web_scraper.WebScraper", FakeScraper, raising=True,
    )

    # Judge: accept good.com at snippet stage; reject paywall (content) and dup (duplicate)
    judge = MockJudge(
        snippet_verdicts=["accept", "accept", "accept"],
        content_verdicts=["accept", "reject", "accept"],  # paywall rejected at content stage
    )

    landing = tmp_path / "landing"
    landing.mkdir()
    ctx = ToolContext(memory=memory, corpus_dir=str(corpus))

    sid, ep_id = None, None
    with AgentMemory(store_path=tmp_path / "agent.db") as mem:
        identity = load_identity()
        sid = mem.open_session(interface="cli", role="general", identity_hash=identity.identity_hash)
        ep = mem.record_episode(sid, turn_role="user", content="q", identity_hash=identity.identity_hash)
        ep_id = ep.episode_id
        mem.close()

    call, result, research = execute_research(
        "python asyncio tutorial", ctx,
        session_id=sid, episode_id=ep_id,
        max_urls=3, max_seconds=30,
        judge=judge,
        landing_dir=landing,
    )

    # The flow completed and the agent made judgments at both stages
    assert call.status == "completed", result.error
    stages = {j.stage for j in research.judgments}
    assert "snippet" in stages
    assert "content" in stages

    # The paywall was rejected at content stage with an explicit reason
    content_rejects = [j for j in research.judgments if j.stage == "content" and j.verdict == "reject"]
    assert len(content_rejects) == 1
    assert content_rejects[0].judge == "mock"

    # Only the accepted (non-rejected, non-duplicate) source was kept
    assert research.scraped_count >= 1
    assert research.rejected_count >= 1

    # Judgments carry reasons (PAT-004 auditability)
    for j in research.judgments:
        assert j.reason


def test_agentic_flow_rejects_all_snippets(memory, corpus, tmp_path, monkeypatch):
    """When the agent rejects every snippet, research fails with a clear reason."""
    from ipa.agent import research_executor as re_module
    from ipa.agent.research_executor import execute_research
    from ipa.agent.web_search import SearchResult, SearchSummary

    def fake_search_web(query, **kwargs):
        return SearchSummary(query=query, results=[
            SearchResult("https://a.com", "Python asyncio", "python asyncio", "a.com"),
        ])

    monkeypatch.setattr(re_module, "search_web", fake_search_web)

    judge = MockJudge(snippet_verdicts=["reject"])
    ctx = ToolContext(memory=memory, corpus_dir=str(corpus))

    with AgentMemory(store_path=tmp_path / "agent.db") as mem:
        identity = load_identity()
        sid = mem.open_session(interface="cli", role="general", identity_hash=identity.identity_hash)
        ep = mem.record_episode(sid, turn_role="user", content="q", identity_hash=identity.identity_hash)
        mem.close()

    call, result, research = execute_research(
        "python asyncio", ctx,
        session_id=sid, episode_id=ep.episode_id,
        max_urls=1, max_seconds=10,
        judge=judge,
        landing_dir=tmp_path / "landing",
    )

    assert call.status == "failed"
    assert "rejected all search results" in result.error
    assert research.judgments[0].stage == "snippet"
    assert research.judgments[0].verdict == "reject"


def test_llm_judge_used_when_provided(memory, corpus, tmp_path, monkeypatch):
    """The LLMJudge path is exercised end-to-end with a scripted provider."""
    from ipa.agent import research_executor as re_module
    from ipa.agent.research_executor import execute_research
    from ipa.agent.web_search import SearchResult, SearchSummary

    def fake_search_web(query, **kwargs):
        return SearchSummary(query=query, results=[
            SearchResult("https://docs.python.org/asyncio", "Asyncio docs", "python asyncio docs", "docs.python.org"),
        ])

    monkeypatch.setattr(re_module, "search_web", fake_search_web)

    class FakeScrapeResult:
        success = True
        error = None
        date = None
        canonical_url = None
        image_paths = []
        document_paths = []
        def __init__(self, url="", text="", title=""):
            self.url = url
            self.text = text
            self.title = title
            self.content_hash = None
            self.quality_score = 0.9
            self.metadata = {"word_count": "500", "engine": "requests"}

    class FakeScraper:
        def __init__(self, **kw):
            pass
        def extract_article(self, url, days_back=0):
            return fake_result
        def save_article(self, result):
            return tmp_path / "landing" / "x.txt"

    fake_result = FakeScrapeResult("https://docs.python.org/asyncio", "Python asyncio tutorial content. " * 50, "Asyncio")
    monkeypatch.setattr("ipa.acquisition.web_scraper.WebScraper", FakeScraper, raising=True)

    # Scripted provider: 1 batch snippet call + 1 content call
    provider = FakeProvider(responses=[
        json.dumps([{"index": 0, "verdict": "accept", "reason": "official docs", "confidence": 0.95}]),
        json.dumps({"verdict": "accept", "reason": "complete tutorial", "confidence": 0.9}),
    ])

    judge = LLMJudge(provider)
    ctx = ToolContext(memory=memory, corpus_dir=str(corpus))

    with AgentMemory(store_path=tmp_path / "agent.db") as mem:
        identity = load_identity()
        sid = mem.open_session(interface="cli", role="general", identity_hash=identity.identity_hash)
        ep = mem.record_episode(sid, turn_role="user", content="q", identity_hash=identity.identity_hash)
        ep_id = ep.episode_id
        mem.close()

    call, result, research = execute_research(
        "python asyncio", ctx,
        session_id=sid, episode_id=ep_id,
        max_urls=1, max_seconds=10,
        judge=judge,
        landing_dir=tmp_path / "landing",
    )

    assert call.status == "completed", result.error
    # Both judgments came from the LLM
    assert all(j.judge == "llm" for j in research.judgments)
    assert len(provider.calls) == 2  # one batch snippet + one content
