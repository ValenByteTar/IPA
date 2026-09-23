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
# Research executor — sub_queries (barrido temático por facetas)
# ---------------------------------------------------------------------------

def test_sub_queries_widen_pool_and_judge_per_facet(memory, tmp_path, monkeypatch):
    """Cada sub-query corre su propia búsqueda y sus resultados entran al
    pool deduplicado; el prefilter/juicio se hace contra la query que los
    produjo — un candidato sin overlap con la query principal sobrevive si
    matchea su faceta."""
    from ipa.agent import research_executor as re_module
    from ipa.acquisition import web_scraper as ws_module

    seen: dict = {"queries": [], "urls": []}
    main_url = "https://a.example/jev-decisions"
    facet_url = "https://b.example/rlcd-paper"

    def mock_search_web(query, **kwargs):
        seen["queries"].append(query)
        if query == "jev typed decisions":
            return SearchSummary(query=query, results=[
                SearchResult(url=main_url, title="Jev typed decisions",
                             snippet="Jev returns typed decisions", domain="a.example"),
            ])
        # Resultado de la faceta: cero overlap con la query principal.
        return SearchSummary(query=query, results=[
            SearchResult(url=facet_url, title="RLCD paper",
                         snippet="reinforcement learning calibrated outputs",
                         domain="b.example"),
        ])

    class _Scrape:
        def __init__(self, text):
            self.success = True
            self.error = None
            self.title = "t"
            self.date = None
            self.canonical_url = None
            self.content_hash = ""
            self.metadata = {"engine": "requests"}
            self.text = text

    class _Scraper:
        def __init__(self, **kwargs):
            pass

        def extract_article(self, url, days_back=0):
            seen["urls"].append(url)
            if url == facet_url:
                return _Scrape("rlcd reinforcement calibrated decisions " * 30)
            return _Scrape("jev typed decisions model " * 30)

        def save_article(self, scrape_result):
            pass

    monkeypatch.setattr(re_module, "search_web", mock_search_web)
    monkeypatch.setattr(ws_module, "WebScraper", _Scraper)

    call, result, research = _run(
        memory, tmp_path, "jev typed decisions",
        sub_queries=["rlcd reinforcement calibrated"],
    )

    assert call.status == "completed", result.error
    assert seen["queries"] == ["jev typed decisions", "rlcd reinforcement calibrated"]
    assert research.search_results_count == 2
    # La URL de la faceta llegó al scrape aunque su snippet no matchea la
    # query principal (relevance 0.0 vs 'jev typed decisions').
    assert facet_url in seen["urls"]
    assert {ws.source_url for ws in research.web_sources} == {main_url, facet_url}
    assert result.result["budget_used"]["sub_queries"] == ["rlcd reinforcement calibrated"]


def test_sub_query_search_failure_does_not_fail_run(memory, tmp_path, monkeypatch):
    """Una sub-query cuyo backend falla se registra como error pero no
    invalida la corrida si la query principal trajo resultados."""
    from ipa.agent import research_executor as re_module

    seen: dict = {}

    def mock_search_web(query, **kwargs):
        if query == "las big tech ia":
            return SearchSummary(query=query, results=[
                SearchResult(url="https://a.example/bigtech", title="Big tech IA",
                             snippet="las big tech de la ia acuerdan frenar",
                             domain="a.example"),
            ])
        return SearchSummary(query=query, error="backend timeout")

    monkeypatch.setattr(re_module, "search_web", mock_search_web)
    _fake_scraper(monkeypatch, seen)

    call, result, research = _run(
        memory, tmp_path, "las big tech ia",
        sub_queries=["rlcd reinforcement calibrated"],
    )

    assert call.status == "completed", result.error
    search_errors = [j for j in result.result["judgments"]
                     if j["stage"] == "search" and j["verdict"] == "error"]
    assert search_errors and "backend timeout" in search_errors[0]["reason"]


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


# ---------------------------------------------------------------------------
# PM-004: dir de trabajo privado + embed acotado + budget post-scrape
# ---------------------------------------------------------------------------

class _FakeChunk:
    def __init__(self, chunk_id: str, document_id: str, text: str = "texto"):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.text = text


class _FakeStore:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def all_chunks(self):
        return list(self._chunks)

    def count_chunks(self):
        return len(self._chunks)


class _FakeLance:
    def __init__(self):
        self.added: list[str] = []

    def is_queryable(self):
        return False

    def add_chunks(self, chunks, vectors, sparse_weights=None):
        self.added.extend(c.chunk_id for c in chunks)

    def create_fts_index(self):
        pass


class _FakeEmbed:
    def __init__(self):
        self.calls: list[list[str]] = []
        self.gpu_released = False

    def embed_texts_hybrid(self, texts):
        self.calls.append(list(texts))
        return [[0.0, 0.0] for _ in texts], [{} for _ in texts]

    def release_gpu(self):
        self.gpu_released = True


class _FakeCtx:
    def __init__(self, store, lance, embed):
        self._store, self._lance, self._embed = store, lance, embed

    def document_store(self):
        return self._store

    def lance_index(self):
        return self._lance

    def embedding_adapter(self):
        return self._embed


def _embed_fixture(chunks):
    from ipa.agent.research_executor import _embed_new_chunks
    store, lance, embed = _FakeStore(chunks), _FakeLance(), _FakeEmbed()
    return _embed_new_chunks, _FakeCtx(store, lance, embed), lance, embed


def test_embed_only_this_runs_documents():
    """El embed de una corrida no toca chunks de otros documentos (PM-004:
    antes embebía todo chunk pendiente del corpus canónico)."""
    chunks = [
        _FakeChunk("c1", "doc:run-a"),
        _FakeChunk("c2", "doc:run-a"),
        _FakeChunk("c3", "doc:otro"),
    ]
    embed_fn, ctx, lance, embed = _embed_fixture(chunks)
    embedded = embed_fn(Path("corpus"), ctx, document_ids={"doc:run-a"})
    assert embedded == 2
    assert lance.added == ["c1", "c2"]
    assert len(embed.calls) == 1
    assert embed.calls[0] == ["texto", "texto"]


def test_embed_without_document_ids_keeps_backcompat():
    chunks = [_FakeChunk("c1", "doc:a"), _FakeChunk("c2", "doc:b")]
    embed_fn, ctx, lance, _ = _embed_fixture(chunks)
    assert embed_fn(Path("corpus"), ctx) == 2
    assert sorted(lance.added) == ["c1", "c2"]


def test_embed_respects_deadline():
    """Un deadline vencido corta el embed: la research no puede colgarse
    embebiendo (PM-004)."""
    import time as _time

    chunks = [_FakeChunk(f"c{i}", "doc:run") for i in range(10)]
    embed_fn, ctx, lance, embed = _embed_fixture(chunks)
    embedded = embed_fn(Path("corpus"), ctx, document_ids={"doc:run"},
                        deadline=_time.monotonic() - 1)
    assert embedded == 0
    assert lance.added == []
    assert embed.calls == []


def test_embed_batches_and_stops_at_deadline():
    """Batchea (no una sola llamada gigante) y corta entre batches."""
    import time as _time

    chunks = [_FakeChunk(f"c{i}", "doc:run") for i in range(5)]
    embed_fn, ctx, lance, embed = _embed_fixture(chunks)
    embedded = embed_fn(Path("corpus"), ctx, document_ids={"doc:run"},
                        deadline=_time.monotonic() + 5, batch_size=2)
    assert embedded == 5
    assert len(embed.calls) == 3  # 2 + 2 + 1
    assert lance.added == ["c0", "c1", "c2", "c3", "c4"]


def test_embed_escalates_to_gpu_bulk_over_threshold(monkeypatch):
    """Backlog >= umbral → el embed de research toma el lote GPU exclusivo
    (claim del job de mantenimiento + _start_bulk_gpu del drain) y lo cierra
    al terminar: el adapter queda liberado a CPU para no recargar BGE en CUDA
    sobre el chat restaurado."""
    import ipa.agentic.embedding_maintenance as maintenance
    import ipa.ingestion.fast_path_cli as fp_cli

    calls: dict[str, int] = {"start": 0, "update": 0, "finish": 0}

    monkeypatch.setattr(fp_cli, "EMBED_GPU_MIN_BACKLOG", 2)
    monkeypatch.setattr(fp_cli, "EMBED_GPU_BULK_ENABLED", True)
    monkeypatch.setattr(maintenance, "claim_job", lambda *a, **kw: True)
    released: list[str] = []
    monkeypatch.setattr(
        maintenance, "release_job", lambda owner="": released.append(owner))

    def _fake_start(embed, **kw):
        calls["start"] += 1
        return {"active": True, "vectorized_before": kw["vectorized"],
                "embedded_before": 0}

    monkeypatch.setattr(fp_cli, "_start_bulk_gpu", _fake_start)
    monkeypatch.setattr(
        fp_cli, "_update_bulk_gpu", lambda *a, **kw: calls.__setitem__("update", calls["update"] + 1))
    monkeypatch.setattr(
        fp_cli, "_finish_bulk_gpu", lambda *a, **kw: calls.__setitem__("finish", calls["finish"] + 1))

    chunks = [_FakeChunk(f"c{i}", "doc:run") for i in range(5)]
    embed_fn, ctx, lance, embed = _embed_fixture(chunks)
    embedded = embed_fn(Path("corpus"), ctx, document_ids={"doc:run"},
                        batch_size=2)
    assert embedded == 5
    assert calls["start"] == 1 and calls["finish"] == 1
    assert calls["update"] == 3  # una por batch
    assert embed.gpu_released
    assert released == ["research_embed"]


def test_embed_stays_cpu_below_gpu_threshold(monkeypatch):
    """Backlog chico → no se publica estado de mantenimiento ni se pide VRAM."""
    import ipa.agentic.embedding_maintenance as maintenance
    import ipa.ingestion.fast_path_cli as fp_cli

    monkeypatch.setattr(fp_cli, "EMBED_GPU_MIN_BACKLOG", 512)
    monkeypatch.setattr(fp_cli, "EMBED_GPU_BULK_ENABLED", True)
    claimed: list[bool] = []
    monkeypatch.setattr(
        maintenance, "claim_job",
        lambda *a, **kw: claimed.append(True) or True)

    def _no_start(*a, **kw):
        raise AssertionError("_start_bulk_gpu no debería llamarse")

    monkeypatch.setattr(fp_cli, "_start_bulk_gpu", _no_start)

    chunks = [_FakeChunk(f"c{i}", "doc:run") for i in range(3)]
    embed_fn, ctx, lance, embed = _embed_fixture(chunks)
    assert embed_fn(Path("corpus"), ctx, document_ids={"doc:run"}) == 3
    assert claimed == []
    assert not embed.gpu_released


def test_research_uses_private_work_dir_by_default(memory, tmp_path, monkeypatch):
    """Sin landing_dir explícito, la corrida trabaja en su propio dir bajo
    outputs/agent/research/ — nunca en Landing/web compartido (PM-004)."""
    from ipa.agent import research_executor as re_module

    monkeypatch.setattr(re_module, "RESEARCH_WORK_ROOT", tmp_path / "research")
    monkeypatch.setattr(re_module, "search_web", lambda q, **kw: SearchSummary(query=q, results=[]))
    _fake_scraper(monkeypatch, {})

    identity = load_identity()
    sid = memory.open_session(interface="cli", role="general",
                              identity_hash=identity.identity_hash)
    ep = memory.record_episode(sid, turn_role="user", content="x",
                               identity_hash=identity.identity_hash)
    ctx = ToolContext(memory=memory)
    call, result, research = execute_research(
        f"acuerdo de las big tech de la ia {PAGINA12}", ctx,
        session_id=sid, episode_id=ep.episode_id,
        max_urls=1, max_seconds=10,
    )
    assert call.status == "completed", result.error
    work_dir = Path(research.budget_used["work_dir"])
    assert work_dir.is_relative_to(tmp_path / "research"), work_dir
    assert "research" in work_dir.name or work_dir.parent.name == "research"


def test_research_records_ingest_budget(memory, tmp_path, monkeypatch):
    """El budget post-scrape y el estado del lock quedan auditados (PAT-004)."""
    from ipa.agent import research_executor as re_module

    monkeypatch.setattr(re_module, "search_web", lambda q, **kw: SearchSummary(query=q, results=[]))
    _fake_scraper(monkeypatch, {})

    identity = load_identity()
    sid = memory.open_session(interface="cli", role="general",
                              identity_hash=identity.identity_hash)
    ep = memory.record_episode(sid, turn_role="user", content="x",
                               identity_hash=identity.identity_hash)
    ctx = ToolContext(memory=memory)
    call, result, research = execute_research(
        f"acuerdo de las big tech de la ia {PAGINA12}", ctx,
        session_id=sid, episode_id=ep.episode_id,
        max_urls=1, max_seconds=10, landing_dir=tmp_path / "landing",
        max_ingest_seconds=123,
    )
    assert call.status == "completed", result.error
    ingest = research.budget_used["ingest"]
    assert ingest["max_ingest_seconds"] == 123
    assert "heavy_lock_acquired" in ingest
    assert "ingest_budget_exceeded" in ingest


# ---------------------------------------------------------------------------
# Staging corpus (DEC-003): research → staging → curación T1 → promoción
# ---------------------------------------------------------------------------

def _fake_scraper_saving(monkeypatch, text: str = _ARTICLE):
    """WebScraper fake que persiste el artefacto con el header ``Source:``
    igual que el real — el path queda en landing.db y la proveniencia se
    resuelve desde ese header (autoritativo)."""
    from ipa.acquisition import web_scraper as ws_module

    class _Scrape:
        success = True
        error = None
        title = "Las big tech y la IA"
        date = None
        canonical_url = None
        content_hash = ""
        metadata = {"engine": "requests"}
        url = ""

        def __init__(self):
            self.text = text

    class _Scraper:
        def __init__(self, **kwargs):
            self._out = Path(kwargs.get("output_dir", "."))
            self._out.mkdir(parents=True, exist_ok=True)

        def extract_article(self, url, days_back=0):
            s = _Scrape()
            s.url = url
            return s

        def save_article(self, scrape_result):
            n = len(list(self._out.glob("*.txt")))
            (self._out / f"doc{n}.txt").write_text(
                f"Source: {scrape_result.url}\n\n{scrape_result.text}",
                encoding="utf-8")

    monkeypatch.setattr(ws_module, "WebScraper", _Scraper)


def _isolate_heavy_lock(monkeypatch, tmp_path):
    """El heavy.lock real vive en outputs/agent — el test usa uno propio."""
    from ipa.agentic import heavy_lock
    monkeypatch.setattr(heavy_lock, "LOCK_PATH", tmp_path / "heavy.lock")
    monkeypatch.setattr(heavy_lock, "WAIT_PATH", tmp_path / "heavy.waiting")


def _stub_embed_and_drain(monkeypatch):
    """Sin modelo real: el embed no produce vectores → el executor lanza el
    drain residual; Popen queda registrado en vez de spawnear procesos."""
    import subprocess
    import ipa.agent.research_executor as re_module
    monkeypatch.setattr(re_module, "_embed_new_chunks", lambda *a, **kw: 0)
    spawned: list[list] = []

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            spawned.append([str(c) for c in cmd])

        def poll(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    return spawned


def test_research_ingests_to_staging_not_main(memory, tmp_path, monkeypatch):
    """Con staging_corpus_dir la ingesta aterriza en el staging — provenance
    registrada ahí — y main queda intacto: la entrada a main la decide la
    curación T1 + promotion_policy, no el run (DEC-003)."""
    import subprocess  # noqa: F401 — el stub de Popen lo parcha
    from ipa.agent import research_executor as re_module
    import ipa.agent.system_tools as st
    import ipa.agent.agent_tools as at
    from ipa.storage.document_store import DocumentStore

    monkeypatch.setattr(
        re_module, "search_web",
        lambda q, **kw: SearchSummary(query=q, results=[]))
    _fake_scraper_saving(monkeypatch)
    _isolate_heavy_lock(monkeypatch, tmp_path)
    main_corpus = tmp_path / "main_corpus"
    staging = tmp_path / "staging"
    # La comparación de duplicados exactos usa _main_corpus_dir — apuntarlo
    # al main del test (vacío), no al E12 real.
    monkeypatch.setattr(st, "_main_corpus_dir", lambda: main_corpus)
    monkeypatch.setattr(
        at, "_search_corpus", lambda args, ctx: ({"hits": []}, []))
    spawned = _stub_embed_and_drain(monkeypatch)

    ctx = ToolContext(memory=memory, corpus_dir=str(main_corpus))
    identity = load_identity()
    sid = memory.open_session(interface="cli", role="general",
                              identity_hash=identity.identity_hash)
    ep = memory.record_episode(sid, turn_role="user", content="x",
                               identity_hash=identity.identity_hash)
    call, result, research = execute_research(
        f"acuerdo de las big tech de la ia {PAGINA12}", ctx,
        session_id=sid, episode_id=ep.episode_id,
        max_urls=1, max_seconds=30, landing_dir=tmp_path / "landing",
        staging_corpus_dir=staging,
    )
    assert call.status == "completed", result.error
    assert research.ingested_count == 1

    # La ingesta aterrizó en staging; main no tiene el documento.
    store = DocumentStore(staging / "document_store.db")
    try:
        docs = store.all_document_texts()
        assert len(docs) == 1
        sources = store.all_sources()
        assert len(sources) == 1
        # Seed URL (pegada por el usuario) → user_provided (auto-promoción).
        src = next(iter(sources.values()))
        assert src["provenance"] == "user_provided"
        assert src["source_url"] == PAGINA12
    finally:
        store.close()
    # Main puede tener el archivo (record_ingest_metadata lo abre para la
    # comparación de duplicados) pero NUNCA el documento.
    if (main_corpus / "document_store.db").exists():
        _m = DocumentStore(main_corpus / "document_store.db")
        try:
            assert _m.all_document_texts() == {}
        finally:
            _m.close()

    # Retrieval: los hits del staging acompañan la respuesta (el material
    # nuevo no está en main todavía).
    staged_hits = [h for h in research.retrieval_hits
                   if h.get("retrieval_backend") == "staging_bm25"]
    assert staged_hits, "el retrieval debe ver el material recién staged"
    assert staged_hits[0]["provenance"] == "user_provided"

    # Backlog residual (embed stubbed → 0 vectores): se lanzó el drain.
    assert any("run_embed_drain" in " ".join(cmd) for cmd in spawned)

    assert research.budget_used["ingest_corpus"] == str(staging)


def test_on_progress_emits_phases_in_order(memory, tmp_path, monkeypatch):
    """on_progress reporta cada transición de fase — el dashboard muestra
    en qué va la investigación en vez de 'running' durante minutos."""
    from ipa.agent import research_executor as re_module
    import ipa.agent.system_tools as st
    import ipa.agent.agent_tools as at

    monkeypatch.setattr(
        re_module, "search_web",
        lambda q, **kw: SearchSummary(query=q, results=[]))
    _fake_scraper_saving(monkeypatch)
    _isolate_heavy_lock(monkeypatch, tmp_path)
    main_corpus = tmp_path / "main_corpus"
    staging = tmp_path / "staging"
    monkeypatch.setattr(st, "_main_corpus_dir", lambda: main_corpus)
    monkeypatch.setattr(
        at, "_search_corpus", lambda args, ctx: ({"hits": []}, []))
    _stub_embed_and_drain(monkeypatch)

    events: list[tuple[str, dict]] = []
    ctx = ToolContext(memory=memory, corpus_dir=str(main_corpus))
    identity = load_identity()
    sid = memory.open_session(interface="cli", role="general",
                              identity_hash=identity.identity_hash)
    ep = memory.record_episode(sid, turn_role="user", content="x",
                               identity_hash=identity.identity_hash)
    call, result, research = execute_research(
        f"acuerdo de las big tech de la ia {PAGINA12}", ctx,
        session_id=sid, episode_id=ep.episode_id,
        max_urls=1, max_seconds=30, landing_dir=tmp_path / "landing",
        staging_corpus_dir=staging,
        on_progress=lambda phase, detail: events.append((phase, detail)),
    )
    assert call.status == "completed", result.error

    phases = [p for p, _ in events]
    # Orden estricto del pipeline — cada fase emite al menos una vez.
    for expected in ("search", "judge", "scrape", "ingest", "embed", "retrieval"):
        assert expected in phases, f"fase '{expected}' no emitida: {phases}"
    assert phases.index("search") < phases.index("judge") < phases.index("scrape")
    assert phases.index("scrape") < phases.index("ingest") < phases.index("embed")
    # El emit por URL lleva conteos accionables.
    scrape_events = [d for p, d in events if p == "scrape"]
    assert scrape_events[-1]["accepted"] == 1
    assert scrape_events[-1]["total"] >= 1
    judge_events = [d for p, d in events if p == "judge"]
    assert judge_events[-1]["accepted"] >= 1

    # Un callback que explota no corta la investigación (best-effort).
    def _boom(phase, detail):
        raise RuntimeError("dashboard down")
    call2, result2, _ = execute_research(
        f"acuerdo de las big tech de la ia {PAGINA12}", ctx,
        session_id=sid, episode_id=ep.episode_id,
        max_urls=1, max_seconds=30, landing_dir=tmp_path / "landing2",
        staging_corpus_dir=staging,
        on_progress=_boom,
    )
    assert call2.status == "completed", result2.error


def test_research_discovered_url_gets_agent_research_provenance(
        memory, tmp_path, monkeypatch):
    """Una URL descubierta por búsqueda (no pegada por el usuario) registra
    provenance=agent_research → gate promotion_score >= 0.70 en T1."""
    from ipa.agent import research_executor as re_module
    import ipa.agent.system_tools as st
    import ipa.agent.agent_tools as at
    from ipa.agent.judge import Judgment
    from ipa.storage.document_store import DocumentStore

    discovered = "https://example.com/big-tech-acuerdo"
    monkeypatch.setattr(
        re_module, "search_web",
        lambda q, **kw: SearchSummary(query=q, results=[
            SearchResult(url=discovered, title="Las big tech y la IA",
                         snippet="acuerdo de las big tech de la ia",
                         domain="example.com")]))
    _fake_scraper_saving(monkeypatch)
    _isolate_heavy_lock(monkeypatch, tmp_path)
    main_corpus = tmp_path / "main_corpus"
    staging = tmp_path / "staging"
    monkeypatch.setattr(st, "_main_corpus_dir", lambda: main_corpus)
    monkeypatch.setattr(
        at, "_search_corpus", lambda args, ctx: ({"hits": []}, []))
    _stub_embed_and_drain(monkeypatch)

    class _AcceptAll:
        def judge_snippets(self, query, payloads):
            return [Judgment(verdict="accept", reason="ok", judge="fake",
                             confidence=1.0) for _ in payloads]

        def judge_content(self, query, title, text, age_days=None):
            return Judgment(verdict="accept", reason="ok", judge="fake",
                            confidence=1.0)

    ctx = ToolContext(memory=memory, corpus_dir=str(main_corpus))
    identity = load_identity()
    sid = memory.open_session(interface="cli", role="general",
                              identity_hash=identity.identity_hash)
    ep = memory.record_episode(sid, turn_role="user", content="x",
                               identity_hash=identity.identity_hash)
    call, result, research = execute_research(
        "acuerdo de las big tech de la ia", ctx,
        session_id=sid, episode_id=ep.episode_id,
        max_urls=1, max_seconds=30, landing_dir=tmp_path / "landing",
        staging_corpus_dir=staging, judge=_AcceptAll(),
    )
    assert call.status == "completed", result.error
    assert research.ingested_count == 1

    store = DocumentStore(staging / "document_store.db")
    try:
        sources = store.all_sources()
        assert len(sources) == 1
        src = next(iter(sources.values()))
        assert src["provenance"] == "agent_research"
        assert src["source_url"] == discovered
    finally:
        store.close()
