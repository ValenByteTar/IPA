"""research_topic executor — agentic research flow (Fase 1→2 bridge).

Flow (the agent reads and judges; tools are deterministic adapters):

  1. web search (DuckDuckGo HTML, no API key) → URLs with snippets
  2. agent reads snippets → semantic judgment of which URLs are worth
     scraping (LLMJudge; HeuristicJudge as cheap scaffold + fallback)
  3. scrape accepted URLs (existing E4 adapter)
  4. agent reads raw scraped text → judges each document:
     - accept  → save to Landing + ingest into corpus
     - reject  → discard with explicit reason (paywall, stub, duplicate, stale)
  5. FastPath ingestion of the accepted material only
  6. retrieval with citations over the corpus

Every judgment is recorded (url, stage, verdict, reason, judge) — PAT-004
budgets + traceability. Web material remains derived and labeled, never
canonical authority (PAT-003).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent_memory import _compact_stamp, content_hash
from .agent_tools import ToolContext, ToolCall, ToolResult, _result_hash, _now
from .judge import (
    CONTENT_QUALITY_THRESHOLD,
    SNIPPET_RELEVANCE_THRESHOLD,
    HeuristicJudge,
    Judgment,
    _content_quality,
    _snippet_relevance,
)
from .web_search import search_web, SearchSummary


@dataclass(frozen=True)
class WebSource:
    """Contract record: provenance for web-acquired material."""
    web_source_id: str
    source_url: str
    fetched_at: str
    content_hash: str
    trust_label: str
    fetch_method: str
    content_type: str
    byte_size: int
    title: str | None = None
    canonical_url: str | None = None
    license: str | None = None
    research_request_id: str | None = None
    artifact_id: str | None = None

    def to_contract(self) -> dict[str, Any]:
        return {
            "web_source_id": self.web_source_id,
            "source_url": self.source_url,
            "canonical_url": self.canonical_url,
            "fetched_at": self.fetched_at,
            "content_hash": self.content_hash,
            "title": self.title,
            "license": self.license,
            "trust_label": self.trust_label,
            "fetch_method": self.fetch_method,
            "content_type": self.content_type,
            "byte_size": self.byte_size,
            "research_request_id": self.research_request_id,
            "artifact_id": self.artifact_id,
        }


@dataclass(frozen=True)
class SourceJudgment:
    """Audit record: one judgment the agent made about one URL (PAT-004)."""
    url: str
    stage: str          # "snippet" | "content" | "date" | "duplicate" | "scrape"
    verdict: str        # "accept" | "reject" | "error"
    reason: str
    judge: str          # "llm" | "heuristic" | "llm_fallback_heuristic"
    confidence: float = 0.0
    kind: str = ""      # scrape failure class: "unreachable" | "blocked" | "extraction_failed" | ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "stage": self.stage,
            "verdict": self.verdict,
            "reason": self.reason,
            "judge": self.judge,
            "confidence": round(self.confidence, 3),
            "kind": self.kind,
        }


def _classify_scrape_error(error: str | None = None, exc: Exception | None = None) -> str:
    """Classify a scrape failure for actionable audit records.

    - "unreachable": network-level failure (transient — worth retrying later)
    - "blocked": active block (403/challenge/captcha — permanent)
    - "extraction_failed": page fetched but no text extracted
    - "unknown": anything else
    """
    combined = f"{error or ''} {exc or ''}".lower()
    if any(sig in combined for sig in (
        "failed to fetch", "connecttimeout", "connection", "timeout",
        "max retries", "getaddrinfo", "nameresolution", "ssl",
    )):
        return "unreachable"
    if any(sig in combined for sig in (
        "403", "forbidden", "captcha", "anomaly", "cloudflare", "access denied",
    )):
        return "blocked"
    if any(sig in combined for sig in ("no text", "extract")):
        return "extraction_failed"
    return "unknown"


@dataclass
class ResearchResult:
    """Structured output of a research_topic execution."""
    query: str
    web_sources: list[WebSource] = field(default_factory=list)
    judgments: list[SourceJudgment] = field(default_factory=list)
    search_results_count: int = 0
    scraped_count: int = 0
    rejected_count: int = 0
    rejection_reasons: list[str] = field(default_factory=list)
    ingested_count: int = 0
    embedded_count: int = 0
    retrieval_hits: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str | None = None
    budget_used: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None and len(self.web_sources) > 0


def _trust_label_for_domain(domain: str) -> str:
    """Heuristic trust classification based on domain."""
    high_trust = {
        "arxiv.org", "github.com", "docs.python.org", "developer.mozilla.org",
        "w3.org", "ietf.org", "nist.gov", "ieee.org", "acm.org",
    }
    medium_trust = {
        "wikipedia.org", "stackoverflow.com", "medium.com", "dev.to",
        "hackernews.com", "thehackernews.com",
    }
    domain_lower = domain.lower()
    for d in high_trust:
        if d in domain_lower:
            return "high"
    for d in medium_trust:
        if d in domain_lower:
            return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Deterministic scaffold helpers
# (snippet/content filters live in judge.py — the HeuristicJudge engine)
# ---------------------------------------------------------------------------

def _is_duplicate(text: str, accepted_texts: list[str], threshold: float = 0.8) -> bool:
    """Near-duplicate detection via token Jaccard against accepted content."""
    from .judge import _tokenize
    tokens = set(_tokenize(text))
    if not tokens:
        return False
    for prev in accepted_texts:
        prev_tokens = set(_tokenize(prev))
        if not prev_tokens:
            continue
        jaccard = len(tokens & prev_tokens) / len(tokens | prev_tokens)
        if jaccard >= threshold:
            return True
    return False


def _article_is_too_old(date_str: str | None, max_age_days: int) -> bool:
    """True if the article date is older than the cutoff (None/unparseable → False)."""
    if max_age_days <= 0 or not date_str:
        return False
    age = _article_age_days(date_str)
    return age is not None and age > max_age_days


def _article_age_days(date_str: str | None) -> int | None:
    """Article age in days, or None if the date is missing/unparseable."""
    if not date_str:
        return None
    from datetime import datetime, timezone
    try:
        article_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if article_date.tzinfo is None:
            article_date = article_date.replace(tzinfo=timezone.utc)
        return max(0, (datetime.now(timezone.utc) - article_date).days)
    except (ValueError, TypeError):
        return None


MAX_ARTICLE_AGE_DAYS = 365  # 0 = no age filter
DUPLICATE_JACCARD_THRESHOLD = 0.80


def _embed_new_chunks(corpus: Path, ctx: Any) -> int:
    """Embed corpus chunks that lack a vector and add them to LanceDB.

    Completes the ingestion pipeline: DocumentStore + BM25 (FastPath) →
    dense vectors (BGE-M3) → LanceDB. Returns the number of chunks embedded.
    Non-fatal: retrieval falls back to BM25 if this fails.
    """
    from ipa.storage.document_store import DocumentStore
    from ipa.indexes.lancedb_index import LanceDBIndex

    store = ctx.document_store()
    lance = ctx.lance_index()
    if store is None or lance is None:
        return 0

    # Chunks in the store that are not yet in the vector table
    existing_ids: set[str] = set()
    try:
        if lance.is_queryable():
            table = lance._table
            existing_ids = {
                row["chunk_id"] for row in table.to_arrow().to_pylist()
            } if table is not None else set()
    except Exception:
        existing_ids = set()

    pending = [
        chunk for chunk in store.all_chunks()
        if chunk.chunk_id not in existing_ids
    ]
    if not pending:
        return 0

    embed = ctx.embedding_adapter()
    vectors, sparse_weights = embed.embed_texts_hybrid([c.text for c in pending])
    lance.add_chunks(pending, vectors, sparse_weights=sparse_weights)
    try:
        lance.create_fts_index()
    except Exception:
        pass  # FTS index may already exist (pre-existing LanceDB quirk)
    return len(pending)


def _enqueue_review(
    url: str,
    title: str | None,
    text: str,
    reason: str,
    query: str,
    research_request_id: str | None,
) -> None:
    """Queue a scraped-but-rejected doc for the short-idle LLM re-read.

    The doc keeps its chance: a worker re-reads it after ~1 min of user
    inactivity and decides promote-or-discard. Non-fatal — the queue is a
    derived convenience, not canonical state.
    """
    try:
        from ipa.agent.research_review import ResearchReviewStore
        store = ResearchReviewStore()
        try:
            store.enqueue(
                url=url, title=title, text=text, reason=reason,
                query=query, research_request_id=research_request_id,
            )
        finally:
            store.close()
    except Exception:
        pass


def execute_research(
    query: str,
    ctx: ToolContext,
    *,
    session_id: str,
    episode_id: str,
    research_request_id: str | None = None,
    max_urls: int = 5,
    max_seconds: int = 120,
    allowed_domains: list[str] | None = None,
    max_age_days: int = MAX_ARTICLE_AGE_DAYS,
    freshness: str = "lenient",
    judge: Any | None = None,
    landing_dir: str | Path = "Landing/web",
) -> tuple[ToolCall, ToolResult, ResearchResult]:
    """Execute a bounded, agent-driven research request end-to-end.

    The agent (via ``judge``) reads search snippets and scraped text and
    decides what is worth ingesting. Without a judge, the deterministic
    HeuristicJudge makes the decisions (Fase 1 behavior).

    Args:
        query: The research question to search for.
        ctx: Tool context with memory and optional corpus.
        session_id / episode_id: Agent session/episode context.
        research_request_id: Optional ResearchRequest that authorized this.
        max_urls: Budget — max URLs to fetch and ingest.
        max_seconds: Budget — max wall-clock time.
        allowed_domains: Optional domain whitelist.
        max_age_days: Age cutoff in days (0 = no filter). Only a hard reject
            in ``freshness="strict"``; in lenient mode it is a signal for the judge.
        freshness: "lenient" (default) — publication date is passed to the
            content judge as context (staleness is a semantic decision).
            "strict" — articles older than max_age_days are rejected outright
            (for news / time-sensitive topics).
        judge: Judge instance (LLMJudge or HeuristicJudge). None → HeuristicJudge.
        landing_dir: Where scraped content lands (Landing/web by default).

    Returns:
        (ToolCall, ToolResult, ResearchResult) — contract records + structured output.
    """
    if freshness not in ("lenient", "strict"):
        raise ValueError(f"freshness must be 'lenient' or 'strict', got: {freshness!r}")
    if judge is None:
        judge = HeuristicJudge()

    call_id = f"tool_call:{_compact_stamp()}"
    result_id = f"tool_result:{_compact_stamp()}"
    started = _now()
    t0 = time.monotonic()

    research = ResearchResult(query=query)
    judgments: list[SourceJudgment] = []

    try:
        # 1. Web search
        search_summary = search_web(query, max_results=max_urls * 3, timeout=15)
        research.search_results_count = len(search_summary.results)

        if search_summary.error:
            raise RuntimeError(f"web search failed: {search_summary.error}")

        candidates = search_summary.results
        if allowed_domains:
            allowed_lower = {d.lower() for d in allowed_domains}
            candidates = [r for r in candidates if any(d in r.domain.lower() for d in allowed_lower)]

        # 2. Agent reads snippets and judges which URLs are worth scraping.
        #    LLMJudge: one batch call for all candidates. HeuristicJudge:
        #    token overlap per candidate. Deterministic scaffold first —
        #    only send plausible candidates to the LLM (cheap → expensive).
        prefiltered = [
            r for r in candidates
            if _snippet_relevance(query, r.title, r.snippet) >= SNIPPET_RELEVANCE_THRESHOLD
        ]
        if not prefiltered:
            raise RuntimeError("no search results passed the deterministic snippet pre-filter")

        snippet_payloads = [
            {"url": r.url, "title": r.title, "snippet": r.snippet} for r in prefiltered
        ]
        snippet_judgments = judge.judge_snippets(query, snippet_payloads)
        for payload, j in zip(snippet_payloads, snippet_judgments):
            judgments.append(SourceJudgment(
                url=payload["url"], stage="snippet", verdict=j.verdict,
                reason=j.reason, judge=j.judge, confidence=j.confidence,
            ))

        # Retry buffer: iterate ALL snippet-accepted candidates in relevance
        # order. max_urls counts successful ingestions, not attempts — when a
        # candidate fails (scrape/date/quality/duplicate), the next one is
        # tried, so rejections rotate instead of shrinking the result set.
        accepted_candidates = [
            result for result, j in zip(prefiltered, snippet_judgments)
            if j.verdict == "accept"
        ]
        if not accepted_candidates:
            raise RuntimeError("agent rejected all search results at the snippet stage")

        # 3. Scrape each accepted URL; the agent then reads the raw text.
        #    engine="auto": requests first, Playwright fallback for JS-heavy
        #    sites (e.g. reddit) that requests+trafilatura cannot extract.
        from ipa.acquisition.web_scraper import WebScraper

        scraper = WebScraper(
            output_dir=str(landing_dir),
            engine="auto",
            download_images=False,
        )

        web_sources: list[WebSource] = []
        accepted_texts: list[str] = []
        accepted_hashes: set[str] = set()
        scraped_count = 0
        attempted_count = 0
        rejected_count = 0
        rejection_reasons: list[str] = []

        for result in accepted_candidates:
            if scraped_count >= max_urls:
                break  # budget: successful ingestions reached
            if time.monotonic() - t0 > max_seconds:
                research.error = f"budget exceeded: max_seconds={max_seconds}"
                break

            attempted_count += 1
            try:
                scrape_result = scraper.extract_article(result.url, days_back=0)
                if not scrape_result.success:
                    failure_kind = _classify_scrape_error(scrape_result.error)
                    judgments.append(SourceJudgment(
                        url=result.url, stage="scrape", verdict="error",
                        reason=f"[{failure_kind}] {scrape_result.error or 'scrape failed'}",
                        judge="heuristic", kind=failure_kind,
                    ))
                    rejection_reasons.append(f"{result.url}: scrape failed ({failure_kind})")
                    continue

                # Record which fetch engine succeeded (auto mode may have
                # retried with Playwright after a requests failure).
                engine_used = (scrape_result.metadata or {}).get("engine")
                if engine_used and engine_used != "requests":
                    judgments.append(SourceJudgment(
                        url=result.url, stage="scrape", verdict="accept",
                        reason=f"extracted via {engine_used} fallback",
                        judge="heuristic", confidence=1.0,
                    ))

                # Deterministic scaffold first (cheap): structure + date.
                quality, reason = _content_quality(scrape_result.text, query)
                if quality < CONTENT_QUALITY_THRESHOLD:
                    rejected_count += 1
                    msg = f"{result.url}: {reason} (quality={quality:.2f})"
                    rejection_reasons.append(msg)
                    judgments.append(SourceJudgment(
                        url=result.url, stage="content", verdict="reject",
                        reason=msg, judge="heuristic", confidence=quality,
                    ))
                    _enqueue_review(
                        result.url, scrape_result.title or result.title,
                        scrape_result.text, msg, query, research_request_id,
                    )
                    continue

                # Freshness: strict mode rejects old articles outright;
                # lenient mode passes the age to the content judge as context
                # (staleness is a semantic decision, not arithmetic).
                age_days = _article_age_days(scrape_result.date)
                if freshness == "strict" and _article_is_too_old(scrape_result.date, max_age_days):
                    rejected_count += 1
                    msg = (f"{result.url}: article dated {scrape_result.date} "
                           f"is older than {max_age_days} days")
                    rejection_reasons.append(msg)
                    judgments.append(SourceJudgment(
                        url=result.url, stage="date", verdict="reject",
                        reason=msg, judge="heuristic", confidence=1.0,
                    ))
                    _enqueue_review(
                        result.url, scrape_result.title or result.title,
                        scrape_result.text, msg, query, research_request_id,
                    )
                    continue

                # Agent reads the raw scraped text and judges it (LLM or
                # heuristic fallback). This is the direct-reading capability:
                # the text never needs to be ingested first to be evaluated.
                content_judgment: Judgment = judge.judge_content(
                    query, scrape_result.title or result.title, scrape_result.text,
                    age_days=age_days,
                )
                judgments.append(SourceJudgment(
                    url=result.url, stage="content", verdict=content_judgment.verdict,
                    reason=content_judgment.reason, judge=content_judgment.judge,
                    confidence=content_judgment.confidence,
                ))
                if content_judgment.verdict != "accept":
                    rejected_count += 1
                    rejection_reasons.append(f"{result.url}: {content_judgment.reason}")
                    _enqueue_review(
                        result.url, scrape_result.title or result.title,
                        scrape_result.text, content_judgment.reason, query,
                        research_request_id,
                    )
                    continue

                # Duplicate check: exact hash or near-duplicate of accepted text.
                raw_hash = scrape_result.content_hash or content_hash(scrape_result.text)
                if not raw_hash.startswith("sha256:"):
                    raw_hash = "sha256:" + raw_hash
                if raw_hash in accepted_hashes or _is_duplicate(scrape_result.text, accepted_texts):
                    rejected_count += 1
                    msg = f"{result.url}: duplicate of already-accepted content"
                    rejection_reasons.append(msg)
                    judgments.append(SourceJudgment(
                        url=result.url, stage="duplicate", verdict="reject",
                        reason="duplicate content", judge="heuristic", confidence=1.0,
                    ))
                    continue

                # Accepted — save immediately (fixes the save-only-last bug)
                try:
                    scraper.save_article(scrape_result)
                except Exception:
                    pass

                ws = WebSource(
                    web_source_id=f"web_source:{_compact_stamp()}",
                    source_url=result.url,
                    fetched_at=_now(),
                    content_hash=raw_hash,
                    trust_label=_trust_label_for_domain(result.domain),
                    fetch_method=("playwright" if engine_used == "playwright" else "requests"),
                    content_type="text/html",
                    byte_size=len(scrape_result.text.encode("utf-8")),
                    title=scrape_result.title or result.title,
                    canonical_url=scrape_result.canonical_url or None,
                    research_request_id=research_request_id,
                )
                web_sources.append(ws)
                accepted_hashes.add(raw_hash)
                accepted_texts.append(scrape_result.text)
                scraped_count += 1
            except Exception as exc:
                # Record the unexpected failure instead of swallowing it
                # (PAT-004: every budget spend leaves an audit trace).
                failure_kind = _classify_scrape_error(None, exc)
                judgments.append(SourceJudgment(
                    url=result.url, stage="scrape", verdict="error",
                    reason=f"[{failure_kind}] {type(exc).__name__}: {str(exc)[:120]}",
                    judge="heuristic", kind=failure_kind,
                ))
                continue  # research is best-effort per URL

        research.web_sources = web_sources
        research.scraped_count = scraped_count
        research.rejected_count = rejected_count
        research.rejection_reasons = rejection_reasons
        research.judgments = judgments

        if not web_sources:
            raise RuntimeError(
                "no URLs accepted: "
                + ("; ".join(rejection_reasons[:3]) if rejection_reasons else "all scrapes failed")
            )

        # 4. Ingestion — FastPath over the landing directory (accepted only,
        #    because rejected content was never saved).
        ingested = 0
        embedded = 0
        if ctx.corpus_dir:
            try:
                from ipa.ingestion.fast_path import FastPathRunner
                corpus = Path(ctx.corpus_dir)
                runner = FastPathRunner(
                    landing_db=str(corpus / "landing.db"),
                    store_db=str(corpus / "document_store.db"),
                    index_db=str(corpus / "bm25_index.db"),
                    landing_root=str(landing_dir),
                )
                results = runner.ingest_directory(str(landing_dir), progress=False, skip_indexed=True)
                ingested = sum(1 for r in results if r.document_id is not None)
                runner.close()

                # 4a. Record provenance — agent research documents get
                #     provenance="agent_research" so the promotion policy
                #     applies the score threshold (>= 0.70).
                if ingested > 0:
                    try:
                        import sqlite3 as _sql3
                        from ipa.storage.document_store import DocumentStore
                        from ipa.ingestion.provenance import record_agent_research
                        agent_store = DocumentStore(corpus / "document_store.db")
                        try:
                            # Build source_uri → web_source URL map from landing zone
                            landing_conn = _sql3.connect(str(corpus / "landing.db"))
                            try:
                                art_rows = landing_conn.execute(
                                    "SELECT artifact_id, source_uri FROM artifacts"
                                ).fetchall()
                            finally:
                                landing_conn.close()

                            # Build artifact_id → source_uri
                            art_to_uri = {row[0]: row[1] for row in art_rows}

                            # For each ingested result, find matching web_source
                            for r in results:
                                if r.document_id is None or not r.artifact_id:
                                    continue
                                source_uri = art_to_uri.get(r.artifact_id, "")
                                if not source_uri:
                                    continue
                                # Match web_source by URL contained in the source_uri path
                                matched_url = ""
                                for ws in web_sources:
                                    if ws.source_url and ws.source_url in source_uri:
                                        matched_url = ws.source_url
                                        break
                                if not matched_url and web_sources:
                                    # Fallback: use the first web_source
                                    matched_url = web_sources[0].source_url
                                if matched_url:
                                    record_agent_research(
                                        agent_store, r.document_id, matched_url,
                                    )
                        finally:
                            agent_store.close()
                    except Exception:
                        pass  # non-fatal; provenance is best-effort
            except Exception:
                pass  # non-fatal; web_sources are still valid

            # 4b. Embedding — index the new chunks into LanceDB so retrieval
            #     uses dense vectors, not only the BM25 fallback.
            if ingested > 0:
                try:
                    embedded = _embed_new_chunks(corpus, ctx)
                except Exception:
                    pass  # non-fatal; BM25 retrieval still works

        research.ingested_count = ingested
        research.embedded_count = embedded

        # 5. Retrieval with citations over the corpus (now including new material)
        retrieval_hits = []
        if ctx.corpus_dir and ingested > 0:
            try:
                from ipa.agent.agent_tools import _search_corpus
                result_dict, _ = _search_corpus({"query": query, "limit": 5}, ctx)
                retrieval_hits = result_dict.get("hits", [])
            except Exception as exc:
                research._retrieval_error = str(exc)  # non-fatal

        research.retrieval_hits = retrieval_hits
        research.budget_used = {
            "max_urls": max_urls,
            "max_seconds": max_seconds,
            "elapsed_seconds": round(time.monotonic() - t0, 2),
            "urls_searched": research.search_results_count,
            "urls_prefiltered": len(prefiltered),
            "urls_attempted": attempted_count,
            "urls_scraped": scraped_count,
            "urls_rejected": rejected_count,
            "urls_ingested": ingested,
            "chunks_embedded": embedded,
        }

        result_dict = {
            "query": query,
            "web_sources": [ws.web_source_id for ws in web_sources],
            "web_source_details": [ws.to_contract() for ws in web_sources],
            "search_results_count": research.search_results_count,
            "scraped_count": scraped_count,
            "rejected_count": rejected_count,
            "rejection_reasons": rejection_reasons,
            "judgments": [j.to_dict() for j in judgments],
            "ingested_count": ingested,
            "embedded_count": embedded,
            "retrieval_hits": retrieval_hits,
            "budget_used": research.budget_used,
        }
        source_refs = [
            {"source_id": ws.web_source_id, "source_type": "artifact", "content_hash": ws.content_hash}
            for ws in web_sources
        ]

        elapsed = int((time.monotonic() - t0) * 1000)
        call = ToolCall(
            tool_call_id=call_id, session_id=session_id, episode_id=episode_id,
            tool_name="research_topic",
            arguments={"query": query, "max_urls": max_urls, "max_seconds": max_seconds,
                       "freshness": freshness, "max_age_days": max_age_days},
            called_at=started, status="completed",
        )
        result = ToolResult(
            tool_result_id=result_id, tool_call_id=call_id, session_id=session_id,
            tool_name="research_topic", result=result_dict,
            result_hash=_result_hash(result_dict),
            source_refs=source_refs,
            started_at=started, completed_at=_now(),
            elapsed_ms=elapsed, status="completed",
        )

    except Exception as exc:
        elapsed = int((time.monotonic() - t0) * 1000)
        error_msg = str(exc)
        research.error = error_msg
        research.judgments = judgments
        result_dict = {
            "query": query,
            "web_sources": [ws.web_source_id for ws in research.web_sources],
            "judgments": [j.to_dict() for j in judgments],
            "error": error_msg,
        }
        call = ToolCall(
            tool_call_id=call_id, session_id=session_id, episode_id=episode_id,
            tool_name="research_topic",
            arguments={"query": query, "max_urls": max_urls, "max_seconds": max_seconds,
                       "freshness": freshness, "max_age_days": max_age_days},
            called_at=started, status="failed", error=error_msg,
        )
        result = ToolResult(
            tool_result_id=result_id, tool_call_id=call_id, session_id=session_id,
            tool_name="research_topic", result=result_dict,
            result_hash=_result_hash(result_dict),
            source_refs=[], started_at=started, completed_at=_now(),
            elapsed_ms=elapsed, status="failed", error=error_msg,
        )

    return call, result, research


__all__ = ["WebSource", "ResearchResult", "SourceJudgment", "execute_research"]
