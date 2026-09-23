"""research_topic executor — agentic research flow (Fase 1→2 bridge).

Flow (the agent reads and judges; tools are deterministic adapters):

  1. web search (DuckDuckGo HTML, no API key) → URLs with snippets
  2. agent reads snippets → semantic judgment of which URLs are worth
     scraping (LLMJudge; HeuristicJudge as cheap scaffold + fallback)
  3. scrape accepted URLs (existing E4 adapter) into a per-run work dir
     (``outputs/agent/research/<run_id>/``, not the shared ``Landing/web``)
  4. agent reads raw scraped text → judges each document:
     - accept  → save to the run dir + ingest into the canonical corpus
     - reject  → discard with explicit reason (paywall, stub, duplicate, stale)
  5. FastPath ingestion of the accepted material only, under the heavy-work
     lock and with a post-scrape budget (``max_ingest_seconds``)
  6. retrieval with citations over the corpus

Steps 4-5 (ingest + embeddings) are the heavy phase: they take the heavy-work
lock with interactive priority and embed only this run's chunks — sharing the
landing dir with a concurrent pipeline made a 120 s research run take 27 min
(PM-004).

Every judgment is recorded (url, stage, verdict, reason, judge) — PAT-004
budgets + traceability. Web material remains derived and labeled, never
canonical authority (PAT-003).
"""
from __future__ import annotations

import os
import re
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
from .web_search import search_web, SearchResult, SearchSummary


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

ROOT = Path(__file__).resolve().parents[3]
# Dir de trabajo por corrida: el scrape NO comparte Landing/web con el scraper
# del pipeline. Compartirlo hacía que la ingesta de la research recorriera los
# ~600 archivos de la ingesta masiva (heredaba su trabajo) y que ambos
# escribieran el mismo árbol — medido 2026-09-22 (PM-004). El handoff al corpus
# principal sigue intacto: la ingesta registra provenance agent_research en el
# corpus canónico; el material queda acá como traza de auditoría.
RESEARCH_WORK_ROOT = ROOT / "outputs" / "agent" / "research"
# Presupuesto de las etapas post-scrape (ingesta + embeddings). El
# ``max_seconds`` de la tool solo acota el loop de scrape; sin este límite una
# corrida con presupuesto de 120 s podía tardar horas en el embed (PM-004).
DEFAULT_INGEST_BUDGET_S = float(os.environ.get("IPA_RESEARCH_INGEST_BUDGET", "600") or 600)
EMBED_BATCH_SIZE = int(os.environ.get("IPA_RESEARCH_EMBED_BATCH", "64") or 64)


def _research_run_dir(query: str) -> Path:
    """Dir de trabajo privado de una corrida de research."""
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:40] or "research"
    return RESEARCH_WORK_ROOT / f"{_compact_stamp()}-{slug}"


def _source_url_from_artifact(path_str: str) -> str:
    """URL real de un artefacto scrapeado: el scraper escribe
    ``Source: <url>`` en la cabecera del .txt (``source_uri`` en landing.db
    es el path local, no la URL). Vacío si no se puede leer."""
    try:
        with open(path_str, "r", encoding="utf-8", errors="replace") as fh:
            for _ in range(12):
                line = fh.readline()
                if not line:
                    break
                if line.startswith("Source:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def _embed_new_chunks(corpus: Path, ctx: Any, *, document_ids: set[str] | None = None,
                      deadline: float | None = None,
                      batch_size: int = EMBED_BATCH_SIZE) -> int:
    """Embed this run's chunks that lack a vector and add them to LanceDB.

    Completes the ingestion pipeline: DocumentStore + BM25 (FastPath) →
    dense vectors (BGE-M3) → LanceDB. Returns the number of chunks embedded.
    Non-fatal: retrieval falls back to BM25 if this fails.

    ``document_ids`` acota el embed a los documentos de ESTA corrida (antes se
    embebía todo chunk pendiente del corpus canónico — miles, en una sola
    llamada sin deadline: PM-004). ``deadline`` (monotonic) corta entre batches.
    """
    store = ctx.document_store()
    lance = ctx.lance_index()
    if store is None or lance is None:
        return 0

    # Chunks in the store that are not yet in the vector table
    existing_ids: set[str] = set()
    try:
        if lance.is_queryable() and lance._table is not None:
            existing_ids = lance.chunk_ids()  # lectura proyectada, sin vectores
    except Exception:
        existing_ids = set()

    pending = [
        chunk for chunk in store.all_chunks()
        if chunk.chunk_id not in existing_ids
        and (document_ids is None or chunk.document_id in document_ids)
    ]
    if not pending:
        return 0

    embed = ctx.embedding_adapter()
    if embed is None:
        return 0

    # Backlog grande → escalar al lote GPU exclusivo: el mismo mecanismo del
    # drain standalone (estado de mantenimiento publicado → el chat se pausa,
    # vram_lock, descarga del LLM, BGE-M3 en CUDA, restauración al terminar).
    # Antes la research embebía siempre inline por CPU — con el chat cargado
    # el gate de VRAM jamás daba CUDA y un backlog de miles tardaba horas en
    # vez de segundos (~45x medido en PM-004).
    from ipa.agentic.chunk_enrichment import enriched_text
    bulk_session: dict | None = None
    job_claimed = False
    maintenance = None
    try:
        from ipa.agentic import embedding_maintenance as maintenance
        from ipa.ingestion.fast_path_cli import (
            EMBED_GPU_BULK_ENABLED, EMBED_GPU_MIN_BACKLOG,
            _finish_bulk_gpu, _start_bulk_gpu, _update_bulk_gpu)
        if (EMBED_GPU_BULK_ENABLED and len(pending) >= EMBED_GPU_MIN_BACKLOG
                and maintenance.claim_job("research_embed", wait_s=0)):
            job_claimed = True
            # El wait por VRAM queda acotado al budget de ingesta (max 120s):
            # una generación en vuelo termina en segundos; si el lock no se
            # libera, cae a CPU y sigue — nunca deadlockea la respuesta.
            remaining = (deadline - time.monotonic()) if deadline is not None else 120.0
            bulk_session = _start_bulk_gpu(
                embed, corpus=Path(corpus), total=store.count_chunks(),
                vectorized=max(0, store.count_chunks() - len(pending)),
                pending=len(pending), embedded_before=0,
                wait_s=max(0.0, min(120.0, remaining)))
    except Exception:
        bulk_session = None

    embedded = 0
    try:
        for start in range(0, len(pending), batch_size):
            if deadline is not None and time.monotonic() > deadline:
                break
            batch = pending[start:start + batch_size]
            vectors, sparse_weights = embed.embed_texts_hybrid(
                [enriched_text(c.text, getattr(c, "metadata", None) or {})
                 for c in batch])
            lance.add_chunks(batch, vectors, sparse_weights=sparse_weights)
            embedded += len(batch)
            if bulk_session:
                _update_bulk_gpu(
                    bulk_session, embedded=embedded,
                    total=store.count_chunks(),
                    vectorized_before=bulk_session["vectorized_before"],
                    pending=max(0, len(pending) - embedded))
    finally:
        if bulk_session:
            _finish_bulk_gpu(embed, bulk_session, status="completed")
            # close() deja device="cuda": sin esto el próximo embed del ctx
            # recargaría BGE-M3 en GPU encima del chat recién restaurado.
            embed.release_gpu()
        if job_claimed and maintenance is not None:
            maintenance.release_job("research_embed")
    try:
        lance.create_fts_index()
    except Exception:
        pass  # FTS index may already exist (pre-existing LanceDB quirk)
    return embedded


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
    landing_dir: str | Path | None = None,
    max_ingest_seconds: float | None = None,
    on_heavy_wait: Any | None = None,
    sub_queries: list[str] | None = None,
    staging_corpus_dir: str | Path | None = None,
    on_progress: Any | None = None,
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
        landing_dir: Where scraped content lands. None (default) → dir de
            trabajo privado por corrida bajo ``outputs/agent/research/``; así
            la research no comparte ``Landing/web`` con el scraper del pipeline
            (PM-004). El override explícito sigue disponible para tests/CLIs.
        max_ingest_seconds: Budget de las etapas post-scrape (ingesta +
            embeddings). None → ``IPA_RESEARCH_INGEST_BUDGET`` (600 s).
        on_heavy_wait: Callback opcional ``(elapsed_s, holder)`` que se llama
            mientras se espera el lock de trabajos pesados (para reportar
            "encolado detrás de <kind>" en el progreso).
        sub_queries: Facetas extra del mismo tema escritas por el agente
            (máx 8). Cada una corre su propia búsqueda web y sus resultados
            entran al pool deduplicado por URL; el prefilter y el juicio de
            snippets/contenido se hacen contra la query que produjo cada
            candidato — una faceta con vocabulario propio no debería fallar
            por no matchear la query principal.
        staging_corpus_dir: Corpus de staging donde aterriza la ingesta
            (DEC-003): juez por fuente → staging → curación T1 → política de
            promoción → main. ``None`` (default) → ``ctx.corpus_dir``
            (comportamiento legacy: directo al corpus dado). El retrieval de
            la respuesta final sigue consultando ``ctx.corpus_dir`` (main)
            y suma los hits del staging recién ingerido.
        on_progress: Callback opcional ``(phase, detail: dict)`` invocado en
            cada transición de fase — search → judge → scrape (por URL) →
            ingest → embed → retrieval. Best-effort: las excepciones del
            callback se tragan, nunca cortan la investigación.

    Returns:
        (ToolCall, ToolResult, ResearchResult) — contract records + structured output.
    """
    if freshness not in ("lenient", "strict"):
        raise ValueError(f"freshness must be 'lenient' or 'strict', got: {freshness!r}")
    if judge is None:
        judge = HeuristicJudge()
    if landing_dir is None:
        landing_dir = _research_run_dir(query)
    if max_ingest_seconds is None:
        max_ingest_seconds = DEFAULT_INGEST_BUDGET_S

    call_id = f"tool_call:{_compact_stamp()}"
    result_id = f"tool_result:{_compact_stamp()}"
    started = _now()
    t0 = time.monotonic()

    research = ResearchResult(query=query)
    judgments: list[SourceJudgment] = []

    def _emit(phase: str, **detail: Any) -> None:
        if on_progress is not None:
            try:
                on_progress(phase, detail)
            except Exception:
                pass

    try:
        # 1. URLs embebidas en la query (pegadas por el usuario, o reinyectadas
        #    desde su mensaje por tool_research_topic): son fuentes explícitas.
        #    Se scrapean directo — saltean el snippet stage (no hay snippet que
        #    juzgar) pero pasan igual por scrape → calidad → juicio de contenido
        #    → ingesta. El remanente textual va a la búsqueda complementaria.
        from urllib.parse import urlparse as _urlparse
        from .web_search import extract_urls, query_from_url, strip_urls
        seed_urls = extract_urls(query)
        text_query = strip_urls(query)
        if not text_query and seed_urls:
            # Query solo-URL: derivar la búsqueda complementaria del slug.
            text_query = query_from_url(seed_urls[0]) or seed_urls[0]
        judge_query = text_query or query
        seed_results = [
            SearchResult(url=u, title=query_from_url(u) or u, snippet="",
                         domain=_urlparse(u).netloc)
            for u in seed_urls[:max_urls]
        ]
        seed_url_set = {s.url for s in seed_results}
        for sr in seed_results:
            judgments.append(SourceJudgment(
                url=sr.url, stage="seed", verdict="accept",
                reason="URL explícita en la query (fuente provista por el usuario)",
                judge="heuristic", confidence=1.0,
            ))

        # 1a. Web search sobre el remanente textual + sub-queries del
        #     agente. Cada sub-query es una faceta del mismo tema: corre su
        #     propia búsqueda y sus resultados entran al pool deduplicado
        #     por URL, recordando qué query los produjo (el prefilter y el
        #     juicio se hacen contra esa query, no contra la principal).
        sub_queries = [
            s.strip() for s in (sub_queries or [])
            if isinstance(s, str) and s.strip() and s.strip() != judge_query
        ][:8]

        summaries: list[tuple[str, SearchSummary]] = [
            (judge_query, search_web(judge_query, max_results=max_urls * 3, timeout=15)),
        ]
        for sq in sub_queries:
            summaries.append((sq, search_web(sq, max_results=max_urls, timeout=15)))

        result_query: dict[str, str] = {}
        merged_results: list[SearchResult] = []
        search_errors: list[str] = []
        for q, summary in summaries:
            if summary.error:
                search_errors.append(f"{q}: {summary.error}")
                continue
            for r in summary.results:
                if r.url not in result_query:
                    result_query[r.url] = q
                    merged_results.append(r)
        research.search_results_count = len(merged_results)
        _emit("search", results=len(merged_results), seeds=len(seed_results),
              sub_queries=len(sub_queries), errors=len(search_errors))

        # Con seeds o resultados de alguna sub-query, un fallo de búsqueda
        # no invalida la investigación (se registra el error igual).
        if search_errors and not merged_results and not seed_results:
            raise RuntimeError(f"web search failed: {search_errors[0]}")
        for err in search_errors:
            judgments.append(SourceJudgment(
                url="(web search)", stage="search", verdict="error",
                reason=err, judge="heuristic",
            ))

        candidates = merged_results
        if allowed_domains:
            allowed_lower = {d.lower() for d in allowed_domains}
            candidates = [r for r in candidates if any(d in r.domain.lower() for d in allowed_lower)]

        # 2. Agent reads snippets and judges which URLs are worth scraping.
        #    LLMJudge: one batch call for all candidates. HeuristicJudge:
        #    token overlap per candidate. Deterministic scaffold first —
        #    only send plausible candidates to the LLM (cheap → expensive).
        prefiltered = [
            r for r in candidates
            if _snippet_relevance(result_query.get(r.url, judge_query), r.title, r.snippet)
            >= SNIPPET_RELEVANCE_THRESHOLD
        ]
        if not prefiltered and not seed_results:
            raise RuntimeError("no search results passed the deterministic snippet pre-filter")

        # El juicio de snippets se hace por faceta: un candidato producido
        # por una sub-query se evalúa contra esa sub-query, no contra la
        # query principal (vocabulario de faceta ≠ vocabulario del tema).
        snippet_judgments: list[Any] = [None] * len(prefiltered)
        by_query: dict[str, list[int]] = {}
        for i, r in enumerate(prefiltered):
            by_query.setdefault(result_query.get(r.url, judge_query), []).append(i)
        for q, idxs in by_query.items():
            payloads = [
                {"url": prefiltered[i].url, "title": prefiltered[i].title,
                 "snippet": prefiltered[i].snippet}
                for i in idxs
            ]
            for i, j in zip(idxs, judge.judge_snippets(q, payloads)):
                snippet_judgments[i] = j
                judgments.append(SourceJudgment(
                    url=prefiltered[i].url, stage="snippet", verdict=j.verdict,
                    reason=j.reason, judge=j.judge, confidence=j.confidence,
                ))

        # Retry buffer: iterate ALL snippet-accepted candidates in relevance
        # order. max_urls counts successful ingestions, not attempts — when a
        # candidate fails (scrape/date/quality/duplicate), the next one is
        # tried, so rejections rotate instead of shrinking the result set.
        # Seeds primero (fuentes explícitas del usuario), después los
        # candidatos aceptados por snippet — sin duplicar URLs ya seedeadas.
        accepted_candidates = seed_results + [
            result for result, j in zip(prefiltered, snippet_judgments)
            if j is not None and j.verdict == "accept" and result.url not in seed_url_set
        ]
        _emit("judge", candidates=len(prefiltered),
              accepted=len(accepted_candidates), seeds=len(seed_results))
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
            _emit("scrape", done=attempted_count, total=len(accepted_candidates),
                  accepted=scraped_count, rejected=rejected_count)
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
                # El candidato se evalúa contra la query que lo produjo
                # (faceta), no contra la query principal.
                effective_query = result_query.get(result.url, judge_query)
                quality, reason = _content_quality(scrape_result.text, effective_query)
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
                        scrape_result.text, msg, judge_query, research_request_id,
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
                        scrape_result.text, msg, judge_query, research_request_id,
                    )
                    continue

                # Agent reads the raw scraped text and judges it (LLM or
                # heuristic fallback). This is the direct-reading capability:
                # the text never needs to be ingested first to be evaluated.
                content_judgment: Judgment = judge.judge_content(
                    effective_query, scrape_result.title or result.title, scrape_result.text,
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
                        scrape_result.text, content_judgment.reason, judge_query,
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

        _emit("scrape", done=attempted_count, total=len(accepted_candidates),
              accepted=scraped_count, rejected=rejected_count)
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

        # 4. Ingestion — FastPath over THIS run's private work dir (accepted
        #    only: rejected content was never saved). Ingesta + embeddings son
        #    la fase pesada: corren bajo el lock de trabajos pesados (prioridad
        #    interactiva) y con budget propio, para no quedar detrás de un job
        #    de background ni heredar su carga (PM-004).
        ingested = 0
        embedded = 0
        ingest_budget_used: dict[str, Any] = {
            "max_ingest_seconds": max_ingest_seconds,
            "heavy_lock_acquired": False,
            "heavy_wait_seconds": 0.0,
        }
        heavy_wait_state: dict[str, Any] = {"seconds": 0.0, "holder": None}

        def _on_heavy_wait(elapsed_s: float, current: dict[str, Any] | None) -> None:
            heavy_wait_state["seconds"] = elapsed_s
            heavy_wait_state["holder"] = (current or {}).get("kind")
            if on_heavy_wait is not None:
                try:
                    on_heavy_wait(elapsed_s, current)
                except Exception:
                    pass

        ingest_phase_start = time.monotonic()
        # DEC-003: la research aterriza en el staging corpus — la promoción a
        # main la decide la curación T1 + promotion_policy, no el juez de
        # fuentes. URLs semilla (pegadas por el usuario) se marcan
        # provenance="user_provided" → auto-promoción como configured_scrape.
        ingest_corpus_dir = staging_corpus_dir or ctx.corpus_dir
        if ingest_corpus_dir:
            from ipa.agentic import heavy_lock

            _emit("ingest")
            with heavy_lock.heavy_phase(
                    "research_ingest", heavy_lock.PRIORITY_INTERACTIVE,
                    wait_s=heavy_lock.DEFAULT_WAIT_S,
                    on_wait=_on_heavy_wait) as held:
                ingest_budget_used["heavy_lock_acquired"] = held
                ingest_budget_used["heavy_wait_seconds"] = round(heavy_wait_state["seconds"], 2)
                ingest_budget_used["heavy_lock_holder"] = heavy_wait_state["holder"]
                ingest_deadline = time.monotonic() + float(max_ingest_seconds)
                run_document_ids: set[str] = set()
                try:
                    from ipa.ingestion.fast_path import FastPathRunner
                    corpus = Path(ingest_corpus_dir)
                    corpus.mkdir(parents=True, exist_ok=True)
                    runner = FastPathRunner(
                        landing_db=str(corpus / "landing.db"),
                        store_db=str(corpus / "document_store.db"),
                        index_db=str(corpus / "bm25_index.db"),
                        landing_root=str(landing_dir),
                    )
                    results = runner.ingest_directory(str(landing_dir), progress=False, skip_indexed=True)
                    ingested = sum(1 for r in results if r.document_id is not None)
                    run_document_ids = {r.document_id for r in results if r.document_id}
                    runner.close()
                    _emit("ingest", docs=ingested)

                    # 4a. Record provenance — agent research documents get
                    #     provenance="agent_research" so the promotion policy
                    #     applies the score threshold (>= 0.70).
                    if ingested > 0:
                        try:
                            import sqlite3 as _sql3
                            from ipa.storage.document_store import DocumentStore
                            from ipa.ingestion.provenance import record_agent_research
                            _seed_norm = {u.rstrip("/") for u in seed_url_set}
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

                                # For each ingested result, resolve its real
                                # URL. El header "Source:" del artefacto es
                                # autoritativo: source_uri es el path LOCAL
                                # (slugs con '-'), así que el match URL-en-path
                                # casi nunca disparaba y el viejo fallback
                                # (web_sources[0]) asignaba una URL ajena —
                                # eso corrompió document_sources y el dedupe
                                # tombstoneó 373 docs legítimos (repair
                                # 2026-09-23). Sin URL → no registrar nada.
                                for r in results:
                                    if r.document_id is None or not r.artifact_id:
                                        continue
                                    source_uri = art_to_uri.get(r.artifact_id, "")
                                    if not source_uri:
                                        continue
                                    matched_url = _source_url_from_artifact(source_uri)
                                    if not matched_url:
                                        for ws in web_sources:
                                            if ws.source_url and ws.source_url in source_uri:
                                                matched_url = ws.source_url
                                                break
                                    if matched_url:
                                        # URL semilla = pegada por el usuario
                                        # → fuente conocida (auto-promoción);
                                        # el resto → agent_research (gate
                                        # promotion_score >= 0.70 en T1).
                                        _prov = (
                                            "user_provided"
                                            if matched_url.rstrip("/") in _seed_norm
                                            else "agent_research"
                                        )
                                        record_agent_research(
                                            agent_store, r.document_id, matched_url,
                                            provenance=_prov,
                                        )
                            finally:
                                agent_store.close()
                        except Exception:
                            pass  # non-fatal; provenance is best-effort
                except Exception:
                    pass  # non-fatal; web_sources are still valid

                # Señales Tier 0: doc metadata derivada (hash/title/fecha) +
                # flag de duplicado exacto vs main — mismas señales que el
                # pipeline, así T1 no las re-deriva en cada ciclo idle.
                if run_document_ids:
                    try:
                        from ipa.ingestion.ingest_metadata import record_ingest_metadata
                        from ipa.agent.system_tools import _main_corpus_dir
                        _run_store = DocumentStore(corpus / "document_store.db")
                        try:
                            _main_path = _main_corpus_dir()
                            _ms = (DocumentStore(_main_path / "document_store.db")
                                   if _main_path and _main_path != corpus else None)
                            try:
                                record_ingest_metadata(
                                    _run_store, run_document_ids,
                                    main_store=_ms)
                            finally:
                                if _ms is not None:
                                    _ms.close()
                        finally:
                            _run_store.close()
                        # Flag "corpus changed" para el gate de topify.
                        try:
                            from ipa.agentic.topic_clusters import TopicClusterStore
                            _cs = TopicClusterStore()
                            try:
                                _cs.set_meta(f"dirty:{corpus.resolve()}", "1")
                            finally:
                                _cs.close()
                        except Exception:
                            pass
                    except Exception:
                        pass  # non-fatal; T1 backfill sigue como fallback

                # 4b. Embedding — only THIS run's chunks (document_ids) into
                #     LanceDB so retrieval uses dense vectors, not only the
                #     BM25 fallback. Batched y con deadline: el embed de un
                #     corpus entero no tenía corte (PM-004).
                if ingested > 0:
                    _emit("embed")
                    try:
                        embedded = _embed_new_chunks(
                            corpus, ctx, document_ids=run_document_ids,
                            deadline=ingest_deadline)
                    except Exception:
                        pass  # non-fatal; BM25 retrieval still works
                    _emit("embed", chunks=embedded)
        ingest_budget_used["ingest_elapsed_seconds"] = round(
            time.monotonic() - ingest_phase_start, 2)
        ingest_budget_used["ingest_budget_exceeded"] = (
            time.monotonic() - ingest_phase_start > float(max_ingest_seconds))

        # Backlog residual de embeddings en el staging (< umbral del lote GPU
        # o cortado por el deadline): la promoción DEFIERE hasta que cada
        # chunk vivo tenga vector en main (PM-004), así que un remanente sin
        # productor dejaría la cola esperando vectores que nadie genera. El
        # drain standalone corre post heavy_phase, sin compartir escritor
        # con la ingesta.
        if ingest_corpus_dir and ingested > 0:
            try:
                from ipa.agent.system_tools import _spawn_embed_drain
                _spawn_embed_drain(Path(ingest_corpus_dir))
            except Exception:
                pass  # non-fatal; la cola de promoción reintenta en ciclos T1

        research.ingested_count = ingested
        research.embedded_count = embedded

        # 5. Retrieval with citations over the corpus (now including new material)
        retrieval_hits = []
        if ingest_corpus_dir and ingested > 0:
            _emit("retrieval")
            _stage_path = Path(ingest_corpus_dir)
            _main_path = Path(ctx.corpus_dir) if ctx.corpus_dir else None
            _staged = (_main_path is None
                       or _stage_path.resolve() != _main_path.resolve())
            if _staged:
                # El material nuevo vive en staging hasta que T1 lo promueva —
                # BM25 directo (recién construido por FastPath), sin cargar un
                # segundo embedding adapter ni depender de que el embed haya
                # terminado.
                try:
                    from ipa.indexes.bm25_index import BM25Index
                    from ipa.storage.document_store import DocumentStore
                    _s_store = DocumentStore(_stage_path / "document_store.db")
                    _s_bm25 = BM25Index(_stage_path / "bm25_index.db")
                    try:
                        _seen_doc: set[str] = set()
                        for hit in _s_bm25.search(judge_query, limit=5):
                            _ch = _s_store.get_chunk(hit.chunk_id)
                            _did = _ch.document_id if _ch else "unknown"
                            if _did in _seen_doc:
                                continue
                            _seen_doc.add(_did)
                            _src = ((_s_store.get_source(_did)
                                     if _did != "unknown" else None) or {})
                            _txt = _ch.text if _ch else ""
                            retrieval_hits.append({
                                "chunk_id": hit.chunk_id,
                                "document_id": _did,
                                "score": round(hit.score, 4),
                                "retrieval_backend": "staging_bm25",
                                "published_at": (_s_store.document_stored_at(_did)
                                                 if _did != "unknown" else None),
                                "source_domain": _src.get("source_domain"),
                                "provenance": _src.get("provenance"),
                                "text_preview": ((_txt[:200] + "...")
                                                 if len(_txt) > 200 else _txt),
                            })
                    finally:
                        _s_bm25.close()
                        _s_store.close()
                except Exception:
                    pass  # non-fatal; el retrieval de main abajo sigue valiendo
            if ctx.corpus_dir:
                try:
                    from ipa.agent.agent_tools import _search_corpus
                    result_dict, _ = _search_corpus(
                        {"query": judge_query, "limit": 5}, ctx)
                    _covered = {h.get("document_id") for h in retrieval_hits}
                    for h in result_dict.get("hits", []):
                        if h.get("document_id") not in _covered:
                            retrieval_hits.append(h)
                except Exception as exc:
                    research._retrieval_error = str(exc)  # non-fatal

        research.retrieval_hits = retrieval_hits
        _emit("retrieval", hits=len(retrieval_hits))
        research.budget_used = {
            "max_urls": max_urls,
            "max_seconds": max_seconds,
            "sub_queries": sub_queries,
            "elapsed_seconds": round(time.monotonic() - t0, 2),
            "urls_searched": research.search_results_count,
            "seed_urls": len(seed_results),
            "urls_prefiltered": len(prefiltered),
            "urls_attempted": attempted_count,
            "urls_scraped": scraped_count,
            "urls_rejected": rejected_count,
            "urls_ingested": ingested,
            "chunks_embedded": embedded,
            "work_dir": str(landing_dir),
            "ingest_corpus": str(ingest_corpus_dir) if ingest_corpus_dir else None,
            "ingest": ingest_budget_used,
        }

        result_dict = {
            "query": query,
            "sub_queries": sub_queries,
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
                       "freshness": freshness, "max_age_days": max_age_days,
                       "sub_queries": sub_queries},
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
                       "freshness": freshness, "max_age_days": max_age_days,
                       "sub_queries": sub_queries},
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
