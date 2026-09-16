"""LLM-as-judge for the agentic research flow (Fase 1→2 bridge).

The agent does not blindly scrape and ingest. It judges:

  1. snippet judgment  — which search results are worth scraping (pre-scrape)
  2. content judgment  — whether scraped text is substantive and relevant
                         (post-scrape, pre-ingest)

Two implementations:

  - HeuristicJudge: deterministic token-overlap filters (cheap scaffold,
    no LLM). Used as pre-filter and as fallback when the LLM fails.
  - LLMJudge: semantic judgment via a chat provider (ExL3Provider or any
    object with generate_chat). JSON-structured verdicts per BM-006
    (structured JSON classification validity 0.90 measured). Falls back to
    the heuristic judge on provider errors or unparseable output.

Andamiaje determinístico + LLM clasifica (roadmap Fase 2, BM-006).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Deterministic scaffold filters (cheap pre-filters + heuristic fallback)
# ---------------------------------------------------------------------------

# Thresholds for the deterministic scaffold
SNIPPET_RELEVANCE_THRESHOLD = 0.15
CONTENT_QUALITY_THRESHOLD = 0.20


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for relevance scoring (no external deps)."""
    return [w for w in re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower()) if len(w) >= 2]


def _snippet_relevance(query: str, title: str, snippet: str) -> float:
    """Deterministic snippet relevance in [0.0, 1.0] (cheap pre-filter)."""
    query_terms = set(_tokenize(query))
    if not query_terms:
        return 0.0
    text = f"{title} {snippet}".lower()
    text_terms = set(_tokenize(text))
    if not text_terms:
        return 0.0
    overlap = query_terms & text_terms
    base_score = len(overlap) / len(query_terms)
    title_terms = set(_tokenize(title.lower()))
    title_overlap = query_terms & title_terms
    title_bonus = 0.15 * (len(title_overlap) / len(query_terms)) if query_terms else 0.0
    return min(1.0, base_score + title_bonus)


def _content_quality(text: str, query: str) -> tuple[float, str]:
    """Deterministic structural quality check (cheap pre-filter).

    Rejects: too-short content, zero query overlap, cookie walls / JS stubs.
    """
    if len(text) < 200:
        return 0.0, "content too short (< 200 chars)"

    query_terms = set(_tokenize(query))
    if query_terms:
        text_terms = set(_tokenize(text.lower()))
        if not (query_terms & text_terms):
            return 0.0, "no query terms found in content"

    garbage_signals = [
        "enable javascript", "please enable javascript",
        "cookie consent", "accept cookies to continue",
        "access denied", "403 forbidden", "captcha",
    ]
    text_start = text[:2000].lower()
    for signal in garbage_signals:
        if signal in text_start:
            return 0.0, f"blocked content detected: {signal}"

    length_score = min(1.0, len(text) / 5000)
    if query_terms:
        text_terms = set(_tokenize(text.lower()))
        term_density = len(query_terms & text_terms) / len(query_terms)
    else:
        term_density = 0.5
    return 0.4 * length_score + 0.6 * term_density, "passed"


@dataclass(frozen=True)
class Judgment:
    """A single accept/reject decision with an explicit reason (PAT-004)."""
    verdict: str        # "accept" | "reject"
    reason: str
    confidence: float
    judge: str          # "llm" | "heuristic" | "llm_fallback_heuristic"

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "judge": self.judge,
        }


class Judge(Protocol):
    """Protocol for snippet and content judgment."""
    name: str

    def judge_snippets(self, query: str, candidates: list[dict[str, str]]) -> list[Judgment]:
        """Judge a batch of search results {url, title, snippet}."""
        ...

    def judge_content(self, query: str, title: str, text: str, *, age_days: int | None = None) -> Judgment:
        """Judge scraped content quality/relevance before ingestion.

        ``age_days`` is a freshness signal (publication age), never a verdict
        by itself: the judge decides whether age makes the content stale.
        """
        ...


# ---------------------------------------------------------------------------
# HeuristicJudge — deterministic scaffold (cheap, no LLM)
# ---------------------------------------------------------------------------

class HeuristicJudge:
    """Deterministic token-overlap judgment. Cheap pre-filter and fallback."""

    name = "heuristic"

    def judge_snippets(self, query: str, candidates: list[dict[str, str]]) -> list[Judgment]:
        judgments = []
        for cand in candidates:
            score = _snippet_relevance(query, cand.get("title", ""), cand.get("snippet", ""))
            if score >= SNIPPET_RELEVANCE_THRESHOLD:
                judgments.append(Judgment("accept", f"snippet relevance {score:.2f}", score, self.name))
            else:
                judgments.append(Judgment("reject", f"snippet relevance {score:.2f} below threshold", score, self.name))
        return judgments

    def judge_content(self, query: str, title: str, text: str, *, age_days: int | None = None) -> Judgment:
        quality, reason = _content_quality(text, query)
        # Age is a recorded signal, never a rejection reason by itself:
        # for evergreen content (tutorials, docs) publication date does not
        # imply obsolescence — semantic staleness is the LLM's call.
        age_note = f", age {age_days}d" if age_days is not None else ""
        if quality >= CONTENT_QUALITY_THRESHOLD:
            return Judgment("accept", f"{reason} (quality {quality:.2f}{age_note})", quality, self.name)
        return Judgment("reject", f"{reason} (quality {quality:.2f}{age_note})", quality, self.name)


# ---------------------------------------------------------------------------
# LLMJudge — semantic judgment with robust fallback
# ---------------------------------------------------------------------------

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_MAX_CONTENT_CHARS = 4000
_MAX_SNIPPETS_PER_CALL = 10


def _extract_json(text: str) -> Any:
    """Extract the first JSON object or array from LLM output."""
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    for pattern, loader in ((_JSON_ARRAY_RE, list), (_JSON_OBJECT_RE, dict)):
        match = pattern.search(text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
    raise ValueError("no valid JSON found in LLM output")


def _normalize_verdict(value: object) -> str | None:
    if isinstance(value, str) and value.lower() in {"accept", "reject"}:
        return value.lower()
    return None


class LLMJudge:
    """Semantic judgment via a chat provider, with heuristic fallback.

    The provider must expose ``generate_chat(messages, *, max_new_tokens,
    temperature) -> object with .text and .error`` (ExL3Provider satisfies
    this). Judgment prompts demand JSON-only output; unparseable responses
    fall back to the heuristic judge so the pipeline never blocks on the LLM.
    """

    name = "llm"

    def __init__(
        self,
        provider: Any,
        *,
        max_judge_tokens: int = 900,
        temperature: float = 0.0,
        max_content_chars: int = _MAX_CONTENT_CHARS,
    ) -> None:
        self.provider = provider
        self.max_judge_tokens = max_judge_tokens
        self.temperature = temperature
        self.max_content_chars = max_content_chars
        self._heuristic = HeuristicJudge()
        # Debug/audit: why the last LLM call fell back to the heuristic judge
        self.last_fallback_reason: str | None = None

    # -- prompts -----------------------------------------------------------

    def _snippet_prompt(self, query: str, candidates: list[dict[str, str]]) -> str:
        lines = [
            "You are evaluating web search results for a research query.",
            f"Query: {query}",
            "",
            "For each result, judge whether the page is likely to contain",
            "substantive content that answers the query (a tutorial, docs,",
            "in-depth article). Reject paywalls, stubs, aggregators, and",
            "unrelated pages.",
            "",
            "Respond with ONLY a JSON array, one object per result, in order:",
            '[{"index": 0, "verdict": "accept", "reason": "<=20 words", "confidence": 0.9}]',
            "",
            "Results:",
        ]
        for i, cand in enumerate(candidates):
            title = (cand.get("title") or "")[:150]
            snippet = (cand.get("snippet") or "")[:300]
            lines.append(f"{i}. Title: {title}")
            lines.append(f"   Snippet: {snippet}")
        return "\n".join(lines)

    def _content_prompt(self, query: str, title: str, text: str, age_days: int | None = None) -> str:
        # Truncate at a sentence boundary so the judge never sees a cut
        # mid-sentence (a raw char cutoff reads as "incomplete content" and
        # causes false rejections).
        truncated = text[: self.max_content_chars]
        if len(text) > self.max_content_chars:
            last_stop = max(truncated.rfind(". "), truncated.rfind("\n"))
            if last_stop > self.max_content_chars // 2:
                truncated = truncated[: last_stop + 1]
            truncated += "\n[... content truncated for evaluation ...]"
        age_line = ""
        if age_days is not None:
            age_line = (
                f"Publication age: {age_days} days. Publication date alone does not "
                "make content stale — judge staleness from the content itself "
                "(outdated syntax, deprecated APIs, superseded versions).\n"
            )
        return (
            "You are evaluating scraped web content for a research query.\n"
            f"Query: {query}\n"
            f"Title: {title}\n"
            f"{age_line}"
            f"Content:\n{truncated}\n"
            "\n"
            "Judge whether this is substantive content that answers the query:\n"
            "- reject paywall teasers, cookie walls, JS-required stubs\n"
            "- reject content unrelated to the query\n"
            "- reject content that is substantively outdated for the query\n"
            "- accept complete, relevant, substantive content\n"
            "\n"
            "IMPORTANT: the content shown may be an excerpt of a longer page "
            "(marked '[... content truncated for evaluation ...]'). Judge only "
            "whether the visible portion indicates a substantive, relevant "
            "resource. Do NOT reject solely because the excerpt ends — a real "
            "page continues beyond the excerpt.\n"
            "\n"
            'Respond with ONLY JSON: {"verdict": "accept", "reason": "<=20 words", "confidence": 0.9}'
        )

    # -- judgment ----------------------------------------------------------

    def _call(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a precise content judge. Respond only with the requested JSON."},
            {"role": "user", "content": prompt},
        ]
        result = self.provider.generate_chat(
            messages, max_new_tokens=self.max_judge_tokens, temperature=self.temperature,
        )
        if getattr(result, "error", None):
            raise RuntimeError(f"provider error: {result.error}")
        return result.text

    def judge_snippets(self, query: str, candidates: list[dict[str, str]]) -> list[Judgment]:
        if not candidates:
            return []
        try:
            raw = self._call(self._snippet_prompt(query, candidates))
            parsed = _extract_json(raw)
            if not isinstance(parsed, list) or len(parsed) != len(candidates):
                raise ValueError(
                    f"judge returned wrong number of verdicts "
                    f"(got {len(parsed) if isinstance(parsed, list) else type(parsed).__name__}, "
                    f"expected {len(candidates)})"
                )
            judgments = []
            for i, item in enumerate(parsed):
                verdict = _normalize_verdict(item.get("verdict")) if isinstance(item, dict) else None
                if verdict is None:
                    raise ValueError(f"invalid verdict at index {i}")
                judgments.append(Judgment(
                    verdict=verdict,
                    reason=str(item.get("reason", ""))[:200],
                    confidence=float(item.get("confidence", 0.5)),
                    judge=self.name,
                ))
            return judgments
        except Exception as exc:
            # Fallback: heuristic judgment, flagged as fallback (PAT-004 traceability).
            # Record why the LLM path failed so fallbacks are diagnosable.
            self.last_fallback_reason = f"snippets: {type(exc).__name__}: {str(exc)[:150]}"
            fallback = self._heuristic.judge_snippets(query, candidates)
            return [Judgment(j.verdict, j.reason, j.confidence, "llm_fallback_heuristic") for j in fallback]

    def judge_content(self, query: str, title: str, text: str, *, age_days: int | None = None) -> Judgment:
        try:
            raw = self._call(self._content_prompt(query, title, text, age_days))
            parsed = _extract_json(raw)
            if not isinstance(parsed, dict):
                raise ValueError("judge did not return a JSON object")
            verdict = _normalize_verdict(parsed.get("verdict"))
            if verdict is None:
                raise ValueError("invalid verdict")
            return Judgment(
                verdict=verdict,
                reason=str(parsed.get("reason", ""))[:200],
                confidence=float(parsed.get("confidence", 0.5)),
                judge=self.name,
            )
        except Exception as exc:
            self.last_fallback_reason = f"content: {type(exc).__name__}: {str(exc)[:150]}"
            fallback = self._heuristic.judge_content(query, title, text, age_days=age_days)
            return Judgment(fallback.verdict, fallback.reason, fallback.confidence, "llm_fallback_heuristic")


__all__ = ["Judgment", "Judge", "HeuristicJudge", "LLMJudge"]
