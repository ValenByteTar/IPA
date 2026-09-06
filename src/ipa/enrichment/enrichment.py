"""E9 â€” Semantic enrichment strategies.

Three strategies for enriching chunks with LLM-generated content:
  1. Synthetic queries: generate 3-5 questions answerable by the chunk
  2. Claim extraction: extract 3-5 atomic facts from the chunk
  3. Summary: generate a 1-2 sentence summary of the chunk

Each strategy produces an "enriched text" that combines the original
chunk text with the LLM output.  This enriched text is then indexed
and retrieval quality is compared against the baseline (original text only).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ipa.enrichment.ollama_adapter import OllamaAdapter, LLMResponse


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------

@dataclass
class EnrichmentResult:
    """Result of enriching a single chunk."""
    chunk_id: str
    strategy: str
    enriched_text: str
    llm_output: str
    latency_seconds: float
    error: str | None = None


type EnrichmentStrategy = Callable[[str, str, OllamaAdapter], EnrichmentResult]
"""Function signature: (chunk_id, chunk_text, llm) -> EnrichmentResult"""


# ---------------------------------------------------------------------------
# Strategy 1: Synthetic queries
# ---------------------------------------------------------------------------

_SYNTH_QUERIES_SYSTEM = "You are a search query generator. Generate realistic questions that a user would ask to find this text. Output one question per line, no numbering, no preamble."

_SYNTH_QUERIES_PROMPT = """Generate 3 questions that this text would answer. Output one question per line.

Text:
\"\"\"
{text}
\"\"\""""


def enrich_synthetic_queries(
    chunk_id: str,
    chunk_text: str,
    llm: OllamaAdapter,
) -> EnrichmentResult:
    """Generate synthetic queries and prepend them to the chunk text."""
    prompt = _SYNTH_QUERIES_PROMPT.format(text=chunk_text[:1500])
    try:
        resp = llm.generate(prompt, system=_SYNTH_QUERIES_SYSTEM)
        queries = _parse_lines(resp.text)
        if not queries:
            return EnrichmentResult(
                chunk_id=chunk_id, strategy="synthetic_queries",
                enriched_text=chunk_text, llm_output=resp.text,
                latency_seconds=resp.latency_seconds, error="no_queries_parsed",
            )
        # Prepend queries as a "synthetic questions" section.
        query_block = "[Synthetic questions]\n" + "\n".join(f"Q: {q}" for q in queries)
        enriched = f"{query_block}\n\n{chunk_text}"
        return EnrichmentResult(
            chunk_id=chunk_id, strategy="synthetic_queries",
            enriched_text=enriched, llm_output=resp.text,
            latency_seconds=resp.latency_seconds,
        )
    except Exception as e:
        return EnrichmentResult(
            chunk_id=chunk_id, strategy="synthetic_queries",
            enriched_text=chunk_text, llm_output="",
            latency_seconds=0.0, error=str(e),
        )


# ---------------------------------------------------------------------------
# Strategy 2: Claim extraction
# ---------------------------------------------------------------------------

_CLAIMS_SYSTEM = "You are a fact extraction system. Extract atomic, verifiable claims from the text. One claim per line. No numbering, no preamble, no commentary."

_CLAIMS_PROMPT = """Extract 3-5 key facts from this text. Each fact should be a single, self-contained sentence.

Text:
\"\"\"
{text}
\"\"\""""


def enrich_claim_extraction(
    chunk_id: str,
    chunk_text: str,
    llm: OllamaAdapter,
) -> EnrichmentResult:
    """Extract atomic claims and append them to the chunk text."""
    prompt = _CLAIMS_PROMPT.format(text=chunk_text[:1500])
    try:
        resp = llm.generate(prompt, system=_CLAIMS_SYSTEM)
        claims = _parse_lines(resp.text)
        if not claims:
            return EnrichmentResult(
                chunk_id=chunk_id, strategy="claim_extraction",
                enriched_text=chunk_text, llm_output=resp.text,
                latency_seconds=resp.latency_seconds, error="no_claims_parsed",
            )
        claim_block = "[Key facts]\n" + "\n".join(f"- {c}" for c in claims)
        enriched = f"{chunk_text}\n\n{claim_block}"
        return EnrichmentResult(
            chunk_id=chunk_id, strategy="claim_extraction",
            enriched_text=enriched, llm_output=resp.text,
            latency_seconds=resp.latency_seconds,
        )
    except Exception as e:
        return EnrichmentResult(
            chunk_id=chunk_id, strategy="claim_extraction",
            enriched_text=chunk_text, llm_output="",
            latency_seconds=0.0, error=str(e),
        )


# ---------------------------------------------------------------------------
# Strategy 3: Summary
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM = "You are a summarization system. Produce a concise summary in 1-2 sentences. No preamble."

_SUMMARY_PROMPT = """Summarize this text in 1-2 sentences.

Text:
\"\"\"
{text}
\"\"\""""


def enrich_summary(
    chunk_id: str,
    chunk_text: str,
    llm: OllamaAdapter,
) -> EnrichmentResult:
    """Generate a summary and prepend it to the chunk text."""
    prompt = _SUMMARY_PROMPT.format(text=chunk_text[:1500])
    try:
        resp = llm.generate(prompt, system=_SUMMARY_SYSTEM)
        summary = resp.text.strip()
        if not summary:
            return EnrichmentResult(
                chunk_id=chunk_id, strategy="summary",
                enriched_text=chunk_text, llm_output=resp.text,
                latency_seconds=resp.latency_seconds, error="empty_summary",
            )
        enriched = f"[Summary] {summary}\n\n{chunk_text}"
        return EnrichmentResult(
            chunk_id=chunk_id, strategy="summary",
            enriched_text=enriched, llm_output=summary,
            latency_seconds=resp.latency_seconds,
        )
    except Exception as e:
        return EnrichmentResult(
            chunk_id=chunk_id, strategy="summary",
            enriched_text=chunk_text, llm_output="",
            latency_seconds=0.0, error=str(e),
        )


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, EnrichmentStrategy] = {
    "synthetic_queries": enrich_synthetic_queries,
    "claim_extraction": enrich_claim_extraction,
    "summary": enrich_summary,
}


# ---------------------------------------------------------------------------
# Natural language query generation (for evaluation, not enrichment)
# ---------------------------------------------------------------------------

_NL_QUERY_SYSTEM = "You are a search engine user. Generate a realistic natural language question that this text would answer. Use your own words, not the exact words from the text. Output only the question, no preamble."

_NL_QUERY_PROMPT = """A user is searching for information. Generate ONE natural language question that this text answers. Use different words than the text itself â€” paraphrase the concept.

Text:
\"\"\"
{text}
\"\"\""""


def generate_nl_query(
    chunk_text: str,
    llm: OllamaAdapter,
) -> tuple[str, float]:
    """Generate a natural language query for a chunk.

    Unlike synthetic_queries (which uses the chunk's own terms), this
    generates a query with DIFFERENT words â€” simulating a real user who
    doesn't know the exact terminology.

    Returns (query_text, latency_seconds).
    """
    prompt = _NL_QUERY_PROMPT.format(text=chunk_text[:1500])
    try:
        resp = llm.generate(prompt, system=_NL_QUERY_SYSTEM)
        query = resp.text.strip().strip('"').strip("?").strip() + "?"
        return query, resp.latency_seconds
    except Exception as e:
        return "", 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_lines(text: str) -> list[str]:
    """Parse LLM output into clean lines, stripping numbering and prefixes."""
    lines = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip common numbering prefixes: "1.", "1)", "- ", "* "
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        line = re.sub(r'^[-*]\s*', '', line)
        line = line.strip()
        if line and len(line) > 5:  # filter trivially short lines
            lines.append(line)
    return lines

