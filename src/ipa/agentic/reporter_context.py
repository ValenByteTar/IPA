"""Deterministic, budgeted context construction for Reporter evidence."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from typing import Any

from ipa.agentic.agentic_contracts import ContextPackage, EvidenceSet

ChunkTextSource = Mapping[str, str] | Callable[[str], str | None]
TokenEstimator = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Return a conservative dependency-free token estimate.

    Four UTF-8 characters per token is a common approximation, with one token
    reserved for every non-empty fragment.
    """

    if not text:
        return 0
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def _resolve_text(source: ChunkTextSource, chunk_id: str) -> str | None:
    value = source(chunk_id) if callable(source) else source.get(chunk_id)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"chunk text for {chunk_id!r} must be a string or None")
    return value


class ReporterContextBuilder:
    """Build a closed citation map without performing retrieval.

    Ranked whole chunks are accepted while both budgets permit.  Missing text
    and over-budget chunks are omitted, and that decision is recorded through
    ``truncation_policy``.  The text resolver and token estimator are injected,
    keeping the component independent from storage and tokenizer packages.
    """

    name = "reporter_context"

    def __init__(
        self,
        chunk_text: ChunkTextSource,
        *,
        max_chunks: int = 12,
        max_context_tokens: int = 8192,
        token_estimator: TokenEstimator = estimate_tokens,
    ) -> None:
        if isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or max_chunks <= 0:
            raise ValueError("max_chunks must be a positive integer")
        if (
            isinstance(max_context_tokens, bool)
            or not isinstance(max_context_tokens, int)
            or max_context_tokens <= 0
        ):
            raise ValueError("max_context_tokens must be a positive integer")
        if not callable(token_estimator):
            raise TypeError("token_estimator must be callable")
        self.chunk_text = chunk_text
        self.max_chunks = max_chunks
        self.max_context_tokens = max_context_tokens
        self.token_estimator = token_estimator

    def build(self, evidence: EvidenceSet) -> ContextPackage:
        if not isinstance(evidence, EvidenceSet):
            raise TypeError("evidence must be an EvidenceSet")

        citation_map: dict[str, Any] = {}
        document_numbers: dict[str, int] = {}
        document_fragments: dict[str, int] = {}
        total_tokens = 0
        omitted = False

        for hit in evidence.hits:
            if len(citation_map) >= self.max_chunks:
                omitted = True
                break
            text = _resolve_text(self.chunk_text, hit.chunk_id)
            if text is None:
                omitted = True
                continue
            raw_tokens = self.token_estimator(text)
            if isinstance(raw_tokens, bool) or not isinstance(raw_tokens, int) or raw_tokens < 0:
                raise ValueError("token_estimator must return a non-negative integer")
            if total_tokens + raw_tokens > self.max_context_tokens:
                omitted = True
                continue

            document_number = document_numbers.setdefault(hit.document_id, len(document_numbers) + 1)
            fragment_number = document_fragments.get(hit.document_id, 0) + 1
            document_fragments[hit.document_id] = fragment_number
            citation_id = f"[Doc {document_number}, fragment {fragment_number}]"
            citation_map[citation_id] = {
                "document_id": hit.document_id,
                "chunk_id": hit.chunk_id,
                "text": text,
                "score": hit.score,
                "retrieval_stage": hit.retrieval_stage,
                "retrieval_backend": hit.retrieval_backend,
                "source_ref": hit.source_ref,
                "source_span": hit.source_span,
                "text_hash": hit.text_hash,
            }
            total_tokens += raw_tokens

        hash_payload = {
            "query_ir": evidence.query_ir.to_dict(),
            "citations": citation_map,
            "token_count": total_tokens,
        }
        encoded = json.dumps(
            hash_payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return ContextPackage(
            query_ir=evidence.query_ir,
            evidence=evidence,
            citation_map=citation_map,
            token_count=total_tokens,
            truncation_policy="preserve-ranked-whole-chunks" if omitted else "none",
            input_hash="sha256:" + hashlib.sha256(encoded).hexdigest(),
        )

    __call__ = build


def build_context(
    evidence: EvidenceSet,
    chunk_text: ChunkTextSource,
    *,
    max_chunks: int = 12,
    max_context_tokens: int = 8192,
    token_estimator: TokenEstimator = estimate_tokens,
) -> ContextPackage:
    """Functional context-building entry point."""

    return ReporterContextBuilder(
        chunk_text,
        max_chunks=max_chunks,
        max_context_tokens=max_context_tokens,
        token_estimator=token_estimator,
    ).build(evidence)


__all__ = [
    "ChunkTextSource",
    "ReporterContextBuilder",
    "TokenEstimator",
    "build_context",
    "estimate_tokens",
]

