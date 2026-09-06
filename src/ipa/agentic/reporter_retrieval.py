"""Bounded lexical retrieval adapter for Reporter investigations.

This component only retrieves and classifies evidence.  It deliberately knows
nothing about context construction, prompting, generation, or orchestration.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Mapping, Protocol

from ipa.agentic.agentic_contracts import EvidenceHit, EvidenceSet, QueryIR


class SearchBackend(Protocol):
    """The part of ``TantivyIndex`` used by the retriever."""

    def search(self, query: str, limit: int = 10) -> list[Any]: ...


class ChunkStore(Protocol):
    """The part of ``DocumentStore`` used by the retriever."""

    def get_chunk(self, chunk_id: str) -> Any | None: ...

    def get_chunks(self, document_id: str) -> Iterable[Any]: ...


_WORD = re.compile(r"\w+", re.UNICODE)


def _value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _positive_int(value: Any, default: int, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return default
    return min(value, maximum)


def _document_scope(query_ir: QueryIR) -> list[str]:
    raw = query_ir.constraints.get("document_ids", [])
    if not isinstance(raw, list):
        return []
    return list(dict.fromkeys(value.strip() for value in raw if isinstance(value, str) and value.strip()))


def _span_dict(span: Any) -> dict[str, Any] | None:
    if span is None:
        return None
    if isinstance(span, Mapping):
        return dict(span)
    if is_dataclass(span):
        return asdict(span)
    names = ("artifact_id", "page", "offset_start", "offset_end")
    result = {name: getattr(span, name) for name in names if hasattr(span, name)}
    return result or None


def _source_ref(chunk: Any, span: dict[str, Any] | None) -> dict[str, Any]:
    metadata = _value(chunk, "metadata", {})
    result: dict[str, Any] = {}
    if isinstance(metadata, Mapping):
        embedded = metadata.get("source_ref")
        if isinstance(embedded, Mapping):
            result.update(embedded)
        for name in ("artifact_id", "source_uri", "canonical_url", "title"):
            if name in metadata and metadata[name] is not None:
                result[name] = metadata[name]
    if span and span.get("artifact_id") is not None:
        result.setdefault("artifact_id", span["artifact_id"])
    return result


def _normal(text: str) -> str:
    return " ".join(_WORD.findall(text.casefold()))


def _covers(text: str, requirement: str) -> bool:
    normalized_text = _normal(text)
    normalized_requirement = _normal(requirement)
    if not normalized_requirement:
        return False
    if normalized_requirement in normalized_text:
        return True
    return set(normalized_requirement.split()).issubset(normalized_text.split())


def _lexical_score(text: str, query: str, requirements: list[str]) -> float:
    terms = set(_normal(query).split())
    terms.update(term for requirement in requirements for term in _normal(requirement).split())
    if not terms:
        return 0.0
    words = set(_normal(text).split())
    return len(terms & words) / len(terms)


class ReporterRetriever:
    """Adapt a lexical index and chunk store into ``EvidenceSet`` records.

    One bounded backend search is always performed first.  With an explicit
    document scope, candidates are then selected only from those documents and
    supplemented from a bounded slice of each scoped document.  Consequently a
    permissive or unfiltered backend cannot leak out-of-scope chunks.
    """

    name = "reporter_retrieval"

    def __init__(
        self,
        index: SearchBackend,
        store: ChunkStore,
        *,
        max_candidates: int = 40,
        max_chunks: int = 12,
    ) -> None:
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates <= 0:
            raise ValueError("max_candidates must be a positive integer")
        if isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or max_chunks <= 0:
            raise ValueError("max_chunks must be a positive integer")
        self.index = index
        self.store = store
        self.max_candidates = max_candidates
        self.max_chunks = max_chunks

    def retrieve(self, query_ir: QueryIR) -> EvidenceSet:
        if not isinstance(query_ir, QueryIR):
            raise TypeError("query_ir must be a QueryIR")
        top_k = _positive_int(query_ir.constraints.get("top_k"), self.max_chunks, maximum=self.max_chunks)
        scope = _document_scope(query_ir)
        allowed = set(scope)

        backend_hits = self.index.search(query_ir.raw_query, limit=self.max_candidates)
        candidates: dict[str, tuple[Any, float, str]] = {}
        for raw_hit in list(backend_hits)[: self.max_candidates]:
            chunk_id = _value(raw_hit, "chunk_id", "")
            if not isinstance(chunk_id, str) or not chunk_id or chunk_id in candidates:
                continue
            chunk = self.store.get_chunk(chunk_id)
            if chunk is None:
                continue
            document_id = _value(chunk, "document_id", "")
            if allowed and document_id not in allowed:
                continue
            raw_score = _value(raw_hit, "score", 0.0)
            score = float(raw_score) if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool) else 0.0
            if not math.isfinite(score):
                score = 0.0
            backend = _value(raw_hit, "retrieval_backend", type(self.index).__name__)
            candidates[chunk_id] = (chunk, score, str(backend or type(self.index).__name__))

        if scope:
            # Give every selected document a bounded opportunity to contribute,
            # even when global BM25 ranking did not place it in the first window.
            per_document = max(1, math.ceil(self.max_candidates / len(scope)))
            for document_id in scope:
                for position, chunk in enumerate(self.store.get_chunks(document_id)):
                    if position >= per_document or len(candidates) >= self.max_candidates:
                        break
                    if _value(chunk, "document_id", "") not in allowed:
                        continue
                    chunk_id = _value(chunk, "chunk_id", "")
                    if not isinstance(chunk_id, str) or not chunk_id or chunk_id in candidates:
                        continue
                    score = _lexical_score(
                        str(_value(chunk, "text", "")), query_ir.raw_query, query_ir.required_evidence
                    )
                    candidates[chunk_id] = (chunk, score, type(self.index).__name__)

        ranked = sorted(candidates.values(), key=lambda item: (-item[1], str(_value(item[0], "chunk_id", ""))))
        selected = ranked[:top_k]
        stage = "scoped" if scope else "initial"
        evidence_hits: list[EvidenceHit] = []
        texts: list[str] = []
        for chunk, score, backend in selected:
            text = str(_value(chunk, "text", ""))
            document_id = str(_value(chunk, "document_id", ""))
            if allowed and document_id not in allowed:
                continue
            span = _span_dict(_value(chunk, "source_span"))
            evidence_hits.append(
                EvidenceHit(
                    chunk_id=str(_value(chunk, "chunk_id")),
                    document_id=document_id,
                    score=score,
                    retrieval_stage=stage,
                    retrieval_backend=backend,
                    source_ref=_source_ref(chunk, span),
                    source_span=span,
                    text_hash="sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            )
            texts.append(text)

        covered = [
            requirement
            for requirement in query_ir.required_evidence
            if any(_covers(text, requirement) for text in texts)
        ]
        missing = [requirement for requirement in query_ir.required_evidence if requirement not in covered]
        if not evidence_hits:
            sufficiency = "insufficient"
        elif not missing:
            sufficiency = "sufficient"
        elif covered:
            sufficiency = "partial"
        else:
            sufficiency = "insufficient"
        return EvidenceSet(
            query_ir=query_ir,
            hits=evidence_hits,
            covered_requirements=covered,
            missing_requirements=missing,
            document_diversity=len({hit.document_id for hit in evidence_hits}),
            sufficiency=sufficiency,
        )

    __call__ = retrieve


def retrieve_evidence(
    query_ir: QueryIR,
    index: SearchBackend,
    store: ChunkStore,
    *,
    max_candidates: int = 40,
    max_chunks: int = 12,
) -> EvidenceSet:
    """Functional entry point for callers that do not need a reusable adapter."""

    return ReporterRetriever(
        index, store, max_candidates=max_candidates, max_chunks=max_chunks
    ).retrieve(query_ir)


__all__ = ["ChunkStore", "ReporterRetriever", "SearchBackend", "retrieve_evidence"]

