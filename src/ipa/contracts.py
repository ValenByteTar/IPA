"""Dataclass materialization of the contract vocabulary.

These types are the in-process representation of the records defined in
contracts/contract_vocabulary.json.  External adapters MUST produce these
types; they never write authoritative state directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    content_hash: str
    source_uri: str
    mime_type: str
    original_filename: str
    byte_size: int
    received_at: str


@dataclass(frozen=True)
class SourceSpan:
    artifact_id: str
    page: int | None
    offset_start: int
    offset_end: int


@dataclass(frozen=True)
class CanonicalDocument:
    document_id: str
    pages: int
    elements: list[dict[str, Any]]
    source_spans: list[SourceSpan]
    text: str
    mime_type: str
    parser_id: str


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    content_hash: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_span: SourceSpan | None = None


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    score: float
    source_span: SourceSpan | None
    retrieval_backend: str


@dataclass(frozen=True)
class ParserResult:
    artifact_id: str
    parser_id: str
    status: str
    canonical_document: CanonicalDocument | None

