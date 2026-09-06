"""Domain-agnostic runtime contracts for bounded agentic investigations."""
from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Self

_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "declined"}
_STATUSES = {"pending", "running", *_TERMINAL_STATUSES}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp must be a non-empty ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be a valid ISO 8601 string") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_string_list(values: list[str], name: str) -> None:
    if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain non-empty strings")


def _validate_json(value: Any, name: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON serializable") from exc


def _validate_hash(value: str, name: str) -> None:
    if value and not _HASH_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be empty or a sha256 hash")


def _payload(data: dict[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{name} payload must be a dictionary")
    return dict(data)


@dataclass(frozen=True)
class QueryIR:
    raw_query: str
    intent: str = "informational"
    entities: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    topic_id: str | None = None
    report_id: str | None = None
    required_evidence: list[str] = field(default_factory=list)
    is_comparison: bool = False
    language: str = "und"

    def __post_init__(self) -> None:
        _require_text(self.raw_query, "raw_query")
        _require_text(self.intent, "intent")
        _validate_string_list(self.entities, "entities")
        _validate_string_list(self.required_evidence, "required_evidence")
        if not isinstance(self.constraints, dict):
            raise ValueError("constraints must be a dictionary")
        _validate_json(self.constraints, "constraints")
        if self.topic_id is not None:
            _require_text(self.topic_id, "topic_id")
        if self.report_id is not None:
            _require_text(self.report_id, "report_id")
        if not isinstance(self.is_comparison, bool):
            raise ValueError("is_comparison must be a boolean")
        _require_text(self.language, "language")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**_payload(data, "QueryIR"))


@dataclass(frozen=True)
class EvidenceHit:
    chunk_id: str
    document_id: str
    score: float = 0.0
    retrieval_stage: str = "initial"
    retrieval_backend: str = "unknown"
    source_ref: dict[str, Any] = field(default_factory=dict)
    source_span: dict[str, Any] | None = None
    text_hash: str = ""

    def __post_init__(self) -> None:
        _require_text(self.chunk_id, "chunk_id")
        _require_text(self.document_id, "document_id")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)) or not math.isfinite(self.score):
            raise ValueError("score must be a finite number")
        _require_text(self.retrieval_stage, "retrieval_stage")
        _require_text(self.retrieval_backend, "retrieval_backend")
        if not isinstance(self.source_ref, dict):
            raise ValueError("source_ref must be a dictionary")
        _validate_json(self.source_ref, "source_ref")
        if self.source_span is not None:
            if not isinstance(self.source_span, dict):
                raise ValueError("source_span must be a dictionary or None")
            start = self.source_span.get("offset_start")
            end = self.source_span.get("offset_end")
            if start is not None and (isinstance(start, bool) or not isinstance(start, int) or start < 0):
                raise ValueError("source_span offset_start must be a non-negative integer")
            if end is not None and (isinstance(end, bool) or not isinstance(end, int) or end < 0):
                raise ValueError("source_span offset_end must be a non-negative integer")
            if start is not None and end is not None and end < start:
                raise ValueError("source_span offsets must be ordered")
            _validate_json(self.source_span, "source_span")
        _validate_hash(self.text_hash, "text_hash")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**_payload(data, "EvidenceHit"))


@dataclass(frozen=True)
class EvidenceSet:
    query_ir: QueryIR
    hits: list[EvidenceHit] = field(default_factory=list)
    covered_requirements: list[str] = field(default_factory=list)
    missing_requirements: list[str] = field(default_factory=list)
    document_diversity: int = 0
    sufficiency: str = "insufficient"

    def __post_init__(self) -> None:
        if not isinstance(self.query_ir, QueryIR):
            raise ValueError("query_ir must be a QueryIR")
        if not isinstance(self.hits, list) or any(not isinstance(hit, EvidenceHit) for hit in self.hits):
            raise ValueError("hits must contain EvidenceHit records")
        _validate_string_list(self.covered_requirements, "covered_requirements")
        _validate_string_list(self.missing_requirements, "missing_requirements")
        if isinstance(self.document_diversity, bool) or not isinstance(self.document_diversity, int) or self.document_diversity < 0:
            raise ValueError("document_diversity must be a non-negative integer")
        if self.document_diversity > len({hit.document_id for hit in self.hits}):
            raise ValueError("document_diversity cannot exceed the number of represented documents")
        if self.sufficiency not in {"sufficient", "partial", "insufficient"}:
            raise ValueError("sufficiency must be sufficient, partial, or insufficient")
        overlap = set(self.covered_requirements) & set(self.missing_requirements)
        if overlap:
            raise ValueError("covered and missing requirements must not overlap")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        payload = _payload(data, "EvidenceSet")
        query_ir = payload.get("query_ir")
        if isinstance(query_ir, dict):
            payload["query_ir"] = QueryIR.from_dict(query_ir)
        hits = payload.get("hits", [])
        if isinstance(hits, list):
            payload["hits"] = [EvidenceHit.from_dict(hit) if isinstance(hit, dict) else hit for hit in hits]
        return cls(**payload)


@dataclass(frozen=True)
class ContextPackage:
    query_ir: QueryIR
    evidence: EvidenceSet
    citation_map: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    truncation_policy: str = "none"
    input_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.query_ir, QueryIR):
            raise ValueError("query_ir must be a QueryIR")
        if not isinstance(self.evidence, EvidenceSet):
            raise ValueError("evidence must be an EvidenceSet")
        if self.query_ir != self.evidence.query_ir:
            raise ValueError("context and evidence must use the same query_ir")
        if not isinstance(self.citation_map, dict) or any(not isinstance(key, str) for key in self.citation_map):
            raise ValueError("citation_map must be a dictionary with string keys")
        _validate_json(self.citation_map, "citation_map")
        if isinstance(self.token_count, bool) or not isinstance(self.token_count, int) or self.token_count < 0:
            raise ValueError("token_count must be a non-negative integer")
        _require_text(self.truncation_policy, "truncation_policy")
        _validate_hash(self.input_hash, "input_hash")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        payload = _payload(data, "ContextPackage")
        query_ir = payload.get("query_ir")
        evidence = payload.get("evidence")
        if isinstance(query_ir, dict):
            payload["query_ir"] = QueryIR.from_dict(query_ir)
        if isinstance(evidence, dict):
            payload["evidence"] = EvidenceSet.from_dict(evidence)
        return cls(**payload)


@dataclass(frozen=True)
class ExecutionBudget:
    max_iterations: int = 8
    max_retrieval_rounds: int = 2
    max_llm_calls: int = 1
    max_repairs: int = 0
    max_documents: int = 6
    max_chunks: int = 12
    max_context_tokens: int = 8192
    max_seconds: float = 120.0

    def __post_init__(self) -> None:
        positive = (
            "max_iterations",
            "max_retrieval_rounds",
            "max_llm_calls",
            "max_documents",
            "max_chunks",
            "max_context_tokens",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.max_repairs, bool) or not isinstance(self.max_repairs, int) or self.max_repairs < 0:
            raise ValueError("max_repairs must be a non-negative integer")
        if isinstance(self.max_seconds, bool) or not isinstance(self.max_seconds, (int, float)) or not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("max_seconds must be a positive finite number")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**_payload(data, "ExecutionBudget"))


@dataclass(frozen=True)
class TraceEvent:
    kind: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_utc_now)
    duration_ms: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.kind, "kind")
        if not isinstance(self.message, str):
            raise ValueError("message must be a string")
        if not isinstance(self.data, dict):
            raise ValueError("data must be a dictionary")
        _validate_json(self.data, "data")
        _timestamp(self.timestamp)
        if self.duration_ms is not None and (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, (int, float))
            or not math.isfinite(self.duration_ms)
            or self.duration_ms < 0
        ):
            raise ValueError("duration_ms must be a non-negative finite number or None")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**_payload(data, "TraceEvent"))


def _default_counters() -> dict[str, int]:
    return {
        "iterations": 0,
        "retrieval_rounds": 0,
        "llm_calls": 0,
        "repairs": 0,
        "documents": 0,
        "chunks": 0,
        "context_tokens": 0,
    }


@dataclass(frozen=True)
class TopicInvestigationState:
    question: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    report_id: str | None = None
    category_id: str | None = None
    query_ir: QueryIR | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    evidence_set: EvidenceSet | None = None
    context_package: ContextPackage | None = None
    answer: str = ""
    claims: list[dict[str, Any]] = field(default_factory=list)
    signals: list[dict[str, Any]] = field(default_factory=list)
    last_decision: dict[str, Any] | None = None
    budget: ExecutionBudget = field(default_factory=ExecutionBudget)
    counters: dict[str, int] = field(default_factory=_default_counters)
    phase_flags: dict[str, bool] = field(default_factory=dict)
    traces: list[TraceEvent] = field(default_factory=list)
    started_at: str = field(default_factory=_utc_now)
    completed_at: str | None = None
    status: str = "pending"

    def __post_init__(self) -> None:
        _require_text(self.question, "question")
        _require_text(self.run_id, "run_id")
        if self.report_id is not None:
            _require_text(self.report_id, "report_id")
        if self.category_id is not None:
            _require_text(self.category_id, "category_id")
        if self.query_ir is not None and not isinstance(self.query_ir, QueryIR):
            raise ValueError("query_ir must be a QueryIR or None")
        if self.evidence_set is not None and not isinstance(self.evidence_set, EvidenceSet):
            raise ValueError("evidence_set must be an EvidenceSet or None")
        if self.context_package is not None and not isinstance(self.context_package, ContextPackage):
            raise ValueError("context_package must be a ContextPackage or None")
        if self.query_ir is not None and self.evidence_set is not None and self.query_ir != self.evidence_set.query_ir:
            raise ValueError("state and evidence_set must use the same query_ir")
        if self.query_ir is not None and self.context_package is not None and self.query_ir != self.context_package.query_ir:
            raise ValueError("state and context_package must use the same query_ir")
        for value, name in ((self.results, "results"), (self.claims, "claims"), (self.signals, "signals")):
            if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
                raise ValueError(f"{name} must contain dictionaries")
            _validate_json(value, name)
        if not isinstance(self.answer, str):
            raise ValueError("answer must be a string")
        if self.last_decision is not None:
            if not isinstance(self.last_decision, dict):
                raise ValueError("last_decision must be a dictionary or None")
            _validate_json(self.last_decision, "last_decision")
        if not isinstance(self.budget, ExecutionBudget):
            raise ValueError("budget must be an ExecutionBudget")
        if not isinstance(self.counters, dict) or any(
            not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int) or value < 0
            for key, value in self.counters.items()
        ):
            raise ValueError("counters must map strings to non-negative integers")
        limits = {
            "iterations": self.budget.max_iterations,
            "retrieval_rounds": self.budget.max_retrieval_rounds,
            "llm_calls": self.budget.max_llm_calls,
            "repairs": self.budget.max_repairs,
            "documents": self.budget.max_documents,
            "chunks": self.budget.max_chunks,
            "context_tokens": self.budget.max_context_tokens,
        }
        for name, limit in limits.items():
            if self.counters.get(name, 0) > limit:
                raise ValueError(f"counter {name} exceeds its execution budget")
        if not isinstance(self.phase_flags, dict) or any(
            not isinstance(key, str) or not isinstance(value, bool) for key, value in self.phase_flags.items()
        ):
            raise ValueError("phase_flags must map strings to booleans")
        if not isinstance(self.traces, list) or any(not isinstance(trace, TraceEvent) for trace in self.traces):
            raise ValueError("traces must contain TraceEvent records")
        started = _timestamp(self.started_at)
        if self.status not in _STATUSES:
            raise ValueError(f"status must be one of {sorted(_STATUSES)}")
        if self.status in _TERMINAL_STATUSES and self.completed_at is None:
            raise ValueError("terminal states require completed_at")
        if self.status not in _TERMINAL_STATUSES and self.completed_at is not None:
            raise ValueError("non-terminal states cannot have completed_at")
        if self.completed_at is not None and started > _timestamp(self.completed_at):
            raise ValueError("started_at must be <= completed_at")

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        payload = _payload(data, "TopicInvestigationState")
        nested = {
            "query_ir": QueryIR,
            "evidence_set": EvidenceSet,
            "context_package": ContextPackage,
            "budget": ExecutionBudget,
        }
        for key, record_type in nested.items():
            value = payload.get(key)
            if isinstance(value, dict):
                payload[key] = record_type.from_dict(value)
        traces = payload.get("traces", [])
        if isinstance(traces, list):
            payload["traces"] = [TraceEvent.from_dict(trace) if isinstance(trace, dict) else trace for trace in traces]
        return cls(**payload)

