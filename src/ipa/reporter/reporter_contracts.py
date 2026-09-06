"""Runtime contracts and helpers for Reporter Agent artifacts."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ReporterDecision(StrEnum):
    PROMOTE = "promote"
    REPORTER_ONLY = "reporter_only"
    DEFER = "defer"
    IRRELEVANT = "irrelevant"
    DUPLICATE = "duplicate"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"
    REVIEWED = "reviewed"


class ReportStatus(StrEnum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    PUBLISHED = "published"


class TopicEvolution(StrEnum):
    NEW = "new"
    STABLE = "stable"
    GROWING = "growing"
    DECLINING = "declining"
    SPLIT = "split"
    MERGED = "merged"
    DISAPPEARED = "disappeared"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ReporterPeriod:
    start: str
    end: str
    label: str


@dataclass(frozen=True)
class ScoreBundle:
    relevance: float
    novelty: float
    source_quality: float
    impact: float
    depth: float = 0.0
    actionability: float = 0.0

    def __post_init__(self) -> None:
        if any(not 0 <= value <= 1 for value in asdict(self).values()):
            raise ValueError("reporter scores must be between 0 and 1")


@dataclass(frozen=True)
class ReporterDocumentDecision:
    decision_id: str
    report_id: str
    document_id: str
    artifact_id: str
    decision: ReporterDecision
    scores: ScoreBundle
    reason: str
    evidence: list[dict[str, Any]]
    generation: dict[str, Any]
    review_status: ReviewStatus = ReviewStatus.PENDING
    content_type: str | None = None
    duplicate_of: str | None = None
    approval: dict[str, Any] | None = None
    field_origins: dict[str, str] = field(default_factory=lambda: {
        "decision": "generated",
        "scores": "generated",
        "reason": "generated",
        "evidence": "source",
        "review_status": "system",
    })

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TopicLink:
    current_category_id: str
    previous_category_id: str | None
    relation: TopicEvolution
    score: float
    evidence: list[dict[str, Any]]
    reason: str
    generation: dict[str, Any]
    review_status: ReviewStatus = ReviewStatus.PENDING
    field_origins: dict[str, str] = field(default_factory=lambda: {
        "relation": "generated",
        "score": "generated",
        "reason": "generated",
    })

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_hash(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def generation_provenance(input_hash: str, model_fingerprint: str = "deterministic-reporter-v1", prompt_fingerprint: str | None = None) -> dict[str, Any]:
    from datetime import datetime, timezone

    return {
        "generator": "ipa.reporter",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input_hash": input_hash,
        "model_fingerprint": model_fingerprint,
        "prompt_fingerprint": prompt_fingerprint,
    }

