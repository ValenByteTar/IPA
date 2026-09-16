"""Runtime data contracts for Tutor Agent records."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class SourceType(StrEnum):
    ARTIFACT = "artifact"
    DOCUMENT = "document"
    CHUNK = "chunk"
    ASSESSMENT = "assessment"
    USER_STATEMENT = "user_statement"


class FieldOrigin(StrEnum):
    SOURCE = "source"
    USER = "user"
    GENERATED = "generated"
    SYSTEM = "system"
    MIXED = "mixed"
    USER_OR_GENERATED = "user_or_generated"
    USER_OR_SYSTEM_POLICY = "user_or_system_policy"
    SYSTEM_POLICY = "system_policy"
    GENERATED_OR_USER = "generated_or_user"


class HumanApprovalDecision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"


class MasteryStatus(StrEnum):
    UNKNOWN = "unknown"
    EXPOSED = "exposed"
    UNDERSTOOD = "understood"
    APPLIED = "applied"
    NEEDS_REVIEW = "needs_review"
    MISCONCEPTION = "misconception"


class LearningGoalStatus(StrEnum):
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ConceptStatus(StrEnum):
    DRAFT = "draft"
    VALIDATED = "validated"
    DEPRECATED = "deprecated"


class RoadmapStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    ACTIVE = "active"
    COMPLETED = "completed"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


class ResearchStatus(StrEnum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AssessmentType(StrEnum):
    RETRIEVAL = "retrieval"
    EXPLANATION = "explanation"
    APPLICATION = "application"
    CRITIQUE = "critique"
    TRANSFER = "transfer"
    MISCONCEPTION_CORRECTION = "misconception_correction"


class RecommendedAction(StrEnum):
    ADVANCE = "advance"
    GUIDED_RETRY = "guided_retry"
    PRACTICAL_RETRY = "practical_retry"
    REVIEW_PREREQUISITE = "review_prerequisite"
    CORRECT_MISCONCEPTION = "correct_misconception"
    RESEARCH_GAP = "research_gap"
    HUMAN_REVIEW = "human_review"


class ResearchTrigger(StrEnum):
    INSUFFICIENT_SOURCES = "insufficient_sources"
    LOW_SOURCE_QUALITY = "low_source_quality"
    SOURCE_CONTRADICTION = "source_contradiction"
    MISSING_PRIMARY_SOURCE = "missing_primary_source"
    STALE_INFORMATION = "stale_information"
    USER_DEEPENING_REQUEST = "user_deepening_request"
    UNKNOWN_PREREQUISITE = "unknown_prerequisite"
    ASSESSMENT_GAP = "assessment_gap"


_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_DOMAIN_PATTERN = re.compile(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _validate_time_order(created_at: str, updated_at: str) -> None:
    if _timestamp(created_at) > _timestamp(updated_at):
        raise ValueError("created_at must be <= updated_at")


def _validate_origins(
    origins: dict[str, str], required: set[str], generation: GenerationProvenance | None
) -> None:
    missing = required - origins.keys()
    if missing:
        raise ValueError(f"field_origins missing required keys: {sorted(missing)}")
    allowed = {origin.value for origin in FieldOrigin}
    invalid = set(origins.values()) - allowed
    if invalid:
        raise ValueError(f"invalid field origins: {sorted(invalid)}")
    if any(origin in {FieldOrigin.GENERATED, FieldOrigin.MIXED} for origin in origins.values()) and generation is None:
        raise ValueError("generated or mixed fields require generation provenance")


@dataclass(frozen=True)
class SourceSpanRef:
    offset_start: int
    offset_end: int
    page: int | None = None

    def __post_init__(self) -> None:
        if self.offset_start < 0 or self.offset_end < self.offset_start:
            raise ValueError("source span offsets must be non-negative and ordered")
        if self.page is not None and self.page < 1:
            raise ValueError("source span page must be >= 1")


@dataclass(frozen=True)
class SourceRef:
    source_id: str
    source_type: SourceType
    source_span: SourceSpanRef | None = None
    content_hash: str | None = None

    def __post_init__(self) -> None:
        if self.source_type not in set(SourceType):
            raise ValueError("invalid source type")
        if self.content_hash is not None and not _HASH_PATTERN.fullmatch(self.content_hash):
            raise ValueError("content_hash must be a sha256 hash")


@dataclass(frozen=True)
class GenerationProvenance:
    generator: str
    generated_at: str
    input_hash: str
    model_fingerprint: str
    prompt_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not _HASH_PATTERN.fullmatch(self.input_hash):
            raise ValueError("input_hash must be a sha256 hash")
        if not self.generator or not self.model_fingerprint:
            raise ValueError("generation provenance requires generator and model fingerprint")


@dataclass(frozen=True)
class HumanApproval:
    decision: HumanApprovalDecision
    decided_at: str
    decided_by: str
    note: str | None = None

    def __post_init__(self) -> None:
        if self.decision not in set(HumanApprovalDecision):
            raise ValueError("invalid human approval decision")
        if not self.decided_by:
            raise ValueError("human approval requires decided_by")

    @property
    def approved(self) -> bool:
        return self.decision == HumanApprovalDecision.APPROVED


@dataclass(frozen=True)
class LearningGoal:
    goal_id: str
    title: str
    description: str
    status: LearningGoalStatus
    success_criteria: list[str]
    created_at: str
    updated_at: str
    approval: HumanApproval | None
    field_origins: dict[str, str]
    constraints: list[str] = field(default_factory=list)
    generation: GenerationProvenance | None = None

    def __post_init__(self) -> None:
        if not self.success_criteria:
            raise ValueError("a learning goal requires at least one success criterion")
        _validate_time_order(self.created_at, self.updated_at)
        _validate_origins(
            self.field_origins, {"title", "description", "success_criteria"}, self.generation
        )
        if self.status in {LearningGoalStatus.CONFIRMED, LearningGoalStatus.ACTIVE, LearningGoalStatus.COMPLETED}:
            if self.approval is None or not self.approval.approved:
                raise ValueError("confirmed goal states require human approval")


@dataclass(frozen=True)
class Concept:
    concept_id: str
    title: str
    definition: str
    status: ConceptStatus
    difficulty: float
    prerequisite_ids: list[str]
    learning_objectives: list[str]
    mastery_criteria: list[str]
    source_refs: list[SourceRef]
    created_at: str
    updated_at: str
    generation: GenerationProvenance | None
    field_origins: dict[str, str]
    common_misconceptions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.concept_id in self.prerequisite_ids:
            raise ValueError("a concept cannot be its own prerequisite")
        if not 0 <= self.difficulty <= 1:
            raise ValueError("difficulty must be between 0 and 1")
        if not self.source_refs:
            raise ValueError("a concept requires source references")
        _validate_time_order(self.created_at, self.updated_at)
        _validate_origins(
            self.field_origins,
            {"title", "definition", "difficulty", "prerequisite_ids", "learning_objectives", "mastery_criteria"},
            self.generation,
        )


@dataclass(frozen=True)
class RoadmapUnit:
    unit_id: str
    order: int
    concept_id: str
    reason: str
    estimated_effort_minutes: int
    source_refs: list[SourceRef]
    assessment_types: list[AssessmentType]

    def __post_init__(self) -> None:
        if not 5 <= self.estimated_effort_minutes <= 1440:
            raise ValueError("estimated effort must be between 5 and 1440 minutes")
        if not self.source_refs:
            raise ValueError("a roadmap unit requires source references")
        if not self.assessment_types or len(set(self.assessment_types)) != len(self.assessment_types):
            raise ValueError("assessment types must be non-empty and unique")


@dataclass(frozen=True)
class Roadmap:
    roadmap_id: str
    goal_id: str
    version: int
    status: RoadmapStatus
    units: list[RoadmapUnit]
    assumptions: list[str]
    uncertainties: list[str]
    change_reason: str | None
    previous_roadmap_id: str | None
    created_at: str
    approval: HumanApproval | None
    generation: GenerationProvenance
    field_origins: dict[str, str]

    def __post_init__(self) -> None:
        if not 3 <= len(self.units) <= 7:
            raise ValueError("a roadmap requires 3 to 7 units")
        orders = sorted(unit.order for unit in self.units)
        if orders != list(range(1, len(self.units) + 1)):
            raise ValueError("roadmap unit order must be contiguous starting at 1")
        if len({unit.unit_id for unit in self.units}) != len(self.units):
            raise ValueError("roadmap unit IDs must be unique")
        if len({unit.concept_id for unit in self.units}) != len(self.units):
            raise ValueError("roadmap concept IDs must be unique")
        if self.version > 1 and (
            not self.previous_roadmap_id or not self.change_reason or not self.change_reason.strip()
        ):
            raise ValueError("roadmap versions after 1 require predecessor and change reason")
        _validate_origins(
            self.field_origins,
            {"goal_id", "units", "assumptions", "uncertainties", "change_reason"},
            self.generation,
        )
        if self.status in {RoadmapStatus.APPROVED, RoadmapStatus.ACTIVE, RoadmapStatus.COMPLETED, RoadmapStatus.SUPERSEDED}:
            if self.approval is None or not self.approval.approved:
                raise ValueError("approved roadmap states require human approval")


@dataclass(frozen=True)
class MisconceptionEvidence:
    claim: str
    correction: str
    evidence: list[SourceRef]


@dataclass(frozen=True)
class AssessmentResult:
    assessment_id: str
    session_id: str
    concept_id: str
    assessment_type: AssessmentType
    answer_hash: str
    rubric_id: str
    score: float
    status: MasteryStatus
    strengths: list[str]
    gaps: list[str]
    misconceptions: list[MisconceptionEvidence]
    evidence: list[SourceRef]
    recommended_action: RecommendedAction
    created_at: str
    generation: GenerationProvenance
    field_origins: dict[str, str]
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 1:
            raise ValueError("assessment score must be between 0 and 1")
        if not self.evidence:
            raise ValueError("an assessment requires evidence")
        if self.status == MasteryStatus.MISCONCEPTION and not self.misconceptions:
            raise ValueError("misconception status requires misconception evidence")
        if self.recommended_action == RecommendedAction.ADVANCE and self.score < 0.7:
            raise ValueError("advance requires a score of at least 0.7")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        _validate_origins(
            self.field_origins,
            {"answer_hash", "score", "status", "strengths", "gaps", "misconceptions", "evidence", "recommended_action"},
            self.generation,
        )


@dataclass(frozen=True)
class ResearchBudget:
    max_urls: int
    max_seconds: int
    max_bytes: int
    max_depth: int

    def __post_init__(self) -> None:
        if not 1 <= self.max_urls <= 1000:
            raise ValueError("max_urls must be between 1 and 1000")
        if not 1 <= self.max_seconds <= 86400:
            raise ValueError("max_seconds must be between 1 and 86400")
        if not 1 <= self.max_bytes <= 10737418240:
            raise ValueError("max_bytes must be between 1 and 10737418240")
        if not 0 <= self.max_depth <= 10:
            raise ValueError("max_depth must be between 0 and 10")


@dataclass(frozen=True)
class ResearchRequest:
    request_id: str
    goal_id: str
    concept_id: str
    question: str
    trigger: ResearchTrigger
    gap_evidence: list[SourceRef]
    allowed_domains: list[str]
    budget: ResearchBudget
    status: ResearchStatus
    created_at: str
    updated_at: str
    approval: HumanApproval | None
    generation: GenerationProvenance
    field_origins: dict[str, str]
    job_id: str | None = None
    result_source_refs: list[SourceRef] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 10 <= len(self.question) <= 2000:
            raise ValueError("research question must contain 10 to 2000 characters")
        if not self.gap_evidence:
            raise ValueError("research requires evidence of an educational gap")
        if not self.allowed_domains:
            raise ValueError("research requires at least one allowed domain")
        if any(not _DOMAIN_PATTERN.fullmatch(domain) for domain in self.allowed_domains):
            raise ValueError("allowed_domains must contain domain names without schemes or paths")
        _validate_time_order(self.created_at, self.updated_at)
        _validate_origins(
            self.field_origins,
            {"question", "trigger", "gap_evidence", "allowed_domains", "budget"},
            self.generation,
        )
        if self.status in {ResearchStatus.APPROVED, ResearchStatus.RUNNING, ResearchStatus.COMPLETED}:
            if self.approval is None or not self.approval.approved:
                raise ValueError("executable research states require human approval")
        if self.status in {ResearchStatus.RUNNING, ResearchStatus.COMPLETED} and not self.job_id:
            raise ValueError("running or completed research requires a job ID")
        if self.status == ResearchStatus.COMPLETED and not self.result_source_refs:
            raise ValueError("completed research requires result source references")


def answer_hash(answer: str) -> str:
    return "sha256:" + hashlib.sha256(answer.encode("utf-8")).hexdigest()


class EvidenceType(StrEnum):
    """Kind of learner evidence recorded in the append-only evidence log."""
    ASSESSMENT = "assessment"
    CONVERSATION = "conversation"
    SELF_REPORT = "self_report"
    OBSERVATION = "observation"


@dataclass(frozen=True)
class UserEvidence:
    """Append-only evidence about the learner (Fase 2).

    Assessments, conversation observations and self-reports accumulate here;
    mastery updates in UserTopicRecord must trace back to this evidence
    (invariant: user_evidence_is_append_only, mastery_is_supported_by_
    assessment_evidence).
    """
    evidence_id: str
    topic_id: str
    evidence_type: EvidenceType
    observation: str
    observed_at: str
    recorded_at: str
    source_refs: list[SourceRef]
    generation: GenerationProvenance
    field_origins: dict[str, str]
    assessment_id: str | None = None
    session_id: str | None = None

    def __post_init__(self) -> None:
        if not self.observation.strip():
            raise ValueError("evidence requires a non-empty observation")
        if not self.source_refs:
            raise ValueError("evidence requires at least one source reference")
        if self.evidence_type == EvidenceType.ASSESSMENT and not self.assessment_id:
            raise ValueError("assessment evidence requires assessment_id")
        _validate_time_order(self.observed_at, self.recorded_at)
        _validate_origins(
            self.field_origins,
            {"observation", "source_refs"},
            self.generation,
        )


@dataclass(frozen=True)
class UserTopicRecord:
    """Mastery state for one topic in the unified user model (Fase 2, DEC-002).

    The Tutor owns this state scope: what the learner knows per topic. Any
    non-unknown mastery state must trace to an assessment (invariant:
    user_topic_records_require_evidence_for_mastery).
    """
    record_id: str
    topic_id: str
    mastery_status: MasteryStatus
    mastery_score: float | None
    attempts: int
    last_assessment_id: str | None
    evidence_ids: list[str]
    updated_at: str
    created_at: str
    generation: GenerationProvenance
    field_origins: dict[str, str]
    prerequisite_ids: list[str] = field(default_factory=list)
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.mastery_score is not None and not 0 <= self.mastery_score <= 1:
            raise ValueError("mastery_score must be between 0 and 1")
        if self.attempts < 0:
            raise ValueError("attempts must be non-negative")
        # mastery_is_supported_by_assessment_evidence
        if self.mastery_status == MasteryStatus.UNKNOWN and self.last_assessment_id:
            raise ValueError("unknown mastery must not reference an assessment")
        if self.mastery_status in {
            MasteryStatus.UNDERSTOOD, MasteryStatus.APPLIED,
            MasteryStatus.NEEDS_REVIEW, MasteryStatus.MISCONCEPTION,
        } and not self.last_assessment_id:
            raise ValueError(
                f"mastery_status '{self.mastery_status.value}' requires assessment evidence"
            )
        _validate_time_order(self.created_at, self.updated_at)
        _validate_origins(
            self.field_origins,
            {"mastery_status", "mastery_score", "attempts", "last_assessment_id", "evidence_ids"},
            self.generation,
        )


def contract_dict(record: Any) -> dict[str, Any]:
    return asdict(record)

