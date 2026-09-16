"""Tests for the tiered curation cascade (ranking-based, not absolute thresholds)."""
import pytest

from ipa.reporter.reporter_contracts import (
    ReporterDecision, ReporterDocumentDecision, ReviewStatus, ScoreBundle,
    generation_provenance,
)
from ipa.reporter.reporter_curation import (
    CurationTier, classify_tiers, promotion_score,
)


def _make_decision(doc_id: str, decision: ReporterDecision = ReporterDecision.PROMOTE,
                   scores: ScoreBundle | None = None,
                   duplicate_of: str | None = None) -> ReporterDocumentDecision:
    scores = scores or ScoreBundle(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    return ReporterDocumentDecision(
        decision_id=f"decision:{doc_id}", report_id="r1", document_id=doc_id,
        artifact_id=f"art:{doc_id}", decision=decision, scores=scores,
        reason="test", evidence=[], generation=generation_provenance("hash"),
        review_status=ReviewStatus.PENDING, duplicate_of=duplicate_of,
    )


def test_classify_tiers_auto_promote_top_percentile():
    """Top 20% del ranking → auto_promote."""
    decisions = [
        _make_decision("d1", scores=ScoreBundle(0.9, 0.9, 0.9, 0.9, 0.9, 0.9)),
        _make_decision("d2", scores=ScoreBundle(0.8, 0.8, 0.8, 0.8, 0.8, 0.8)),
        _make_decision("d3", scores=ScoreBundle(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)),
        _make_decision("d4", scores=ScoreBundle(0.4, 0.4, 0.4, 0.4, 0.4, 0.4)),
        _make_decision("d5", scores=ScoreBundle(0.3, 0.3, 0.3, 0.3, 0.3, 0.3)),
    ]
    tiers = classify_tiers(decisions, auto_promote_percentile=0.80, auto_reject_percentile=0.40)
    # d1 tiene el score más alto → auto_promote
    assert tiers[0].tier == "auto_promote"
    assert tiers[0].document_id == "d1"
    assert tiers[0].percentile == 1.0


def test_classify_tiers_auto_reject_bottom_percentile():
    """Bottom 60% → auto_reject."""
    decisions = [
        _make_decision("d1", scores=ScoreBundle(0.9, 0.9, 0.9, 0.9, 0.9, 0.9)),
        _make_decision("d2", scores=ScoreBundle(0.3, 0.3, 0.3, 0.3, 0.3, 0.3)),
    ]
    tiers = classify_tiers(decisions, auto_promote_percentile=0.80, auto_reject_percentile=0.40)
    # d2 tiene el score más bajo → auto_reject
    assert tiers[1].tier == "auto_reject"
    assert tiers[1].percentile == 0.0


def test_classify_tiers_gray_middle_zone():
    """Entre 0.40 y 0.80 → gray."""
    decisions = [
        _make_decision("d1", scores=ScoreBundle(0.9, 0.9, 0.9, 0.9, 0.9, 0.9)),
        _make_decision("d2", scores=ScoreBundle(0.6, 0.6, 0.6, 0.6, 0.6, 0.6)),
        _make_decision("d3", scores=ScoreBundle(0.3, 0.3, 0.3, 0.3, 0.3, 0.3)),
    ]
    tiers = classify_tiers(decisions, auto_promote_percentile=0.80, auto_reject_percentile=0.40)
    # d2 está en el medio → gray
    assert tiers[1].tier == "gray"


def test_classify_tiers_hard_reject_for_duplicates():
    """DUPLICATE → hard_reject, no entra al ranking."""
    decisions = [
        _make_decision("d1", scores=ScoreBundle(0.9, 0.9, 0.9, 0.9, 0.9, 0.9)),
        _make_decision("d2", decision=ReporterDecision.DUPLICATE, duplicate_of="d1"),
    ]
    tiers = classify_tiers(decisions)
    assert tiers[1].tier == "hard_reject"
    assert "duplicado" in tiers[1].reason


def test_classify_tiers_hard_reject_for_irrelevant():
    """IRRELEVANT → hard_reject."""
    decisions = [
        _make_decision("d1", decision=ReporterDecision.IRRELEVANT),
    ]
    tiers = classify_tiers(decisions)
    assert tiers[0].tier == "hard_reject"


def test_classify_tiers_empty_list():
    """Lista vacía → lista vacía."""
    assert classify_tiers([]) == []


def test_classify_tiers_single_candidate():
    """Un solo candidato → auto_promote (percentil 1.0)."""
    decisions = [_make_decision("d1", scores=ScoreBundle(0.9, 0.9, 0.9, 0.9, 0.9, 0.9))]
    tiers = classify_tiers(decisions)
    assert tiers[0].tier == "auto_promote"


def test_classify_tiers_preserves_order():
    """El output respeta el orden del input."""
    decisions = [
        _make_decision("d3", scores=ScoreBundle(0.3, 0.3, 0.3, 0.3, 0.3, 0.3)),
        _make_decision("d1", scores=ScoreBundle(0.9, 0.9, 0.9, 0.9, 0.9, 0.9)),
        _make_decision("d2", scores=ScoreBundle(0.6, 0.6, 0.6, 0.6, 0.6, 0.6)),
    ]
    tiers = classify_tiers(decisions)
    assert [t.document_id for t in tiers] == ["d3", "d1", "d2"]


def test_classify_tiers_rank_is_1_based():
    """rank=1 es el mejor score dentro de los candidatos."""
    decisions = [
        _make_decision("low", scores=ScoreBundle(0.3, 0.3, 0.3, 0.3, 0.3, 0.3)),
        _make_decision("high", scores=ScoreBundle(0.9, 0.9, 0.9, 0.9, 0.9, 0.9)),
        _make_decision("mid", scores=ScoreBundle(0.6, 0.6, 0.6, 0.6, 0.6, 0.6)),
    ]
    tiers = classify_tiers(decisions)
    high_tier = next(t for t in tiers if t.document_id == "high")
    assert high_tier.rank == 1  # mejor score
