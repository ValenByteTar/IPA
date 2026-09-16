"""Tests for the LLM curation judge (zona gris) and calibration."""
import json
import pytest

from ipa.reporter.curation_judge import (
    JudgeVerdict, judge_gray_batch, _parse_verdicts,
)
from ipa.reporter.curation_calibration import (
    CalibrationResult, calibrate_curation, load_calibration,
)
from ipa.reporter.reporter_contracts import (
    ReporterDecision, ReporterDocumentDecision, ReviewStatus, ScoreBundle,
    generation_provenance,
)
from ipa.reporter.reporter_curation import classify_tiers


class FakeProvider:
    """Fake LLM provider que devuelve JSON predefinido."""
    def __init__(self, responses: list[str], loaded: bool = True):
        self._responses = responses
        self._idx = 0
        self._loaded = loaded
        self.no_think = True  # el endpoint lo togglea

    def generate_chat(self, messages, **kwargs):
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        from types import SimpleNamespace
        return SimpleNamespace(text=resp)

    def is_loaded(self):
        return self._loaded


def _make_decision(doc_id, scores=None, decision=ReporterDecision.PROMOTE, dup=None):
    scores = scores or ScoreBundle(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    return ReporterDocumentDecision(
        decision_id=f"d:{doc_id}", report_id="r", document_id=doc_id,
        artifact_id=f"a:{doc_id}", decision=decision, scores=scores,
        reason="t", evidence=[], generation=generation_provenance("h"),
        review_status=ReviewStatus.PENDING, duplicate_of=dup,
    )


# --- Judge tests ---

def test_judge_parses_valid_json():
    raw = '[{"document_id": "d1", "verdict": "promote", "confidence": 0.9, "reason": "bueno"}]'
    verdicts = _parse_verdicts(raw, ["d1"])
    assert len(verdicts) == 1
    assert verdicts[0].verdict == "promote"
    assert verdicts[0].confidence == 0.9


def test_judge_parses_json_with_surrounding_text():
    raw = 'Pensando... [{"document_id": "d1", "verdict": "reject", "confidence": 0.8, "reason": "malo"}] fin'
    verdicts = _parse_verdicts(raw, ["d1"])
    assert verdicts[0].verdict == "reject"


def test_judge_fallback_on_invalid_json():
    raw = "no json here"
    verdicts = _parse_verdicts(raw, ["d1"])
    assert verdicts[0].verdict == "defer"
    assert "no produjo JSON" in verdicts[0].reason


def test_judge_fallback_on_missing_docs():
    raw = '[{"document_id": "d1", "verdict": "promote", "confidence": 0.9, "reason": "ok"}]'
    verdicts = _parse_verdicts(raw, ["d1", "d2"])
    assert len(verdicts) == 2
    assert verdicts[0].document_id == "d1"
    assert verdicts[1].document_id == "d2"
    assert verdicts[1].verdict == "defer"  # omitido → defer


def test_judge_invalid_verdict_becomes_defer():
    raw = '[{"document_id": "d1", "verdict": "maybe", "confidence": 0.5, "reason": "?"}]'
    verdicts = _parse_verdicts(raw, ["d1"])
    assert verdicts[0].verdict == "defer"


def test_judge_confidence_clamped():
    raw = '[{"document_id": "d1", "verdict": "promote", "confidence": 1.5, "reason": ""}]'
    verdicts = _parse_verdicts(raw, ["d1"])
    assert verdicts[0].confidence == 1.0


def test_judge_batch_with_fake_provider():
    docs = [
        {"document_id": "d1", "title": "AI", "text": "buen contenido", "source_domain": "arxiv.org", "promotion_score": 0.7},
        {"document_id": "d2", "title": "Spam", "text": "clickbait", "source_domain": "spam.com", "promotion_score": 0.5},
    ]
    response = json.dumps([
        {"document_id": "d1", "verdict": "promote", "confidence": 0.9, "reason": "relevante"},
        {"document_id": "d2", "verdict": "reject", "confidence": 0.8, "reason": "clickbait"},
    ])
    provider = FakeProvider([response])
    verdicts = judge_gray_batch(docs, provider)
    assert len(verdicts) == 2
    assert verdicts[0].verdict == "promote"
    assert verdicts[1].verdict == "reject"


def test_judge_empty_docs():
    provider = FakeProvider([])
    assert judge_gray_batch([], provider) == []


def test_judge_provider_not_loaded():
    docs = [{"document_id": "d1", "title": "", "text": "", "source_domain": "", "promotion_score": 0.5}]
    provider = FakeProvider([], loaded=False)
    verdicts = judge_gray_batch(docs, provider)
    assert verdicts[0].verdict == "defer"
    assert "no cargado" in verdicts[0].reason


def test_judge_generation_error_becomes_defer():
    class ErrorProvider:
        def is_loaded(self): return True
        def generate_chat(self, messages, **kwargs): raise RuntimeError("boom")
    docs = [{"document_id": "d1", "title": "", "text": "", "source_domain": "", "promotion_score": 0.5}]
    verdicts = judge_gray_batch(docs, ErrorProvider())
    assert verdicts[0].verdict == "defer"
    assert "Error" in verdicts[0].reason


# --- Calibration tests ---

def test_calibration_persists_and_loads(tmp_path):
    out = tmp_path / "cal.json"
    # Provider que siempre dice "promote" para docs con score alto
    def make_response(docs):
        return json.dumps([
            {"document_id": d["document_id"], "verdict": "promote" if d["promotion_score"] > 0.6 else "reject",
             "confidence": 0.9, "reason": "test"} for d in docs
        ])
    docs = [
        {"document_id": f"d{i}", "title": f"doc {i}", "text": f"content {i}",
         "source_domain": "test.com", "promotion_score": 0.95 - i * 0.04}
        for i in range(20)
    ]
    decisions = [_make_decision(f"d{i}", scores=ScoreBundle(max(0.05, 0.95 - i*0.04), 0.5, 0.5, 0.5, 0.5, 0.5))
                 for i in range(20)]
    provider = FakeProvider([make_response(docs[:15]), make_response(docs[15:])])
    result = calibrate_curation(docs, decisions, provider, sample_size=20, output_path=out)
    # Muestra estratificada: 70% PROMOTE (14) + 30% otros (0, todos son PROMOTE)
    assert result.sample_size == 14
    assert out.exists()
    loaded = load_calibration(out)
    assert loaded is not None
    assert loaded["sample_size"] == 14


def test_calibration_small_sample_returns_defaults(tmp_path):
    out = tmp_path / "cal.json"
    docs = [{"document_id": "d1", "title": "", "text": "", "source_domain": "", "promotion_score": 0.5}]
    decisions = [_make_decision("d1")]
    provider = FakeProvider([])
    result = calibrate_curation(docs, decisions, provider, sample_size=100, output_path=out)
    # Muestra demasiado chica → defaults
    assert result.auto_promote_percentile == 0.80
    assert result.auto_reject_percentile == 0.40


def test_load_calibration_missing_file():
    assert load_calibration("/nonexistent/path.json") is None


def test_load_calibration_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json", encoding="utf-8")
    assert load_calibration(p) is None
