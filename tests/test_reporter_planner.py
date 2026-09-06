from __future__ import annotations

import pytest

from ipa.agentic_contracts import QueryIR
from ipa.reporter_planner import ReporterPlanner, plan_report_query


def test_topic_plan_is_typed_deterministic_and_preserves_scope():
    report = {"report_id": "report:2026-09"}
    topic = {
        "category_id": "topic:storage",
        "label": "Distributed storage",
        "description": "Replication and consistency trade-offs",
        "subtopics": ["failure recovery"],
        "document_ids": ["doc:2", "doc:1", "doc:2"],
    }
    question = "Compare Aurora and Borealis recovery approaches"

    first = plan_report_query(question, report=report, topic=topic)
    second = plan_report_query(question, report=report, topic=topic)

    assert isinstance(first, QueryIR)
    assert first == second
    assert first.intent == "comparison"
    assert first.is_comparison
    assert first.report_id == "report:2026-09"
    assert first.topic_id == "topic:storage"
    assert first.constraints["retrieval_scope"] == "restricted"
    assert first.constraints["retrieval_strategy"] == "topic_restricted"
    assert first.constraints["document_ids"] == ["doc:1", "doc:2"]
    assert first.constraints["min_document_diversity"] == 2
    assert {"Aurora", "Borealis"} <= set(first.entities)
    assert "replication" in first.required_evidence


def test_explicit_document_scope_wins_over_category_scope():
    plan = ReporterPlanner().plan(
        "What evidence supports the result?",
        category={"category_id": "topic:1", "document_ids": ["doc:category"]},
        document_ids=["doc:selected"],
    )

    assert plan.intent == "evidence"
    assert plan.topic_id == "topic:1"
    assert plan.constraints["document_ids"] == ["doc:selected"]
    assert plan.constraints["min_document_diversity"] == 1
    assert plan.language == "en"


@pytest.mark.parametrize(
    ("question", "intent", "language"),
    [
        ("¿Cómo configurar el proceso reproducible?", "procedural", "es"),
        ("How many records are in the collection?", "numeric", "en"),
        ("Explain the observed mechanism", "explanation", "en"),
        ("Resume el documento seleccionado", "document_reading", "es"),
    ],
)
def test_general_linguistic_patterns_classify_without_domain_rules(question, intent, language):
    plan = plan_report_query(question)
    assert plan.intent == intent
    assert plan.language == language
    assert plan.constraints["retrieval_scope"] == "global"
    assert plan.constraints["document_ids"] == []


@pytest.mark.parametrize("question", ["", "   ", None])
def test_planner_rejects_missing_question(question):
    with pytest.raises(ValueError, match="question"):
        plan_report_query(question)  # type: ignore[arg-type]
