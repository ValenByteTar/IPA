"""Deterministic, domain-agnostic query planning for Reporter topics.

The planner only interprets a question and its optional Reporter scope.  It does
not retrieve documents, build context, or call an LLM.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

from ipa.agentic.agentic_contracts import QueryIR


_COMPARISON_TERMS = {
    "compare", "comparison", "contrast", "difference", "differences", "versus", "vs",
    "compara", "comparar", "comparacion", "contrasta", "diferencia", "diferencias",
}
_PROCEDURAL_TERMS = {
    "how", "implement", "configure", "install", "procedure", "steps", "guide",
    "como", "implementar", "configurar", "instalar", "procedimiento", "pasos", "guia",
}
_NUMERIC_TERMS = {
    "amount", "count", "how many", "number", "percentage", "rate", "total", "version",
    "cantidad", "cuanto", "cuantos", "numero", "porcentaje", "tasa", "total", "version",
}
_EVIDENCE_TERMS = {
    "according", "citation", "cite", "evidence", "source", "support",
    "cita", "citar", "evidencia", "fuente", "respalda", "segun",
}
_READING_TERMS = {
    "document", "documents", "paper", "report", "source", "text",
    "documento", "documentos", "informe", "fuente", "texto",
}
_EXPLANATION_TERMS = {
    "define", "definition", "explain", "meaning", "what is", "what are", "why",
    "define", "definicion", "explica", "explicar", "por que", "que es", "que son",
}

_STOPWORDS = {
    "a", "al", "an", "and", "are", "as", "at", "be", "by", "con", "cual", "cuales",
    "de", "del", "describe", "dice", "do", "document", "documento", "el", "en", "es",
    "esta", "estan", "explain", "explica", "for", "from", "has", "have", "how", "in",
    "informe", "is", "it", "la", "las", "lo", "los", "of", "on", "or", "para", "por",
    "que", "report", "segun", "sobre", "son", "source", "su", "sus", "text", "the",
    "this", "to", "un", "una", "what", "which", "with", "y",
}
_SPANISH_MARKERS = {
    "como", "cual", "cuales", "cuanto", "de", "del", "donde", "el", "en", "es", "evidencia",
    "fuente", "la", "las", "los", "para", "por", "que", "segun", "sobre", "una", "y",
}
_ENGLISH_MARKERS = {
    "and", "are", "compare", "document", "evidence", "for", "from", "how", "in", "is", "of",
    "report", "source", "the", "to", "what", "where", "which", "with",
}


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _words(value: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:[-/][^\W_]+)*|\d+(?:[.,]\d+)*%?", value, re.UNICODE)


def _field(record: Mapping[str, Any] | str | None, *names: str) -> Any:
    if not isinstance(record, Mapping):
        return None
    for name in names:
        value = record.get(name)
        if value is not None and value != "":
            return value
    return None


def _record_id(record: Mapping[str, Any] | str | None, *names: str) -> str | None:
    if isinstance(record, str):
        value = record.strip()
        return value or None
    value = _field(record, *names)
    return str(value).strip() if value is not None and str(value).strip() else None


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Iterable) and not isinstance(value, (bytes, Mapping)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _document_scope(
    document_ids: Iterable[str] | None,
    topic: Mapping[str, Any] | str | None,
    category: Mapping[str, Any] | str | None,
) -> list[str]:
    values = list(document_ids or [])
    if not values:
        values = _string_values(_field(topic, "document_ids"))
    if not values:
        values = _string_values(_field(category, "document_ids"))
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _contains_phrase(folded_question: str, terms: set[str]) -> bool:
    padded = f" {folded_question} "
    return any(f" {term} " in padded for term in terms)


def _intent(question: str) -> tuple[str, bool]:
    folded = " ".join(_words(_fold(question)))
    is_comparison = _contains_phrase(folded, _COMPARISON_TERMS)
    if is_comparison:
        return "comparison", True
    # Numeric phrases such as "how many" also contain procedural question
    # words, so the more specific intent must win.
    if _contains_phrase(folded, _NUMERIC_TERMS):
        return "numeric", False
    if _contains_phrase(folded, _PROCEDURAL_TERMS):
        return "procedural", False
    if _contains_phrase(folded, _EVIDENCE_TERMS):
        return "evidence", False
    if _contains_phrase(folded, _READING_TERMS):
        return "document_reading", False
    if _contains_phrase(folded, _EXPLANATION_TERMS):
        return "explanation", False
    return "informational", False


def _language(question: str) -> str:
    folded_words = {_fold(word) for word in _words(question)}
    spanish = len(folded_words & _SPANISH_MARKERS)
    english = len(folded_words & _ENGLISH_MARKERS)
    if spanish > english:
        return "es"
    if english > spanish:
        return "en"
    return "und"


def _entities(question: str) -> list[str]:
    candidates: list[str] = []
    candidates.extend(match.strip() for match in re.findall(r'["â€œâ€]([^"â€œâ€]+)["â€œâ€]', question))

    words = question.split()
    for index, word in enumerate(words):
        clean = re.sub(r"[^\w/&.-]", "", word, flags=re.UNICODE).strip(".-")
        if len(clean) < 2:
            continue
        is_acronym = len(clean) >= 2 and any(char.isalpha() for char in clean) and clean.upper() == clean
        is_named = index > 0 and clean[0].isupper()
        has_identifier = any(char.isdigit() for char in clean) and any(char.isalpha() for char in clean)
        if is_acronym or is_named or has_identifier:
            candidates.append(clean)

    seen: set[str] = set()
    entities: list[str] = []
    for candidate in candidates:
        key = _fold(candidate)
        if key and key not in seen and key not in _STOPWORDS:
            seen.add(key)
            entities.append(candidate)
    return entities[:12]


def _required_evidence(question: str, topic_fields: list[str], entities: list[str]) -> list[str]:
    candidates = [*entities]
    candidates.extend(word.casefold() for word in _words(question) if len(word) >= 3)
    for field in topic_fields:
        candidates.extend(word.casefold() for word in _words(field) if len(word) >= 3)

    seen: set[str] = set()
    requirements: list[str] = []
    for candidate in candidates:
        cleaned = candidate.strip()
        key = _fold(cleaned)
        if not key or key in seen or key in _STOPWORDS:
            continue
        seen.add(key)
        requirements.append(cleaned)
    return requirements[:15]


def plan_report_query(
    question: str,
    *,
    report: Mapping[str, Any] | str | None = None,
    category: Mapping[str, Any] | str | None = None,
    topic: Mapping[str, Any] | str | None = None,
    document_ids: Iterable[str] | None = None,
) -> QueryIR:
    """Create an operational query plan, preserving any selected topic scope.

    ``report``, ``category``, and ``topic`` may be Reporter dictionaries or ID
    strings.  A topic takes precedence over a category for topic metadata, while
    explicit ``document_ids`` take precedence over IDs embedded in either.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    raw_query = question.strip()

    selected_topic = topic if topic is not None else category
    report_id = _record_id(report, "report_id", "id")
    topic_id = _record_id(selected_topic, "category_id", "topic_id", "id")
    scoped_document_ids = _document_scope(document_ids, topic, category)
    intent, is_comparison = _intent(raw_query)
    entities = _entities(raw_query)

    topic_fields: list[str] = []
    for record in (category, topic):
        for name in ("label", "title", "description"):
            value = _field(record, name)
            if isinstance(value, str) and value.strip():
                topic_fields.append(value.strip())
        topic_fields.extend(_string_values(_field(record, "subtopics")))

    diversity = 2 if intent in {"comparison", "evidence"} else 1
    if scoped_document_ids:
        diversity = min(diversity, len(scoped_document_ids))
    top_k_by_intent = {
        "comparison": 12,
        "evidence": 10,
        "explanation": 8,
        "procedural": 8,
        "informational": 8,
        "document_reading": 6,
        "numeric": 6,
    }
    constraints = {
        "retrieval_scope": "restricted" if scoped_document_ids else "global",
        "document_ids": scoped_document_ids,
        "top_k": top_k_by_intent[intent],
        "min_document_diversity": diversity,
        "retrieval_strategy": "topic_restricted" if scoped_document_ids else "global",
    }

    return QueryIR(
        raw_query=raw_query,
        intent=intent,
        entities=entities,
        constraints=constraints,
        topic_id=topic_id,
        report_id=report_id,
        required_evidence=_required_evidence(raw_query, topic_fields, entities),
        is_comparison=is_comparison,
        language=_language(raw_query),
    )


# A short generic name is useful to capability callers; both names are kept
# explicit so this module remains independent from Reporter orchestration.
plan_query = plan_report_query


class ReporterPlanner:
    """Stateless adapter for runtimes that resolve planners as capabilities."""

    name = "reporter_planner"

    def plan(
        self,
        question: str,
        *,
        report: Mapping[str, Any] | str | None = None,
        category: Mapping[str, Any] | str | None = None,
        topic: Mapping[str, Any] | str | None = None,
        document_ids: Iterable[str] | None = None,
    ) -> QueryIR:
        return plan_report_query(
            question,
            report=report,
            category=category,
            topic=topic,
            document_ids=document_ids,
        )

    __call__ = plan


__all__ = ["ReporterPlanner", "plan_query", "plan_report_query"]

