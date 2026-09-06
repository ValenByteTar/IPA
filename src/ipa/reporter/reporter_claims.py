"""Claim extraction and citation validation for Reporter answers."""
from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"[\w-]{4,}", re.UNICODE)
_CITATION_RE = re.compile(r"\[(?:Doc\s*)?(?:n\s*[=:]\s*)?(\d+)\]", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?(?:\s?[%xkKMGbgm]+)?", text))


def extract_claims(answer: str) -> list[dict[str, Any]]:
    normalized = re.sub(r"(?m)^\s*\d+[.)]\s*$", "", answer.strip())
    normalized = re.sub(r"(?m)^\s*\d+[.)]\s+", "", normalized)
    normalized = re.sub(r"(?m)^\s*[-*]\s+", "", normalized)
    if "Respuesta final:" in normalized:
        normalized = normalized.split("Respuesta final:", 1)[1].strip()
    claims = []
    for index, sentence in enumerate(re.split(r"(?<=[.!?])\s+(?=[A-ZÃÃ‰ÃÃ“ÃšÃ‘Â¿Â¡])", normalized)):
        sentence = sentence.strip()
        if not sentence:
            continue
        citations = [int(value) for value in _CITATION_RE.findall(sentence)]
        clean = _CITATION_RE.sub("", sentence).strip()
        claims.append({"claim_id": f"claim-{index + 1}", "text": clean, "citations": citations, "numbers": sorted(_numbers(clean))})
    return claims


def validate_claims(answer: str, evidence: list[str]) -> list[dict[str, Any]]:
    results = []
    for claim in extract_claims(answer):
        cited = [index for index in claim["citations"] if 1 <= index <= len(evidence)]
        if not cited:
            results.append({**claim, "support_level": "unsupported", "support_score": 0.0, "validated_citations": []})
            continue
        scores = []
        number_support = True
        for citation in cited:
            source = evidence[citation - 1]
            source_tokens = _tokens(source)
            claim_tokens = _tokens(claim["text"])
            overlap = len(claim_tokens & source_tokens) / len(claim_tokens) if claim_tokens else 0.0
            scores.append(overlap)
            if claim["numbers"] and not set(claim["numbers"]) <= _numbers(source):
                number_support = False
        score = max(scores, default=0.0)
        if score >= 0.35 and number_support:
            level = "supported"
        elif score >= 0.12:
            level = "partial"
        else:
            level = "unsupported"
        if not number_support:
            level = "partial" if score >= 0.25 else "unsupported"
        results.append({
            **claim,
            "support_level": level,
            "support_score": round(score, 4),
            "validated_citations": cited,
            "number_support": number_support,
        })
    return results


def citation_summary(claims: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"supported": 0, "partial": 0, "unsupported": 0}
    for claim in claims:
        summary[claim["support_level"]] = summary.get(claim["support_level"], 0) + 1
    return summary

