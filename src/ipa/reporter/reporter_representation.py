"""Clean document representations for semantic analysis."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentRepresentation:
    title: str
    title_source: str
    title_confidence: str
    abstract: str
    keywords: tuple[str, ...]
    embedding_text: str


def extract_title(text: str, fallback: str) -> tuple[str, str, str]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    for line in lines[:30]:
        if line.lower().startswith(("abstract", "keywords", "contents", "references")):
            continue
        if 12 <= len(line) <= 240 and not re.search(r"\b(arxiv|doi|http|copyright)\b", line, re.I):
            return line.rstrip(".:"), "first_page_heading", "medium"
    return fallback, "filename", "low"


def extract_abstract(text: str) -> str:
    match = re.search(r"\babstract\b\s*[:\n]?\s*(.*?)(?:\n\s*(?:keywords|introduction|1\.?\s+introduction)\b|\Z)", text, re.I | re.S)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()[:4000]


def build_representation(text: str, fallback_title: str) -> DocumentRepresentation:
    title, source, confidence = extract_title(text, fallback_title)
    abstract = extract_abstract(text)
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", (title + " " + abstract).lower())
    keywords = tuple(dict.fromkeys(words[:20]))
    embedding_text = "\n".join(part for part in (title, abstract, " ".join(keywords)) if part)
    return DocumentRepresentation(title, source, confidence, abstract, keywords, embedding_text)

