"""Domain-agnostic topic discovery over Reporter documents."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Callable

from ipa.reporter.reporter_contracts import TopicEvolution, TopicLink, generation_provenance, sha256_hash

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "una", "para", "con", "que", "los", "las", "del", "por", "una", "como",
    "sobre", "into", "their", "will", "have", "has", "not", "you", "your",
    # Generic terms that appear in many topic labels and cause mega-categories
    "topic", "topico", "document", "documento", "related", "relacionado",
    "across", "through", "traves", "nuevo", "nueva", "new", "recent",
    "research", "investigacion", "analysis", "analisis", "study", "estudio",
    "system", "sistema", "model", "modelo", "data", "datos", "using",
    "usando", "based", "basado", "approach", "enfoque", "method", "metodo",
    "report", "reporte", "article", "articulo", "paper", "paper",
}


def _tokens(text: str) -> set[str]:
    return {word.lower() for word in re.findall(r"[\w-]{4,}", text, re.UNICODE) if word.lower() not in _STOPWORDS}


def _lexical_similarity(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    return len(left & right) / len(left | right) if left and right else 0.0


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _label(texts: list[str]) -> tuple[str, str]:
    counts = Counter(token for text in texts for token in _tokens(text))
    terms = [term for term, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]]
    label = " / ".join(term.replace("-", " ").title() for term in terms[:4]) or "Tema pendiente de nombrar"
    description = "Tema emergente identificado por similitud entre documentos: " + ", ".join(terms[:5]) + "."
    return label, description


def _fallback_label(group: list[dict[str, Any]]) -> tuple[str, str]:
    # Extract key terms from all documents in the group
    all_texts = []
    titles = []
    for item in group:
        title = str(item.get("title", "")).strip()
        if title:
            titles.append(title)
        text = item.get("representation_text") or item.get("text", "")
        if text:
            all_texts.append(str(text)[:2000])

    # Count term frequencies across all texts
    counts = Counter(token for text in all_texts for token in _tokens(text))
    # Filter out generic terms
    generic = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
               "have", "has", "had", "do", "does", "did", "will", "would", "could",
               "should", "may", "might", "must", "can", "this", "that", "these",
               "those", "with", "from", "into", "onto", "upon", "about", "after",
               "before", "during", "through", "between", "among", "under", "over",
               "also", "more", "most", "some", "such", "only", "very", "than",
               "then", "now", "here", "there", "where", "when", "what", "which",
               "who", "whom", "whose", "how", "why", "not", "no", "nor", "but",
               "and", "or", "if", "so", "as", "at", "by", "for", "in", "of", "on",
               "to", "up", "out", "off", "down", "all", "any", "each", "few",
               "many", "other", "same", "such", "own", "one", "two", "three",
               "new", "said", "says", "say", "said", "like", "well", "even",
               "still", "just", "also", "el", "la", "los", "las", "un", "una",
               "de", "del", "en", "es", "por", "para", "con", "que", "se",
               "su", "sus", "al", "lo", "le", "les", "ya", "ha", "han",
               "fue", "son", "ser", "estÃ¡", "estÃ¡n", "mÃ¡s", "muy", "tan"}
    key_terms = [term for term, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
                 if term.lower() not in generic and len(term) > 2][:6]

    # Build label from top terms or titles
    if key_terms:
        label = " / ".join(term.replace("-", " ").title() for term in key_terms[:4])
    elif titles:
        usable = [t for t in titles if len(re.findall(r"[A-Za-zÃÃ‰ÃÃ“ÃšÃ¡Ã©Ã­Ã³ÃºÃ‘Ã±]{3,}", t)) >= 2]
        label = usable[0][:120] if usable else titles[0][:120]
    else:
        label = "Tema pendiente de nombrar"

    # Build description from key terms and document count
    if key_terms:
        description = f"TÃ³pico sobre {', '.join(key_terms[:4])} â€” {len(group)} documento(s) relacionado(s)."
    else:
        description = f"TÃ³pico con {len(group)} documento(s) relacionados. TÃ©rminos clave: {', '.join(key_terms[:3])}."

    return label[:180], description


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, (str, int, float))]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, dict):
        return [str(item) for item in value.values() if isinstance(item, (str, int, float))]
    return []


def discover_topics(
    documents: list[dict[str, Any]],
    similarity_threshold: float = 0.52,
    min_documents: int = 2,
    allow_singletons: bool = True,
    embeddings: list[list[float]] | None = None,
    labeler: Callable[[list[dict[str, Any]]], dict[str, str]] | None = None,
    labeler_batch: Callable[..., list[dict[str, str]]] | None = None,
    report_id: str = "report:unspecified",
    progress_callback=None,
) -> list[dict[str, Any]]:
    if not documents:
        return []
    threshold = max(similarity_threshold, 0.45) if embeddings is not None else similarity_threshold
    similarities: dict[tuple[int, int], float] = {}
    for left in range(len(documents)):
        for right in range(left + 1, len(documents)):
            if embeddings is not None and len(embeddings) == len(documents):
                similarity = _cosine(embeddings[left], embeddings[right])
            else:
                left_text = documents[left].get("representation_text") or documents[left].get("title", "") + " " + documents[left].get("text", "")[:2000]
                right_text = documents[right].get("representation_text") or documents[right].get("title", "") + " " + documents[right].get("text", "")[:2000]
                similarity = _lexical_similarity(left_text, right_text)
            similarities[left, right] = similarity

    def pair_similarity(left: int, right: int) -> float:
        return similarities[min(left, right), max(left, right)] if left != right else 1.0

    groups: dict[int, list[int]] = {index: [index] for index in range(len(documents))}
    distances = {
        (left, right): pair_similarity(left, right)
        for left in groups
        for right in groups
        if left < right
    }
    while True:
        eligible = [(score, left, right) for (left, right), score in distances.items() if score >= threshold]
        if not eligible:
            break
        _, left, right = max(eligible)
        merged = sorted(groups[left] + groups[right])
        del groups[right]
        groups[left] = merged
        distances = {
            (a, b): score for (a, b), score in distances.items()
            if a != right and b != right and a != left and b != left
        }
        for other in groups:
            if other == left:
                continue
            a, b = sorted((left, other))
            distances[a, b] = min(pair_similarity(member, candidate) for member in merged for candidate in groups[other])

    grouped_documents = [groups[index] for index in sorted(groups)]
    grouped_documents = [[documents[index] for index in group] for group in grouped_documents]

    topics = []
    total_groups = len(grouped_documents)
    llm_progress = None
    if progress_callback is not None:
        def llm_progress(current: int, total: int) -> None:
            progress_callback(current, total)
    batch_labels = labeler_batch(grouped_documents, llm_progress) if labeler_batch else []
    for group_index, group in enumerate(grouped_documents, start=1):
        if progress_callback is not None:
            progress_callback(group_index, total_groups)
        if len(group) < min_documents and not allow_singletons:
            continue
        texts = [item.get("title", "") + " " + item.get("text", "") for item in group]
        generated = batch_labels[group_index - 1] if group_index - 1 < len(batch_labels) else (labeler(group) if labeler else {})
        fallback_label, fallback_description = _fallback_label(group)
        label = generated.get("label") or fallback_label
        description = generated.get("description") or fallback_description
        ids = sorted(str(item["document_id"]) for item in group)
        category_id = "topic:" + hashlib.sha256("|".join(ids).encode()).hexdigest()[:32]
        sources = sorted({item.get("source_domain") or "unknown" for item in group})
        topic = {
            "category_id": category_id,
            "label": label,
            "description": description,
            "document_count": len(group),
            "source_count": len(sources),
            "cohesion": round(min(1.0, len(group) / max(len(group), 1)), 4),
            "importance": round(min(1.0, 0.5 + len(group) / 20), 4),
            "novelty": 0.5,
            "evolution": TopicEvolution.NEW.value,
            "subtopics": _string_list(generated.get("subtopics", [])),
            "representative_sources": [{"source_id": str(group[0]["document_id"]), "source_type": "document"}],
            "document_ids": ids,
            "uncertainties": _string_list(generated.get("uncertainties", [])),
            "source_refs": [{"source_id": str(item["document_id"]), "source_type": "document"} for item in group],
            "generation": generation_provenance(sha256_hash("|".join(ids))),
            "field_origins": {"label": "generated", "description": "generated", "document_ids": "source", "importance": "generated"},
        }
        topics.append(topic)
    return sorted(topics, key=lambda topic: (-topic["importance"], topic["category_id"]))


def match_topic_continuity(current: list[dict[str, Any]], previous: list[dict[str, Any]]) -> list[TopicLink]:
    links: list[TopicLink] = []
    for topic in current:
        best = None
        best_score = 0.0
        for old in previous:
            score = _lexical_similarity(topic["label"] + " " + topic["description"], old.get("label", "") + " " + old.get("description", ""))
            shared = len(set(topic["document_ids"]) & set(old.get("document_ids", [])))
            score = max(score, min(1.0, score + shared * 0.2))
            if score > best_score:
                best_score, best = score, old
        if best is None or best_score < 0.25:
            relation = TopicEvolution.NEW
            previous_id = None
        else:
            previous_id = best["category_id"]
            relation = TopicEvolution.STABLE if best_score >= 0.6 else TopicEvolution.AMBIGUOUS
        links.append(TopicLink(
            current_category_id=topic["category_id"], previous_category_id=previous_id,
            relation=relation, score=round(best_score, 4),
            evidence=topic["source_refs"],
            reason="Correspondencia basada en similitud semÃ¡ntica, tÃ©rminos y documentos compartidos.",
            generation=topic["generation"],
        ))
    return links


def group_topics_into_categories(
    topics: list[dict[str, Any]],
    *,
    llm_grouper: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Group fine-grained topics into broader parent categories.

    Uses the LLM to identify thematic clusters among topics. Each parent category
    contains a list of subtopic_ids pointing to the original topics.

    Returns a list of parent categories with:
    - category_id, label, description, subtopic_ids, document_count, importance
    """
    if not topics:
        return []
    # If <= 5 topics, no need to group further
    if len(topics) <= 5:
        return []

    # Build a compact representation of each topic for the LLM
    topic_summaries = [
        {
            "topic_id": t["category_id"],
            "label": t["label"],
            "description": t.get("description", ""),
            "doc_count": t.get("document_count", 0),
        }
        for t in topics
    ]

    parent_categories = []
    if llm_grouper:
        try:
            candidate = llm_grouper(topic_summaries)
            if _valid_parent_grouping(candidate, len(topics)):
                parent_categories = candidate
        except Exception:
            parent_categories = []

    # Fallback: deterministic grouping by shared keywords in labels
    if not parent_categories:
        parent_categories = _fallback_group_topics(topics)

    if progress_callback:
        progress_callback(len(parent_categories), len(parent_categories))

    # Enrich parent categories with document counts and subtopic references
    topic_by_id = {t["category_id"]: t for t in topics}
    result = []
    for parent in parent_categories:
        subtopic_ids = parent.get("subtopic_ids", [])
        # Resolve subtopic IDs â€” if they're indices, convert to category_ids
        resolved_ids = []
        for sid in subtopic_ids:
            if isinstance(sid, int) and 0 <= sid < len(topics):
                resolved_ids.append(topics[sid]["category_id"])
            elif isinstance(sid, str) and sid in topic_by_id:
                resolved_ids.append(sid)
            elif isinstance(sid, str):
                # Try matching by label
                for t in topics:
                    if t["label"] == sid:
                        resolved_ids.append(t["category_id"])
                        break

        # If no subtopics resolved, skip
        if not resolved_ids:
            continue

        total_docs = sum(topic_by_id.get(sid, {}).get("document_count", 0) for sid in resolved_ids)
        all_doc_ids = []
        for sid in resolved_ids:
            all_doc_ids.extend(topic_by_id.get(sid, {}).get("document_ids", []))

        parent_id = "category:" + hashlib.sha256("|".join(sorted(resolved_ids)).encode()).hexdigest()[:32]
        result.append({
            "category_id": parent_id,
            "label": parent.get("label", "CategorÃ­a general"),
            "description": parent.get("description", ""),
            "subtopic_ids": resolved_ids,
            "document_count": total_docs,
            "document_ids": sorted(set(all_doc_ids)),
            "importance": round(min(1.0, 0.5 + total_docs / 20), 4),
            "cohesion": 1.0,
            "novelty": 0.5,
            "evolution": TopicEvolution.NEW.value,
            "source_refs": [],
            "generation": generation_provenance(sha256_hash("|".join(sorted(resolved_ids)))),
        })

    return sorted(result, key=lambda c: (-c["importance"], c["category_id"]))


def _valid_parent_grouping(groups: Any, topic_count: int) -> bool:
    """Reject malformed or mega-category LLM output."""
    if not isinstance(groups, list) or topic_count > 12 and len(groups) < 3 or len(groups) > 8:
        return False
    assigned = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("subtopic_ids"), list):
            return False
        assigned.extend(group["subtopic_ids"])
    if topic_count > 12 and max((sum(1 for _ in group.get("subtopic_ids", [])) for group in groups), default=0) > topic_count * 0.5:
        return False
    return bool(assigned)


def _fallback_group_topics(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic fallback: group topics by shared significant keywords."""
    # Extract keywords from each topic label
    topic_keywords = []
    for t in topics:
        tokens = _tokens(t["label"] + " " + t.get("description", ""))
        topic_keywords.append(tokens)

    # Build adjacency: topics sharing >= 2 keywords are connected (stricter to avoid mega-categories)
    groups: dict[int, list[int]] = {i: [i] for i in range(len(topics))}
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            shared = topic_keywords[i] & topic_keywords[j]
            if len(shared) >= 2:  # Require 2+ shared keywords to merge
                # Merge j into i's group
                for key, members in list(groups.items()):
                    if j in members and key != i:
                        groups[i] = sorted(set(groups.get(i, []) + members))
                        del groups[key]
                        break

    # Split mega-categories: if a group has > 8 topics, split by re-clustering
    max_group_size = 8
    final_groups = []
    for group_indices in groups.values():
        if len(group_indices) > max_group_size:
            # Re-cluster the large group by tighter keyword overlap
            sub_groups = _split_large_group(group_indices, topic_keywords, min_shared=3)
            final_groups.extend(sub_groups)
        else:
            final_groups.append(group_indices)

    # Build parent categories from groups
    result = []
    for group_indices in final_groups:
        if len(group_indices) < 2:
            continue  # Only create parent categories for groups of 2+ topics
        group_topics_list = [topics[i] for i in group_indices]
        # Use most common keywords as label
        all_keywords = Counter()
        for tokens in [topic_keywords[i] for i in group_indices]:
            all_keywords.update(tokens)
        top_keywords = [kw for kw, _ in all_keywords.most_common(4)]
        label = " / ".join(k.capitalize() for k in top_keywords) if top_keywords else "CategorÃ­a general"
        result.append({
            "label": label,
            "description": f"AgrupaciÃ³n de {len(group_indices)} tÃ³picos relacionados: " + ", ".join(t["label"] for t in group_topics_list[:5]),
            "subtopic_ids": [topics[i]["category_id"] for i in group_indices],
        })
    return result


def _split_large_group(indices: list[int], topic_keywords: list[set[str]], min_shared: int = 3) -> list[list[int]]:
    """Split a large group into smaller sub-groups using tighter keyword overlap."""
    groups: dict[int, list[int]] = {i: [i] for i in indices}
    for i_pos, i in enumerate(indices):
        for j in indices[i_pos + 1:]:
            shared = topic_keywords[i] & topic_keywords[j]
            if len(shared) >= min_shared:
                for key, members in list(groups.items()):
                    if j in members and key != i:
                        groups[i] = sorted(set(groups.get(i, []) + members))
                        del groups[key]
                        break
    return list(groups.values())

