"""Optional structured LLM layer for Reporter curation and topic labels."""
from __future__ import annotations

import json
from typing import Any


class ReporterLLM:
    def __init__(self, provider) -> None:
        self.provider = provider

    @staticmethod
    def _parse(text: str) -> dict[str, Any]:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(text[start:end + 1])
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _generate_resilient(self, messages: list[list[dict[str, str]]], *, max_new_tokens: int, batch_size: int, progress_callback=None, base_index: int = 0) -> list[Any]:
        """Generate a batch with progressive splitting and per-item fallback."""
        if not messages:
            return []
        results: list[Any] = []
        for start in range(0, len(messages), batch_size):
            batch = messages[start:start + batch_size]
            if progress_callback is not None:
                progress_callback(base_index + start, len(messages) + base_index)
            # Reset generator state before each batch to prevent recurrent cache
            # saturation from previous batches (gated_delta_net accumulates state).
            self.provider.reset_generator()
            try:
                current = self.provider.generate_chat_batch(batch, max_new_tokens=max_new_tokens, timeout=60)
                if len(current) == len(batch):
                    results.extend(current)
                    self.provider.reset_generator()
                    continue
                raise RuntimeError("LLM batch returned an unexpected number of results")
            except Exception:
                self.provider.reset_generator()
                if len(batch) > 1:
                    midpoint = max(1, len(batch) // 2)
                    results.extend(self._generate_resilient(batch[:midpoint], max_new_tokens=max_new_tokens, batch_size=midpoint, progress_callback=progress_callback, base_index=base_index + start))
                    results.extend(self._generate_resilient(batch[midpoint:], max_new_tokens=max_new_tokens, batch_size=len(batch) - midpoint, progress_callback=progress_callback, base_index=base_index + start + midpoint))
                else:
                    try:
                        results.append(self.provider.generate_chat(batch[0], max_new_tokens=max_new_tokens, timeout=60))
                    except Exception:
                        results.append(None)
                    self.provider.reset_generator()
        return results

    def classify(self, document: dict[str, Any], interests: tuple[str, ...] = ()) -> dict[str, Any]:
        prompt = (
            "Clasifica este documento para un reporte periÃ³dico. No inventes hechos. "
            "Devuelve SOLO JSON con relevance, novelty, source_quality, impact, depth, "
            "actionability, content_type, decision y reason. Scores entre 0 y 1. "
            "decision debe ser reporter_only, promote, defer, irrelevant o insufficient_evidence.\n\n"
            f"Intereses opcionales: {', '.join(interests) or 'ninguno'}\n"
            f"TÃ­tulo: {document.get('title', '')}\nTexto:\n{document.get('representative_text') or document.get('text', '')[:2000]}"
        )
        result = self.provider.generate_chat([{"role": "user", "content": prompt}], max_new_tokens=300)
        return self._parse(result.text) if result.ok else {}

    def classify_many(self, documents: list[dict[str, Any]], interests: tuple[str, ...] = (), progress_callback=None) -> list[dict[str, Any]]:
        messages = []
        for document in documents:
            messages.append([{"role": "user", "content": (
                "Clasifica este documento para un reporte periÃ³dico. No inventes hechos. "
                "Devuelve SOLO JSON con relevance, novelty, source_quality, impact, depth, "
                "actionability, content_type, decision y reason. Scores entre 0 y 1. "
                "decision debe ser reporter_only, promote, defer, irrelevant o insufficient_evidence.\n\n"
                f"Intereses opcionales: {', '.join(interests) or 'ninguno'}\n"
                f"TÃ­tulo: {document.get('title', '')}\nTexto:\n{document.get('representative_text') or document.get('text', '')[:2000]}"
            )}])
        if not messages:
            return []
        titles = [doc.get('title', '') for doc in documents]
        def _llm_progress(current: int, total: int) -> None:
            if progress_callback is not None:
                title = titles[current] if current < len(titles) else ''
                progress_callback(current, total, title)
        results = self._generate_resilient(messages, max_new_tokens=300, batch_size=2, progress_callback=_llm_progress)
        return [self._parse(result.text) if result is not None and result.ok else {} for result in results]

    def label(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        evidence = "\n\n".join(f"- {doc.get('title', '')}: {doc.get('text', '')[:500]}" for doc in documents[:8])
        prompt = (
            "Nombra y describe el tema comÃºn de estos documentos. Devuelve SOLO JSON con "
            "label, description, subtopics y uncertainties. No uses una taxonomÃ­a fija.\n\n" + evidence
        )
        result = self.provider.generate_chat([{"role": "user", "content": prompt}], max_new_tokens=250)
        if not result.ok:
            return {}
        value = self._parse(result.text)
        label = value.get("label")
        description = value.get("description")
        if not isinstance(label, str) or not 3 <= len(label.strip()) <= 200:
            return {}
        if not isinstance(description, str) or not 10 <= len(description.strip()) <= 2000:
            return {}
        value["label"] = label.strip()
        value["description"] = description.strip()
        return value

    def label_many(self, groups: list[list[dict[str, Any]]], progress_callback=None) -> list[dict[str, Any]]:
        messages = []
        for documents in groups:
            evidence = "\n\n".join(f"- {doc.get('title', '')}: {doc.get('text', '')[:500]}" for doc in documents[:8])
            messages.append([{"role": "user", "content": (
                "Nombra y describe el tema comÃºn de estos documentos. Devuelve SOLO JSON con "
                "label, description, subtopics y uncertainties. No uses una taxonomÃ­a fija.\n\n" + evidence
            )}])
        if not messages:
            return []
        results = self._generate_resilient(messages, max_new_tokens=250, batch_size=2, progress_callback=progress_callback)
        values = []
        for result in results:
            value = self._parse(result.text) if result is not None and result.ok else {}
            label, description = value.get("label"), value.get("description")
            if not isinstance(label, str) or not 3 <= len(label.strip()) <= 200 or not isinstance(description, str) or not 10 <= len(description.strip()) <= 2000:
                values.append({})
            else:
                values.append({**value, "label": label.strip(), "description": description.strip()})
        return values

    def group_topics(self, topic_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Group fine-grained topics into broader parent categories using the LLM."""
        if not topic_summaries:
            return []
        summaries_text = "\n".join(
            f"{i}: {t.get('label', '')} â€” {t.get('description', '')[:100]} ({t.get('doc_count', 0)} docs)"
            for i, t in enumerate(topic_summaries)
        )
        prompt = (
            "AgrupÃ¡ estos tÃ³picos en categorÃ­as mÃ¡s generales (3-8 categorÃ­as). "
            "Cada categorÃ­a debe agrupar tÃ³picos relacionados temÃ¡ticamente. "
            "Devuelve SOLO un JSON array donde cada elemento tiene: "
            "label (nombre de la categorÃ­a, mÃ¡ximo 4 palabras), "
            "description (1-2 oraciones), "
            "subtopic_ids (array de Ã­ndices numÃ©ricos de los tÃ³picos que pertenecen). "
            "Todos los tÃ³picos deben estar en exactamente una categorÃ­a.\n\n"
            f"TÃ³picos:\n{summaries_text}"
        )
        result = self.provider.generate_chat([{"role": "user", "content": prompt}], max_new_tokens=600)
        if not result.ok:
            return []
        text = result.text
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            return []
        try:
            value = json.loads(text[start:end + 1])
            return value if isinstance(value, list) else []
        except json.JSONDecodeError:
            return []

