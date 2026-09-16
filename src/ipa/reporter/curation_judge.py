"""LLM judge para la zona gris de la cascada de promoción.

Diseño:
  - Recibe lotes de 10-20 documentos ambiguos (tier=gray).
  - Usa think_mode (no_think=False) para que el modelo razone sobre evidencia.
  - Devuelve, por documento: promote / reject / defer.
  - El agente orquesta; este módulo solo juzga.
  - No toca DocumentStore ni copia nada — es puramente un clasificador.

Arquitectura:
  - El provider se setea con no_think=False para esta generación puntual.
  - Se restaura el estado original después (el chat del dashboard sigue
    en no_think=True para respuestas directas).
  - Respeta CHAT_BUSY: si el chat está generando, NO usa el provider.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class JudgeVerdict:
    """Veredicto del LLM judge sobre un documento de la zona gris."""
    document_id: str
    verdict: str  # "promote" | "reject" | "defer"
    confidence: float  # 0.0 .. 1.0
    reason: str


class LLMProvider(Protocol):
    """Protocolo mínimo que cumple ExL3Provider (y cualquier fake de test)."""
    def generate_chat(self, messages: list[dict[str, str]], *,
                      max_new_tokens: int | None = None,
                      temperature: float | None = None,
                      stop_sequences: list[str] | None = None,
                      timeout: float | None = None) -> Any: ...
    def is_loaded(self) -> bool: ...


def _build_batch_prompt(docs: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Construye el prompt del juez: lotes de docs con contexto comparativo.

    El modelo ve todos los docs del lote juntos, lo que le permite comparar
    y razonar sobre cuáles merecen promoción relativa al resto.
    """
    items = []
    for i, doc in enumerate(docs, 1):
        title = str(doc.get("title", ""))[:120]
        text = str(doc.get("text", ""))[:800]
        domain = str(doc.get("source_domain", ""))
        score = doc.get("promotion_score", 0.0)
        items.append(
            f"--- Documento {i} ---\n"
            f"ID: {doc['document_id']}\n"
            f"Título: {title}\n"
            f"Dominio: {domain}\n"
            f"Score heurístico: {score:.2f}\n"
            f"Contenido (extracto):\n{text}\n"
        )
    batch_text = "\n".join(items)

    system = (
        "Sos un juez de curación de documentos para un corpus de conocimiento personal. "
        "Recibís un lote de documentos que la heurística determinística no pudo clasificar "
        "con confianza (zona gris). Tu trabajo es decidir, para cada documento, si merece "
        "ser promovido al corpus principal (promote), rechazado (reject), o dejado para "
        "revisión humana explícita (defer).\n\n"
        "Criterios:\n"
        "- promote: contenido relevante, novedoso, de fuente confiable, con sustancia. "
        "No lo promuevas solo porque parece bien escrito — evaluá si aporta valor real.\n"
        "- reject: irrelevante, duplicado conceptual, superficial, o de fuente no confiable.\n"
        "- defer: genuinamente ambiguo. Usá defer solo cuando no podés decidir con la "
        "evidencia disponible — no como cajón de sastre.\n\n"
        "Pensá paso a paso sobre cada documento (podés razonar internamente). "
        "Luego devolvé EXCLUSIVAMENTE un JSON array con un objeto por documento:\n\n"
        '[{"document_id": "...", "verdict": "promote|reject|defer", '
        '"confidence": 0.0-1.0, "reason": "..."}]\n\n'
        "No agregues texto fuera del JSON."
    )
    user = (
        f"Juzgá estos {len(docs)} documentos de la zona gris. "
        f"Comparalos entre sí — el contexto comparativo es la ventaja del lote.\n\n"
        f"{batch_text}\n\n"
        "Devolvé el JSON array ahora."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse_verdicts(raw: str, doc_ids: list[str]) -> list[JudgeVerdict]:
    """Extrae veredictos del output del LLM. Tolerante a texto extra."""
    # Buscar el primer JSON array en el output
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if not match:
        # Fallback: todo defer
        return [JudgeVerdict(did, "defer", 0.0, "LLM no produjo JSON válido") for did in doc_ids]
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return [JudgeVerdict(did, "defer", 0.0, "LLM produjo JSON inválido") for did in doc_ids]
    verdicts: list[JudgeVerdict] = []
    seen: set[str] = set()
    for item in items:
        did = str(item.get("document_id", ""))
        v = str(item.get("verdict", "defer")).lower().strip()
        if v not in ("promote", "reject", "defer"):
            v = "defer"
        conf = float(item.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))
        reason = str(item.get("reason", ""))[:300]
        if did in doc_ids and did not in seen:
            verdicts.append(JudgeVerdict(did, v, conf, reason))
            seen.add(did)
    # Fallback para docs que el LLM omitió
    for did in doc_ids:
        if did not in seen:
            verdicts.append(JudgeVerdict(did, "defer", 0.0, "LLM omitió este documento"))
    return verdicts


def judge_gray_batch(
    docs: list[dict[str, Any]],
    provider: LLMProvider,
    *,
    batch_size: int = 15,
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
) -> list[JudgeVerdict]:
    """Juzga un lote de documentos de la zona gris con think_mode.

    Args:
        docs: lista de dicts con document_id, title, text, source_domain,
              promotion_score.
        provider: LLM provider (ExL3Provider) con no_think=False temporal.
        batch_size: docs por llamada al LLM (10-20 recomendado).
        max_new_tokens: límite de generación (think + JSON).
        temperature: baja para juicio consistente.

    Returns:
        Lista de JudgeVerdict, uno por documento de entrada, en orden.
    """
    if not docs:
        return []
    if not provider.is_loaded():
        return [JudgeVerdict(d["document_id"], "defer", 0.0, "Provider no cargado") for d in docs]

    all_verdicts: list[JudgeVerdict] = []
    doc_ids = [d["document_id"] for d in docs]

    # Procesar en sub-lotes para no exceder el contexto
    for start in range(0, len(docs), batch_size):
        sub = docs[start:start + batch_size]
        sub_ids = [d["document_id"] for d in sub]
        messages = _build_batch_prompt(sub)
        try:
            result = provider.generate_chat(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_sequences=["<|im_end|>", "</s>", "<|im_start|>"],
            )
            raw = getattr(result, "text", "") or str(result)
        except Exception as exc:
            all_verdicts.extend(
                JudgeVerdict(did, "defer", 0.0, f"Error de generación: {exc}")
                for did in sub_ids
            )
            continue
        all_verdicts.extend(_parse_verdicts(raw, sub_ids))

    # Asegurar orden y completitud
    by_id = {v.document_id: v for v in all_verdicts}
    return [by_id.get(did, JudgeVerdict(did, "defer", 0.0, "Faltante"))
            for did in doc_ids]


__all__ = [
    "JudgeVerdict",
    "LLMProvider",
    "judge_gray_batch",
]
