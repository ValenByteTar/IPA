"""Calibración de la cascada de curación contra decisiones del LLM judge.

Opción 3 del diseño: usar el LLM judge (con think_mode) como pseudo-ground-truth
para calibrar los percentiles de la cascada determinística.

Flujo:
  1. Tomar una muestra representativa de documentos del corpus del reporter.
  2. Correr curate_documents() → decisions → classify_tiers().
  3. Correr judge_gray_batch() sobre TODA la muestra (no solo la zona gris)
     para tener etiquetas LLM sobre auto_promote y auto_reject también.
  4. Para cada tier, comparar la decisión del LLM contra el tier asignado:
     - Si el LLM dice "promote" pero el tier es auto_reject → el percentil
       auto_reject_percentile es demasiado alto (rechazando buenos docs).
     - Si el LLM dice "reject" pero el tier es auto_promote → el percentil
       auto_promote_percentile es demasiado bajo (promoviendo malos docs).
  5. Ajustar los percentiles para minimizar disagreement con el LLM.
  6. Persistir los parámetros calibrados.

El resultado es un JSON con:
  - auto_promote_percentile óptimo
  - auto_reject_percentile óptimo
  - métricas de agreement/disagreement
  - tamaño de la muestra
  - fecha de calibración

Esto hace que la heurística determinística quede calibrada contra el juicio
semántico del LLM, sin necesidad de etiquetas humanas. El LLM se usa una vez
para calibrar; después la cascada corre sin LLM (excepto la zona gris).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ipa.reporter.reporter_contracts import ReporterDecision, ReporterDocumentDecision
from ipa.reporter.reporter_curation import (
    CurationTier,
    classify_tiers,
    promotion_score,
)
from ipa.reporter.curation_judge import JudgeVerdict, LLMProvider, judge_gray_batch

DEFAULT_CALIBRATION_PATH = Path("outputs/reporter/curation_calibration.json")


@dataclass(frozen=True)
class CalibrationResult:
    """Resultado de una calibración de la cascada."""
    auto_promote_percentile: float
    auto_reject_percentile: float
    sample_size: int
    agreement: float  # fracción de docs donde tier coincide con veredicto LLM
    false_promotes: int  # LLM=reject pero tier=auto_promote
    false_rejects: int  # LLM=promote pero tier=auto_reject
    gray_correct: int  # LLM=promote/reject en zona gris (judge útil)
    gray_defer: int  # LLM=defer en zona gris
    calibrated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _verdict_to_tier(verdict: str) -> str:
    """Mapea veredicto del LLM al tier esperado."""
    if verdict == "promote":
        return "auto_promote"
    if verdict == "reject":
        return "auto_reject"
    return "gray"  # defer → debería estar en zona gris


def _search_optimal_percentiles(
    decisions: list[ReporterDocumentDecision],
    verdicts: list[JudgeVerdict],
    *,
    promote_range: tuple[float, float] = (0.60, 0.95),
    reject_range: tuple[float, float] = (0.20, 0.60),
    step: float = 0.05,
) -> tuple[float, float, float]:
    """Busca los percentiles que maximizan agreement con el LLM.

    Returns: (best_promote_pct, best_reject_pct, best_agreement)
    """
    verdict_by_id = {v.document_id: v for v in verdicts}
    best = (0.80, 0.40, 0.0)

    p = promote_range[0]
    while p <= promote_range[1]:
        r = reject_range[0]
        while r < p:
            tiers = classify_tiers(decisions, auto_promote_percentile=p, auto_reject_percentile=r)
            agree = 0
            total = 0
            for tier in tiers:
                v = verdict_by_id.get(tier.document_id)
                if v is None:
                    continue
                total += 1
                expected = _verdict_to_tier(v.verdict)
                # auto_promote y auto_reject deben coincidir; gray siempre "acierta"
                # porque el LLM los va a juzgar de todos modos
                if expected == "gray":
                    agree += 1
                elif tier.tier == expected:
                    agree += 1
            agreement = agree / total if total else 0.0
            if agreement > best[2]:
                best = (p, r, agreement)
            r += step
        p += step
    return best


def calibrate_curation(
    documents: list[dict[str, Any]],
    decisions: list[ReporterDocumentDecision],
    provider: LLMProvider,
    *,
    sample_size: int = 100,
    output_path: Path | str = DEFAULT_CALIBRATION_PATH,
) -> CalibrationResult:
    """Calibra los percentiles de la cascada contra decisiones del LLM judge.

    Args:
        documents: documentos del reporter (con text, title, source_domain).
        decisions: ReporterDocumentDecision ya calculados por curate_documents().
        provider: LLM provider cargado (con think_mode).
        sample_size: tamaño de la muestra para calibrar (default 100).
        output_path: dónde persistir el resultado.

    Returns:
        CalibrationResult con los percentiles óptimos y métricas.
    """
    # 1. Muestrear: priorizar candidatos PROMOTE (los que entran al ranking)
    #    más algunos hard_reject para validar que el LLM también los rechaza.
    promote_indices = [i for i, d in enumerate(decisions)
                       if d.decision == ReporterDecision.PROMOTE and d.duplicate_of is None]
    other_indices = [i for i, d in enumerate(decisions)
                     if i not in promote_indices]

    # Muestra estratificada: 70% PROMOTE, 30% otros
    n_promote = min(int(sample_size * 0.7), len(promote_indices))
    n_other = min(sample_size - n_promote, len(other_indices))
    sampled = promote_indices[:n_promote] + other_indices[:n_other]

    if len(sampled) < 10:
        # Muestra demasiado chica para calibrar
        return CalibrationResult(
            auto_promote_percentile=0.80, auto_reject_percentile=0.40,
            sample_size=len(sampled), agreement=0.0,
            false_promotes=0, false_rejects=0,
            gray_correct=0, gray_defer=0,
            calibrated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    # 2. Preparar docs para el juez
    sampled_decisions = [decisions[i] for i in sampled]
    sampled_docs = []
    for i in sampled:
        doc = documents[i] if i < len(documents) else {}
        sampled_docs.append({
            "document_id": decisions[i].document_id,
            "title": doc.get("title", ""),
            "text": doc.get("text", ""),
            "source_domain": doc.get("source_domain", ""),
            "promotion_score": promotion_score(decisions[i].scores),
        })

    # 3. Correr el juez sobre TODA la muestra (no solo zona gris)
    verdicts = judge_gray_batch(sampled_docs, provider, batch_size=15)

    # 4. Buscar percentiles óptimos
    best_p, best_r, best_agreement = _search_optimal_percentiles(
        sampled_decisions, verdicts
    )

    # 5. Calcular métricas con los percentiles óptimos
    final_tiers = classify_tiers(
        sampled_decisions,
        auto_promote_percentile=best_p,
        auto_reject_percentile=best_r,
    )
    verdict_by_id = {v.document_id: v for v in verdicts}
    false_promotes = 0
    false_rejects = 0
    gray_correct = 0
    gray_defer = 0
    for tier in final_tiers:
        v = verdict_by_id.get(tier.document_id)
        if v is None:
            continue
        if tier.tier == "auto_promote" and v.verdict == "reject":
            false_promotes += 1
        elif tier.tier == "auto_reject" and v.verdict == "promote":
            false_rejects += 1
        elif tier.tier == "gray":
            if v.verdict in ("promote", "reject"):
                gray_correct += 1
            else:
                gray_defer += 1

    result = CalibrationResult(
        auto_promote_percentile=round(best_p, 4),
        auto_reject_percentile=round(best_r, 4),
        sample_size=len(sampled),
        agreement=round(best_agreement, 4),
        false_promotes=false_promotes,
        false_rejects=false_rejects,
        gray_correct=gray_correct,
        gray_defer=gray_defer,
        calibrated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    # 6. Persistir
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                   encoding="utf-8")

    return result


def load_calibration(path: Path | str = DEFAULT_CALIBRATION_PATH) -> dict[str, Any] | None:
    """Carga parámetros calibrados si existen."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


__all__ = [
    "CalibrationResult",
    "calibrate_curation",
    "load_calibration",
]
