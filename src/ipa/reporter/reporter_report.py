"""Reporter report materialization and Markdown rendering."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ipa.reporter.reporter_contracts import ReportStatus, generation_provenance, sha256_hash


def build_report(
    report_id: str,
    corpus_id: str,
    period: dict[str, str],
    categories: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    source_refs: list[dict[str, Any]],
    uncertainties: list[str] | None = None,
    cross_cutting_signals: list[str] | None = None,
    recommended_readings: list[dict[str, Any]] | None = None,
    parent_categories: list[dict[str, Any]] | None = None,
    status: str = ReportStatus.DRAFT.value,
) -> dict[str, Any]:
    payload_hash = sha256_hash(json.dumps({"categories": categories, "decisions": decisions, "parent_categories": parent_categories or []}, sort_keys=True))
    counts: dict[str, int] = {}
    for decision in decisions:
        key = str(decision.get("decision", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    readings = recommended_readings or []
    if not readings:
        for category in categories[:5]:
            if category.get("document_ids"):
                readings.append({
                    "document_id": category["document_ids"][0],
                    "reason": f"Documento representativo de {category['label']}.",
                    "priority": category.get("importance", 0.5),
                })
    return {
        "report_id": report_id,
        "corpus_id": corpus_id,
        "period": period,
        "status": status,
        "categories": categories,
        "parent_categories": parent_categories or [],
        "cross_cutting_signals": cross_cutting_signals or [],
        "recommended_readings": readings,
        "uncertainties": uncertainties or [],
        "curation_summary": counts,
        "source_refs": source_refs,
        "generation": generation_provenance(payload_hash),
        "field_origins": {
            "categories": "generated",
            "cross_cutting_signals": "generated",
            "recommended_readings": "generated",
            "uncertainties": "generated",
            "curation_summary": "system",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    period = report["period"]
    lines = [f"# Reporter â€” {period['label']}", "", f"PerÃ­odo: `{period['start']}` â†’ `{period['end']}`", ""]
    lines.append("## Panorama emergente")
    lines.append("")
    if not report["categories"]:
        lines.append("No se detectaron categorÃ­as con la configuraciÃ³n actual.")
    for index, category in enumerate(report["categories"], 1):
        lines.extend([
            f"### {index}. {category['label']}",
            "",
            category["description"],
            "",
            f"Documentos: {category['document_count']} Â· Fuentes: {category['source_count']} Â· "
            f"Importancia: {category['importance']:.2f} Â· Novedad: {category['novelty']:.2f}",
            f"EvoluciÃ³n: `{category['evolution']}`",
            "",
        ])
        if category.get("subtopics"):
            lines.append("Subtemas:")
            lines.extend(f"- {topic}" for topic in category["subtopics"])
            lines.append("")
        if category.get("uncertainties"):
            lines.append("Incertidumbres:")
            lines.extend(f"- {item}" for item in category["uncertainties"])
            lines.append("")
        lines.append("Fuentes representativas:")
        lines.extend(f"- `{source['source_id']}`" for source in category["representative_sources"])
        lines.append("")
    lines.extend(["## SeÃ±ales transversales", ""])
    signals = report.get("cross_cutting_signals", [])
    if signals:
        lines.extend(f"- {signal}" for signal in signals)
    else:
        lines.append("- No se detectaron seÃ±ales transversales.")
    lines.extend(["", "## Lecturas recomendadas", ""])
    for reading in report.get("recommended_readings", []):
        lines.append(f"- `{reading['document_id']}` â€” {reading['reason']} (prioridad {reading['priority']:.2f})")
    lines.extend(["", "## Incertidumbres", ""])
    report_uncertainties = report.get("uncertainties", [])
    if report_uncertainties:
        lines.extend(f"- {item}" for item in report_uncertainties)
    else:
        lines.append("- No registradas.")
    lines.extend(["", "## CuraciÃ³n", "", "```json", json.dumps(report["curation_summary"], indent=2, ensure_ascii=False), "```"])
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path

