"""Generador de expansiones de profundidad (dimensión Y del roadmap).

Determinístico: dado (unidad, nivel), expande evidencia del corpus —
el LLM solo renderea lo que este módulo produce (PAT-004 EvidenceSet).

Niveles:
  L0 OVERVIEW      — cabeza del documento primario de la unidad
  L1 CORE          — chunks referenciados por source_refs de la unidad
  L2 CLAIMS        — bloques [Key facts] enriquecidos de esos chunks
  L3 NEIGHBORHOOD  — documentos hermanos del topic cluster del concepto
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ipa.tutor.tutor_contracts import (
    ExpansionDepth,
    GenerationProvenance,
    RoadmapExpansion,
    RoadmapUnit,
    SourceRef,
    SourceType,
)

_CLAIMS_MARKER = "[Key facts]"
_MAX_SUMMARY_CHARS = 3800


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _hash12(*parts: str) -> str:
    import hashlib
    return hashlib.sha256("".join(parts).encode()).hexdigest()[:12]


class ExpansionGenerator:
    """Expande (unidad, profundidad) → RoadmapExpansion con evidencia real."""

    def __init__(self, corpus_dir: Path | str, clusters_db: Path | str | None = None) -> None:
        from ipa.storage.document_store import DocumentStore
        self.corpus_dir = Path(corpus_dir)
        self.store = DocumentStore(self.corpus_dir / "document_store.db")
        self.clusters_db = Path(clusters_db) if clusters_db else None

    def close(self) -> None:
        try:
            self.store.close()
        except Exception:
            pass

    # -- cluster lookup -------------------------------------------------------

    def _cluster_for_doc(self, document_id: str) -> dict[str, Any] | None:
        if not self.clusters_db or not self.clusters_db.exists():
            return None
        try:
            conn = sqlite3.connect(str(self.clusters_db))
            rows = conn.execute("SELECT payload_json FROM topic_clusters").fetchall()
            conn.close()
        except sqlite3.Error:
            return None
        for (payload,) in rows:
            try:
                data = json.loads(payload)
            except ValueError:
                continue
            if document_id in (data.get("member_document_ids") or []):
                return data
        return None

    # -- niveles --------------------------------------------------------------

    def _overview(self, unit: RoadmapUnit) -> tuple[str, list[SourceRef]]:
        doc_id = unit.concept_id
        text = ""
        try:
            doc = self.store.get_document(doc_id)
            text = (doc.text or "") if doc else ""
        except Exception:
            text = ""
        if not text:
            chunk = self.store.get_chunk(doc_id)
            text = (chunk.text or "") if chunk else ""
        return text[:_MAX_SUMMARY_CHARS], [SourceRef(
            source_id=doc_id, source_type=SourceType.DOCUMENT)]

    def _chunks_for_ref(self, ref: SourceRef) -> list[Any]:
        """Resuelve un source_ref a chunks reales.

        Los roadmaps actuales guardan doc-ids con source_type=CHUNK
        (_shape_units); el fallback doc→chunks los absorbe.
        """
        chunk = self.store.get_chunk(ref.source_id)
        if chunk is not None:
            return [chunk]
        return list(self.store.get_chunks(ref.source_id))[:3]

    def _core(self, unit: RoadmapUnit) -> tuple[str, list[SourceRef]]:
        parts: list[str] = []
        refs: list[SourceRef] = []
        for ref in unit.source_refs:
            for chunk in self._chunks_for_ref(ref):
                parts.append((chunk.text or "")[:1200])
                refs.append(SourceRef(source_id=chunk.chunk_id, source_type=SourceType.CHUNK))
        return "\n\n".join(parts)[:_MAX_SUMMARY_CHARS], refs

    def _claims(self, unit: RoadmapUnit) -> tuple[str, list[SourceRef]]:
        parts: list[str] = []
        refs: list[SourceRef] = []
        for ref in unit.source_refs:
            for chunk in self._chunks_for_ref(ref):
                text = chunk.text or ""
                idx = text.find(_CLAIMS_MARKER)
                if idx < 0:
                    continue
                block = text[idx + len(_CLAIMS_MARKER):].strip()
                end = block.find("\n[")
                if end >= 0:
                    block = block[:end]
                if block:
                    parts.append(block[:1200])
                    refs.append(SourceRef(source_id=chunk.chunk_id, source_type=SourceType.CHUNK))
        return "\n\n".join(parts)[:_MAX_SUMMARY_CHARS], refs

    def _neighborhood(self, unit: RoadmapUnit) -> tuple[str, list[SourceRef], dict[str, Any] | None]:
        cluster = self._cluster_for_doc(unit.concept_id)
        if not cluster:
            return "", [], None
        siblings = [d for d in (cluster.get("member_document_ids") or [])
                    if d != unit.concept_id][:4]
        parts: list[str] = []
        refs: list[SourceRef] = []
        for doc_id in siblings:
            chunk = next(self.store.get_chunks(doc_id), None)
            if chunk is None:
                continue
            parts.append((chunk.text or "")[:800])
            refs.append(SourceRef(source_id=doc_id, source_type=SourceType.DOCUMENT))
        coverage = {
            "cluster_id": cluster.get("cluster_id"),
            "docs_total": len(cluster.get("member_document_ids") or []),
            "docs_covered": len(siblings) + 1,
        }
        return "\n\n".join(parts)[:_MAX_SUMMARY_CHARS], refs, coverage

    # -- API -------------------------------------------------------------------

    def generate(self, roadmap_id: str, unit: RoadmapUnit, depth: int) -> RoadmapExpansion:
        level = ExpansionDepth(depth)
        coverage = None
        if level is ExpansionDepth.OVERVIEW:
            summary, refs = self._overview(unit)
            title = f"Overview: {unit.reason[:80] or unit.concept_id}"
        elif level is ExpansionDepth.CORE:
            summary, refs = self._core(unit)
            title = f"Material nuclear — unidad {unit.order}"
        elif level is ExpansionDepth.CLAIMS:
            summary, refs = self._claims(unit)
            title = f"Claims enriquecidos — unidad {unit.order}"
        else:
            summary, refs, coverage = self._neighborhood(unit)
            title = f"Vecindario del cluster — unidad {unit.order}"
        if not refs:
            raise ValueError(
                f"sin material de corpus para la unidad {unit.order} "
                f"(nivel L{depth}): el concepto {unit.concept_id} no está en el corpus")
        now = _now()
        return RoadmapExpansion(
            expansion_id=f"expansion:{_hash12(roadmap_id, unit.unit_id, str(int(level)), now)}",
            roadmap_id=roadmap_id,
            unit_id=unit.unit_id,
            depth_level=level,
            title=title[:200],
            summary=summary or None,
            evidence_refs=refs,
            coverage=coverage,
            created_at=now,
            generation=GenerationProvenance(
                generator="tutor-expansion",
                generated_at=now,
                input_hash="sha256:" + _hash12(roadmap_id, unit.unit_id, str(int(level))).ljust(64, "0"),
                model_fingerprint="deterministic-expansion-v1",
            ),
            field_origins={"title": "generated", "evidence_refs": "source"},
        )
