"""Estación de montaje de roadmaps (workbench).

Operaciones determinísticas de montaje sobre el contrato Roadmap v2:

- create_draft / add_unit / remove_unit / reorder_units / edit_unit /
  set_stages: CRUD de borrador, cada op es un evento con field_origins.
- run_validation: pasada P2 del pipeline de refinamiento (determinística).
- freeze_draft: draft → proposed; exige spine 3-7 grounded y cero findings
  de severidad error. El gate humano sigue siendo approve_roadmap.
- reopen_draft: proposed → draft (solo pre-aprobación).
- refine_with_findings: pasada P3 (LLM revisor) — contexto corto: solo los
  hallazgos de P2 y las unidades afectadas (RES-005: el 9B degrada en
  cadenas largas).

El LLM nunca decide estado: posición, orden y validez son scaffold.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Callable

from ipa.tutor.tutor_contracts import (
    AssessmentType,
    FieldOrigin,
    GenerationProvenance,
    Roadmap,
    RoadmapStage,
    RoadmapStatus,
    RoadmapUnit,
    RoadmapValidation,
    SourceRef,
    SourceType,
    ValidationFinding,
)


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _hash12(*parts: str) -> str:
    return hashlib.sha256("".join(parts).encode()).hexdigest()[:12]


class RoadmapWorkbench:
    """CRUD determinístico de drafts + validación P2 + revisión P3."""

    def __init__(self, store: Any, *, provider: Any | None = None) -> None:
        self.store = store
        self.provider = provider

    # -- helpers -------------------------------------------------------------

    def _get(self, roadmap_id: str) -> Roadmap:
        roadmap = self.store.get_roadmap(roadmap_id)
        if roadmap is None:
            raise ValueError(f"unknown roadmap: {roadmap_id}")
        return roadmap

    def _save(self, roadmap: Roadmap) -> Roadmap:
        self.store.save_roadmap(roadmap)
        return roadmap

    @staticmethod
    def _generation(generator: str, input_key: str) -> GenerationProvenance:
        now = _now()
        return GenerationProvenance(
            generator=generator,
            generated_at=now,
            input_hash="sha256:" + _hash12(input_key, now).ljust(64, "0"),
            model_fingerprint="deterministic-workbench-v1",
        )

    @staticmethod
    def _unit_field_origins(fields: dict[str, str]) -> dict[str, str]:
        return dict(fields)

    # -- CRUD de borrador -----------------------------------------------------

    def create_draft(
        self,
        goal_id: str,
        *,
        base_roadmap_id: str | None = None,
        change_reason: str | None = None,
    ) -> Roadmap:
        """Borrador vacío (o copia editable de un roadmap existente).

        Con base_roadmap_id: versión N+1 del roadmap base, unidades copiadas
        (editables), gate humano intacto — el base no se muta jamás.
        """
        now = _now()
        version, previous, units = 1, None, []
        if base_roadmap_id is not None:
            base = self._get(base_roadmap_id)
            version = base.version + 1
            previous = base.roadmap_id
            units = [
                replace(u, field_origins={"reason": "user", "concept_id": "user"})
                for u in base.units
            ]
        roadmap = Roadmap(
            roadmap_id=f"roadmap:{_hash12(goal_id, now)}",
            goal_id=goal_id,
            version=version,
            status=RoadmapStatus.DRAFT,
            units=units,
            assumptions=[],
            uncertainties=[],
            change_reason=change_reason or ("revisión desde la estación de montaje" if previous else None),
            previous_roadmap_id=previous,
            created_at=now,
            approval=None,
            generation=self._generation("tutor-workbench", goal_id),
            field_origins={
                "goal_id": "user", "units": "user", "assumptions": "generated",
                "uncertainties": "generated", "change_reason": "user_or_generated",
            },
        )
        return self._save(roadmap)

    def add_unit(
        self,
        roadmap_id: str,
        *,
        concept_id: str,
        reason: str,
        estimated_effort_minutes: int = 30,
        stage_id: str | None = None,
        source_refs: list[SourceRef] | None = None,
        assessment_types: list[str] | None = None,
    ) -> Roadmap:
        roadmap = self._get(roadmap_id)
        if roadmap.status != RoadmapStatus.DRAFT:
            raise ValueError("solo los drafts aceptan unidades (usá create_draft(base_roadmap_id=...) para revisar)")
        if len(roadmap.units) >= 50:
            raise ValueError("el draft alcanzó el máximo de 50 unidades")
        if any(u.concept_id == concept_id for u in roadmap.units):
            raise ValueError(f"concept_id duplicado en el draft: {concept_id}")
        atypes = [
            AssessmentType(at) for at in (assessment_types or ["explanation"])
            if str(at) in {t.value for t in AssessmentType}
        ] or [AssessmentType.EXPLANATION]
        order = len(roadmap.units) + 1
        unit = RoadmapUnit(
            # unit_id por concepto (dup ya rechazado arriba): conservar el
            # concept_id entre versiones hereda el progreso de la unidad.
            unit_id=f"roadmap_unit:{_hash12(roadmap.goal_id, concept_id)}",
            order=order,
            concept_id=concept_id,
            reason=reason[:1000],
            estimated_effort_minutes=max(5, min(1440, int(estimated_effort_minutes))),
            source_refs=list(source_refs or []),
            assessment_types=atypes,
            stage_id=stage_id,
            field_origins=self._unit_field_origins({
                "reason": "user", "concept_id": "user",
                "estimated_effort_minutes": "user", "source_refs": "user",
            }),
        )
        return self._save(replace(roadmap, units=[*roadmap.units, unit]))

    def remove_unit(self, roadmap_id: str, unit_id: str) -> Roadmap:
        roadmap = self._get(roadmap_id)
        if roadmap.status != RoadmapStatus.DRAFT:
            raise ValueError("solo los drafts aceptan quitar unidades")
        remaining = [u for u in roadmap.units if u.unit_id != unit_id]
        if len(remaining) == len(roadmap.units):
            raise ValueError(f"unidad desconocida: {unit_id}")
        # Las expansiones del unit quedarían huérfanas — la unidad ya no
        # existe, la evidencia tampoco debe persistir bajo su clave.
        self.store.delete_expansions(roadmap_id, unit_id)
        remaining = [replace(u, order=i + 1) for i, u in enumerate(remaining)]
        return self._save(replace(roadmap, units=remaining))

    def reorder_units(self, roadmap_id: str, ordered_unit_ids: list[str]) -> Roadmap:
        roadmap = self._get(roadmap_id)
        if roadmap.status != RoadmapStatus.DRAFT:
            raise ValueError("solo los drafts aceptan reordenar")
        by_id = {u.unit_id: u for u in roadmap.units}
        if sorted(ordered_unit_ids) != sorted(by_id):
            raise ValueError("reorder requiere exactamente las unidades del draft")
        units = [replace(by_id[uid], order=i + 1) for i, uid in enumerate(ordered_unit_ids)]
        return self._save(replace(roadmap, units=units))

    def edit_unit(
        self,
        roadmap_id: str,
        unit_id: str,
        *,
        reason: str | None = None,
        estimated_effort_minutes: int | None = None,
        concept_id: str | None = None,
        stage_id: str | None = None,
        clear_stage: bool = False,
        source_refs: list[SourceRef] | None = None,
    ) -> Roadmap:
        roadmap = self._get(roadmap_id)
        if roadmap.status != RoadmapStatus.DRAFT:
            raise ValueError("solo los drafts aceptan editar unidades")
        unit = next((u for u in roadmap.units if u.unit_id == unit_id), None)
        if unit is None:
            raise ValueError(f"unidad desconocida: {unit_id}")
        if concept_id is not None and any(
            u.concept_id == concept_id and u.unit_id != unit_id for u in roadmap.units
        ):
            raise ValueError(f"concept_id duplicado en el draft: {concept_id}")
        origins = dict(unit.field_origins or {})
        updates: dict[str, Any] = {}
        if reason is not None:
            updates["reason"] = reason[:1000]
            origins["reason"] = "user"
        if estimated_effort_minutes is not None:
            updates["estimated_effort_minutes"] = max(5, min(1440, int(estimated_effort_minutes)))
            origins["estimated_effort_minutes"] = "user"
        if concept_id is not None:
            updates["concept_id"] = concept_id
            origins["concept_id"] = "user"
        if clear_stage:
            updates["stage_id"] = None
            origins["stage_id"] = "user"
        elif stage_id is not None:
            updates["stage_id"] = stage_id
            origins["stage_id"] = "user"
        if source_refs is not None:
            updates["source_refs"] = list(source_refs)
            origins["source_refs"] = "user"
        if concept_id is not None and concept_id != unit.concept_id:
            # El ancla cambió: las expansiones L0-L3 del unit describen el
            # material viejo — quedan 'stale' hasta re-expandir.
            self.store.mark_expansions_stale(roadmap_id, unit_id)
        new_unit = replace(unit, field_origins=origins, **updates)
        units = [new_unit if u.unit_id == unit_id else u for u in roadmap.units]
        return self._save(replace(roadmap, units=units))

    def set_stages(self, roadmap_id: str, stages: list[dict[str, Any]]) -> Roadmap:
        roadmap = self._get(roadmap_id)
        if roadmap.status != RoadmapStatus.DRAFT:
            raise ValueError("solo los drafts aceptan editar etapas")
        parsed = [
            RoadmapStage(
                stage_id=str(s["stage_id"]),
                title=str(s["title"])[:200],
                order=int(s["order"]),
            )
            for s in stages
        ]
        return self._save(replace(roadmap, stages=parsed or None))

    # -- P2: validación determinística ----------------------------------------

    def run_validation(
        self,
        roadmap_id: str,
        *,
        corpus_lookup: Callable[[str], list[dict[str, Any]]] | None = None,
        strict: bool = False,
    ) -> Roadmap:
        """Pasada P2: hallazgos determinísticos adjuntos al roadmap.

        corpus_lookup(concept_id) → docs del corpus para ese concepto; si se
        provee, evalúa cobertura material (ungrounded / cluster vacío).
        strict=True (momento del freeze): las condiciones de spine congelada
        (grounding, 3-7 unidades) suben a severidad error.
        """
        roadmap = self._get(roadmap_id)
        is_draft = roadmap.status == RoadmapStatus.DRAFT and not strict
        findings: list[ValidationFinding] = []

        if is_draft and not roadmap.units:
            findings.append(ValidationFinding(
                code="empty_draft", severity="info",
                message="Borrador vacío: agregá unidades desde la estación de montaje."))
        for unit in roadmap.units:
            if not unit.source_refs:
                findings.append(ValidationFinding(
                    code="ungrounded_unit",
                    severity="error" if not is_draft else "warning",
                    unit_id=unit.unit_id,
                    message=f"Unidad {unit.order} sin source_refs: sin ancla en el corpus."))
            if corpus_lookup is not None:
                try:
                    docs = corpus_lookup(unit.concept_id)
                except Exception:
                    docs = []
                if not docs:
                    findings.append(ValidationFinding(
                        code="no_corpus_material",
                        severity="warning",
                        unit_id=unit.unit_id,
                        message=f"Unidad {unit.order}: el corpus no tiene material para {unit.concept_id}."))
            if unit.estimated_effort_minutes > 120:
                findings.append(ValidationFinding(
                    code="effort_outlier", severity="info",
                    unit_id=unit.unit_id,
                    message=f"Unidad {unit.order}: {unit.estimated_effort_minutes} min — considerá dividirla."))
        if len(roadmap.units) < 3 and not is_draft:
            findings.append(ValidationFinding(
                code="too_few_units", severity="error",
                message="La spine congelada requiere 3-7 unidades."))
        if len(roadmap.units) > 7 and not is_draft:
            findings.append(ValidationFinding(
                code="too_many_units", severity="error",
                message="La spine congelada admite 3-7 unidades."))

        validation = RoadmapValidation(run_at=_now(), findings=findings)
        return self._save(replace(roadmap, validation=validation))

    # -- freeze / reopen -------------------------------------------------------

    def freeze_draft(self, roadmap_id: str, *, change_reason: str | None = None) -> Roadmap:
        """draft → proposed. Exige cero findings de severidad error.

        El snapshot queda inmutable (3-7 unidades grounded); el gate humano
        (approve_roadmap) sigue siendo el único camino a active.
        """
        roadmap = self.run_validation(roadmap_id, strict=True)
        errors = [f for f in (roadmap.validation.findings if roadmap.validation else []) if f.severity == "error"]
        if errors:
            raise ValueError(
                "freeze bloqueado por validación P2: "
                + "; ".join(f"{f.code}: {f.message}" for f in errors[:3]))
        reason = change_reason or roadmap.change_reason or "freeze desde la estación de montaje"
        return self._save(replace(
            roadmap, status=RoadmapStatus.PROPOSED, change_reason=reason))

    def reopen_draft(self, roadmap_id: str) -> Roadmap:
        """proposed → draft (solo pre-aprobación; active exige revisión nueva)."""
        roadmap = self._get(roadmap_id)
        if roadmap.status != RoadmapStatus.PROPOSED:
            raise ValueError(f"solo los proposed reabren a draft (status: {roadmap.status.value})")
        return self._save(replace(roadmap, status=RoadmapStatus.DRAFT))

    # -- P3: revisión LLM con contexto corto -----------------------------------

    def refine_with_findings(
        self,
        roadmap_id: str,
        *,
        max_units_touched: int = 3,
    ) -> tuple[Roadmap, list[dict[str, Any]]]:
        """Pasada P3: el LLM revisa SOLO las unidades señaladas por P2.

        Contexto corto por diseño (RES-005): hallazgos + unidades afectadas,
        no el roadmap completo. Devuelve (roadmap, cambios aplicados).
        """
        if self.provider is None:
            raise ValueError("P3 requiere provider (LLM revisor)")
        roadmap = self.run_validation(roadmap_id)
        validation = roadmap.validation
        actionable = [
            f for f in (validation.findings if validation else [])
            if f.severity in {"warning", "error"} and f.unit_id
        ]
        if not actionable:
            return roadmap, []

        unit_by_id = {u.unit_id: u for u in roadmap.units}
        targets: list[RoadmapUnit] = []
        seen: set[str] = set()
        for f in actionable:
            u = unit_by_id.get(f.unit_id or "")
            if u and u.unit_id not in seen and len(targets) < max_units_touched:
                targets.append(u)
                seen.add(u.unit_id)
        if not targets:
            return roadmap, []

        findings_lines = "\n".join(
            f"- [{f.severity}] {f.code}: {f.message}" for f in actionable[:6])
        units_lines = "\n".join(
            f"- {u.unit_id} (orden {u.order}, concepto {u.concept_id}): {u.reason[:120]}"
            for u in targets)
        prompt = (
            "Corregí las unidades señaladas por la validación determinística. "
            "Respondé SOLO JSON:\n"
            '{"units": [{"unit_id": "...", "reason": "<=30 palabras", '
            '"estimated_effort_minutes": 30}]}\n'
            "Solo unidades listadas; no inventes unit_ids.\n\n"
            f"Hallazgos:\n{findings_lines}\n\nUnidades a revisar:\n{units_lines}"
        )
        messages = [
            {"role": "system", "content": "Sos un revisor pedagógico. Respondé solo con el JSON pedido."},
            {"role": "user", "content": prompt},
        ]
        result = self.provider.generate_chat(messages, max_new_tokens=400, temperature=0.0)
        if getattr(result, "error", None):
            raise RuntimeError(f"P3 refinement failed: {result.error}")
        parsed = _extract_json(result.text)
        proposed = {u.get("unit_id"): u for u in (parsed.get("units") or [])
                    if isinstance(u, dict) and u.get("unit_id") in unit_by_id}

        changes: list[dict[str, Any]] = []
        units = list(roadmap.units)
        for i, unit in enumerate(units):
            fix = proposed.get(unit.unit_id)
            if not fix:
                continue
            origins = dict(unit.field_origins or {})
            updates: dict[str, Any] = {}
            new_reason = str(fix.get("reason", "")).strip()
            if new_reason and new_reason != unit.reason:
                updates["reason"] = new_reason[:1000]
                origins["reason"] = "generated"
            try:
                minutes = int(fix.get("estimated_effort_minutes", unit.estimated_effort_minutes))
                if 5 <= minutes <= 1440 and minutes != unit.estimated_effort_minutes:
                    updates["estimated_effort_minutes"] = minutes
                    origins["estimated_effort_minutes"] = "generated"
            except (TypeError, ValueError):
                pass
            if updates:
                units[i] = replace(unit, field_origins=origins, **updates)
                changes.append({"unit_id": unit.unit_id, **updates})
        updated = replace(
            roadmap, units=units,
            generation=self._generation("tutor-workbench-p3", roadmap.roadmap_id),
        )
        return self._save(updated), changes


def _extract_json(text: str) -> dict[str, Any]:
    """Primer objeto JSON balanceado del texto del LLM."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in LLM response")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON in LLM response")
