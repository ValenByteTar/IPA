"""Estación de montaje de roadmaps: CRUD de drafts, P2 determinística, P3 LLM.

Covers:
  - create_draft: vacío y como revisión de un roadmap base (versión N+1)
  - add_unit / remove_unit / reorder_units / edit_unit: invariantes y renumeración
  - set_stages: etapas válidas y rechazo de stage desconocido
  - run_validation (P2): hallazgos determinísticos según estado
  - freeze_draft: bloqueado con errores, ok con spine 3-7 grounded
  - reopen_draft: solo proposed → draft
  - refine_with_findings (P3): contexto corto, solo unidades afectadas
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "validation"))

from ipa.tutor.tutor_contracts import (  # noqa: E402
    GenerationProvenance,
    Roadmap,
    RoadmapStatus,
    RoadmapUnit,
    SourceRef,
    SourceType,
)
from ipa.tutor.tutor_runtime import TutorStore  # noqa: E402
from ipa.tutor.tutor_workbench import RoadmapWorkbench  # noqa: E402


class FakeGenerationResult:
    def __init__(self, text: str = "", error: str | None = None):
        self.text = text
        self.error = error


class FakeProvider:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def generate_chat(self, messages, *, max_new_tokens=None, temperature=None, **kw):
        self.calls.append({"messages": messages, "max_new_tokens": max_new_tokens})
        return FakeGenerationResult(text=self.responses.pop(0) if self.responses else "{}")


@pytest.fixture()
def wb_env(tmp_path):
    store = TutorStore(tmp_path / "tutor.db")
    return RoadmapWorkbench(store), store


def _source(sid: str) -> SourceRef:
    return SourceRef(source_id=sid, source_type=SourceType.CHUNK)


def _grounded_unit(index: int, concept_id: str) -> RoadmapUnit:
    return RoadmapUnit(
        unit_id=f"roadmap_unit:seed{index:04d}",
        order=index,
        concept_id=concept_id,
        reason=f"unidad {index} sembrada",
        estimated_effort_minutes=30,
        source_refs=[_source(f"chunk:{concept_id}")],
        assessment_types=["explanation"],
    )


# ---------------------------------------------------------------------------
# create_draft
# ---------------------------------------------------------------------------

def test_create_draft_empty(wb_env):
    wb, _ = wb_env
    draft = wb.create_draft("goal:x")
    assert draft.status == RoadmapStatus.DRAFT
    assert draft.units == []
    assert draft.version == 1
    assert draft.previous_roadmap_id is None
    assert wb.store.get_roadmap(draft.roadmap_id) is not None


def test_create_draft_from_base_copies_units_as_user(wb_env):
    wb, store = wb_env
    base = Roadmap(
        roadmap_id="roadmap:base000000001", goal_id="goal:x", version=2,
        status=RoadmapStatus.PROPOSED,
        units=[_grounded_unit(1, "concept:a"), _grounded_unit(2, "concept:b"),
               _grounded_unit(3, "concept:c")],
        assumptions=[], uncertainties=[], change_reason="v2", previous_roadmap_id="roadmap:v1",
        created_at="2026-09-25T00:00:00Z", approval=None,
        generation=GenerationProvenance("t", "2026-09-25T00:00:00Z",
                                        "sha256:" + "0" * 64, "fp"),
        field_origins={"goal_id": "user", "units": "generated", "assumptions": "generated",
                       "uncertainties": "generated", "change_reason": "user_or_generated"},
    )
    store.save_roadmap(base)
    draft = wb.create_draft("goal:x", base_roadmap_id=base.roadmap_id)
    assert draft.version == 3
    assert draft.previous_roadmap_id == base.roadmap_id
    assert len(draft.units) == 3
    assert all(u.field_origins == {"reason": "user", "concept_id": "user"} for u in draft.units)
    # el base no se muta
    assert store.get_roadmap(base.roadmap_id).status == RoadmapStatus.PROPOSED


# ---------------------------------------------------------------------------
# add / remove / reorder / edit
# ---------------------------------------------------------------------------

def test_add_unit_appends_and_rejects_duplicates(wb_env):
    wb, _ = wb_env
    draft = wb.create_draft("goal:x")
    wb.add_unit(draft.roadmap_id, concept_id="concept:a", reason="primera")
    wb.add_unit(draft.roadmap_id, concept_id="concept:b", reason="segunda")
    roadmap = wb.store.get_roadmap(draft.roadmap_id)
    assert [u.order for u in roadmap.units] == [1, 2]
    assert roadmap.units[0].field_origins["reason"] == "user"
    with pytest.raises(ValueError, match="duplicado"):
        wb.add_unit(draft.roadmap_id, concept_id="concept:a", reason="repetida")


def test_add_unit_rejected_on_frozen(wb_env):
    wb, store = wb_env
    frozen = Roadmap(
        roadmap_id="roadmap:frozen0000001", goal_id="goal:x", version=1,
        status=RoadmapStatus.PROPOSED,
        units=[_grounded_unit(1, "concept:a"), _grounded_unit(2, "concept:b"),
               _grounded_unit(3, "concept:c")],
        assumptions=[], uncertainties=[], change_reason=None, previous_roadmap_id=None,
        created_at="2026-09-25T00:00:00Z", approval=None,
        generation=GenerationProvenance("t", "2026-09-25T00:00:00Z", "sha256:" + "0" * 64, "fp"),
        field_origins={"goal_id": "user", "units": "generated", "assumptions": "generated",
                       "uncertainties": "generated", "change_reason": "user_or_generated"},
    )
    store.save_roadmap(frozen)
    with pytest.raises(ValueError, match="solo los drafts"):
        wb.add_unit(frozen.roadmap_id, concept_id="concept:d", reason="x")


def test_remove_unit_renumbers(wb_env):
    wb, _ = wb_env
    draft = wb.create_draft("goal:x")
    for cid in ("concept:a", "concept:b", "concept:c"):
        wb.add_unit(draft.roadmap_id, concept_id=cid, reason=f"u-{cid}")
    roadmap = wb.store.get_roadmap(draft.roadmap_id)
    victim = next(u for u in roadmap.units if u.concept_id == "concept:b")
    updated = wb.remove_unit(draft.roadmap_id, victim.unit_id)
    assert [u.order for u in updated.units] == [1, 2]
    assert [u.concept_id for u in updated.units] == ["concept:a", "concept:c"]


def test_reorder_units(wb_env):
    wb, _ = wb_env
    draft = wb.create_draft("goal:x")
    for cid in ("concept:a", "concept:b", "concept:c"):
        wb.add_unit(draft.roadmap_id, concept_id=cid, reason=f"u-{cid}")
    roadmap = wb.store.get_roadmap(draft.roadmap_id)
    ids = [u.unit_id for u in roadmap.units]
    updated = wb.reorder_units(draft.roadmap_id, [ids[2], ids[0], ids[1]])
    assert [u.concept_id for u in updated.units] == ["concept:c", "concept:a", "concept:b"]
    assert [u.order for u in updated.units] == [1, 2, 3]


def test_edit_unit_updates_origins(wb_env):
    wb, _ = wb_env
    draft = wb.create_draft("goal:x")
    wb.add_unit(draft.roadmap_id, concept_id="concept:a", reason="original")
    roadmap = wb.store.get_roadmap(draft.roadmap_id)
    unit = roadmap.units[0]
    updated = wb.edit_unit(draft.roadmap_id, unit.unit_id, reason="editada por el usuario",
                           estimated_effort_minutes=45)
    u = updated.units[0]
    assert u.reason == "editada por el usuario"
    assert u.estimated_effort_minutes == 45
    assert u.field_origins["reason"] == "user"
    assert u.field_origins["estimated_effort_minutes"] == "user"


def test_set_stages_and_unknown_stage_rejected(wb_env):
    wb, _ = wb_env
    draft = wb.create_draft("goal:x")
    updated = wb.set_stages(draft.roadmap_id, [
        {"stage_id": "stage:1", "title": "Fundamentos", "order": 1},
        {"stage_id": "stage:2", "title": "Avanzado", "order": 2},
    ])
    assert [s.stage_id for s in updated.stages] == ["stage:1", "stage:2"]
    wb.add_unit(draft.roadmap_id, concept_id="concept:a", reason="x", stage_id="stage:1")
    with pytest.raises(ValueError, match="unknown stage"):
        wb.add_unit(draft.roadmap_id, concept_id="concept:b", reason="y", stage_id="stage:9")


# ---------------------------------------------------------------------------
# P2: run_validation
# ---------------------------------------------------------------------------

def test_validation_flags_ungrounded_as_warning_in_draft(wb_env):
    wb, _ = wb_env
    draft = wb.create_draft("goal:x")
    wb.add_unit(draft.roadmap_id, concept_id="concept:a", reason="sin ancla")
    validated = wb.run_validation(draft.roadmap_id)
    codes = {(f.code, f.severity) for f in validated.validation.findings}
    assert ("ungrounded_unit", "warning") in codes


def test_validation_with_corpus_lookup(wb_env):
    wb, _ = wb_env
    draft = wb.create_draft("goal:x")
    wb.add_unit(draft.roadmap_id, concept_id="concept:a", reason="x",
                source_refs=[_source("chunk:concept-a")])
    validated = wb.run_validation(
        draft.roadmap_id, corpus_lookup=lambda cid: [] if cid == "concept:a" else [{"x": 1}])
    codes = {f.code for f in validated.validation.findings}
    assert "no_corpus_material" in codes


# ---------------------------------------------------------------------------
# freeze / reopen
# ---------------------------------------------------------------------------

def _seed_grounded_draft(wb, goal: str = "goal:x") -> str:
    draft = wb.create_draft(goal)
    for i, cid in enumerate(("concept:a", "concept:b", "concept:c"), start=1):
        wb.add_unit(draft.roadmap_id, concept_id=cid, reason=f"u{i}",
                    source_refs=[_source(f"chunk:{cid}")])
    return draft.roadmap_id


def test_freeze_blocked_on_ungrounded(wb_env):
    wb, _ = wb_env
    draft = wb.create_draft("goal:x")
    wb.add_unit(draft.roadmap_id, concept_id="concept:a", reason="sin ancla")
    with pytest.raises(ValueError, match="freeze bloqueado"):
        wb.freeze_draft(draft.roadmap_id)


def test_freeze_ok_with_grounded_spine(wb_env):
    wb, _ = wb_env
    rid = _seed_grounded_draft(wb)
    frozen = wb.freeze_draft(rid)
    assert frozen.status == RoadmapStatus.PROPOSED
    assert all(u.source_refs for u in frozen.units)
    assert frozen.validation is not None


def test_reopen_draft_only_from_proposed(wb_env):
    wb, _ = wb_env
    rid = _seed_grounded_draft(wb)
    frozen = wb.freeze_draft(rid)
    reopened = wb.reopen_draft(frozen.roadmap_id)
    assert reopened.status == RoadmapStatus.DRAFT
    with pytest.raises(ValueError, match="solo los proposed"):
        wb.reopen_draft(frozen.roadmap_id)


# ---------------------------------------------------------------------------
# P3: refine_with_findings
# ---------------------------------------------------------------------------

def test_refine_revises_only_flagged_units_with_short_context(wb_env):
    wb, _ = wb_env
    rid = _seed_grounded_draft(wb)
    # una unidad queda sin ancla → P2 la marca; P3 debe tocar solo esa
    roadmap = wb.store.get_roadmap(rid)
    victim = roadmap.units[0]
    wb.edit_unit(rid, victim.unit_id, source_refs=[])
    provider = FakeProvider(responses=[json.dumps({
        "units": [{"unit_id": victim.unit_id, "reason": "razón corregida",
                   "estimated_effort_minutes": 25}],
    })])
    wb.provider = provider
    updated, changes = wb.refine_with_findings(rid)
    assert len(changes) == 1
    assert changes[0]["unit_id"] == victim.unit_id
    fixed = next(u for u in updated.units if u.unit_id == victim.unit_id)
    assert fixed.reason == "razón corregida"
    assert fixed.estimated_effort_minutes == 25
    assert fixed.field_origins["reason"] == "generated"
    # contexto corto: el prompt no contiene TODAS las unidades
    prompt = provider.calls[0]["messages"][1]["content"]
    assert "unidad 2 sembrada" not in prompt
    assert "Hallazgos" in prompt


def test_refine_noop_when_no_findings(wb_env):
    wb, _ = wb_env
    rid = _seed_grounded_draft(wb)
    provider = FakeProvider()
    wb.provider = provider
    _, changes = wb.refine_with_findings(rid)
    assert changes == []
    assert provider.calls == []


# ---------------------------------------------------------------------------
# Keying por unit_id: cleanup de expansiones + herencia entre versiones
# ---------------------------------------------------------------------------

def _expansion(rid: str, unit_id: str, depth: int = 0):
    from ipa.tutor.tutor_contracts import ExpansionDepth, RoadmapExpansion
    return RoadmapExpansion(
        expansion_id=f"expansion:{unit_id[-8:]}l{depth}",
        roadmap_id=rid, unit_id=unit_id, depth_level=ExpansionDepth(depth),
        title=f"L{depth}", evidence_refs=[_source("chunk:x")],
        created_at="2026-01-01T00:00:00Z",
        generation=GenerationProvenance(
            generator="test", generated_at="2026-01-01T00:00:00Z",
            input_hash="sha256:" + "0" * 64, model_fingerprint="test"),
        field_origins={"title": "generated", "evidence_refs": "generated"},
    )


def test_remove_unit_deletes_unit_expansions(wb_env):
    wb, store = wb_env
    rid = _seed_grounded_draft(wb)
    victim = wb.store.get_roadmap(rid).units[0]
    store.save_expansion(_expansion(rid, victim.unit_id, 0))
    store.save_expansion(_expansion(rid, victim.unit_id, 1))
    survivor = wb.store.get_roadmap(rid).units[1]
    store.save_expansion(_expansion(rid, survivor.unit_id, 0))
    wb.remove_unit(rid, victim.unit_id)
    exps = store.list_expansions(rid)
    assert len(exps) == 1
    assert exps[0].unit_id == survivor.unit_id


def test_edit_unit_concept_change_marks_expansions_stale(wb_env):
    wb, store = wb_env
    rid = _seed_grounded_draft(wb)
    unit = wb.store.get_roadmap(rid).units[0]
    store.save_expansion(_expansion(rid, unit.unit_id, 0))
    # edición sin tocar concept_id → sigue generated
    wb.edit_unit(rid, unit.unit_id, reason="razón nueva")
    assert store.list_expansions(rid, unit_id=unit.unit_id)[0].status == "generated"
    # cambio de concept_id → la evidencia describe el material viejo
    wb.edit_unit(rid, unit.unit_id, concept_id="concept:z")
    assert store.list_expansions(rid, unit_id=unit.unit_id)[0].status == "stale"


def test_inherit_progress_carries_surviving_units(wb_env):
    wb, store = wb_env
    old_rid = _seed_grounded_draft(wb)
    units = wb.store.get_roadmap(old_rid).units
    uid_done, uid_survivor = units[0].unit_id, units[1].unit_id
    store.set_unit_status(old_rid, uid_done, "done")
    store.set_unit_status(old_rid, uid_survivor, "current")
    store.save_unit_summary(old_rid, uid_done, "resumen de la unidad 1")
    store.set_position(old_rid, uid_survivor, 2)
    store.save_expansion(_expansion(old_rid, uid_done, 0))
    new = wb.create_draft("goal:x", base_roadmap_id=old_rid)
    # una unidad no sobrevive a la revisión
    wb.remove_unit(new.roadmap_id, uid_survivor)
    n = store.inherit_progress(
        old_rid, new.roadmap_id,
        {u.unit_id for u in wb.store.get_roadmap(new.roadmap_id).units})
    assert n == 1  # solo la unidad done sobrevivió
    statuses = store.unit_statuses(new.roadmap_id)
    assert statuses == {uid_done: "done"}
    assert store.list_unit_summaries(new.roadmap_id)[0]["unit_id"] == uid_done
    assert store.get_position(new.roadmap_id)["unit_id"] is None
    exps = store.list_expansions(new.roadmap_id)
    assert len(exps) == 1 and exps[0].unit_id == uid_done
    assert exps[0].expansion_id != f"expansion:{uid_done[-8:]}l0"


def test_migration_unit_order_to_unit_id(tmp_path):
    """DBs viejas con columnas unit_order se migran y backfillean con el payload."""
    import sqlite3
    db = tmp_path / "old.db"
    con = sqlite3.connect(str(db))
    con.executescript("""
        CREATE TABLE roadmaps (
            roadmap_id TEXT PRIMARY KEY, goal_id TEXT NOT NULL,
            payload_json TEXT NOT NULL, status TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE unit_progress (
            roadmap_id TEXT NOT NULL, unit_order INTEGER NOT NULL,
            status TEXT NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY (roadmap_id, unit_order));
        CREATE TABLE roadmap_position (
            roadmap_id TEXT PRIMARY KEY, unit_order INTEGER NOT NULL,
            depth_level INTEGER NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE unit_summaries (
            roadmap_id TEXT NOT NULL, unit_order INTEGER NOT NULL,
            summary TEXT NOT NULL, created_at TEXT NOT NULL,
            PRIMARY KEY (roadmap_id, unit_order));
    """)
    payload = {
        "roadmap_id": "roadmap:old", "goal_id": "goal:x", "version": 1,
        "status": "draft", "change_reason": "",
        "units": [
            {"unit_id": "roadmap_unit:aaa", "order": 1, "concept_id": "c:a",
             "reason": "a", "estimated_effort_minutes": 30, "source_refs": [],
             "assessment_types": [], "stage_id": None, "status": "pending",
             "field_origins": {}},
            {"unit_id": "roadmap_unit:bbb", "order": 2, "concept_id": "c:b",
             "reason": "b", "estimated_effort_minutes": 30, "source_refs": [],
             "assessment_types": [], "stage_id": None, "status": "pending",
             "field_origins": {}},
        ],
        "stages": [], "uncertainties": [], "assumptions": [],
        "success_criteria": [], "source_refs": [], "field_origins": {},
        "generation": None, "validation": None,
    }
    con.execute("INSERT INTO roadmaps VALUES (?,?,?,?,?,?)",
                ("roadmap:old", "goal:x", json.dumps(payload), "draft",
                 "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"))
    con.execute("INSERT INTO unit_progress VALUES (?,?,?,?)",
                ("roadmap:old", 2, "done", "2026-01-01T00:00:00Z"))
    con.execute("INSERT INTO roadmap_position VALUES (?,?,?,?)",
                ("roadmap:old", 2, 1, "2026-01-01T00:00:00Z"))
    con.execute("INSERT INTO unit_summaries VALUES (?,?,?,?)",
                ("roadmap:old", 2, "viejo resumen", "2026-01-01T00:00:00Z"))
    con.commit(); con.close()

    store = TutorStore(db)  # abrir dispara la migración
    statuses = store.unit_statuses("roadmap:old")
    assert statuses == {"roadmap_unit:bbb": "done"}
    assert store.get_position("roadmap:old")["unit_id"] == "roadmap_unit:bbb"
    assert store.get_position("roadmap:old")["depth_level"] == 1
    assert store.list_unit_summaries("roadmap:old")[0]["summary"] == "viejo resumen"
