"""Validate Tutor Agent records against their authoritative JSON Schemas."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
CONTRACTS = ROOT / "contracts"
SCHEMAS = {
    "LearningGoal": "learning_goal.schema.json",
    "Concept": "concept.schema.json",
    "Roadmap": "roadmap.schema.json",
    "AssessmentResult": "assessment_result.schema.json",
    "ResearchRequest": "research_request.schema.json",
}


def load_schema(record_type: str) -> dict[str, Any]:
    if record_type not in SCHEMAS:
        raise ValueError(f"Unknown record type: {record_type}")
    return json.loads((CONTRACTS / SCHEMAS[record_type]).read_text(encoding="utf-8"))


def _registry():
    from referencing import Registry, Resource

    registry = Registry()
    for path in CONTRACTS.glob("*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        if "$id" in schema:
            registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    return registry


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _source_ref_errors(source_refs: list[dict[str, Any]], prefix: str) -> list[str]:
    errors = []
    for index, ref in enumerate(source_refs):
        span = ref.get("source_span")
        if span and span["offset_start"] > span["offset_end"]:
            errors.append(f"{prefix}[{index}].source_span offset_start must be <= offset_end")
    return errors


def integrity_errors(record_type: str, data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    created_at = data.get("created_at")
    updated_at = data.get("updated_at")
    if created_at and updated_at and _timestamp(created_at) > _timestamp(updated_at):
        errors.append("created_at must be <= updated_at")

    origins = data.get("field_origins", {})
    if any(origin in {"generated", "mixed"} for origin in origins.values()) and not data.get("generation"):
        errors.append("generation provenance is required when a field origin is generated or mixed")

    if record_type == "LearningGoal":
        if data.get("status") in {"confirmed", "active", "completed"}:
            approval = data.get("approval") or {}
            if approval.get("decision") != "approved":
                errors.append("confirmed, active, or completed goals require an approved human decision")

    elif record_type == "Concept":
        if data.get("concept_id") in data.get("prerequisite_ids", []):
            errors.append("a concept cannot be its own prerequisite")
        errors.extend(_source_ref_errors(data.get("source_refs", []), "source_refs"))

    elif record_type == "Roadmap":
        units = data.get("units", [])
        orders = [unit.get("order") for unit in units]
        if orders and sorted(orders) != list(range(1, len(orders) + 1)):
            errors.append("roadmap unit order must be unique and contiguous starting at 1")
        unit_ids = [unit.get("unit_id") for unit in units]
        if len(unit_ids) != len(set(unit_ids)):
            errors.append("roadmap unit_id values must be unique")
        concept_ids = [unit.get("concept_id") for unit in units]
        if len(concept_ids) != len(set(concept_ids)):
            errors.append("roadmap concept_id values must be unique")
        if data.get("status") in {"approved", "active", "completed", "superseded"}:
            approval = data.get("approval") or {}
            if approval.get("decision") != "approved":
                errors.append("approved roadmap states require an approved human decision")
        for index, unit in enumerate(units):
            errors.extend(_source_ref_errors(unit.get("source_refs", []), f"units[{index}].source_refs"))

    elif record_type == "AssessmentResult":
        errors.extend(_source_ref_errors(data.get("evidence", []), "evidence"))
        if data.get("status") == "misconception" and not data.get("misconceptions"):
            errors.append("misconception status requires at least one misconception")

    elif record_type == "ResearchRequest":
        if data.get("status") in {"approved", "running", "completed"}:
            approval = data.get("approval") or {}
            if approval.get("decision") != "approved":
                errors.append("executable research states require an approved human decision")
        errors.extend(_source_ref_errors(data.get("gap_evidence", []), "gap_evidence"))
        errors.extend(_source_ref_errors(data.get("result_source_refs", []), "result_source_refs"))

    return errors


def validate(record_type: str, data: dict[str, Any]) -> list[str]:
    from jsonschema import Draft202012Validator, FormatChecker

    schema = load_schema(record_type)
    validator = Draft202012Validator(schema, registry=_registry(), format_checker=FormatChecker())
    errors = []
    for error in sorted(validator.iter_errors(data), key=lambda item: list(item.absolute_path)):
        path = ".".join(str(part) for part in error.absolute_path)
        errors.append(f"{path}: {error.message}" if path else error.message)
    if not errors:
        errors.extend(integrity_errors(record_type, data))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", choices=SCHEMAS)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    data = json.loads(args.path.read_text(encoding="utf-8"))
    errors = validate(args.record, data)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Valid {args.record}: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
