"""Validate Reporter JSON artifacts against authoritative schemas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
SCHEMAS = {
    "ReporterReport": "reporter_report.schema.json",
    "ReporterDocumentDecision": "reporter_document_decision.schema.json",
    "TopicLink": "topic_link.schema.json",
}


def _registry():
    from referencing import Registry, Resource
    registry = Registry()
    for path in (ROOT / "contracts").glob("*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        if "$id" in schema:
            registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    return registry


def validate(record_type: str, payload: dict[str, Any]) -> list[str]:
    from jsonschema import Draft202012Validator, FormatChecker
    schema = json.loads((ROOT / "contracts" / SCHEMAS[record_type]).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, registry=_registry(), format_checker=FormatChecker())
    return [f"{'.'.join(map(str, error.absolute_path))}: {error.message}" if error.absolute_path else error.message for error in validator.iter_errors(payload)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", choices=SCHEMAS)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    errors = validate(args.record, json.loads(args.path.read_text(encoding="utf-8")))
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    print(f"Valid {args.record}: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
