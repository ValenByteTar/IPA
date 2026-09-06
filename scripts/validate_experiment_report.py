"""Validate an experiment report JSON against the lab schema.

Loads ``contracts/experiment_report.schema.json`` and validates with
``jsonschema`` (Draft 2020-12) for structural checks, then runs additional
integrity checks not expressible in JSON Schema.

``jsonschema`` is a dev/test-only dependency (see contracts/validation_contract.md).

Usage:
    python scripts/validate_experiment_report.py path/to/report.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "contracts" / "experiment_report.schema.json"


def load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _structural_errors(report: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    import jsonschema
    from jsonschema import Draft202012Validator
    validator = Draft202012Validator(schema)
    return [
        f"schema: {error.message} (at {'/'.join(str(p) for p in error.absolute_path) or '<root>'})"
        for error in validator.iter_errors(report)
    ]


def _integrity_errors(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    started = report.get("started_at")
    finished = report.get("finished_at")
    if isinstance(started, str) and isinstance(finished, str) and started > finished:
        errors.append(f"integrity: started_at ({started}) is after finished_at ({finished})")

    output_uri = report.get("output_uri")
    output_hash = report.get("output_hash")
    if output_uri and output_hash:
        path = Path(output_uri)
        if path.exists():
            actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != output_hash:
                errors.append(
                    f"integrity: output_hash mismatch for {output_uri}: "
                    f"expected {output_hash}, got {actual}"
                )
        else:
            errors.append(f"integrity: output_uri does not exist: {output_uri}")

    for artifact in report.get("output_artifacts", []) or []:
        apath = Path(artifact.get("path", ""))
        ahash = artifact.get("hash", "")
        if apath.exists():
            actual = "sha256:" + hashlib.sha256(apath.read_bytes()).hexdigest()
            if actual != ahash:
                errors.append(
                    f"integrity: output_artifact hash mismatch for {apath}: "
                    f"expected {ahash}, got {actual}"
                )
        else:
            errors.append(f"integrity: output_artifact does not exist: {apath}")

    return errors


def validate(report: dict[str, Any]) -> list[str]:
    schema = load_schema()
    return _structural_errors(report, schema) + _integrity_errors(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    args = parser.parse_args()
    path = Path(args.report)
    if not path.exists():
        raise SystemExit(f"Report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    errors = validate(report)
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        raise SystemExit(f"{len(errors)} validation error(s)")
    print(f"OK: report {report.get('experiment_id', '?')} is valid")


if __name__ == "__main__":
    main()
