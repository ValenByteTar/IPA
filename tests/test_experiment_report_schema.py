from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from validate_experiment_report import validate, load_schema  # noqa: E402

ROOT = Path(__file__).parents[1]


def _valid_report() -> dict:
    return {
        "experiment_id": "E0.1",
        "candidate_id": "lab-baseline",
        "capability": "contract_compliance",
        "adapter_version": "0.1.0",
        "tool_version": "native",
        "hardware": {"cpu": "x86_64", "ram_gb": 16, "gpu": None},
        "input_manifest_hash": "sha256:" + "a" * 64,
        "configuration_fingerprint": "cfg:abc123",
        "output_hash": "sha256:" + "b" * 64,
        "output_uri": None,
        "output_artifacts": [],
        "started_at": "2026-08-24T22:00:00Z",
        "finished_at": "2026-08-24T22:05:00Z",
        "status": "completed",
        "warnings": [],
        "errors": [],
        "controls": {
            "same_input_manifest": True,
            "same_contract_version": True,
            "same_ground_truth": True,
            "same_hardware": True,
            "isolated_output_namespace": True,
            "no_production_writes": True,
        },
        "results": {
            "quality": "pass",
            "throughput": None,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "memory_mb": None,
            "recovery": None,
            "backpressure": None,
            "privacy_licensing": "local-only",
            "custom": {"contracts_validated": 10},
        },
        "decision": {
            "level": "observed",
            "rationale": "Stage 0 contract compliance baseline established.",
            "applicable_workloads": ["all"],
        },
    }


# --- Schema meta-validation ---

def test_schema_file_exists_and_is_valid_json():
    schema = json.loads((ROOT / "contracts" / "experiment_report.schema.json").read_text(encoding="utf-8"))
    assert schema["title"] == "ExperimentReport"
    assert "required" in schema


def test_schema_is_valid_draft202012():
    import jsonschema
    from jsonschema import Draft202012Validator
    schema = load_schema()
    Draft202012Validator.check_schema(schema)  # raises on invalid schema


def test_schema_hardware_is_closed():
    schema = load_schema()
    assert schema["properties"]["hardware"]["additionalProperties"] is False


# --- Structural validation via jsonschema ---

def test_valid_report_passes():
    assert validate(_valid_report()) == []


def test_missing_required_field_fails():
    report = _valid_report()
    del report["output_hash"]
    errors = validate(report)
    assert any("output_hash" in e for e in errors)


def test_invalid_experiment_id_fails():
    report = _valid_report()
    report["experiment_id"] = "bad-id"
    errors = validate(report)
    assert any("experiment_id" in e for e in errors)


def test_invalid_capability_fails():
    report = _valid_report()
    report["capability"] = "magic"
    errors = validate(report)
    assert any("capability" in e for e in errors)


def test_invalid_hash_format_fails():
    report = _valid_report()
    report["input_manifest_hash"] = "md5:abc"
    errors = validate(report)
    assert any("input_manifest_hash" in e for e in errors)


def test_invalid_decision_level_fails():
    report = _valid_report()
    report["decision"]["level"] = "champion"
    errors = validate(report)
    assert any("level" in e for e in errors)


def test_missing_control_flag_fails():
    report = _valid_report()
    del report["controls"]["no_production_writes"]
    errors = validate(report)
    assert any("no_production_writes" in e for e in errors)


def test_unknown_top_level_field_fails():
    report = _valid_report()
    report["unexpected_field"] = True
    errors = validate(report)
    assert any("unexpected_field" in e for e in errors)


def test_unknown_result_key_fails():
    report = _valid_report()
    report["results"]["bogus_metric"] = 1
    errors = validate(report)
    assert any("bogus_metric" in e for e in errors)


def test_unknown_hardware_field_fails():
    """H4: hardware must be closed under additionalProperties."""
    report = _valid_report()
    report["hardware"]["secret_field"] = "leaked"
    errors = validate(report)
    assert any("secret_field" in e for e in errors)


def test_invalid_timestamp_type_fails():
    report = _valid_report()
    report["started_at"] = 12345
    errors = validate(report)
    assert any("started_at" in e for e in errors)


def test_negative_ram_fails():
    report = _valid_report()
    report["hardware"]["ram_gb"] = -1
    errors = validate(report)
    assert any("ram_gb" in e for e in errors)


# --- Integrity validation ---

def test_integrity_started_after_finished_fails():
    report = _valid_report()
    report["started_at"] = "2026-08-24T23:00:00Z"
    report["finished_at"] = "2026-08-24T22:00:00Z"
    errors = validate(report)
    assert any("started_at" in e and "finished_at" in e for e in errors)


def test_integrity_output_uri_hash_mismatch_fails(tmp_path):
    artifact = tmp_path / "out.txt"
    artifact.write_bytes(b"hello")
    report = _valid_report()
    report["output_uri"] = str(artifact)
    report["output_hash"] = "sha256:" + "0" * 64
    errors = validate(report)
    assert any("output_hash mismatch" in e for e in errors)


def test_integrity_output_uri_hash_match_passes(tmp_path):
    artifact = tmp_path / "out.txt"
    artifact.write_bytes(b"hello")
    report = _valid_report()
    report["output_uri"] = str(artifact)
    report["output_hash"] = "sha256:" + hashlib.sha256(b"hello").hexdigest()
    errors = validate(report)
    assert errors == []


def test_integrity_output_uri_missing_file_fails(tmp_path):
    report = _valid_report()
    report["output_uri"] = str(tmp_path / "nonexistent.txt")
    report["output_hash"] = "sha256:" + "0" * 64
    errors = validate(report)
    assert any("output_uri does not exist" in e for e in errors)


def test_integrity_output_artifact_hash_mismatch_fails(tmp_path):
    artifact = tmp_path / "out.txt"
    artifact.write_bytes(b"hello")
    report = _valid_report()
    report["output_artifacts"] = [{"path": str(artifact), "hash": "sha256:" + "0" * 64}]
    errors = validate(report)
    assert any("output_artifact hash mismatch" in e for e in errors)


def test_integrity_output_artifact_hash_match_passes(tmp_path):
    artifact = tmp_path / "out.txt"
    artifact.write_bytes(b"hello")
    report = _valid_report()
    report["output_artifacts"] = [
        {"path": str(artifact), "hash": "sha256:" + hashlib.sha256(b"hello").hexdigest()}
    ]
    errors = validate(report)
    assert errors == []
