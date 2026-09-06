from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from build_landing_manifest import build_records, main as build_manifest  # noqa: E402
from validate_contracts import validate_structure, validate_integrity  # noqa: E402

ROOT = Path(__file__).parents[1]
REQUIRED_FIELDS = {
    "artifact_id", "content_hash", "source_uri", "original_filename",
    "mime_type", "byte_size", "received_at", "status", "attempts", "stages",
}
STAGE_KEYS = {"acquisition", "parsing", "chunking", "bm25", "embedding", "enrichment"}


def test_workspace_contract_files_exist():
    assert (ROOT / "README.md").exists()
    assert (ROOT / "contracts" / "README.md").exists()
    assert (ROOT / "configs" / "lab.yaml").exists()
    assert (ROOT / "docs" / "development" / "experiment-matrix.md").exists()


def _make_fixtures(input_dir: Path) -> dict[str, bytes]:
    input_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "note.txt": b"Hello RES-023 lab.\nLine two.\n",
        "page.html": b"<html><body><h1>Title</h1><p>Body text</p></body></html>",
        "data.json": b'{"key": "value", "n": 42}\n',
        "empty.bin": b"",
    }
    for name, content in files.items():
        (input_dir / name).write_bytes(content)
    return files


def _run_builder(input_dir: Path, manifest_path: Path) -> list[dict]:
    import argparse
    orig_argv = sys.argv
    try:
        sys.argv = ["build_landing_manifest", "--input", str(input_dir),
                    "--manifest", str(manifest_path)]
        build_manifest()
    finally:
        sys.argv = orig_argv
    return [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- Builder behavior ---

def test_manifest_has_all_required_fields(tmp_path):
    input_dir = tmp_path / "input"
    _make_fixtures(input_dir)
    manifest = tmp_path / "processing.jsonl"
    records = _run_builder(input_dir, manifest)
    assert len(records) == 4
    for r in records:
        missing = REQUIRED_FIELDS - r.keys()
        assert not missing, f"{r.get('original_filename')} missing: {missing}"


def test_manifest_artifact_ids_are_unique(tmp_path):
    input_dir = tmp_path / "input"
    _make_fixtures(input_dir)
    manifest = tmp_path / "processing.jsonl"
    records = _run_builder(input_dir, manifest)
    ids = [r["artifact_id"] for r in records]
    assert len(ids) == len(set(ids))


def test_manifest_content_hashes_are_sha256_prefixed(tmp_path):
    input_dir = tmp_path / "input"
    _make_fixtures(input_dir)
    manifest = tmp_path / "processing.jsonl"
    records = _run_builder(input_dir, manifest)
    for r in records:
        assert r["content_hash"].startswith("sha256:")
        assert r["artifact_id"] == r["content_hash"]


def test_manifest_hash_matches_file_content(tmp_path):
    input_dir = tmp_path / "input"
    files = _make_fixtures(input_dir)
    manifest = tmp_path / "processing.jsonl"
    records = _run_builder(input_dir, manifest)
    by_name = {r["original_filename"]: r for r in records}
    for name, content in files.items():
        expected = "sha256:" + hashlib.sha256(content).hexdigest()
        assert by_name[name]["content_hash"] == expected


def test_manifest_is_idempotent(tmp_path):
    input_dir = tmp_path / "input"
    _make_fixtures(input_dir)
    manifest = tmp_path / "processing.jsonl"
    first = _run_builder(input_dir, manifest)
    first_text = manifest.read_text(encoding="utf-8")
    # Re-run with --force to allow overwrite for idempotency check
    import argparse
    orig_argv = sys.argv
    try:
        sys.argv = ["build_landing_manifest", "--input", str(input_dir),
                    "--manifest", str(manifest), "--force"]
        build_manifest()
    finally:
        sys.argv = orig_argv
    second_text = manifest.read_text(encoding="utf-8")
    assert first_text == second_text


def test_manifest_stages_declare_all_stage_keys(tmp_path):
    input_dir = tmp_path / "input"
    _make_fixtures(input_dir)
    manifest = tmp_path / "processing.jsonl"
    records = _run_builder(input_dir, manifest)
    for r in records:
        assert set(r["stages"].keys()) == STAGE_KEYS
        assert r["stages"]["acquisition"] == "success"
        for stage in ("parsing", "chunking", "bm25", "embedding", "enrichment"):
            assert r["stages"][stage] == "pending"


def test_manifest_initial_status_is_received(tmp_path):
    input_dir = tmp_path / "input"
    _make_fixtures(input_dir)
    manifest = tmp_path / "processing.jsonl"
    records = _run_builder(input_dir, manifest)
    for r in records:
        assert r["status"] == "received"
        assert r["attempts"] == 0


def test_manifest_byte_size_matches_file(tmp_path):
    input_dir = tmp_path / "input"
    files = _make_fixtures(input_dir)
    manifest = tmp_path / "processing.jsonl"
    records = _run_builder(input_dir, manifest)
    by_name = {r["original_filename"]: r for r in records}
    for name, content in files.items():
        assert by_name[name]["byte_size"] == len(content)


def test_manifest_preserves_original_artifacts(tmp_path):
    input_dir = tmp_path / "input"
    files = _make_fixtures(input_dir)
    originals = {name: (input_dir / name).read_bytes() for name in files}
    manifest = tmp_path / "processing.jsonl"
    _run_builder(input_dir, manifest)
    for name, content in originals.items():
        assert (input_dir / name).read_bytes() == content


# --- Append-safe behavior (H1) ---

def test_builder_refuses_to_overwrite_existing_manifest(tmp_path):
    input_dir = tmp_path / "input"
    _make_fixtures(input_dir)
    manifest = tmp_path / "processing.jsonl"
    _run_builder(input_dir, manifest)
    original = manifest.read_text(encoding="utf-8")
    import argparse
    orig_argv = sys.argv
    try:
        sys.argv = ["build_landing_manifest", "--input", str(input_dir),
                    "--manifest", str(manifest)]
        with __import__("pytest").raises(SystemExit):
            build_manifest()
    finally:
        sys.argv = orig_argv
    assert manifest.read_text(encoding="utf-8") == original


def test_builder_force_overwrites_existing_manifest(tmp_path):
    input_dir = tmp_path / "input"
    _make_fixtures(input_dir)
    manifest = tmp_path / "processing.jsonl"
    _run_builder(input_dir, manifest)
    import argparse
    orig_argv = sys.argv
    try:
        sys.argv = ["build_landing_manifest", "--input", str(input_dir),
                    "--manifest", str(manifest), "--force"]
        build_manifest()
    finally:
        sys.argv = orig_argv
    assert manifest.exists()


# --- Structural validation (H5) ---

def test_validate_structure_passes_on_valid_manifest(tmp_path):
    input_dir = tmp_path / "input"
    _make_fixtures(input_dir)
    manifest = tmp_path / "processing.jsonl"
    _run_builder(input_dir, manifest)
    errors, count = validate_structure(manifest)
    assert errors == []
    assert count == 4


def test_validate_structure_catches_duplicate_ids(tmp_path):
    manifest = tmp_path / "bad.jsonl"
    record = {
        "artifact_id": "sha256:" + "a" * 64,
        "content_hash": "sha256:" + "a" * 64,
        "source_uri": "x", "original_filename": "x", "mime_type": "text/plain",
        "byte_size": 1, "received_at": "2026-01-01T00:00:00Z",
        "status": "received", "attempts": 0,
        "stages": {"acquisition": "success", "parsing": "pending",
                   "chunking": "pending", "bm25": "pending",
                   "embedding": "pending", "enrichment": "pending"},
    }
    manifest.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
    errors, _ = validate_structure(manifest)
    assert any("duplicate" in e for e in errors)


def test_validate_structure_catches_invalid_status(tmp_path):
    manifest = tmp_path / "bad.jsonl"
    record = {
        "artifact_id": "sha256:" + "a" * 64,
        "content_hash": "sha256:" + "a" * 64,
        "source_uri": "x", "original_filename": "x", "mime_type": "text/plain",
        "byte_size": 1, "received_at": "2026-01-01T00:00:00Z",
        "status": "bogus", "attempts": 0,
        "stages": {"acquisition": "success", "parsing": "pending",
                   "chunking": "pending", "bm25": "pending",
                   "embedding": "pending", "enrichment": "pending"},
    }
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    errors, _ = validate_structure(manifest)
    assert any("invalid status" in e for e in errors)


def test_validate_structure_catches_artifact_id_mismatch(tmp_path):
    manifest = tmp_path / "bad.jsonl"
    record = {
        "artifact_id": "sha256:" + "a" * 64,
        "content_hash": "sha256:" + "b" * 64,
        "source_uri": "x", "original_filename": "x", "mime_type": "text/plain",
        "byte_size": 1, "received_at": "2026-01-01T00:00:00Z",
        "status": "received", "attempts": 0,
        "stages": {"acquisition": "success", "parsing": "pending",
                   "chunking": "pending", "bm25": "pending",
                   "embedding": "pending", "enrichment": "pending"},
    }
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    errors, _ = validate_structure(manifest)
    assert any("artifact_id != content_hash" in e for e in errors)


# --- Integrity validation (H5) ---

def test_validate_integrity_passes_on_fresh_manifest(tmp_path):
    input_dir = tmp_path / "input"
    _make_fixtures(input_dir)
    manifest = tmp_path / "processing.jsonl"
    _run_builder(input_dir, manifest)
    errors = validate_integrity(manifest)
    assert errors == []


def test_validate_integrity_catches_modified_source(tmp_path):
    input_dir = tmp_path / "input"
    files = _make_fixtures(input_dir)
    manifest = tmp_path / "processing.jsonl"
    _run_builder(input_dir, manifest)
    (input_dir / "note.txt").write_bytes(b"MODIFIED CONTENT")
    errors = validate_integrity(manifest)
    assert any("content_hash mismatch" in e for e in errors)


def test_validate_integrity_catches_deleted_source(tmp_path):
    input_dir = tmp_path / "input"
    _make_fixtures(input_dir)
    manifest = tmp_path / "processing.jsonl"
    _run_builder(input_dir, manifest)
    (input_dir / "note.txt").unlink()
    errors = validate_integrity(manifest)
    assert any("source_uri does not exist" in e for e in errors)


def test_validate_integrity_catches_size_mismatch(tmp_path):
    input_dir = tmp_path / "input"
    _make_fixtures(input_dir)
    manifest = tmp_path / "processing.jsonl"
    _run_builder(input_dir, manifest)
    (input_dir / "note.txt").write_bytes(b"short")
    errors = validate_integrity(manifest)
    assert any("byte_size mismatch" in e for e in errors)
