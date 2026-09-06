"""Validate a processing manifest for structure and integrity.

Structural validation checks field presence, uniqueness and hash format.
Integrity validation (enabled with --integrity) verifies that source files
still exist and that recorded hashes/sizes match the actual files.

Usage:
    python scripts/validate_contracts.py outputs/manifests/processing.jsonl
    python scripts/validate_contracts.py outputs/manifests/processing.jsonl --integrity
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

REQUIRED = {
    "artifact_id", "content_hash", "source_uri", "original_filename",
    "mime_type", "byte_size", "received_at", "status", "attempts", "stages",
}
VALID_LIFECYCLE = {
    "received", "accepted", "quarantined", "parsing", "chunked", "indexed",
    "embedding_pending", "embedding_running", "embedding_complete",
    "enrichment_pending", "enrichment_running", "enriched",
    "validated", "published", "failed", "dead_letter",
}
VALID_STAGE_KEYS = {
    "acquisition", "parsing", "chunking", "bm25", "embedding", "enrichment",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_structure(path: Path) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    count = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        count += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON: {exc}")
            continue

        missing = REQUIRED - record.keys()
        if missing:
            errors.append(f"line {line_no}: missing fields: {sorted(missing)}")

        aid = record.get("artifact_id", "")
        if aid in seen:
            errors.append(f"line {line_no}: duplicate artifact_id: {aid}")
        seen.add(aid)

        chash = record.get("content_hash", "")
        if not chash.startswith("sha256:"):
            errors.append(f"line {line_no}: invalid content_hash format: {chash}")
        if aid and chash and aid != chash:
            errors.append(f"line {line_no}: artifact_id != content_hash")

        if not isinstance(record.get("byte_size"), int) or record["byte_size"] < 0:
            errors.append(f"line {line_no}: invalid byte_size: {record.get('byte_size')}")

        if not isinstance(record.get("attempts"), int) or record["attempts"] < 0:
            errors.append(f"line {line_no}: invalid attempts: {record.get('attempts')}")

        status = record.get("status", "")
        if status not in VALID_LIFECYCLE:
            errors.append(f"line {line_no}: invalid status '{status}'; expected one of {sorted(VALID_LIFECYCLE)}")

        stages = record.get("stages", {})
        if not isinstance(stages, dict):
            errors.append(f"line {line_no}: stages must be an object")
        elif set(stages.keys()) != VALID_STAGE_KEYS:
            errors.append(f"line {line_no}: stages keys mismatch: {set(stages.keys()) ^ VALID_STAGE_KEYS}")

    return errors, count  # type: ignore[return-value]


def validate_integrity(path: Path) -> list[str]:
    errors: list[str] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        source = Path(record.get("source_uri", ""))
        if not source.exists():
            errors.append(f"line {line_no}: source_uri does not exist: {source}")
            continue
        actual_hash = "sha256:" + _sha256(source)
        if actual_hash != record.get("content_hash"):
            errors.append(
                f"line {line_no}: content_hash mismatch for {source.name}: "
                f"expected {record.get('content_hash')}, got {actual_hash}"
            )
        actual_size = source.stat().st_size
        if actual_size != record.get("byte_size"):
            errors.append(
                f"line {line_no}: byte_size mismatch for {source.name}: "
                f"expected {record.get('byte_size')}, got {actual_size}"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument(
        "--integrity",
        action="store_true",
        help="Also verify that source files exist and hashes/sizes match.",
    )
    args = parser.parse_args()
    path = Path(args.manifest)
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")

    errors, count = validate_structure(path)
    if args.integrity:
        errors.extend(validate_integrity(path))

    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        raise SystemExit(f"{len(errors)} validation error(s) in {count} records")

    mode = "structure + integrity" if args.integrity else "structure"
    print(f"OK: {count} manifest records valid ({mode})")


if __name__ == "__main__":
    main()
