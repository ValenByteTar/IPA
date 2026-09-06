"""Create a durable JSONL manifest without modifying input files.

Each run writes to a timestamped file under outputs/manifests/ so that prior
manifests are never overwritten. A ``--manifest`` path may be supplied
explicitly; if it already exists the run is rejected unless ``--force`` is
passed, preserving the lab invariant "never overwrite previous artifacts".
"""
from __future__ import annotations
import argparse
import hashlib
import json
import mimetypes
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_records(root: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        digest = sha256(path)
        records.append({
            "artifact_id": f"sha256:{digest}",
            "content_hash": f"sha256:{digest}",
            "source_uri": str(path.resolve()),
            "original_filename": path.name,
            "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            "byte_size": stat.st_size,
            "received_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
            "status": "received",
            "attempts": 0,
            "stages": {
                "acquisition": "success",
                "parsing": "pending",
                "chunking": "pending",
                "bm25": "pending",
                "embedding": "pending",
                "enrichment": "pending",
            },
        })
    return records


def default_manifest_path() -> Path:
    stamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    return Path("outputs/manifests") / f"processing.{stamp}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--manifest",
        default=None,
        help="Explicit manifest path. Defaults to a timestamped file under outputs/manifests/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing manifest file. Use only for throwaway runs.",
    )
    args = parser.parse_args()
    root = Path(args.input)
    if not root.exists():
        raise SystemExit(f"Input directory does not exist: {root}")
    output = Path(args.manifest) if args.manifest else default_manifest_path()
    if output.exists() and not args.force:
        raise SystemExit(
            f"Manifest already exists: {output}\n"
            f"Use --force to overwrite, or omit --manifest to write a new timestamped file."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    records = build_records(root)
    output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    manifest_hash = "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"Wrote {len(records)} records to {output}")
    print(f"manifest_hash={manifest_hash}")


if __name__ == "__main__":
    main()
