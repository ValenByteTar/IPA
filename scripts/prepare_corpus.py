"""Distribute PDFs from a source directory into Landing/ across 4 format categories.

Each source PDF is assigned to exactly one category (no repeats):
  - pdf:  copied as-is
  - txt:  text extracted with PyMuPDF, saved as .txt
  - html: text extracted, wrapped in minimal HTML
  - json: metadata + text extracted, saved as .json

Distribution is deterministic (sorted by filename) so re-runs produce the
same assignment.  Files larger than --max-convert-mb are always assigned to
the pdf category to avoid slow conversions.

Usage:
    python scripts/prepare_corpus.py --source "C:\\path\\to\\pdfs" --landing Landing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def slugify(name: str) -> str:
    """Make a filename-safe slug from a PDF name."""
    stem = Path(name).stem
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return safe[:120]  # cap length


def extract_text(pdf_path: Path, max_pages: int = 500) -> tuple[str, int]:
    """Extract text from a PDF. Returns (text, page_count)."""
    import pymupdf
    doc = pymupdf.open(str(pdf_path))
    pages = min(len(doc), max_pages)
    parts: list[str] = []
    for i in range(pages):
        parts.append(doc[i].get_text("text"))
    doc.close()
    return "\n".join(parts), pages


def make_html(title: str, text: str) -> str:
    """Wrap text in a minimal HTML document."""
    import html as html_mod
    escaped = html_mod.escape(text)
    title_esc = html_mod.escape(title)
    return (
        f"<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{title_esc}</title>\n</head>\n<body>\n"
        f"<pre>{escaped}</pre>\n</body>\n</html>\n"
    )


def make_json_record(pdf_path: Path, text: str, pages: int) -> dict:
    """Build a structured JSON record from a PDF."""
    return {
        "source_file": pdf_path.name,
        "source_size_bytes": pdf_path.stat().st_size,
        "pages_extracted": pages,
        "text": text,
        "text_length": len(text),
        "content_hash": "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Distribute PDFs into Landing/ format categories.")
    parser.add_argument("--source", required=True, help="Source directory with PDFs.")
    parser.add_argument("--landing", default="Landing", help="Landing output directory.")
    parser.add_argument("--max-convert-mb", type=int, default=20,
                        help="Skip conversion for PDFs larger than this (MB); assign to pdf category.")
    parser.add_argument("--pdf-count", type=int, default=180, help="Target PDF category size.")
    parser.add_argument("--txt-count", type=int, default=180, help="Target TXT category size.")
    parser.add_argument("--html-count", type=int, default=180, help="Target HTML category size.")
    parser.add_argument("--json-count", type=int, default=180, help="Target JSON category size.")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"Source directory not found: {source}")

    landing = Path(args.landing)
    landing.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(source.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs found in source directory.")

    max_convert_bytes = args.max_convert_mb * 1024 * 1024

    # Deterministic assignment: sort by name, split into contiguous ranges.
    # Large files always go to pdf category.
    pdf_pool: list[Path] = []
    convert_pool: list[Path] = []

    for pdf in pdfs:
        if pdf.stat().st_size > max_convert_bytes:
            pdf_pool.append(pdf)
        else:
            convert_pool.append(pdf)

    # Allocate convert_pool across txt, html, json (in order).
    txt_files = convert_pool[:args.txt_count]
    html_files = convert_pool[args.txt_count:args.txt_count + args.html_count]
    json_files = convert_pool[args.txt_count + args.html_count:
                              args.txt_count + args.html_count + args.json_count]
    # Remaining convert_pool files go to pdf category.
    pdf_files = pdf_pool + convert_pool[args.txt_count + args.html_count + args.json_count:]

    # Trim pdf category if it exceeds the target (keep all large files though).
    # Actually, keep everything — more PDFs is fine for scale testing.

    stats = {"pdf": 0, "txt": 0, "html": 0, "json": 0, "errors": []}

    # --- PDF category: copy ---
    print(f"Copying {len(pdf_files)} PDFs...")
    for i, pdf in enumerate(pdf_files):
        dest = landing / f"{slugify(pdf.name)}.pdf"
        try:
            shutil.copy2(pdf, dest)
            stats["pdf"] += 1
        except Exception as exc:
            stats["errors"].append(f"pdf copy {pdf.name}: {exc}")
        if (i + 1) % 50 == 0:
            print(f"  pdf: {i + 1}/{len(pdf_files)}")

    # --- TXT category: extract text ---
    print(f"Converting {len(txt_files)} PDFs to TXT...")
    for i, pdf in enumerate(txt_files):
        slug = slugify(pdf.name)
        dest = landing / f"{slug}.txt"
        try:
            text, pages = extract_text(pdf)
            dest.write_text(text, encoding="utf-8")
            stats["txt"] += 1
        except Exception as exc:
            stats["errors"].append(f"txt {pdf.name}: {exc}")
        if (i + 1) % 50 == 0:
            print(f"  txt: {i + 1}/{len(txt_files)}")

    # --- HTML category: extract text, wrap in HTML ---
    print(f"Converting {len(html_files)} PDFs to HTML...")
    for i, pdf in enumerate(html_files):
        slug = slugify(pdf.name)
        dest = landing / f"{slug}.html"
        try:
            text, pages = extract_text(pdf)
            html_content = make_html(slug, text)
            dest.write_text(html_content, encoding="utf-8")
            stats["html"] += 1
        except Exception as exc:
            stats["errors"].append(f"html {pdf.name}: {exc}")
        if (i + 1) % 50 == 0:
            print(f"  html: {i + 1}/{len(html_files)}")

    # --- JSON category: extract metadata + text ---
    print(f"Converting {len(json_files)} PDFs to JSON...")
    for i, pdf in enumerate(json_files):
        slug = slugify(pdf.name)
        dest = landing / f"{slug}.json"
        try:
            text, pages = extract_text(pdf)
            record = make_json_record(pdf, text, pages)
            dest.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            stats["json"] += 1
        except Exception as exc:
            stats["errors"].append(f"json {pdf.name}: {exc}")
        if (i + 1) % 50 == 0:
            print(f"  json: {i + 1}/{len(json_files)}")

    # --- Summary ---
    print("\n=== Corpus preparation complete ===")
    print(f"  PDF:  {stats['pdf']}")
    print(f"  TXT:  {stats['txt']}")
    print(f"  HTML: {stats['html']}")
    print(f"  JSON: {stats['json']}")
    print(f"  Total artifacts in Landing/: {stats['pdf'] + stats['txt'] + stats['html'] + stats['json']}")
    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])}")
        for e in stats["errors"][:10]:
            print(f"    {e}")
        if len(stats["errors"]) > 10:
            print(f"    ... and {len(stats['errors']) - 10} more")

    # Write summary file
    summary_path = landing.parent / "outputs" / "corpus_preparation.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
