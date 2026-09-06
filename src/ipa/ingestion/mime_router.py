"""MIME router â€” detect MIME type by extension then content sniffing.

Uses Python's stdlib ``mimetypes`` for extension-based detection and falls
back to magic-byte sniffing for robustness.  No external dependencies.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path

# Magic-byte signatures for common formats where extension may be wrong/missing.
_MAGIC_SIGNATURES: list[tuple[bytes, str]] = [
    (b"%PDF-", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"\x1f\x8b", "application/gzip"),
    (b"PK\x03\x04", "application/zip"),
    (b"\x7fELF", "application/x-elf"),
]


def detect_mime(path: Path) -> str:
    """Detect MIME type, preferring content signatures over extensions."""
    guessed, _ = mimetypes.guess_type(path.name)

    # Magic bytes win over extensions: a renamed PDF must remain a PDF.
    try:
        with path.open("rb") as stream:
            header = stream.read(16)
    except OSError:
        return guessed or "application/octet-stream"

    for signature, mime in _MAGIC_SIGNATURES:
        if header.startswith(signature):
            return mime

    if guessed:
        return guessed

    # Heuristic: try to decode as UTF-8 text.
    if header:
        try:
            header.decode("utf-8")
            return "text/plain"
        except UnicodeDecodeError:
            pass

    return "application/octet-stream"


def route_to_parser(mime_type: str) -> str:
    """Map a MIME type to a parser identifier."""
    if mime_type == "application/pdf":
        return "pymupdf"
    if mime_type.startswith("text/html"):
        return "html"
    if mime_type == "application/json":
        return "json"
    if mime_type.startswith("text/"):
        return "text"
    if mime_type.startswith("image/"):
        return "image"
    return "unknown"

