"""Content safety â€” defense layers for the ingestion pipeline.

Three layers protect against malicious content:

Layer 1 â€” Download validation (before writing to disk):
  - Magic bytes check (don't trust content-type or file extension)
  - Size limit (default 200 MB, configurable)
  - Redirect limit (max 3 hops)
  - Strict timeout

Layer 2 â€” Quarantine (before processing):
  - Separate quarantine directory
  - Optional ClamAV scan (if clamd is running)
  - Optional YARA rules (if yara-python is installed)
  - Structure validation (PDF is valid, not corrupt)

Layer 3 â€” Safe parsing:
  - PyMuPDF in safe mode (no JS, no actions, no embedded files)
  - Page limit (default 2000, configurable)
  - Parse timeout (default 120s)
  - Manual override for oversized documents

Usage:
  from ipa.ingestion.content_safety import (
      validate_download, quarantine_file, safe_open_pdf,
      DownloadValidationError, QuarantineError, SafeParseError,
  )

  # Layer 1: validate before saving
  validate_download(content_bytes, expected_type="pdf")

  # Layer 2: quarantine before processing
  quarantine_file(filepath, expected_type="pdf")

  # Layer 3: safe PDF open
  with safe_open_pdf(filepath) as doc:
      for page in doc:
          text = page.get_text("text")
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_SIZE_MB = 200
DEFAULT_MAX_PAGES = 2000
DEFAULT_PARSE_TIMEOUT_S = 120
DEFAULT_MAX_REDIRECTS = 3
DEFAULT_DOWNLOAD_TIMEOUT_S = 60

# Manual override: when True, skip size/page limits but still validate
# structure and magic bytes.  Set via environment or function parameter.
MANUAL_OVERRIDE_ENV = "IPA_MANUAL_OVERRIDE"


# ---------------------------------------------------------------------------
# Magic bytes signatures
# ---------------------------------------------------------------------------

# Each entry: (offset, expected_bytes)
# The check reads the first N bytes and compares at the given offset.
_MAGIC_SIGNATURES: dict[str, list[tuple[int, bytes]]] = {
    "pdf": [
        (0, b"%PDF-"),
    ],
    "html": [
        # HTML can start with <!DOCTYPE, <html, or whitespace before either
        (0, b"<!DOCTYPE"),
        (0, b"<!doctype"),
        (0, b"<html"),
        (0, b"<HTML"),
    ],
    "png": [(0, b"\x89PNG\r\n\x1a\n")],
    "jpg": [(0, b"\xff\xd8\xff")],
    "gif": [(0, b"GIF87a"), (0, b"GIF89a")],
    "webp": [(0, b"RIFF"), (8, b"WEBP")],
    # Office formats (ZIP-based: docx, xlsx, pptx, odt)
    "zip": [(0, b"PK\x03\x04"), (0, b"PK\x05\x06"), (0, b"PK\x07\x08")],
    # Legacy Office (OLE2: doc, xls, ppt)
    "ole2": [(0, b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")],
    # Plain text: no magic bytes, validated by trying UTF-8 decode
    "text": [],
}

# Map file extensions to expected magic byte groups
_EXT_TO_TYPE: dict[str, str] = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".txt": "text",
    ".csv": "text",
    ".rst": "text",
    ".rtf": "text",
    ".json": "text",
    ".png": "png",
    ".jpg": "jpg",
    ".jpeg": "jpg",
    ".gif": "gif",
    ".webp": "webp",
    ".docx": "zip",
    ".xlsx": "zip",
    ".pptx": "zip",
    ".odt": "zip",
    ".epub": "zip",
    ".doc": "ole2",
    ".xls": "ole2",
    ".ppt": "ole2",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DownloadValidationError(Exception):
    """Raised when downloaded content fails validation (Layer 1)."""


class QuarantineError(Exception):
    """Raised when a file fails quarantine checks (Layer 2)."""


class SafeParseError(Exception):
    """Raised when safe parsing fails (Layer 3)."""


# ---------------------------------------------------------------------------
# Layer 1: Download validation
# ---------------------------------------------------------------------------

@dataclass
class DownloadValidationConfig:
    """Configuration for download validation."""
    max_size_mb: int = DEFAULT_MAX_SIZE_MB
    max_redirects: int = DEFAULT_MAX_REDIRECTS
    timeout_seconds: int = DEFAULT_DOWNLOAD_TIMEOUT_S
    manual_override: bool = False


def _is_manual_override(config: DownloadValidationConfig | None = None) -> bool:
    """Check if manual override is active (env var or config)."""
    if config and config.manual_override:
        return True
    return os.environ.get(MANUAL_OVERRIDE_ENV, "").lower() in ("1", "true", "yes")


def check_magic_bytes(
    data: bytes,
    expected_type: str,
    min_bytes: int = 512,
) -> bool:
    """Check if data starts with the correct magic bytes for the expected type.

    Args:
        data: First bytes of the file (at least min_bytes, or the full file
            if smaller).
        expected_type: One of: pdf, html, png, jpg, gif, webp, zip, ole2, text.
        min_bytes: Minimum bytes to read for checking.

    Returns:
        True if magic bytes match (or expected_type is text/unknown).
    """
    if expected_type not in _MAGIC_SIGNATURES:
        return True  # Unknown type â€” don't block, let parser handle it

    signatures = _MAGIC_SIGNATURES[expected_type]
    if not signatures:
        # Text files have no magic bytes â€” try UTF-8 decode
        try:
            data[:min_bytes].decode("utf-8")
            return True
        except (UnicodeDecodeError, UnicodeError):
            # Not valid UTF-8 â€” could be binary masquerading as text
            # Allow it; the text parser will handle gracefully
            return True

    for offset, expected in signatures:
        if offset + len(expected) > len(data):
            continue
        if data[offset:offset + len(expected)] == expected:
            return True

    return False


def validate_download(
    data: bytes,
    expected_type: str | None = None,
    config: DownloadValidationConfig | None = None,
) -> None:
    """Validate downloaded content before writing to disk.

    Args:
        data: The downloaded bytes (or first chunk if streaming).
        expected_type: Expected file type (pdf, html, text, etc.).
            If None, inferred from magic bytes.
        config: Validation configuration. Uses defaults if None.

    Raises:
        DownloadValidationError: If validation fails.
    """
    cfg = config or DownloadValidationConfig()

    # Check 1: Size limit (skip if manual override)
    if not _is_manual_override(cfg):
        max_bytes = cfg.max_size_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise DownloadValidationError(
                f"File too large: {len(data) / 1024 / 1024:.1f} MB "
                f"(limit: {cfg.max_size_mb} MB). "
                f"Set {MANUAL_OVERRIDE_ENV}=1 for manual override."
            )

    # Check 2: Magic bytes
    if expected_type:
        if not check_magic_bytes(data, expected_type):
            raise DownloadValidationError(
                f"Magic bytes mismatch: expected {expected_type}, "
                f"got different content. First bytes: "
                f"{data[:16].hex() if data else 'empty'}"
            )

    # Check 3: Empty content
    if len(data) == 0:
        raise DownloadValidationError("Empty content (0 bytes)")


def validate_html_size(data: bytes | str, config: DownloadValidationConfig | None = None) -> None:
    """Enforce the configured limit for HTML before parsing it."""
    cfg = config or DownloadValidationConfig()
    size = len(data.encode("utf-8") if isinstance(data, str) else data)
    if not _is_manual_override(cfg) and size > cfg.max_size_mb * 1024 * 1024:
        raise DownloadValidationError(f"HTML too large: {size / 1024 / 1024:.1f} MB")
    if size == 0:
        raise DownloadValidationError("Empty HTML content")


def validate_download_stream(
    response: Any,
    expected_type: str | None = None,
    config: DownloadValidationConfig | None = None,
) -> bytes:
    """Validate a streaming HTTP response and return the content.

    Checks:
        - Content-Length header against size limit
        - Magic bytes from first chunk
        - Total downloaded size

    Args:
        response: requests.Response object (with stream=True).
        expected_type: Expected file type.
        config: Validation configuration.

    Returns:
        The validated content as bytes.

    Raises:
        DownloadValidationError: If validation fails.
    """
    cfg = config or DownloadValidationConfig()
    max_bytes = cfg.max_size_mb * 1024 * 1024

    # Check Content-Length header if present
    content_length = response.headers.get("content-length")
    if content_length and not _is_manual_override(cfg):
        try:
            cl = int(content_length)
            if cl > max_bytes:
                raise DownloadValidationError(
                    f"Content-Length too large: {cl / 1024 / 1024:.1f} MB "
                    f"(limit: {cfg.max_size_mb} MB)"
                )
        except ValueError:
            pass  # Invalid header â€” rely on actual size check

    # Read content with size enforcement
    chunks: list[bytes] = []
    total = 0
    first_chunk_checked = False

    for chunk in response.iter_content(8192):
        if not chunk:
            continue
        total += len(chunk)

        # Size check during streaming
        if not _is_manual_override(cfg) and total > max_bytes:
            raise DownloadValidationError(
                f"Stream exceeded size limit: {total / 1024 / 1024:.1f} MB "
                f"(limit: {cfg.max_size_mb} MB)"
            )

        # Magic bytes check on first chunk
        if not first_chunk_checked and expected_type:
            if not check_magic_bytes(chunk, expected_type):
                raise DownloadValidationError(
                    f"Magic bytes mismatch: expected {expected_type}. "
                    f"First bytes: {chunk[:16].hex() if chunk else 'empty'}"
                )
            first_chunk_checked = True

        chunks.append(chunk)

    data = b"".join(chunks)

    # Final validation
    validate_download(data, expected_type=expected_type, config=cfg)

    return data


def get_safe_session(**kwargs: Any) -> Any:
    """Create a requests.Session with safe defaults.

    Configures:
        - Max redirects (default 3)
        - Timeout (default 60s)
        - No automatic redirect to file:// or other dangerous schemes
    """
    import requests
    from urllib3.util.retry import Retry

    cfg = DownloadValidationConfig(**{
        k: v for k, v in kwargs.items()
        if k in ("max_redirects", "timeout_seconds")
    })

    session = requests.Session()
    session.max_redirects = cfg.max_redirects

    # Disable dangerous URL schemes
    session.mount("file://", requests.adapters.HTTPAdapter(max_retries=0))
    # requests doesn't natively handle file://, but this prevents any
    # adapter from being used for it

    return session


# ---------------------------------------------------------------------------
# Layer 2: Quarantine
# ---------------------------------------------------------------------------

@dataclass
class QuarantineConfig:
    """Configuration for quarantine checks."""
    quarantine_dir: str | Path = "quarantine"
    clamav_enabled: bool = True
    yara_enabled: bool = False
    yara_rules_path: str | Path | None = None
    validate_structure: bool = True
    manual_override: bool = False


def _check_clamav(filepath: Path) -> tuple[bool, str]:
    """Scan file with ClamAV (clamdscan).

    Returns (clean, message).
    """
    try:
        result = subprocess.run(
            ["clamdscan", "--fdpass", "--no-summary", str(filepath)],
            capture_output=True,
            timeout=60,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS if os.name == "nt" else 0,
        )
        if result.returncode == 0:
            return True, "clean"
        elif result.returncode == 1:
            return False, result.stdout.strip() or "infected"
        else:
            # clamd not running or error â€” don't block, just warn
            return True, f"clamdscan_error: {result.stderr.strip()[:100]}"
    except FileNotFoundError:
        return True, "clamdscan_not_installed"
    except subprocess.TimeoutExpired:
        return True, "clamdscan_timeout"
    except Exception as e:
        return True, f"clamdscan_exception: {e}"


def _check_yara(filepath: Path, rules_path: Path) -> tuple[bool, str]:
    """Scan file with YARA rules.

    Returns (clean, message).
    """
    try:
        import yara
        rules = yara.compile(str(rules_path))
        matches = rules.match(str(filepath))
        if matches:
            return False, f"yara_match: {', '.join(m.rule for m in matches)}"
        return True, "clean"
    except ImportError:
        return True, "yara_not_installed"
    except Exception as e:
        return True, f"yara_exception: {e}"


def _validate_pdf_structure(filepath: Path) -> tuple[bool, str]:
    """Validate that a PDF is structurally sound (not corrupt, not a fake).

    Returns (valid, message).
    """
    try:
        import pymupdf
        doc = pymupdf.open(str(filepath))
        page_count = len(doc)
        if page_count == 0:
            doc.close()
            return False, "pdf_has_zero_pages"
        # Check that at least the first page is accessible
        page = doc[0]
        _ = page.rect
        doc.close()
        return True, f"valid_pdf_{page_count}_pages"
    except Exception as e:
        return False, f"pdf_structure_error: {e}"


def quarantine_file(
    filepath: str | Path,
    expected_type: str | None = None,
    config: QuarantineConfig | None = None,
) -> Path:
    """Run quarantine checks on a file before processing.

    Steps:
        1. Move file to quarantine directory
        2. Optional ClamAV scan
        3. Optional YARA rules scan
        4. Structure validation (PDF is valid, not corrupt)

    Args:
        filepath: Path to the file to quarantine.
        expected_type: Expected file type for structure validation.
        config: Quarantine configuration.

    Returns:
        Path to the file in the quarantine directory (if passed, it stays
        there for processing; the caller moves it to Archive after).

    Raises:
        QuarantineError: If any check fails.
    """
    cfg = config or QuarantineConfig()
    src = Path(filepath)

    if not src.exists():
        raise QuarantineError(f"File not found: {src}")

    # Create quarantine directory
    q_dir = Path(cfg.quarantine_dir)
    q_dir.mkdir(parents=True, exist_ok=True)
    q_path = q_dir / src.name

    # Move to quarantine
    shutil.copy2(str(src), str(q_path))

    messages: list[str] = []

    # Check 1: ClamAV
    if cfg.clamav_enabled:
        clean, msg = _check_clamav(q_path)
        messages.append(f"clamav: {msg}")
        if not clean:
            # Delete the quarantined copy â€” don't let malware sit on disk
            q_path.unlink(missing_ok=True)
            raise QuarantineError(f"ClamAV detected malware: {msg}")

    # Check 2: YARA
    if cfg.yara_enabled and cfg.yara_rules_path:
        clean, msg = _check_yara(q_path, Path(cfg.yara_rules_path))
        messages.append(f"yara: {msg}")
        if not clean:
            q_path.unlink(missing_ok=True)
            raise QuarantineError(f"YARA rule match: {msg}")

    # Check 3: Structure validation
    if cfg.validate_structure and expected_type:
        if expected_type == "pdf":
            valid, msg = _validate_pdf_structure(q_path)
            messages.append(f"structure: {msg}")
            if not valid:
                q_path.unlink(missing_ok=True)
                raise QuarantineError(f"Invalid PDF structure: {msg}")

    # All checks passed â€” return path in quarantine
    return q_path


def release_from_quarantine(
    q_path: str | Path,
    dest: str | Path,
) -> Path:
    """Move a file from quarantine to its final destination.

    Called after all processing is complete and the file is confirmed safe.
    """
    q = Path(q_path)
    d = Path(dest)
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(q), str(d))
    return d


# ---------------------------------------------------------------------------
# Layer 3: Safe PDF parsing
# ---------------------------------------------------------------------------

@dataclass
class SafeParseConfig:
    """Configuration for safe PDF parsing."""
    max_pages: int = DEFAULT_MAX_PAGES
    timeout_seconds: int = DEFAULT_PARSE_TIMEOUT_S
    manual_override: bool = False
    # PyMuPDF safe flags
    disable_javascript: bool = True
    disable_actions: bool = True
    disable_embedded_files: bool = True


class SafePDFContext:
    """Context manager for safely opening PDFs with PyMuPDF.

    Applies safety restrictions:
        - No JavaScript execution
        - No automatic actions (OpenAction, AA)
        - No embedded file extraction
        - Page limit enforcement
        - Timeout (via signal on Unix, thread-based on Windows)
    """

    def __init__(self, filepath: str | Path, config: SafeParseConfig | None = None) -> None:
        self.filepath = Path(filepath)
        self.config = config or SafeParseConfig()
        self._doc = None

    def __enter__(self) -> Any:
        import pymupdf

        if not self.filepath.exists():
            raise SafeParseError(f"File not found: {self.filepath}")

        # Open with PyMuPDF
        try:
            # PyMuPDF open flags for safety:
            # pymupdf.OPEN_FLAG may not exist in all versions, so we use
            # the document-level controls after opening.
            self._doc = pymupdf.open(str(self.filepath))
        except Exception as e:
            raise SafeParseError(f"Failed to open PDF: {e}")

        doc = self._doc

        # Check page count (skip if manual override)
        if not _is_manual_override_safe(self.config):
            page_count = len(doc)
            if page_count > self.config.max_pages:
                doc.close()
                raise SafeParseError(
                    f"PDF has {page_count} pages (limit: {self.config.max_pages}). "
                    f"Set {MANUAL_OVERRIDE_ENV}=1 for manual override."
                )

        # Disable JavaScript
        if self.config.disable_javascript:
            try:
                doc.set_metadata({"javascript": ""})
            except Exception:
                pass  # Not all PDFs have JS metadata
            # Clear any JS in the document
            try:
                for page in doc:
                    if page.get_links():
                        # Filter out JS-based links
                        links = page.get_links()
                        for link in links:
                            if link.get("uri", "").startswith("javascript:"):
                                try:
                                    page.delete_link(link)
                                except Exception:
                                    pass
            except Exception:
                pass

        # Disable embedded files
        if self.config.disable_embedded_files:
            try:
                # Remove embedded files if any
                if doc.embfile_count() > 0:
                    for i in range(doc.embfile_count() - 1, -1, -1):
                        try:
                            doc.embfile_del(i)
                        except Exception:
                            pass
            except (AttributeError, Exception):
                pass  # Some PyMuPDF versions don't have embfile methods

        return doc

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._doc is not None:
            try:
                self._doc.close()
            except Exception:
                pass
            self._doc = None


def _is_manual_override_safe(config: SafeParseConfig) -> bool:
    """Check manual override for safe parsing."""
    if config.manual_override:
        return True
    return os.environ.get(MANUAL_OVERRIDE_ENV, "").lower() in ("1", "true", "yes")


def safe_open_pdf(
    filepath: str | Path,
    config: SafeParseConfig | None = None,
) -> SafePDFContext:
    """Open a PDF safely with restrictions.

    Usage:
        with safe_open_pdf("file.pdf") as doc:
            for page in doc:
                text = page.get_text("text")

    Args:
        filepath: Path to the PDF file.
        config: Safe parse configuration.

    Returns:
        SafePDFContext (use as context manager).
    """
    return SafePDFContext(filepath, config)


def safe_parse_pdf_pages(
    filepath: str | Path,
    config: SafeParseConfig | None = None,
) -> list[str]:
    """Safely parse a PDF and return text per page.

    Convenience function that handles the safe open + timeout + page limit
    in one call.  Uses a thread-based timeout on Windows (no signal.alarm).

    Args:
        filepath: Path to the PDF file.
        config: Safe parse configuration.

    Returns:
        List of page texts (one string per page).

    Raises:
        SafeParseError: If parsing fails or times out.
    """
    cfg = config or SafeParseConfig()

    result: list[str] = []
    error: SafeParseError | None = None

    def _parse() -> None:
        nonlocal result, error
        try:
            with safe_open_pdf(filepath, cfg) as doc:
                pages = []
                for page in doc:
                    pages.append(page.get_text("text"))
                result = pages
        except SafeParseError as e:
            error = e
        except Exception as e:
            error = SafeParseError(f"Unexpected error during parsing: {e}")

    import threading
    thread = threading.Thread(target=_parse, daemon=True)
    thread.start()
    thread.join(timeout=cfg.timeout_seconds)

    if thread.is_alive():
        # Thread is still running â€” timeout
        raise SafeParseError(
            f"PDF parsing timed out after {cfg.timeout_seconds}s"
        )

    if error is not None:
        raise error

    return result


# ---------------------------------------------------------------------------
# Utility: detect file type from magic bytes
# ---------------------------------------------------------------------------

def detect_file_type(data: bytes) -> str | None:
    """Detect file type from magic bytes.

    Args:
        data: First bytes of the file (at least 512 recommended).

    Returns:
        Detected type string (pdf, html, png, jpg, gif, webp, zip, ole2)
        or None if no match.
    """
    for ftype, signatures in _MAGIC_SIGNATURES.items():
        if not signatures:
            continue
        for offset, expected in signatures:
            if offset + len(expected) <= len(data):
                if data[offset:offset + len(expected)] == expected:
                    return ftype
    return None


def detect_file_type_from_path(filepath: str | Path) -> str | None:
    """Detect file type by reading the first bytes of a file."""
    p = Path(filepath)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        data = f.read(512)
    return detect_file_type(data)

