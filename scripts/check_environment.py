"""Check optional dependencies without importing the production repository."""
from __future__ import annotations
import importlib.util, platform, sys

PACKAGES = {
    "pymupdf": "PDF fast path",
    "docling": "structured PDF candidate",
    "mineru": "structured PDF candidate",
    "ocrmypdf": "OCR candidate",
    "sentence_transformers": "dense embeddings",
    "chromadb": "prototype vector backend",
    "qdrant_client": "Qdrant vector backend",
    "lancedb": "LanceDB vector backend",
    "tantivy": "Tantivy lexical backend",
    "ollama": "local LLM client",
    "faster_whisper": "audio transcription",
    "crawl4ai": "web crawler",
    "opentelemetry": "observability",
}
print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
print("Optional capabilities:")
for package, purpose in PACKAGES.items():
    available = importlib.util.find_spec(package) is not None
    print(f"  {'OK' if available else 'MISSING':7} {package:22} {purpose}")
