"""Shared corpus access boundary for Reporter and agentic consumers."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ipa.storage.document_store import DocumentStore


class CorpusService:
    """Own canonical corpus paths and read-side access for consumers.

    Ingestion and index publication remain owned by their respective services.
    Reporter uses this boundary instead of reconstructing storage paths and
    opening canonical stores ad hoc throughout the pipeline.
    """

    def __init__(self, corpus_dir: str | Path, main_corpus: str | Path | None = None) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.main_corpus = Path(main_corpus) if main_corpus else None

    @property
    def store_path(self) -> Path:
        return self.corpus_dir / "document_store.db"

    @property
    def landing_path(self) -> Path:
        return self.corpus_dir / "landing.db"

    @property
    def lexical_path(self) -> Path:
        return self.corpus_dir / "tantivy"

    @property
    def vector_path(self) -> Path:
        return self.corpus_dir / "vector" / "lancedb"

    @contextmanager
    def open_store(self) -> Iterator[DocumentStore]:
        with DocumentStore(self.store_path) as store:
            yield store

    def counts(self) -> tuple[int, int]:
        if not self.store_path.exists():
            return 0, 0
        with self.open_store() as store:
            return store.count_documents(), store.count_chunks()

    def all_chunks(self):
        with self.open_store() as store:
            return list(store.all_chunks())

    def get_document_texts(self) -> dict[str, str]:
        if not self.store_path.exists():
            return {}
        with self.open_store() as store:
            return {
                str(artifact_id): text
                for artifact_id, text in store._conn.execute(
                    "SELECT artifact_id, text FROM documents WHERE tombstoned=0"
                )
            }
