from __future__ import annotations

from pathlib import Path

from ipa.contracts import CanonicalDocument, DocumentChunk, SourceSpan
from ipa.reporter.corpus_service import CorpusService


def test_corpus_service_owns_canonical_paths_and_reads_documents(tmp_path: Path):
    service = CorpusService(tmp_path / "corpus")
    service.corpus_dir.mkdir(parents=True)
    with service.open_store() as store:
        document = CanonicalDocument(
            document_id="doc:1",
            pages=1,
            elements=[],
            source_spans=[SourceSpan("artifact:1", 1, 0, 4)],
            text="hello corpus",
            mime_type="text/plain",
            parser_id="text",
        )
        store.put_document(document, "artifact:1")
        store.put_chunks([
            DocumentChunk("chunk:1", "doc:1", "hash:1", "hello corpus", source_span=document.source_spans[0])
        ])
        store.commit()

    assert service.store_path == tmp_path / "corpus" / "document_store.db"
    assert service.counts() == (1, 1)
    assert service.get_document_texts() == {"artifact:1": "hello corpus"}
    assert [chunk.chunk_id for chunk in service.all_chunks()] == ["chunk:1"]
