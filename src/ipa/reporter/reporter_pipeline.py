"""Reporter orchestration over an isolated corpus."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ipa.ingestion.fast_path import FastPathRunner
from ipa.reporter.corpus_service import CorpusService
from ipa.reporter.reporter_config import ReporterConfig
from ipa.reporter.reporter_contracts import sha256_hash
from ipa.reporter.reporter_curation import curate_documents
from ipa.reporter.reporter_metadata import load_scrape_records, normalize_article
from ipa.reporter.reporter_report import build_report, write_report
from ipa.reporter.reporter_representation import build_representation
from ipa.reporter.reporter_store import ReporterStore
from ipa.reporter.reporter_topics import discover_topics, match_topic_continuity


class ReporterPipeline:
    """Run acquisition-independent analysis in outputs/reporter only."""

    def __init__(self, config: ReporterConfig, output_dir: str | Path, embedding_adapter=None, llm_provider=None, main_corpus=None) -> None:
        config.validate()
        self.config = config
        self.output_dir = Path(output_dir)
        self.corpus_dir = self.output_dir / "corpus"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.store = ReporterStore(self.output_dir / "reporter.db")
        self.main_corpus = Path(main_corpus) if main_corpus else None
        self.corpus = CorpusService(self.corpus_dir, self.main_corpus)
        self.embedding_adapter = embedding_adapter
        self.llm = None
        if llm_provider is not None:
            from ipa.reporter.reporter_ai import ReporterLLM
            self.llm = ReporterLLM(llm_provider)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "ReporterPipeline":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _fingerprint(self) -> str:
        value = json.dumps({
            "corpus_id": self.config.corpus_id,
            "period": self.config.period.__dict__,
            "threshold": self.config.similarity_threshold,
            "weights": self.config.weights,
            "interests": self.config.interests,
        }, sort_keys=True)
        return sha256_hash(value)

    def _ingest(self, input_dir: Path, progress_callback=None) -> None:
        # Check if corpus was already populated (e.g. by fast path watch mode)
        existing_docs, existing_chunks = self.corpus.counts()
        store_db = self.corpus.store_path

        if existing_docs > 0 and existing_chunks > 0:
            # Corpus already populated by fast path â€” skip re-ingestion
            if progress_callback:
                progress_callback(15, f"corpus ya indexado ({existing_docs} docs, {existing_chunks} chunks)")
            # Still ensure Tantivy and LanceDB are up to date
            chunks = self.corpus.all_chunks()
            from ipa.indexes.tantivy_index import TantivyIndex
            tantivy_path = self.corpus_dir / "tantivy"
            if not tantivy_path.exists() and chunks:
                if progress_callback:
                    progress_callback(16, "indexando Tantivy")
                with TantivyIndex(tantivy_path) as index:
                    index.add_chunks(chunks)
            if chunks:
                lance_path = self.corpus_dir / "vector" / "lancedb"
                try:
                    import lancedb as _ldb
                    ldb = _ldb.connect(str(lance_path))
                    _resp = ldb.list_tables() if hasattr(ldb, "list_tables") else ldb.table_names()
                    tables = list(_resp.tables if hasattr(_resp, "tables") else _resp)
                    lance_count = ldb.open_table("chunks").count_rows() if "chunks" in tables else 0
                except Exception:
                    lance_count = 0
                if lance_count < len(chunks):
                    if progress_callback:
                        progress_callback(18, f"embeddings LanceDB ({len(chunks)} chunks)")
                    self._index_lancedb(chunks, progress_callback=progress_callback)
            return

        # Full ingestion from scratch
        runner = FastPathRunner(
            landing_db=self.corpus_dir / "landing.db",
            store_db=self.corpus_dir / "document_store.db",
            index_db=self.corpus_dir / "bm25_index.db",
            landing_root=input_dir,
            index_backend="bm25",
        )
        try:
            supported = {".txt", ".md", ".html", ".htm", ".json", ".pdf"}
            paths = [
                path for path in sorted(input_dir.rglob("*"))
                if path.is_file() and not path.name.startswith(".") and path.name != "scrape_report.json" and path.suffix.lower() in supported
            ]
            total = len(paths)
            for i, path in enumerate(paths):
                runner.ingest(path)
                if progress_callback and total > 0:
                    progress_callback(5 + int(10 * (i + 1) / total), f"ingesta {i+1}/{total}")
        finally:
            runner.close()
        from ipa.indexes.tantivy_index import TantivyIndex
        if progress_callback:
            progress_callback(16, "indexando Tantivy")
        chunks = self.corpus.all_chunks()
        with TantivyIndex(self.corpus_dir / "tantivy") as index:
            index.add_chunks(chunks)
        # Also index into LanceDB with BGE-M3 embeddings for hybrid retrieval
        if chunks:
            if progress_callback:
                progress_callback(18, f"embeddings LanceDB ({len(chunks)} chunks)")
            self._index_lancedb(chunks, progress_callback=progress_callback)

    def _index_lancedb(self, chunks: list, progress_callback=None) -> None:
        """Index chunks into LanceDB using BGE-M3 embeddings (dense + sparse).

        Tries to copy vectors from the main corpus LanceDB first (fast path).
        Only embeds NEW chunks that aren't in the main corpus (idempotent).
        """
        try:
            from ipa.indexes.embedding_adapter import EmbeddingAdapter
            from ipa.indexes.lancedb_index import LanceDBIndex
            lance_path = self.corpus_dir / "vector" / "lancedb"
            lance = LanceDBIndex(lance_path, vector_dim=1024)

            # Get existing chunk_ids in reporter LanceDB to skip already-indexed
            existing_ids: set[str] = set()
            if lance._table is not None:
                try:
                    tbl = lance._table.to_arrow()
                    if "chunk_id" in tbl.column_names:
                        existing_ids = set(tbl.column("chunk_id").to_pylist())
                except Exception:
                    pass

            # Filter to only new chunks
            new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
            if not new_chunks:
                lance.close()
                return

            # Try to copy vectors from main corpus LanceDB (avoids re-embedding)
            copied = 0
            remaining_chunks = new_chunks
            if self.main_corpus and (self.main_corpus / "vector" / "lancedb").exists():
                try:
                    import lancedb as _ldb
                    import pyarrow as pa
                    main_lance = _ldb.connect(str(self.main_corpus / "vector" / "lancedb"))
                    _resp = main_lance.list_tables() if hasattr(main_lance, "list_tables") else main_lance.table_names()
                    main_tables = list(_resp.tables if hasattr(_resp, "tables") else _resp)
                    if "chunks" in main_tables:
                        main_tbl = main_lance.open_table("chunks")
                        main_arrow = main_tbl.to_arrow()
                        if "chunk_id" in main_arrow.column_names:
                            main_ids = set(main_arrow.column("chunk_id").to_pylist())
                            # Find chunks that exist in main corpus
                            to_copy = [c for c in new_chunks if c.chunk_id in main_ids]
                            if to_copy:
                                if progress_callback:
                                    progress_callback(19, f"copiando {len(to_copy)} vectores del corpus principal")
                                # Build a lookup from main LanceDB
                                main_df = main_arrow.to_pandas()
                                main_lookup = {row["chunk_id"]: row for _, row in main_df.iterrows()}
                                batch = []
                                for chunk in to_copy:
                                    row = main_lookup.get(chunk.chunk_id)
                                    if row is None:
                                        continue
                                    vec = list(row["vector"])
                                    sparse_json = row.get("sparse_json", "")
                                    sparse = None
                                    if sparse_json:
                                        import json as _json
                                        try:
                                            sparse = {int(k): float(v) for k, v in _json.loads(sparse_json).items()}
                                        except Exception:
                                            pass
                                    batch.append((chunk, vec, sparse))
                                    if len(batch) >= 64:
                                        lance.add_chunks([b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch])
                                        copied += len(batch)
                                        batch.clear()
                                if batch:
                                    lance.add_chunks([b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch])
                                    copied += len(batch)
                                    batch.clear()
                                # Only embed chunks that weren't in main corpus
                                remaining_chunks = [c for c in new_chunks if c.chunk_id not in main_ids]
                                if progress_callback:
                                    progress_callback(20, f"copiados {copied} vectores Â· {len(remaining_chunks)} nuevos a embedear")
                except Exception as exc:
                    print(f"  [reporter] copy from main corpus failed: {exc}", flush=True)

            # Embed remaining chunks that weren't in main corpus
            if remaining_chunks:
                embedding = EmbeddingAdapter(batch_size=64, show_progress=False)
                BATCH = 64
                total_batches = (len(remaining_chunks) + BATCH - 1) // BATCH
                for i in range(0, len(remaining_chunks), BATCH):
                    batch = remaining_chunks[i:i + BATCH]
                    texts = [c.text for c in batch]
                    dense, sparse = embedding.embed_texts_hybrid(texts)
                    lance.add_chunks(batch, dense, sparse)
                    if progress_callback:
                        batch_num = i // BATCH + 1
                        progress_callback(20 + int(3 * batch_num / max(total_batches, 1)),
                                          f"embeddings nuevos {batch_num}/{total_batches}")
                embedding.close()

            # Compute document centroids (representative chunks per document)
            try:
                from ipa.indexes.lancedb_index import _compute_centroids
                from ipa.storage.document_store import DocumentStore as _DS
                with _DS(self.corpus_dir / "document_store.db") as ds:
                    _compute_centroids(ds, lance)
                    ds.commit()
            except Exception as exc:
                print(f"  [reporter] centroid computation skipped: {exc}", flush=True)

            lance.close()
        except Exception as exc:
            # LanceDB is optional â€” reporter can work with BM25 only
            print(f"  [reporter] LanceDB indexing skipped: {exc}", flush=True)

    def _documents(self, input_dir: Path, scrape_report: Path | None) -> list[dict[str, Any]]:
        records = load_scrape_records(scrape_report) if scrape_report and scrape_report.exists() else {}
        parsed_text = self.corpus.get_document_texts()
        parsed_db = self.corpus.store_path
        # Load representative chunk texts from centroids (computed during LanceDB indexing)
        representative_texts: dict[str, str] = {}
        if parsed_db.exists():
            try:
                with self.corpus.open_store() as ds:
                    centroids = ds.all_centroids()
                    for doc_id, chunk_ids in centroids.items():
                        parts = []
                        total_len = 0
                        for cid in chunk_ids:
                            chunk = ds.get_chunk(cid)
                            if chunk and chunk.text:
                                parts.append(chunk.text)
                                total_len += len(chunk.text)
                                if total_len >= 5000:
                                    break
                        if parts:
                            representative_texts[doc_id] = "\n\n".join(parts)[:5000]
            except Exception as exc:
                print(f"  [reporter] centroid load skipped: {exc}", flush=True)
        documents = []
        for path in sorted(input_dir.rglob("*")):
            if not path.is_file() or path.name.startswith(".") or path.name == "scrape_report.json" or path.suffix.lower() not in {".txt", ".md", ".html", ".htm", ".json", ".pdf"}:
                continue
            record = records.get(str(path)) or records.get(str(path.resolve()))
            try:
                metadata = normalize_article(path, record)
                raw_text = path.read_text(encoding="utf-8", errors="replace") if path.suffix.lower() != ".pdf" else ""
                text = parsed_text.get(metadata.artifact_id, raw_text) or metadata.title
                item = metadata.to_dict()
                representation = build_representation(text, item["title"])
                item["title"] = representation.title
                item["title_source"] = representation.title_source
                item["title_confidence"] = representation.title_confidence
                item["abstract"] = representation.abstract
                item["keywords"] = list(representation.keywords)
                item["representation_text"] = representation.embedding_text
                item["text"] = text
                # Use representative chunks (centroid-based) if available, else fallback to first 5000 chars
                item["representative_text"] = representative_texts.get(item["document_id"]) or text[:5000]
                documents.append(item)
                self.store.put_metadata(metadata)
            except (OSError, UnicodeError):
                continue
        self.store.commit()
        return documents

    def run(
        self,
        input_dir: str | Path,
        scrape_report: str | Path | None = None,
        previous_report: str | Path | None = None,
        use_embeddings: bool = False,
        progress_callback=None,
    ) -> dict[str, Any]:
        def progress(percent: int, stage: str, detail: str = "") -> None:
            if progress_callback is not None:
                progress_callback(percent, stage, detail)

        progress(5, "ingesta")
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(input_dir)
        report_id = "report:" + self.config.period.label + ":" + hashlib.sha256(self._fingerprint().encode()).hexdigest()[:16]
        self._ingest(input_dir, progress_callback=progress)
        progress(25, "documentos")
        documents = self._documents(input_dir, Path(scrape_report) if scrape_report else None)

        # --- Load document embeddings from LanceDB (centroids from chunk vectors) ---
        # This reuses the BGE-M3 embeddings already computed during fast path ingestion,
        # eliminating the need for a separate embedding pass later.
        progress(30, "cargando embeddings")
        doc_embeddings: dict[str, list[float]] = {}
        interest_embeddings: list[list[float]] = []
        historical_embeddings: list[list[float]] = []
        lancedb_path = self.corpus_dir / "vector" / "lancedb"
        if lancedb_path.exists():
            try:
                from ipa.indexes.lancedb_index import LanceDBIndex
                with LanceDBIndex(lancedb_path) as lance:
                    doc_embeddings = lance.document_embeddings()
            except Exception as exc:
                print(f"  [reporter] LanceDB embedding load skipped: {exc}", flush=True)
        # Embed interests once for semantic relevance scoring
        if self.config.interests and self.embedding_adapter and doc_embeddings:
            try:
                interest_embeddings = self.embedding_adapter.embed_texts(list(self.config.interests))
            except Exception as exc:
                print(f"  [reporter] interest embedding skipped: {exc}", flush=True)

        # --- Curation: use embeddings for semantic relevance/novelty (no LLM needed) ---
        progress(40, "curaciÃ³n")
        decisions = curate_documents(
            documents, report_id, self.config.period.start, self.config.period.end,
            self.config.interests, self.config.quality_threshold,
            classifier=(lambda document: self.llm.classify(document, self.config.interests)) if self.llm else None,
            classifier_batch=(lambda batch, llm_progress=None: self.llm.classify_many(batch, self.config.interests, progress_callback=llm_progress)) if self.llm else None,
            progress_callback=lambda current, total, title="": progress(
                40 + int(35 * current / max(total, 1)),
                f"curaciÃ³n {current}/{total}",
                f"Documento {current}/{total}: {title}" if title else f"Documento {current}/{total}",
            ),
            document_embeddings=doc_embeddings,
            interest_embeddings=interest_embeddings,
            historical_embeddings=historical_embeddings,
        )
        for decision in decisions:
            self.store.put_decision(decision, self.config.period.start or "")
        selected = [doc for doc, decision in zip(documents, decisions) if decision.decision.value not in {"duplicate", "irrelevant", "insufficient_evidence"}]

        # --- Topic clustering: reuse the same document embeddings (no re-embedding) ---
        # Build embeddings list for selected docs in the same order
        selected_embeddings = [doc_embeddings.get(doc["document_id"]) for doc in selected]
        # If any doc is missing embeddings, fall back to None (lexical similarity)
        if any(e is None for e in selected_embeddings):
            selected_embeddings = None
        progress(75, "agrupando tÃ³picos")
        categories = discover_topics(
            selected,
            similarity_threshold=self.config.similarity_threshold,
            min_documents=self.config.min_documents,
            allow_singletons=self.config.allow_singleton_topics,
            embeddings=selected_embeddings,
            labeler=(lambda group: self.llm.label(group)) if self.llm else None,
            labeler_batch=(lambda groups, llm_progress=None: self.llm.label_many(groups, progress_callback=llm_progress)) if self.llm else None,
            report_id=report_id,
            progress_callback=lambda current, total: progress(75 + int(15 * current / max(total, 1)), f"tÃ³pico {current}/{total}", f"Inferencia {current}/{total}: generando etiqueta de tÃ³pico"),
        )
        # Second pass: group fine-grained topics into broader parent categories
        from ipa.reporter.reporter_topics import group_topics_into_categories
        progress(88, "agrupando categorÃ­as generales")
        parent_categories = []
        if self.llm:
            import threading
            result_holder = {"result": None}
            def _group_worker():
                try:
                    result_holder["result"] = group_topics_into_categories(
                        categories,
                        llm_grouper=lambda summaries: self.llm.group_topics(summaries),
                    )
                except Exception as exc:
                    print(f"  [reporter] group_topics LLM failed: {exc}", flush=True)
                    result_holder["result"] = []
            worker = threading.Thread(target=_group_worker, daemon=True)
            worker.start()
            worker.join(timeout=300)  # 5 minute timeout (was 120s â€” too short for 20+ topics)
            if worker.is_alive():
                print("  [reporter] group_topics LLM timed out after 300s, using fallback", flush=True)
                parent_categories = group_topics_into_categories(categories, llm_grouper=None)
            else:
                parent_categories = result_holder["result"] or []
        else:
            parent_categories = group_topics_into_categories(categories, llm_grouper=None)
        previous = []
        if previous_report and Path(previous_report).exists():
            previous = json.loads(Path(previous_report).read_text(encoding="utf-8")).get("categories", [])
        links = match_topic_continuity(categories, previous)
        links_by_category = {link.current_category_id: link for link in links}
        for category in categories:
            link = links_by_category.get(category["category_id"])
            if link:
                category["evolution"] = link.relation.value
            self.store.put_topic(category["category_id"], report_id, category)
            for document_id in category["document_ids"]:
                self.store.put_topic_document(category["category_id"], document_id, 1.0, "Membership by connected similarity component")
        for link in links:
            self.store.put_topic_link(link)
        self.store.commit()
        source_refs = [{"source_id": doc["document_id"], "source_type": "document"} for doc in selected]
        progress(90, "redactando informe")
        report = build_report(
            report_id, self.config.corpus_id,
            {"start": self.config.period.start, "end": self.config.period.end, "label": self.config.period.label},
            categories, [decision.to_dict() for decision in decisions], source_refs,
            uncertainties=["La fecha puede ser desconocida o depender de metadata de la fuente."],
            parent_categories=parent_categories,
        )
        json_path, _ = write_report(report, self.output_dir)
        corpus_fingerprint = sha256_hash("|".join(sorted(str(doc.get("content_hash", "")) for doc in documents)))
        self.store.put_run(
            report_id, self.config.corpus_id, self.config.period.start, self.config.period.end,
            report["status"], self._fingerprint(), corpus_fingerprint, str(json_path),
            report["generation"]["generated_at"],
        )
        self.store.commit()
        progress(100, "completado")
        return report

