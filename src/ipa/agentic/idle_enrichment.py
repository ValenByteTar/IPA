"""Idle enrichment: deterministic topic discovery + heuristic curation.

Two levels:
  Level 1 (idle light, 5 min): full discover_topics() on all unclustered docs
    + curate_documents() heuristic (no classifier) + match_topic_continuity
    + group_topics_into_categories (deterministic fallback).
    No LLM, no VRAM — reuses BGE-M3 embeddings from LanceDB.

  Level 2 (idle deep, 30+ min): loads the LLM for rich topic labels
    + LLM classification of gray documents + LLM topic grouping.
    Unloads the model when done so VRAM returns to the chat path.

Architectural invariants:
  - DocumentStore stays canonical; results go to derived stores only.
  - TopicClusterStore is rebuildable (derived index).
  - No physical promotion to MAIN_CORPUS — curation is analysis-only.
  - Provenance is preserved (generation.generator records the source).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ipa.agentic.topic_clusters import TopicCluster, TopicClusterStore
from ipa.tutor.tutor_contracts import FieldOrigin, GenerationProvenance


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def build_document_dicts(
    store: Any,
    lance: Any,
    doc_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    """Build document dicts + embeddings from DocumentStore + LanceDB.

    The DocumentStore is canonical. We read provenance metadata (source_url,
    source_domain, quality_score) from the document_sources table when available,
    so curate_documents() can produce real scores instead of degraded defaults.

    ``doc_ids``: build dicts only for those documents — Tier 0 already wrote
    title/published_at/normalized_hash into ``document_metadata`` at ingest,
    so the per-cycle full-corpus scan is no longer needed. An empty set
    short-circuits (no doc fetches, no LanceDB scan).

    Falls back to LanceDB document_embeddings() for document IDs when
    all_centroids() is empty (e.g. centroids not yet computed).
    """
    if doc_ids is not None and not doc_ids:
        return [], {}
    centroids = store.all_centroids()
    try:
        doc_embeddings = lance.document_embeddings(doc_ids)
    except TypeError:
        # Fakes/implementaciones sin el filtro: full read + filtro en Python.
        all_embs = lance.document_embeddings()
        doc_embeddings = (
            {k: v for k, v in all_embs.items() if doc_ids is None or k in doc_ids}
            if isinstance(all_embs, dict) else all_embs)

    # If centroids are empty, fall back to document embeddings keys
    if not centroids and doc_embeddings:
        centroids = {doc_id: [] for doc_id in doc_embeddings}

    # Read provenance + ingest-time derived metadata (both cheap row fetches)
    sources_map: dict[str, dict] = {}
    try:
        sources_map = store.all_sources()
    except Exception:
        pass  # table may not exist yet on older corpora
    meta_map: dict[str, dict] = {}
    try:
        meta_map = store.all_doc_meta()
    except Exception:
        pass  # document_metadata may not exist on older corpora

    documents: list[dict[str, Any]] = []
    for doc_id, chunk_ids in centroids.items():
        if doc_ids is not None and doc_id not in doc_ids:
            continue
        doc = store.get_document(doc_id)
        if doc is None:
            continue
        text = doc.text or ""
        meta = meta_map.get(doc_id) or {}
        # Title persists from ingest; derive only as fallback.
        title = meta.get("title") or ""
        if not title:
            for line in text.split("\n"):
                stripped = line.strip()
                if stripped:
                    title = stripped[:200]
                    break
        if not title:
            title = doc_id

        # Representative text from centroid chunks (if available)
        rep_parts: list[str] = []
        for cid in chunk_ids[:3]:
            chunk = store.get_chunk(cid)
            if chunk and chunk.text:
                rep_parts.append(chunk.text)
        rep_text = "\n\n".join(rep_parts)[:5000] if rep_parts else text[:5000]

        # Provenance metadata from document_sources (if available)
        src = sources_map.get(doc_id, {})

        documents.append({
            "document_id": doc_id,
            "title": title,
            "text": text,
            "representation_text": rep_text,
            "representative_text": rep_text,
            "source_domain": src.get("source_domain", ""),
            "source_url": src.get("source_url", ""),
            "published_at": meta.get("published_at") or src.get("published_at") or "",
            "quality_score": float(src.get("quality_score") or 0.0),
            "canonical_url": src.get("source_url", ""),
            "content_hash": hashlib.sha256(text.encode()).hexdigest()[:32],
            "normalized_hash": meta.get("normalized_hash") or "",
            "content_type": None,
        })

    return documents, doc_embeddings


def _discover_to_topic_clusters(
    topics: list[dict[str, Any]],
    parent_categories: list[dict[str, Any]],
    centroids: dict[str, list[str]],
    *,
    generator: str = "idle-enrichment",
    model_fingerprint: str = "bge-m3-centroids",
) -> list[TopicCluster]:
    """Convert discover_topics() output dicts to TopicCluster records."""
    parent_map: dict[str, str] = {}
    for parent in parent_categories:
        for sub_id in parent.get("subtopic_ids", []):
            parent_map[sub_id] = parent["category_id"]

    now = _now()
    records: list[TopicCluster] = []
    for topic in topics:
        doc_ids = topic["document_ids"]
        rep_chunk = ""
        if doc_ids:
            chunk_ids = centroids.get(doc_ids[0], [])
            if chunk_ids:
                rep_chunk = chunk_ids[0]

        gen = topic.get("generation", {})
        generation = GenerationProvenance(
            generator=generator,
            generated_at=gen.get("generated_at", now),
            input_hash=gen.get("input_hash", ""),
            model_fingerprint=model_fingerprint,
        )

        records.append(TopicCluster(
            cluster_id=topic["category_id"],
            label=topic["label"],
            description=topic.get("description"),
            member_document_ids=doc_ids,
            member_concept_ids=[],
            parent_cluster_id=parent_map.get(topic["category_id"]),
            coherence_score=float(topic.get("cohesion", 0.0)),
            representative_chunk_id=rep_chunk,
            created_at=now,
            generation=generation,
            field_origins=topic.get("field_origins", {
                "label": FieldOrigin.GENERATED.value,
                "member_document_ids": FieldOrigin.SOURCE.value,
                "coherence_score": FieldOrigin.GENERATED.value,
            }),
        ))

    # Save parent categories as clusters too
    for parent in parent_categories:
        gen = parent.get("generation", {})
        records.append(TopicCluster(
            cluster_id=parent["category_id"],
            label=parent["label"],
            description=parent.get("description"),
            member_document_ids=parent.get("document_ids", []),
            member_concept_ids=parent.get("subtopic_ids", []),
            parent_cluster_id=None,
            coherence_score=float(parent.get("cohesion", 1.0)),
            representative_chunk_id="",
            created_at=now,
            generation=GenerationProvenance(
                generator=f"{generator}-grouper",
                generated_at=gen.get("generated_at", now),
                input_hash=gen.get("input_hash", ""),
                model_fingerprint=model_fingerprint,
            ),
            field_origins={
                "label": FieldOrigin.GENERATED.value,
                "member_document_ids": FieldOrigin.GENERATED.value,
            },
        ))

    return records


def _previous_topics_from_store(
    cluster_store: TopicClusterStore,
) -> list[dict[str, Any]]:
    """Extract previous topics from the cluster store for continuity matching."""
    previous: list[dict[str, Any]] = []
    for c in cluster_store.list_clusters():
        previous.append({
            "category_id": c.cluster_id,
            "label": c.label,
            "description": c.description or "",
            "document_ids": c.member_document_ids,
        })
    return previous


def _load_interests() -> tuple[str, ...]:
    """Load interests from reporter config + user model (TutorStore).

    Config provides a static baseline. The user model's active topic records
    augment with topics the user is actively learning. Both are combined
    into a tuple of interest strings for curate_documents().
    """
    interests: list[str] = []

    # 1. Static config baseline
    try:
        import yaml
        from pathlib import Path as _P
        cfg_path = _P("configs/reporter.yaml")
        if cfg_path.exists():
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            cfg_interests = data.get("reporter", {}).get("interests", [])
            if cfg_interests:
                interests.extend(str(i) for i in cfg_interests)
    except Exception:
        pass

    # 2. User model augmentation (active topic records from TutorStore)
    try:
        from ipa.tutor.tutor_runtime import TutorStore
        from pathlib import Path as _P
        tutor_db = _P("outputs/agent/tutor.db")
        if tutor_db.exists():
            store = TutorStore(tutor_db)
            try:
                records = store.list_topic_records()
                for r in records:
                    if r.topic_id and r.topic_id not in interests:
                        interests.append(r.topic_id)
            finally:
                store.close()
    except Exception:
        pass

    return tuple(interests)


def _refresh_novelty_hint(
    hint: dict[str, Any],
    doc_emb: list[float],
    new_embs: dict[str, list[float]],
) -> dict[str, Any]:
    """Recomputa el max cosine de un hint solo contra los docs NUEVOS de main.

    Mantiene el máximo viejo si ningún doc nuevo es más cercano; devuelve una
    copia del hint con max_cosine/nearest_doc_id actualizados.
    """
    from ipa.reporter.reporter_curation import _cosine_similarity
    best = float(hint.get("max_cosine") or 0.0)
    best_id = hint.get("nearest_doc_id")
    for nid, emb in new_embs.items():
        if not emb:
            continue
        sim = _cosine_similarity(list(doc_emb), list(emb))
        if sim > best:
            best, best_id = sim, nid
    return {**hint, "max_cosine": best, "nearest_doc_id": best_id}


def _load_historical_embeddings(main_corpus_path: Path) -> tuple[list[str], list[list[float]]]:
    """Load document embeddings from the main corpus LanceDB for novelty scoring.

    These represent the "already known" documents. curate_documents() uses them
    to compute novelty as cosine distance to the nearest historical document.
    Returns (document_ids, embeddings) aligned — the ids let the novelty gate
    confirm a near-identical match lexically against that document's text.
    """
    try:
        from ipa.indexes.lancedb_index import LanceDBIndex
        main_lance = main_corpus_path / "vector" / "lancedb"
        if main_lance.exists():
            lance = LanceDBIndex(main_lance)
            try:
                emb_map = lance.document_embeddings()
                pairs = [(k, v) for k, v in emb_map.items() if v]
                return [k for k, _ in pairs], [v for _, v in pairs]
            finally:
                lance.close()
    except Exception:
        pass
    return [], []


def enrich_corpus_level1(
    corpus_path: Path,
    cluster_store: TopicClusterStore,
    *,
    main_corpus_path: Path | None = None,
) -> dict[str, Any]:
    """Level 1: deterministic enrichment (no LLM, no VRAM).

    Runs full discover_topics() on all unclustered documents, heuristic
    curation (no classifier), topic continuity matching, and deterministic
    topic grouping. Results are saved atomically to the TopicClusterStore.

    Atomicity: clusters are saved in a single batch transaction. Curation
    decisions are persisted in a separate batch. A checkpoint table tracks
    which documents have been clustered and curated, so a crash mid-pass
    is resumable — the next run skips already-processed docs.
    """
    from ipa import DocumentStore
    from ipa.indexes.lancedb_index import LanceDBIndex
    from ipa.reporter.reporter_topics import (
        discover_topics,
        group_topics_into_categories,
        match_topic_continuity,
    )
    from ipa.reporter.reporter_curation import curate_documents

    store_db = corpus_path / "document_store.db"
    lance_dir = corpus_path / "vector" / "lancedb"
    if not store_db.exists() or not lance_dir.exists():
        return {"topics_new": 0, "parents_new": 0, "curated": 0,
                "skipped": "corpus not found"}

    store = DocumentStore(store_db)
    lance = LanceDBIndex(lance_dir)
    try:
        # --- Filter-first: set diffs on cheap id/metadata queries BEFORE
        #     fetching any document text. The per-cycle O(corpus) scan is gone;
        #     only docs with actual pending work get dicts built. ---
        centroids = store.all_centroids()
        all_doc_ids = set(centroids)
        if not all_doc_ids:
            doc_embs_probe = lance.document_embeddings()
            all_doc_ids = set(doc_embs_probe)
        if not all_doc_ids:
            return {"topics_new": 0, "parents_new": 0, "curated": 0,
                    "skipped": "no documents"}

        try:
            sources_map = store.all_sources()
        except Exception:
            sources_map = {}
        try:
            meta_map = store.all_doc_meta()
        except Exception:
            meta_map = {}

        already_clustered = cluster_store.processed_doc_ids(stage="clustered")
        existing_clusters = cluster_store.list_clusters()
        for c in existing_clusters:
            already_clustered.update(c.member_document_ids)
        already_curated = cluster_store.processed_doc_ids(stage="curated")

        unclustered_ids = all_doc_ids - already_clustered
        uncurated_ids = all_doc_ids - already_curated
        # Policy re-eval runs for every doc with provenance not already
        # tracked by the queue — ids only, no text needed. Una sola query a
        # promotion_queue (incluye 'promoted': re-evaluarlos re-encolaba un
        # doc ya promovido → churn pendiente→done en cada ciclo).
        queued_ids = cluster_store.promotion_queue_doc_ids()
        pending_eval_ids = [
            d for d in all_doc_ids
            if d in sources_map and d not in queued_ids
        ]

        if not unclustered_ids and not uncurated_ids and not pending_eval_ids:
            return {"topics_new": 0, "parents_new": 0, "curated": 0,
                    "promoted": 0, "unclustered_before": 0,
                    "continuity_links": 0, "skipped": "nothing to do"}

        needed_ids = unclustered_ids | uncurated_ids
        documents, doc_embeddings = build_document_dicts(
            store, lance, doc_ids=needed_ids)
        by_id = {d["document_id"]: d for d in documents}
        unclustered = [by_id[i] for i in unclustered_ids if i in by_id]
        to_curate = [by_id[i] for i in uncurated_ids if i in by_id]

        topics_new = 0
        parents_new = 0
        continuity_links = 0

        if unclustered:
            # Embeddings for unclustered docs
            unclustered_embeddings = [
                doc_embeddings.get(d["document_id"]) for d in unclustered
            ]
            if any(e is None for e in unclustered_embeddings):
                unclustered_embeddings = None

            # --- discover_topics (deterministic, no labeler) ---
            topics = discover_topics(
                unclustered,
                similarity_threshold=0.52,
                min_documents=2,
                allow_singletons=False,
                embeddings=unclustered_embeddings,
                report_id="idle-enrichment",
            )

            # --- group_topics_into_categories (deterministic fallback) ---
            parent_categories = group_topics_into_categories(topics, llm_grouper=None)

            # --- match_topic_continuity against existing clusters ---
            previous = _previous_topics_from_store(cluster_store)
            links = match_topic_continuity(topics, previous)
            continuity_links = len(links)

            # --- Convert to TopicCluster and save atomically ---
            centroids = store.all_centroids()
            new_clusters = _discover_to_topic_clusters(
                topics, parent_categories, centroids,
            )
            cluster_store.save_clusters_batch(new_clusters)
            topics_new = len(topics)
            parents_new = len(parent_categories)

            # --- Checkpoint: mark clustered docs ---
            clustered_doc_ids: set[str] = set()
            for t in topics:
                clustered_doc_ids.update(t["document_ids"])
            # Singletons (not in any topic) are also "processed" — they
            # were considered and didn't match anything. Mark them so we
            # don't reprocess them next cycle.
            singleton_ids = {d["document_id"] for d in unclustered} - clustered_doc_ids
            cluster_store.mark_processed(
                list(clustered_doc_ids | singleton_ids), stage="clustered",
            )

        # --- Heuristic curation on docs not yet curated (no LLM) ---
        # Tier 0 ya marcó duplicados exactos contra main al ingerir: decisión
        # DUPLICATE directa, sin gastar scoring ni embeddings.
        dup_flagged = [
            d for d in to_curate
            if (meta_map.get(d["document_id"], {}).get("extra") or {})
            .get("duplicate_of_main")
        ]
        curated_count = 0
        promoted_count = 0
        if dup_flagged:
            try:
                from ipa.reporter.reporter_contracts import (
                    ReporterDecision as _RD, ReviewStatus as _RS,
                )
                dup_decisions = [{
                    "decision_id": f"dup:{d['document_id']}",
                    "report_id": "tier0-ingest",
                    "document_id": d["document_id"],
                    "artifact_id": d.get("content_hash", ""),
                    "decision": _RD.DUPLICATE.value,
                    "reason": "Contenido idéntico (hash normalizado) ya presente en main.",
                    "duplicate_of": "__main__",
                    "scores": {},
                    "evidence": [],
                    "review_status": _RS.APPROVED.value,
                } for d in dup_flagged]
                cluster_store.save_curation_decisions_batch(dup_decisions)
                cluster_store.mark_processed(
                    [d["document_id"] for d in dup_flagged], stage="curated")
                curated_count += len(dup_flagged)
                to_curate = [d for d in to_curate
                             if d["document_id"] not in {x["document_id"] for x in dup_flagged}]
            except Exception as exc:
                print(f"  [idle-enrich L1] dup-flag decisions failed: {exc!r}", flush=True)

        if to_curate:
            interests = _load_interests()
            main_url_hashes: dict[str, str] = {}
            novelty_hints: dict[str, dict] = {}
            historical_embeddings: list[list[float]] = []
            historical_documents: list[dict[str, Any]] = []
            need_historical = bool(to_curate)
            # Cuando el corpus curado ES main (topify_main), "novelty vs main"
            # incluiría al propio doc: cosine/histórico/refresh darían
            # DUPLICATE de sí mismo (self-match = 1.0). Se excluyen los docs
            # en curación de todas las comparaciones contra main.
            same_corpus = bool(
                main_corpus_path
                and Path(main_corpus_path).resolve() == corpus_path.resolve())
            _curate_ids = {d["document_id"] for d in to_curate} if same_corpus else set()
            _curate_urls = (
                {str(d.get("source_url") or d.get("canonical_url") or "")
                 for d in to_curate} if same_corpus else set())
            if main_corpus_path is not None:
                _main_db = Path(main_corpus_path) / "document_store.db"
                if _main_db.exists():
                    from ipa.storage.document_store import DocumentStore as _MainDS
                    _mstore = _MainDS(_main_db)
                    try:
                        # url → normalized_hash ya persistido por Tier 0 /
                        # backfill acotado para docs legacy — nada de hashing
                        # de textos por ciclo.
                        from ipa.ingestion.ingest_metadata import backfill_doc_metadata
                        # Acotado por ciclo: el corpus main legacy son ~4k
                        # docs — sin límite el primer ciclo T1 los hashearía
                        # todos de una pasada.
                        backfill_doc_metadata(_mstore, limit=int(
                            os.environ.get("IPA_T1_META_BACKFILL_LIMIT", "1000") or 1000))
                        main_url_hashes = _mstore.url_normalized_hashes()
                        if _curate_urls:
                            main_url_hashes = {
                                u: h for u, h in main_url_hashes.items()
                                if u not in _curate_urls}
                        main_count = _mstore.count_documents()
                        main_latest = (_mstore._conn.execute(
                            "SELECT MAX(stored_at) FROM documents WHERE tombstoned = 0"
                        ).fetchone() or [None])[0]
                        stale_hints: dict[str, dict] = {}

                        def _accept_hint(did: str, h: dict) -> None:
                            if float(h.get("max_cosine") or 0.0) >= 0.95:
                                # Gate de dos factores (DEC-003): la
                                # confirmación Jaccard necesita el texto del
                                # doc matcheado — se fetchea solo ese.
                                mdoc = _mstore.get_document(h.get("nearest_doc_id"))
                                h = {**h, "nearest_text": (mdoc.text if mdoc else "")}
                            novelty_hints[did] = h

                        for d in to_curate:
                            did = d["document_id"]
                            h = (meta_map.get(did, {}).get("extra") or {}).get("novelty_hint")
                            if h is None:
                                continue  # sin hint → path histórico
                            # Válido si count Y snapshot coinciden. El count
                            # detecta adds en el mismo segundo; el ts detecta
                            # tombstone+add que deja el count igual. Hints
                            # viejos sin ts se validan solo por count.
                            h_ts = h.get("main_latest_stored_at")
                            if (h.get("main_doc_count") == main_count
                                    and (h_ts is None or h_ts == main_latest)):
                                _accept_hint(did, h)
                            else:
                                stale_hints[did] = h

                        # Refresh incremental: un hint stale NO descarta — se
                        # re-verifica solo contra los docs agregados a main
                        # desde el snapshot (main_latest_stored_at). O(nuevos)
                        # en vez de O(corpus): sin esto cada promoción
                        # invalidaba TODOS los hints y forzaba la recarga
                        # completa de embeddings+textos históricos.
                        if stale_hints:
                            try:
                                from ipa.indexes.lancedb_index import LanceDBIndex as _MainLance
                                earliest = min(
                                    str(h.get("main_latest_stored_at") or "")
                                    for h in stale_hints.values())
                                stored_map = _mstore.all_document_stored_at()
                                new_ids = [d for d, ts in stored_map.items()
                                           if ts > earliest]
                                _mlance_dir = Path(main_corpus_path) / "vector" / "lancedb"
                                new_embs: dict[str, Any] = {}
                                if new_ids and _mlance_dir.exists():
                                    _ml = _MainLance(_mlance_dir)
                                    try:
                                        new_embs = _ml.document_embeddings(new_ids)
                                    finally:
                                        _ml.close()
                                elif not new_ids:
                                    new_embs = {}  # solo removals → nada nuevo que comparar
                                else:
                                    new_embs = {}  # lance ausente → no refrescable
                                for did, h in list(stale_hints.items()):
                                    emb = doc_embeddings.get(did)
                                    if emb is None or (new_ids and not new_embs):
                                        continue  # no refrescable → histórico
                                    # same_corpus: el propio doc puede estar en
                                    # new_ids → cosine 1.0 consigo mismo.
                                    h2 = _refresh_novelty_hint(
                                        h, emb, {k: v for k, v in new_embs.items()
                                                 if k not in _curate_ids})
                                    h2["main_doc_count"] = main_count
                                    if main_latest:
                                        h2["main_latest_stored_at"] = main_latest
                                    # Self-heal: el hint refrescado se
                                    # persiste — el próximo ciclo ya es válido.
                                    try:
                                        store.put_doc_meta(
                                            did, extra={"novelty_hint": h2})
                                        store.commit()
                                    except Exception:
                                        pass
                                    _accept_hint(did, h2)
                                    del stale_hints[did]
                            except Exception as exc:
                                print(f"  [idle-enrich L1] hint refresh failed: "
                                      f"{exc!r}", flush=True)
                        need_historical = any(
                            d["document_id"] not in novelty_hints
                            for d in to_curate)
                        if need_historical:
                            hist_ids, historical_embeddings = _load_historical_embeddings(
                                main_corpus_path)
                            if _curate_ids:
                                # same_corpus: excluir los docs en curación —
                                # cosine(self)=1.0 → falso DUPLICATE.
                                keep = [i for i, d in enumerate(hist_ids)
                                        if d not in _curate_ids]
                                hist_ids = [hist_ids[i] for i in keep]
                                historical_embeddings = [
                                    historical_embeddings[i] for i in keep]
                            _texts = _mstore.all_document_texts()
                            # Aligned with historical_embeddings: the novelty gate
                            # confirms a >0.95 cosine match lexically against this
                            # document's text before marking a duplicate.
                            historical_documents = [
                                {"document_id": d, "text": _texts.get(d, "")}
                                for d in hist_ids
                            ]
                    finally:
                        _mstore.close()

            decisions = curate_documents(
                to_curate,
                report_id="idle-enrichment",
                period_start="2000-01-01T00:00:00Z",
                period_end="2100-01-01T00:00:00Z",
                interests=interests,
                quality_threshold=0.0,
                document_embeddings=doc_embeddings,
                historical_embeddings=historical_embeddings or None,
                historical_documents=historical_documents or None,
                known_url_hashes=main_url_hashes or None,
                novelty_hints=novelty_hints or None,
            )
            # Persist decisions atomically
            decision_dicts = [d.to_dict() for d in decisions]
            cluster_store.save_curation_decisions_batch(decision_dicts)

            # Checkpoint: mark curated docs
            curated_ids = [d["document_id"] for d in to_curate]
            cluster_store.mark_processed(curated_ids, stage="curated")
            curated_count += len(decisions)

        # --- Evaluate promotion policy for ALL documents with provenance ---
        # This runs on every cycle, not just newly curated docs, because:
        # 1. Documents curated before the promotion policy existed need evaluation
        # 2. Provenance may have been backfilled after curation
        # 3. The policy is the gate to physical promotion, not curation
        try:
            from ipa.agentic.promotion_policy import evaluate_batch
            sources_map = store.all_sources()
            if sources_map:
                # Build decisions map from stored curation decisions
                stored_decisions = cluster_store.list_curation_decisions()
                decision_map: dict[str, Any] = {}
                for sd in stored_decisions:
                    decision_map[sd.get("document_id")] = sd

                # Evaluate all documents that have provenance but aren't
                # already in the promotion queue — ids only (set diff ya
                # calculado arriba; no hace falta el texto del doc).
                docs_with_provenance = [
                    {"document_id": d} for d in pending_eval_ids
                ]
                if docs_with_provenance:
                    # Build lightweight decision objects for the policy
                    from ipa.reporter.reporter_contracts import (
                        ReporterDocumentDecision, ReporterDecision, ScoreBundle,
                        ReviewStatus,
                    )
                    policy_decisions = []
                    for d in docs_with_provenance:
                        doc_id = d["document_id"]
                        stored = decision_map.get(doc_id)
                        if stored and "scores" in stored:
                            scores = stored["scores"]
                            sb = ScoreBundle(
                                relevance=float(scores.get("relevance", 0.5)),
                                novelty=float(scores.get("novelty", 0.5)),
                                source_quality=float(scores.get("source_quality", 0.5)),
                                impact=float(scores.get("impact", 0.5)),
                                depth=float(scores.get("depth", 0.5)),
                                actionability=float(scores.get("actionability", 0.5)),
                            )
                            decision_str = stored.get("decision", "reporter_only")
                            try:
                                decision_enum = ReporterDecision(decision_str)
                            except ValueError:
                                decision_enum = ReporterDecision.REPORTER_ONLY
                            try:
                                review_enum = ReviewStatus(str(stored.get("review_status", "pending")))
                            except ValueError:
                                review_enum = ReviewStatus.PENDING
                            policy_decisions.append(ReporterDocumentDecision(
                                decision_id=stored.get("decision_id", ""),
                                report_id=stored.get("report_id", "idle-enrichment"),
                                document_id=doc_id,
                                artifact_id=stored.get("artifact_id", ""),
                                scores=sb,
                                decision=decision_enum,
                                reason=stored.get("reason", ""),
                                evidence=stored.get("evidence", []),
                                generation=stored.get("generation") or {},
                                duplicate_of=stored.get("duplicate_of"),
                                review_status=review_enum,
                                approval=stored.get("approval"),
                                field_origins=stored.get("field_origins", {}),
                            ))
                        else:
                            # No curation decision — use defaults
                            # configured_scrape doesn't need scores, so this
                            # is fine for auto-promote
                            policy_decisions.append(None)

                    # Evaluate batch — pass None decisions where we don't have them
                    promo_inputs = []
                    rejected_doc_ids: list[str] = []
                    for i, d in enumerate(docs_with_provenance):
                        prov = sources_map.get(d["document_id"], {}).get("provenance", "unknown")
                        decision = policy_decisions[i] if i < len(policy_decisions) else None
                        scores = decision.scores if decision else None
                        from ipa.agentic.promotion_policy import evaluate_promotion
                        pd = evaluate_promotion(d["document_id"], prov, decision, scores)
                        if pd.should_promote:
                            cluster_store.mark_promotion_pending(
                                pd.document_id, pd.reason, pd.provenance,
                                source_corpus=str(corpus_path),
                            )
                            promoted_count += 1
                        elif pd.reason in ("duplicate document", "insufficient evidence"):
                            # Definitive rejection (DEC-007): the duplicate's
                            # content already lives in main / the doc has no
                            # usable text. Mark rejected so the Landing sweep
                            # deletes the redundant file instead of leaving it
                            # in Transit forever. agent_research sub-threshold
                            # stays pending (human confirmation).
                            rejected_doc_ids.append(pd.document_id)
                    if rejected_doc_ids:
                        try:
                            n_rejected = cluster_store.mark_curation_rejected(rejected_doc_ids)
                            print(f"  [idle-enrich L1] {n_rejected} definitive rejections marked "
                                  f"(sweep deletes their files)", flush=True)
                        except Exception as exc:
                            print(f"  [idle-enrich L1] rejection marking failed: {exc!r}", flush=True)
        except Exception as exc:
            # Visible in the dashboard stdout — a silent failure here blocks
            # all promotions and looks like "promoted=0" with no cause.
            print(f"  [idle-enrich L1] promotion evaluation error: {exc!r}", flush=True)

        return {
            "topics_new": topics_new,
            "parents_new": parents_new,
            "curated": curated_count,
            "promoted": promoted_count,
            "unclustered_before": len(unclustered),
            "continuity_links": continuity_links,
        }
    finally:
        store.close()
        lance.close()


def enrich_corpus_level2(
    corpus_path: Path,
    cluster_store: TopicClusterStore,
    llm_provider: Any,
    *,
    is_busy: Any = lambda: False,
) -> dict[str, Any]:
    """Level 2: LLM enrichment for rich labels + gray classification.

    Loads the LLM provider (assumed already loaded by caller), re-labels
    clusters that have fallback labels, and runs LLM-assisted discovery
    on unclustered docs. Checks ``is_busy`` before each LLM call and
    aborts gracefully if the user returns.

    Atomicity: re-labeled clusters are saved in a batch. New LLM-discovered
    clusters are saved in a separate batch. LLM curation decisions are
    persisted atomically. Checkpoint prevents reprocessing on resume.
    """
    from ipa import DocumentStore
    from ipa.indexes.lancedb_index import LanceDBIndex
    from ipa.reporter.reporter_ai import ReporterLLM
    from ipa.reporter.reporter_topics import (
        discover_topics,
        group_topics_into_categories,
    )
    from ipa.reporter.reporter_curation import curate_documents

    store_db = corpus_path / "document_store.db"
    lance_dir = corpus_path / "vector" / "lancedb"
    if not store_db.exists() or not lance_dir.exists():
        return {"labeled": 0, "classified": 0, "skipped": "corpus not found"}

    store = DocumentStore(store_db)
    lance = LanceDBIndex(lance_dir)
    reporter_llm = ReporterLLM(llm_provider)
    try:
        existing_clusters = cluster_store.list_clusters()
        # Filter-first igual que L1: los dicts solo alimentan el discovery de
        # docs no clusterizados — el relabel fetchea por id, la zona gris usa
        # sus propias decisiones.
        already_clustered = cluster_store.processed_doc_ids(stage="clustered")
        for c in existing_clusters:
            already_clustered.update(c.member_document_ids)
        unclustered_ids = set(store.all_centroids()) - already_clustered
        documents, doc_embeddings = build_document_dicts(
            store, lance, doc_ids=unclustered_ids or set())

        # Gray-zone eligibility se computa antes del early-exit: los
        # candidatos no necesitan dicts ni clusters — un corpus quieto puede
        # seguir teniendo agent_research borderline pendientes de revisión.
        gray_lo = gray_hi = gray_limit = 0
        gray_candidates: list[tuple[str, dict, Any]] = []
        try:
            from ipa.agentic.promotion_policy import AGENT_RESEARCH_THRESHOLD
            from ipa.reporter.reporter_curation import (
                normalized_hash, promotion_score)
            from ipa.reporter.reporter_contracts import ScoreBundle
            gray_lo = float(os.environ.get("IPA_GRAY_LO", "0.5") or 0.5)
            gray_hi = float(os.environ.get(
                "IPA_GRAY_HI", str(AGENT_RESEARCH_THRESHOLD))
                or AGENT_RESEARCH_THRESHOLD)
            gray_limit = int(os.environ.get("IPA_GRAY_LIMIT", "24") or 24)
            reviewed = cluster_store.stage_doc_ids("gray_reviewed")
            try:
                gray_sources = store.all_sources()
            except Exception:
                gray_sources = {}
            for sd in cluster_store.list_curation_decisions():
                did = sd.get("document_id")
                if not did or did in reviewed:
                    continue
                if cluster_store.is_promotion_pending(did):
                    continue
                if sd.get("decision") != "reporter_only":
                    continue
                if gray_sources.get(did, {}).get("provenance") != "agent_research":
                    continue
                sc = sd.get("scores") or {}
                try:
                    sb = ScoreBundle(
                        relevance=float(sc.get("relevance", 0.5)),
                        novelty=float(sc.get("novelty", 0.5)),
                        source_quality=float(sc.get("source_quality", 0.5)),
                        impact=float(sc.get("impact", 0.5)),
                        depth=float(sc.get("depth", 0.5)),
                        actionability=float(sc.get("actionability", 0.5)),
                    )
                except (TypeError, ValueError):
                    continue
                if gray_lo <= promotion_score(sb) < gray_hi:
                    gray_candidates.append((did, sd, sb))
            gray_candidates = gray_candidates[:gray_limit]
        except Exception as exc:
            print(f"  [idle-enrich L2] gray-zone scan failed: {exc!r}",
                  flush=True)

        if not documents and not existing_clusters and not gray_candidates:
            return {"labeled": 0, "classified": 0, "gray_reviewed": 0,
                    "gray_rescued": 0, "skipped": "no documents"}

        # --- Re-label clusters that have deterministic fallback labels ---
        # Collect re-labeled clusters and save them in a batch at the end.
        relabeled: list[TopicCluster] = []
        for cluster in existing_clusters:
            if is_busy():
                break
            # Only re-label clusters created by Level 1 (deterministic)
            if cluster.generation.generator != "idle-enrichment":
                continue
            if cluster.generation.model_fingerprint != "bge-m3-centroids":
                continue  # already LLM-labeled

            group: list[dict[str, Any]] = []
            for doc_id in cluster.member_document_ids[:8]:
                doc = store.get_document(doc_id)
                if doc:
                    title = ""
                    for line in (doc.text or "").split("\n"):
                        s = line.strip()
                        if s:
                            title = s[:200]
                            break
                    if not title:
                        title = doc_id
                    group.append({
                        "document_id": doc_id,
                        "title": title,
                        "text": (doc.text or "")[:500],
                    })
            if not group:
                continue
            try:
                label_result = reporter_llm.label(group)
            except Exception:
                continue
            if label_result.get("label"):
                updated = replace(
                    cluster,
                    label=label_result["label"],
                    description=label_result.get("description") or cluster.description,
                    generation=GenerationProvenance(
                        generator="idle-enrichment-llm",
                        generated_at=_now(),
                        input_hash=cluster.generation.input_hash,
                        model_fingerprint="qwen35-9b-exl3",
                    ),
                )
                relabeled.append(updated)

        # Save re-labeled clusters atomically
        if relabeled:
            cluster_store.save_clusters_batch(relabeled)

        if is_busy():
            return {"labeled": len(relabeled), "classified": 0, "aborted": "user returned"}

        # --- LLM-assisted discovery on unclustered docs ---
        already_clustered = cluster_store.processed_doc_ids(stage="clustered")
        for c in existing_clusters:
            already_clustered.update(c.member_document_ids)
        unclustered = [d for d in documents if d["document_id"] not in already_clustered]

        classified = 0
        parents_new = 0
        if unclustered:
            unclustered_embeddings = [
                doc_embeddings.get(d["document_id"]) for d in unclustered
            ]
            if any(e is None for e in unclustered_embeddings):
                unclustered_embeddings = None

            topics = discover_topics(
                unclustered,
                similarity_threshold=0.52,
                min_documents=2,
                allow_singletons=False,
                embeddings=unclustered_embeddings,
                labeler_batch=(
                    lambda groups, progress_callback=None:
                    reporter_llm.label_many(groups, progress_callback=progress_callback)
                ),
                report_id="idle-enrichment-llm",
            )

            if is_busy():
                return {"labeled": len(relabeled), "classified": 0,
                        "aborted": "user returned during discovery"}

            parent_categories = group_topics_into_categories(
                topics,
                llm_grouper=lambda summaries: reporter_llm.group_topics(summaries),
            )
            parents_new = len(parent_categories)

            centroids = store.all_centroids()
            new_clusters = _discover_to_topic_clusters(
                topics, parent_categories, centroids,
                generator="idle-enrichment-llm",
                model_fingerprint="qwen35-9b-exl3",
            )
            cluster_store.save_clusters_batch(new_clusters)
            classified = len(topics)

            # Checkpoint: mark clustered docs
            clustered_doc_ids: set[str] = set()
            for t in topics:
                clustered_doc_ids.update(t["document_ids"])
            singleton_ids = {d["document_id"] for d in unclustered} - clustered_doc_ids
            cluster_store.mark_processed(
                list(clustered_doc_ids | singleton_ids), stage="clustered",
            )

        # --- Gray zone: segunda opinión LLM sobre agent_research borderline ---
        # El bloque original de "curación LLM de docs sin curar" era dead code:
        # L1 marca TODOS los docs como stage="curated" en el mismo checkpoint,
        # así que to_curate siempre salía vacío. Lo que sí falta en el sistema
        # es resolver la zona gris: agent_research con promotion_score en
        # [GRAY_LO, AGENT_RESEARCH_THRESHOLD) — quedan pending a confirmación
        # humana que puede no llegar nunca.
        gray_rescued = 0
        gray_reviewed = 0
        if gray_candidates and not is_busy():
            try:
                from ipa.agentic.promotion_policy import evaluate_promotion
                gray_docs = []
                for did, sd, sb in gray_candidates:
                    doc = store.get_document(did)
                    if doc is None or not (doc.text or "").strip():
                        cluster_store.mark_stage([did], "gray_reviewed")
                        gray_reviewed += 1
                        continue
                    gray_docs.append((did, sd, sb, doc))
                if gray_docs:
                    batch = [{
                        "document_id": did,
                        "title": did,
                        "text": (doc.text or "")[:2000],
                    } for did, _, _, doc in gray_docs]
                    semantics = reporter_llm.classify_many(batch, ())
                    fingerprint = (
                        getattr(llm_provider, "model_fingerprint", None)
                        or getattr(llm_provider, "engine", None)
                        or "llm")
                    reviewed_ids: list[str] = []
                    for (did, sd, sb, doc), sem in zip(gray_docs, semantics):
                        if is_busy():
                            break
                        if not isinstance(sem, dict) or not sem:
                            continue  # sin veredicto → no marcar: reintenta

                        def _sf(raw: Any, fallback: float) -> float:
                            try:
                                return max(0.0, min(1.0, float(raw)))
                            except (TypeError, ValueError):
                                return fallback
                        new_scores = ScoreBundle(
                            relevance=_sf(sem.get("relevance"), sb.relevance),
                            novelty=_sf(sem.get("novelty"), sb.novelty),
                            source_quality=_sf(sem.get("source_quality"), sb.source_quality),
                            impact=_sf(sem.get("impact"), sb.impact),
                            depth=_sf(sem.get("depth"), sb.depth),
                            actionability=_sf(sem.get("actionability"), sb.actionability),
                        )
                        new_decision = dict(sd)
                        new_decision["report_id"] = "idle-enrichment-llm"
                        new_decision["scores"] = {
                            k: getattr(new_scores, k) for k in (
                                "relevance", "novelty", "source_quality",
                                "impact", "depth", "actionability")}
                        new_decision["reason"] = (
                            f"{sem.get('reason') or sd.get('reason', '')} "
                            "[segunda opinión LLM — zona gris]")
                        new_decision["generation"] = {
                            "generator": "idle-enrichment-grayzone",
                            "generated_at": _now(),
                            "model_fingerprint": fingerprint,
                            "input_hash": normalized_hash(doc.text or ""),
                        }
                        cluster_store.save_curation_decisions_batch([new_decision])
                        pd = evaluate_promotion(
                            did, "agent_research", None, new_scores)
                        if pd.should_promote:
                            cluster_store.mark_promotion_pending(
                                did, pd.reason + " (gray-zone LLM)",
                                "agent_research",
                                source_corpus=str(corpus_path))
                            gray_rescued += 1
                        reviewed_ids.append(did)
                        gray_reviewed += 1
                    if reviewed_ids:
                        cluster_store.mark_stage(
                            reviewed_ids, "gray_reviewed")
            except Exception as exc:
                print(f"  [idle-enrich L2] gray zone failed: {exc!r}", flush=True)

        return {
            "labeled": len(relabeled),
            "classified": classified,
            "parents_new": parents_new,
            "gray_rescued": gray_rescued,
            "gray_reviewed": gray_reviewed,
        }
    finally:
        store.close()
        lance.close()


__all__ = [
    "build_document_dicts",
    "enrich_corpus_level1",
    "enrich_corpus_level2",
]
