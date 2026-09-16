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
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    """Build document dicts + embeddings from DocumentStore + LanceDB.

    The DocumentStore is canonical. We read provenance metadata (source_url,
    source_domain, quality_score) from the document_sources table when available,
    so curate_documents() can produce real scores instead of degraded defaults.

    Falls back to LanceDB document_embeddings() for document IDs when
    all_centroids() is empty (e.g. centroids not yet computed).
    """
    centroids = store.all_centroids()
    doc_embeddings = lance.document_embeddings()

    # If centroids are empty, fall back to document embeddings keys
    if not centroids and doc_embeddings:
        centroids = {doc_id: [] for doc_id in doc_embeddings}

    # Read provenance metadata (if document_sources table exists)
    sources_map: dict[str, dict] = {}
    try:
        sources_map = store.all_sources()
    except Exception:
        pass  # table may not exist yet on older corpora

    documents: list[dict[str, Any]] = []
    for doc_id, chunk_ids in centroids.items():
        doc = store.get_document(doc_id)
        if doc is None:
            continue
        text = doc.text or ""
        # Derive title from first non-empty line
        title = ""
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
            "published_at": "",
            "quality_score": float(src.get("quality_score") or 0.0),
            "canonical_url": src.get("source_url", ""),
            "content_hash": hashlib.sha256(text.encode()).hexdigest()[:32],
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


def _load_historical_embeddings(main_corpus_path: Path) -> list[list[float]]:
    """Load document embeddings from the main corpus LanceDB for novelty scoring.

    These represent the "already known" documents. curate_documents() uses them
    to compute novelty as cosine distance to the nearest historical document.
    """
    try:
        from ipa.indexes.lancedb_index import LanceDBIndex
        main_lance = main_corpus_path / "vector" / "lancedb"
        if main_lance.exists():
            lance = LanceDBIndex(main_lance)
            try:
                emb_map = lance.document_embeddings()
                return [v for v in emb_map.values() if v]
            finally:
                lance.close()
    except Exception:
        pass
    return []


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
        documents, doc_embeddings = build_document_dicts(store, lance)
        if not documents:
            return {"topics_new": 0, "parents_new": 0, "curated": 0,
                    "skipped": "no documents"}

        all_doc_ids = [d["document_id"] for d in documents]

        # --- Checkpoint: skip docs already clustered ---
        already_clustered = cluster_store.processed_doc_ids(stage="clustered")
        # Also include docs that are in existing clusters (pre-checkpoint era)
        existing_clusters = cluster_store.list_clusters()
        for c in existing_clusters:
            already_clustered.update(c.member_document_ids)

        unclustered = [d for d in documents if d["document_id"] not in already_clustered]

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
        already_curated = cluster_store.processed_doc_ids(stage="curated")
        to_curate = [d for d in documents if d["document_id"] not in already_curated]

        # Load real interests (config + user model) and historical embeddings
        interests = _load_interests()
        historical_embeddings: list[list[float]] = []
        main_url_hashes: dict[str, str] = {}
        if main_corpus_path is not None:
            historical_embeddings = _load_historical_embeddings(main_corpus_path)
            try:
                from ipa.storage.document_store import DocumentStore as _MainDS
                from ipa.reporter.reporter_curation import normalized_hash as _norm_hash
                _main_db = Path(main_corpus_path) / "document_store.db"
                if _main_db.exists():
                    _mstore = _MainDS(_main_db)
                    try:
                        _texts = _mstore.all_document_texts()
                        for doc_id, info in _mstore.all_sources().items():
                            u = info.get("source_url")
                            if u and doc_id in _texts:
                                main_url_hashes[u] = _norm_hash(_texts[doc_id])
                    finally:
                        _mstore.close()
            except Exception as exc:
                print(f"  [idle-enrich L1] main URL seed failed: {exc!r}", flush=True)

        curated_count = 0
        promoted_count = 0
        if to_curate:
            decisions = curate_documents(
                to_curate,
                report_id="idle-enrichment",
                period_start="2000-01-01T00:00:00Z",
                period_end="2100-01-01T00:00:00Z",
                interests=interests,
                quality_threshold=0.0,
                document_embeddings=doc_embeddings,
                historical_embeddings=historical_embeddings or None,
                known_url_hashes=main_url_hashes or None,
            )
            # Persist decisions atomically
            decision_dicts = [d.to_dict() for d in decisions]
            cluster_store.save_curation_decisions_batch(decision_dicts)

            # Checkpoint: mark curated docs
            curated_ids = [d["document_id"] for d in to_curate]
            cluster_store.mark_processed(curated_ids, stage="curated")
            curated_count = len(decisions)

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
                # already in the promotion queue
                docs_with_provenance = [
                    d for d in documents
                    if d["document_id"] in sources_map
                    and not cluster_store.is_promotion_pending(d["document_id"])
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
        documents, doc_embeddings = build_document_dicts(store, lance)
        if not documents:
            return {"labeled": 0, "classified": 0, "skipped": "no documents"}

        existing_clusters = cluster_store.list_clusters()

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

        # --- LLM classification of uncurated docs ---
        if not is_busy():
            already_curated = cluster_store.processed_doc_ids(stage="curated")
            to_curate = [d for d in documents if d["document_id"] not in already_curated]
            if to_curate:
                try:
                    decisions = curate_documents(
                        to_curate,
                        report_id="idle-enrichment-llm",
                        period_start="2000-01-01T00:00:00Z",
                        period_end="2100-01-01T00:00:00Z",
                        interests=(),
                        quality_threshold=0.0,
                        classifier_batch=(
                            lambda batch, llm_progress=None:
                            reporter_llm.classify_many(batch, (), progress_callback=llm_progress)
                        ),
                        document_embeddings=doc_embeddings,
                    )
                    decision_dicts = [d.to_dict() for d in decisions]
                    cluster_store.save_curation_decisions_batch(decision_dicts)
                    cluster_store.mark_processed(
                        [d["document_id"] for d in to_curate], stage="curated",
                    )
                except Exception:
                    pass  # curation is best-effort in Level 2

        return {
            "labeled": len(relabeled),
            "classified": classified,
            "parents_new": parents_new,
        }
    finally:
        store.close()
        lance.close()


__all__ = [
    "build_document_dicts",
    "enrich_corpus_level1",
    "enrich_corpus_level2",
]
