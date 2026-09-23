"""Topic clusters: emergent grouping over existing centroids (Fase 3).

Deterministic agglomerative clustering over document centroids (BGE-M3
embeddings already computed). No fixed K — the number of clusters emerges
from a cosine similarity threshold, mirroring the approach validated in
Reporter's discover_topics().

Contracts: TopicCluster (contracts/topic_cluster.schema.json).
Invariants: topic_clusters_are_emergent, topic_clusters_are_derived_not_
authoritative (clusters are a derived index; DocumentStore stays canonical).

Consumers: TopicNavigator (multi-hop retrieval, Fase 3), Tutor roadmap
prerequisites, memory consolidation.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ipa.tutor.tutor_contracts import (
    FieldOrigin,
    GenerationProvenance,
    SourceRef,
    SourceType,
)

DEFAULT_CLUSTER_STORE = Path("outputs/agent/topic_clusters.db")

# Similarity thresholds (empirical, mirroring Reporter discover_topics):
# - >= MERGE_THRESHOLD: two documents belong to the same cluster
# - cluster centroid similarity >= PARENT_THRESHOLD: parent/child hierarchy
MERGE_THRESHOLD = 0.52
PARENT_THRESHOLD = 0.65
MIN_CLUSTER_SIZE = 2  # singletons stay unclustered (they are their own topic)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass(frozen=True)
class TopicCluster:
    """Contract-shaped runtime record (contracts/topic_cluster.schema.json)."""
    cluster_id: str
    label: str
    description: str | None
    member_document_ids: list[str]
    member_concept_ids: list[str]
    parent_cluster_id: str | None
    coherence_score: float
    representative_chunk_id: str
    created_at: str
    generation: GenerationProvenance
    field_origins: dict[str, str]

    def to_contract(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["field_origins"] = dict(self.field_origins)
        return payload


def _label_from_texts(texts: list[str], max_terms: int = 4) -> str:
    """Deterministic label: most frequent non-stopword terms across members."""
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for",
        "on", "with", "as", "by", "that", "this", "it", "from", "at", "be",
        "el", "la", "los", "las", "de", "y", "o", "en", "es", "un", "una",
        "para", "con", "por", "que", "del", "al", "como", "más", "su",
    }
    counts: dict[str, int] = {}
    for text in texts:
        for word in re.findall(r"[a-zA-Záéíóúñü]{3,}", text.lower()):
            if word not in stop:
                counts[word] = counts.get(word, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_terms]
    return " / ".join(word for word, _ in top) if top else "cluster"


class TopicClusterStore:
    """SQLite persistence for TopicCluster records (derived index, rebuildable).

    Also tracks enrichment progress (checkpoint) and curation decisions
    so the idle enrichment worker is atomic, resumable, and doesn't
    recompute already-finished work.
    """

    def __init__(self, store_path: str | Path = DEFAULT_CLUSTER_STORE) -> None:
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.store_path), timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout = 30000")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS topic_clusters (
                cluster_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                coherence_score REAL NOT NULL DEFAULT 0,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS enrichment_progress (
                document_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                processed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS curation_decisions (
                decision_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                report_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS promotion_queue (
                document_id TEXT PRIMARY KEY,
                reason TEXT NOT NULL,
                provenance TEXT NOT NULL,
                source_corpus TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                queued_at TEXT NOT NULL,
                promoted_at TEXT
            );
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            -- Checkpoints NO ordenados (enrichment_progress es una
            -- progresión clustered<curated de una fila por doc; marcas
            -- ortogonales como gray_reviewed necesitan su propio espacio
            -- para no pisar ese estado).
            CREATE TABLE IF NOT EXISTS aux_progress (
                document_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                processed_at TEXT NOT NULL,
                PRIMARY KEY (document_id, stage)
            );
        """)
        # Additive migration for stores created before coherence_score was
        # materialized. Derived stores remain rebuildable, but opening an old
        # store must not crash the runtime.
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(topic_clusters)")}
        if "coherence_score" not in columns:
            self._conn.execute("ALTER TABLE topic_clusters ADD COLUMN coherence_score REAL NOT NULL DEFAULT 0")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # --- Meta KV (dirty flags, last-seen counts — gate "corpus changed") ----

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value))
        self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def save_cluster(self, cluster: TopicCluster) -> None:
        payload = cluster.to_contract()
        self._conn.execute(
            "INSERT OR REPLACE INTO topic_clusters "
            "(cluster_id, label, coherence_score, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (cluster.cluster_id, cluster.label, cluster.coherence_score,
             json.dumps(payload, ensure_ascii=False), cluster.created_at),
        )
        self._conn.commit()

    def save_clusters_batch(self, clusters: list[TopicCluster]) -> None:
        """Save all clusters in a single transaction (atomic batch).

        Either all clusters are committed or none — if the process crashes
        mid-batch, no partial results are visible.
        """
        self._conn.execute("BEGIN")
        try:
            for cluster in clusters:
                payload = cluster.to_contract()
                self._conn.execute(
                    "INSERT OR REPLACE INTO topic_clusters "
                    "(cluster_id, label, coherence_score, payload_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (cluster.cluster_id, cluster.label, cluster.coherence_score,
                     json.dumps(payload, ensure_ascii=False), cluster.created_at),
                )
            self._conn.commit()
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # --- Enrichment checkpoint ------------------------------------------------

    def mark_processed(self, document_ids: list[str], stage: str = "clustered") -> None:
        """Mark documents as processed by the enrichment worker.

        stage values: "clustered" (topic discovery done), "curated" (curation done).
        Uses INSERT OR REPLACE (UPSERT) so re-marking with a higher stage
        (e.g. clustered → curated) updates the existing row.
        """
        now = _now()
        self._conn.execute("BEGIN")
        try:
            for doc_id in document_ids:
                self._conn.execute(
                    "INSERT OR REPLACE INTO enrichment_progress (document_id, stage, processed_at) "
                    "VALUES (?, ?, ?)",
                    (doc_id, stage, now),
                )
            self._conn.commit()
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def update_stage(self, document_ids: list[str], stage: str) -> None:
        """Update the processing stage for already-tracked documents."""
        now = _now()
        self._conn.execute("BEGIN")
        try:
            for doc_id in document_ids:
                self._conn.execute(
                    "UPDATE enrichment_progress SET stage = ?, processed_at = ? "
                    "WHERE document_id = ?",
                    (stage, now, doc_id),
                )
            self._conn.commit()
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def processed_doc_ids(self, stage: str | None = None) -> set[str]:
        """Return document_ids that have been processed.

        If stage is given, return docs at that stage or beyond
        (clustered < curated): a curated doc is also considered clustered.
        If None, return all tracked docs.
        """
        order = {"clustered": 0, "curated": 1}
        if stage is None:
            rows = self._conn.execute(
                "SELECT document_id FROM enrichment_progress"
            ).fetchall()
        else:
            min_level = order.get(stage, 0)
            rows = self._conn.execute(
                "SELECT document_id, stage FROM enrichment_progress"
            ).fetchall()
            return {row[0] for row in rows if order.get(row[1], 0) >= min_level}
        return {row[0] for row in rows}

    def is_processed(self, document_id: str, stage: str = "clustered") -> bool:
        row = self._conn.execute(
            "SELECT stage FROM enrichment_progress WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if row is None:
            return False
        order = {"clustered": 0, "curated": 1}
        return order.get(row[0], 0) >= order.get(stage, 0)

    # --- Aux progress: marcas ortogonales NO ordenadas (gray_reviewed, etc.) --

    def mark_stage(self, document_ids: list[str], stage: str) -> None:
        """Marca docs en una stage independiente de la progresión principal."""
        now = _now()
        self._conn.executemany(
            "INSERT OR IGNORE INTO aux_progress "
            "(document_id, stage, processed_at) VALUES (?, ?, ?)",
            [(d, stage, now) for d in document_ids],
        )
        self._conn.commit()

    def stage_doc_ids(self, stage: str) -> set[str]:
        rows = self._conn.execute(
            "SELECT document_id FROM aux_progress WHERE stage = ?",
            (stage,),
        ).fetchall()
        return {row[0] for row in rows}

    # --- Curation decisions ---------------------------------------------------

    def save_curation_decisions_batch(self, decisions: list[dict[str, Any]]) -> None:
        """Persist curation decisions atomically.

        Each decision is a dict from ReporterDocumentDecision.to_dict().
        Uses INSERT OR REPLACE so re-running curation updates rather than
        duplicates.
        """
        now = _now()
        self._conn.execute("BEGIN")
        try:
            for d in decisions:
                self._conn.execute(
                    "INSERT OR REPLACE INTO curation_decisions "
                    "(decision_id, document_id, report_id, payload_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        d.get("decision_id", ""),
                        d.get("document_id", ""),
                        d.get("report_id", ""),
                        json.dumps(d, ensure_ascii=False),
                        now,
                    ),
                )
            self._conn.commit()
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def get_curation_decision(self, document_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT payload_json FROM curation_decisions WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def list_curation_decisions(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT payload_json FROM curation_decisions ORDER BY created_at DESC"
        ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def mark_curation_rejected(self, document_ids: list[str]) -> int:
        """Mark curation decisions as review_status='rejected'.

        Used for definitive policy rejections (duplicate, insufficient
        evidence): the Landing sweep deletes files whose decision is
        rejected, closing the DEC-007 lifecycle (processed+rejected →
        deleted). The content of a duplicate already lives in the main
        corpus, so deleting the redundant file loses nothing.
        Returns the number of decisions updated.
        """
        if not document_ids:
            return 0
        updated = 0
        self._conn.execute("BEGIN")
        try:
            for doc_id in document_ids:
                row = self._conn.execute(
                    "SELECT payload_json FROM curation_decisions WHERE document_id = ?",
                    (doc_id,),
                ).fetchone()
                if not row:
                    continue
                try:
                    payload = json.loads(row[0])
                except (TypeError, json.JSONDecodeError):
                    continue
                if payload.get("review_status") == "rejected":
                    continue
                payload["review_status"] = "rejected"
                self._conn.execute(
                    "UPDATE curation_decisions SET payload_json = ? WHERE document_id = ?",
                    (json.dumps(payload, ensure_ascii=False), doc_id),
                )
                updated += 1
            self._conn.commit()
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return updated

    # --- Promotion queue ------------------------------------------------------

    def mark_promotion_pending(self, document_id: str, reason: str, provenance: str,
                               source_corpus: str = "") -> None:
        """Queue a document for promotion to the main corpus.

        The promotion is not executed here — it is recorded as pending.
        A separate promotion worker (or the dashboard) executes the physical
        copy to the main corpus.

        Args:
            source_corpus: Path to the corpus the document lives in (where to
                copy from). Empty string means "unknown" — the worker will
                try to find it.
        """
        now = _now()
        self._conn.execute(
            "INSERT OR REPLACE INTO promotion_queue "
            "(document_id, reason, provenance, source_corpus, status, queued_at, promoted_at) "
            "VALUES (?, ?, ?, ?, 'pending', ?, NULL)",
            (document_id, reason, provenance, source_corpus, now),
        )
        self._conn.commit()

    def pending_promotions(self) -> list[dict[str, Any]]:
        """Return all pending promotions in the queue."""
        rows = self._conn.execute(
            "SELECT document_id, reason, provenance, source_corpus, status, queued_at, promoted_at "
            "FROM promotion_queue WHERE status = 'pending' ORDER BY queued_at"
        ).fetchall()
        return [
            {
                "document_id": row[0], "reason": row[1], "provenance": row[2],
                "source_corpus": row[3], "status": row[4], "queued_at": row[5],
                "promoted_at": row[6],
            }
            for row in rows
        ]

    def mark_promotion_done(self, document_id: str) -> None:
        """Mark a promotion as completed (physical copy done)."""
        now = _now()
        self._conn.execute(
            "UPDATE promotion_queue SET status = 'promoted', promoted_at = ? WHERE document_id = ?",
            (now, document_id),
        )
        self._conn.commit()

    def recent_promotions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the most recently completed promotions (status='promoted')."""
        rows = self._conn.execute(
            "SELECT document_id, reason, provenance, source_corpus, status, queued_at, promoted_at "
            "FROM promotion_queue WHERE status = 'promoted' ORDER BY promoted_at DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
        return [
            {
                "document_id": row[0], "reason": row[1], "provenance": row[2],
                "source_corpus": row[3], "status": row[4], "queued_at": row[5],
                "promoted_at": row[6],
            }
            for row in rows
        ]

    def is_promotion_pending(self, document_id: str) -> bool:
        """Check if a document is already in the promotion queue (pending)."""
        row = self._conn.execute(
            "SELECT status FROM promotion_queue WHERE document_id = ?", (document_id,)
        ).fetchone()
        return row is not None and row[0] == "pending"

    def promotion_queue_doc_ids(self) -> set[str]:
        """Doc_ids ya rastreados por la cola (cualquier estado).

        Un doc 'promoted' ya no necesita re-evaluación de política — sin este
        set T1 lo re-encolaba cada ciclo (INSERT OR REPLACE resetea a
        'pending' y el executor lo volvía a marcar done: churn eterno).
        """
        rows = self._conn.execute(
            "SELECT document_id FROM promotion_queue").fetchall()
        return {r[0] for r in rows}

    # --- Cluster queries ------------------------------------------------------

    def get_cluster(self, cluster_id: str) -> TopicCluster | None:
        row = self._conn.execute(
            "SELECT payload_json FROM topic_clusters WHERE cluster_id = ?", (cluster_id,)
        ).fetchone()
        return _deserialize(row[0]) if row else None

    def list_clusters(self) -> list[TopicCluster]:
        rows = self._conn.execute(
            "SELECT payload_json FROM topic_clusters ORDER BY coherence_score DESC"
        ).fetchall()
        return [_deserialize(payload) for (payload,) in rows]

    def find_by_document(self, document_id: str) -> TopicCluster | None:
        for cluster in self.list_clusters():
            if document_id in cluster.member_document_ids:
                return cluster
        return None


def _deserialize(payload: str) -> TopicCluster:
    data = json.loads(payload)
    gen = data["generation"]
    data["generation"] = GenerationProvenance(**gen)
    return TopicCluster(**data)


def build_clusters(
    documents: dict[str, str],
    embeddings: dict[str, list[float]],
    representative_chunks: dict[str, str],
    *,
    merge_threshold: float = MERGE_THRESHOLD,
    parent_threshold: float = PARENT_THRESHOLD,
    min_size: int = MIN_CLUSTER_SIZE,
) -> list[TopicCluster]:
    """Deterministic agglomerative clustering over document embeddings.

    Args:
        documents: {document_id: text} for labeling.
        embeddings: {document_id: dense vector} (from centroids' representative
                    chunks or mean-pooled chunk embeddings).
        representative_chunks: {document_id: chunk_id} for provenance.
        merge_threshold: cosine similarity above which documents merge.
        parent_threshold: cluster-centroid similarity above which a parent
                          hierarchy link is created.

    Returns:
        TopicCluster records sorted by coherence (descending). Singletons are
        excluded (they are their own topic, not a cluster).
    """
    doc_ids = [did for did in embeddings if did in documents]
    if len(doc_ids) < min_size:
        return []

    vectors = np.array([embeddings[did] for did in doc_ids], dtype=np.float64)

    # Deterministic single-pass agglomeration (stable order): each document
    # joins the cluster containing its most-similar member, or starts a new
    # cluster. Single-linkage above merge_threshold.
    clusters: list[list[int]] = []
    for i in range(len(doc_ids)):
        best_ci, best_sim = None, 0.0
        for ci, members in enumerate(clusters):
            sim = max(cosine(vectors[i], vectors[j]) for j in members)
            if sim > best_sim:
                best_sim, best_ci = sim, ci
        if best_ci is not None and best_sim >= merge_threshold:
            clusters[best_ci].append(i)
        else:
            clusters.append([i])

    now = _now()
    records: list[TopicCluster] = []
    for members in clusters:
        if len(members) < min_size:
            continue  # singleton: not a cluster
        member_ids = [doc_ids[i] for i in members]
        member_vecs = [vectors[i] for i in members]
        centroid = np.mean(member_vecs, axis=0)
        sims = [cosine(v, centroid) for v in member_vecs]
        coherence = float(np.clip(np.mean(sims), 0.0, 1.0))
        # Representative: member closest to the centroid
        rep_idx = int(np.argmax(sims))
        rep_doc = member_ids[rep_idx]
        texts = [documents[did] for did in member_ids]
        cluster_id = "topic_cluster:" + hashlib.sha256(
            ("".join(sorted(member_ids)) + now).encode()
        ).hexdigest()[:16]
        records.append(TopicCluster(
            cluster_id=cluster_id,
            label=_label_from_texts(texts),
            description=None,
            member_document_ids=member_ids,
            member_concept_ids=[],
            parent_cluster_id=None,
            coherence_score=round(coherence, 4),
            representative_chunk_id=representative_chunks.get(rep_doc, rep_doc),
            created_at=now,
            generation=GenerationProvenance(
                generator="topic-clusterer",
                generated_at=now,
                input_hash="sha256:" + hashlib.sha256(
                    json.dumps(sorted(member_ids), sort_keys=True).encode()
                ).hexdigest(),
                model_fingerprint="bge-m3-centroids",
            ),
            field_origins={
                "label": FieldOrigin.GENERATED.value,
                "member_document_ids": FieldOrigin.SOURCE.value,
                "coherence_score": FieldOrigin.GENERATED.value,
            },
        ))

    # Hierarchy: link each cluster to the most similar other cluster centroid
    if len(records) > 1 and parent_threshold < 1.0:
        centroids = [
            np.mean([vectors[doc_ids.index(d)] for d in r.member_document_ids], axis=0)
            for r in records
        ]
        for i, rec in enumerate(records):
            best_j, best_sim = None, parent_threshold
            for j in range(i):
                sim = float(cosine(centroids[i], centroids[j]))
                if sim > best_sim:
                    best_sim, best_j = sim, j
            if best_j is not None:
                # Parent candidates are restricted to earlier records. This
                # makes the hierarchy a DAG; unrestricted nearest-neighbor
                # assignment previously allowed parent cycles.
                records[i] = _with_parent(rec, records[best_j].cluster_id)

    records.sort(key=lambda r: -r.coherence_score)
    return records


def _with_parent(rec: TopicCluster, parent_id: str) -> TopicCluster:
    return TopicCluster(
        cluster_id=rec.cluster_id, label=rec.label, description=rec.description,
        member_document_ids=rec.member_document_ids,
        member_concept_ids=rec.member_concept_ids,
        parent_cluster_id=parent_id,
        coherence_score=rec.coherence_score,
        representative_chunk_id=rec.representative_chunk_id,
        created_at=rec.created_at, generation=rec.generation,
        field_origins=rec.field_origins,
    )


def cosine(a: Any, b: Any) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def backfill_topics(
    cluster_store: "TopicClusterStore",
    document_store: Any,
    lance_index: Any,
    *,
    merge_threshold: float = MERGE_THRESHOLD,
    min_size: int = MIN_CLUSTER_SIZE,
    max_docs: int = 20,
) -> dict[str, Any]:
    """Asigna tópicos a documentos sin cluster (pre-proceso de fondo).

    Determinístico, sin LLM: reutiliza los embeddings que FastPath ya
    calculó (centroide = media de los vectores de chunks del documento).
    Corre en inactividad; no toca VRAM del LLM.

    Estrategia:
      1. Documentos sin cluster (no aparecen en member_document_ids).
      2. Por cada uno: similitud contra el centroide de cada cluster
         existente → si >= merge_threshold, se asigna (mismo cluster_id,
         membresía actualizada; las referencias en episodios siguen válidas).
      3. Los que no matchean: si hay suficientes mutuamente similares,
         forman clusters nuevos (build_clusters sobre el lote restante).

    Returns: {"assigned": [...], "new_clusters": [...], "pending": int}
    """
    # 1. Documentos del corpus (con sus centroides representativos)
    doc_centroids = document_store.all_centroids()
    if not doc_centroids:
        return {"assigned": [], "new_clusters": [], "pending": 0}

    clusters = cluster_store.list_clusters()
    topicized: set[str] = set()
    for c in clusters:
        topicized.update(c.member_document_ids)

    pending = [d for d in doc_centroids if d not in topicized][:max_docs]
    if not pending:
        return {"assigned": [], "new_clusters": [], "pending": 0}

    # Centroide por documento: media de los vectores LanceDB de sus chunks
    def _doc_vector(doc_id: str) -> list[float] | None:
        chunk_ids = doc_centroids.get(doc_id) or []
        if not chunk_ids:
            return None
        vectors = lance_index.get_vectors(chunk_ids[:12])
        vecs = [vectors[cid] for cid in chunk_ids[:12] if cid in vectors]
        if not vecs:
            return None
        return list(np.mean(np.array(vecs, dtype=np.float64), axis=0))

    # Centroide de cada cluster existente (media de los vectores de miembros).
    # Fallback: si ningún miembro tiene centroide disponible (p. ej. fueron
    # purgados del corpus pero el cluster derivado sigue registrado), se usa
    # el vector del representative_chunk_id del cluster.
    def _cluster_centroid(cluster: TopicCluster) -> Any | None:
        vecs = []
        for member in cluster.member_document_ids[:12]:
            v = _doc_vector(member)
            if v is not None:
                vecs.append(v)
        if vecs:
            return np.mean(np.array(vecs, dtype=np.float64), axis=0)
        rep = cluster.representative_chunk_id
        if rep:
            rv = lance_index.get_vectors([rep])
            if rep in rv:
                return np.asarray(rv[rep], dtype=np.float64)
        return None

    def _assign_to_cluster(cluster: TopicCluster, doc_id: str) -> None:
        """Agrega el documento a la membresía del cluster (mismo cluster_id)."""
        from dataclasses import replace
        updated = replace(
            cluster,
            member_document_ids=sorted(set(cluster.member_document_ids) | {doc_id}),
        )
        cluster_store.save_cluster(updated)

    assigned: list[dict[str, str]] = []
    unmatched: list[str] = []
    cluster_centroids: dict[str, Any] = {}

    for doc_id in pending:
        vec = _doc_vector(doc_id)
        if vec is None:
            continue
        best_cid, best_sim = None, 0.0
        for c in clusters:
            if not c.member_document_ids:
                continue
            if c.cluster_id not in cluster_centroids:
                cluster_centroids[c.cluster_id] = _cluster_centroid(c)
            centroid = cluster_centroids[c.cluster_id]
            if centroid is None:
                continue
            sim = cosine(vec, centroid)
            if sim > best_sim:
                best_sim, best_cid = sim, c.cluster_id
        if best_cid is not None and best_sim >= merge_threshold:
            target = next(c for c in clusters if c.cluster_id == best_cid)
            _assign_to_cluster(target, doc_id)
            assigned.append({"document_id": doc_id, "cluster_id": best_cid})
        else:
            unmatched.append(doc_id)

    # 4. Los no asignados: intentar formar clusters nuevos entre sí
    new_cluster_ids: list[str] = []
    if len(unmatched) >= min_size:
        embeddings: dict[str, list[float]] = {}
        texts: dict[str, str] = {}
        for doc_id in unmatched:
            vec = _doc_vector(doc_id)
            if vec is not None:
                embeddings[doc_id] = vec
                doc = document_store.get_document(doc_id)
                texts[doc_id] = (doc.text if doc else "")[:500]
        if len(embeddings) >= min_size:
            new_clusters = build_clusters(texts, embeddings, {}, merge_threshold=merge_threshold, min_size=min_size)
            for nc in new_clusters:
                cluster_store.save_cluster(nc)
            new_cluster_ids = [c.cluster_id for c in new_clusters]

    return {
        "assigned": assigned,
        "new_clusters": new_cluster_ids,
        "pending": len(unmatched),
    }


__all__ = [
    "DEFAULT_CLUSTER_STORE",
    "MERGE_THRESHOLD",
    "PARENT_THRESHOLD",
    "TopicCluster",
    "TopicClusterStore",
    "backfill_topics",
    "build_clusters",
    "cosine",
]
