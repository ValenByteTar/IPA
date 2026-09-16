"""TopicNavigator: bounded multi-hop retrieval over topic clusters (Fase 3).

Vertical retrieval first (what already works). If coverage is insufficient
(same deterministic gap check as Fase 1), the planner hops through the topic
cluster hierarchy: sibling/parent clusters of the best hit's cluster provide
additional candidate documents. Bounded: max 2 hops, budgeted chunks per hop
(PAT-004).

This is a hypothesis under test (horizontalization.md): multi-hop must beat
vertical on recall without exploding latency, measured not assumed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ipa.agentic.topic_clusters import TopicCluster, TopicClusterStore

MAX_HOPS = 2
CHUNKS_PER_HOP = 5


@dataclass
class NavigationTrace:
    """Audit trail of the navigation decisions (PAT-004)."""
    vertical_hits: int = 0
    coverage_sufficient: bool = True
    hops: list[dict[str, Any]] = field(default_factory=list)
    total_chunks_considered: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "vertical_hits": self.vertical_hits,
            "coverage_sufficient": self.coverage_sufficient,
            "hops": list(self.hops),
            "total_chunks": self.total_chunks_considered,
        }


@dataclass
class NavigationResult:
    """Multi-hop retrieval result with provenance of how each hit was found."""
    hits: list[dict[str, Any]] = field(default_factory=list)
    trace: NavigationTrace = field(default_factory=NavigationTrace)

    @property
    def used_multi_hop(self) -> bool:
        return len(self.trace.hops) > 0


class TopicNavigator:
    """Bounded multi-hop retrieval over the topic cluster graph."""

    def __init__(
        self,
        cluster_store: TopicClusterStore,
        *,
        max_hops: int = MAX_HOPS,
        chunks_per_hop: int = CHUNKS_PER_HOP,
    ) -> None:
        self.clusters = cluster_store
        self.max_hops = max_hops
        self.chunks_per_hop = chunks_per_hop

    def _cluster_of(self, document_id: str) -> TopicCluster | None:
        return self.clusters.find_by_document(document_id)

    def _query_seed_clusters(self, query: str, limit: int = 2) -> list[TopicCluster]:
        """Fallback seed selection when vertical hits have no cluster.

        This keeps the production navigator usable before all corpus documents
        have cluster membership: deterministic token overlap against labels.
        It is a seed only; ranking/retrieval remains in the vertical adapter.
        """
        terms = set(re.findall(r"[a-zA-Záéíóúñü]{3,}", query.lower()))
        scored: list[tuple[int, TopicCluster]] = []
        for cluster in self.clusters.list_clusters():
            label_terms = set(re.findall(r"[a-zA-Záéíóúñü]{3,}", cluster.label.lower()))
            score = len(terms & label_terms)
            if score:
                scored.append((score, cluster))
        scored.sort(key=lambda pair: (-pair[0], pair[1].cluster_id))
        return [cluster for _, cluster in scored[:limit]]

    def _neighbor_clusters(self, cluster: TopicCluster) -> list[TopicCluster]:
        """Neighbors of a cluster: its parent, its children, and its siblings
        (clusters sharing the same parent). Deterministic order, no cycles."""
        all_clusters = self.clusters.list_clusters()
        neighbors: list[TopicCluster] = []
        seen = {cluster.cluster_id}
        # Parent
        if cluster.parent_cluster_id:
            parent = self.clusters.get_cluster(cluster.parent_cluster_id)
            if parent is not None and parent.cluster_id not in seen:
                neighbors.append(parent)
                seen.add(parent.cluster_id)
        # Children and siblings
        for other in all_clusters:
            if other.cluster_id in seen:
                continue
            is_child = other.parent_cluster_id == cluster.cluster_id
            is_sibling = (
                cluster.parent_cluster_id is not None
                and other.parent_cluster_id == cluster.parent_cluster_id
            )
            if is_child or is_sibling:
                neighbors.append(other)
                seen.add(other.cluster_id)
        return neighbors

    def navigate(
        self,
        query: str,
        vertical_search: Any,
        document_store: Any,
        *,
        min_hits: int = 3,
        limit: int = 10,
        seed_clusters: list[TopicCluster] | None = None,
    ) -> NavigationResult:
        """Vertical first; if coverage is insufficient, hop through clusters.

        Args:
            query: the user query.
            vertical_search: callable(query, limit) -> hits with document_id.
            document_store: DocumentStore for fetching chunks of neighbor docs.
            min_hits: coverage threshold that triggers multi-hop.
            seed_clusters: optional explicit entry points into the cluster
                graph (e.g. by query→cluster similarity). When None, seeds are
                derived from the vertical hits' documents.

        Returns:
            NavigationResult with fused hits (vertical first, then hops) and
            the navigation trace.
        """
        trace = NavigationTrace()
        vertical = vertical_search(query, limit=10)
        trace.vertical_hits = len(vertical)

        # Deduplicate by chunk_id, preserving order (vertical first)
        hits: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in vertical:
            cid = hit.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                hits.append(hit)
        trace.total_chunks_considered = len(hits)

        # Coverage = distinct documents, not raw chunk count: 10 chunks from
        # 2 documents is narrow coverage even when the count looks fine.
        distinct_docs = {h.get("document_id") for h in hits}
        if len(hits) >= min_hits and len(distinct_docs := {h.get("document_id") for h in hits}) >= min_hits:
            trace.coverage_sufficient = True
            return NavigationResult(hits=hits, trace=trace)

        # Coverage insufficient → multi-hop through the cluster graph
        trace.coverage_sufficient = False
        visited_clusters: set[str] = set()
        hop = 0
        # Entry points: explicit seeds, or clusters of the vertical hits' docs
        if seed_clusters:
            frontier = list(seed_clusters)
        else:
            frontier = []
            for hit in hits:
                cluster = self.clusters.find_by_document(hit.get("document_id", ""))
                if cluster is not None and cluster.cluster_id not in {c.cluster_id for c in frontier}:
                    frontier.append(cluster)
            # If vertical documents are not clustered, use deterministic label
            # overlap as a bounded entry point instead of silently doing zero
            # hops despite an available cluster graph.
            if not frontier:
                frontier = self._query_seed_clusters(query, limit=2)
        while frontier and hop < self.max_hops:
            hop += 1
            next_clusters: list[TopicCluster] = []
            for cluster in frontier:
                if cluster.cluster_id in visited_clusters:
                    continue
                visited_clusters.add(cluster.cluster_id)
                neighbors = self._neighbor_clusters(cluster)
                for neighbor in neighbors:
                    if neighbor.cluster_id in visited_clusters:
                        continue
                    visited_clusters.add(neighbor.cluster_id)
                    hop_chunks = 0
                    for doc_id in neighbor.member_document_ids:
                        if hop_chunks >= self.chunks_per_hop:
                            break
                        for chunk in document_store.get_chunks(doc_id):
                            if hop_chunks >= self.chunks_per_hop:
                                break
                            if chunk.chunk_id in seen:
                                continue
                            hits.append({
                                "chunk_id": chunk.chunk_id,
                                "document_id": doc_id,
                                "score": 0.0,  # no query-time score for hop hits
                                "retrieval_backend": f"multi_hop_hop{hop}",
                                "text_preview": (chunk.text[:200] + "...") if len(chunk.text) > 200 else chunk.text,
                                "via_cluster": neighbor.cluster_id,
                                "via_label": neighbor.label,
                            })
                            seen.add(chunk.chunk_id)
                            hop_chunks += 1
                            trace.total_chunks_considered += 1
                    next_clusters.append(neighbor)
                trace.hops.append({
                    "hop": hop,
                    "from_cluster": cluster.cluster_id,
                    "from_label": cluster.label,
                    "neighbors_explored": [n.cluster_id for n in neighbors],
                })
            frontier = next_clusters

        return NavigationResult(hits=hits, trace=trace)


__all__ = [
    "CHUNKS_PER_HOP",
    "MAX_HOPS",
    "NavigationResult",
    "NavigationTrace",
    "TopicNavigator",
]
