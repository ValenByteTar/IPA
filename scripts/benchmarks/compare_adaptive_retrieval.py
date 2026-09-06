"""Compare retrieval on original vs adaptively re-chunked activated documents.

Uses the same document-level queries and ground truth.  Chunk IDs may change
after adaptive re-chunking, so document recall is the primary metric.
"""
from __future__ import annotations
import argparse, json, re, shutil, sqlite3, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ipa.contracts import DocumentChunk, SourceSpan
from ipa.indexes.tantivy_index import TantivyIndex
from ipa.agentic.retrieval_eval import document_recall_at_k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True)
    p.add_argument("--adaptive-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-queries", type=int, default=200)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    report = json.loads(Path(args.adaptive_report).read_text(encoding="utf-8"))
    active = [r for r in report["results"] if r["reason"] != "no_action"]
    active_ids = {r["document_id"] for r in active}
    conn = sqlite3.connect(args.store)
    docs = {r[0]: (r[1], r[5]) for r in conn.execute("SELECT document_id, artifact_id, parser_id, mime_type, pages, text FROM documents WHERE tombstoned=0") if r[0] in active_ids}
    original = {}
    for r in conn.execute("SELECT chunk_id, document_id, content_hash, text, metadata_json, span_json FROM chunks WHERE tombstoned=0 ORDER BY document_id, rowid"):
        if r[1] in active_ids:
            original.setdefault(r[1], []).append(DocumentChunk(r[0], r[1], r[2], r[3], json.loads(r[4] or "{}"), None))
    conn.close()
    adaptive = {}
    for item in active:
        adaptive[item["document_id"]] = [DocumentChunk(c["chunk_id"], c["document_id"], c["content_hash"], c["text"], c["metadata"], None) for c in item["chunks"]]

    # Queries are document-level: distinctive terms from each activated document.
    queries = []
    for doc_id, (_, text) in sorted(docs.items())[:args.n_queries]:
        terms = re.findall(r"[A-Za-z]{4,}", text.lower())
        freq = {}
        for t in terms: freq[t] = freq.get(t, 0) + 1
        q = " ".join(t for t, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:6])
        if q: queries.append((q, doc_id))

    def build(name, chunks_by_doc):
        path = out / name
        if path.exists(): shutil.rmtree(path)
        idx = TantivyIndex(path); all_chunks=[]
        for cs in chunks_by_doc.values(): all_chunks.extend(cs)
        idx.add_chunks(all_chunks); return idx
    base = build("tantivy_original", original)
    adapt = build("tantivy_adaptive", adaptive)
    metrics = {}
    for name, idx in (("original", base), ("adaptive", adapt)):
        vals = []
        for q, doc_id in queries:
            hits = idx.search(q, limit=20)
            docs_ranked = [h.source_span.artifact_id if False else h.chunk_id for h in hits]
            # Resolve hit document IDs from local chunk maps.
            lookup = {c.chunk_id: c.document_id for cs in (original if name == "original" else adaptive).values() for c in cs}
            vals.append(document_recall_at_k([lookup.get(cid, "") for cid in docs_ranked], doc_id, 10))
        metrics[name] = {"n_queries": len(vals), "document_recall@10": round(sum(vals)/len(vals), 4) if vals else 0.0, "chunks": idx.count()}
    base.close(); adapt.close()
    result = {"activated_documents": len(active), "results": metrics}
    path = out / "adaptive_retrieval_comparison.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)); print(f"Report: {path}")

if __name__ == "__main__": main()
