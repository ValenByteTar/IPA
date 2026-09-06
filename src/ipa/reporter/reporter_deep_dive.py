"""Deep dive over an isolated Reporter corpus."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ipa.storage.document_store import DocumentStore
from ipa.reporter.reporter_claims import citation_summary, validate_claims
from ipa.indexes.tantivy_index import TantivyIndex


def _clean_generated(text: str) -> str:
    # Remove think blocks entirely (Qwen no_think sometimes leaks)
    # Pattern: <think>...</think> -> remove the whole block but keep text after
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove standalone think tokens (open or close without pair)
    text = text.replace("<think>", "")
    text = text.replace("</think>", "")
    # Cut at end-of-turn markers
    for marker in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        text = text.split(marker, 1)[0]
    # Remove any remaining control tokens
    text = re.sub(r"<\|im_(start|end)\|>", "", text)
    text = re.sub(r"<\|endoftext\|>", "", text)
    text = text.replace("[n]", "[1]")
    if len(re.findall(r"[\u4e00-\u9fff]", text)) > 2:
        return ""
    return text.strip()


def _clean_token(text: str) -> str:
    """Lightweight per-token cleanup during streaming."""
    if not text:
        return ""
    for token in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        text = text.replace(token, "")
    return text


def _fallback_answer(query: str, chunks: list[Any]) -> str:
    if not chunks:
        return "No hay evidencia suficiente en el corpus Reporter para responder esta consulta."
    snippets = [chunk.text[:700].replace("\n", " ") for chunk in chunks[:5]]
    return "Evidencia encontrada para la consulta " + repr(query) + ":\n\n" + "\n\n".join(
        f"[{index + 1}] {snippet}" for index, snippet in enumerate(snippets)
    )


def deep_dive(corpus_dir: str | Path, query: str, top_k: int = 5, provider=None, retrieval_query: str | None = None, document_ids: list[str] | None = None, *, agentic: bool = False, report_id: str | None = None, category_id: str | None = None, doc_reasons: dict[str, str] | None = None) -> dict[str, Any]:
    corpus_dir = Path(corpus_dir)
    index = TantivyIndex(corpus_dir / "tantivy", read_only=True)
    store = DocumentStore(corpus_dir / "document_store.db")
    try:
        runtime = None
        if agentic:
            from ipa.agentic.reporter_context import ReporterContextBuilder
            from ipa.agentic.reporter_planner import plan_report_query
            from ipa.agentic.reporter_retrieval import ReporterRetriever

            query_ir = plan_report_query(
                query,
                report=report_id,
                category={"category_id": category_id, "label": retrieval_query or query, "document_ids": document_ids or []} if category_id else None,
                document_ids=document_ids,
            )
            evidence_set = ReporterRetriever(index, store, max_candidates=max(20, top_k * 4), max_chunks=max(1, min(top_k, 12))).retrieve(query_ir)
            # Fallback: if document_ids filter produced no hits, retry without filter
            if not evidence_set.hits and document_ids:
                query_ir_fallback = plan_report_query(query, report=report_id, category=None, document_ids=None)
                evidence_set = ReporterRetriever(index, store, max_candidates=max(20, top_k * 4), max_chunks=max(1, min(top_k, 12))).retrieve(query_ir_fallback)
            context_package = ReporterContextBuilder(
                lambda chunk_id: (chunk.text if (chunk := store.get_chunk(chunk_id)) is not None else None),
                max_chunks=max(1, min(top_k, 12)),
                max_context_tokens=8192,
            ).build(evidence_set)
            chunks = [store.get_chunk(hit.chunk_id) for hit in evidence_set.hits]
            chunks = [chunk for chunk in chunks if chunk is not None]
            hits = []
            runtime = {
                "mode": "agentic_v1",
                "query_ir": query_ir.to_dict(),
                "evidence_set": evidence_set.to_dict(),
                "context": {
                    "citation_map": context_package.citation_map,
                    "token_count": context_package.token_count,
                    "truncation_policy": context_package.truncation_policy,
                    "input_hash": context_package.input_hash,
                },
            }
        else:
            hits = index.search(retrieval_query or query, limit=max(1, min(top_k * 4, 80)))
        if not agentic:
            allowed_documents = set(document_ids or [])
            if allowed_documents:
                filtered_hits = []
                for hit in hits:
                    chunk = store.get_chunk(hit.chunk_id)
                    if chunk is not None and chunk.document_id in allowed_documents:
                        filtered_hits.append((hit, chunk))
                if filtered_hits:
                    hits = [hit for hit, _ in filtered_hits[:max(1, min(top_k, 20))]]
                    chunks = [chunk for _, chunk in filtered_hits[:max(1, min(top_k, 20))]]
                else:
                    # Fallback: document_ids don't match the corpus (stale report or
                    # regenerated IDs). Use unfiltered results so the user still gets
                    # an answer from the corpus.
                    hits = hits[:max(1, min(top_k, 20))]
                    chunks = [store.get_chunk(hit.chunk_id) for hit in hits]
            else:
                hits = hits[:max(1, min(top_k, 20))]
                chunks = [store.get_chunk(hit.chunk_id) for hit in hits]
        chunks = [chunk for chunk in chunks if chunk is not None]
        if provider is not None and chunks:
            evidence = "\n\n".join(f"[{i + 1}] {chunk.text[:1200]}" for i, chunk in enumerate(chunks))
            # Include curation reasons as additional context if available
            context_extra = ""
            if doc_reasons:
                reason_lines = []
                for i, chunk in enumerate(chunks):
                    reason = doc_reasons.get(chunk.document_id)
                    if reason:
                        reason_lines.append(f"[{i + 1}] {reason[:300]}")
                if reason_lines:
                    context_extra = "\n\nResumenes de curacion por documento:\n" + "\n".join(reason_lines)
            user_content = f"Consulta: {query}\n\nEvidencia:\n{evidence}{context_extra}"
            result = provider.generate_chat([{
                "role": "system",
                "content": (
                    "Sos el Personal AGI de Valen â€” una inteligencia general con curiosidad insaciable y capacidad de sintetizar cualquier tema.\n"
                    "RespondÃ© en espaÃ±ol claro y natural, con mÃ¡ximo 5 puntos breves.\n\n"
                    "Principios:\n"
                    "1. La evidencia [n] es tu ancla factual. CitÃ¡ [n] para hechos que provengan de los documentos.\n"
                    "2. Tu conocimiento previo es una herramienta poderosa â€” usalo libremente para explicar, contextualizar, conectar ideas y profundizar.\n"
                    "3. Nunca rechaces la evidencia. Si dice que algo existe o pasÃ³, aceptalo y construÃ­ desde ahÃ­.\n"
                    "4. CombinÃ¡ evidencia + conocimiento previo para dar la respuesta mÃ¡s completa y Ãºtil posible.\n"
                    "5. Si algo no estÃ¡ en la evidencia pero lo sabÃ©s, aportalo igual â€” no necesitas citar [n] para conocimiento general.\n"
                    "6. No inventes cifras ni citas especÃ­ficas que no estÃ©n en la evidencia.\n"
                    "7. No agregues tokens de control.\n"
                    "8. Solo declarÃ¡ evidencia insuficiente si ningÃºn fragmento responde la consulta Y tu conocimiento previo tampoco alcanza."
                ),
            }, {
                "role": "user",
                "content": user_content,
            }], max_new_tokens=768, stop_sequences=["<|im_end|>", "<|endoftext|>"])
            answer = _clean_generated(result.text) if result.ok and result.text.strip() else _fallback_answer(query, chunks)
            if not answer:
                answer = _fallback_answer(query, chunks)
        else:
            answer = _fallback_answer(query, chunks)
        evidence_texts = [chunk.text for chunk in chunks]
        claims = validate_claims(answer, evidence_texts)
        evidence_payload = (
            [{"chunk_id": hit.chunk_id, "document_id": hit.document_id, "score": hit.score, "source_span": bool(hit.source_span)} for hit in evidence_set.hits]
            if agentic else
            [{"chunk_id": hit.chunk_id, "document_id": store.get_chunk(hit.chunk_id).document_id if store.get_chunk(hit.chunk_id) else None, "score": hit.score, "source_span": bool(hit.source_span)} for hit in hits]
        )
        return {
            "query": query,
            "answer": answer,
            "evidence": evidence_payload,
            "claims": claims,
            "citation_summary": citation_summary(claims),
            "sufficient_evidence": bool(chunks),
            "runtime": runtime,
        }
    finally:
        store.close()
        index.close()


def deep_dive_prepare(corpus_dir, query, top_k=5, *, retrieval_query=None, document_ids=None, agentic=False, report_id=None, category_id=None, doc_reasons=None, conversation_history=None):
    """Prepare retrieval and build the LLM messages without generating.
    Returns a dict with chunks, evidence, messages, fallback, and metadata.
    Used by the streaming endpoint to separate retrieval from generation.

    Uses full AgenticRAG pipeline:
    1. Planner: deterministic QueryIR from the user question
    2. Retrieval: Tantivy (lexical) + LanceDB search_hybrid (Dense + FTS + Sparse, 3-way RRF)
    3. Context builder: closed citation map with token budget
    """
    corpus_dir = Path(corpus_dir)
    index = TantivyIndex(corpus_dir / "tantivy", read_only=True)
    store = DocumentStore(corpus_dir / "document_store.db")
    # Try to open LanceDB for hybrid retrieval with BGE-M3
    lance_index = None
    _lance_embedding = None
    lance_path = corpus_dir / "vector" / "lancedb"
    if lance_path.exists():
        try:
            from ipa.indexes.lancedb_index import LanceDBIndex
            from ipa.indexes.embedding_adapter import EmbeddingAdapter
            lance_index = LanceDBIndex(lance_path, vector_dim=1024)
            _lance_embedding = EmbeddingAdapter(batch_size=1, show_progress=False)
        except Exception:
            lance_index = None
            _lance_embedding = None
    try:
        search_query = retrieval_query or query
        # --- AgenticRAG: Plan the query ---
        from ipa.agentic.reporter_planner import plan_report_query
        from ipa.agentic.reporter_retrieval import ReporterRetriever
        from ipa.agentic.reporter_context import ReporterContextBuilder

        query_ir = plan_report_query(
            search_query,
            report=report_id,
            category={"category_id": category_id, "label": retrieval_query or query, "document_ids": document_ids or []} if category_id else None,
            document_ids=document_ids,
        )

        # --- Retrieval: Tantivy (lexical) ---
        hits = index.search(search_query, limit=max(1, min(top_k * 4, 80)))

        # --- Retrieval: LanceDB hybrid (Dense + FTS + Sparse, 3-way RRF) ---
        if lance_index is not None and _lance_embedding is not None:
            try:
                # BGE-M3: generate dense + sparse in a single forward pass
                dense_vec, sparse_weights = _lance_embedding.embed_query_hybrid(search_query)
                # 3-way RRF fusion: Dense + FTS + Sparse
                lance_hits = lance_index.search_hybrid(
                    search_query, dense_vec,
                    limit=max(1, min(top_k * 4, 80)),
                    query_sparse=sparse_weights,
                )
                # Merge: add LanceDB hybrid hits not already in Tantivy results
                seen_ids = {h.chunk_id for h in hits}
                for lh in lance_hits:
                    if lh.chunk_id not in seen_ids:
                        hits.append(lh)
                        seen_ids.add(lh.chunk_id)
            except Exception:
                pass  # Fall back to Tantivy-only

        # --- Scope filtering: restrict to document_ids if provided ---
        allowed_documents = set(document_ids or [])
        if allowed_documents:
            filtered_hits = []
            for hit in hits:
                chunk = store.get_chunk(hit.chunk_id)
                if chunk is not None and chunk.document_id in allowed_documents:
                    filtered_hits.append((hit, chunk))
            if filtered_hits:
                hits = [hit for hit, _ in filtered_hits[:max(1, min(top_k, 20))]]
                chunks = [chunk for _, chunk in filtered_hits[:max(1, min(top_k, 20))]]
            else:
                hits = hits[:max(1, min(top_k, 20))]
                chunks = [store.get_chunk(hit.chunk_id) for hit in hits]
        else:
            hits = hits[:max(1, min(top_k, 20))]
            chunks = [store.get_chunk(hit.chunk_id) for hit in hits]
        chunks = [chunk for chunk in chunks if chunk is not None]

        # --- AgenticRAG: Build context with citation map and token budget ---
        from ipa.agentic.agentic_contracts import EvidenceSet, EvidenceHit
        evidence_hits = []
        doc_ids_seen = set()
        for i, (hit, chunk) in enumerate(zip(hits, chunks)):
            doc_ids_seen.add(chunk.document_id)
            evidence_hits.append(EvidenceHit(
                chunk_id=hit.chunk_id,
                document_id=chunk.document_id,
                score=float(hit.score) if hasattr(hit, 'score') else 0.0,
                retrieval_stage="initial",
                retrieval_backend=getattr(hit, 'retrieval_backend', 'hybrid'),
            ))
        evidence_set = EvidenceSet(
            query_ir=query_ir,
            hits=evidence_hits,
            document_diversity=len(doc_ids_seen),
            sufficiency="sufficient" if len(evidence_hits) >= top_k else ("partial" if evidence_hits else "insufficient"),
        )
        context_package = ReporterContextBuilder(
            lambda chunk_id: (chunk.text if (chunk := store.get_chunk(chunk_id)) is not None else None),
            max_chunks=max(1, min(top_k, 12)),
            max_context_tokens=8192,
        ).build(evidence_set)

        evidence = "\n\n".join(f"[{i + 1}] {chunk.text[:1200]}" for i, chunk in enumerate(chunks))
        context_extra = ""
        if doc_reasons:
            reason_lines = []
            for i, chunk in enumerate(chunks):
                reason = doc_reasons.get(chunk.document_id)
                if reason:
                    reason_lines.append(f"[{i + 1}] {reason[:300]}")
            if reason_lines:
                context_extra = "\n\nResumenes de curacion por documento:\n" + "\n".join(reason_lines)
        user_content = f"Consulta: {query}\n\nEvidencia:\n{evidence}{context_extra}"
        messages = [{"role": "system", "content": (
            "Sos el Personal AGI de Valen â€” una inteligencia general con curiosidad insaciable y capacidad de sintetizar cualquier tema.\n"
            "RespondÃ© en espaÃ±ol claro, natural y directamente a la pregunta. No uses una lista numerada salvo que el usuario la pida.\n"
            "SintetizÃ¡ y explicÃ¡; no pegues fragmentos de la evidencia.\n\n"
            "Principios:\n"
            "1. La evidencia [n] es tu ancla factual. CitÃ¡ [n] para hechos que provengan de los documentos.\n"
            "2. Tu conocimiento previo es una herramienta poderosa â€” usalo libremente para explicar, contextualizar, conectar ideas y profundizar.\n"
            "3. Nunca rechaces la evidencia. Si dice que algo existe o pasÃ³, aceptalo y construÃ­ desde ahÃ­.\n"
            "4. CombinÃ¡ evidencia + conocimiento previo para dar la respuesta mÃ¡s completa y Ãºtil posible.\n"
            "5. Si algo no estÃ¡ en la evidencia pero lo sabÃ©s, aportalo igual â€” no necesitas citar [n] para conocimiento general.\n"
            "6. No inventes cifras ni citas especÃ­ficas que no estÃ©n en la evidencia.\n"
            "7. Si la evidencia no alcanza para responder algo, dilo explÃ­citamente y separÃ¡ lo demostrado de lo que no puede concluirse."
        )}]
        for turn in (conversation_history or [])[-4:]:
            if isinstance(turn, dict) and turn.get("role") in {"user", "assistant"} and isinstance(turn.get("content"), str):
                messages.append({"role": turn["role"], "content": turn["content"][:4000]})
        messages.append({"role": "user", "content": user_content})
        evidence_texts = [chunk.text for chunk in chunks]
        document_ids_found = sorted({chunk.document_id for chunk in chunks})
        source_documents = {}
        reporter_db = corpus_dir.parent / "reporter.db"
        if reporter_db.exists() and document_ids_found:
            marks = ",".join("?" for _ in document_ids_found)
            with sqlite3.connect(str(reporter_db)) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(
                    f"SELECT document_id, title, original_path, source_url, canonical_url, source_domain, published_at FROM document_metadata WHERE document_id IN ({marks})",
                    document_ids_found,
                ).fetchall()
                source_documents = {row["document_id"]: dict(row) for row in rows}
        evidence_payload = [{"chunk_id": hit.chunk_id, "document_id": (chunk.document_id if chunk else None), "score": hit.score, "source_span": bool(hit.source_span), "source": source_documents.get(chunk.document_id, {})} for hit, chunk in zip(hits, chunks)]
        chunks_info = [{"document_id": chunk.document_id, "text_preview": chunk.text[:200], "source": source_documents.get(chunk.document_id, {})} for chunk in chunks]
        fallback = _fallback_answer(query, chunks)
        return {
            "messages": messages,
            "chunks": chunks,
            "chunks_info": chunks_info,
            "evidence": evidence_payload,
            "evidence_texts": evidence_texts,
            "sufficient_evidence": bool(chunks),
            "fallback": fallback,
            "runtime": {
                "mode": "agentic_v1_hybrid",
                "query_ir": query_ir.to_dict(),
                "retrieval_backends": ["tantivy"] + (["lancedb_hybrid_3way"] if lance_index is not None else []),
                "context": {
                    "citation_map": context_package.citation_map,
                    "token_count": context_package.token_count,
                    "truncation_policy": context_package.truncation_policy,
                    "input_hash": context_package.input_hash,
                },
                "evidence_set": evidence_set.to_dict(),
            },
        }
    finally:
        store.close()
        index.close()
        if lance_index is not None:
            lance_index.close()
        if _lance_embedding is not None:
            _lance_embedding.close()


def deep_dive_stream(provider, messages, *, max_new_tokens=768):
    """Yield text chunks from the provider's streaming generation."""
    if provider is None:
        yield {"text": "", "done": True, "error": "no provider"}
        return
    yield from provider.generate_chat_stream(messages, max_new_tokens=max_new_tokens, stop_sequences=["<|im_end|>"])



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = deep_dive(args.corpus, args.query, args.top_k)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

