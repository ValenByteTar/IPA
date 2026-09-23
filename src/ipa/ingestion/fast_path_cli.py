"""Fast path pipeline CLI over a directory of artifacts.

Canonical implementation; entrypoint is a thin wrapper
(``scripts/cli/run_fast_path.py``).

Usage:
    python scripts/cli/run_fast_path.py --input data/sample/input --output outputs/experiments/E1
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

from ipa import FastPathRunner, TraceLog
from ipa.agentic import heavy_lock, tier0

ROOT = Path(__file__).resolve().parents[3]


def _resolve_main_corpus(output: Path) -> Path | None:
    """Corpus principal para señales Tier 0 (dup flag / novelty hints).

    ``IPA_MAIN_CORPUS`` lo sobreescribe; None cuando el output del run ES el
    main corpus o cuando main no existe todavía.
    """
    cand = Path(os.environ.get(
        "IPA_MAIN_CORPUS",
        str(ROOT / "outputs" / "experiments" / "E12-corpus")))
    try:
        if cand.resolve() == output.resolve():
            return None
    except OSError:
        pass
    return cand if (cand / "document_store.db").exists() else None


def _record_run_metadata(store, document_ids, *, input_dir: Path,
                         output: Path, main_store=None) -> None:
    """Señales Tier 0 post-ingesta: doc metadata + provenance + dup flag."""
    try:
        from ipa.ingestion.ingest_metadata import record_ingest_metadata
        stats = record_ingest_metadata(
            store, document_ids,
            landing_db_path=output / "landing.db",
            web_root=input_dir,
            scrape_report_dir=input_dir,
            main_store=main_store,
        )
        if any(stats.values()):
            print(f"  [tier0-meta] {stats}", flush=True)
            _mark_corpus_dirty(output)
    except Exception as exc:
        print(f"  [tier0-meta] skipped: {exc}", flush=True)


def _mark_corpus_dirty(corpus_path: Path) -> None:
    """Flag "corpus changed" para el gate de topify: un ciclo T1 sin docs
    nuevos no construye dicts ni carga embeddings de main."""
    try:
        from ipa.agentic.topic_clusters import TopicClusterStore
        cs = TopicClusterStore()
        try:
            cs.set_meta(f"dirty:{corpus_path.resolve()}", "1")
        finally:
            cs.close()
    except Exception:
        pass

# Chunks máximos por pasada del drain de embeddings. Una pasada acotada deja
# progreso visible en las stats (antes la pasada 1 recorría el corpus entero y
# el watch reportaba "0 chunks embedded" durante horas) y da un punto de cesión
# frecuente al lock de trabajos pesados (should_yield).
EMBED_PASS_CHUNKS = int(os.environ.get("IPA_EMBED_PASS_CHUNKS", "256") or 256)

# Un backlog grande activa un lote GPU exclusivo: el usuario acepta que el chat
# quede temporalmente fuera de servicio para evitar horas de CPU. BGE permanece
# en CUDA hasta que LanceDB no tenga chunks pendientes; nunca hay ventanas que
# descarguen/reinicien el embedder a mitad del lote.
EMBED_GPU_BULK_ENABLED = os.environ.get(
    "IPA_EMBED_GPU_BULK", "1").strip().lower() not in ("0", "false", "no", "off")
EMBED_GPU_MIN_BACKLOG = int(os.environ.get("IPA_EMBED_GPU_MIN_BACKLOG", "512") or 512)
EMBED_GPU_WAIT_SECONDS = float(os.environ.get("IPA_EMBED_GPU_WAIT_SECONDS", "1800") or 1800)
BULK_GPU_OWNER = "bulk_embedding"


def _parent_alive() -> bool:
    """True while the spawning process (pipeline/dashboard) still exists.

    On Windows the PID of a dead parent is not recycled into getppid(), so a
    dead parent means this watcher was orphaned and the --idle-gate sentinel
    may never arrive.
    """
    ppid = os.getppid()
    if ppid <= 1:
        return False
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, ppid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(ppid, 0)
    except OSError:
        return False
    return True


def _restore_ollama(models: list[str]) -> str | None:
    if not models:
        return None
    configured = os.environ.get("IPA_OLLAMA_MODEL", "qwen3.5:9b-q4_K_M")
    selected = configured if configured in models else (models[0] if len(models) == 1 else None)
    if selected is None:
        return "No se pudo elegir un único modelo Ollama para warmup; cargará bajo demanda."
    from ipa.providers.ollama_provider import warmup_ollama_model
    try:
        warmup_ollama_model(selected, keep_alive=os.environ.get("IPA_OLLAMA_KEEP_ALIVE", "30m"))
        return None
    except Exception as exc:
        return f"Warmup de {selected} falló; el próximo turno cargará el modelo: {exc}"


def _start_bulk_gpu(embedding, *, corpus: Path, total: int, vectorized: int,
                    pending: int, embedded_before: int = 0,
                    cancel: threading.Event | None = None,
                    wait_s: float | None = None) -> dict | None:
    """Toma la GPU por todo el backlog si se supera el umbral (512 por default).

    Publica primero el estado de mantenimiento para que dashboard/agent no
    acepten chats nuevos mientras se espera una generación en vuelo. Luego toma
    el lock exclusivo, descarga Ollama, carga BGE-M3 FP16 y mantiene ambos el
    tiempo necesario para drenar todo. None = no aplica (feature off / backlog
    chico); dict activo = la sesión quedó dueña de la GPU.
    """
    if not EMBED_GPU_BULK_ENABLED or pending < EMBED_GPU_MIN_BACKLOG:
        return None

    from ipa.agentic import embedding_maintenance as maintenance
    from ipa.providers import vram_lock
    from ipa.providers.ollama_provider import loaded_ollama_models, unload_ollama_models

    session: dict = {
        "active": False, "lock": False, "models": [], "restore_models": [],
        "started_at": time.monotonic(),
        "vectorized_before": vectorized, "embedded_before": embedded_before,
    }
    maintenance.start_state(
        corpus=str(corpus), total_chunks=total, vectorized=vectorized,
        pending=pending, mode="bulk_gpu",
    )
    deadline = time.monotonic() + (
        EMBED_GPU_WAIT_SECONDS if wait_s is None else float(wait_s))
    last_wait_update = 0.0
    try:
        while not vram_lock.acquire(BULK_GPU_OWNER):
            if cancel and cancel.is_set():
                raise InterruptedError("lote GPU cancelado mientras esperaba la VRAM")
            holder = vram_lock.holder()
            if time.monotonic() - last_wait_update >= 5.0:
                maintenance.update_state(
                    status="waiting_for_vram", phase="waiting_for_vram",
                    blocked_by=(holder or {}).get("owner"),
                    pending=pending,
                )
                last_wait_update = time.monotonic()
            if time.monotonic() >= deadline:
                raise TimeoutError("Tiempo máximo de espera por VRAM excedido")
            time.sleep(0.5)
        session["lock"] = True
        maintenance.update_state(status="unloading_llm", phase="unloading_llm")
        session["models"] = [
            str(item.get("name") or item.get("model"))
            for item in loaded_ollama_models()
            if item.get("name") or item.get("model")
        ]
        configured_model = os.environ.get("IPA_OLLAMA_MODEL", "qwen3.5:9b-q4_K_M")
        if os.environ.get("IPA_LLM_PROVIDER", "ollama").strip().lower() == "ollama":
            session["restore_models"] = [configured_model]
        elif configured_model in session["models"]:
            session["restore_models"] = [configured_model]
        elif len(session["models"]) == 1:
            session["restore_models"] = list(session["models"])
        if session["models"]:
            unload_ollama_models(timeout_s=20.0)
        maintenance.update_state(status="loading_bge", phase="loading_bge")
        if not embedding.try_move_to_gpu():
            raise RuntimeError("BGE-M3 no pudo cargar en CUDA tras liberar Ollama")
        session["active"] = True
        session["embedding_started_at"] = time.monotonic()
        maintenance.update_state(status="embedding", phase="embedding")
        print(f"  [embed] lote GPU exclusivo iniciado: {pending:,} chunks pendientes "
              f"(umbral {EMBED_GPU_MIN_BACKLOG})", flush=True)
        return session
    except Exception as exc:
        status = "cancelled" if isinstance(exc, InterruptedError) else "cpu_fallback"
        _finish_bulk_gpu(embedding, session, status=status, error=str(exc))
        if status == "cpu_fallback":
            print(f"  [embed] lote GPU no disponible; continúa en CPU: {exc}", flush=True)
        return None


def _update_bulk_gpu(session: dict, *, embedded: int, total: int,
                     vectorized_before: int, pending: int) -> None:
    if not session.get("active"):
        return
    from ipa.agentic import embedding_maintenance as maintenance
    from ipa.providers import vram_lock

    vram_lock.renew(BULK_GPU_OWNER)
    elapsed = max(0.001, time.monotonic() - session["embedding_started_at"])
    gpu_embedded = max(0, embedded - session.get("embedded_before", 0))
    rate = gpu_embedded / elapsed
    maintenance.update_state(
        status="embedding", phase="embedding", embedded=gpu_embedded,
        total_chunks=total, vectorized=vectorized_before + gpu_embedded,
        pending=pending, chunks_per_second=round(rate, 2),
        eta_seconds=(round(pending / rate) if rate > 0 else None),
    )


def _finish_bulk_gpu(embedding, session: dict | None, *, status: str,
                     error: str | None = None) -> None:
    if not session:
        return
    from ipa.agentic import embedding_maintenance as maintenance
    from ipa.providers import vram_lock

    warning = None
    try:
        if session.get("active"):
            try:
                maintenance.update_state(status="restoring_chat", phase="restoring_chat")
            except Exception:
                pass
            embedding.close()  # free BGE CUDA before restoring the chat model
        if session.get("restore_models"):
            warning = _restore_ollama(session["restore_models"])
        elif session.get("models"):
            warning = "El proveedor activo no es Ollama; el modelo original se restaurará bajo demanda."
    except Exception as exc:
        warning = f"No se pudo restaurar el modelo de chat: {exc}"
    finally:
        if session.get("lock"):
            try:
                vram_lock.release(BULK_GPU_OWNER)
            except Exception:
                pass
            session["lock"] = False
        if status not in ("cpu_fallback", "failed", "cancelled"):
            status = "completed"
        try:
            maintenance.update_state(
                status=status, phase=status, chat_blocked=False,
                error=error, warning=warning, finished_at=time.time(),
            )
        except Exception as exc:
            print(f"  [embed] no se pudo guardar estado final: {exc}", flush=True)
    if warning:
        print(f"  [embed] aviso al restaurar chat: {warning}", flush=True)
    if session.get("active"):
        print("  [embed] lote GPU terminado; chat habilitado", flush=True)


def index_lancedb(store_db: Path, lance_path: Path, batch_size: int = 64) -> dict:
    """Index chunks from document_store.db into LanceDB using BGE-M3 embeddings.

    Only embeds NEW chunks (not already in LanceDB) â€” idempotent and incremental.
    Runs after BM25 indexing so the corpus is queryable immediately.
    Returns a summary dict with counts and timing.
    """
    from ipa import DocumentStore
    from ipa.indexes.embedding_adapter import EmbeddingAdapter
    from ipa.indexes.lancedb_index import LanceDBIndex

    store = DocumentStore(store_db)
    # `batch_size` acá es el tamaño de flush hacia LanceDB, no el del forward
    # del modelo (que el adapter resuelve por device: 4 CPU / 4 GPU).
    embedding = EmbeddingAdapter(show_progress=True)
    lance = LanceDBIndex(lance_path, vector_dim=1024)
    result = _index_lancedb_incremental(store, lance, embedding, batch_size)
    embedding.close()
    lance.close()
    store.close()
    print(f"  LanceDB: {result['new_chunks']} new chunks embedded + indexed ({result['skipped']} skipped) in {result['elapsed_seconds']}s", flush=True)
    return result


def _index_lancedb_incremental(store, lance, embedding, batch_size: int = 64,
                               known_ids: set | None = None,
                               centroids: bool = True,
                               max_chunks: int | None = None) -> dict:
    """Index only new chunks into LanceDB. Reuses already-loaded embedding model.

    known_ids: optional caller-owned set of already-indexed chunk_ids. When
    provided it is used as the skip set and updated in place — avoids
    re-reading the whole LanceDB table on every pass (used by the
    background drain loop).
    centroids: recompute document centroids after indexing. Pass False for
    intermediate passes; run once with True at the end.
    max_chunks: acota la pasada (progreso incremental + punto de cesión del
    lock de trabajos pesados). None = sin límite.
    """
    import time as _time

    # Get existing chunk_ids in LanceDB to skip already-indexed chunks
    existing_ids: set[str] = known_ids if known_ids is not None else set()
    if known_ids is None and lance._table is not None:
        existing_ids = lance.chunk_ids()  # lectura proyectada, sin vectores

    total_chunks = 0
    skipped = len(existing_ids)
    start = _time.monotonic()

    batch: list = []
    BATCH = batch_size

    def flush_batch():
        nonlocal total_chunks
        if not batch:
            return
        # Representación derivada: chunks enriquecidos se embeben con
        # [Summary]/[Questions] + canónico (el texto del store queda intacto).
        from ipa.agentic.chunk_enrichment import enriched_text
        texts = [enriched_text(c.text, getattr(c, "metadata", None) or {})
                 for c in batch]
        dense, sparse = embedding.embed_texts_hybrid(texts)
        lance.add_chunks(batch, dense, sparse)
        if known_ids is not None:
            known_ids.update(c.chunk_id for c in batch)
        total_chunks += len(batch)
        batch.clear()

    for chunk in store.all_chunks():
        if chunk.chunk_id in existing_ids:
            continue
        batch.append(chunk)
        if len(batch) >= BATCH:
            flush_batch()
            if max_chunks is not None and total_chunks >= max_chunks:
                break
    flush_batch()

    elapsed = _time.monotonic() - start
    result = {"chunks_indexed": total_chunks + skipped, "new_chunks": total_chunks, "skipped": skipped, "elapsed_seconds": round(elapsed, 2)}
    # Compute document centroids (representative chunks per document)
    if centroids and (total_chunks > 0 or skipped > 0):
        try:
            from ipa.indexes.lancedb_index import _compute_centroids
            _compute_centroids(store, lance)
        except Exception as exc:
            print(f"  [centroid] computation skipped: {exc}", flush=True)
    return result


def _embed_drain_loop(store_db: Path, lance_path: Path, done: threading.Event,
                      stats: dict, batch_size: int = 64,
                      cancel: threading.Event | None = None,
                      main_corpus_path: Path | None = None) -> None:
    """Drain new chunks to LanceDB and optionally hold a GPU bulk lease.

    At 512+ pending chunks, the worker announces maintenance, waits for any
    active Ollama request, unloads the model, and keeps BGE-M3 on GPU until the
    complete backlog is embedded. LanceDB chunk_ids remain the resumable
    checkpoint; chat status is only an operational indicator.
    """
    from ipa import DocumentStore
    from ipa.agentic import embedding_maintenance
    from ipa.indexes.embedding_adapter import EmbeddingAdapter
    from ipa.indexes.lancedb_index import LanceDBIndex

    if not embedding_maintenance.claim_job(
            "embedding_drain", wait_s=EMBED_GPU_WAIT_SECONDS, cancel=cancel):
        stats.update({"indexed": 0, "idle": True,
                      "error": "Cancelado o timeout esperando el lease de indexación"})
        print(f"  [embed] {stats['error']}", flush=True)
        return

    store = embedding = lance = None
    bulk_session: dict | None = None
    bulk_attempted = False
    fatal_error: str | None = None
    cancelled = False
    stats.update({"indexed": 0, "idle": False, "error": None})
    known: set[str] = set()
    try:
        store = DocumentStore(store_db)
        embedding = EmbeddingAdapter(show_progress=False)
        lance = LanceDBIndex(lance_path, vector_dim=1024)
        # Resumable checkpoint: only IDs already in LanceDB are skipped.
        if lance.is_queryable() and lance._table is not None:
            known = lance.chunk_ids()  # lectura proyectada, sin vectores
            print(f"  [embed] reanudando: {len(known)} chunks ya vectorizados",
                  flush=True)

        total = store.count_chunks()
        pending_queue = [chunk for chunk in store.all_chunks()
                         if chunk.chunk_id not in known]
        pending = len(pending_queue)
        pending_offset = 0
        vectorized_before = total - pending
        stats.update({"total": total, "vectorized_before": vectorized_before,
                      "pending": pending})
        print(f"  [embed] pendientes={pending} / chunks={total}", flush=True)

        def _refill_pending_queue() -> None:
            nonlocal pending_queue, pending_offset
            if pending_offset >= len(pending_queue):
                # Snapshot once per drained queue, not once per 256-chunk pass.
                # Re-scanning 127k store rows for every pass made the old drain
                # quadratic in the backlog and capped the GPU at ~60 chunks/s.
                pending_queue = [chunk for chunk in store.all_chunks()
                                 if chunk.chunk_id not in known]
                pending_offset = 0

        def _run_pass(max_chunks: int | None) -> dict | None:
            nonlocal bulk_session, bulk_attempted, pending_offset
            while not cancel or not cancel.is_set():
                if not bulk_session and heavy_lock.should_yield(heavy_lock.PRIORITY_BACKGROUND):
                    stats["idle"] = False
                    if done.is_set():
                        time.sleep(1)
                    else:
                        done.wait(2)
                    continue
                with heavy_lock.heavy_phase(
                        "fast_path_embed", heavy_lock.PRIORITY_BACKGROUND) as held:
                    if not held:
                        stats["idle"] = False
                        time.sleep(1)
                        continue
                    embedding_maintenance.renew_job("embedding_drain")
                    current_total = store.count_chunks()
                    current_pending = max(0, current_total - len(known))
                    threshold_pending = (
                        pending if stats["indexed"] == 0 else current_pending)
                    if (not bulk_attempted and EMBED_GPU_BULK_ENABLED
                            and threshold_pending >= EMBED_GPU_MIN_BACKLOG):
                        bulk_attempted = True
                        current_pending = max(current_pending, threshold_pending)
                        bulk_session = _start_bulk_gpu(
                            embedding, corpus=store_db.parent,
                            total=current_total,
                            vectorized=max(0, current_total - current_pending),
                            pending=current_pending,
                            embedded_before=stats["indexed"], cancel=cancel,
                        )
                    _refill_pending_queue()
                    pass_limit = len(pending_queue) if max_chunks is None else max_chunks
                    pass_chunks = pending_queue[pending_offset:pending_offset + pass_limit]
                    if not pass_chunks:
                        return {"new_chunks": 0, "pending": 0}
                    started = time.monotonic()
                    indexed = 0
                    for start in range(0, len(pass_chunks), batch_size):
                        if cancel and cancel.is_set():
                            break
                        batch = pass_chunks[start:start + batch_size]
                        from ipa.agentic.chunk_enrichment import enriched_text
                        dense, sparse = embedding.embed_texts_hybrid(
                            [enriched_text(chunk.text,
                                           getattr(chunk, "metadata", None) or {})
                             for chunk in batch])
                        lance.add_chunks(batch, dense, sparse)
                        known.update(chunk.chunk_id for chunk in batch)
                        indexed += len(batch)
                        pending_offset += len(batch)
                    result = {
                        "new_chunks": indexed,
                        "elapsed_seconds": round(time.monotonic() - started, 2),
                        "pending": max(0, current_pending - indexed),
                    }
                return result
            return None

        while not done.is_set() and not (cancel and cancel.is_set()):
            result = _run_pass(EMBED_PASS_CHUNKS)
            if result is None:
                if cancel and cancel.is_set():
                    cancelled = True
                    break
                continue
            stats["indexed"] += result["new_chunks"]
            current_total = store.count_chunks()
            current_pending = max(0, current_total - len(known))
            stats.update({"total": current_total, "pending": current_pending,
                          "idle": result["new_chunks"] == 0})
            if result["new_chunks"] > 0:
                print(f"  [embed] +{result['new_chunks']} chunks "
                      f"(total {stats['indexed']}; pending {current_pending})",
                      flush=True)
            if bulk_session:
                _update_bulk_gpu(
                    bulk_session, embedded=stats["indexed"], total=current_total,
                    vectorized_before=bulk_session["vectorized_before"],
                    pending=current_pending)
            if result["new_chunks"] == 0:
                done.wait(10)

        cancelled = cancelled or bool(cancel and cancel.is_set())
        # Once ingestion is finished, repeat bounded passes until a full scan
        # confirms there is no tail left. A graceful cancel skips this section;
        # rows already committed to LanceDB are the resume checkpoint.
        if done.is_set() and not cancelled:
            while True:
                result = _run_pass(EMBED_PASS_CHUNKS)
                if result is None:
                    if cancel and cancel.is_set():
                        cancelled = True
                        break
                    continue
                stats["indexed"] += result["new_chunks"]
                current_total = store.count_chunks()
                current_pending = max(0, current_total - len(known))
                stats.update({"total": current_total, "pending": current_pending,
                              "idle": result["new_chunks"] == 0})
                if bulk_session:
                    _update_bulk_gpu(
                        bulk_session, embedded=stats["indexed"], total=current_total,
                        vectorized_before=bulk_session["vectorized_before"],
                        pending=current_pending)
                if result["new_chunks"] > 0:
                    print(f"  [embed] final drain +{result['new_chunks']} chunks "
                          f"(total {stats['indexed']}; pending {current_pending})",
                          flush=True)
                else:
                    break

        if cancelled:
            stats["idle"] = True
            print("  [embed] cancelado entre pasadas; progreso durable en LanceDB",
                  flush=True)
        elif done.is_set():
            stats["idle"] = True
            try:
                from ipa.indexes.lancedb_index import _compute_centroids
                _compute_centroids(store, lance)
            except Exception as exc:
                print(f"  [centroid] computation skipped: {exc}", flush=True)
            # Novelty hints (Tier 0): max coseno vs main por doc nuevo — T1
            # los usa para saltear la carga completa de embeddings históricos.
            if main_corpus_path is not None:
                try:
                    from ipa import DocumentStore as _DS
                    from ipa.indexes.lancedb_index import LanceDBIndex as _LI
                    from ipa.ingestion.ingest_metadata import compute_novelty_hints
                    _ms = _DS(Path(main_corpus_path) / "document_store.db")
                    _ml = _LI(Path(main_corpus_path) / "vector" / "lancedb")
                    try:
                        n_hints = compute_novelty_hints(store, lance, _ms, _ml)
                    finally:
                        _ml.close()
                        _ms.close()
                    if n_hints:
                        print(f"  [tier0-meta] novelty hints: {n_hints} docs",
                              flush=True)
                except Exception as exc:
                    print(f"  [tier0-meta] novelty hints skipped: {exc}", flush=True)
    except Exception as exc:
        fatal_error = str(exc)
        stats["error"] = fatal_error
        stats["idle"] = True
        print(f"  [embed] ERROR: {fatal_error}", flush=True)
    finally:
        if bulk_session and bulk_session.get("active"):
            final_status = "cancelled" if cancelled else ("failed" if fatal_error else "completed")
            _finish_bulk_gpu(embedding, bulk_session, status=final_status,
                             error=fatal_error)
        elif embedding is not None:
            embedding.close()
        if lance is not None:
            lance.close()
        if store is not None:
            store.close()
        embedding_maintenance.release_job("embedding_drain")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RES-023 fast path pipeline.")
    parser.add_argument("--input", default="Landing", help="Landing directory with artifacts (default: Landing).")
    parser.add_argument(
        "--output",
        default="outputs/experiments/E1",
        help="Output directory for databases and results.",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--query", default=None, help="Optional query to run after ingestion.")
    parser.add_argument("--trace-db", default=None, help="Enable E11 traceability, writing events to this SQLite DB.")
    parser.add_argument("--no-lancedb", action="store_true", help="Skip LanceDB vector indexing (BM25 only).")
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS", help="Keep running: re-ingest + re-index every N seconds (keeps BGE-M3 in memory).")
    parser.add_argument("--idle-exit", type=int, default=0, metavar="N", help="In watch mode: exit after N consecutive iterations with no new artifacts/chunks. 0 = run forever (default).")
    parser.add_argument("--idle-gate", default=None, metavar="PATH", help="In watch mode: only start counting idle iterations once this file exists (e.g. scraper-done sentinel).")
    args = parser.parse_args()

    # Tier 0: una sola ingesta activa contra el corpus. El lease es
    # cross-process con heartbeat — los watchers pueden sobrevivir al
    # dashboard que los lanzó, y el scheduler idle no inicia Tier 1/2
    # mientras esté vivo (la cuenta de idle arranca al liberarse).
    if not tier0.claim("fast_path"):
        owner = tier0.holder() or {}
        print(f"[tier0] ya hay una ingesta activa "
              f"(pid {owner.get('pid', '?')}) — esta instancia no aporta "
              "nada, saliendo", flush=True)
        return
    tier0.start_heartbeat("fast_path")

    input_dir = Path(args.input)
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    trace = None
    if args.trace_db:
        trace = TraceLog(args.trace_db)
        print(f"Tracing enabled: {args.trace_db}")

    runner = FastPathRunner(
        landing_db=output / "landing.db",
        store_db=output / "document_store.db",
        index_db=output / "bm25_index.db",
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        landing_root=input_dir,
        trace_log=trace,
    )

    main_corpus_path = _resolve_main_corpus(output)

    # Stage 2: LanceDB vector indexing runs CONCURRENTLY in a background
    # thread — CPU parsing/chunking overlaps GPU embedding (BGE-M3). The
    # thread drains committed chunks as the ingest produces them.
    embed_done: threading.Event | None = None
    embed_thread: threading.Thread | None = None
    embed_stats: dict | None = None
    if not args.no_lancedb:
        embed_done = threading.Event()
        embed_stats = {"indexed": 0, "idle": False}
        embed_thread = threading.Thread(
            target=_embed_drain_loop,
            args=(output / "document_store.db", output / "vector" / "lancedb",
                  embed_done, embed_stats),
            kwargs={"batch_size": 64, "main_corpus_path": main_corpus_path},
            daemon=True,
            name="lancedb-embed-drain",
        )
        embed_thread.start()
        print("LanceDB embedder: background drain started (BGE-M3)", flush=True)

    start = time.monotonic()
    results = runner.ingest_directory(input_dir)
    total_elapsed = time.monotonic() - start

    # Señales Tier 0 en la misma corrida: provenance + doc metadata + flag de
    # duplicado exacto vs main. T1 deja de re-derivarlas cada ciclo idle.
    _main_store = None
    if main_corpus_path is not None:
        try:
            from ipa import DocumentStore as _DS
            _main_store = _DS(main_corpus_path / "document_store.db")
        except Exception:
            _main_store = None
    _record_run_metadata(
        runner.store,
        [r.document_id for r in results if r.document_id],
        input_dir=input_dir, output=output, main_store=_main_store)

    report = {
        "input": str(input_dir),
        "output": str(output),
        "artifacts_ingested": len(results),
        "total_chunks": sum(r.chunks_created for r in results),
        "first_queryable": all(r.first_queryable for r in results) if results else False,
        "total_elapsed_seconds": round(total_elapsed, 4),
        "results": [
            {
                "artifact_id": r.artifact_id,
                "mime_type": r.mime_type,
                "parser_id": r.parser_id,
                "document_id": r.document_id,
                "pages": r.pages,
                "chunks_created": r.chunks_created,
                "first_queryable": r.first_queryable,
                "elapsed_seconds": round(r.elapsed_seconds, 4),
                "errors": r.errors,
            }
            for r in results
        ],
    }

    if args.query:
        hits = runner.search(args.query, limit=10)
        report["query"] = args.query
        report["search_hits"] = [
            {
                "chunk_id": h.chunk_id,
                "score": h.score,
                "retrieval_backend": h.retrieval_backend,
                "page": h.source_span.page if h.source_span else None,
            }
            for h in hits
        ]

    runner.close()

    # LanceDB indexing is already running in the background drain thread;
    # record a snapshot (final counts land when the thread is joined below).
    if embed_stats is not None:
        report["lancedb"] = {
            "mode": "background-drain",
            "chunks_indexed_so_far": embed_stats.get("indexed", 0),
        }

    if trace:
        s = trace.summary()
        report["trace_summary"] = s
        trace.close()
        print(f"Trace: {s['total_events']} events, {s['artifacts']} artifacts, {s['failed_events']} failed")

    report_path = output / "fast_path_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Ingested {len(results)} artifacts, {report['total_chunks']} chunks in {total_elapsed:.3f}s")
    print(f"first_queryable={report['first_queryable']}")
    print(f"Report: {report_path}")
    if args.query:
        hits = report.get("search_hits", [])
        print(f"Query '{args.query}': {len(hits)} hits")
        for h in hits[:5]:
            print(f"  {h['chunk_id']} score={h['score']:.4f} page={h['page']}")

    # --- Watch mode: keep re-ingesting BM25; embeddings drain in background ---
    if args.watch > 0 and not args.no_lancedb:
        import time as _time

        print(f"\nWatch mode: re-ingesting every {args.watch}s "
              f"(embeddings drain in background)", flush=True)

        iteration = 0
        idle_rounds = 0
        idle_gate = Path(args.idle_gate) if args.idle_gate else None
        while True:
            _time.sleep(args.watch)
            iteration += 1
            print(f"\n[watch] Iteration {iteration} — re-ingesting...", flush=True)

            # Re-run fast path (BM25 — idempotent, skips already-indexed)
            new_artifacts = 0
            new_chunks = 0
            runner2 = FastPathRunner(
                landing_db=output / "landing.db",
                store_db=output / "document_store.db",
                index_db=output / "bm25_index.db",
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                landing_root=input_dir,
            )
            try:
                results2 = runner2.ingest_directory(input_dir)
                new_artifacts = sum(1 for r in results2 if r.chunks_created > 0)
                new_chunks = sum(r.chunks_created for r in results2)
                print(f"  BM25: {new_artifacts} new artifacts, {new_chunks} new chunks", flush=True)
                _record_run_metadata(
                    runner2.store,
                    [r.document_id for r in results2 if r.document_id],
                    input_dir=input_dir, output=output, main_store=_main_store)
            except (PermissionError, OSError) as exc:
                # File might be being written by scraper — skip this iteration
                print(f"  BM25: skipped iteration (file busy: {exc})", flush=True)
                new_artifacts = -1  # busy file is not idle
            finally:
                runner2.close()

            embed_indexed = embed_stats.get("indexed", 0) if embed_stats else 0
            embed_idle = embed_stats.get("idle", True) if embed_stats else True
            print(f"  LanceDB: {embed_indexed} chunks embedded "
                  f"({'idle' if embed_idle else 'draining'})", flush=True)

            # Orphan exit: if the parent (pipeline/dashboard) died before the
            # idle gate was written, it will never arrive — leave watch mode
            # instead of holding the embedding lease forever.
            if idle_gate is not None and not idle_gate.exists() and not _parent_alive():
                print("  [watch] parent process gone and idle gate missing — "
                      "exiting watch mode", flush=True)
                break

            # Idle-exit: once the gate file exists (scraper done), stop the
            # watch after N consecutive iterations with no new work AND the
            # embedding drain fully caught up.
            if args.idle_exit > 0 and (idle_gate is None or idle_gate.exists()):
                if new_artifacts == 0 and embed_idle:
                    idle_rounds += 1
                    print(f"  [watch] idle {idle_rounds}/{args.idle_exit}", flush=True)
                    if idle_rounds >= args.idle_exit:
                        print(f"  [watch] {args.idle_exit} consecutive idle iterations — exiting watch mode", flush=True)
                        break
                else:
                    idle_rounds = 0

    # Shutdown: signal the embedder that ingestion is over and wait for the
    # final drain so no committed chunk is left unembedded.
    if embed_thread is not None:
        print("Waiting for LanceDB embedder final drain...", flush=True)
        embed_done.set()
        embed_thread.join()
        print(f"LanceDB embedder done: {embed_stats.get('indexed', 0)} chunks embedded", flush=True)

    if _main_store is not None:
        try:
            _main_store.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
