"""Drain de embeddings de LanceDB standalone — pausable y resumible.

Corre el mismo drain que el fast path (``_embed_drain_loop``) sin arrastrar el
pipeline: sirve para trabajar el backlog vectorial de un corpus aparte, con la
posibilidad de cortarlo y retomarlo.

- **Pausar**: matar el proceso (Ctrl+C o ``taskkill``). El drain escribe por
  batch; nada queda a medias.
- **Reanudar**: volver a correrlo. Precarga los ``chunk_id`` ya vectorizados,
  así que no re-embebe lo hecho (PM-004).
- **Estado**: ``--status`` reporta total / vectorizados / pendientes sin
  cargar el modelo.

Uso:
    python scripts/operations/run_embed_drain.py --corpus outputs/reporter/quality-check/<out>/corpus
    python scripts/operations/run_embed_drain.py --corpus <dir> --status
    python scripts/operations/run_embed_drain.py --corpus <dir> --max-seconds 3600

Config (env): IPA_EMBED_GPU_MIN_BACKLOG (512 chunks),
IPA_EMBED_GPU_BULK=0 desactiva el lease bulk pero no fija CPU; usar
IPA_EMBED_DEVICE=cpu para pinnear el adapter. Batches CPU/GPU default 4;
IPA_EMBED_PASS_CHUNKS (256). El lote GPU bloquea el chat hasta finalizar.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _status(corpus: Path) -> int:
    """Reporta cuántos chunks del store están vectorizados."""
    import sqlite3

    from ipa.indexes.lancedb_index import LanceDBIndex

    store_db = corpus / "document_store.db"
    if not store_db.exists():
        print(f"no hay document_store.db en {corpus}")
        return 1
    conn = sqlite3.connect(f"file:{store_db}?mode=ro", uri=True, timeout=30)
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE tombstoned=0").fetchone()[0]
    finally:
        conn.close()

    lance = LanceDBIndex(corpus / "vector" / "lancedb", vector_dim=1024)
    try:
        vectorized = lance._table.count_rows() if (
            lance.is_queryable() and lance._table is not None) else 0
    finally:
        lance.close()

    pending = max(0, total - vectorized)
    from ipa.agentic.embedding_maintenance import read_state
    maintenance = read_state()
    print(f"corpus: {corpus}")
    active_statuses = {"preparing", "waiting_for_vram", "unloading_llm",
                       "loading_bge", "embedding", "restoring_chat"}
    active_corpus = Path(str(maintenance.get("corpus") or "")).resolve()
    active_here = (maintenance.get("status") in active_statuses
                   and active_corpus == corpus.resolve())
    if active_here:
        print(f"  lote GPU           : {maintenance.get('status')} · "
              f"{maintenance.get('embedded', 0)}/{maintenance.get('pending_initial', 0)}")
    elif maintenance.get("status") in active_statuses:
        print(f"  otro lote GPU activo en {active_corpus}")
    print(f"  chunks en el store : {total}")
    print(f"  vectorizados       : {vectorized}")
    print(f"  pendientes         : {pending}")
    if pending:
        print(f"  ETA CPU (~2.9 chunks/s): {pending / 2.9 / 3600:.1f} h")
        from ipa.agentic.embedding_maintenance import GPU_MIN_BACKLOG
        gpu_enabled = os.environ.get("IPA_EMBED_GPU_BULK", "1").strip().lower() not in (
            "0", "false", "no", "off")
        if active_here and maintenance.get("chunks_per_second", 0) > 0:
            rate = float(maintenance["chunks_per_second"])
            print(f"  ETA GPU observada (~{rate:.1f} chunks/s): "
                  f"{pending / rate / 60:.1f} min")
        elif gpu_enabled and pending >= GPU_MIN_BACKLOG:
            eta_gpu = pending / 129 / 60 + 40 / 60
            print(f"  Umbral GPU: alcanzado ({GPU_MIN_BACKLOG}); lote exclusivo, chat pausado")
            print(f"  ETA GPU teórica (~129 chunks/s más transición): {eta_gpu:.1f} min")
        else:
            reason = "forzado a CPU" if not gpu_enabled else "por debajo del umbral"
            print(f"  Lote GPU no aplica ({reason}; umbral {GPU_MIN_BACKLOG})")
    return 0


def _spawn_background(args: argparse.Namespace, corpus: Path) -> int:
    """Relaunch under pythonw so Windows never opens a console for the drain."""
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.exists():
        print(f"pythonw.exe no encontrado junto a {sys.executable}; ejecutá sin --background")
        return 1
    log = ROOT / "outputs" / "web_dashboard" / "logs" / "embed_drain.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(pythonw), "-u", str(Path(__file__).resolve()),
               "--corpus", str(corpus)]
    if args.max_seconds:
        command.extend(("--max-seconds", str(args.max_seconds)))
    if args.cpu_only:
        command.append("--cpu-only")
    command.extend(("--batch-size", str(args.batch_size)))
    flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    with log.open("ab") as output:
        process = subprocess.Popen(
            command, cwd=str(ROOT), stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT,
            creationflags=flags, close_fds=True,
        )
    print(f"drain sin consola iniciado (PID {process.pid}); log: {log}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="LanceDB embed drain (pausable/resumible)")
    parser.add_argument("--corpus", required=True,
                        help="Dir del corpus (con document_store.db y vector/lancedb)")
    parser.add_argument("--status", action="store_true",
                        help="Solo reportar estado (no carga el modelo)")
    parser.add_argument("--background", action="store_true",
                        help="Relaunch with pythonw, no console window; progress goes to embed_drain.log")
    parser.add_argument("--max-seconds", type=float, default=0,
                        help="Pausa de forma reanudable después de N segundos (0 = hasta terminar)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Desactiva el lease GPU bulk; para pinnear CPU añadir IPA_EMBED_DEVICE=cpu")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Chunks por flush hacia LanceDB (el forward usa el batch por device)")
    args = parser.parse_args()

    corpus = Path(args.corpus).resolve()
    if args.status:
        sys.exit(_status(corpus))
    if args.background:
        sys.exit(_spawn_background(args, corpus))

    store_db = corpus / "document_store.db"
    if not store_db.exists():
        print(f"no hay document_store.db en {corpus}")
        sys.exit(1)

    if args.cpu_only:
        os.environ["IPA_EMBED_GPU_BULK"] = "0"
    from ipa.ingestion.fast_path_cli import _embed_drain_loop

    done = threading.Event()
    cancel = threading.Event()
    stats: dict = {"indexed": 0, "idle": False}
    worker = threading.Thread(
        target=_embed_drain_loop,
        args=(store_db, corpus / "vector" / "lancedb", done, stats),
        kwargs={"batch_size": args.batch_size, "cancel": cancel},
        daemon=True, name="lancedb-embed-drain",
    )
    t0 = time.monotonic()
    worker.start()
    print(f"drain iniciado sobre {corpus} (Ctrl+C para pausar; reanudable)", flush=True)
    try:
        while worker.is_alive():
            time.sleep(10)
            elapsed = time.monotonic() - t0
            rate = stats["indexed"] / elapsed if elapsed else 0.0
            print(f"  +{stats['indexed']} chunks en {elapsed:.0f}s "
                  f"= {rate:.2f} chunks/s", flush=True)
            if stats["idle"]:
                break
            if args.max_seconds and elapsed >= args.max_seconds:
                print("  tope de tiempo alcanzado — pausa reanudable", flush=True)
                cancel.set()
                done.set()
                break
    except KeyboardInterrupt:
        print("  interrumpido — pausa reanudable", flush=True)
        cancel.set()
    finally:
        done.set()
        worker.join()
    elapsed = time.monotonic() - t0
    print(f"total: {stats['indexed']} chunks en {elapsed:.0f}s "
          f"({stats['indexed'] / elapsed if elapsed else 0:.2f} chunks/s)")


if __name__ == "__main__":
    main()
