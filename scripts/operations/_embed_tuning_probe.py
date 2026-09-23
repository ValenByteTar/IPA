"""Sonda temporal: throughput de BGE-M3 (hybrid) según device / batch.

Medición operativa para calibrar la config del drain y del idle (PM-004), no un
benchmark de EKS (no escribe report). Textos del tamaño del corpus real
(~512 chars por chunk).

Uso:
    python scripts/operations/_embed_tuning_probe.py --device cpu
    python scripts/operations/_embed_tuning_probe.py --device cuda --batches 16,64,128 --hold-vram-lock

En GPU hay que liberar VRAM antes (el 9B del chat la ocupa):
    ollama stop qwen3.5:9b-q4_K_M
`--hold-vram-lock` toma outputs/agent/vram.lock durante la corrida para que
ExL3/Ollama no carguen su modelo encima (se libera al salir).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

_TEXT = ("El sistema de recuperación híbrida combina BM25 con vectores densos "
         "y pesos sparse de BGE-M3 para rankear evidencia del corpus local. ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batches", default="16,64", help="batches a comparar")
    parser.add_argument("--texts", type=int, default=64)
    parser.add_argument("--threads", type=int, default=0, help="0 = default de torch")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--hold-vram-lock", action="store_true")
    args = parser.parse_args()

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)

    lock_owner = None
    if args.hold_vram_lock:
        from ipa.providers import vram_lock
        if not vram_lock.acquire("embed_tuning_probe"):
            print("vram.lock ocupado por otro proceso — abortando", flush=True)
            sys.exit(2)
        lock_owner = "embed_tuning_probe"
        print(f"vram.lock tomado ({lock_owner})", flush=True)

    try:
        from ipa.indexes.embedding_adapter import EmbeddingAdapter

        texts = [_TEXT * 2] * args.texts
        t0 = time.monotonic()
        adapter = EmbeddingAdapter(device=args.device, show_progress=False,
                                   max_length=args.max_length)
        adapter._ensure_model()
        load_s = time.monotonic() - t0
        print(f"load: {load_s:.1f}s  device={adapter.active_device}  "
              f"fp16={adapter.use_fp16 and adapter.active_device == 'cuda'}  "
              f"max_length={adapter.max_length}  torch_threads={torch.get_num_threads()}",
              flush=True)

        adapter.embed_texts_hybrid(texts[:2])  # warmup (fuera de la medición)
        for batch in [int(b) for b in args.batches.split(",")]:
            adapter.batch_size = batch
            t = time.monotonic()
            adapter.embed_texts_hybrid(texts)
            elapsed = time.monotonic() - t
            print(f"device={args.device} batch={batch:4d}: {args.texts} texts en "
                  f"{elapsed:.2f}s = {args.texts / elapsed:.1f} texts/s", flush=True)
        adapter.close()
    finally:
        if lock_owner:
            from ipa.providers import vram_lock
            vram_lock.release(lock_owner)
            print("vram.lock liberado", flush=True)


if __name__ == "__main__":
    main()
