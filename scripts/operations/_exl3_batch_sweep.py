"""Sweep de batch_size para ExL3 en 6 GB — buscar el sweet spot de throughput.

batch_size es load-time (dimensiona el cache), así que cada punto recarga el
modelo. Mide: throughput agregado, por secuencia (varianza = thrashing) y TTFT.

Uso: python scripts/operations/_exl3_batch_sweep.py [--sizes 1,2,3,4] [--ctx 2048]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ipa.providers.exl3_provider import ExL3Provider  # noqa: E402

MODEL = str(ROOT / "models" / "Qwen3.5-9B-exl3-3.0bpw")
CTX_TEXT = (ROOT / "docs" / "Horizontalidad.md").read_text(encoding="utf-8", errors="ignore")[:3000]
PREFIX = "Contexto compartido del proyecto:\n\n" + CTX_TEXT + "\n\n---\n\n"
QUESTIONS = [
    "¿Qué es la recuperación híbrida?",
    "¿Qué aporta BM25?",
    "¿Qué aporta la búsqueda densa?",
    "¿Cómo funciona RRF?",
    "¿Cuándo falla BM25?",
    "¿Cuándo falla la búsqueda densa?",
]


def build(batch_size: int, ctx: int, use_mtp: bool = True) -> ExL3Provider:
    return ExL3Provider(
        model_path=MODEL, model_id="Qwen3.5-9B-EXL3-3.0bpw", quantization="EXL3-3.0bpw",
        context_length=ctx, max_output_tokens=120, temperature=0.1,
        no_think=True, batch_size=batch_size, use_mtp=use_mtp, mtp_draft_tokens=2,
        mtp_cache_tokens=ctx if use_mtp else 0, cache_k_bits=8, cache_v_bits=8,
        suppress_cjk=True, rep_p=1.15,
    )


def run_batch(provider: ExL3Provider, n_prompts: int, max_tokens: int = 120) -> dict:
    msgs = [[{"role": "user", "content": PREFIX + q}] for q in QUESTIONS[:n_prompts]]
    t0 = time.monotonic()
    results = provider.generate_chat_batch(msgs, max_new_tokens=max_tokens)
    wall = time.monotonic() - t0
    per = [r.tokens_per_second for r in results]
    total = sum(r.tokens_generated for r in results)
    return {
        "wall_s": round(wall, 1),
        "total_tok": total,
        "agg_tok_s": round(total / wall, 1) if wall > 0 else 0.0,
        "per_seq": [round(x, 1) for x in per],
        "min_seq": round(min(per), 1) if per else 0.0,
        "stdev": round(statistics.pstdev(per), 1) if len(per) > 1 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1,2,3,4")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=2, help="corridas por tamaño (la 1ª calienta)")
    ap.add_argument("--no-mtp", action="store_true", help="correr sin MTP (comparativa)")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    out = {}
    for bs in sizes:
        p = build(bs, args.ctx, use_mtp=not args.no_mtp)
        t0 = time.monotonic()
        p.load()
        load_s = time.monotonic() - t0
        print(f"\n=== batch_size={bs} ctx={args.ctx} (load {load_s:.0f}s) ===", flush=True)
        rows = []
        for rep in range(args.reps):
            r = run_batch(p, len(QUESTIONS))
            rows.append(r)
            print(f"  rep {rep + 1}: {r['total_tok']} tok en {r['wall_s']}s = "
                  f"{r['agg_tok_s']} tok/s agregado | por seq {r['per_seq']} | "
                  f"min {r['min_seq']} stdev {r['stdev']}", flush=True)
        out[f"bs{bs}"] = {"load_s": round(load_s, 1), "ctx": args.ctx, "reps": rows}
        p.reset_generator()
        p.unload()

    print("\n===== RESUMEN (última rep de cada tamaño) =====", flush=True)
    for k, v in out.items():
        last = v["reps"][-1]
        print(f"{k}: {last['agg_tok_s']} tok/s agregado | min_seq {last['min_seq']} "
              f"| stdev {last['stdev']} | wall {last['wall_s']}s")
    path = ROOT / "outputs" / "experiments" / "exl3-batch-sweep" / f"sweep_{int(time.time())}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        __import__("json").dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"reporte: {path}", flush=True)


if __name__ == "__main__":
    main()
