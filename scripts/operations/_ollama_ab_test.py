"""A/B de sampler en Ollama: rep_p × repeat_last_n con métricas de drift.

Sin carga de modelo (Ollama ya está arriba). Mide tok/s, repetición y
densidad de stopwords españolas por ventana.

Uso: python scripts/operations/_ollama_ab_test.py [--repeat 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "operations"))
from _exl3_fatigue_test import drift_metrics  # noqa: E402

MODEL = "qwen3.5:9b-q4_K_M"
PROMPT = (
    "Explica con detalle cómo funciona la recuperación híbrida en un sistema RAG "
    "moderno: qué aporta BM25, qué aporta la búsqueda densa con embeddings, cómo "
    "se fusionan ambos rankings (RRF) y por qué el resultado combinado es mejor "
    "que cada método por separado. Da ejemplos concretos de consultas donde cada "
    "uno falla."
)

CONFIGS = {
    "rep1.15_rln64":  {"repeat_penalty": 1.15, "repeat_last_n": 64},
    "rep1.00_rln64":  {"repeat_penalty": 1.00, "repeat_last_n": 64},
    "rep1.15_rln256": {"repeat_penalty": 1.15, "repeat_last_n": 256},
    "rep1.00_rln256": {"repeat_penalty": 1.00, "repeat_last_n": 256},
}


def generate(options: dict, max_tokens: int = 300) -> dict:
    opts = {"num_predict": max_tokens, "num_ctx": 6144, "temperature": 0.1,
            "top_p": 0.9, "top_k": 40}
    opts.update(options)
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False, "think": False, "keep_alive": "30m",
        "options": opts,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        return {"data": json.loads(r.read()), "wall_s": time.monotonic() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--configs", default="rep1.15_rln64,rep1.00_rln64,rep1.15_rln256,rep1.00_rln256")
    args = ap.parse_args()

    out = {}
    for name in [n.strip() for n in args.configs.split(",") if n.strip()]:
        cfg = CONFIGS[name]
        rows = []
        for i in range(args.repeat):
            r = generate(cfg)
            d = r["data"]
            text = (d.get("message") or {}).get("content", "")
            ev, ed = d.get("eval_count", 0), (d.get("eval_duration") or 1) / 1e9
            pe, pc = d.get("prompt_eval_count", 0), d.get("prompt_eval_cached_count", 0)
            m = drift_metrics(text)
            stop = round(sum(x["stopword_ratio"] for x in m) / max(1, len(m)), 2)
            rep = max((x["repeat_ratio"] for x in m), default=0.0)
            weird = max((x["weird_ratio"] for x in m), default=0.0)
            rows.append({"tok_s": round(ev / ed, 1), "eval": ev,
                         "prompt": pe, "cached": pc,
                         "stopword_avg": stop, "max_repeat": rep,
                         "max_weird": weird, "wall_s": round(r["wall_s"], 1)})
            print(f"  [{name} #{i + 1}] {ev} tok {ev / ed:.1f} tok/s | "
                  f"prompt {pe} (cached {pc}) | stopwords {stop} rep {rep} "
                  f"weird {weird}", flush=True)
        out[name] = rows
        mean = lambda k: round(sum(r[k] for r in rows) / len(rows), 2)  # noqa: E731
        print(f"=== {name}: {mean('tok_s')} tok/s | stopwords {mean('stopword_avg')} "
              f"| rep {mean('max_repeat')} | weird {mean('max_weird')}\n", flush=True)

    path = ROOT / "outputs" / "experiments" / "ollama-ab" / f"ab_{int(time.time())}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"reporte: {path}")


if __name__ == "__main__":
    main()
