"""Medición puntual de prefill/decode tok/s de Ollama (experimento de config).

Uso: python scripts/operations/_measure_llm.py [--ctx 8192] [--predict 120]
"""
from __future__ import annotations

import argparse
import json
import urllib.request

PROMPT = (
    "Contexto de prueba sobre sistemas de recuperación híbrida. " * 55
    + "\n\nExplica en detalle cómo se combinan BM25 y embeddings densos:"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--predict", type=int, default=120)
    ap.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    args = ap.parse_args()

    body = json.dumps({
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {"num_predict": args.predict, "num_ctx": args.ctx},
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())

    pe = data.get("prompt_eval_count", 0)
    pc = data.get("prompt_eval_cached_count", 0)
    pd_ = data.get("prompt_eval_duration", 1) / 1e9
    ev = data.get("eval_count", 0)
    ed = data.get("eval_duration", 1) / 1e9
    load = data.get("load_duration", 0) / 1e9
    print(f"load    : {load:.1f}s")
    print(f"prefill : {pe} tok ({pc} desde cache) en {pd_:.2f}s = {pe / pd_:.0f} tok/s")
    print(f"decode  : {ev} tok en {ed:.2f}s = {ev / ed:.1f} tok/s")


if __name__ == "__main__":
    main()
