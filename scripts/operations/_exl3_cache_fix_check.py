"""Verificación del fix de cache ExL3: mtp_cache_tokens >= context_length.

Prueba el mismo contexto que fallaba (4,376 tokens) con el cap viejo (4096)
y con el cache alineado a context_length (6144).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ipa.providers.exl3_provider import ExL3Provider  # noqa: E402

MODEL = str(ROOT / "models" / "Qwen3.5-9B-exl3-3.0bpw")
CTX = (ROOT / "docs" / "Horizontalidad.md").read_text(encoding="utf-8", errors="ignore")[:16000]
PROMPT = (
    "Contexto técnico del proyecto:\n\n" + CTX
    + "\n\n---\n\nCon el contexto de arriba, explica cómo funciona la recuperación "
    "híbrida: qué aporta BM25, qué aporta la búsqueda densa y cómo se fusionan."
)


def run(mtp_cache_tokens: int, context_length: int) -> None:
    p = ExL3Provider(
        model_path=MODEL, model_id="Qwen3.5-9B-EXL3-3.0bpw", quantization="EXL3-3.0bpw",
        context_length=context_length, max_output_tokens=300, temperature=0.1,
        no_think=True, batch_size=1, use_mtp=True, mtp_draft_tokens=2,
        mtp_cache_tokens=mtp_cache_tokens, cache_k_bits=8, cache_v_bits=8,
        suppress_cjk=True, rep_p=1.15,
    )
    p.load()
    toks = int(p._tokenizer.encode(PROMPT, add_bos=False).shape[-1])
    r = p.generate_chat([{"role": "user", "content": PROMPT}], max_new_tokens=300)
    print(f"\n### cache={mtp_cache_tokens} ctx={context_length} | prompt={toks} tok")
    print(f"    generados: {r.tokens_generated} tok en {r.latency_s:.1f}s "
          f"({r.tokens_per_second:.1f} tok/s) | error: {r.error}")
    print(f"    inicio: {r.text[:220]!r}")
    print(f"    final : {r.text[-220:]!r}")
    p.unload()


if __name__ == "__main__":
    run(mtp_cache_tokens=4096, context_length=6144)   # config vieja (cap 4096)
    run(mtp_cache_tokens=6144, context_length=6144)   # fix: cache = context
