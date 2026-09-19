"""ExL3: tuning de mtp_draft_tokens + throughput batch con prefijo compartido.

- draft 2 vs 4: profundidad del speculative decoding (más draft = más tokens
  por paso, pero menor aceptación).
- batch 6: 6 prompts que comparten el prefijo de sistema (caso real de los
  jobs batch: mismo template) → mide el throughput agregado.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ipa.providers.exl3_provider import ExL3Provider  # noqa: E402

MODEL = str(ROOT / "models" / "Qwen3.5-9B-exl3-3.0bpw")
CTX = (ROOT / "docs" / "Horizontalidad.md").read_text(encoding="utf-8", errors="ignore")[:3000]
PREFIX = "Contexto compartido del proyecto:\n\n" + CTX + "\n\n---\n\n"
QUESTIONS = [
    "¿Qué es la recuperación híbrida?",
    "¿Qué aporta BM25?",
    "¿Qué aporta la búsqueda densa?",
    "¿Cómo funciona RRF?",
    "¿Cuándo falla BM25?",
    "¿Cuándo falla la búsqueda densa?",
]


def build(draft: int) -> ExL3Provider:
    # ctx 2048: los jobs batch son de prompts cortos (clasificar/etiquetar);
    # batch 6 × 6144 KV no entra en 6 GB (medido: OOM al cargar).
    return ExL3Provider(
        model_path=MODEL, model_id="Qwen3.5-9B-EXL3-3.0bpw", quantization="EXL3-3.0bpw",
        context_length=2048, max_output_tokens=120, temperature=0.1,
        no_think=True, batch_size=6, use_mtp=True, mtp_draft_tokens=draft,
        mtp_cache_tokens=2048, cache_k_bits=8, cache_v_bits=8,
        suppress_cjk=True, rep_p=1.15,
    )


def main() -> None:
    for draft in (2, 4):
        p = build(draft)
        t0 = time.monotonic()
        p.load()
        load_s = time.monotonic() - t0
        print(f"\n=== mtp_draft_tokens={draft} (load {load_s:.0f}s) ===")
        # single, dos veces (la 2ª mide reuse de prefijo)
        for i in (1, 2):
            r = p.generate_chat([{"role": "user", "content": PREFIX + QUESTIONS[0]}],
                                max_new_tokens=120)
            print(f"  single #{i}: TTFT {r.time_to_first_token_ms:.0f}ms | "
                  f"{r.tokens_generated} tok | {r.tokens_per_second:.1f} tok/s")
        # batch 6 con prefijo compartido
        msgs = [[{"role": "user", "content": PREFIX + q}] for q in QUESTIONS]
        t0 = time.monotonic()
        results = p.generate_chat_batch(msgs, max_new_tokens=120)
        wall = time.monotonic() - t0
        total_tok = sum(r.tokens_generated for r in results)
        per = [round(r.tokens_per_second, 1) for r in results]
        print(f"  batch6: {total_tok} tok en {wall:.1f}s = {total_tok / wall:.1f} tok/s agregado "
              f"(por seq: {per})")
        p.reset_generator()
        p.unload()


if __name__ == "__main__":
    main()
