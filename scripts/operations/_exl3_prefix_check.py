"""¿ExLlamaV3 ya reusa prefijo entre jobs? Misma prompt 3 veces → TTFT.

Si el TTFT cae fuerte en la 2ª/3ª corrida, el PT cache de la librería está
activo (hash de páginas) y no hace falta implementar nada en el provider.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ipa.providers.exl3_provider import ExL3Provider  # noqa: E402

MODEL = str(ROOT / "models" / "Qwen3.5-9B-exl3-3.0bpw")
CTX = (ROOT / "docs" / "Horizontalidad.md").read_text(encoding="utf-8", errors="ignore")[:8000]
PROMPT = "Contexto:\n\n" + CTX + "\n\n---\n\nExplica en 3 oraciones qué es la recuperación híbrida."


def main() -> None:
    p = ExL3Provider(
        model_path=MODEL, model_id="Qwen3.5-9B-EXL3-3.0bpw", quantization="EXL3-3.0bpw",
        context_length=6144, max_output_tokens=80, temperature=0.1,
        no_think=True, batch_size=1, use_mtp=True, mtp_draft_tokens=2,
        mtp_cache_tokens=4096, cache_k_bits=8, cache_v_bits=8,
        suppress_cjk=True, rep_p=1.15,
    )
    p.load()
    toks = int(p._tokenizer.encode(PROMPT, add_bos=False).shape[-1])
    print(f"prompt: {toks} tokens\n")
    for i in range(3):
        r = p.generate_chat([{"role": "user", "content": PROMPT}], max_new_tokens=80)
        print(f"run {i + 1}: TTFT {r.time_to_first_token_ms:.0f}ms | "
              f"{r.tokens_generated} tok | {r.latency_s:.1f}s | {r.tokens_per_second:.1f} tok/s")
    # Prompt distinto (prefijo distinto) para comparar
    other = "Pregunta totalmente distinta: ¿qué es BM25?"
    r = p.generate_chat([{"role": "user", "content": other}], max_new_tokens=80)
    print(f"\nprompt nuevo (sin prefijo compartido): TTFT {r.time_to_first_token_ms:.0f}ms | "
          f"{r.tokens_generated} tok | {r.latency_s:.1f}s")
    p.unload()


if __name__ == "__main__":
    main()
