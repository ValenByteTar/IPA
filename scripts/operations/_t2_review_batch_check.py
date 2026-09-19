"""Smoke de integración: review batched contra el ExL3 real (batch 4, ctx 2048).

Valida el path completo del pase Tier 2: config del motor + guard de MTP +
helper de batch + parseo de veredictos.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ipa.agent.research_review import review_docs_with_llm  # noqa: E402
from ipa.agentic.batch_llm import supports_batch  # noqa: E402
from ipa.providers.exl3_provider import ExL3Provider  # noqa: E402

ITEMS = [
    {"text": "BM25 es un algoritmo de ranking léxico usado en recuperación de "
             "información. Combina TF e IDF con saturación de frecuencia.",
     "query": "bm25", "url": "https://ejemplo.com/bm25", "title": "BM25",
     "reason": "quality below threshold"},
    {"text": "Receta de pan casero: harina, agua, sal y levadura. Amasar 10 "
             "minutos y hornear 40 minutos a 200 grados.",
     "query": "retrieval", "url": "https://ejemplo.com/pan", "title": "Pan",
     "reason": "quality below threshold"},
    {"text": "RRF (Reciprocal Rank Fusion) fusiona rankings sumando 1/(k+rank) "
             "de cada lista; k=60 es el valor habitual en la literatura.",
     "query": "rrf", "url": "https://ejemplo.com/rrf", "title": "RRF",
     "reason": "date out of range"},
    {"text": "Ofertas de viajes y descuentos de temporada. Reservá ahora con "
             "30% de descuento en vuelos nacionales.",
     "query": "retrieval", "url": "https://ejemplo.com/viajes", "title": "Viajes",
     "reason": "judge rejected"},
]


def main() -> None:
    p = ExL3Provider(
        model_path=str(ROOT / "models" / "Qwen3.5-9B-exl3-3.0bpw"),
        model_id="Qwen3.5-9B-EXL3-3.0bpw", quantization="EXL3-3.0bpw",
        context_length=2048, max_output_tokens=128, temperature=0.1,
        no_think=True, batch_size=4, use_mtp=True, mtp_draft_tokens=2,
        mtp_cache_tokens=2048, cache_k_bits=8, cache_v_bits=8,
        suppress_cjk=True, rep_p=1.15,
    )
    print(f"supports_batch antes de cargar: {supports_batch(p)}")
    t0 = time.monotonic()
    p.load()
    print(f"load {time.monotonic() - t0:.0f}s | MTP activo: {p.use_mtp} "
          f"(guard: batch {p.batch_size} > 2)")
    t0 = time.monotonic()
    verdicts = review_docs_with_llm(p, ITEMS)
    dt = time.monotonic() - t0
    print(f"\n{len(ITEMS)} veredictos en {dt:.1f}s "
          f"({len(ITEMS) / dt:.2f} docs/s)")
    for item, v in zip(ITEMS, verdicts):
        flag = "PROMOTE" if v.get("promote") else "discard"
        print(f"  [{flag}] {item['title']}: {v.get('reason', '')[:70]} "
              f"{'ERROR: ' + v['error'] if v.get('error') else ''}")
    p.unload()


if __name__ == "__main__":
    main()
