"""Smoke test del modelo estrella Qwen3.5-9B EXL3 3.0bpw + MTP.

Verifica que el provider carga correctamente, genera texto coherente,
y reporta metricas de rendimiento (tok/s, VRAM, latencia).

Uso:
  .venv\\Scripts\\python.exe scripts\\test_exl3_provider.py
  .venv\\Scripts\\python.exe scripts\\test_exl3_provider.py --interactive
  .venv\\Scripts\\python.exe scripts\\test_exl3_provider.py --batch
  .venv\\Scripts\\python.exe scripts\\test_exl3_provider.py --all
"""
import argparse
import sys
import time
from pathlib import Path

# Asegurar que src/ esta en el path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ipa.exl3_provider import (
    ExL3Provider,
    create_star_provider,
    query_gpu_info,
    query_vram_mb,
)


def test_single_generation(provider: ExL3Provider) -> bool:
    """Test de generacion individual (interactivo)."""
    print("\n" + "=" * 60)
    print("Test 1: Generacion individual (single prompt)")
    print("=" * 60)

    messages = [
        {"role": "user", "content": "Explica en 2 oraciones que es BM25 en recuperacion de informacion."},
    ]

    t0 = time.monotonic()
    result = provider.generate_chat(
        messages,
        max_new_tokens=200,
        temperature=0.0,
        stop_sequences=["<|im_end|>"],
    )
    elapsed = time.monotonic() - t0

    print(f"  Respuesta: {result.text[:300]}")
    print(f"  Latencia: {result.latency_s:.2f}s")
    print(f"  Tokens generados: {result.tokens_generated}")
    print(f"  Tokens/s: {result.tokens_per_second:.1f}")
    print(f"  TTFT: {result.time_to_first_token_ms:.0f}ms")
    print(f"  VRAM antes: {result.vram_before_mb} MB")
    print(f"  VRAM despues: {result.vram_after_mb} MB")
    print(f"  VRAM pico: {result.vram_peak_mb} MB")
    print(f"  Error: {result.error or 'none'}")
    print(f"  OK: {result.ok}")

    if not result.ok or len(result.text.strip()) < 10:
        print("\n  [FAIL] generacion vacia o con error")
        return False
    print("\n  [PASS]")
    return True


def test_batch_generation(provider: ExL3Provider) -> bool:
    """Test de generacion batch (continuous batching con MTP)."""
    print("\n" + "=" * 60)
    print("Test 2: Generacion batch (continuous batching)")
    print("=" * 60)

    batch_messages = [
        [{"role": "user", "content": "Explica que es BM25 en una oracion."}],
        [{"role": "user", "content": "Explica que son los embeddings en una oracion."}],
        [{"role": "user", "content": "Explica que es la recuperacion hibrida en una oracion."}],
        [{"role": "user", "content": "Explica que es el reranking en una oracion."}],
    ]

    t0 = time.monotonic()
    results = provider.generate_chat_batch(
        batch_messages,
        max_new_tokens=150,
        temperature=0.0,
        stop_sequences=["<|im_end|>"],
    )
    elapsed = time.monotonic() - t0

    all_ok = True
    total_tokens = 0
    for i, r in enumerate(results):
        print(f"\n  [{i+1}] {r.text[:150]}")
        print(f"      {r.tokens_per_second:.1f} tok/s, {r.latency_s:.2f}s, {r.tokens_generated} tokens")
        total_tokens += r.tokens_generated
        if not r.ok or len(r.text.strip()) < 5:
            print(f"      [FAIL]")
            all_ok = False

    print(f"\n  Total: {total_tokens} tokens en {elapsed:.2f}s")
    print(f"  Throughput agregado: {total_tokens / elapsed:.1f} tok/s")
    print(f"  VRAM pico: {max(r.vram_peak_mb for r in results)} MB")

    if all_ok:
        print("\n  [PASS]")
    else:
        print("\n  [FAIL] alguna generacion fallo")
    return all_ok


def test_structured_output(provider: ExL3Provider) -> bool:
    """Test de salida estructurada (JSON) - funcion critica del tutor."""
    print("\n" + "=" * 60)
    print("Test 3: Salida estructurada (JSON)")
    print("=" * 60)

    messages = [
        {"role": "user", "content": (
            "Evalua esta respuesta de un estudiante sobre BM25. "
            "Responde SOLO con JSON valido.\n\n"
            "Pregunta: Que es BM25?\n"
            "Respuesta del estudiante: BM25 es un algoritmo de busqueda que usa TF-IDF.\n\n"
            'Formato: {"score": 0.0-1.0, "status": "understood|applied|misconception", '
            '"feedback": "texto breve", "gaps": ["lista de gaps"]}'
        )},
    ]

    result = provider.generate_chat(
        messages,
        max_new_tokens=256,
        temperature=0.0,
        stop_sequences=["<|im_end|>"],
    )

    print(f"  Respuesta: {result.text[:400]}")
    print(f"  {result.tokens_per_second:.1f} tok/s, {result.latency_s:.2f}s")

    import json
    try:
        text = result.text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end+1])
            print(f"  JSON parseado OK: score={parsed.get('score')}, status={parsed.get('status')}")
            print("\n  [PASS]")
            return True
        else:
            print(f"  [FAIL] no se encontro JSON en la respuesta")
            return False
    except json.JSONDecodeError as e:
        print(f"  [FAIL] JSON invalido: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Smoke test del modelo estrella EXL3")
    parser.add_argument("--interactive", action="store_true",
                        help="Modo interactivo (batch_size=1, 45 tok/s)")
    parser.add_argument("--batch", action="store_true",
                        help="Ejecutar tambien test de batch")
    parser.add_argument("--structured", action="store_true",
                        help="Ejecutar tambien test de salida estructurada")
    parser.add_argument("--all", action="store_true",
                        help="Ejecutar todos los tests")
    parser.add_argument("--model-path", default="models/Qwen3.5-9B-exl3-3.0bpw",
                        help="Path al directorio del modelo")
    args = parser.parse_args()

    gpu = query_gpu_info()
    print("=" * 60)
    print("Smoke Test - Qwen3.5-9B EXL3 3.0bpw + MTP")
    print("=" * 60)
    if gpu:
        print(f"  GPU: {gpu['name']}")
        print(f"  VRAM total: {gpu['vram_total_mb']} MB")
        print(f"  VRAM usada: {gpu['vram_used_mb']} MB")
        print(f"  VRAM libre: {gpu['vram_free_mb']} MB")
    else:
        print("  [WARN] No se detecto GPU via nvidia-smi")

    mode = "interactivo (batch=1)" if args.interactive else "paralelo (batch=6)"
    print(f"  Modo: {mode}")
    print(f"  MTP: habilitado (speculative decoding)")
    print(f"  no_think: habilitado (respuestas directas)")

    provider = create_star_provider(
        model_path=args.model_path,
        interactive=args.interactive,
    )

    print(f"\nCargando modelo desde {args.model_path}...")
    t0 = time.monotonic()
    load_time = provider.load()
    print(f"  Modelo cargado en {load_time:.1f}s")
    print(f"  VRAM tras carga: {provider.vram_usage_mb()} MB (allocated)")
    print(f"  VRAM tras carga: {query_vram_mb()} MB (nvidia-smi)")

    if not provider.is_loaded():
        print("\n[FAIL] el modelo no se cargo correctamente")
        sys.exit(1)

    results = []
    results.append(test_single_generation(provider))

    if args.batch or args.all:
        provider.reset_generator()
        results.append(test_batch_generation(provider))

    if args.structured or args.all:
        provider.reset_generator()
        results.append(test_structured_output(provider))

    print("\n" + "=" * 60)
    print("Resumen")
    print("=" * 60)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"  Tests: {passed}/{total} pasados")
    print(f"  VRAM final: {query_vram_mb()} MB")

    provider.unload()
    print(f"  VRAM tras unload: {query_vram_mb()} MB")

    if all(results):
        print("\n  [PASS] Todos los tests pasaron - modelo estrella operativo")
        sys.exit(0)
    else:
        print("\n  [FAIL] Algunos tests fallaron")
        sys.exit(1)


if __name__ == "__main__":
    main()
