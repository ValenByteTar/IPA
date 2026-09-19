"""Test de fatiga ExL3 9B 3.0bpw — aislamiento de la causa del drift.

Genera ~300 tokens con la misma prompt bajo variantes de config (KV bits,
rep_p, no_think, suppress_cjk) con MTP activo, y mide drift por ventana:
ratio de stopwords españolas, repetición de n-gramas y palabras raras.

Uso: python scripts/operations/_exl3_fatigue_test.py [--configs A,B,C]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ipa.providers.exl3_provider import ExL3Provider  # noqa: E402

MODEL = str(ROOT / "models" / "Qwen3.5-9B-exl3-3.0bpw")
PROMPT = (
    "Explica con detalle cómo funciona la recuperación híbrida en un sistema RAG "
    "moderno: qué aporta BM25, qué aporta la búsqueda densa con embeddings, cómo "
    "se fusionan ambos rankings (RRF) y por qué el resultado combinado es mejor "
    "que cada método por separado. Da ejemplos concretos de consultas donde cada "
    "uno falla."
)

STOPWORDS = {
    "de", "la", "que", "el", "en", "y", "los", "se", "del", "las", "un", "por",
    "con", "no", "una", "su", "para", "es", "al", "lo", "como", "mas", "más",
    "o", "pero", "sus", "le", "ha", "me", "si", "sí", "sin", "sobre", "este",
    "ya", "entre", "cuando", "todo", "esta", "ser", "son", "dos", "también",
    "fue", "era", "muy", "hasta", "desde", "está", "mi", "porque", "qué",
    "han", "yo", "hay", "puede", "todos", "así", "nos", "ni", "parte", "tiene",
    "él", "uno", "donde", "bien", "mismo", "ese", "ahora", "cada", "vida",
    "otro", "después", "te", "otros", "aunque", "esa", "eso", "hace", "otra",
    "tan", "siempre", "día", "tanto", "ella", "tres", "dijo", "sido", "gran",
    "según", "menos", "mundo", "año", "antes", "estado", "cinco", "nada",
    "hacer", "algo", "fuerza", "esos", "mucho", "quienes", "muchos", "misma",
    "les", "esa", "esas", "esos", "estos", "estas", "del", "cual", "cuales",
}

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
VOWELS = set("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _weird_ratio(words: list[str]) -> float:
    """Palabras con pinta de gibberish: sin vocales o runs de 5+ consonantes."""
    if not words:
        return 0.0
    weird = 0
    for w in words:
        lw = w.lower()
        if len(lw) >= 4 and not (set(lw) & VOWELS):
            weird += 1
        elif re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", lw):
            weird += 1
    return weird / len(words)


def _repeat_ratio(words: list[str], n: int = 4) -> float:
    if len(words) < n * 2:
        return 0.0
    grams = [tuple(w.lower() for w in words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


_SENT_RE = re.compile(r"[^.!?\n]+[.!?]?")


def _dup_sentence_ratio(text: str) -> float:
    """Fracciones de oraciones repetidas (drift típico: re-genera una sección)."""
    sents = [s.strip().lower() for s in _SENT_RE.findall(text)]
    sents = [s for s in sents if len(s.split()) >= 5]
    if len(sents) < 3:
        return 0.0
    return 1.0 - len(set(sents)) / len(sents)


def _tail_echo_ratio(text: str) -> float:
    """Cuánto del último 25% del texto reaparece antes (eco de sección)."""
    words = _words(text)
    if len(words) < 80:
        return 0.0
    cut = int(len(words) * 0.75)
    head = " ".join(w.lower() for w in words[:cut])
    tail = " ".join(w.lower() for w in words[cut:])
    grams = [tuple(tail.split()[i:i + 6]) for i in range(max(1, len(tail.split()) - 5))]
    if not grams:
        return 0.0
    hits = sum(1 for g in grams if " ".join(g) in head)
    return hits / len(grams)


def drift_metrics(text: str, window: int = 45) -> list[dict]:
    """Métricas por ventana de ~`window` palabras."""
    words = _words(text)
    out = []
    for start in range(0, len(words), window):
        chunk = words[start:start + window]
        if len(chunk) < 10:
            continue
        low = [w.lower() for w in chunk]
        out.append({
            "words": f"{start}-{start + len(chunk)}",
            "stopword_ratio": round(sum(1 for w in low if w in STOPWORDS) / len(chunk), 2),
            "repeat_ratio": round(_repeat_ratio(chunk), 2),
            "weird_ratio": round(_weird_ratio(chunk), 2),
        })
    return out


def build_provider(*, kv_bits: int, rep_p: float, no_think: bool,
                   suppress_cjk: bool) -> ExL3Provider:
    return ExL3Provider(
        model_path=MODEL,
        model_id="Qwen3.5-9B-EXL3-3.0bpw",
        quantization="EXL3-3.0bpw",
        context_length=6144,
        max_output_tokens=300,
        temperature=0.1,
        no_think=no_think,
        batch_size=1,
        use_mtp=True,
        mtp_draft_tokens=2,
        mtp_cache_tokens=4096,
        cache_k_bits=kv_bits,
        cache_v_bits=kv_bits,
        suppress_cjk=suppress_cjk,
        rep_p=rep_p,
    )


# (nombre, kv_bits, rep_p, no_think, suppress_cjk) — el load cambia solo con kv_bits
CONFIGS = {
    "A_actual":   dict(kv_bits=8,  rep_p=1.15, no_think=True,  suppress_cjk=True),
    "C_rep1.0":   dict(kv_bits=8,  rep_p=1.0,  no_think=True,  suppress_cjk=True),
    "D_think":    dict(kv_bits=8,  rep_p=1.15, no_think=False, suppress_cjk=True),
    "E_nocjk":    dict(kv_bits=8,  rep_p=1.15, no_think=True,  suppress_cjk=False),
    "B_kvfp16":   dict(kv_bits=0,  rep_p=1.15, no_think=True,  suppress_cjk=True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="A_actual,C_rep1.0,D_think,E_nocjk,B_kvfp16")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--repeat", type=int, default=1,
                    help="generaciones por config (el drift es estocástico a temp>0)")
    ap.add_argument("--context-file", default="",
                    help="archivo a inyectar como contexto largo (reproduce 'mucho contexto')")
    ap.add_argument("--context-chars", type=int, default=0,
                    help="recortar el contexto a N caracteres")
    args = ap.parse_args()

    global PROMPT
    if args.context_file:
        ctx = (ROOT / args.context_file).read_text(encoding="utf-8", errors="ignore")
        if args.context_chars:
            ctx = ctx[:args.context_chars]
        PROMPT = (
            "Contexto técnico del proyecto:\n\n" + ctx
            + "\n\n---\n\nCon el contexto de arriba, " + PROMPT[0].lower() + PROMPT[1:]
        )
        print(f"[ctx] contexto de {len(ctx)} chars inyectado", flush=True)

    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    out_dir = ROOT / "outputs" / "experiments" / "exl3-fatigue"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Agrupar por kv_bits: un load por grupo (kv_bits es load-time)
    order = sorted(names, key=lambda n: (CONFIGS[n]["kv_bits"] == 0, n))
    provider = None
    current_kv = None
    report: dict = {}

    for name in order:
        cfg = CONFIGS[name]
        if provider is None or cfg["kv_bits"] != current_kv:
            if provider is not None:
                provider.unload()
            provider = build_provider(**cfg)
            t0 = time.monotonic()
            provider.load()
            current_kv = cfg["kv_bits"]
            print(f"[{name}] load con kv_bits={cfg['kv_bits']} en {time.monotonic() - t0:.1f}s",
                  flush=True)
        else:
            # Knobs runtime: aplicar sin recargar (kv_bits es load-time).
            provider.rep_p = cfg["rep_p"]
            provider.no_think = cfg["no_think"]
            if cfg["suppress_cjk"] != provider.suppress_cjk:
                provider.suppress_cjk = cfg["suppress_cjk"]
                if cfg["suppress_cjk"]:
                    from ipa.providers.exl3_provider import _build_cjk_bias
                    provider._cjk_bias = _build_cjk_bias(provider._tokenizer)
                else:
                    provider._cjk_bias = None
        t0 = time.monotonic()
        runs = []
        drift_events = 0
        for i in range(args.repeat):
            result = provider.generate_chat(
                [{"role": "user", "content": PROMPT}],
                max_new_tokens=args.max_tokens,
            )
            text = result.text or ""
            dup = _dup_sentence_ratio(text)
            echo = _tail_echo_ratio(text)
            weird = max((m["weird_ratio"] for m in drift_metrics(text)), default=0.0)
            drifted = dup > 0.05 or echo > 0.3 or weird > 0.05
            drift_events += int(drifted)
            runs.append({
                "tokens": result.tokens_generated,
                "tok_s": round(result.tokens_per_second, 1),
                "dup_sentence_ratio": round(dup, 2),
                "tail_echo_ratio": round(echo, 2),
                "max_weird_ratio": weird,
                "drifted": drifted,
                "text": text,
            })
            print(f"  [{name} #{i + 1}] {result.tokens_generated} tok "
                  f"{result.tokens_per_second:.1f} tok/s dup={dup:.2f} "
                  f"echo={echo:.2f} weird={weird:.2f} "
                  f"{'DRIFT' if drifted else 'ok'}", flush=True)
        elapsed = time.monotonic() - t0
        metrics = drift_metrics(runs[-1]["text"])
        report[name] = {
            "cfg": cfg,
            "n": args.repeat,
            "drift_events": drift_events,
            "mean_tok_s": round(
                sum(r["tok_s"] for r in runs) / max(1, len(runs)), 1),
            "latency_s": round(elapsed, 1),
            "metrics": metrics,
            "runs": runs,
        }
        print(f"\n=== {name} {cfg} ===", flush=True)
        print(f"  {drift_events}/{args.repeat} con drift | "
              f"mean {report[name]['mean_tok_s']} tok/s", flush=True)
        for m in metrics:
            print(f"  words {m['words']:>8}: stopwords={m['stopword_ratio']} "
                  f"repeat={m['repeat_ratio']} weird={m['weird_ratio']}", flush=True)

    if provider is not None:
        provider.unload()
    path = out_dir / f"fatigue_{int(time.time())}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreporte: {path}", flush=True)


if __name__ == "__main__":
    main()
