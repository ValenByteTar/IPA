"""Tests del fallback CPU-only (providers/device + factory + llm_status)."""
from __future__ import annotations

import pytest

from ipa.providers import device


@pytest.fixture(autouse=True)
def _clean_cache():
    device.reset_gpu_cache()
    yield
    device.reset_gpu_cache()


def test_force_cpu_returns_false(monkeypatch):
    monkeypatch.setenv("IPA_FORCE_CPU", "1")
    assert device.has_gpu() is False


def test_cache_is_stable(monkeypatch):
    device.reset_gpu_cache()
    a = device.has_gpu()
    b = device.has_gpu()
    assert a == b


def test_factory_falls_back_to_ollama_without_gpu(monkeypatch):
    """IPA_LLM_PROVIDER=exl3 sin GPU → Ollama (CPU), nunca un crash."""
    monkeypatch.setenv("IPA_LLM_PROVIDER", "exl3")
    monkeypatch.setenv("IPA_FORCE_CPU", "1")
    device.reset_gpu_cache()

    from ipa.providers import factory
    provider = factory.create_star_provider(interactive=True)
    # El provider de Ollama expone model_id/model y generate_chat
    assert hasattr(provider, "generate_chat")
    assert factory.DEFAULT_OLLAMA_MODEL in str(getattr(provider, "model", "")) or \
        getattr(provider, "model_id", None) is not None


def test_factory_respects_ollama_default(monkeypatch):
    monkeypatch.setenv("IPA_LLM_PROVIDER", "ollama")
    from ipa.providers import factory
    provider = factory.create_star_provider()
    assert hasattr(provider, "generate_chat")


def test_exl3_disables_mtp_for_batch(monkeypatch):
    """MTP + batch > 2 thrashea (medido: batch 6 con MTP 17.5 tok/s vs 83.2
    sin MTP; realineación del speculative decoding — arXiv 2510.22876). El
    guard lo desactiva solo, salvo IPA_EXL3_FORCE_MTP=1."""
    from ipa.providers.exl3_provider import ExL3Provider

    def _mk(batch_size: int) -> ExL3Provider:
        return ExL3Provider(
            model_path="x", model_id="x", quantization="x",
            batch_size=batch_size, use_mtp=True,
        )

    assert _mk(1).use_mtp is True          # interactivo: el MTP aporta (+14%)
    assert _mk(2).use_mtp is True
    assert _mk(3).use_mtp is False         # batch: guard lo apaga
    assert _mk(6).use_mtp is False
    # Override explícito.
    monkeypatch.setenv("IPA_EXL3_FORCE_MTP", "1")
    assert _mk(6).use_mtp is True


class _FakeResp:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        import json
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _patch_ollama_api(monkeypatch, ps_sequence: list[list[dict]]):
    """Mockea /api/ps (secuencia de respuestas) y captura POSTs a /api/generate."""
    import urllib.request
    calls = {"posts": [], "ps": 0}

    def _urlopen(req_or_url, timeout=None):
        url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
        if url.endswith("/api/ps"):
            idx = min(calls["ps"], len(ps_sequence) - 1)
            calls["ps"] += 1
            return _FakeResp({"models": ps_sequence[idx]})
        if url.endswith("/api/generate"):
            calls["posts"].append(getattr(req_or_url, "data", b""))
            return _FakeResp({})
        raise AssertionError(f"url inesperada: {url}")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    return calls


def test_unload_ollama_models_waits_for_release(monkeypatch):
    """La liberación de VRAM es async: hay que esperar a /api/ps vacío antes
    de que ExL3 mida la libre para el split (sino muere con 'Insufficient
    VRAM in split')."""
    from ipa.providers.exl3_provider import _unload_ollama_models

    calls = _patch_ollama_api(monkeypatch, [
        [{"name": "qwen3.5:9b-q4_K_M"}],   # /api/ps inicial: modelo cargado
        [{"name": "qwen3.5:9b-q4_K_M"}],   # primer poll: aún liberando
        [],                                 # segundo poll: liberado
    ])
    _unload_ollama_models(timeout_s=10)
    assert len(calls["posts"]) == 1        # POST keep_alive=0 enviado
    assert calls["ps"] >= 3                # siguió polleando hasta vacío


def test_unload_ollama_models_noop_when_empty(monkeypatch):
    from ipa.providers.exl3_provider import _unload_ollama_models

    calls = _patch_ollama_api(monkeypatch, [[]])
    _unload_ollama_models()
    assert calls["posts"] == []            # nada que descargar: no POSTea
    assert calls["ps"] == 1


def test_unload_ollama_models_bounded_by_timeout(monkeypatch):
    """Si Ollama nunca libera, el wait está acotado (no cuelga la carga)."""
    import time
    from ipa.providers.exl3_provider import _unload_ollama_models

    calls = _patch_ollama_api(monkeypatch, [[{"name": "m"}]])
    t0 = time.monotonic()
    _unload_ollama_models(timeout_s=1.5)
    assert time.monotonic() - t0 < 4       # ~timeout + overhead de la llamada
    assert len(calls["posts"]) == 1
