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
