"""Shared test fixtures.

Rerank is ON by default in production (opt-out via IPA_RERANK=0). Tests must
stay hermetic: the cross-encoder (BGE-reranker-v2-m3, ~2.1 GB) must never load
from the env default — a test that wants rerank behavior sets IPA_RERANK=1
explicitly via monkeypatch.

Embeddings por el mismo motivo: BGE-M3 (~2.2 GB) en CUDA mientras el
dashboard mantiene el modelo del chat en VRAM agota la GPU de 6 GB y congela
la UI de Windows (el compositor se queda sin memoria de video). Los tests
fuerzan CPU vía IPA_EMBED_DEVICE; un test que necesite GPU lo pisa explícito.
"""
import pytest


@pytest.fixture(autouse=True)
def _rerank_off_in_tests(monkeypatch):
    monkeypatch.setenv("IPA_RERANK", "0")


@pytest.fixture(autouse=True)
def _embeddings_on_cpu_in_tests(monkeypatch):
    monkeypatch.setenv("IPA_EMBED_DEVICE", "cpu")


@pytest.fixture(autouse=True)
def _clear_module_caches():
    """Los caches de proceso (tools, retrieval, respuestas, reranker) persisten
    entre tests y un test que monkeypatchea un store recibe un hit viejo
    (bug real: list_promotions servía un resultado de otro test)."""
    yield
    try:
        from ipa.agent import system_tools
        system_tools._TOOL_CACHE.clear()
    except Exception:
        pass
    try:
        from ipa.indexes import reranker_adapter
        reranker_adapter._rerank_cache._data.clear()
    except Exception:
        pass
    try:
        from ipa.dashboard import api
        api._RETRIEVAL_CACHE._data.clear()
        api._RESPONSE_CACHE._data.clear()
    except Exception:
        pass
