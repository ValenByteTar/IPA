"""Shared test fixtures.

Rerank is ON by default in production (opt-out via IPA_RERANK=0). Tests must
stay hermetic: the cross-encoder (BGE-reranker-v2-m3, ~2.1 GB) must never load
from the env default — a test that wants rerank behavior sets IPA_RERANK=1
explicitly via monkeypatch.
"""
import pytest


@pytest.fixture(autouse=True)
def _rerank_off_in_tests(monkeypatch):
    monkeypatch.setenv("IPA_RERANK", "0")
