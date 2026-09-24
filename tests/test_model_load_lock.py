"""Regression tests for PM-007 — concurrent model construction race.

transformers/accelerate patch ``nn.Module.register_parameter`` globally
during ``from_pretrained``; a concurrent nn.Module construction leaves
params on device ``meta`` permanently ("Cannot copy out of meta tensor").
Every heavyweight model ctor must run under ``MODEL_LOAD_LOCK``.
"""
from __future__ import annotations

import sys
import threading
import types

import pytest

from ipa.model_load_lock import MODEL_LOAD_LOCK


def _lock_held_by_another_thread() -> bool:
    """True if MODEL_LOAD_LOCK is currently held by someone else."""
    box: list[bool] = []

    def _probe() -> None:
        box.append(MODEL_LOAD_LOCK.acquire(blocking=False))

    t = threading.Thread(target=_probe)
    t.start()
    t.join()
    acquired = box[0]
    if acquired:
        MODEL_LOAD_LOCK.release()
    return not acquired


def _fake_flagembedding(m3_cls=None, rerank_cls=None) -> types.ModuleType:
    fake = types.ModuleType("FlagEmbedding")
    if m3_cls is not None:
        fake.BGEM3FlagModel = m3_cls
    if rerank_cls is not None:
        fake.FlagReranker = rerank_cls
    return fake


def _make_adapter():
    from ipa.indexes.embedding_adapter import EmbeddingAdapter

    ad = EmbeddingAdapter(device="cpu", show_progress=False)
    ad._resolve_device = lambda: "cpu"  # no nvidia-smi probe in tests
    return ad


def test_embedding_adapter_constructs_under_model_load_lock(monkeypatch):
    observed: list[bool] = []

    class FakeM3:
        def __init__(self, *a, **k):
            observed.append(_lock_held_by_another_thread())
            self.model = None

    monkeypatch.setitem(
        sys.modules, "FlagEmbedding", _fake_flagembedding(m3_cls=FakeM3)
    )
    _make_adapter()._ensure_model()
    assert observed == [True]


def test_embedding_adapter_single_construction_under_contention(monkeypatch):
    calls: list[int] = []

    class FakeM3:
        def __init__(self, *a, **k):
            calls.append(1)
            self.model = None

    monkeypatch.setitem(
        sys.modules, "FlagEmbedding", _fake_flagembedding(m3_cls=FakeM3)
    )
    ad = _make_adapter()
    threads = [threading.Thread(target=ad._ensure_model) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == [1]
    assert ad._model is not None


def test_embedding_adapter_meta_params_fail_loud_and_reset(monkeypatch):
    """Tripwire: if params still land on meta, raise instead of leaving a
    corrupted singleton that fails every query silently."""

    class _FakeParam:
        is_meta = True

    class _FakeInner:
        def parameters(self):
            return iter([_FakeParam()])

    class FakeM3:
        def __init__(self, *a, **k):
            self.model = _FakeInner()

    monkeypatch.setitem(
        sys.modules, "FlagEmbedding", _fake_flagembedding(m3_cls=FakeM3)
    )
    ad = _make_adapter()
    with pytest.raises(RuntimeError, match="meta"):
        ad._ensure_model()
    assert ad._model is None


def test_reranker_constructs_under_model_load_lock(monkeypatch):
    observed: list[bool] = []

    class FakeReranker:
        def __init__(self, *a, **k):
            observed.append(_lock_held_by_another_thread())

    monkeypatch.setitem(
        sys.modules, "FlagEmbedding", _fake_flagembedding(rerank_cls=FakeReranker)
    )
    from ipa.indexes.reranker_adapter import RerankerAdapter

    rr = RerankerAdapter.__new__(RerankerAdapter)
    rr._model = None
    rr._device_resolved = None
    rr.use_fp16 = False
    rr.model_name = "fake"
    rr._resolve_device = lambda: "cpu"
    rr._ensure_model()
    assert observed == [True]


def test_ocr_reader_constructs_under_model_load_lock(monkeypatch):
    observed: list[bool] = []

    class FakeReader:
        def __init__(self, *a, **k):
            observed.append(_lock_held_by_another_thread())

    fake_easyocr = types.ModuleType("easyocr")
    fake_easyocr.Reader = FakeReader
    monkeypatch.setitem(sys.modules, "easyocr", fake_easyocr)

    from ipa.acquisition.ocr_adapter import OCRAdapter

    ad = OCRAdapter(languages=["en"], gpu=False)
    ad._load_reader()
    assert observed == [True]
    assert ad._reader is not None
