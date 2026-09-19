"""Tests del helper de batch (batch_llm) y del review batched.

Cubre: uso de generate_chat_batch cuando el provider lo soporta (ExL3),
fallback serial (Ollama), aislamiento de errores por lote y parseo de
veredictos en orden.
"""
from __future__ import annotations

from dataclasses import dataclass

from ipa.agentic.batch_llm import generate_many, supports_batch


@dataclass
class _Result:
    text: str = ""
    error: str | None = None


class _BatchProvider:
    """Provider tipo ExL3: implementa generate_chat_batch."""

    engine = "exllamav3"

    def __init__(self, batch_size: int = 4, fail_on: int | None = None):
        self.batch_size = batch_size
        self.calls: list[list[str]] = []
        self.fail_on = fail_on  # índice de lote que debe fallar

    def generate_chat(self, messages, *, max_new_tokens=None, temperature=None):
        return _Result(text="serial:" + messages[-1]["content"])

    def generate_chat_batch(self, batch_messages, *, max_new_tokens=None,
                            temperature=None):
        idx = len(self.calls)
        self.calls.append([m[-1]["content"] for m in batch_messages])
        if self.fail_on is not None and idx == self.fail_on:
            raise RuntimeError("lote roto")
        return [_Result(text="batch:" + m[-1]["content"]) for m in batch_messages]


class _SerialProvider:
    """Provider tipo Ollama: sin generate_chat_batch, devuelve str."""

    engine = "ollama"

    def __init__(self):
        self.calls = 0

    def generate_chat(self, messages, *, max_new_tokens=None, temperature=None):
        self.calls += 1
        return "str:" + messages[-1]["content"]


def _conv(text: str):
    return [{"role": "user", "content": text}]


def test_supports_batch_detects_capability():
    assert supports_batch(_BatchProvider()) is True
    assert supports_batch(_SerialProvider()) is False


def test_generate_many_uses_batch_and_preserves_order():
    p = _BatchProvider(batch_size=4)
    out = generate_many(p, [_conv(f"a{i}") for i in range(3)], max_new_tokens=16)
    assert [t for t, _ in out] == ["batch:a0", "batch:a1", "batch:a2"]
    assert all(e is None for _, e in out)
    assert len(p.calls) == 1  # 3 items en un solo lote


def test_generate_many_chunks_by_batch_size():
    p = _BatchProvider(batch_size=2)
    out = generate_many(p, [_conv(f"a{i}") for i in range(5)], max_new_tokens=16)
    assert [len(c) for c in p.calls] == [2, 2, 1]
    assert [t for t, _ in out] == [f"batch:a{i}" for i in range(5)]


def test_generate_many_serial_fallback_for_ollama():
    p = _SerialProvider()
    out = generate_many(p, [_conv("x"), _conv("y")], max_new_tokens=16)
    assert [t for t, _ in out] == ["str:x", "str:y"]
    assert p.calls == 2


def test_generate_many_isolates_batch_failure():
    """Un lote que falla no tumba a los demás."""
    p = _BatchProvider(batch_size=2, fail_on=1)
    out = generate_many(p, [_conv(f"a{i}") for i in range(6)], max_new_tokens=16)
    assert out[0][0] == "batch:a0" and out[1][0] == "batch:a1"
    assert out[2][1] and "lote roto" in out[2][1]  # lote 2 falló
    assert out[3][1] and "lote roto" in out[3][1]
    assert out[4][0] == "batch:a4"  # lote 3 siguió


def test_generate_many_empty():
    assert generate_many(_BatchProvider(), [], max_new_tokens=8) == []


def test_review_docs_with_llm_parses_verdicts_in_order():
    from ipa.agent.research_review import review_docs_with_llm

    class _VerdictProvider(_BatchProvider):
        def generate_chat_batch(self, batch_messages, *, max_new_tokens=None,
                                temperature=None):
            # El primero promueve, el segundo no.
            return [
                _Result(text='{"promote": true, "reason": "aporta datos"}'),
                _Result(text='{"promote": false, "reason": "ruido"}'),
            ]

    items = [{"text": "doc A"}, {"text": "doc B"}]
    verdicts = review_docs_with_llm(_VerdictProvider(), items)
    assert verdicts[0]["promote"] is True
    assert verdicts[0]["reason"] == "aporta datos"
    assert verdicts[1]["promote"] is False
    assert verdicts[0]["error"] is None


def test_review_docs_with_llm_surfaces_provider_error():
    from ipa.agent.research_review import review_docs_with_llm

    class _BrokenProvider(_BatchProvider):
        def generate_chat_batch(self, batch_messages, *, max_new_tokens=None,
                                temperature=None):
            raise RuntimeError("sin VRAM")

    verdicts = review_docs_with_llm(_BrokenProvider(), [{"text": "doc"}])
    assert verdicts[0]["promote"] is False
    assert "sin VRAM" in verdicts[0]["error"]
