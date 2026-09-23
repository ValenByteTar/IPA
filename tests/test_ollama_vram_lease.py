"""An Ollama generation owns the VRAM lease for its whole streaming lifetime."""
from __future__ import annotations

import json

from ipa.agentic import embedding_maintenance
from ipa.providers import ollama_provider, vram_lock


def _isolate_locks(tmp_path, monkeypatch):
    monkeypatch.setattr(vram_lock, "LOCK_PATH", tmp_path / "vram.lock")
    monkeypatch.setattr(vram_lock, "_pid_alive_cache", {})
    monkeypatch.setattr(embedding_maintenance, "STATE_PATH", tmp_path / "embed.json")
    monkeypatch.setattr(embedding_maintenance, "JOB_LOCK_PATH", tmp_path / "embed.lock")
    monkeypatch.setattr(ollama_provider, "_log_perf", lambda *a, **k: None)


class _FakeResponse:
    def __init__(self, lines):
        self.lines = iter(lines)

    def readline(self):
        return next(self.lines, b"")


class _FakeConnection:
    lines = []
    instances = []

    def __init__(self, *args, **kwargs):
        self.closed = False
        self.instances.append(self)

    def request(self, *args, **kwargs):
        pass

    def getresponse(self):
        return _FakeResponse(self.lines)

    def close(self):
        self.closed = True


def _stream_lines():
    return [
        (json.dumps({"message": {"content": "token"}, "done": False}) + "\n").encode(),
        (json.dumps({"done": True, "eval_count": 1}) + "\n").encode(),
    ]


def test_ollama_owns_vram_until_stream_finishes(tmp_path, monkeypatch):
    import http.client

    _isolate_locks(tmp_path, monkeypatch)
    _FakeConnection.lines = _stream_lines()
    _FakeConnection.instances = []
    monkeypatch.setattr(http.client, "HTTPConnection", _FakeConnection)
    provider = ollama_provider.OllamaProvider()
    provider._loaded = True

    stream = provider.generate_chat_stream([{"role": "user", "content": "hola"}])
    first = next(stream)
    assert first["text"] == "token"
    holder = vram_lock.holder()
    assert holder is not None and holder["owner"] == "ollama"
    assert vram_lock.acquire("bulk_embedding") is False

    assert list(stream)[0]["done"] is True
    assert vram_lock.holder() is None
    assert _FakeConnection.instances[0].closed is True


def test_ollama_generation_fails_clearly_during_bulk_lease(tmp_path, monkeypatch):
    import http.client

    _isolate_locks(tmp_path, monkeypatch)
    _FakeConnection.instances = []
    monkeypatch.setattr(http.client, "HTTPConnection", _FakeConnection)
    assert vram_lock.acquire("bulk_embedding") is True
    provider = ollama_provider.OllamaProvider()
    provider._loaded = True

    chunks = list(provider.generate_chat_stream([
        {"role": "user", "content": "hola"},
    ]))
    assert chunks[0]["error"]
    assert "ingesta masiva" in chunks[0]["error"].lower()
    assert not _FakeConnection.instances
    vram_lock.release("bulk_embedding")
