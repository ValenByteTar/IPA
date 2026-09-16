"""OllamaProvider server-bootstrap tests.

The bug: when /api/tags timed out, the provider spawned its own
`ollama serve`; its llama-server.exe grandchild popped a visible console
window on every model load/unload. Fix: if the tray app is alive, poll the
API instead of spawning a duplicate server.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ipa.providers.ollama_provider import OllamaProvider  # noqa: E402


def _provider() -> OllamaProvider:
    return OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model")


def test_ensure_server_returns_fast_when_api_up():
    p = _provider()
    with patch("urllib.request.urlopen") as mock_open:
        mock_open.return_value = MagicMock()
        p._ensure_server()
    assert mock_open.call_count == 1


def test_ensure_server_waits_for_tray_instead_of_spawning():
    """Tray app running → poll API, never spawn 'ollama serve'."""
    p = _provider()
    calls = {"n": 0}

    def flaky_urlopen(url, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("not yet")
        return MagicMock()

    with patch("urllib.request.urlopen", side_effect=flaky_urlopen), \
         patch.object(OllamaProvider, "_tray_app_running", return_value=True), \
         patch("subprocess.Popen") as mock_popen:
        p._ensure_server()
    mock_popen.assert_not_called()
    assert calls["n"] >= 3


def test_ensure_server_spawns_only_without_tray(monkeypatch):
    """No tray app → fall back to launching 'ollama serve' hidden."""
    p = _provider()
    calls = {"n": 0}

    def flaky_urlopen(url, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("not yet")
        return MagicMock()

    with patch("urllib.request.urlopen", side_effect=flaky_urlopen), \
         patch.object(OllamaProvider, "_tray_app_running", return_value=False), \
         patch("shutil.which", return_value="C:\\Ollama\\ollama.exe"), \
         patch("subprocess.Popen") as mock_popen:
        p._ensure_server()
    mock_popen.assert_called_once()
    args = mock_popen.call_args
    assert args[0][0][1] == "serve"
    # Windows: detached + no window so OUR process stays hidden.
    if sys.platform == "win32":
        import subprocess
        flags = args[1]["creationflags"]
        assert flags & subprocess.CREATE_NO_WINDOW
        assert flags & subprocess.DETACHED_PROCESS
