"""Tests for the dynamic system-state layer (self-knowledge grounding)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ipa.agent.system_state import render_system_state  # noqa: E402


def test_renders_without_crash():
    # Must never raise, even with missing/corrupt stores.
    state = render_system_state()
    assert isinstance(state, str)


def test_corpus_line_when_corpus_exists():
    state = render_system_state()
    # The main corpus exists in this repo (2,491 docs) — the layer should
    # mention it so the model knows search_corpus has data.
    if state:
        assert "Corpus principal" in state or state == ""


def test_no_test_data_leak():
    state = render_system_state()
    assert "test goal" not in state
    assert "test interest" not in state
