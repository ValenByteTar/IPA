"""Fase 0 tests: agent core contracts, memory, identity gate and omnipresence."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "validation"))

from ipa.agent import AgentCore, AgentMemory, load_identity  # noqa: E402
from validate_agent_contract import validate  # noqa: E402


# ---------------------------------------------------------------------------
# Identity (behavior gate: YAML changes behavior without touching code)
# ---------------------------------------------------------------------------

def test_identity_loads_with_hash():
    identity = load_identity()
    assert identity.name
    assert identity.identity_hash.startswith("sha256:")
    assert identity.user == "Valen"


def test_identity_yaml_change_changes_prompt_without_code(tmp_path):
    base = tmp_path / "identity.yaml"
    base.write_text(
        "name: Agente A\nuser: Tester\nlanguage: español\npersona: |\n  Personalidad A.\nprinciples:\n  - P1\n",
        encoding="utf-8",
    )
    identity_a = load_identity(base)
    prompt_a = identity_a.system_prompt(role="general")

    base.write_text(
        "name: Agente B\nuser: Tester\nlanguage: español\npersona: |\n  Personalidad B distinta.\nprinciples:\n  - P2\n",
        encoding="utf-8",
    )
    identity_b = load_identity(base)
    prompt_b = identity_b.system_prompt(role="general")

    assert prompt_a != prompt_b
    assert "Agente A" in prompt_a and "Personalidad A" in prompt_a
    assert "Agente B" in prompt_b and "Personalidad B" in prompt_b
    assert identity_a.identity_hash != identity_b.identity_hash


def test_identity_tutor_role_extends_prompt():
    identity = load_identity()
    general = identity.system_prompt(role="general")
    tutor = identity.system_prompt(role="tutor")
    assert tutor != general
    assert "tutor pedagógico" in tutor


# ---------------------------------------------------------------------------
# Memory: sessions + episodes (append-only, contract-shaped)
# ---------------------------------------------------------------------------

@pytest.fixture()
def memory(tmp_path):
    with AgentMemory(store_path=tmp_path / "agent.db") as store:
        yield store


def test_session_and_episode_round_trip(memory):
    identity = load_identity()
    session_id = memory.open_session(interface="cli", role="general", identity_hash=identity.identity_hash)
    episode = memory.record_episode(
        session_id, turn_role="user", content="Hola memoria", identity_hash=identity.identity_hash,
    )
    loaded = memory.get_episodes(session_id)
    assert len(loaded) == 1
    assert loaded[0].content == "Hola memoria"
    assert loaded[0].content_hash == episode.content_hash
    session = memory.get_session(session_id)
    assert session.episode_count == 1
    assert session.status == "active"


def test_episode_contract_validates(memory):
    identity = load_identity()
    session_id = memory.open_session(interface="cli", role="general", identity_hash=identity.identity_hash)
    episode = memory.record_episode(
        session_id, turn_role="assistant", content="Respuesta", identity_hash=identity.identity_hash,
    )
    errors = validate("AgentEpisode", episode.to_contract())
    assert errors == []
    session = memory.get_session(session_id)
    assert validate("AgentSession", session.to_contract()) == []


def test_episode_content_hash_detects_tampering(memory):
    identity = load_identity()
    session_id = memory.open_session(interface="cli", role="general", identity_hash=identity.identity_hash)
    episode = memory.record_episode(
        session_id, turn_role="user", content="texto original", identity_hash=identity.identity_hash,
    )
    tampered = episode.to_contract()
    tampered["content"] = "texto alterado"
    errors = validate("AgentEpisode", tampered)
    assert any("content_hash" in error for error in errors)


def test_closed_session_rejects_episodes(memory):
    identity = load_identity()
    session_id = memory.open_session(interface="cli", role="general", identity_hash=identity.identity_hash)
    memory.close_session(session_id)
    with pytest.raises(ValueError, match="closed"):
        memory.record_episode(session_id, turn_role="user", content="x", identity_hash=identity.identity_hash)


# ---------------------------------------------------------------------------
# Omnipresence gate: the session survives across processes
# ---------------------------------------------------------------------------

_WRITE_PROCESS = """
import sys
sys.path.insert(0, r"{src}")
from ipa.agent import AgentCore

core = AgentCore(interface="cli", role="general")
core.start_session(title="gate")
core.submit("mensaje escrito desde el proceso A")
print(core.session_id)
"""

_READ_PROCESS = """
import sys
sys.path.insert(0, r"{src}")
from ipa.agent import AgentMemory

memory = AgentMemory()
episodes = memory.recent_episodes(limit=5)
match = [e for e in episodes if e.content == "mensaje escrito desde el proceso A"]
print(match[0].session_id if match else "NOT_FOUND")
"""


def test_omnipresence_session_survives_across_processes(tmp_path):
    """The Fase 0 gate: a turn written by one process is visible to another."""
    env_root = Path(__file__).parents[1]
    src = str(env_root / "src")
    store = tmp_path / "agent.db"
    identity = load_identity()

    # Process A: open session and write a turn.
    writer = _WRITE_PROCESS.format(src=src)
    result = subprocess.run(
        [sys.executable, "-c", writer],
        capture_output=True, text=True, cwd=str(env_root), timeout=60,
        env={"PYTHONPATH": src, "IPA_AGENT_STORE": str(store), "PATH": "", "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")},
    )
    assert result.returncode == 0, result.stderr
    session_id = result.stdout.strip().splitlines()[-1]
    assert session_id.startswith("agent_session:")

    # Process B: a fresh process reads the same memory.
    reader = _READ_PROCESS.format(src=src)
    result = subprocess.run(
        [sys.executable, "-c", reader],
        capture_output=True, text=True, cwd=str(env_root), timeout=60,
        env={"PYTHONPATH": src, "IPA_AGENT_STORE": str(store), "PATH": "", "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == session_id


# ---------------------------------------------------------------------------
# AgentCore end-to-end (identity-driven prompt + episode recording)
# ---------------------------------------------------------------------------

def test_core_submit_records_both_turns_and_builds_identity_prompt(tmp_path):
    memory = AgentMemory(store_path=tmp_path / "agent.db")
    core = AgentCore(interface="cli", role="general", memory=memory)
    result = core.submit("¿Qué recordás de esta sesión?")
    session = core.get_session()
    assert session.episode_count == 2
    assert result["messages"][0]["role"] == "system"
    assert core.identity.name in result["messages"][0]["content"]
    core.close_session()
    assert core.get_session() is None or core.get_session().status == "closed"
    memory.close()


def test_core_responder_is_used(tmp_path):
    memory = AgentMemory(store_path=tmp_path / "agent.db")
    core = AgentCore(interface="cli", role="general", memory=memory)
    result = core.submit("ping", responder=lambda messages: f"eco con {len(messages)} mensajes")
    assert result["reply"] == "eco con 2 mensajes"  # system + user
    core.close_session()
    memory.close()
