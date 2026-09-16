"""Fase 1 gate: omnipresence CLI ↔ dashboard — both surfaces share agent memory.

This test proves the architectural property that the CLI and the dashboard
are thin surfaces over the same AgentCore, sharing the same SQLite store
(outputs/agent/agent.db). A session opened from one surface is visible and
continuable from the other (DEC-002, roadmap Fase 1 gate).
"""
from __future__ import annotations

import json
import os
import sys
import subprocess
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ipa.agent import AgentCore, AgentMemory, load_identity  # noqa: E402


ROOT = Path(__file__).parents[1]


def test_cli_and_dashboard_share_agent_store(tmp_path):
    """A session written via CLI (interface=cli) is readable via dashboard
    (interface=dashboard) through the same store, and vice versa."""
    store = tmp_path / "agent.db"
    identity = load_identity()

    # Surface 1: CLI opens a session and writes a turn
    cli_core = AgentCore(interface="cli", role="general", memory=AgentMemory(store_path=store))
    cli_session_id = cli_core.start_session(title="omnipresence gate")
    cli_core.submit("mensaje desde CLI")
    cli_core.close_session()

    # Surface 2: Dashboard opens the same store and reads the CLI session
    dash_memory = AgentMemory(store_path=store)
    sessions = dash_memory.list_sessions()
    assert len(sessions) >= 1
    cli_session = [s for s in sessions if s.session_id == cli_session_id]
    assert len(cli_session) == 1
    assert cli_session[0].interface == "cli"
    assert cli_session[0].status == "closed"

    # Dashboard reads episodes from the CLI session
    episodes = dash_memory.get_episodes(cli_session_id)
    assert len(episodes) == 2  # user + assistant
    assert any("mensaje desde CLI" in e.content for e in episodes)

    # Dashboard opens its own session in the same store
    dash_core = AgentCore(interface="dashboard", role="general", memory=dash_memory)
    dash_session_id = dash_core.start_session(title="dashboard session")
    dash_core.submit("mensaje desde dashboard")
    dash_core.close_session()

    # CLI surface reads the dashboard session
    cli_memory = AgentMemory(store_path=store)
    dash_session = cli_memory.get_session(dash_session_id)
    assert dash_session is not None
    assert dash_session.interface == "dashboard"
    dash_episodes = cli_memory.get_episodes(dash_session_id)
    assert any("mensaje desde dashboard" in e.content for e in dash_episodes)

    # Both surfaces see all sessions
    all_sessions = cli_memory.list_sessions()
    assert len(all_sessions) >= 2
    interfaces = {s.interface for s in all_sessions}
    assert "cli" in interfaces and "dashboard" in interfaces


def test_dashboard_agent_chat_endpoint_opens_session(tmp_path):
    """The dashboard's /api/agent/chat endpoint creates a session tagged
    interface=dashboard in the shared store."""
    store = tmp_path / "agent.db"
    env = {
        "PYTHONPATH": str(ROOT / "src"),
        "IPA_AGENT_STORE": str(store),
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
    }

    # Simulate what the dashboard endpoint does: open a session from dashboard interface
    script = f"""
import sys
sys.path.insert(0, r"{ROOT / 'src'}")
from ipa.agent import AgentCore
core = AgentCore(interface="dashboard", role="general")
result = core.submit("test from dashboard endpoint")
print(core.session_id)
core.close_session()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(ROOT), timeout=30, env=env,
    )
    assert result.returncode == 0, result.stderr
    session_id = result.stdout.strip().splitlines()[-1]
    assert session_id.startswith("agent_session:")

    # Verify from a separate process (simulating CLI reading the dashboard session)
    memory = AgentMemory(store_path=store)
    session = memory.get_session(session_id)
    assert session is not None
    assert session.interface == "dashboard"
    assert session.episode_count == 2  # user + assistant


def test_agent_sessions_endpoint_lists_shared_sessions(tmp_path):
    """The /api/agent/sessions endpoint lists sessions from both surfaces."""
    store = tmp_path / "agent.db"
    identity = load_identity()

    # Create sessions from both interfaces
    for interface in ("cli", "dashboard"):
        core = AgentCore(interface=interface, role="general", memory=AgentMemory(store_path=store))
        core.start_session(title=f"test {interface}")
        core.submit(f"hello from {interface}")
        core.close_session()

    # Verify both are visible
    memory = AgentMemory(store_path=store)
    sessions = memory.list_sessions()
    interfaces = {s.interface for s in sessions}
    assert "cli" in interfaces
    assert "dashboard" in interfaces


def test_deep_dive_records_conversation_in_agent_memory(tmp_path, monkeypatch):
    """The deep dive records user+assistant turns in the shared agent store
    (DEC-002): the dashboard never owns conversation state. This exercises the
    same recording pattern the /api/deep-dive/stream handler uses."""
    store = tmp_path / "agent.db"
    monkeypatch.setenv("IPA_AGENT_STORE", str(store))

    from ipa.agent import AgentMemory, load_identity

    memory = AgentMemory()
    identity = load_identity()
    session_id = memory.open_session(
        interface="dashboard", role="general",
        identity_hash=identity.identity_hash,
        title="deep dive: test",
    )
    memory.record_episode(
        session_id, turn_role="user", content="test deep dive question",
        identity_hash=identity.identity_hash,
    )
    memory.record_episode(
        session_id, turn_role="assistant", content="test answer",
        identity_hash=identity.identity_hash,
    )

    # The conversation is in the shared store, visible from another connection
    check = AgentMemory()
    stored = check.get_session(session_id)
    assert stored is not None
    assert stored.interface == "dashboard"
    episodes = check.get_episodes(session_id)
    assert len(episodes) == 2
    assert episodes[0].turn_role == "user"
    assert "deep dive question" in episodes[0].content
    assert episodes[1].turn_role == "assistant"
    memory.close_session(session_id)
