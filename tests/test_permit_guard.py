"""Tests for the Devin lifecycle hook that enforces work permits.

The hook is loaded by path (it is a script, not a package module) and pointed
at a temporary project root through `DEVIN_PROJECT_DIR`, which is the same
variable Devin sets when it runs it. `tools/` is put on `sys.path` so the
hook's `from eks_repository import ...` resolves to the real modules.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_PATH = REPO_ROOT / "scripts" / "hooks" / "permit_guard.py"

GOVERNING_RECORD = """---
id: PAT-001
category: pattern
status: accepted
created: 2026-09-05
updated: 2026-09-05
author: test
components: [agentic_runtime]
tags: [t]
related: []
supersedes: null
superseded_by: null
affects: ["src/ipa/agentic/**"]
---

# PAT-001 — Governing pattern
"""


def _load_hook() -> object:
    spec = importlib.util.spec_from_file_location("permit_guard_under_test", HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved
    return module


def _payload(capsys) -> dict:
    out = capsys.readouterr().out.strip()
    assert out, "hook produced no output"
    return json.loads(out)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "proj"
    (root / "knowledge" / "patterns").mkdir(parents=True)
    (root / "outputs" / "devin" / "permits").mkdir(parents=True)
    monkeypatch.setenv("DEVIN_PROJECT_DIR", str(root))
    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools"))
    return root, _load_hook()


def _permit_store(root: Path):
    from tools.work_permits import PermitStore
    return PermitStore(root / "outputs" / "devin" / "permits")


def _acquire(root: Path, session: str, scope: list[str], **kwargs):
    permit, payload = _permit_store(root).acquire(session, scope, "task", **kwargs)
    assert payload["issued"], payload
    return permit


# --- PreToolUse -----------------------------------------------------------

def test_pre_tool_use_blocks_a_foreign_exclusive_scope(project, capsys):
    root, hook = project
    _acquire(root, "s1", ["src/ipa/**"])
    code = hook.on_pre_tool_use({
        "tool_name": "edit",
        "tool_input": {"file_path": str(root / "src" / "ipa" / "x.py")},
        "session_id": "s2",
    })
    assert code == 0
    payload = _payload(capsys)
    assert payload["decision"] == "block"
    assert "PW-" in payload["reason"]


def test_pre_tool_use_allows_the_permit_holder(project, capsys):
    root, hook = project
    _acquire(root, "s1", ["src/ipa/**"])
    hook.on_pre_tool_use({
        "tool_name": "edit",
        "tool_input": {"file_path": str(root / "src" / "ipa" / "x.py")},
        "session_id": "s1",
    })
    assert capsys.readouterr().out.strip() == ""


def test_pre_tool_use_ignores_paths_outside_every_scope(project, capsys):
    root, hook = project
    _acquire(root, "s1", ["src/ipa/**"])
    hook.on_pre_tool_use({
        "tool_name": "edit",
        "tool_input": {"file_path": str(root / "docs" / "x.md")},
        "session_id": "s2",
    })
    assert capsys.readouterr().out.strip() == ""


def test_governing_records_are_injected_once_per_session(project, capsys):
    root, hook = project
    (root / "knowledge" / "patterns" / "PAT-001.md").write_text(
        GOVERNING_RECORD, encoding="utf-8")
    event = {
        "tool_name": "edit",
        "tool_input": {"file_path": str(root / "src" / "ipa" / "agentic" / "x.py")},
        "session_id": "s1",
    }
    hook.on_pre_tool_use(event)
    first = _payload(capsys)
    assert "PAT-001" in first["hookSpecificOutput"]["additionalContext"]
    assert first["hookSpecificOutput"]["hookEventName"] == "PreToolUse"

    hook.on_pre_tool_use(event)
    assert capsys.readouterr().out.strip() == "", "the record was injected twice"


# --- SessionStart / SessionEnd -------------------------------------------

def test_session_start_lists_active_permits(project, capsys):
    root, hook = project
    permit = _acquire(root, "s1", ["src/ipa/**"])
    hook.on_session_start({})
    payload = _payload(capsys)
    assert permit.permit_id in payload["hookSpecificOutput"]["additionalContext"]


def test_session_start_is_silent_without_permits(project, capsys):
    _root, hook = project
    hook.on_session_start({})
    assert capsys.readouterr().out.strip() == ""


def test_session_end_closes_and_flags_unharvested(project, capsys):
    root, hook = project
    permit = _acquire(root, "s1", ["src/ipa/**"])

    hook.on_session_end({"session_id": "s1"})
    assert capsys.readouterr().out.strip() == ""
    assert _permit_store(root).get(permit.permit_id).status == "closed"

    marker = root / "outputs" / "devin" / "permits" / ".unharvested-s1.json"
    assert marker.is_file()
    assert [entry["permit_id"] for entry in json.loads(marker.read_text("utf-8"))] \
        == [permit.permit_id]

    # The next session reports it and clears the marker (report once).
    hook.on_session_start({})
    context = _payload(capsys)["hookSpecificOutput"]["additionalContext"]
    assert permit.permit_id in context and "WITHOUT an EKS harvest" in context
    assert not marker.exists()


def test_session_end_does_not_flag_a_harvested_permit(project, capsys):
    root, hook = project
    permit = _acquire(root, "s1", ["src/ipa/**"])
    _permit_store(root).close(permit.permit_id, notes="done", eks_draft="PAT-001")

    hook.on_session_end({"session_id": "s1"})
    assert capsys.readouterr().out.strip() == ""
    assert not (root / "outputs" / "devin" / "permits" / ".unharvested-s1.json").exists()


def test_session_end_only_touches_its_own_session(project, capsys):
    root, hook = project
    mine = _acquire(root, "s1", ["src/ipa/**"])
    other = _acquire(root, "s2", ["docs/**"])
    hook.on_session_end({"session_id": "s1"})
    store = _permit_store(root)
    assert store.get(mine.permit_id).status == "closed"
    assert store.get(other.permit_id).status == "active"


# --- Stop -----------------------------------------------------------------

def test_stop_blocks_once_then_stays_quiet(project, capsys):
    root, hook = project
    permit = _acquire(root, "s1", ["src/ipa/**"])

    hook.on_stop({"session_id": "s1"})
    payload = _payload(capsys)
    assert payload["decision"] == "block"
    assert permit.permit_id in payload["reason"]

    hook.on_stop({"session_id": "s1"})
    assert capsys.readouterr().out.strip() == "", "Stop blocked more than once"


def test_stop_never_blocks_when_the_hook_is_already_active(project, capsys):
    root, hook = project
    _acquire(root, "s1", ["src/ipa/**"])
    hook.on_stop({"session_id": "s1", "stop_hook_active": True})
    assert capsys.readouterr().out.strip() == ""


def test_stop_without_permits_does_not_block(project, capsys):
    _root, hook = project
    hook.on_stop({"session_id": "s1"})
    assert capsys.readouterr().out.strip() == ""


# --- PostCompaction / maintenance ----------------------------------------

def test_post_compaction_reinjects_the_sessions_permits(project, capsys):
    root, hook = project
    permit = _acquire(root, "s1", ["src/ipa/**"])
    hook.on_post_compaction({"session_id": "s1"})
    payload = _payload(capsys)
    assert payload["hookSpecificOutput"]["hookEventName"] == "PostCompaction"
    assert permit.permit_id in payload["hookSpecificOutput"]["additionalContext"]


def test_session_start_prunes_stale_seen_files(project, capsys):
    root, hook = project
    seen = root / "outputs" / "devin" / "permits" / ".seen-old.json"
    seen.write_text("[]", encoding="utf-8")
    stale = 1_000_000_000  # 2001 → clearly outside the 7-day window
    os.utime(seen, (stale, stale))
    hook.on_session_start({})
    assert not seen.exists()


def test_main_routes_every_event(project, capsys, monkeypatch):
    root, hook = project
    permit = _acquire(root, "s1", ["src/ipa/**"])

    def run(payload: dict):
        monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(payload)))
        return hook.main()

    assert run({"hook_event_name": "SessionStart"}) == 0
    assert permit.permit_id in _payload(capsys)["hookSpecificOutput"]["additionalContext"]

    assert run({"hook_event_name": "PreToolUse", "tool_name": "read",
                "tool_input": {"file_path": "x"}, "session_id": "s1"}) == 0
    assert capsys.readouterr().out.strip() == ""

    assert run({"hook_event_name": "unknown"}) == 0
    assert capsys.readouterr().out.strip() == ""
