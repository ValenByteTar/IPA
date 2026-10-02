"""Devin lifecycle hook: work-permit enforcement + EKS governing injection.

Events handled (hook_event_name in the stdin payload):
- PreToolUse (edit/write/apply_patch/notebook_edit): blocks edits to paths
  under a foreign exclusive permit, and injects the EKS records governing
  the path as additionalContext — once per record per session.
- SessionStart: lists active permits and any permit the previous session
  closed without an EKS harvest, then prunes stale per-session state.
- SessionEnd: closes this session's permits, recording an `unharvested`
  marker for those closed without `--eks-draft` (TTL still covers hard kills).
- Stop: blocks once per session asking for the closeout before finishing.
- PostCompaction: re-injects this session's permits — compaction may have
  dropped them.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("DEVIN_PROJECT_DIR") or
                    Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from eks_repository import EKSRepository  # noqa: E402
from work_permits import PermitStore  # noqa: E402

WRITE_TOOLS = {"edit", "write", "apply_patch", "notebook_edit"}

# Zonas blandas: el riesgo real de conflicto es de código. En docs/ y
# knowledge/ un exclusive ajeno advierte pero no bloquea (el STOP solo
# donde importa) — lección de PM-009 en el port desde RIAPP.
SOFT_ZONE_PREFIXES = ("docs/", "knowledge/")

SEEN_MAX_AGE_S = 7 * 24 * 3600


def _heartbeat_mine(store: PermitStore, session: str) -> None:
    """Renueva el lease de los permits de ESTA sesión en cada tool call.

    Mismo patrón que vram_lock.renew (PAT-007): mientras la sesión trabaja,
    su permit no caduca (un batch nocturno >ttl ya no expira en vuelo);
    cuando la sesión muere, deja de latir y pid_alive/TTL lo matan solos.
    """
    if not session:
        return
    for permit in store.active():
        if permit.session == session:
            store.heartbeat(permit.permit_id)


def _emit(payload: dict) -> None:
    # Windows cp1252 no codifica →/— de los títulos EKS: forzar UTF-8 en
    # stdout (bug real: la inyección de records moría en UnicodeEncodeError
    # y el hook salía 1 — el .seen se escribía pero la UI nunca veía el
    # additionalContext).
    out = json.dumps(payload, ensure_ascii=False)
    try:
        sys.stdout.buffer.write(out.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()
    except AttributeError:
        print(out)


def _rel(path: str) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except (ValueError, OSError):
        return path.replace("\\", "/").lstrip("/")


def _permits_dir() -> Path:
    return PROJECT_ROOT / "outputs" / "devin" / "permits"


def _seen_path(session: str) -> Path:
    return _permits_dir() / f".seen-{session}.json"


def _unharvested_path(session: str) -> Path:
    return _permits_dir() / f".unharvested-{session}.json"


def _reminded_path(session: str) -> Path:
    return _permits_dir() / f".stop-reminded-{session}.json"


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path: Path, payload) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _load_seen(session: str) -> list[str]:
    value = _read_json(_seen_path(session), [])
    return value if isinstance(value, list) else []


def _save_seen(session: str, seen: list[str]) -> None:
    _write_json(_seen_path(session), seen)


def _pending_unharvested() -> list[Path]:
    directory = _permits_dir()
    if not directory.is_dir():
        return []
    return sorted(directory.glob(".unharvested-*.json"))


def _prune_session_state() -> int:
    """Drop per-session seen files older than SEEN_MAX_AGE_S."""
    directory = _permits_dir()
    if not directory.is_dir():
        return 0
    removed = 0
    cutoff = time.time() - SEEN_MAX_AGE_S
    for path in directory.glob(".seen-*.json"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue
    return removed


def on_pre_tool_use(data: dict) -> int:
    tool_input = data.get("tool_input") or {}
    path = tool_input.get("file_path") or tool_input.get("path")
    if not path:
        return 0
    session = data.get("session_id") or ""
    rel = _rel(path)
    store = PermitStore.default(PROJECT_ROOT)

    _heartbeat_mine(store, session)

    check = store.check_path(rel, session=session)
    if check["blocked"]:
        owners = ", ".join(f"{p['permit_id']} ({p['session']})" for p in check["permits"])
        if rel.startswith(SOFT_ZONE_PREFIXES):
            _emit({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "additionalContext": (
                        f"Aviso: '{rel}' está bajo un permit exclusivo ajeno "
                        f"({owners}) — zona blanda (docs/knowledge): podés "
                        "editar, pero coordiná para no pisar trabajo en curso."),
                },
            })
            return 0
        _emit({
            "decision": "block",
            "reason": (f"'{rel}' is under an active exclusive work permit held by "
                       f"{owners}. Coordinate with the user or pick a different scope. "
                       "Si tu edición es acotada, adquirí el permit con scope a "
                       "nivel de archivo (p. ej. --scope "
                       "'src/ipa/agentic/promotion_executor.py')."),
        })
        return 0

    repo = EKSRepository(PROJECT_ROOT / "knowledge")
    governing = repo.governing([rel])
    if not governing:
        return 0
    seen = _load_seen(session)
    fresh = [item for item in governing if item["id"] not in seen]
    if not fresh:
        return 0
    seen.extend(item["id"] for item in fresh)
    _save_seen(session, seen)
    lines = "\n".join(
        f"  {item['id']} [{item['status']}] {item['title']} — governs {item['affects_matched']}"
        for item in fresh)
    _emit({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": (
                f"EKS records governing '{rel}' (read before editing):\n{lines}"),
        },
    })
    return 0


def on_session_start(data: dict) -> int:
    store = PermitStore.default(PROJECT_ROOT)
    session = data.get("session_id")
    lines: list[str] = []
    if session:
        # El guard matchea edits contra ESTE id. Adquirir con una etiqueta
        # de rol ("A", "V1") en su lugar hace que tu propio permit te
        # bloquee — mismo hallazgo que PM-002 en RIAPP.
        lines.append(
            f"Your Devin session_id: {session} — pass it as --session in "
            "permit.py acquire so the guard recognizes your own edits.")

    active = store.active()
    if active:
        lines.append("Active work permits in this workspace (do not edit inside "
                     "their scopes):")
        lines.extend(
            f"  {p.permit_id} [{p.type}] session={p.session} scope={p.scope} task={p.task!r}"
            for p in active)

    unharvested = _pending_unharvested()
    if unharvested:
        lines.append("")
        lines.append("Permits closed WITHOUT an EKS harvest (previous session "
                     "ended before closeout) — if that session produced durable "
                     "knowledge, log it with the experiment-logging skill:")
        for path in unharvested:
            for entry in _read_json(path, []):
                lines.append(f"  {entry.get('permit_id')} scope={entry.get('scope')} "
                             f"task={entry.get('task')!r}")

    _prune_session_state()
    if not lines:
        return 0
    _emit({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "\n".join(lines),
        },
    })
    for path in unharvested:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    return 0


def on_session_end(data: dict) -> int:
    """Release this session's permits, flagging any that close unharvested.

    The scope must be released when the session dies, but closing silently
    would lose the knowledge the closeout is meant to capture — so permits
    closed here without an `eks_draft` leave a marker the next SessionStart
    reports.
    """
    session = data.get("session_id")
    if not session:
        return 0
    store = PermitStore.default(PROJECT_ROOT)
    mine = [p for p in store.active() if p.session == session]
    if not mine:
        return 0
    unharvested = []
    for permit in mine:
        store.close(permit.permit_id, notes="session ended without closeout")
        if not permit.eks_draft:
            unharvested.append({
                "permit_id": permit.permit_id,
                "scope": permit.scope,
                "task": permit.task,
            })
    if unharvested:
        _write_json(_unharvested_path(session), unharvested)
    return 0


def on_stop(data: dict) -> int:
    """Block once per session (never loop) so the closeout runs."""
    session = data.get("session_id")
    if not session or data.get("stop_hook_active"):
        return 0
    reminded = _reminded_path(session)
    if reminded.exists():
        return 0
    store = PermitStore.default(PROJECT_ROOT)
    mine = [p for p in store.active() if p.session == session]
    if not mine:
        return 0
    _write_json(reminded, {
        "at": time.time(),
        "permits": [p.permit_id for p in mine],
    })
    lines = "\n".join(f"  {p.permit_id} scope={p.scope}" for p in mine)
    _emit({
        "decision": "block",
        "reason": (
            "You still hold open work permits — run the session-closeout skill "
            "or close them before finishing:\n" + lines +
            "\n  permit.py close --permit <id> --notes \"...\" --eks-draft <ID>"),
    })
    return 0


def on_post_compaction(data: dict) -> int:
    """Re-inject this session's permits — compaction may have dropped them."""
    session = data.get("session_id")
    if not session:
        return 0
    store = PermitStore.default(PROJECT_ROOT)
    mine = [p for p in store.active() if p.session == session]
    lines = [
        f"Reminder after compaction — your Devin session_id: {session} "
        "(use it as --session in permit.py acquire).",
    ]
    if mine:
        lines.append("Your active work permits:")
        lines.extend(
            f"  {p.permit_id} [{p.type}] scope={p.scope} task={p.task!r}" for p in mine)
    _emit({
        "hookSpecificOutput": {
            "hookEventName": "PostCompaction",
            "additionalContext": "\n".join(lines),
        },
    })
    return 0


def main() -> int:
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return 0
    event = data.get("hook_event_name")
    if event == "PreToolUse" and data.get("tool_name") in WRITE_TOOLS:
        return on_pre_tool_use(data)
    if event == "SessionStart":
        return on_session_start(data)
    if event == "SessionEnd":
        return on_session_end(data)
    if event == "Stop":
        return on_stop(data)
    if event == "PostCompaction":
        return on_post_compaction(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
