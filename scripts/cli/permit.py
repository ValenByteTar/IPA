"""Dev-time work permits for parallel agent sessions (Permit-to-Work).

Examples:
    permit.py acquire --session devin-abc --scope "src/ipa/agentic/**" \
        --task "refactor promotion" --type exclusive
    permit.py check --scope "src/ipa/**"
    permit.py list
    permit.py heartbeat --permit PW-20260923-01
    permit.py close --permit PW-20260923-01 --notes "done" --eks-draft PAT-010
    permit.py close-session --stdin   # hook: reads session_id from stdin
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from work_permits import PERMIT_TYPES, PermitStore  # noqa: E402


def _store() -> PermitStore:
    return PermitStore.default(Path(os.environ.get("DEVIN_PROJECT_DIR") or PROJECT_ROOT))


def _print_permit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    acquire = sub.add_parser("acquire", help="Request a work permit for a path scope")
    acquire.add_argument("--session", required=True)
    acquire.add_argument("--scope", nargs="+", required=True)
    acquire.add_argument("--task", default="")
    acquire.add_argument("--type", choices=PERMIT_TYPES, default="exclusive")
    acquire.add_argument("--ttl", type=int, default=7200)
    acquire.add_argument("--force", action="store_true",
                         help="Issue despite overlapping exclusive permits")

    check = sub.add_parser("check", help="Report permits conflicting with a scope")
    check.add_argument("--scope", nargs="+", required=True)

    sub.add_parser("list", help="List permits (active first)")
    sub.add_parser("prune", help="Report and drop expired permit files")

    hb = sub.add_parser("heartbeat", help="Refresh a permit's TTL")
    hb.add_argument("--permit", required=True)

    close = sub.add_parser("close", help="Close a permit (closeout)")
    close.add_argument("--permit", required=True)
    close.add_argument("--notes", default=None)
    close.add_argument("--eks-draft", dest="eks_draft", default=None)

    cs = sub.add_parser("close-session", help="Close all permits of a session")
    cs.add_argument("--session", default=None)
    cs.add_argument("--stdin", action="store_true",
                    help="Read session_id from a Devin hook JSON payload on stdin")
    cs.add_argument("--reason", default="session ended")

    args = parser.parse_args()
    store = _store()

    if args.command == "acquire":
        permit, payload = store.acquire(
            args.session, args.scope, args.task,
            type=args.type, ttl_s=args.ttl, force=args.force)
        _print_permit(payload)
        return 0 if permit else 2

    if args.command == "check":
        conflicts = store.conflicts(args.scope)
        _print_permit({
            "scope": args.scope,
            "conflicts": [p.to_dict() for p in conflicts],
            "clear": not conflicts,
        })
        return 0 if not conflicts else 1

    if args.command == "list":
        permits = store.all()
        _print_permit({
            "permits": [p.to_dict() for p in permits],
            "active": [p.permit_id for p in permits if p.alive()],
        })
        return 0

    if args.command == "prune":
        removed = 0
        for permit in store.all():
            if not permit.alive() and permit.status == "active":
                permit.status = "expired"
                store._save(permit)
                removed += 1
        _print_permit({"expired": removed})
        return 0

    if args.command == "heartbeat":
        permit = store.heartbeat(args.permit)
        _print_permit({"permit": permit.to_dict() if permit else None,
                       "refreshed": permit is not None})
        return 0 if permit else 1

    if args.command == "close":
        permit = store.close(args.permit, notes=args.notes, eks_draft=args.eks_draft)
        _print_permit({"permit": permit.to_dict() if permit else None,
                       "closed": permit is not None})
        return 0 if permit else 1

    if args.command == "close-session":
        session = args.session
        if args.stdin:
            try:
                payload = json.loads(sys.stdin.read() or "{}")
                session = session or payload.get("session_id")
            except ValueError:
                session = session
        if not session:
            print("no session id (flag or stdin)", file=sys.stderr)
            return 1
        _print_permit({"closed": store.close_session(session, reason=args.reason)})
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
