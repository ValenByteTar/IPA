"""Public entrypoint for the personal agent (thin wrapper over ipa.agent).

Surfaces are thin clients: they open sessions against the agent core and never
define personality nor keep agent state (DEC-002).

Usage:
    python scripts/cli/agent.py chat                     # interactive REPL
    python scripts/cli/agent.py chat -m "mensaje"        # single turn
    python scripts/cli/agent.py sessions                 # list sessions
    python scripts/cli/agent.py audit --session <id>     # export session to outputs/agent/exports/
    python scripts/cli/agent.py audit --recent 5         # export recent episodes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ipa.agent import AgentCore, AgentMemory, load_identity  # noqa: E402


def _export_session(memory: AgentMemory, session_id: str, exports_dir: Path) -> Path:
    session = memory.get_session(session_id)
    if session is None:
        raise ValueError(f"unknown session: {session_id}")
    episodes = memory.get_episodes(session_id, limit=1000)
    exports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "session": session.to_contract(),
        "episodes": [episode.to_contract() for episode in episodes],
    }
    out = exports_dir / f"{session_id.replace(':', '_')}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _load_star_provider(model_path: str):
    """Load the star model once and return (provider, responder, judge).

    Shared by chat --llm and research --llm: one VRAM load serves both the
    responder and the LLM judge (DEC-001: provider is replaceable).
    """
    from ipa.agent.provider_wiring import build_llm_judge, build_responder
    from ipa.providers.exl3_provider import create_star_provider

    provider = create_star_provider(
        model_path=model_path,
        interactive=True,
    )
    load_seconds = provider.load()
    print(f"[llm] {provider.model_id} cargado en {load_seconds:.1f}s")
    return provider, build_responder(provider), build_llm_judge(provider)


def cmd_chat(args: argparse.Namespace) -> int:
    responder = None
    provider = None
    if args.llm:
        provider, responder, _judge = _load_star_provider(args.model_path)
        print("[llm] respuestas generadas por el modelo estrella")
    core = AgentCore(interface="cli", role=args.role)
    print(f"[{core.identity.name}] sesión {core.ensure_session()} (rol {args.role}); 'exit' para cerrar.")
    try:
        if args.message:
            result = core.submit(args.message, responder=responder)
            print(result["reply"])
            core.close_session()
            return 0
        while True:
            try:
                line = input(f"{core.identity.user}> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line.lower() in {"exit", "quit", "salir"}:
                break
            result = core.submit(line, responder=responder)
            print(result["reply"])
    finally:
        core.close_session()
        if provider is not None:
            provider.unload()
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    with AgentMemory() as memory:
        for session in memory.list_sessions(limit=args.limit):
            print(f"{session.session_id}  [{session.status}]  role={session.role}  interface={session.interface}  episodes={session.episode_count}  last={session.last_active_at}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    exports_dir = Path("outputs") / "agent" / "exports"
    with AgentMemory() as memory:
        if args.session:
            out = _export_session(memory, args.session, exports_dir)
            print(f"export: {out}")
            return 0
        exported = []
        for session in memory.list_sessions(limit=args.recent):
            exported.append(_export_session(memory, session.session_id, exports_dir))
        for path in exported:
            print(f"export: {path}")
    return 0


def cmd_research(args: argparse.Namespace) -> int:
    """Agentic research flow: gap check → web search → judge → selective ingest."""
    from ipa.agent import ToolContext, assess_corpus_coverage, execute_research

    with AgentMemory() as memory:
        identity = load_identity()
        sid = memory.open_session(
            interface="cli", role=args.role,
            identity_hash=identity.identity_hash,
            title=f"research: {args.query[:50]}",
        )
        ep = memory.record_episode(
            sid, turn_role="user", content=args.query,
            identity_hash=identity.identity_hash,
        )

        ctx = ToolContext(memory=memory, corpus_dir=args.corpus)

        # Step 1: knowledge gap detection (deterministic scaffold)
        coverage = assess_corpus_coverage(args.query, ctx, min_hits=args.min_hits)
        print(f"[coverage] {coverage.reason}")
        if coverage.sufficient and not args.force:
            print("El corpus local es suficiente; no se requiere investigación web.")
            print("Usa --force para investigar de todos modos.")
            return 0

        # Step 2: agentic research (judge = heuristic unless --llm)
        judge = None
        provider = None
        if args.llm:
            provider, _responder, judge = _load_star_provider(args.model_path)
            print("[judge] LLMJudge con modelo estrella cargado")

        try:
            print(f"[research] investigando: {args.query}")
            call, result, research = execute_research(
                args.query, ctx,
                session_id=sid, episode_id=ep.episode_id,
                max_urls=args.max_urls, max_seconds=args.max_seconds,
                freshness=args.freshness,
                max_age_days=args.max_age_days,
                judge=judge,
                landing_dir=args.landing,
            )

            print(f"\nstatus: {result.status}")
            if result.error:
                print(f"error: {result.error}")
            print(f"busqueda: {research.search_results_count} resultados | "
                  f"scrapeados: {research.scraped_count} | "
                  f"rechazados: {research.rejected_count} | "
                  f"ingestados: {research.ingested_count}")
            for j in research.judgments:
                marker = "+" if j.verdict == "accept" else "-"
                print(f"  [{marker}] {j.stage:9s} {j.judge:22s} {j.url[:70]}")
                print(f"      {j.reason[:100]}")
            for h in research.retrieval_hits[:5]:
                print(f"  hit: {h['chunk_id'][:35]} score={h['score']}")
                print(f"       {h['text_preview'][:100]}...")
            return 0 if result.status == "completed" else 1
        finally:
            memory.close_session(sid)
            if provider is not None:
                provider.unload()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="interactive session with the agent")
    chat.add_argument("-m", "--message", default=None, help="single turn instead of REPL")
    chat.add_argument("--role", default="general", choices=["general", "tutor"])
    chat.add_argument("--llm", action="store_true", help="generate replies with the star model")
    chat.add_argument("--model-path", default="models/Qwen3.5-9B-exl3-3.0bpw")

    sessions = sub.add_parser("sessions", help="list recent sessions")
    sessions.add_argument("--limit", type=int, default=20)

    audit = sub.add_parser("audit", help="export sessions/episodes for review")
    audit.add_argument("--session", default=None, help="export one session")
    audit.add_argument("--recent", type=int, default=5, help="export N most recent sessions")

    research = sub.add_parser("research", help="agentic web research with LLM judgment")
    research.add_argument("query", help="research question")
    research.add_argument("--corpus", default="outputs/experiments/E12-corpus", help="corpus dir")
    research.add_argument("--max-urls", type=int, default=5)
    research.add_argument("--max-seconds", type=int, default=120)
    research.add_argument("--min-hits", type=int, default=3, help="gap threshold")
    research.add_argument("--force", action="store_true", help="research even if corpus suffices")
    research.add_argument("--llm", action="store_true", help="use the star model as judge")
    research.add_argument("--model-path", default="models/Qwen3.5-9B-exl3-3.0bpw")
    research.add_argument("--freshness", default="lenient", choices=["lenient", "strict"],
                          help="lenient: date as judge signal; strict: hard age cutoff")
    research.add_argument("--max-age-days", type=int, default=365,
                          help="age cutoff (hard reject only in strict mode)")
    research.add_argument("--landing", default="Landing/web")
    research.add_argument("--role", default="general", choices=["general", "tutor"])

    args = parser.parse_args()
    if args.command == "chat":
        return cmd_chat(args)
    if args.command == "sessions":
        return cmd_sessions(args)
    if args.command == "audit":
        return cmd_audit(args)
    if args.command == "research":
        return cmd_research(args)
    parser.print_usage()
    return 2


if __name__ == "__main__":
    sys.exit(main())
