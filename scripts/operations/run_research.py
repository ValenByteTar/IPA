"""Standalone research executor — runs research_topic as a subprocess.

Launched by tool_research_topic when the dashboard needs web research to
survive a dashboard restart. Writes progress to
outputs/web_dashboard/research_progress.json.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PROGRESS = ROOT / "outputs" / "web_dashboard" / "research_progress.json"


def write_progress(payload: dict) -> None:
    # Merge con el estado inicial: el dashboard escribe session_id/notified
    # antes de lanzar este subprocess — si se pisa el archivo, el watcher
    # pierde la sesión destino y nunca entrega el episodio de cierre.
    existing = {}
    if PROGRESS.exists():
        try:
            existing = json.loads(PROGRESS.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    merged = {**existing, **payload}
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: run_research.py <query> [max_urls] [max_seconds]")
        sys.exit(1)
    query = sys.argv[1]
    max_urls = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    max_seconds = int(sys.argv[3]) if len(sys.argv) > 3 else 180

    sys.path.insert(0, str(ROOT / "src"))

    from ipa.agent.agent_memory import AgentMemory
    from ipa.agent.agent_tools import ToolContext
    from ipa.agent.research_executor import execute_research
    from ipa.agent.system_tools import _main_corpus_dir

    corpus_dir = _main_corpus_dir()
    if corpus_dir is None:
        write_progress({
            "status": "failed", "query": query,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error": "no corpus with document_store.db found",
        })
        sys.exit(1)

    memory = AgentMemory()
    sid = memory.open_session(
        interface="system_tool", role="general",
        identity_hash="system", title=f"research: {query[:50]}",
    )
    ep = memory.record_episode(
        sid, turn_role="user", content=query, identity_hash="system",
    )
    ctx = ToolContext(memory=memory, corpus_dir=str(corpus_dir))
    try:
        _call, result, research = execute_research(
            query, ctx,
            session_id=sid, episode_id=ep.episode_id,
            max_urls=max_urls, max_seconds=max_seconds,
        )
        write_progress({
            "status": "done" if result.status == "completed" else "failed",
            "query": query,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "result": {
                "search_results": research.search_results_count,
                "scraped": research.scraped_count,
                "ingested": research.ingested_count,
                "rejected": research.rejected_count,
            },
            "error": result.error,
        })
    except Exception as exc:
        write_progress({
            "status": "failed", "query": query,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error": str(exc),
        })
    finally:
        memory.close_session(sid)
        ctx.close()
        memory.close()


if __name__ == "__main__":
    main()
