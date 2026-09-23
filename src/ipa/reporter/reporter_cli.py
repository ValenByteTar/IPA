"""Generate a domain-agnostic periodic Reporter report.

Canonical implementation; entrypoint is a thin wrapper
(``scripts/cli/run_reporter.py``).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ipa.reporter.reporter_config import ReporterConfig
from ipa.reporter.reporter_pipeline import ReporterPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Landing/web", help="Reporter input directory")
    parser.add_argument("--config", default="configs/reporter.yaml")
    parser.add_argument("--output", default=None, help="Output directory; defaults to outputs/reporter/<corpus>/<period>")
    parser.add_argument("--scrape-report", default=None, help="Optional scrape_report.json")
    parser.add_argument("--previous-report", default=None)
    parser.add_argument("--embeddings", action="store_true", help="Use BGE-M3 embeddings for clustering")
    parser.add_argument("--llm", action="store_true", help="Use the local Qwen3.5-9B EXL3 provider for curation and labels")
    parser.add_argument("--interactive", action="store_true", help="Use EXL3 batch_size=1 instead of batch_size=6")
    parser.add_argument("--main-corpus", default=None, help="Path to main corpus (to reuse LanceDB vectors instead of re-embedding)")
    parser.add_argument("--period-start", default=None, help="Override period start (ISO format)")
    parser.add_argument("--period-end", default=None, help="Override period end (ISO format)")
    parser.add_argument("--period-label", default=None, help="Override period label")
    args = parser.parse_args()
    config = ReporterConfig.from_yaml(args.config)
    # Apply period overrides if provided
    if args.period_start or args.period_end or args.period_label:
        from ipa.reporter.reporter_config import ReporterPeriodConfig
        config = ReporterConfig(
            corpus_id=config.corpus_id,
            period=ReporterPeriodConfig(
                start=args.period_start or config.period.start,
                end=args.period_end or config.period.end,
                label=args.period_label or config.period.label,
            ),
            min_documents=config.min_documents,
            allow_singleton_topics=config.allow_singleton_topics,
            max_hierarchy_depth=config.max_hierarchy_depth,
            similarity_threshold=config.similarity_threshold,
            weights=config.weights,
            quality_threshold=config.quality_threshold,
            interests=config.interests,
            research_enabled=config.research_enabled,
            research_require_approval=config.research_require_approval,
            research_budget=config.research_budget,
        )
    progress_file = Path("outputs/web_dashboard/reporter_progress.json")

    def report_progress(percent: int, stage: str, detail: str = "") -> None:
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.write_text(json.dumps({"status": "running", "percent": percent, "stage": stage, "detail": detail, "updated_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False), encoding="utf-8")

    report_progress(0, "iniciando")
    output = Path(args.output or f"outputs/reporter/{config.corpus_id}/{config.period.label}")
    embedding = None
    llm_provider = None
    if args.embeddings:
        from ipa.indexes.embedding_adapter import EmbeddingAdapter
        embedding = EmbeddingAdapter(show_progress=False)
    if args.llm:
        from ipa.providers.exl3_provider import create_star_provider
        llm_provider = create_star_provider(interactive=args.interactive)
        llm_provider.load()
    try:
        main_corpus = Path(args.main_corpus) if args.main_corpus else None
        with ReporterPipeline(config, output, embedding_adapter=embedding, llm_provider=llm_provider, main_corpus=main_corpus) as pipeline:
            report = pipeline.run(args.input, args.scrape_report, args.previous_report, args.embeddings, report_progress)
        progress_file.write_text(json.dumps({"status": "completed", "percent": 100, "stage": "completado", "detail": "", "updated_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        progress_file.write_text(json.dumps({"status": "failed", "percent": 100, "stage": "error", "detail": "", "error": str(exc), "updated_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False), encoding="utf-8")
        raise
    finally:
        if llm_provider is not None:
            llm_provider.unload()
        if embedding is not None:
            embedding.close()
    print(f"Reporter: {len(report['categories'])} categories, {sum(report['curation_summary'].values())} decisions")
    print(f"JSON: {output / 'report.json'}")
    print(f"Markdown: {output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
