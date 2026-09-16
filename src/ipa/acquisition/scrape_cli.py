"""Web scraping + OCR CLI over configured sites.

Canonical implementation; entrypoints are thin wrappers
(``scripts/cli/run_web_scrape.py``).

Usage:
    # Scrape all sites from configs/scrape_sites.yaml
    python scripts/cli/run_web_scrape.py

    # Scrape a single URL with explicit pattern
    python scripts/cli/run_web_scrape.py --url https://developer.nvidia.com/blog \\
        --url-pattern "^/blog/[^/]+/$" \\
        --exclude "/blog/category/" "/blog/tag/" "/blog/recent-posts/"

    # Scrape a single URL with CSS selector
    python scripts/cli/run_web_scrape.py --url https://thehackernews.com/ \\
        --selector "article a[href]"

    # Scrape a single URL with heuristics (no pattern/selector)
    python scripts/cli/run_web_scrape.py --url https://example.com/news --days-back 2

    # Skip OCR (just scrape text + images)
    python scripts/cli/run_web_scrape.py --no-ocr

    # Specify languages for OCR
    python scripts/cli/run_web_scrape.py --ocr-langs en es

    # Use a custom config file
    python scripts/cli/run_web_scrape.py --config configs/scrape_sites.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from ipa import WebScraper, ScrapeSite, OCRAdapter

# Lock for thread-safe printing
_print_lock = threading.Lock()


def load_sites_from_yaml(config_path: str | Path) -> list[ScrapeSite]:
    """Load site configurations from a YAML file."""
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    sites = []
    for s in data.get("sites", []):
        sites.append(ScrapeSite(
            url=s["url"],
            days_back=s.get("days_back", 2),
            max_articles=s.get("max_articles", 20),
            delay_seconds=s.get("delay_seconds", 1.0),
            article_selector=s.get("article_selector"),
            url_pattern=s.get("url_pattern"),
            exclude_paths=s.get("exclude_paths", []),
            allowed_domains=s.get("allowed_domains", []),
            rss_feed=s.get("rss_feed"),
            sitemap_url=s.get("sitemap_url"),
            trust_article_dates=s.get("trust_article_dates", True),
            engine=s.get("engine"),
            json_api_url=s.get("json_api_url"),
            json_api_id_field=s.get("json_api_id_field", "id"),
            json_api_url_template=s.get("json_api_url_template"),
            paginate=s.get("paginate", False),
            max_pages=s.get("max_pages", 50),
            paginate_url_template=s.get("paginate_url_template"),
        ))
    return sites


def main() -> None:
    parser = argparse.ArgumentParser(description="Web scraper + OCR for intelligence gathering")
    parser.add_argument("--config", default="configs/scrape_sites.yaml",
                        help="YAML config file with site definitions")
    parser.add_argument("--output", default="Landing", help="Output directory for scraped articles (default: Landing)")
    parser.add_argument("--url", default=None, help="Scrape a single URL (overrides config file)")
    parser.add_argument("--selector", default=None, help="CSS selector for article links (with --url)")
    parser.add_argument("--url-pattern", default=None, help="Regex for article URL paths (with --url)")
    parser.add_argument("--exclude", nargs="*", default=None, help="URL substrings to exclude (with --url)")
    parser.add_argument("--allowed-domains", nargs="*", default=None, help="Additional domains to allow (with --url)")
    parser.add_argument("--days-back", type=int, default=None, help="Only articles from last N days")
    parser.add_argument("--max-articles", type=int, default=None, help="Max articles per site")
    parser.add_argument("--delay", type=float, default=None, help="Delay between requests (seconds)")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR on images")
    parser.add_argument("--ocr-langs", nargs="+", default=["en", "es"], help="OCR languages")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU for OCR")
    parser.add_argument("--no-images", action="store_true", help="Skip image download")
    parser.add_argument("--engine", choices=["requests", "playwright", "auto"],
                        default="auto", help="Fetch engine: requests (fast), playwright (JS-rendered), auto (try requests then playwright)")
    parser.add_argument("--headed", action="store_true", help="Show browser window (Playwright, for debugging)")
    parser.add_argument("--report", default=None, help="Save JSON report to this path")
    parser.add_argument("--clear-history", action="store_true",
                        help="Clear scrape_history.db before scraping (re-scrape everything). "
                             "Use when you want to re-discover articles that were already scraped.")
    args = parser.parse_args()

    # Build site list
    if args.url:
        sites = [ScrapeSite(
            url=args.url,
            days_back=args.days_back or 2,
            max_articles=args.max_articles or 20,
            delay_seconds=args.delay or 1.0,
            article_selector=args.selector,
            url_pattern=args.url_pattern,
            exclude_paths=args.exclude or [],
            allowed_domains=args.allowed_domains or [],
        )]
    else:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            sys.exit(1)
        sites = load_sites_from_yaml(config_path)
        # Allow CLI overrides for days_back/max_articles/delay
        if args.days_back is not None:
            sites = [ScrapeSite(**{**s.__dict__, "days_back": args.days_back}) for s in sites]
        if args.max_articles is not None:
            sites = [ScrapeSite(**{**s.__dict__, "max_articles": args.max_articles}) for s in sites]
        if args.delay is not None:
            sites = [ScrapeSite(**{**s.__dict__, "delay_seconds": args.delay}) for s in sites]

    if not sites:
        print("No sites to scrape. Edit configs/scrape_sites.yaml or use --url.")
        sys.exit(1)

    print(f"Scraping {len(sites)} site(s) with {min(5, len(sites))} parallel workers")
    for s in sites:
        mode = "selector" if s.article_selector else ("pattern" if s.url_pattern else "heuristic")
        print(f"  {s.url}  [{mode}, days_back={s.days_back}, max={s.max_articles}]")
    print(f"Output: {args.output}")
    print(f"Engine: {args.engine}")
    print(f"OCR: {'disabled' if args.no_ocr else f'enabled (langs={args.ocr_langs}, gpu={not args.no_gpu})'}")
    print()

    # Initialize OCR adapter (lazy load)
    ocr = None
    if not args.no_ocr:
        ocr = OCRAdapter(
            languages=args.ocr_langs,
            gpu=not args.no_gpu,
            paragraph=False,
        )

    # Scrape
    output_dir = Path(args.output)
    # Clear history if requested (re-scrape everything).
    # Limpiar las tablas en vez de borrar el archivo: si lo borramos,
    # 5 workers paralelos intentan recrearlo simultáneamente y SQLite
    # tira "database is locked" (race condition del 2026-09-08).
    if args.clear_history:
        history_db = output_dir / "scrape_history.db"
        if history_db.exists():
            import sqlite3
            conn = sqlite3.connect(str(history_db), timeout=30)
            try:
                conn.execute("DELETE FROM scraped_urls")
                conn.execute("DELETE FROM scrape_jobs")
                conn.commit()
                print(f"  [scraper] Cleared scrape history: {history_db}")
            except Exception:
                # Si las tablas no existen, borrar y recrear
                conn.close()
                history_db.unlink()
                print(f"  [scraper] Cleared scrape history (file): {history_db}")
            else:
                conn.close()
    all_results = []
    total_articles = 0
    total_images = 0
    total_documents = 0
    total_ocr_texts = 0
    t0 = time.monotonic()

    def scrape_one_site(site: ScrapeSite) -> list[dict]:
        """Scrape a single site in its own WebScraper instance. Thread-safe."""
        site_results = []
        with WebScraper(
            output_dir=output_dir,
            download_images=not args.no_images,
            engine=args.engine,
            playwright_headless=not args.headed,
            history_db=str(output_dir / "scrape_history.db"),
        ) as scraper:
            with _print_lock:
                print(f"--- {site.url} ---")
            summary = scraper.scrape_site(site)
            with _print_lock:
                print(f"  [{site.url}] Found: {summary.total_articles_found} articles")
                print(f"  [{site.url}] Scraped: {summary.articles_scraped}, Skipped: {summary.articles_skipped}")
                if summary.errors:
                    print(f"  [{site.url}] Errors: {len(summary.errors)}")
                    for err in summary.errors[:2]:
                        print(f"    {err}")

            for result in summary.results:
                # OCR on images (intelligent mode — skip decorative images)
                if result.image_paths and ocr:
                    with _print_lock:
                        print(f"  [{site.url}] Smart OCR on {len(result.image_paths)} images for: {result.title[:60]}")
                    ocr_results = ocr.extract_texts_smart(result.image_paths)
                    for ocr_res in ocr_results:
                        if ocr_res.success and ocr_res.text.strip():
                            result.ocr_texts.append(ocr_res.text)

                # Save article
                filepath = scraper.save_article(result)
                with _print_lock:
                    doc_info = f", {len(result.document_paths)} docs" if result.document_paths else ""
                    print(f"  [{site.url}] Saved: {filepath.name} ({len(result.text)} chars, {len(result.image_paths)} imgs{doc_info})")

                site_results.append({
                    "url": result.url,
                    "title": result.title,
                    "date": result.date,
                    "text_length": len(result.text),
                    "image_count": len(result.image_paths),
                    "document_count": len(result.document_paths),
                    "document_paths": result.document_paths,
                    "ocr_text_count": len(result.ocr_texts),
                    "saved_to": str(filepath),
                    "canonical_url": result.canonical_url,
                    "content_hash": result.content_hash,
                    "quality_score": result.quality_score,
                    "metadata": result.metadata,
                    "error": result.error,
                    "elapsed_seconds": round(result.elapsed_seconds, 3),
                })
        return site_results

    # Run sites in parallel (5 workers)
    max_workers = min(5, len(sites))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_site = {executor.submit(scrape_one_site, site): site for site in sites}
        for future in as_completed(future_to_site):
            site = future_to_site[future]
            try:
                site_results = future.result()
                all_results.extend(site_results)
                total_articles += len(site_results)
                for r in site_results:
                    total_images += r["image_count"]
                    total_documents += r["document_count"]
                    total_ocr_texts += r["ocr_text_count"]
            except Exception as e:
                with _print_lock:
                    print(f"  [{site.url}] FATAL: {e}")

    elapsed = time.monotonic() - t0

    # Summary
    print()
    print("=" * 60)
    print(f"Scrape complete in {elapsed:.1f}s")
    print(f"  Articles: {total_articles}")
    print(f"  Images:   {total_images}")
    print(f"  Documents: {total_documents}")
    print(f"  OCR texts: {total_ocr_texts}")
    print(f"  Output:   {output_dir}")

    # Report
    report = {
        "sites": [{"url": s.url, "days_back": s.days_back,
                    "article_selector": s.article_selector,
                    "url_pattern": s.url_pattern,
                    "exclude_paths": s.exclude_paths,
                    "allowed_domains": s.allowed_domains} for s in sites],
        "ocr_enabled": not args.no_ocr,
        "ocr_languages": args.ocr_langs if not args.no_ocr else [],
        "total_articles": total_articles,
        "total_images": total_images,
        "total_documents": total_documents,
        "total_ocr_texts": total_ocr_texts,
        "elapsed_seconds": round(elapsed, 3),
        "results": all_results,
    }

    report_path = args.report or str(output_dir / "scrape_report.json")
    Path(report_path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Report:   {report_path}")

    if ocr:
        ocr.close()


if __name__ == "__main__":
    main()
