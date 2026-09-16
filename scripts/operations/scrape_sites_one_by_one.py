"""Run the scraper site-by-site and aggregate discovery effectiveness.

For each site in configs/scrape_sites.yaml: write a one-site temp config, run
run_web_scrape.py with a JSON report, and collect
(total_articles_found, articles_scraped, articles_skipped, errors).

Usage:
    .venv/Scripts/python.exe scripts/operations/scrape_sites_one_by_one.py [--days-back N]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "scrape-audit"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "scrape_sites.yaml"))
    ap.add_argument("--output", default=str(ROOT / "Landing" / "web"))
    ap.add_argument("--days-back", type=int, default=None,
                    help="Override days_back for every site")
    ap.add_argument("--only", default=None,
                    help="Comma-separated site url substrings to run")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sites = config.get("sites", [])
    if args.only:
        wanted = [s.strip() for s in args.only.split(",")]
        sites = [s for s in sites if any(w in s["url"] for w in wanted)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reports: list[dict] = []
    for i, site in enumerate(sites, 1):
        if args.days_back is not None:
            site = {**site, "days_back": args.days_back}
        slug = site["url"].replace("https://", "").replace("http://", "")
        slug = "".join(c if c.isalnum() else "-" for c in slug).strip("-")[:60]
        report_path = OUT_DIR / f"{slug}.json"
        cfg_path = OUT_DIR / f"config-{slug}.yaml"

        # Drop stale report so we never read a previous run's file.
        report_path.unlink(missing_ok=True)
        cfg_path.write_text(
            yaml.safe_dump({"sites": [site]}, allow_unicode=True), encoding="utf-8")

        cmd = [
            sys.executable, "-u", str(ROOT / "scripts" / "cli" / "run_web_scrape.py"),
            "--config", str(cfg_path),
            "--output", args.output,
            "--no-images", "--no-ocr",
            "--report", str(report_path),
        ]
        print(f"[{i}/{len(sites)}] {site['url']}", flush=True)
        t0 = time.time()
        stdout = stderr = ""
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=900)
            rc = proc.returncode
            stdout, stderr = proc.stdout or "", proc.stderr or ""
            tail = stdout[-400:]
        except subprocess.TimeoutExpired as exc:
            rc, tail = -1, "TIMEOUT"
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        elapsed = time.time() - t0

        entry = {
            "url": site["url"], "rc": rc, "elapsed": round(elapsed, 1),
            "found": None, "scraped": None, "skipped": None,
            "documents": None, "errors": [], "stdout_tail": tail,
        }
        # The JSON report only counts *scraped* results; discovery counts
        # (Found/Skipped, including dedup) live in stdout.
        import re
        m = re.search(r"Found:\s*(\d+)\s*articles", stdout)
        if m:
            entry["found"] = int(m.group(1))
        m = re.search(r"Scraped:\s*(\d+),\s*Skipped:\s*(\d+)", stdout)
        if m:
            entry["scraped"], entry["skipped"] = int(m.group(1)), int(m.group(2))
        if report_path.exists():
            try:
                data = json.loads(report_path.read_text(encoding="utf-8"))
                results = data.get("results", data if isinstance(data, list) else [])
                entry["documents"] = data.get("total_documents")
                if results:
                    r = results[0]
                    entry.setdefault("scraped", r.get("articles_scraped"))
                    entry["errors"] = (r.get("errors") or [])[:5]
            except (ValueError, KeyError) as exc:
                entry["errors"] = [f"report parse: {exc}"]
        # Site-level failures only appear on stderr/stdout
        for line in stdout.splitlines() + stderr.splitlines():
            low = line.lower()
            if any(k in low for k in ("failed", "error", "blocked", "forbidden",
                                      "timeout", "captcha", "cloudflare", "403", "429")):
                if len(entry["errors"]) < 5 and line.strip() not in entry["errors"]:
                    entry["errors"].append(line.strip()[:200])
        reports.append(entry)
        print(f"    rc={rc} found={entry['found']} scraped={entry['scraped']} "
              f"skipped={entry['skipped']} docs={entry['documents']} "
              f"errors={len(entry['errors'])} ({elapsed:.0f}s)", flush=True)

    summary = OUT_DIR / "summary.json"
    summary.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsummary -> {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
