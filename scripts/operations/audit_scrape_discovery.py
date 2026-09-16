"""Audit discovery effectiveness per configured site.

For each site: fetch the listing page (same UA as the scraper), count links
matching the config's url_pattern after excludes, and compare with the
scraper's discovered count. Also flags JS-rendered pages (few static links).

Usage:
    .venv/Scripts/python.exe scripts/operations/audit_scrape_discovery.py [--only substr]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
UA = "RES023-Research-Bot/1.0"


def _links(html: str, base: str) -> list[str]:
    hrefs = re.findall(r'<a[^>]+href=["\']([^"\']+)["\']', html, re.IGNORECASE)
    return [urljoin(base, h) for h in hrefs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "scrape_sites.yaml"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--audit-summary", default=str(ROOT / "outputs" / "scrape-audit" / "summary.json"))
    args = ap.parse_args()

    sites = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")).get("sites", [])
    if args.only:
        wanted = [s.strip() for s in args.only.split(",")]
        sites = [s for s in sites if any(w in s["url"] for w in wanted)]

    scraper_found: dict[str, int | None] = {}
    if Path(args.audit_summary).exists():
        for r in json.loads(Path(args.audit_summary).read_text(encoding="utf-8")):
            scraper_found[r["url"]] = r.get("found")

    s = requests.Session()
    s.headers["User-Agent"] = UA
    rows = []
    for site in sites:
        url = site["url"]
        pattern = site.get("url_pattern")
        excludes = site.get("exclude_paths") or []
        strategy = ("json_api" if site.get("json_api_url") else
                    "rss" if site.get("rss_feed") else
                    "sitemap" if site.get("sitemap_url") else "html")
        try:
            resp = s.get(url, timeout=25, allow_redirects=True)
            html, status = resp.text, resp.status_code
        except requests.RequestException as exc:
            rows.append({"url": url, "strategy": strategy, "http": f"ERROR {exc}",
                         "static_links": None, "pattern_matches": None,
                         "scraper_found": scraper_found.get(url)})
            continue
        links = _links(html, url)
        rx = re.compile(pattern) if pattern else None
        matches = []
        for link in links:
            if any(ex in link for ex in excludes):
                continue
            if rx and not rx.match(urlparse(link).path):
                continue
            if rx or (urlparse(link).netloc and urlparse(link).netloc in urlparse(url).netloc):
                matches.append(link)
        rows.append({
            "url": url, "strategy": strategy, "http": status,
            "static_links": len(links), "pattern_matches": len(set(matches)),
            "scraper_found": scraper_found.get(url),
        })

    print(f"{'strategy':9} {'http':5} {'links':>6} {'matches':>7} {'scraper':>7}  site")
    for r in rows:
        print(f"{r['strategy']:9} {str(r['http']):5} {str(r['static_links']):>6} "
              f"{str(r['pattern_matches']):>7} {str(r['scraper_found']):>7}  {r['url']}")
    out = ROOT / "outputs" / "scrape-audit" / "discovery.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
