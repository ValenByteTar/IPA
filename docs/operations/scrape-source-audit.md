# Scrape source audit

Effectiveness of each configured source in `configs/scrape_sites.yaml`, measured
by running the scraper site-by-site (`scripts/operations/scrape_sites_one_by_one.py`)
and comparing discovery against the live listing pages
(`scripts/operations/audit_scrape_discovery.py`). Audit date: 2026-09-13.

"Found" = article links discovered after pattern/date filters, before dedup.
A high `Skipped` count is healthy — it means the content was already ingested.

## Working as intended

| Site | Found | Strategy |
| --- | --- | --- |
| developer.nvidia.com/blog | 100 | RSS feed |
| vllm.ai/blog | 134 | URL pattern |
| thehackernews.com | 160 | Blogger pagination (10 pages) |
| microsoft.com/en-us/research/blog | 60 | pagination (5 pages) |
| offsec.com/blog | 56 | pagination (5 pages) |
| lablab.ai/ai-tutorials | 50 | sitemap |
| cobalt.io/blog | 40 | pagination (5 pages) |
| claude.com/blog | 23 | single page |
| blog.google/rss | 19 | RSS feed |
| emergentmind.com | 18 | single page |
| netspi.com blog / podcast / newsroom | 15 / 12 / 12 | single page each |
| anthropic.com news / research | 12 / 10 | single page each |
| blog.isecauditors.com/en | 9 | single page |
| cisa.gov | 8 | single page |

For the single-page sources the scraper retrieves every matching link the
listing exposes — the low count is the listing size, not lost content.

## Broken sources and root causes

### bleepingcomputer.com — 0 found (Cloudflare IP ban)

`requests` returns 403; the RSS feed returns 403; Playwright returns 403 in
both headless and headed mode. The block page is **Cloudflare error 1006**
("the owner of this website has banned your IP address") — an explicit IP ban,
not a JS challenge, so no engine or user-agent change can bypass it.

Fix: route through a different egress (proxy/VPN) or remove the source.

### developer.meta.com Muse posts — redirect to product landing

The `/ai/resources/blog/<slug>/` links (Muse, Muse Spark, Muse Code) redirect
to `developer.meta.com/ai/` — there is no article page to extract. Not
configurable; the viable Meta developer sources are the Facebook platform blog
and the Horizon blog (both added to the config on 2026-09-13).

## Fixes applied (2026-09-13)

- **CISA advisories**: pattern widened to `^/news-events/(alerts|cybersecurity-advisories)/`
  + `engine: playwright`. Verified: 8 links discovered (was 5), 3 new advisories scraped.
- **qwen.ai**: `days_back: 92 → 400` (the API list is stale; newest item
  2025-12-23). Verified: 18 links discovered (was 0), 15 articles scraped.
- **Meta**: added `developers.facebook.com/blog/` (3 posts, pattern
  `^/blog/post/\d{4}/`) and `developers.meta.com/horizon/blog/` (10 posts
  scraped on first run). The old `ai.meta.com/research/` source is kept.

### qwen.ai/blog — 0 found (stale source, not a scraper bug)

The JSON API (`/api/page_config?code=research.research-list`) responds 200 and
`_parse_json_api` parses all 60 items correctly. The newest item is dated
**2025-12-23**; with `days_back: 92` from 2026-09-13 the cutoff (2026-06-13)
filters out every item. The source has not published in ~9 months.

Fix applied: `days_back: 400` for this site. Verified: 18 links discovered,
15 articles scraped (SPA pages render via Playwright fallback; ~20s/article).

### bleepingcomputer.com — 0 found (Cloudflare IP ban)

`requests` returns 403; the RSS feed returns 403; Playwright returns 403 in
both headless and headed mode. The block page is **Cloudflare error 1006**
("the owner of this website has banned your IP address") — an explicit IP ban,
not a JS challenge, so no engine or user-agent change can bypass it.

Fix: route through a different egress (proxy/VPN) or remove the source.

### cisa.gov/news-events/cybersecurity-advisories — 5 found (stale URL pattern) — FIXED

CISA moved advisory items to `/news-events/alerts/YYYY/MM/DD/<slug>`. The
configured pattern `^/news-events/cybersecurity-advisories/[a-z0-9-]+` no longer
matched them, so only the single legacy-format link was picked up. The listing
also uses a "Show more" progressive-load control.

Fix applied: pattern `^/news-events/(alerts|cybersecurity-advisories)/` plus
`engine: playwright`. Verified: 8 links discovered (was 5), 3 new advisories
scraped on the first run after the fix.

### ai.meta.com/research — 6 found (listing structure)

The rendered page exposes only 3 real `/blog/<slug>/` links (the other ~50 are
navigation). The blog index `ai.meta.com/blog/` times out and
`/research/publications/` returns HTTP 500, so there is no richer listing to
switch to. Low yield is inherent to the site.

### New Meta sources (2026-09-13)

The hub `developers.meta.com/resources/blog/` aggregates posts from three
sub-properties. Evaluation:

- `developer.meta.com/ai/resources/blog/<slug>/` (Muse posts) — the listing
  works, but each "post" **redirects to the product landing**
  (`developer.meta.com/ai/`); there is no article body to extract. Not viable.
- `developers.facebook.com/blog/post/YYYY/MM/DD/<slug>/` — real articles,
  requests-extractable (2.9k chars, correct dates) on first evaluation.
  **Re-checked 2026-09-13 (pipeline run): the article body no longer
  renders** — Playwright + networkidle yields ~1.5KB of nav chrome only
  ("No se encontraron revisiones", menus, footer); `extract_article`
  returns "quality gate rejected: too little article text". The source
  produced 0 text documents and only orphan images, so it was DISABLED in
  `configs/scrape_sites.yaml`.
- `developers.meta.com/horizon/blog/<slug>/` — 10 posts, all scrapeable with
  plain requests (12k chars each). Added. Verified live 2026-09-13: 10/10
  scraped and ingested.

### Discovery audit — ~607 unregistered articles discarded per run (2026-09-13)

A discovery-only audit (listings/feeds/sitemaps + pagination, no article
fetches; `outputs/scrape-audit/discovery_unregistered.json`) showed the
scraper DISCOVERS far more than it registers, then throws the difference
away every run:

| site | found | registered | unregistered | cause |
|---|---|---|---|---|
| lablab.ai | 436 | 50 | 386 | max_articles cap |
| vllm.ai | 134 | 35 | 99 | date filter |
| microsoft | 60 | 15 | 45 | date filter |
| offsec | 56 | 13 | 43 | date filter |
| cobalt | 40 | 17 | 23 | date filter |
| claude.com | 23 | 16 | 7 | date filter |
| qwen | 18 | 15 | 3 | date filter |

Root cause (all four crawl paths): articles whose extracted date fell
outside `days_back` were counted as skipped but **never recorded**, so
every run re-fetched and re-discarded them. A 92-day window is therefore
NOT "exhausted" — the discovery finds ~600 more articles per run and the
date filter silently drops them.

Fixes applied to `web_scraper.py` (2026-09-13):

- Date-filtered articles are now recorded with `status='filtered'` plus the
  window width used (`filtered_days_back` column). Subsequent runs skip
  them via `ScrapeHistory.is_filtered()`, but a WIDER window re-evaluates
  them (stored window >= current window → skip).
- Extraction errors are recorded with `status='error'` in all paths (RSS
  path previously dropped them unrecorded). `is_scraped()` only matches
  `status='ok'`, so errors are still retried on later runs (self-healing).
- `ScrapeHistory.claim()` stale-claim comparison fixed: `claimed_at` is
  stored ISO-with-'T' and was compared against SQLite `datetime('now')`
  (space separator) — string comparison always False, so stale 'running'
  claims never expired and blocked their URLs permanently. The stale
  cutoff is now computed in Python as an ISO string.

Config changes (2026-09-13):

- **Window policy: the AGENT determines the scrape window per launch —
  dates are not hardcoded.** `run_ingestion` accepts `days_back` (global
  override; >30 also clears scrape history for re-discovery),
  `date_from`/`date_to` (exact range), or nothing (per-site baselines
  apply, no global override). The tool's previous hardcoded default (90)
  was removed. `sources.json` keeps the per-site baseline (92) — it is a
  fallback, not the decision mechanism.
- lablab.ai `max_articles` 50 → 0 (cap removed).
- arXiv deep archive added: cs.LG, cs.CL, cs.AI via the export.arxiv.org
  Atom API (200 most recent per category, baseline `days_back: 365`).
  arXiv rate-limits parallel bursts (HTTP 429) — `_parse_rss_feed` now
  retries 429s with jittered backoff (2 retries) so concurrent site
  workers desynchronize; a persistent 429 yields an empty result for
  that run and retries on the next ingestion.
- developers.facebook.com disabled (renders no article body, see above).

### Limit removal pass (2026-09-13, second)

First wide run (365d + clear-history) downloaded 1,319 files — the
ceiling of the previous discovery surface. Removed the remaining caps:

- `max_pages`: thehackernews 10→40, microsoft 5→20, offsec/cobalt/CISA
  5→15.
- `max_articles: 20` removed from developers.openai.com and
  platform.claude.com (sources.json overrides — these two sources live
  only there, not in the yaml).
- thehackernews `url_pattern` `^/2026/` → `^/20\d{2}/`: the hardcoded
  year silently dropped Sep–Dec 2025 articles inside the 365-day window.
- Sitemap discovery where the archive is deep (replaces feed/listing
  ceilings):
  - NVIDIA: `rss_feed` (~100-item feed cap) → `sitemap_url`
    `blog/wp-sitemap.xml`. Discovery within 365d: 4,795 total → 661
    after lastmod filtering.
  - Microsoft: added `sitemap_url` `en-us/research/sitemap.xml`
    (listing+pagination stays as fallback). 1,833 total → 69 within
    365d.
- `_parse_sitemap` now filters by `<lastmod>` during discovery when
  `days_back > 0` — avoids fetching years of articles just to discard
  them one by one at extraction time.

### Image leakage in Landing/web (2026-09-13)

16 `img_*.{png,jpg,webp}` files appeared under `developers-meta-com/` and
`developers-facebook-com/`. Root cause: the source-specific verification
scrapes were run without `--no-images`. The pipeline itself always passes
`--no-images` (every run in scraper.log reports `Images: 0`), so the
automated path was never affected. Fixes: orphan images deleted;
`/api/scraper/run` now also passes `--no-images --no-ocr` to match
`run_full_pipeline`.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts\operations\scrape_sites_one_by_one.py
.venv\Scripts\python.exe scripts\operations\audit_scrape_discovery.py
# outputs: outputs/scrape-audit/summary.json, discovery.json
```

### Pipeline premature termination killed the fast path watch (2026-09-14)

After the 365-day run, `pipeline_progress` reported "902 sin procesar" —
the landing sweep ran while the fast path watch was still indexing.

Root cause: `run_full_pipeline` waited a fixed `max_wait = 300`s for the
watch to catch up, and its completion check compared `bm25_docs >=
scraped_files` (documents vs files — never equal), so the loop always
timed out and `fp_proc.terminate()` killed indexing mid-flight.

Fix (structural, not a longer timeout):
- `run_fast_path.py --watch` gained `--idle-exit N` + `--idle-gate PATH`:
  once the gate file exists, the watch exits after N consecutive
  iterations with 0 new artifacts and 0 new chunks.
- The pipeline writes `Landing/web/.scraper_done` when the scraper exits
  (deleted at run start), passes `--idle-exit 3 --idle-gate <sentinel>`,
  and waits on `fp_proc.poll()` until natural exit. The 5-min kill is now
  a 6h safety net only.
- Effect: the landing sweep always runs after indexing is truly finished;
  no more "unprocessed" leftovers caused by premature termination.
