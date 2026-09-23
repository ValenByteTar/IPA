---
id: PAT-006
category: pattern
status: accepted
created: 2026-09-23
updated: 2026-09-23
author: agent
components: [acquisition, ingestion, landing_zone, provenance]
tags: [scraper, rss, playwright, smart-ocr, arxiv, dedup, scrape-history, waf]
related: [PAT-002, PAT-003, DEC-007]
supersedes: null
superseded_by: null
affects: ["src/ipa/acquisition/**", "scripts/cli/run_web_scrape.py"]
evidence: ["scripts/cli/run_web_scrape.py"]
author_model: swe-2
---

# PAT-006 — Adquisición web determinística (RSS-first, engine auto, SmartOCR)

## Problema

Adquirir material heterogéneo (blogs con HTML semántico, SPAs React, sitios
detrás de WAF/CDN, feeds) sin bloquear el fast path, sin re-descargar lo ya
visto y sin bajar ruido (iconos, READMEs de repos, páginas de error).

## Solución

- **Config declarativa por sitio** (`ScrapeSite` en
  `configs/scrape_sites.yaml`): `article_selector`, `url_pattern`,
  `exclude_paths`, `allowed_domains`, `rss_feed`, `trust_article_dates`.
  Las heurísticas de extracción de links solo actúan como fallback cuando el
  sitio no tiene config — lo determinístico gana siempre.
- **RSS/Atom primero**: cuando hay feed, se parsea (`<item>`/`<entry>`,
  `days_back`, `url_pattern`) en vez de crawlear el listing HTML. Orden de
  magnitud más confiable (OpenAI: 1155 items → 20 artículos).
- **Engine `auto`**: `requests` primero; si 0 links, fallback a Playwright
  (lazy Chromium). Meta AI (SPA) solo es scrapeable vía Playwright.
- **Descarga de documentos**: PDF/DOCX/etc. a Landing con validación de
  content-type (saltar error pages HTML) y dedup por URL; links a
  `arxiv.org/abs|pdf` resuelven el PDF y deduplican abs+pdf del mismo paper.
- **SmartOCR**: `ImageClassifier` (edge density, color, tamaño) decide si una
  imagen porta texto antes de gastar OCR; íconos <2KB, fotos y gigantes
  (>20MB, downscale) se descartan. En PDFs, OCR solo en páginas con baja
  densidad de texto.
- **`ScrapeHistory`**: SQLite `url PRIMARY KEY`; lookup O(log n) medido
  0.046 ms @1M URLs — re-scrape imposible por construcción.
- **Filtros de ruido**: `REPO_DOMAINS` (github/gitlab/...) y `.md` fuera de
  `DOCUMENT_EXTENSIONS` (los md linkeados son READMEs, no papers).
- `trust_article_dates=false` para sitios donde trafilatura toma fechas de
  footer (Anthropic: 0→24 artículos).

## Trade-offs

- La config por sitio requiere curación manual (conocido: Qwen.ai sin `<a>`
  es inscrapeable; BleepingComputer bloquea por Cloudflare).
- Playwright es pesado; se justifica solo cuando `auto` lo invoca.
- `trust_article_dates=false` desactiva el filtro temporal para ese sitio.

## Ejemplos locales

- `src/ipa/acquisition/` (web_scraper, ocr_adapter), `configs/scrape_sites.yaml`.
- Evidencia original: filas 2026-08-27/28 de `docs/DECISION_LOG.md` (pre-EKS).
- Tests: `tests/test_web_scraper.py` (heurísticas, ImageClassifier, RSS).
