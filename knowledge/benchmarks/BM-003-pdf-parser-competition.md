---
id: BM-003
category: benchmark
status: accepted
created: 2026-09-05
updated: 2026-09-06
author: human
components: [parsing, ingestion, performance, provenance]
tags: [pdf, pymupdf, docling, unstructured, parser]
related: [PAT-001]
supersedes: null
superseded_by: null
---

# BM-003 â€” Competencia de parsers PDF

## Objetivo

Comparar PyMuPDF, Docling y Unstructured para el fast path de PDFs born-digital.

## Entorno

- Dataset: 20 PDFs balanceados.
- Artefacto: `outputs/experiments/E3/benchmark_report.json`.
- Métricas: tiempo, texto extraído y errores.

## Resultados

- PyMuPDF: 0,21 s por PDF, 1,2% más texto que Docling y 0 errores.
- Docling: aproximadamente 33,4 s por PDF.
- Unstructured fast: aproximadamente 35% menos texto y menor detección de páginas.
- Unstructured hi_res: aproximadamente 26 veces más lento que el baseline sin mejora equivalente.

## Conclusión

PyMuPDF es el baseline operativo del fast path para PDFs born-digital. Docling y Unstructured permanecen como adapters para workloads donde layout, tablas u OCR justifiquen el coste adicional.

## Alcance

Este benchmark no selecciona un parser universal para PDFs escaneados, multimodales o con tablas complejas.

## Provenance

Artefacto histórico (`outputs/experiments/E3/benchmark_report.json`) no disponible localmente (ver PM-002). Conclusiones retenidas; runner reproducible en `scripts/benchmarks/run_parser_benchmark.py`. Relacionado: [PM-002].
