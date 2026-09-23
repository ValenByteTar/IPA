---
id: EXP-006
category: experiment
status: accepted
created: 2026-09-11
updated: 2026-09-23
author: human
components: [agentic_runtime, document_store, reporter, ingestion]
tags: [promotion, provenance, idle-enrichment, decoupling, reporter-independent]
related: [DEC-003, PAT-001, PAT-005, EXP-003, PM-001]
supersedes: null
superseded_by: null
affects: ["src/ipa/reporter/**", "src/ipa/agentic/promotion_executor.py"]
---

# EXP-006 — Promoción independiente del Reporter

## Hipótesis

Separar la promoción física al corpus principal del Reporter (report approval) permite mantener el corpus principal actualizado continuamente, sin intervención humana para fuentes confiables, mientras se aplica un umbral de calidad a las fuentes de investigación del agente.

## Configuración

- **Corpus fuente**: `outputs/reporter/quality-check/optimized-llm-2026-09/corpus`
  - 736 documentos, 41,952 chunks, 41,952 vectores LanceDB
  - 701 con proveniencia registrada (666 configured_scrape, 35 agent_research)
- **Corpus destino**: `outputs/experiments/E12-corpus` (MAIN_CORPUS)
  - 103 documentos antes del experimento
- **Política**: `promotion_policy.py`
  - configured_scrape → auto-promote
  - agent_research → promotion_score >= 0.70
- **Idle enrichment**: `enrich_corpus_level1()` con `main_corpus_path=MAIN_CORPUS`
  - interests de config + user model (TutorStore)
  - historical embeddings del main corpus LanceDB
  - metadata de `document_sources` (source_domain, quality_score)

## Ejecución (2026-09-11)

### Paso 1: Backfill de proveniencia

Backfill desde `reporter.db` + `scrape_report.json` matcheando por `source_uri` del LandingZone.

- Reporter corpus: 701 documentos con proveniencia (95% de 736)
- Main corpus: 13 documentos con proveniencia (los 90 restantes son pre-scraper)

### Paso 2: Evaluación de política

Evaluación sobre los 701 documentos con proveniencia, usando decisiones de curación guardadas.

- **Encolados para promoción**: 550 (configured_scrape → auto-promote)
- **Rechazados**: 35 (agent_research con score < 0.70)
- **Ya en cola**: 116 (de un run anterior)

### Paso 3: Promoción física

`process_promotion_queue()` agrupó por `source_corpus` y copió documentos + chunks + vectores.

- **Procesados**: 666 documentos
- **Documentos nuevos**: 116 (los otros 550 ya existían — idempotente)
- **Chunks nuevos**: 2,786
- **Vectores nuevos**: 2,786

### Paso 4: Sincronización de índices

Auditoría post-promoción detectó desincronización:

| Índice | Antes | Después |
|---|---:|---:|
| DocumentStore chunks | 40,903 | 40,903 |
| BM25 chunks | 40,744 | 40,903 |
| LanceDB vectors | 40,970 (67 dup) | 40,903 |

Corrección: 159 chunks re-indexados en BM25, 67 vectores duplicados eliminados de LanceDB.

## Resultados

| Métrica | Antes | Después |
|---|---:|---:|
| Main corpus documentos | 103 | 689 |
| Main corpus chunks | ~38k | 40,903 |
| Main corpus vectores | ~38k | 40,903 |
| Main corpus con proveniencia | 0 | 666 |
| DS == BM25 == LanceDB | No | **Sí** |
| Reporter requerido para promoción | Sí | **No** |

## Lectura

- **La promoción continua funciona**: 666 documentos promovidos sin intervención humana. Los 35 documentos agent_research con score < 0.70 fueron correctamente rechazados.
- **Idempotencia verificada**: los 550 documentos ya presentes no se duplicaron (excepto 67 vectores LanceDB por un bug de check — ver PM-003).
- **Idle enrichment con métricas reales**: ahora recibe `source_domain`, `quality_score`, interests de config + user model, y historical embeddings del main corpus.
- **Reporter desacoplado**: la promoción no requiere `report.json` aprobado. El endpoint `/api/reports/review` ahora usa `promote_documents_to_main()` directamente.

## Limitaciones

- El backfill de proveniencia no recuperó los 90 documentos pre-scraper del main corpus (sin `scrape_report.json` ni `reporter.db` para ellos).
- El matching por `source_uri` puede fallar si los archivos se mueven entre Landing y Archive.
- La evaluación de política corre en el idle enrichment, que usa checkpoints — si todos los documentos ya están curados, la evaluación de promoción debe correr como paso separado (implementado).

## Estado

Aceptado: la hipótesis se confirma. La promoción independiente del Reporter es viable y mantiene el corpus principal actualizado. La política de proveniencia es auditable y los índices se mantienen sincronizados.

## Recomendación

- [x] Mantener como Experiment
- [ ] Congelar como Benchmark
- [x] Crear Decision (DEC-003)
- [ ] Proponer ADR
- [ ] Nothing
