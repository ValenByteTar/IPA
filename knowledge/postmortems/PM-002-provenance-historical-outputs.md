---
id: PM-002
category: postmortem
status: accepted
created: 2026-09-06
updated: 2026-09-23
author: human
components: [provenance, benchmarks, eks]
tags: [provenance, raw-outputs, cleanup, reproducibility, benchmarks]
related: [BM-001, BM-002, BM-003, BM-004, BM-005, EXP-001, EXP-002, RES-001]
supersedes: null
superseded_by: null
affects: ["src/ipa/storage/**", "src/ipa/ingestion/provenance.py"]
evidence: ["src/ipa/ingestion/provenance.py", "tests/test_ingest_metadata.py"]
---

# PM-002 — Pérdida de artefactos crudos de benchmarks históricos (E0-E11)

## Impacto

Los benchmarks aceptados BM-001..BM-005 y los experimentos EXP-001/EXP-002 referencian artefactos crudos bajo `outputs/experiments/E0-E11/` (`benchmark_report.json`, reportes de retrieval, comparaciones adaptativas) que ya no existen en el workspace. Solo permanece `outputs/experiments/E12-corpus/` (corpus de producción). Las conclusiones registradas en EKS siguen en pie, pero la evidencia cruda local de esas corridas se perdió en limpiezas anteriores al EKS.

## Línea de tiempo

1. E3/E5/E6/E7/E9/E10 se ejecutaron y sus reportes se congelaron en el decision log (2026-08-27).
2. El workspace se reorganizó (MACRO-ORDEN) y `outputs/experiments/` se redujo a `E12-corpus/`.
3. El EKS se creó el 2026-09-05 transcribiendo conclusiones, no artefactos.
4. La revisión de provenance del 2026-09-06 detectó que ningún `benchmark_report.json` histórico permanece local.

## Causa raíz

Los outputs de experimentos se trataron como espacio temporal regenerable sin registrar antes una manifestación de provenance (hash + copia de artefactos crudos) ni una declaración formal de dónde vive la evidencia.

## Estado resultante y política

- **BM-001..BM-005, EXP-001, EXP-002**: conclusiones retenidas; artefactos crudos históricos no disponibles localmente. Cada runner es reproducible (`scripts/benchmarks/`) pero requiere reconstruir el corpus histórico de 166k chunks; la regeneración no es bit-a-bit equivalente (corpus y modelos actuales difieren).
- **EXP-004 / BM-006 / DEC-001**: la evidencia cruda vive en el repositorio fuente `C:\Users\Valen\Desktop\Proyectos\small-model-deliberation` (`engine_benchmark/results/`, 58 runs con run_id) — verificados el 2026-09-06.
- **EXP-003 (E13-agentic) y EXP-005 (E8)**: artefactos crudos locales completos y validados contra contrato.

## Prevención

1. Todo experimento nuevo escribe su reporte conforme a `contracts/experiment_report.schema.json` con `output_hash`/`output_artifacts` verificables (ya aplicado en E8 y E13).
2. Antes de limpiar `outputs/experiments/<id>/`, registrar en EKS el hash del artefacto primario o moverlo al repositorio fuente.
3. Los benchmarks aceptados deben declarar explícitamente dónde vive su evidencia cruda (local, repositorio fuente, o no disponible).

## Lección reutilizable

La provenance es parte del experimento, no un adjunto posterior: un benchmark sin artefactos localizables es una afirmación, no evidencia.
