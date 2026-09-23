---
id: DEC-007
category: decision
status: accepted
created: 2026-09-13
updated: 2026-09-13
author: human
components: [ingestion, landing_zone, dashboard, promotion_executor, provenance]
tags: [landing, transit, archive, lifecycle, human-confirmation, provenance, sweep, deletion-policy]
related: [DEC-003, PAT-001, PAT-003, PAT-005, PM-003]
supersedes: null
superseded_by: null
affects: ["Landing/**", "src/ipa/ingestion/**"]
---

# DEC-007 — Ciclo de vida Landing → Transit → Archive con confirmación humana

## Contexto

Antes de esta decisión, `Landing/` acumulaba indefinidamente los artefactos ya
procesados: el pipeline indexaba el contenido pero ningún mecanismo movía los
archivos fuente fuera de la zona de ingesta. Además, la relación entre el
corpus de staging (reporter) y el corpus principal era opaca:

1. **Landing sin política de salida**: 800+ archivos procesados seguían en
   `Landing/web` después de estar indexados. Landing era almacenamiento
   permanente de facto, no zona de paso.
2. **Staging acumulaba copias promovidas**: `promote_documents_to_main()`
   copia documentos, chunks, vectores y provenance al corpus principal, pero
   dejaba intacta la copia en staging — las métricas mostraban 50,710 chunks
   en staging y 51,060 en main siendo casi el mismo contenido dos veces.
3. **Documentos varados por proveniencia faltante**: 110 documentos del scraper
   sin fila en `document_sources` eran descartados por `promotion_policy` como
   "unknown provenance" para siempre — el backfill existente
   (`backfill_from_scrape_report`) no estaba wireado al pipeline y el
   `scrape_report.json` del último run estaba vacío.
4. **Sin zona de confirmación humana**: no existía un estado físico para
   "procesado, esperando decisión humana sobre su ingreso al corpus principal".

## Decisión

Tres zonas físicas con clasificación automática y señal humana explícita:

```text
Landing/ (solo intake)
    ├─ aprobado: doc vivo en main corpus (E12)      → Archive/
    ├─ pendiente: en staging pero no en main,
    │   promotion_queue pending, o curation
    │   review_status pending/changes_requested     → Transit/
    ├─ rechazado: failed en todo registro, o
    │   review_status='rejected' humano             → eliminado
    └─ sin registrar / en vuelo                     → se queda
```

Reglas operativas:

- **El sweep** (`src/ipa/ingestion/landing_sweep.py`) clasifica cada artefacto
  registrado usando `landing.db`, membresía en `document_store.db`, y las
  decisiones humanas de **ambos** stores (`topic_clusters.db` curation +
  `reporter.db` document_decisions — el review del dashboard escribe en el
  segundo). El rechazo humano es la señal negativa más fuerte: gana sobre pending.
- **Transit se re-escanea en cada sweep por hash de contenido**: promovido a
  main → Archive; rechazado → delete; sigue pendiente → permanece. El registry
  conserva el `source_uri` original; la identificación es por contenido.
- **El sweep corre automáticamente** en cuatro puntos donde el estado cambia:
  fin de `run_full_pipeline`, tras procesar la promotion queue en el idle
  worker, y tras cada acción de review del dashboard
  (`/api/reports/review`, `/api/decisions/review`, `/api/promotions/process`).
  Manual: `scripts/operations/sweep_landing.py [--dry-run]`.
- **Purga post-promoción** (`purge_promoted_from_source` en
  `promotion_executor.py`): tras copiar a main, la copia de staging se
  tombstonea (documents+chunks), se saca del índice BM25 FTS, se borran sus
  vectores de LanceDB y sus `embedding_jobs` stale. Solo se purgan documentos
  confirmados vivos en main.
- **Provenance self-heal**: `backfill_from_landing_registry()` marca artefactos
  bajo `Landing/web/` sin `document_sources` como `configured_scrape`
  (los de agent research se auto-registran al fetch, así que provenance
  faltante implica scraper); extrae la URL real de la línea `Source:` embebida
  en el texto. Wireado al pipeline y al idle enrichment antes de evaluar policy.
- **Archivos operativos intocables**: `scrape_history.db`,
  `scrape_report.json`, `*.db`, ocultos y `*.pending_delete` nunca se mueven ni
  eliminan.
- **Dedup por contenido**: mover a Transit/Archive con hash — el scraper puede
  re-descargar el mismo contenido bajo otro nombre sin crear duplicados.

## Consecuencias

- **Gana**: Landing queda vacío tras cada ingesta (solo estado operativo);
  Transit refleja exactamente el conjunto pendiente de confirmación humana;
  las métricas de staging/main dejan de duplicarse; el ciclo converge sin
  intervención manual (scrape → staging → policy/humano → Transit → Archive).
- **Coste**: el sweep hashea todos los archivos de Landing+Transit por corrida
  (O(archivos) de IO — aceptable a escala actual, optimizable con cache mtime);
  lee dos stores de decisiones; la purga post-promoción añade escrituras al
  final de cada promoción.
- **Seguridad de datos**: los deletes requieren señal explícita (`failed` en
  todos los registros, o `review_status='rejected'` humano); el contenido
  desconocido nunca se elimina; los locks de Windows se manejan con
  copy+retry+`.pending_delete` limpiado en la corrida siguiente.
- **Reversibilidad**: la clasificación es un módulo independiente
  (`landing_sweep.py`); las zonas son directorios físicos movibles; ninguna
  frontera arquitectónica cambia — el DocumentStore sigue siendo canónico y
  los índices derivados.

## Evidencia

- `src/ipa/ingestion/landing_sweep.py` — clasificación de 3 vías + 3 pasadas
  (registry, hash en Landing, re-evaluación de Transit por hash)
- `scripts/operations/sweep_landing.py` — CLI operativo con `--dry-run`
- `src/ipa/agentic/promotion_executor.py` — `purge_promoted_from_source()`
- `src/ipa/ingestion/provenance.py` — `backfill_from_landing_registry()`
- `tests/test_landing_sweep.py` — 7 tests del ciclo completo (fixtures sintéticos)
- Estado final verificado: Landing 0, Transit 0, Archive 918,
  main 858 docs / 51,060 chunks / 51,060 vectores, staging 1 doc / 11 chunks
- `docs/operations/landing-and-archive.md` — política documentada

## Por qué no es ADR

Igual que DEC-003: es una decisión local y reversible que no altera fronteras
arquitectónicas. El DocumentStore sigue siendo la autoridad canónica; los
índices siguen siendo derivados y rebuildables; Landing/Transit/Archive son
zonas físicas del mismo sistema, no nuevos subsistemas. La purga post-promoción
refuerza la frontera existente (main canónico, staging derivado) en lugar de
crear una nueva.
