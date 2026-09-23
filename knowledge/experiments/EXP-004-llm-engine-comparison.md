---
id: EXP-004
category: experiment
status: accepted
created: 2026-09-05
updated: 2026-09-23
author: human
components: [tutor, providers, evaluation, performance]
tags: [qwen3.5-9b, exllamav3, exl3, ollama, mtp, vram, deliberation]
related: [RES-004, BM-006, DEC-001]
supersedes: null
superseded_by: null
evidence: ["scripts/operations/test_exl3_provider.py", "src/ipa/providers/exl3_provider.py"]
affects: ["src/ipa/providers/**", "scripts/operations/test_exl3_provider.py"]
---

# EXP-004 — Comparación de engine y modelo LLM para Tutor

## Hipótesis

Qwen3.5-9B EXL3 con ExLlamaV3 puede ofrecer una combinación superior de calidad y rendimiento frente a Qwen3.5-9B GGUF ejecutado con Ollama, siempre que el uso de VRAM, contexto y estabilidad sea viable.

## Motivación

El Tutor Agent depende de explicaciones, ejercicios, assessment, structured output, abstención y tool use. Elegir el engine por tokens/s únicamente podría producir un sistema rápido pero pedagógicamente inferior.

## Procedencia verificada

La evidencia cruda fue localizada y verificada (2026-09-06) en el repositorio externo `C:\Users\Valen\Desktop\Proyectos\small-model-deliberation`:

- Runs bajo `engine_benchmark/results/<run_id>/` con `config.json`, `environment.json`, `metrics.json`, `raw_outputs.jsonl`, `parsed_results.jsonl`, `failures.jsonl` y `summary.md` por corrida.
- Documentación: `README2.md` (seguimiento del engine benchmark) y `docs/06-llm-benchmark-summary.md` (9 LLMs).
- Datasets: `engine_benchmark/benchmarks/pedagogical_v1.json` (48 casos x 15 tareas) y `benchmarks/semantic_assessment_v2.json` (55 casos, 10 categorías).

Runs relevantes: `full-qwen9b-3.0`, `full-qwen9b-3.5`, `full-qwen9b-4.0`, `full-mtp-batch6-3.0`, `diag-qwen9b-gguf-q4`, `full-granite41-8b-gguf-q4-20`, `delib-final-{independent,debate-on-disagreement,debate-all}-9b-exl3-3.0`.

## Configuración verificada

- Hardware: RTX 4050 Laptop GPU (~6 GB), Python 3.12.8, CUDA 12.6, ExLlamaV3 1.4.4 con extensión nativa compilada (sm_89).
- EXL3: context 4096, temperature 0, `no_think=true`, prompt/dataset versionados en `config.json` (`pedagogical_v1`, dataset 1.0).
- MTP: draft model 122 MB (4.0bpw), `mtp_cache_tokens=4096`, `mtp_draft_tokens=2`.
- Deliberación: 4 workers con roles (entailment, skeptical, contradiction, context) + challenge + judge; 3 modos sobre 55 casos.

## Resultados verificados contra runs crudos

| Configuración | n | Quality raw | JSON | Groundedness | Tok/s | VRAM peak |
|---|---:|---:|---:|---:|---:|---:|
| EXL3 3.0bpw, batch 8 | 720 | 0.9054 | 0.885 | 0.968 | 15.1 | 4648 MB |
| EXL3 3.0bpw + MTP, batch 6 | 720 | 0.9155 | 0.903 | 0.952 | 41.7 | 5025 MB |
| EXL3 3.5bpw, batch 8 | 720 | 0.9207 | 0.885 | 0.968 | 14.2 | 5890 MB |
| EXL3 4.0bpw, batch 4 | 720 | 0.9139 | 0.885 | 0.968 | 12.3 | 5050 MB |
| GGUF Q4_K_M, Ollama | 288 | 0.8906 | 0.975 | 0.894 | 30.5 | 5832 MB |
| Granite 4.1-8B GGUF Q4_K_M | 300 | 0.4426 | 0.817 | 0.565 | 17.6 | 5886 MB |

Todos los runs full reportan 0 failures, 0 OOM y 0 timeouts. `failures.jsonl` vacío en cada corrida.

## Correcciones respecto de la transcripción previa

1. **Quality 0.9225 es un re-score, no el número del run.** El run crudo `full-qwen9b-3.0` reporta 0.9054 con `next_step_accuracy: 0.0`. El 0.9225 proviene de re-puntuar los mismos outputs con el rubric t12 corregido (iteración 3: next_step_correct 0% -> 66.7%, +0.017 global). Ambos números deben citarse con su versión de rubric.
2. **La comparación EXL3 vs GGUF no es apples-to-apples.** GGUF corrió 288 generaciones (6 tareas diagnósticas t09-t12, t14, t15 x 48 casos), no 720. Re-score sobre el subset común (288 pares exactos por task_id + case_id, `analysis/rescore_gate6_subset.py` del repo fuente): EXL3 3.0bpw + MTP quality **0.8160** (diagnosis 0.6707, json 0.8875, groundedness 0.9272) vs GGUF **0.8906** (diagnosis 0.7634, json 0.9750, groundedness 0.8936); pareado 31-58-199 a favor de GGUF (gana t09/t10/t11/t15; EXL3 solo gana t12). **GGUF obtuvo mejor calidad en el subset comparable; la ventaja limpia de EXL3 es operacional**: throughput agregado 9.7x (295 vs 30.5 tok/s), menor VRAM (5025 vs 5832 MB) y batching nativo.
3. **Granite quedó penalizado por el rubric viejo.** Corrió antes del fix t12 (`next_step_accuracy: 0.0`) y con 20 casos en lugar de 48. Su 0.4426 no es comparable con el 0.9155 post-fix de Qwen.

## Deliberación verificada (semantic_assessment_v2, 55 casos)

Runs `delib-final-*` (EXL3 3.0bpw + MTP, batch 6):

| Modo | Init | Final | Correcciones | Damage | Net |
|---|---:|---:|---:|---:|---:|
| independent | 76.4% | 76.4% | 0 | 0 | 0 |
| debate-on-disagreement | 69.1% | 65.5% | 10 | 12 | -3 |
| debate-all | 67.3% | 72.7% | 8 | 5 | +3 |

- La accuracy inicial varía entre corridas (67-80%) aun con temperature 0: el MTP introduce no-determinismo. Los rangos históricos (-26 a +3 en debate-all) provienen de las iteraciones `delib-v2`/`delib-v3`.
- Decisión operacional registrada en el repo fuente: **no usar deliberación con el 9B; operar en modo independent**. El judge sobrescribe diagnósticos correctos y el damage (5-12 casos por corrida) no compensa el net dentro del ruido.

## Gaps

- [x] raw outputs, run_ids y configuración localizados y verificados
- [x] deliberación vinculada a artefactos crudos
- [x] Granite 4.1-8B ejecutado (con caveat de rubric y casos)
- [x] `environment.json` registra solo GPU y Python → **resuelto 2026-09-06**: el runner ahora registra `library_versions` (torch, exllamav3 desde version.py, extensión, fla, transformers, safetensors); versiones históricas capturadas post-hoc del mismo venv
- [x] prueba formal de sesión larga, cancelación y reset del generator → **resuelto 2026-09-08**: 30 turnos secuenciales con Qwen3.5-9B EXL3 3.0bpw, 60 episodios registrados, 0 fallos, VRAM estable (crecimiento neto +20 MB, pico 4735 MB), unload limpio a 718 MB
- [x] re-score EXL3 vs GGUF sobre el subset común de 6 tareas: GGUF superior en calidad (0.8906 vs 0.8160; pareado 31-58-199)
- [ ] re-run de Granite con rubric v2 y 48 casos, o declarar formalmente la comparación no comparable

## Conclusión

Los runs crudos confirman el perfil operacional EXL3 3.0bpw + MTP (41.7 tok/s por secuencia, 295 tok/s agregados, 5025 MB VRAM, 0 failures). El re-score del subset común muestra que **GGUF es superior en calidad** (0.8906 vs 0.8160; EXL3 solo supera en groundedness y en t12); la debilidad principal del perfil EXL3 sigue siendo diagnóstico pedagógico (diagnosis_accuracy 0.6707 en el subset).

## Gate Fase 2 del Tutor (2026-09-08)

Corrida formal `tutor-fase2-qwen35-9b-3.0` (720 generaciones, 15 tareas × 48 casos, EXL3 3.0bpw, batch 8 óptimo empírico, sin MTP — configuración idéntica a la baseline `full-qwen9b-3.0` para comparabilidad), ejecutada como gate cuantitativo de Fase 2 del agent-core roadmap:

| Métrica | Baseline `full-qwen9b-3.0` (rubric v1) | Gate Fase 2 (rubric v2, t12 corregido) | Δ |
|---|---:|---:|---:|
| n | 720 | 720 | — |
| quality_score_avg | 0.9054 | 0.9124 | +0.007 |
| diagnosis_accuracy | 0.6977 | 0.6628 | −0.035 |
| next_step_accuracy | 0.0 (rubric vieja) | 0.75 | +0.75 |
| json_valid_rate | 0.8854 | 0.9028 | +0.017 |
| abstention_accuracy | 0.7708 | 0.8125 | +0.042 |
| groundedness | 0.968 | 0.9437 | −0.024 |
| no_unsupported_claims | 0.9985 | 0.9986 | +0.000 |
| tok/s | 15.1 | 27.5 | +12.4 |
| VRAM peak | 4648 MB | 4924 MB | +276 MB |

- 720/720 generaciones, 0 failures, 0 OOM, 0 timeouts. Decision del runner: `candidate`.
- `next_step_accuracy` 0.75 con la rubrica t12 corregida (la baseline 0.0 era un artefacto de parsing, ver corrección 1 arriba).
- `diagnosis_accuracy` 0.6628 vs 0.6977 baseline (−3.5pp): dentro del rango 0.67-0.70 medido para el 9B (BM-006); el andamiaje determinístico del Tutor (TutorSession.diagnose) es la compensación.
- `abstention_accuracy` 0.8125: consistente con el rango 0.81-0.89 del roadmap.
- Artefactos crudos: `small-model-deliberation/engine_benchmark/results/tutor-fase2-qwen35-9b-3.0/` (metrics.json, raw_outputs.jsonl, 720 records).

**Conclusión del gate**: el gate cuantitativo de Fase 2 se cumple — el loop del Tutor (diagnóstico → lección → assessment → mastery) está soportado cuantitativamente por el modelo estrella en la configuración de producción (EXL3 3.0bpw, batch 8, sin MTP).

## Recomendación

- [x] Mantener como Experiment
- [x] Congelar como Benchmark (congelado y aceptado 2026-09-06; gate 5 sesión larga queda como validación operacional)
- [ ] Crear Decision
- [ ] Proponer ADR
- [ ] Nothing
