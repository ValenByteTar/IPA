---
id: EXP-004
category: experiment
status: draft
created: 2026-09-05
updated: 2026-09-05
author: human
components: [tutor, providers, evaluation, performance]
tags: [qwen3.5-9b, exllamav3, exl3, ollama, mtp, vram, deliberation]
related: [RES-004, BM-006, DEC-001]
supersedes: null
superseded_by: null
---

# EXP-004 — Comparación de engine y modelo LLM para Tutor

## Hipótesis

Qwen3.5-9B EXL3 con ExLlamaV3 puede ofrecer una combinación superior de calidad y rendimiento frente a Qwen3.5-9B GGUF ejecutado con Ollama, siempre que el uso de VRAM, contexto y estabilidad sea viable.

## Motivación

El Tutor Agent depende de explicaciones, ejercicios, assessment, structured output, abstención y tool use. Elegir el engine por tokens/s únicamente podría producir un sistema rápido pero pedagógicamente inferior.

## Configuración reportada

Los documentos fuente reportan 720 generaciones sobre 15 tareas pedagógicas y 55 casos de deliberación. También reportan:

- Qwen3.5-9B EXL3 3.0bpw + MTP;
- Qwen3.5-9B EXL3 3.5bpw;
- Qwen3.5-9B EXL3 4.0bpw;
- Qwen3.5-9B GGUF Q4_K_M en Ollama.

La configuración operacional reportada para EXL3 es batch 6/1, context 4096, `mtp_cache_tokens=4096`, `temperature=0`, `no_think=true` y MTP habilitado.

## Resultados reportados, pendientes de verificación

| Configuración | Quality | JSON | Groundedness | Tok/s reportado | VRAM peak reportada |
|---|---:|---:|---:|---:|---:|
| EXL3 3.0bpw, batch 8 | 0.9225 | 0.885 | 0.968 | 15,1 | 4648 MB |
| EXL3 3.0bpw + MTP, batch 6 | 0.9155 | 0.903 | 0.952 | 41,7 | 5025 MB |
| EXL3 3.5bpw, batch 8 | 0.9207 | 0.885 | 0.968 | 14,2 | 5890 MB |
| EXL3 4.0bpw, batch 4 | 0.9139 | 0.885 | 0.968 | 12,3 | 5050 MB |
| GGUF Q4_K_M, Ollama | 0.8906 | 0.975 | 0.894 | 30,5 | 5832 MB |

Los valores son transcritos desde `RECOMENDACION_OPERACIONAL.md`; no se consideran evidencia final hasta localizar runs, dataset, configuración y outputs crudos.

## Deliberación reportada

El documento fuente reporta daño neto para debate-on-disagreement y debate-all en el conjunto `semantic_assessment_v2`. Este resultado debe reproducirse o vincularse a artefactos antes de convertirlo en una política permanente.

## Gaps

- no se localizaron en IPA los outputs crudos de las 720 generaciones;
- no está asociado un `run_id` verificable;
- Granite 4.1-8B no fue comparado en el resultado final;
- algunas instrucciones apuntan al repositorio `engine_benchmark`, no a los paths actuales de IPA;
- las versiones de dependencias deben reconciliarse con `AGENTS.md`;
- falta una prueba de sesión larga y recuperación de fallos.

## Conclusión provisional

Los datos reportados favorecen operacionalmente EXL3 3.0bpw + MTP para el hardware actual, pero el experimento queda en `draft` hasta validar provenance local y reproducibilidad.

## Recomendación

- [x] Mantener como Experiment
- [ ] Congelar como Benchmark
- [ ] Crear Decision
- [ ] Proponer ADR
- [ ] Nothing
