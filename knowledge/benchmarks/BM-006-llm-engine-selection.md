---
id: BM-006
category: benchmark
status: draft
created: 2026-09-05
updated: 2026-09-05
author: human
components: [tutor, providers, evaluation, performance, configuration]
tags: [llm, engine-selection, qwen3.5-9b, exllamav3, ollama, exl3, mtp]
related: [RES-004, EXP-004, DEC-001]
supersedes: null
superseded_by: null
---

# BM-006 — Selección operacional de modelo y engine para Tutor

## Objetivo

Congelar una baseline comparable para decidir el proveedor LLM del Tutor Agent local.

## Estado del benchmark

`draft`. El documento de recomendación contiene resultados resumidos, pero no se encontraron todavía en IPA todos los artefactos crudos necesarios para aceptar formalmente el benchmark.

## Configuración reportada

- Hardware: RTX 4050 Laptop GPU, aproximadamente 6 GB.
- Candidato principal reportado: Qwen3.5-9B EXL3 3.0bpw + MTP.
- Engine reportado: ExLlamaV3 con continuous batching.
- Contexto: 4096.
- MTP cache: 4096 tokens.
- Temperatura: 0.
- `no_think`: true.
- Batch: 6 para procesamiento paralelo y 1 para interacción.

## Resultados reportados

| Dimensión | EXL3 3.0bpw + MTP | GGUF Q4_K_M + Ollama |
|---|---:|---:|
| Quality | 0.9155 | 0.8906 |
| JSON | 0.903 | 0.975 |
| Groundedness | 0.952 | 0.894 |
| Tok/s por secuencia | 49 | 30,5 |
| Tok/s agregado reportado | 295 | 30,5 |
| VRAM peak | 5025 MB | 5832 MB |
| Failures reportados | 0 | 0 |
| Batching nativo | Sí | No |

## Calidad por categoría reportada para EXL3 3.0bpw

- explicación: 0,996;
- ejercicios: 0,974;
- diagnóstico: 0,786;
- structured output: 0,972;
- abstención: 0,885;
- tool use: 0,906.

La debilidad principal reportada es diagnóstico pedagógico. Eso impide interpretar el resultado como “solución completa” del Tutor.

## Gates pendientes

1. Recuperar raw outputs, métricas y `run_id`.
2. Confirmar dataset/versiones y número de repeticiones.
3. Reconciliar dependencias y paths con IPA actual.
4. Ejecutar Granite 4.1-8B o declarar formalmente que queda fuera del universo comparado.
5. Probar sesiones largas, cancelación, reset del generator y OOM.
6. Separar resultados del modelo de resultados del engine.
7. Verificar que la deliberación no aporta valor neto negativo en el flujo que IPA realmente utilizará.

## Decisión

No se congela todavía un ganador como benchmark aceptado. La recomendación provisional es continuar con EXL3 3.0bpw + MTP como candidato operacional, sujeto a completar los gates.
