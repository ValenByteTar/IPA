---
id: BM-006
category: benchmark
status: accepted
created: 2026-09-05
updated: 2026-09-06
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

**Congelado y aceptado (2026-09-06).** La evidencia cruda vive en `C:\Users\Valen\Desktop\Proyectos\small-model-deliberation` (`engine_benchmark/results/`, 58 runs con run_id). Gates 1, 2, 3, 4, 6 y 7 cerrados; gate 5 (sesión larga/cancelación/OOM) queda como validación operacional pendiente antes de uso productivo del Tutor, sin bloquear el congelamiento de la comparación.

## Entorno reconciliado (gate 2, 2026-09-06)

- GPU: NVIDIA RTX 4050 Laptop (~6 GB, sm_89); Python 3.12.8; CUDA 12.6.
- torch 2.13.0+cu126; ExLlamaV3 **1.4.4** (checkout local `exllamav3-dev/` con extensión nativa `exllamav3_ext.cp312-win_amd64.pyd` compilada para sm_89); flash-linear-attention 0.5.2; transformers 5.16.1; safetensors 0.8.0; huggingface_hub 1.29.0.
- Nota: el venv del repo fuente conserva un registro pip stale de exllamav3 1.4.2; el código efectivamente cargado es el checkout local 1.4.4 (verificado vía `version.py` del módulo). El runner ahora registra `library_versions` en cada `environment.json` (runs futuros); para los runs históricos las versiones se capturaron post-hoc del mismo venv.

## Reconciliación de providers (gate 3, 2026-09-06)

- Núcleo compartido entre `engine_benchmark/runners/lib/exl3_provider.py` (485 líneas) y `src/ipa/providers/exl3_provider.py` (882 líneas): `load`, `reset_generator` (clear_queue + defrag del fix de cache saturado), `unload`, `generate_batch` con MTP (draft 2, cache 4096 efectivo), ChatML `no_think=True`, temperature 0.0.
- IPA extiende para uso interactivo: clase standalone (sin base `ModelProvider`), `generate_chat`/`generate_stream`/`generate_chat_stream`, `VRAMMonitor`, auto-descubrimiento de CUDA/extensión, `config_dict`, `create_star_provider` (fábrica del perfil DEC-001) y protección de timeout con `clear_queue` en `generate_batch`.
- El provider del benchmark recibe `mtp_cache_tokens=4096` por parámetro (fix documentado); IPA lo tiene como default. Mismo comportamiento operacional.

## Configuración verificada

- Candidato principal: Qwen3.5-9B EXL3 3.0bpw + MTP (draft 122 MB, `mtp_cache_tokens=4096`).
- Contexto 4096, temperature 0, `no_think=true`.
- Batch: 6 con MTP para procesamiento paralelo, 1 para interacción; 8 sin MTP.
- Runs: `full-qwen9b-3.0`, `full-mtp-batch6-3.0`, `diag-qwen9b-gguf-q4`, `full-granite41-8b-gguf-q4-20`, `delib-final-*`.

## Resultados verificados

| Dimensión | EXL3 3.0bpw + MTP (n=720) | GGUF Q4_K_M + Ollama (n=288) | Granite 4.1-8B GGUF (n=300) |
|---|---:|---:|---:|
| Quality | 0.9155 | 0.8906 | 0.4426 |
| JSON | 0.903 | 0.975 | 0.817 |
| Groundedness | 0.952 | 0.894 | 0.565 |
| Tok/s por secuencia | 49 | 30.5 | 17.6 |
| Tok/s agregado | 295 | 30.5 | 30.5 |
| VRAM peak | 5025 MB | 5832 MB | 5886 MB |
| Failures | 0 | 0 | 0 |
| Batching nativo | Sí | No | No |

Notas de comparabilidad:

- El quality de EXL3 batch 8 sin MTP es 0.9054 en el run crudo; 0.9225 corresponde al re-score con el rubric t12 v2. El 0.9155 del run MTP ya usa el rubric corregido.
- GGUF corrió solo las 6 tareas diagnósticas (t09-t12, t14, t15). Sobre ese subset EXL3 batch 8 promedia ~0.82: **GGUF tuvo mejor calidad diagnóstica**. La comparación de quality entre tablas no es apples-to-apples.
- Granite corrió con el rubric t12 previo al fix (`next_step_accuracy: 0.0`) y 20 casos; su quality no es comparable con los runs post-fix.

## Calidad por categoría reportada para EXL3 3.0bpw (batch 8, rubric v2)

- explicación: 0.996; ejercicios: 0.974; diagnóstico: 0.786; structured output: 0.972; abstención: 0.885; tool use: 0.906.

La debilidad principal es diagnóstico pedagógico (t09-t12: 0.75-0.80; diagnosis_accuracy 0.67-0.70 en runs crudos). Eso impide interpretar el resultado como solución completa del Tutor.

## Estado de los gates

1. Recuperar raw outputs, métricas y `run_id` — **cerrado**: 58 runs con artefactos completos por corrida.
2. Confirmar dataset/versiones y repeticiones — **cerrado (2026-09-06)**: `config.json` versiona prompt y dataset; `library_versions` agregado al runner; versiones capturadas post-hoc del venv para los runs históricos; sin repeticiones (variabilidad por MTP documentada: init 67-80% con temperature 0).
3. Reconciliar dependencias y paths con IPA — **cerrado (2026-09-06)**: núcleo del provider compartido y verificado; divergencias documentadas (streaming, watchdog de timeout, fábrica `create_star_provider`); mismo ExLlamaV3 1.4.4 + extensión sm_89 en ambos lados.
4. Ejecutar Granite 4.1-8B — **cerrado con caveat**: ejecutado (quality 0.4426, groundedness 0.565, VRAM 5886 MB) con rubric pre-fix y 20 casos; se declara la comparación **no comparable** con los runs post-fix de Qwen; un eventual re-run con rubric v2 y 48 casos queda como mejora opcional, no como gate.
5. Sesiones largas, cancelación, reset y OOM — **diferido por decisión del usuario (2026-09-06)**: no se ejecutará la prueba de sesión larga por ahora. Evidencia indirecta: 0 OOM/timeout en 720 generaciones, `reset_generator()` entre fases validado, protección de timeout con clear_queue en el provider de IPA. Riesgo cubierto por el rollback de DEC-001; reabrir si aparece inestabilidad en uso real.
6. Separar resultados del modelo de resultados del engine — **cerrado (re-score 2026-09-06)**: sobre el subset común de 6 tareas x 48 casos (288 pares exactos, script `analysis/rescore_gate6_subset.py` del repo fuente), EXL3 3.0bpw + MTP obtiene quality 0.8160 (diagnosis 0.6707, json 0.8875, groundedness 0.9272) frente a GGUF 0.8906 (diagnosis 0.7634, json 0.9750, groundedness 0.8936). Comparación pareada: EXL3 mejor en 31 casos, GGUF mejor en 58, empate en 199. **En calidad, GGUF es superior en el subset comparable (+0.075 quality); la ventaja de EXL3 es exclusivamente operacional.**
7. Verificar que la deliberación no aporta valor neto negativo — **cerrado**: `delib-final-*` muestra independent net 0 (0 damage), debate-on-disagreement net -3 (12 damage), debate-all net +3 (5 damage), con variabilidad entre corridas. Decisión: no usar deliberación con el 9B; modo independent.

## Decisión

**Benchmark congelado (2026-09-06).** Ganador operacional: **Qwen3.5-9B EXL3 3.0bpw + MTP con ExLlamaV3** (throughput agregado 9.7x, menor VRAM, batching nativo, MTP +176% en batch 6). Veredicto de calidad: **GGUF Q4_K_M es superior en el subset comparable** (0.8906 vs 0.8160; EXL3 solo supera en groundedness y en t12). La selección operacional prioriza throughput/VRAM/batching para el Tutor local; si la calidad pedagógica pasa a ser el criterio dominante, reabrir la comparación con GGUF como candidato de calidad. Pendiente de validación operacional: sesión larga/cancelación/OOM (gate 5) antes de uso productivo.
