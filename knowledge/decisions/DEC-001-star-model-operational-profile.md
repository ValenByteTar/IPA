---
id: DEC-001
category: decision
status: accepted
created: 2026-09-05
updated: 2026-09-23
author: human
components: [tutor, providers, configuration, performance]
tags: [qwen3.5-9b, exllama, exl3, mtp, no-think, batch, provisional]
related: [RES-004, EXP-004, BM-006]
supersedes: null
superseded_by: null
evidence: ["src/ipa/providers/exl3_provider.py", "tests/test_device_fallback.py"]
affects: ["src/ipa/providers/**", "configs/**"]
---

# DEC-001 — Perfil operacional del modelo estrella

## Contexto

IPA integra `ExL3Provider` y documenta Qwen3.5-9B EXL3 3.0bpw + MTP como modelo estrella. La evidencia cruda fue localizada y verificada (2026-09-06) en `C:\Users\Valen\Desktop\Proyectos\small-model-deliberation` (`engine_benchmark/results/`, runs `full-qwen9b-3.0`, `full-mtp-batch6-3.0`, `diag-qwen9b-gguf-q4`, `full-granite41-8b-gguf-q4-20`, `delib-final-*`). El re-score sobre el subset comparable (6 tareas x 48 casos, 288 pares exactos) muestra GGUF superior en calidad; la selección se justifica operacionalmente. `BM-006` fue congelado y aceptado el 2026-09-06.

## Decisión

Usar el siguiente perfil cuando se ejecute el Tutor Agent o inferencia paralela de alto volumen:

```text
model: Qwen3.5-9B EXL3 3.0bpw
engine: ExLlamaV3 1.4.4 (checkout local + extensión sm_89)
use_mtp: true
batch_size: 6 en procesamiento paralelo, 1 en interacción
context_length: 4096
mtp_cache_tokens: 4096
temperature: 0.0
no_think: true
```

La justificación de este perfil es **operacional**: throughput agregado 295 tok/s (9.7x sobre Ollama), 49 tok/s por secuencia (MTP +176% en batch 6), VRAM peak 5025 MB con ~1 GB de margen, batching nativo y 0 failures/OOM/timeouts en 720 generaciones. No se afirma superioridad de calidad pedagógica: el re-score sobre el subset comparable muestra **GGUF superior en calidad** (quality 0.8906 vs 0.8160; diagnosis 0.7634 vs 0.6707; json 0.975 vs 0.8875; pareado 31-58-199). EXL3 solo supera en groundedness (0.9272 vs 0.8936) y en t12_next_step.

No activar deliberación: la evidencia verificada (`delib-final-*`, 55 casos) muestra net -3 con 12 damage en debate-on-disagreement, net +3 con 5 damage en debate-all y variabilidad entre corridas (init 67-80% con temperature 0 por no-determinismo del MTP). El 9B opera en modo independent.

## Consecuencias

- Se aprovecha la implementación existente del provider (`create_star_provider` materializa este perfil).
- El perfil cabe con margen verificado en la GPU de aproximadamente 6 GB.
- Se acepta una debilidad conocida y medida en diagnóstico pedagógico (diagnosis_accuracy 0.6707 en el subset).
- El uso de MTP introduce variabilidad residual aun con temperatura 0.
- Cambiar modelo, engine, cuantización o hardware requiere actualizar benchmark y revisar esta decisión.

## Condiciones de aceptación

- [x] artefactos crudos y run IDs verificables (repo fuente, 58 runs);
- [x] entorno y dependencias reconciliados (2026-09-06: ExLlamaV3 1.4.4 + ext sm_89 verificados en ambos lados; runner registra `library_versions`);
- [~] prueba de estabilidad prolongada (sesión larga, cancelación, reset del generator): **diferida por decisión del usuario (2026-09-06)** — no se ejecutará por ahora; queda registrada como validación operacional no realizada. Evidencia indirecta disponible (0 OOM/timeout en 720 gens, `reset_generator()` validado, protección de timeout en el provider); el rollback cubre el riesgo si aparece inestabilidad en uso real;
- [x] evaluación de diagnóstico pedagógico (realizada: debilidad conocida y medida);
- [x] comparación contra controles: re-score EXL3 vs GGUF sobre subset común completado (GGUF superior en calidad, decisión operacional documentada); Granite declarado no comparable (rubric pre-fix, 20 casos);
- [x] revisión de fallos y recuperación (0 failures en runs full; `reset_generator()` validado entre fases).

## Rollback

Volver al perfil anterior o a un provider alternativo (GGUF/Ollama como candidato de calidad) mediante configuración/composition root, sin modificar contratos del corpus. El proveedor es una implementación reemplazable y no una autoridad de arquitectura.
