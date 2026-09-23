---
id: RES-004
category: research
status: superseded
created: 2026-09-05
updated: 2026-09-23
author: human
components: [tutor, providers, configuration, evaluation, performance]
tags: [llm, engine, exllamav3, ollama, exl3, gguf, benchmark, vram]
related: [RES-003, EXP-004, BM-006, DEC-001]
supersedes: null
superseded_by: BM-006
affects: ["src/ipa/providers/**"]
---

# RES-004 — Protocolo de benchmark de engines y modelos LLM

## Tema

Definir una comparación reproducible de modelo + engine + cuantización para el futuro Tutor Agent local en una GPU RTX 4050 de aproximadamente 6 GB.

## Pregunta

¿Qué combinación ofrece el mejor equilibrio entre calidad pedagógica, groundedness, structured output, tool use, latencia, estabilidad y VRAM?

## Diseño

Separar tres comparaciones:

1. Mismo modelo, distinto engine: Qwen3.5-9B EXL3 frente a Qwen3.5-9B GGUF/Ollama.
2. Mismo modelo y engine, distinta cuantización: EXL3 3.0, 3.5 y 4.0 bpw.
3. Familias distintas: Qwen, Ministral, Granite, LFM y candidatos condicionales.

Mantener constante, cuando aplique, system prompt, user prompt, contexto, dataset, temperatura, top-p/top-k, max output tokens, seed, formato y repeticiones.

## Métricas

### Calidad

- diagnosis accuracy;
- next-step accuracy;
- groundedness;
- feedback completeness;
- misconception detection;
- structured-output validity;
- abstention accuracy;
- tool-use correctness.

### Operación

- load time;
- time to first token;
- tokens/s;
- latency p50/p95;
- VRAM peak;
- RAM peak;
- OOM, timeout y format errors;
- degradación en sesiones largas.

La métrica pedagógica principal es `diagnóstico correcto + siguiente intervención correcta`; fluidez y velocidad no deben dominar por sí solas.

## Hardware y controles

- GPU objetivo: NVIDIA GeForce RTX 4050 Laptop, aproximadamente 6 GB.
- Medir VRAM con `nvidia-smi` durante cada corrida.
- Registrar si BGE-M3, reranker u OCR están descargados de GPU.
- Diferenciar modelo completamente en VRAM de modelo con offload a RAM.
- Guardar versión de Python, CUDA, torch, engine, extensión nativa, modelo, quant y configuración.

## Takeaways

El ranking debe ser multidimensional: tutoría, assessment, tool use, eficiencia, velocidad y estabilidad. Un ranking preliminar o una afirmación de model card no constituye evidencia local suficiente para aceptar una decisión.

## Estado

Research propuesto. La ejecución y los resultados deben registrarse en `EXP-004` y, si se congelan, en `BM-006`.

## Cierre (2026-09-23)

Ejecutado y congelado según lo previsto: el protocolo corrió en EXP-004 y el
benchmark quedó como autoridad vigente en **BM-006** (que supersede a este
documento como referencia del protocolo y sus resultados; DEC-001 fija el
perfil operacional resultante).
