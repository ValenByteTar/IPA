---
id: EXP-009
category: experiment
status: proposed
created: 2026-09-23
updated: 2026-09-23
author: agent
components: [indexes, embeddings, retrieval, evaluation, configuration]
tags: [bge-m3, fp8, nvfp8, ada, rtx4050, sm89, tensor-cores, quantization, dense-sparse]
related: [EXP-008, PM-004, BM-002]
supersedes: null
superseded_by: null
author_model: swe-2
affects: ["src/ipa/indexes/embedding_adapter.py"]
---

# EXP-009 — FP8 E4M3 (“NVFP8”) para BGE-M3 en RTX 4050 Ada

## Estado y alcance

**Propuesto; diseño solamente.** A petición del usuario, no se implementa código,
no se instala ninguna dependencia, no se ejecuta benchmark ni prueba y no se
cambia el default de producción en este registro.

Alcance: inferencia de `BAAI/bge-m3` para producir **dense + sparse** en la RTX
4050 Laptop. No incluye el BGE Reranker (cross-encoder), la generación LLM ni
el formato/almacenamiento de los vectores en LanceDB.

“NVIDIA FP8” y “NVFP8” se usan aquí como nombre corto del candidato FP8 E4M3 con
scaling de NVIDIA; no son sinónimos de los formatos **MXFP8** o **NVFP4**. La
RTX 4050 Ada es compute capability 8.9 (SM89): NVIDIA documenta Tensor Cores
FP8 en Ada; MXFP8/NVFP4 tienen requisitos de arquitectura distintos. La
compatibilidad de una API o un motor con SM89 no garantiza que FlagEmbedding o
BGE-M3 puedan ejecutarlos directamente.

## Hipótesis

Una ruta BGE-M3 en FP8 E4M3 podría reducir el tiempo de cómputo o la memoria
pico frente a FP16 en esta RTX, manteniendo la calidad híbrida suficiente para
IPA. Es igualmente plausible que no mejore el batch pequeño de IPA: el scaling
FP8 agrega trabajo, y la carga es inferencia de chunks relativamente cortos.

## Motivación y baselines

El benchmark local pareado de PM-004 usó los mismos 64 chunks reales (~512
caracteres; ~159 tokens promedio) y `dense+sparse`:

| Baseline medido | Throughput | Uso en el experimento |
|---|---:|---|
| CPU FP32, 6 threads, batch 4 | 2,86–2,99 chunks/s | Referencia de CPU; los 6 threads se fijaron en el probe, no en el adapter |
| GPU FP16, batch 4 | 123,8–135 chunks/s | Baseline GPU principal |
| GPU FP16, batch 8 | 126,6–129,1 chunks/s | Control de sensibilidad al batch |

Estos son microbenchmarks de dispositivo, no SLA de extremo a extremo. En IPA,
`EmbeddingAdapter` usa batch 4 por default en ambos devices y fuerza FP32 cuando
el device resuelto es CPU. El número de threads no está fijado por el adapter;
el runtime actual reportó 6, pero ese valor depende del proceso/entorno. Otros
callers pueden pasar un batch explícito y anular el default; auditar esos caminos
forma parte del diseño, no de un cambio de producción en este EXP.

## Puerta de factibilidad técnica

No se debe comenzar por cuantizar el modelo completo ni asumir que existe una
perilla `use_nvfp8`:

1. El adapter actual usa `BGEM3FlagModel` y le pasa `use_fp16`; no contiene ruta
   FP8. La API pública consultada de M3Embedder expone `use_fp16` y `use_bf16`,
   no un argumento FP8.
2. Verificar, para las versiones locales de Python/PyTorch/CUDA/FlagEmbedding,
   si la ruta elegida ejecuta en SM89 las capas reales de BGE-M3. El resultado
   debe conservar el forward que devuelve tanto `dense_vecs` como
   `lexical_weights`.
3. Candidato primario de investigación: FP8 E4M3 con `Float8CurrentScaling`
   de Transformer Engine, **solo si** se puede integrar en las capas lineales
   relevantes sin sustituir ni desalinear las salidas dense/sparse. La receta
   calcula `amax` y aplica scaling; ese overhead se debe medir.
4. Ruta alternativa, solo si la primera no es viable: exportar el grafo de BGE-M3
   (incluidas las cabezas sparse) a un motor que admita FP8 en SM89. La
   exportación debe probar paridad de todas las salidas; no se asume compatibilidad
   de ONNX/TensorRT por adelantado.
5. No usar NVFP4/MXFP8 como atajo: no son el mismo formato FP8 E4M3 y su soporte
   de hardware/escala no corresponde automáticamente a Ada.

Si dense y sparse no pueden conservarse en la misma ejecución experimental, la
puerta falla y no se compara esa ruta como reemplazo del BGE-M3 híbrido.

## Diseño de la comparación

### Controles

- Misma RTX 4050 Laptop, misma versión de modelo BGE-M3, mismo conjunto ordenado
  de textos, mismo `max_length`, mismos callers y mismo número de warmups.
- Baseline CPU: FP32, batch 4 y registrar el `torch.get_num_threads()` real; el
  probe de referencia fijó 6 explícitamente.
- Baseline GPU: FP16, batches 4 y 8.
- Candidato: FP8 E4M3 con scaling NVIDIA, batches 4 y 8; batch 16 solo si los
  dos primeros no exceden memoria y muestran throughput creciente.
- Correr con GPU exclusiva bajo `vram.lock`: descargar Ollama/ExL3 antes del
  ensayo, mantener el chat en maintenance mode y restaurar el modelo al salir.
  Registrar cualquier fallback como fallo de configuración, no como resultado
  FP8.

### Datos y aislamiento

1. Microset pareado de los mismos 64 chunks reales usado en PM-004, para comparar
   el baseline publicado.
2. Un subconjunto reproducible y representativo de 1.024 chunks del corpus, con
   longitudes/tipos anotados; no usar una sola longitud sintética como resultado.
3. Evaluación de retrieval con el mismo query set/candidate pool versionado de
   `E10-rerank` o una versión ampliada fijada antes de correrlo. No modificar
   `E12-corpus`, el staging Reporter, Transit ni Archive durante el diseño/ensayo.
4. Crear una salida experimental aislada y versionar modelo, backend, receta,
   scalings, batch, max length, versions y hash del dataset. No mezclar filas FP8
   con vectores FP16 existentes ni promover el corpus experimental.

### Matriz de métricas

- **Compatibilidad**: carga en SM89; capas cuantizadas; ausencia de NaN/Inf; shape
  dense 1024; sparse presente y finito; conteo de chunks procesados igual al
  input.
- **Rendimiento**: tiempo de carga, warmup, chunks/s de forward y de escritura,
  latencia p50/p95 por batch, VRAM pico/libre y fallback/OOM. Separar throughput
  de modelo de throughput de pipeline.
- **Fidelidad numérica**: distribución de similitud coseno dense vs FP16; top-k
  de pesos sparse/solapamiento y diferencias de pesos.
- **Calidad retrieval**: `recall@1/5/10/20`, MRR y `nDCG@10` en evaluación pareada;
  reportar diferencias con intervalo bootstrap, no solo un promedio agregado.
- **Repetibilidad**: warmup separado, al menos 3 repeticiones por configuración,
  orden alternado y misma carga GPU/CPU. Registrar temperatura/reloj para detectar
  throttling del portátil.

## Criterios propuestos (pendientes de aprobación antes de ejecutar)

- **Calidad**: no más de 1 punto porcentual de pérdida en recall@10 ni 0,01 en
  `nDCG@10` frente a FP16 en la evaluación pareada; cualquier cambio de ranking
  se informa aunque cumpla los umbrales. Revisar dense/sparse por separado.
- **Beneficio**: al menos 1,2× throughput end-to-end frente a FP16 batch 4, o una
  reducción de al menos 20% de VRAM pico sin pérdida material de rendimiento.
  Si no se obtiene una de esas mejoras, no se justifica mantener otra ruta.
- **Seguridad operacional**: cero OOM, cero mezcla de precisiones dentro del
  mismo LanceDB y fallback explícito y observable si la ruta FP8 no puede cargar.

Los umbrales anteriores son diseño propuesto, no resultados ni política aceptada.
El usuario debe aprobarlos antes de una ejecución real.

## Decisión esperada

- Si la puerta de factibilidad falla: cerrar el EXP como no viable para el stack
  actual; conservar CPU FP32/GPU FP16 y no añadir dependencias.
- Si FP8 es funcional pero no cumple calidad/rendimiento: no cambiar el default;
  dejar evidencia y, si procede, optimizar otra etapa.
- Si cumple todo: proponer un adapter/backend FP8 **opt-in**, con metadata de
  versión/precisión y corpus experimental separado; no activar por umbral ni
  re-embebir Transit/E12 sin aprobación explícita.

## Resultados

No ejecutados por instrucción del usuario. No hubo código, configuración runtime,
prueba ni benchmark en esta propuesta.

## Fuentes

- EKS local: EXP-008 (VRAM/hardware), PM-004 (baseline BGE-M3), EXP-007
  (reranker separado), BM-006 (hardware/stack).
- NVIDIA, *Ada GPU Architecture Tuning Guide*:
  https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
- NVIDIA, *CUDA Compute Capabilities* (tabla FP8 por compute capability):
  https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- NVIDIA Transformer Engine, *FP8 Current Scaling* (SM89+, formatos/scaling y
  coste de `amax`):
  https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html
- NVIDIA Transformer Engine, *NVFP4* (formato distinto; requisito Blackwell):
  https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.18/user-guide/features/low_precision_training/nvfp4/nvfp4.html
- BGE documentation, *M3Embedder API*:
  https://bge-model.com/API/inference/embedder/encoder_only/M3Embedder
