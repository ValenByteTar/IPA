"""Lock global para la construcción de modelos pesados in-process.

transformers/accelerate (``init_empty_weights``, ``init_on_device``)
parchean ``torch.nn.Module.register_parameter`` a nivel CLASE durante
``from_pretrained`` — el monkey-patch es global al proceso, no
thread-local. Si otro thread construye ``nn.Module`` en esa ventana (el
dashboard arranca warmups paralelos: BGE-M3 en un thread y el provider
estrella en otro), sus parámetros nacen en device ``meta`` y nunca se
materializan: cada inferencia falla con ``Cannot copy out of meta
tensor``. Bug real 2026-09-24: el retrieval del dashboard quedó roto de
forma permanente tras un restart con warmups concurrentes, y el pipeline
SSE lo enmascaraba como "corpus vacío" (PM-007).

Todo constructor de modelo pesado que pueda crear ``nn.Module``
(``BGEM3FlagModel``, ``FlagReranker``, ``easyocr.Reader``,
``Model.from_config`` de ExL3) debe ejecutarse bajo este lock, con
double-check del singleton adentro para no construir dos veces.

Complemento intra-proceso de PAT-007 (leases de archivo cross-process):
los file locks serializan procesos; este serializa threads que comparten
el estado global de torch/accelerate.

Orden de locks: ``vram.lock`` (archivo) se adquiere ANTES que este —
nunca al revés — así no hay espera circular con los providers GPU.
"""
from __future__ import annotations

import threading

# RLock: permite anidar construcciones (un provider que cargue un modelo
# auxiliar dentro de su propio load) sin deadlock.
MODEL_LOAD_LOCK = threading.RLock()
