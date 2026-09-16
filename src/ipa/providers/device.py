"""Detección de GPU con caché — base del fallback CPU-only.

El proyecto debe funcionar 100% por CPU cuando no hay GPU disponible.
Toda decisión de dispositivo (OCR, Docling, embeddings, provider LLM)
debe pasar por `has_gpu()` en vez de asumir CUDA.

La señal primaria es `torch.cuda.is_available()`; si torch no está o no
compiló con CUDA, `nvidia-smi` en PATH es la señal secundaria. El resultado
se cachea: la GPU no aparece a mitad de ejecución.
"""
from __future__ import annotations

import os
import shutil

_cached: bool | None = None


def _detect() -> bool:
    if os.environ.get("IPA_FORCE_CPU", "") == "1":
        return False
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    return shutil.which("nvidia-smi") is not None


def has_gpu() -> bool:
    """True si hay GPU CUDA disponible. Resultado cacheado."""
    global _cached
    if _cached is None:
        _cached = _detect()
    return _cached


def reset_gpu_cache() -> None:
    """Invalida la caché (solo tests)."""
    global _cached
    _cached = None


__all__ = ["has_gpu", "reset_gpu_cache"]
