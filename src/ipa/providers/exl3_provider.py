"""ExLlamaV3 provider â€” inferencia nativa GPU sin overhead HTTP.

Modelo estrella: Qwen3.5-9B EXL3 3.0bpw + MTP speculative decoding.
Seleccionado por benchmark pedagÃ³gico (720 generaciones, 15 tareas,
quality 0.9225) y benchmark de deliberaciÃ³n (55 casos, debate daÃ±ino).

Flujo: Config.from_directory -> Model.from_config -> Cache -> Generator
      Job + ComboSampler para generaciÃ³n.

Soporta:
  - MTP speculative decoding (draft model para +40-176% velocidad)
  - Continuous batching (encolar todos los jobs a la vez)
  - ChatML con no_think (respuestas directas sin reasoning)
  - reset_generator() entre fases batch grandes (bug crÃ­tico)
  - VRAM monitoring vÃ­a nvidia-smi

ParÃ¡metros crÃ­ticos (RECOMENDACION_OPERACIONAL.md):
  batch_size: 6 (paralelo) o 1 (interactivo)
  context_length: 4096
  mtp_cache_tokens: 4096 (default 2048 rompe prompts >750 tokens)
  no_think: True
  temperature: 0.0
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

# Windows: suppress console window flashes from subprocesses (nvidia-smi, etc.)
# CREATE_NO_WINDOW (0x08000000) prevents console creation for non-console parents.
# DETACHED_PROCESS (0x00000008) ensures no window even from pythonw.exe.
_NO_WINDOW_FLAGS = 0
if os.name == "nt":
    _NO_WINDOW_FLAGS = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Environment setup â€” debe ejecutarse antes de importar exllamav3
# ---------------------------------------------------------------------------

def _setup_cuda_env() -> None:
    """Setea CUDA_PATH y PATH antes de importar exllamav3.

    Sin esto, exllamav3_ext.pyd no encuentra las DLLs de CUDA y falla con
    'DLL load failed'.
    """
    _CUDA_CANDIDATES = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
        os.environ.get("CUDA_PATH", ""),
    ]
    cuda_path = None
    for candidate in _CUDA_CANDIDATES:
        if candidate and Path(candidate).exists():
            cuda_path = candidate
            break
    if cuda_path:
        os.environ["CUDA_PATH"] = cuda_path
        bin_dir = os.path.join(cuda_path, "bin")
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def _setup_exllamav3_path() -> Optional[Path]:
    """AÃ±ade exllamav3-dev al sys.path si tiene la extensiÃ³n compilada.

    El pip package exllamav3 no incluye la extensiÃ³n C++/CUDA compilada.
    El checkout local exllamav3-dev/ contiene exllamav3_ext*.pyd compilado
    para la GPU especÃ­fica (sm_89 = RTX 4050).

    Sin esto, `import exllamav3_ext` falla con ModuleNotFoundError.
    """
    # Buscar exllamav3-dev relativo al project root
    # src/ipa/providers/exl3_provider.py -> ../../../exllamav3-dev
    here = Path(__file__).resolve().parent
    project_root = here.parents[2]  # src/ipa/providers -> src -> project_root
    checkout = project_root / "exllamav3-dev"
    candidates = (checkout / "build", checkout, checkout / "source")
    for candidate in candidates:
        if not candidate.exists():
            continue
        ext_files = list(candidate.glob("exllamav3_ext*.pyd"))
        if ext_files:
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    return None


def _init_cuda() -> None:
    """Inicializa CUDA antes de importar exllamav3_ext.

    Sin esto, el .pyd puede entrar en deadlock al hacer su propia
    inicializaciÃ³n de CUDA si torch no inicializÃ³ el contexto primero.
    """
    import torch
    if torch.cuda.is_available():
        torch.cuda.init()


# Ejecutar setup al importar el mÃ³dulo
_setup_cuda_env()
_EXL3_DEV_PATH = _setup_exllamav3_path()


# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Resultado de una generaciÃ³n individual."""
    text: str
    latency_s: float
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    time_to_first_token_ms: float = 0.0
    vram_before_mb: int = 0
    vram_after_mb: int = 0
    vram_peak_mb: int = 0
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Chat templates
# ---------------------------------------------------------------------------

def _chatml(messages: List[Dict[str, str]]) -> str:
    """ChatML â€” Qwen3.5, Qwen3, modelos basados en Qwen."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _chatml_no_think(messages: List[Dict[str, str]]) -> str:
    """ChatML con thinking deshabilitado (bloque think vacÃ­o).

    Para Qwen3.5 y modelos reasoning: fuerza respuesta directa sin <think>.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "\n".join(parts)


def _detect_family(model_id: str) -> str:
    """Detecta la familia de template a partir del model_id."""
    mid = model_id.lower()
    if "granite" in mid:
        return "granite"
    if "lfm" in mid:
        return "lfm"
    if "ministral" in mid:
        return "ministral"
    if "ornith" in mid:
        return "ornith"
    # Qwen y derivados usan ChatML
    return "chatml"


def _build_cjk_bias(tokenizer) -> Dict[int, float]:
    """Construye {token_id: -inf} para tokens con caracteres CJK.

    Escanea el vocabulario decodificando por chunks (decode exige Tensor).
    Si un token decodifica con caracteres de los rangos CJK (han, kana,
    hangul, compatibilidad), se banea. Scan O(vocab), ~0.2s para 248k tokens.
    """
    import torch
    bias: Dict[int, float] = {}
    vocab_size = getattr(tokenizer, "actual_vocab_size", None) or 150000
    chunk = 2048
    for start in range(0, vocab_size, chunk):
        end = min(start + chunk, vocab_size)
        try:
            id_tensor = torch.arange(start, end, dtype=torch.long)
            texts = tokenizer.decode(id_tensor)
        except Exception:
            continue
        for tid, piece in zip(range(start, end), texts):
            if piece and _has_cjk(piece):
                bias[tid] = float("-inf")
    return bias


def _has_cjk(text: str) -> bool:
    """True si el texto contiene caracteres no latinos (CJK o cirílico).

    El modelo es bilingüe zh/en y en español desliza caracteres chinos
    Y cirílicos ("Модель" visto en producción). Ambos alfabetos están
    baneados a nivel sampler para respuestas en alfabeto latino puro.
    """
    for ch in text:
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF    # CJK Unified Ideographs
            or 0x3400 <= code <= 0x4DBF  # CJK Extension A
            or 0x3040 <= code <= 0x30FF  # Hiragana + Katakana
            or 0xAC00 <= code <= 0xD7AF  # Hangul
            or 0xF900 <= code <= 0xFAFF  # CJK Compatibility Ideographs
            or 0x0400 <= code <= 0x04FF  # Cyrillic
            or 0x0500 <= code <= 0x052F  # Cyrillic Supplement
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# VRAM monitor
# ---------------------------------------------------------------------------

class VRAMMonitor:
    """Monitor de VRAM que hace polling de nvidia-smi en thread separado.

    Registra VRAM pico durante la generaciÃ³n. No bloquea el hilo principal.
    """

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._samples: list[int] = []
        self._running = False

    def _query_vram(self) -> int:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
                creationflags=_NO_WINDOW_FLAGS,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass
        return 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            vram = self._query_vram()
            if vram > 0:
                self._samples.append(vram)
            self._stop_event.wait(self.interval_s)

    def stop(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def peak_mb(self) -> int:
        return max(self._samples) if self._samples else 0

    @property
    def current_mb(self) -> int:
        return self._samples[-1] if self._samples else self._query_vram()


def query_vram_mb() -> int:
    """Query puntual de VRAM usada en MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=_NO_WINDOW_FLAGS,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return 0


def query_gpu_info() -> dict:
    """Info completa de la GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=_NO_WINDOW_FLAGS,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return {
                "name": parts[0],
                "vram_total_mb": int(parts[1]),
                "vram_used_mb": int(parts[2]),
                "vram_free_mb": int(parts[3]),
            }
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# ExL3Provider
# ---------------------------------------------------------------------------

def _unload_ollama_models(timeout_s: float = 20.0) -> None:
    """Compatibility wrapper over the shared Ollama VRAM release routine."""
    from .ollama_provider import unload_ollama_models
    try:
        unload_ollama_models(timeout_s=timeout_s)
    except Exception:
        pass


class ExL3Provider:
    """Provider para ExLlamaV3 con modelos EXL3.

    Modelo estrella: Qwen3.5-9B EXL3 3.0bpw + MTP.
    Ver RECOMENDACION_OPERACIONAL.md para justificaciÃ³n de parÃ¡metros.

    Uso tÃ­pico:
        provider = ExL3Provider(
            model_path="models/Qwen3.5-9B-exl3-3.0bpw",
            model_id="Qwen3.5-9B-EXL3-3.0bpw",
            quantization="EXL3-3.0bpw",
            use_mtp=True,
            mtp_cache_tokens=4096,
            batch_size=6,
            no_think=True,
        )
        provider.load()
        results = provider.generate_chat_batch(batch_messages)
        provider.reset_generator()  # entre fases batch grandes
        provider.unload()
    """

    name = "exllamav3"

    def __init__(
        self,
        model_path: str,
        model_id: str,
        quantization: str,
        context_length: int = 4096,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: Optional[int] = None,
        no_think: bool = True,
        batch_size: int = 6,
        use_mtp: bool = True,
        mtp_draft_tokens: int = 2,
        mtp_cache_tokens: int = 4096,
        cache_k_bits: int = 0,
        cache_v_bits: int = 0,
        suppress_cjk: bool = False,
        rep_p: float = 1.1,
    ) -> None:
        self.model_path = model_path
        self.model_id = model_id
        self.engine = "exllamav3"
        self.quantization = quantization
        self.context_length = context_length
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.no_think = no_think
        self.rep_p = rep_p
        self.batch_size = batch_size
        self.use_mtp = use_mtp
        # MTP y batch son incompatibles a partir de batch ~3 (medido + literatura:
        # el overhead de realineación del speculative decoding en batch crece
        # superlinealmente — arXiv 2510.22876 lo mide en 40% del cómputo a
        # batch 8; arXiv 2310.18813: "larger batch sizes require a smaller
        # speculation length"). Medido en RTX 4050: batch 6 con MTP = 17.5
        # tok/s (thrashing, secuencias de 3 a 31 tok/s); sin MTP = 83.2 tok/s
        # uniforme. A batch 1 el MTP da +14% (37.1 vs 32.5). Guard automático:
        # batch > 2 → MTP off (IPA_EXL3_FORCE_MTP=1 lo fuerza).
        if self.use_mtp and self.batch_size > 2 and \
                os.environ.get("IPA_EXL3_FORCE_MTP", "0") != "1":
            print(
                f"  [EXL3] batch_size={self.batch_size} > 2 -> MTP desactivado "
                "(medido: thrashing por realineación; el MTP aporta solo a batch 1)",
                flush=True,
            )
            self.use_mtp = False
        self.mtp_draft_tokens = mtp_draft_tokens
        self.mtp_cache_tokens = mtp_cache_tokens
        # KV cache cuantizado: 0 = fp16 (default). 8 = q8 (mitad de VRAM).
        # Solo aplica a capas de atención global; las recurrentes usan O(1).
        self.cache_k_bits = cache_k_bits
        self.cache_v_bits = cache_v_bits
        # Ban de tokens CJK (chino/japonés/coreano): el modelo es bilingüe
        # zh/en y en español a veces desliza caracteres chinos. El bias -inf
        # los prohíbe a nivel sampler (capa dura); el prompt es la capa blanda.
        self.suppress_cjk = suppress_cjk
        self._cjk_bias: Optional[Dict[int, float]] = None
        self._mtp_model = None
        self._mtp_cache = None
        self._config = None
        self._model = None
        self._tokenizer = None
        self._cache = None
        self._generator = None
        self._load_time = 0.0
        self._family = _detect_family(model_id)

    def load(self) -> float:
        """Carga el modelo en GPU. Retorna segundos de carga."""
        # Si Ollama está generando, esperar a que cierre su lease antes de
        # descargar el modelo server-side. Un batch de embeddings o una segunda
        # instancia ExL3, en cambio, conserva prioridad/ownership y hace fallar
        # este load con un error claro.
        from . import vram_lock
        deadline = time.monotonic() + 300
        while True:
            holder = vram_lock.holder()
            if holder is None or holder.get("owner") != "ollama":
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Ollama sigue usando la VRAM; ExL3 no inició la carga")
            time.sleep(0.2)
        if not vram_lock.acquire("exl3"):
            h = vram_lock.holder() or {}
            raise RuntimeError(
                f"VRAM ocupada por {h.get('owner', '?')} (pid {h.get('pid', '?')}); "
                "esperá a que termine o limpiá outputs/agent/vram.lock")
        try:
            return self._load_locked()
        except Exception:
            # No filtrar el lock si la carga falla (OOM, VRAM ocupada por el
            # dashboard, etc.): el próximo intento debe poder tomarlo.
            vram_lock.release("exl3")
            raise

    def _load_locked(self) -> float:
        """Carga con el lock de VRAM ya tomado."""
        try:
            _unload_ollama_models()
        except Exception:
            pass
        _init_cuda()
        from exllamav3 import Config, Model, Cache, Tokenizer

        t0 = time.monotonic()
        self._config = Config.from_directory(self.model_path)
        self._model = Model.from_config(self._config, component="text")
        self._tokenizer = Tokenizer(self._config)

        # Modelos con estados recurrentes (Mamba/SSM, atenciÃ³n lineal hÃ­brida)
        # usan memoria O(1) por token para esas capas.
        is_recurrent = bool(self._model.caps.get("recurrent_states", False))
        if is_recurrent:
            print(f"  [EXL3] Model has recurrent states (Mamba/SSM/linear-attn)", flush=True)

        # Construir ban de tokens CJK una sola vez por carga (scan del vocab).
        if self.suppress_cjk:
            self._cjk_bias = _build_cjk_bias(self._tokenizer)
            print(f"  [EXL3] CJK suppress: {len(self._cjk_bias)} tokens baneados", flush=True)

        # mtp_cache_tokens: permite override del cache limit.
        # CRÃTICO: default 2048 causa outputs vacÃ­os en prompts >750 tokens.
        # mtp_cache_tokens solo puede AMPLIAR el cache, nunca recortarlo por
        # debajo del contexto declarado. Recortarlo causaba el fallo real
        # "Job requires N pages (only M available) and cannot be enqueued"
        # → salida vacía en prompts grandes (6144 ctx + 4096 cache = todo
        # prompt >4096 tokens fallaba en silencio).
        if self.mtp_cache_tokens > 0:
            cache_tokens = max(self.context_length, self.mtp_cache_tokens)
        else:
            cache_tokens = self.context_length

        # KV cache cuantizado (opcional): reduce VRAM del cache principal.
        # El cache MTP queda en fp16: el speculative decoding depende de él.
        _quant_layer = None
        if self.cache_k_bits and self.cache_v_bits:
            try:
                from exllamav3.cache import CacheLayer_quant as _CacheLayer_quant
                _quant_layer = _CacheLayer_quant
                print(f"  [EXL3] KV cache principal cuantizado: k={self.cache_k_bits}b v={self.cache_v_bits}b (MTP fp16)", flush=True)
            except ImportError:
                print("  [EXL3] CacheLayer_quant no disponible; usando fp16", flush=True)

        if self.use_mtp:
            self._mtp_model = Model.from_config(self._config, component="mtp")
            self._mtp_cache = Cache(
                self._mtp_model,
                max_num_tokens=cache_tokens,
                max_batch_size=self.batch_size,
                max_history=self.mtp_draft_tokens,
            )
            self._mtp_model.load(reserve_per_device=0.05)

        if _quant_layer is not None:
            self._cache = Cache(
                self._model,
                max_num_tokens=cache_tokens,
                max_batch_size=self.batch_size,
                max_history=self.mtp_draft_tokens if self.use_mtp else 0,
                layer_type=_quant_layer,
                k_bits=self.cache_k_bits,
                v_bits=self.cache_v_bits,
            )
        else:
            self._cache = Cache(
                self._model,
                max_num_tokens=cache_tokens,
                max_batch_size=self.batch_size,
                max_history=self.mtp_draft_tokens if self.use_mtp else 0,
            )
        if self.use_mtp:
            self._model.load(reserve_per_device=0.05)
        else:
            self._model.load()

        from exllamav3.generator import Generator
        # Optimizaciones del Generator:
        # - max_chunk_size=4096: prefill en chunks mÃ¡s grandes = menos overhead
        # - max_q_size=16: mÃ¡s queries encoladas para decode = mejor throughput
        # - enable_defrag=True: defragmentaciÃ³n automÃ¡tica del KV cache
        self._generator = Generator(
            self._model, self._cache, self._tokenizer,
            draft_model=self._mtp_model if self.use_mtp else None,
            draft_cache=self._mtp_cache if self.use_mtp else None,
            num_draft_tokens=self.mtp_draft_tokens if self.use_mtp else None,
            max_chunk_size=4096,
            max_q_size=16,
            enable_defrag=True,
        )
        self._load_time = time.monotonic() - t0
        return self._load_time

    def reset_generator(self) -> None:
        """Clear generator queue y defrag cache entre fases batch grandes.

        Bug crÃ­tico: despuÃ©s de un batch grande (>100 gens), el cache del
        generator queda saturado y los jobs siguientes no pueden allocar
        pÃ¡ginas, produciendo outputs vacÃ­os.
        """
        if self._generator is not None:
            self._generator.clear_queue()
            if hasattr(self._generator, 'pagetable'):
                self._generator.pagetable.defrag()

    def unload(self) -> None:
        """Descarga el modelo y libera VRAM/RAM."""
        import gc
        import torch

        if self._model is not None:
            try:
                self._model.unload()
            except Exception:
                pass
        if self._mtp_model is not None:
            try:
                self._mtp_model.unload()
            except Exception:
                pass
        self._generator = None
        self._mtp_cache = None
        self._mtp_model = None
        self._cache = None
        self._model = None
        self._tokenizer = None
        self._config = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # Liberar el lock de VRAM: Ollama puede volver a cargar.
        from . import vram_lock
        vram_lock.release("exl3")

    def is_loaded(self) -> bool:
        return self._generator is not None and self._model is not None

    def vram_usage_mb(self) -> int:
        import torch
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.memory_allocated() / 1024 / 1024)

    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        if self.no_think and self._family == "chatml":
            return _chatml_no_think(messages)
        return _chatml(messages)

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> GenerationResult:
        """Genera a partir de mensajes chat (single prompt)."""
        prompt = self.format_prompt(messages)
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
            timeout=timeout,
        )

    def generate_stream(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ):
        """Generator that yields text chunks as they are produced."""
        from exllamav3.generator import Job
        from exllamav3.generator.sampler import ComboSampler

        if not self.is_loaded():
            yield {"text": "", "error": "Model not loaded", "done": True}
            return

        max_tokens = max_new_tokens or self.max_output_tokens
        temp = temperature if temperature is not None else self.temperature

        sampler = ComboSampler(logit_bias=self._cjk_bias, rep_p=self.rep_p)
        sampler.temperature = temp
        sampler.top_p = self.top_p if self.top_p < 1.0 else 0.0
        sampler.top_k = self.top_k if self.top_k > 0 else 0
        if self.seed is not None:
            sampler.seed = self.seed

        stops = list(stop_sequences) if stop_sequences else []
        if self._family in ("chatml", "ornith", "lfm"):
            for s in ("<|im_end|>", "</s>", "<|im_start|>", "</think>"):
                if s not in stops:
                    stops.append(s)
        elif self._family == "granite":
            for s in ("<|end_of_text|>", "</s>"):
                if s not in stops:
                    stops.append(s)

        try:
            input_ids = self._tokenizer.encode(prompt, add_bos=False)
            job = Job(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                sampler=sampler,
                stop_conditions=stops,
                identifier=0,
            )
            self._generator.enqueue(job)

            while self._generator.num_remaining_jobs() > 0:
                for result in self._generator.iterate():
                    if result["stage"] == "streaming":
                        chunk = result.get("text", "")
                        if chunk:
                            yield {"text": chunk, "done": False}
                    elif result["stage"] == "end":
                        pass
            yield {"text": "", "done": True}
        except Exception as e:
            yield {"text": "", "error": str(e), "done": True}

    def generate_chat_stream(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ):
        """Generator that yields text chunks from chat messages."""
        prompt = self.format_prompt(messages)
        yield from self.generate_stream(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )

    def generate_chat_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> List[GenerationResult]:
        """Batch generation: formatea N sets de mensajes y los procesa juntos."""
        prompts = [self.format_prompt(msgs) for msgs in batch_messages]
        return self.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
            timeout=timeout,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> GenerationResult:
        """Genera texto a partir de un prompt ya formateado (single prompt)."""
        from exllamav3.generator import Job
        from exllamav3.generator.sampler import ComboSampler

        if not self.is_loaded():
            return GenerationResult(text="", latency_s=0.0, error="Model not loaded")

        vram_before = query_vram_mb()
        monitor = VRAMMonitor(interval_s=2.0)
        monitor.start()

        max_tokens = max_new_tokens or self.max_output_tokens
        temp = temperature if temperature is not None else self.temperature

        sampler = ComboSampler(logit_bias=self._cjk_bias, rep_p=self.rep_p)
        sampler.temperature = temp
        sampler.top_p = self.top_p if self.top_p < 1.0 else 0.0
        sampler.top_k = self.top_k if self.top_k > 0 else 0
        if self.seed is not None:
            sampler.seed = self.seed

        stops = list(stop_sequences) if stop_sequences else []
        if self._family in ("chatml", "ornith", "lfm"):
            for s in ("<|im_end|>", "</s>", "<|im_start|>", "</think>"):
                if s not in stops:
                    stops.append(s)
        elif self._family == "granite":
            for s in ("<|end_of_text|>", "</s>"):
                if s not in stops:
                    stops.append(s)

        try:
            input_ids = self._tokenizer.encode(prompt, add_bos=False)
            job = Job(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                sampler=sampler,
                stop_conditions=stops,
                identifier=0,
            )
            self._generator.enqueue(job)

            result_text = ""
            t_start = time.monotonic()
            ttft_ms = 0.0
            first_token_received = False
            tokens_count = 0

            while self._generator.num_remaining_jobs() > 0:
                for result in self._generator.iterate():
                    if result["stage"] == "streaming":
                        if not first_token_received:
                            ttft_ms = (time.monotonic() - t_start) * 1000
                            first_token_received = True
                        chunk = result.get("text", "")
                        result_text += chunk
                        tokens_count += 1
                    elif result["stage"] == "end":
                        pass

            latency_s = time.monotonic() - t_start

        except Exception as e:
            monitor.stop()
            return GenerationResult(
                text="", latency_s=0.0,
                vram_before_mb=vram_before, vram_after_mb=query_vram_mb(),
                error=str(e),
            )

        monitor.stop()
        vram_after = query_vram_mb()

        for stop in stops:
            if stop in result_text:
                result_text = result_text.split(stop)[0]

        tps = tokens_count / latency_s if latency_s > 0 and tokens_count > 0 else 0.0

        return GenerationResult(
            text=result_text.strip(),
            latency_s=latency_s,
            tokens_generated=tokens_count,
            tokens_per_second=tps,
            time_to_first_token_ms=ttft_ms,
            vram_before_mb=vram_before,
            vram_after_mb=vram_after,
            vram_peak_mb=monitor.peak_mb,
        )

    def generate_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> List[GenerationResult]:
        """Genera mÃºltiples prompts con continuous batching.

        Encola TODOS los jobs de una vez. El generator de ExLlamaV3 maneja el
        scheduling: arranca jobs hasta llenar max_batch_size, y cuando uno
        termina, arranca el siguiente pending automÃ¡ticamente.
        """
        from exllamav3.generator import Job
        from exllamav3.generator.sampler import ComboSampler

        if not self.is_loaded():
            return [GenerationResult(text="", latency_s=0.0, error="Model not loaded")] * len(prompts)

        n = len(prompts)
        if n == 0:
            return []

        # Single prompt â†’ fall back to generate()
        if n == 1:
            return [self.generate(
                prompts[0], max_new_tokens=max_new_tokens,
                temperature=temperature, stop_sequences=stop_sequences,
                timeout=timeout)]

        vram_before = query_vram_mb()
        monitor = VRAMMonitor(interval_s=2.0)
        monitor.start()

        max_tokens = max_new_tokens or self.max_output_tokens
        temp = temperature if temperature is not None else self.temperature

        sampler = ComboSampler(logit_bias=self._cjk_bias, rep_p=self.rep_p)
        sampler.temperature = temp
        sampler.top_p = self.top_p if self.top_p < 1.0 else 0.0
        sampler.top_k = self.top_k if self.top_k > 0 else 0
        if self.seed is not None:
            sampler.seed = self.seed

        stops = list(stop_sequences) if stop_sequences else []
        if self._family in ("chatml", "ornith", "lfm"):
            for s in ("<|im_end|>", "</s>", "<|im_start|>", "</think>"):
                if s not in stops:
                    stops.append(s)
        elif self._family == "granite":
            for s in ("<|end_of_text|>", "</s>"):
                if s not in stops:
                    stops.append(s)

        results = [GenerationResult(text="", latency_s=0.0) for _ in range(n)]
        result_texts = [""] * n
        result_token_counts = [0] * n
        result_ttft = [0.0] * n
        result_latency = [0.0] * n
        first_token_received = [False] * n
        job_done = [False] * n

        try:
            t_start = time.monotonic()
            default_timeout = max(120, 30 * (n // 8 + 1) * 2)
            hard_timeout = timeout or default_timeout
            deadline = t_start + hard_timeout

            for i, prompt in enumerate(prompts):
                input_ids = self._tokenizer.encode(prompt, add_bos=False)
                job = Job(
                    input_ids=input_ids,
                    max_new_tokens=max_tokens,
                    sampler=sampler,
                    stop_conditions=stops,
                    identifier=i,
                )
                self._generator.enqueue(job)

            # Watchdog: iterate() is blocking and may never return if the model
            # gets stuck in prefill (e.g. gated_delta_net recurrent cache saturation
            # with large batches). This thread forcibly clears the generator queue
            # after the hard timeout, causing iterate() to return and the loop to exit.
            watchdog_fired = threading.Event()

            def _watchdog():
                if not watchdog_fired.wait(timeout=hard_timeout):
                    try:
                        self._generator.clear_queue()
                    except Exception:
                        pass
                    watchdog_fired.set()

            wd = threading.Thread(target=_watchdog, daemon=True)
            wd.start()

            while self._generator.num_remaining_jobs() > 0:
                if watchdog_fired.is_set() or time.monotonic() > deadline:
                    raise TimeoutError(f"Generation exceeded {hard_timeout}s timeout (watchdog fired={watchdog_fired.is_set()})")
                for result in self._generator.iterate():
                    idx = result.get("identifier")
                    if idx is None:
                        continue
                    if result["stage"] == "streaming":
                        if not first_token_received[idx]:
                            result_ttft[idx] = (time.monotonic() - t_start) * 1000
                            first_token_received[idx] = True
                        chunk = result.get("text", "")
                        result_texts[idx] += chunk
                        result_token_counts[idx] += 1
                        if result.get("eos"):
                            t_enq = result.get("time_enqueued", 0.0)
                            t_pre = result.get("time_prefill", 0.0)
                            t_gen = result.get("time_generate", 0.0)
                            result_latency[idx] = t_enq + t_pre + t_gen
                            job_done[idx] = True

            watchdog_fired.set()  # signal watchdog to stop waiting
            wall_time = time.monotonic() - t_start

        except Exception as e:
            monitor.stop()
            return [GenerationResult(
                text="", latency_s=0.0,
                vram_before_mb=vram_before, vram_after_mb=query_vram_mb(),
                error=str(e),
            ) for _ in range(n)]

        monitor.stop()
        vram_after = query_vram_mb()

        for i in range(n):
            text = result_texts[i]
            for stop in stops:
                if stop in text:
                    text = text.split(stop)[0]

            lat = result_latency[i] if job_done[i] else wall_time
            gen_time = lat - result_ttft[i] / 1000 if result_ttft[i] > 0 else lat
            tps = result_token_counts[i] / gen_time if gen_time > 0 and result_token_counts[i] > 0 else 0.0

            results[i] = GenerationResult(
                text=text.strip(),
                latency_s=lat,
                tokens_generated=result_token_counts[i],
                tokens_per_second=tps,
                time_to_first_token_ms=result_ttft[i],
                vram_before_mb=vram_before,
                vram_after_mb=vram_after,
                vram_peak_mb=monitor.peak_mb,
            )

        return results

    def config_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "engine": self.engine,
            "quantization": self.quantization,
            "context_length": self.context_length,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "no_think": self.no_think,
            "batch_size": self.batch_size,
            "use_mtp": self.use_mtp,
            "mtp_draft_tokens": self.mtp_draft_tokens,
            "mtp_cache_tokens": self.mtp_cache_tokens,
        }


# ---------------------------------------------------------------------------
# Factory â€” configuraciÃ³n Ã³ptima del modelo estrella
# ---------------------------------------------------------------------------

def create_star_provider(
    model_path: str = "models/Qwen3.5-4B-exl3-4bpw",
    batch_size: int = 4,
    interactive: bool = False,
) -> ExL3Provider:
    """Crea el provider con la configuración óptima del modelo estrella.

    Qwen3.5-4B EXL3 4.0bpw — más chico pero más estable en chat libre.

    NOTA (2026-09, medido): la atribución vieja "el 9B se degrada después de
    ~80 tokens" era en gran parte un artefacto de configuración, no del
    modelo: el cache real quedaba en min(context_length, mtp_cache_tokens) y
    un prompt que cruzaba ese límite a mitad de generación hacía perder el
    inicio del contexto (divagación) o fallaba entero con salida vacía.
    Con el cache alineado al contexto declarado, 27+ generaciones de 300
    tokens con contexto de hasta ~4.4k tokens dieron 0 drift a 24-42 tok/s
    (MTP on). Ver scripts/operations/_exl3_fatigue_test.py.

    batch_size=4 (medido, sweep en RTX 4050 / 6 GB, ctx 2048): throughput
    agregado 81 tok/s con varianza mínima; 6 thrashea (reencola por presión
    de páginas, 17.5 tok/s) y 5 no mejora a 4. Ver
    scripts/operations/_exl3_batch_sweep.py.

    Args:
        model_path: Path al directorio del modelo.
        batch_size: 4 para deliberación/paralelo (sweet spot medido), 1 para
            interactivo.
        interactive: Si True, usa batch_size=1 (modo interactivo).
    """
    if interactive:
        batch_size = 1
    return ExL3Provider(
        model_path=model_path,
        model_id="Qwen3.5-4B-EXL3-4.0bpw",
        quantization="EXL3-4.0bpw",
        # 4B 4.0bpw pesa ~4.2GB. En VRAM con KV cache q8:
        #   6144 ctx: ~4.8 GB peak → margen holgado en 4050 de 6GB
        context_length=6144,
        max_output_tokens=1024,
        temperature=0.1,
        no_think=True,
        batch_size=batch_size,
        use_mtp=False,  # el 4B no tiene MTP
        mtp_draft_tokens=0,
        mtp_cache_tokens=0,
        cache_k_bits=8,
        cache_v_bits=8,
        suppress_cjk=True,
        rep_p=1.15,
    )

