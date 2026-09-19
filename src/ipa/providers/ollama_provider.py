"""OllamaProvider — local LLM via Ollama HTTP API with streaming.

Implementa la misma interfaz que ExL3Provider para que el dashboard pueda
usarlo como drop-in replacement. Ollama usa GGUF con K-quants que suelen ser
más estables que EXL3 para chat libre.

Modelo recomendado: qwen3.5:9b-q4_K_M (6.6 GB, Q4_K_M).
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Optional, List


def _perf_log_path() -> Path:
    return Path(os.environ.get(
        "IPA_LLM_PERF_LOG", "outputs/web_dashboard/logs/llm_perf.jsonl"))


def _log_perf(model: str, data: dict) -> None:
    """Append one JSONL line per completed generation with Ollama's own
    metrics. prompt_eval_cached_count es la métrica DIRECTA del PT cache:
    cuántos tokens del prompt se reusaron del slot en vez de re-evaluarse."""
    try:
        path = _perf_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": model,
            "prompt_eval_count": data.get("prompt_eval_count"),
            "prompt_eval_cached_count": data.get("prompt_eval_cached_count"),
            "eval_count": data.get("eval_count"),
            "prompt_eval_ms": round((data.get("prompt_eval_duration") or 0) / 1e6),
            "eval_ms": round((data.get("eval_duration") or 0) / 1e6),
            "total_ms": round((data.get("total_duration") or 0) / 1e6),
        })
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


class OllamaProvider:
    """Provider que habla con Ollama's /api/chat endpoint con streaming."""

    def __init__(
        self,
        model: str = "qwen3.5:9b-q4_K_M",
        base_url: str = "http://localhost:11434",
        context_length: int = 8192,
        max_output_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 40,
        seed: Optional[int] = None,
        no_think: bool = True,
        rep_p: float = 1.15,
        keep_alive: Optional[str] = None,
        num_keep: Optional[int] = None,
        repeat_last_n: Optional[int] = None,
        num_batch: Optional[int] = None,
    ) -> None:
        self.model = model
        self.model_id = model
        self.engine = "ollama"
        self.quantization = "GGUF-Q4_K_M"
        self.base_url = base_url.rstrip("/")
        self.context_length = context_length
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.no_think = no_think
        self.rep_p = rep_p
        # keep_alive: cuánto retiene Ollama el modelo cargado tras cada request.
        # Default 30m: cubre sesiones interactivas espaciadas sin pinneo eterno
        # de VRAM (ExL3 necesita esa memoria cuando se usa). -1 = nunca
        # descargar; 0 = descargar al terminar la request.
        self.keep_alive = keep_alive or os.environ.get(
            "IPA_OLLAMA_KEEP_ALIVE", "30m")
        # num_keep: tokens del INICIO que Ollama preserva cuando el contexto se
        # llena. Default de llama.cpp = 4 → al desbordar, el system prompt
        # (identidad, grounding, user model) es lo PRIMERO en evaporarse y el
        # modelo pierde el hilo. 2048 ≈ tamaño del system prompt real.
        _nk = num_keep if num_keep is not None else os.environ.get("IPA_OLLAMA_NUM_KEEP")
        self.num_keep = int(_nk) if _nk else 2048
        # repeat_last_n: ventana hacia atrás del repetition penalty (default
        # llama.cpp = 64). Más ancha = más contexto penalizado.
        _rln = repeat_last_n if repeat_last_n is not None else os.environ.get("IPA_OLLAMA_REPEAT_LAST_N")
        self.repeat_last_n = int(_rln) if _rln else None
        # num_batch: batch de prefill (default servidor = OLLAMA_NUM_BATCH).
        _nb = num_batch if num_batch is not None else os.environ.get("IPA_OLLAMA_NUM_BATCH")
        self.num_batch = int(_nb) if _nb else None
        self._loaded = False

    def load(self) -> None:
        """Verify Ollama is running and the model is available.

        If the server is not responding, try to spawn ``ollama serve``
        detached and wait for readiness — the dashboard should not depend
        on Ollama having been started manually.

        Si ExL3 tiene el lock de VRAM (batch en curso), falla claro: cargar
        Ollama encima en 6 GB mata a ExL3 con OOM. Ollama NO toma el lock:
        su modelo queda cargado con keep_alive y tomarlo bloquearía a ExL3
        para siempre — solo lo respeta.
        """
        from . import vram_lock
        h = vram_lock.holder()
        if h is not None and h.get("owner") != "ollama":
            raise RuntimeError(
                f"VRAM ocupada por {h.get('owner', '?')} (pid {h.get('pid', '?')}): "
                "el chat queda sin backend hasta que termine el batch ExL3")
        self._ensure_server()
        try:
            resp = urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=10)
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            if self.model not in models:
                raise RuntimeError(f"Model {self.model} not found in Ollama. Available: {models}")
            self._loaded = True
        except Exception as e:
            raise RuntimeError(f"Ollama not available: {e}")

    def _ensure_server(self) -> None:
        """Start ``ollama serve`` if the API is not responding."""
        from urllib.parse import urlparse
        parsed = urlparse(self.base_url)
        if parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            return  # remote server — not ours to start
        try:
            urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3)
            return  # already running
        except Exception:
            pass
        # If the Ollama tray app is already running, its own server brings
        # the API up shortly — spawning `ollama serve` ourselves makes its
        # llama-server.exe grandchild pop a visible console window on every
        # model load/unload (our CREATE_NO_WINDOW only hides the direct
        # child). Poll the API instead of launching a duplicate server.
        if self._tray_app_running():
            for _ in range(120):  # up to ~60s for the tray server to come up
                try:
                    urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3)
                    return
                except Exception:
                    time.sleep(0.5)
            raise RuntimeError("Ollama tray app running but API never came up")
        import shutil
        import subprocess
        exe = shutil.which("ollama")
        if not exe:
            default = os.path.expandvars(
                r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
            exe = default if os.path.exists(default) else None
        if not exe:
            raise RuntimeError(
                "Ollama server not running and 'ollama' executable not found")
        kwargs: dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
        }
        if os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW)
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen([exe, "serve"], **kwargs)
        for _ in range(60):
            try:
                urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3)
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("Ollama server did not become ready in 30s")

    @staticmethod
    def _tray_app_running() -> bool:
        """Is the Ollama tray app (ollama app.exe) alive? Its managed server
        spawns llama-server hidden — no console popups."""
        if os.name != "nt":
            return False
        try:
            import subprocess
            out = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq ollama app.exe", "/NH"],
                capture_output=True, text=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            ).stdout
            return "ollama" in out.lower()
        except Exception:
            return False

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._loaded = False

    def _build_options(self, max_new_tokens: int, temperature: float | None) -> dict:
        temp = temperature if temperature is not None else self.temperature
        options = {
            "temperature": temp,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_predict": max_new_tokens,
            "repeat_penalty": self.rep_p,
            "num_ctx": self.context_length,
            "num_keep": self.num_keep,
        }
        if self.repeat_last_n is not None:
            options["repeat_last_n"] = self.repeat_last_n
        if self.num_batch is not None:
            options["num_batch"] = self.num_batch
        # num_gpu: el auto-fit de Ollama es conservador (deja ~1.4 GB libres y
        # manda la mitad del modelo a CPU). Forzarlo sube el decode ~2x en la
        # RTX 4050 (18/34 → 30/34 capas, 10.4 → 20 tok/s). Configurable por
        # máquina: sin la var, Ollama decide solo.
        n_gpu = os.environ.get("IPA_OLLAMA_NUM_GPU", "").strip()
        if n_gpu.isdigit():
            options["num_gpu"] = int(n_gpu)
        return options

    def _build_body(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float | None,
        stop_sequences: list[str] | None,
    ) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": not self.no_think,
            "keep_alive": self.keep_alive,
            "options": self._build_options(max_new_tokens, temperature),
        }
        if stop_sequences:
            body["stop"] = stop_sequences
        return body

    def generate_chat_stream(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ):
        """Stream tokens from Ollama. Yields {"text": ..., "done": bool}."""
        if not self.is_loaded():
            yield {"text": "", "error": "model not loaded", "done": True}
            return
        # Un request HTTP dispara la carga del modelo server-side aunque el
        # provider ya esté "loaded": si ExL3 tiene el lock, cortamos acá para
        # no matarlo con OOM.
        from . import vram_lock
        _h = vram_lock.holder()
        if _h is not None and _h.get("owner") not in (None, "ollama"):
            yield {"text": "", "done": True,
                   "error": f"GPU ocupada por {_h.get('owner')} (batch ExL3 en curso)"}
            return

        max_tokens = max_new_tokens or self.max_output_tokens
        stops = list(stop_sequences) if stop_sequences else []
        # ChatML stop tokens (same as ExL3)
        for s in ("<|im_end|>", "</s>", "<|im_start|>"):
            if s not in stops:
                stops.append(s)

        body = self._build_body(messages, max_tokens, temperature, stops)
        body_json = json.dumps(body).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body_json,
            headers={"Content-Type": "application/json"},
        )

        try:
            import http.client
            from urllib.parse import urlparse
            parsed = urlparse(self.base_url)
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=300)
            conn.request("POST", "/api/chat", body=body_json, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except (ValueError, json.JSONDecodeError):
                    continue
                if data.get("done"):
                    _log_perf(self.model, data)
                    yield {
                        "text": "", "done": True,
                        "metrics": {
                            "prompt_eval_count": data.get("prompt_eval_count"),
                            "prompt_eval_cached_count": data.get("prompt_eval_cached_count"),
                            "eval_count": data.get("eval_count"),
                            "prompt_eval_ms": round((data.get("prompt_eval_duration") or 0) / 1e6),
                            "eval_ms": round((data.get("eval_duration") or 0) / 1e6),
                        },
                    }
                    conn.close()
                    return
                msg = data.get("message", {})
                text = msg.get("content", "")
                if text:
                    yield {"text": text, "done": False}
            yield {"text": "", "done": True}
            conn.close()
        except Exception as e:
            yield {"text": "", "error": str(e), "done": True}

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Non-streaming version. Returns full text."""
        text = ""
        for chunk in self.generate_chat_stream(
            messages, max_new_tokens, temperature, stop_sequences
        ):
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            if chunk.get("text"):
                text += chunk["text"]
        return text

    def status(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "model": self.model,
            "loaded": self._loaded,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }


def create_star_provider(
    model: str = "qwen3.5:9b-q4_K_M",
    interactive: bool = False,
) -> OllamaProvider:
    """Crea el provider de Ollama con la configuración óptima.

    Q4_K_M de Ollama usa K-quants que preservan mejor las capas de atención
    que EXL3, resultando en chat libre más estable.

    Args:
        model: Nombre del modelo en Ollama.
        interactive: Sin efecto (compatibilidad con ExL3 interface).
    """
    # num_ctx 6144: los prompts reales del chat miden 2.3-4.1k tokens; 6144
    # cubre con margen y libera ~68 MiB de KV por slot (q8) → más margen de
    # VRAM para capas GPU. Override: IPA_OLLAMA_NUM_CTX.
    _ctx = os.environ.get("IPA_OLLAMA_NUM_CTX", "").strip()
    return OllamaProvider(
        model=model,
        context_length=int(_ctx) if _ctx.isdigit() else 6144,
        max_output_tokens=512,
        temperature=0.1,
        top_p=0.9,
        top_k=40,
        no_think=True,
        rep_p=1.15,
    )
