"""OllamaAdapter â€” local LLM client via Ollama HTTP API.

Thin wrapper around Ollama's /api/chat endpoint.  Supports:
  - think=False (disables reasoning mode for Qwen3 and similar models)
  - Configurable timeout and retries
  - Batch generation (one call per item, but concurrent via threads)

Used by E9 (semantic enrichment) to generate synthetic queries, extract
claims, and summarize chunks.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    """Response from a single LLM call."""
    text: str
    latency_seconds: float
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OllamaAdapter:
    """Client for Ollama's local HTTP API.

    Ollama must be running (default: http://localhost:11434).
    Models must be pre-pulled via `ollama pull <model>`.
    """

    def __init__(
        self,
        model: str = "qwen3.5:4b-q4_K_M",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        think: bool = False,
        temperature: float = 0.3,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.think = think
        self.temperature = temperature
        self.max_retries = max_retries

    def generate(
        self,
        prompt: str,
        system: str | None = None,
    ) -> LLMResponse:
        """Send a single prompt and return the response.

        Args:
            prompt: User message content.
            system: Optional system message for instruction tuning.

        Returns:
            LLMResponse with text, latency, and token counts.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": {"temperature": self.temperature},
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                t0 = time.monotonic()
                resp = urllib.request.urlopen(req, timeout=self.timeout)
                data = json.loads(resp.read())
                elapsed = time.monotonic() - t0
                return LLMResponse(
                    text=data["message"]["content"].strip(),
                    latency_seconds=elapsed,
                    model=self.model,
                    prompt_tokens=data.get("prompt_eval_count", 0),
                    completion_tokens=data.get("eval_count", 0),
                )
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))
                continue

        raise ConnectionError(f"Ollama request failed after {self.max_retries + 1} attempts: {last_error}")

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            return self.model in models
        except Exception:
            return False

    def close(self) -> None:
        """No persistent connection to close."""
        pass

    def __enter__(self) -> "OllamaAdapter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

