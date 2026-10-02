"""Servicio IPA Push (DEC-011): hub que une store + transporte + config.

``notify()`` es best-effort: si el servicio no corre, el aviso queda encolado
en la DB y se entrega al reconectar el dispositivo. Un fallo de push NUNCA
rompe al llamador (watcher del dashboard) — los errores se loguean.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

from .config import PushConfig
from .store import Store
from .transport import Wake, build_server


def resolve_bind_ip(bind_ip: str) -> str:
    """IP LAN concreta para el bind. "auto" resuelve la interfaz de salida
    (truco UDP connect — no envía paquetes). Fallback: hostname local."""
    if bind_ip and bind_ip != "auto":
        return bind_ip
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


class PushHub:
    """Estado compartido del servicio: store, wake y config."""

    def __init__(self, config: PushConfig | None = None, store: Store | None = None):
        self.config = config or PushConfig.load()
        self.store = store or Store()
        self.wake = Wake()

    def notify(
        self,
        event_type: str,
        title: str,
        body: str = "",
        payload: dict[str, Any] | None = None,
        *,
        priority: int = 1,
        collapse_key: str | None = None,
    ) -> int:
        nid = self.store.enqueue(
            event_type,
            title,
            body,
            payload,
            priority=priority,
            collapse_key=collapse_key,
        )
        self.wake.ping()
        return nid


def notify(
    event_type: str,
    title: str,
    body: str = "",
    payload: dict[str, Any] | None = None,
    *,
    priority: int = 1,
    collapse_key: str | None = None,
) -> int:
    """Encola un aviso en el hub por defecto (best-effort, thread-safe)."""
    return _default_hub().notify(
        event_type, title, body, payload, priority=priority, collapse_key=collapse_key
    )


def notify_research(progress: dict[str, Any], hub: PushHub | None = None) -> bool:
    """Hook del watcher de research (DEC-005): transición done/failed.

    Contenido: SOLO metadatos del evento (DEC-002 — nada de sesiones/episodios
    viaja al móvil). Devuelve True si el evento quedó consumido (encolado o
    deshabilitado por config); False si debe reintentarse.
    """
    hub = hub or _default_hub()
    if not hub.config.events.get("research", True):
        return True
    status = progress.get("status")
    query = str(progress.get("query") or "?")
    result = progress.get("result") or {}
    if status == "done":
        urls = result.get("ingested") if isinstance(result, dict) else None
        if urls is None and isinstance(result, dict):
            urls = result.get("urls") or result.get("count")
        title = f"Investigación lista: {query}"
        body = f"{urls} URLs · material indexado" if urls not in (None, "") else "material indexado"
        event = "research.done"
    elif status == "failed":
        title = f"Investigación falló: {query}"
        body = str(progress.get("error") or "error desconocido")[:200]
        event = "research.failed"
    else:
        return True
    payload = {
        "query": query,
        "status": status,
        "urls_ingested": urls if status == "done" else None,
        "finished_at": progress.get("finished_at"),
    }
    hub.notify(event, title, body, payload, collapse_key=query)
    return True


_default_hub_instance: PushHub | None = None
_default_hub_lock = threading.Lock()


def _default_hub() -> PushHub:
    global _default_hub_instance
    with _default_hub_lock:
        if _default_hub_instance is None:
            _default_hub_instance = PushHub()
        return _default_hub_instance


class PushService:
    """Servidor long-poll mTLS en un thread daemon."""

    def __init__(self, config: PushConfig | None = None, store: Store | None = None):
        self.hub = PushHub(config=config, store=store)
        self._server = None
        self._thread: threading.Thread | None = None

    @property
    def bind_ip(self) -> str:
        return self._bind_ip

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> tuple[str, int]:
        self._bind_ip = resolve_bind_ip(self.hub.config.bind_ip)
        self._server = build_server(self._bind_ip, int(self.hub.config.port), self.hub)
        self._port = self._server.server_address[1]  # real (config puede ser 0)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.5},
            daemon=True,
            name="ipa-push",
        )
        self._thread.start()
        return self._bind_ip, self._port

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def start_service_thread(
    config: PushConfig | None = None, store: Store | None = None
) -> PushService | None:
    """Arranca el servicio si está habilitado. Devuelve None si está
    deshabilitado; reintenta el bind (TIME_WAIT tras restart del dashboard)
    y lanza si falla definitivo (el llamador degrada sin push)."""
    cfg = config or PushConfig.load()
    if not cfg.enabled:
        return None
    service = PushService(config=cfg, store=store)
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            service.start()
            return service
        except OSError as exc:
            last_exc = exc
            time.sleep(2.0 * (attempt + 1))
    raise last_exc  # type: ignore[misc]
