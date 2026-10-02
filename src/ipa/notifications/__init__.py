"""IPA Push — notificaciones propias al móvil (DEC-011).

Bounded context separado del dashboard y del agent core. Superficie pública:
``notify_research`` (hook del watcher) y ``PushService`` (servidor long-poll
mTLS). Contenido de los avisos: SOLO metadatos del evento (DEC-002).
"""

from .service import PushHub, PushService, notify, notify_research, start_service_thread

__all__ = [
    "PushHub",
    "PushService",
    "notify",
    "notify_research",
    "start_service_thread",
]
