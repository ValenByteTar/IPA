"""Transporte de IPA Push (DEC-011): HTTP long-poll sobre TLS con mTLS.

Superficie mínima (v1 congelado): 3 rutas. El handshake TLS exige certificado
de cliente firmado por nuestra CA (``CERT_REQUIRED``) — sin cert válido no hay
ruta HTTP que atacar. Encima del TLS, el fingerprint sha256 del cert cliente
debe coincidir con el del slot único (403 si no). Límites duros anti-exhaustión
y todo rechazo queda logueado con IP y motivo.
"""

from __future__ import annotations

import hashlib
import json
import ssl
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import tls_material

MAX_BODY = 64 * 1024
READ_TIMEOUT_S = 90


class Wake:
    """Señal de eventos nuevos para los pollers colgados (long-poll)."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._version = 0

    def ping(self) -> None:
        with self._cond:
            self._version += 1
            self._cond.notify_all()

    def wait(self, timeout_s: float) -> bool:
        """True si hubo eventos nuevos durante la espera."""
        with self._cond:
            start = self._version
            self._cond.wait(timeout_s)
            return self._version != start


class PushHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ipa-push/0.1"
    timeout = READ_TIMEOUT_S

    # Referencias inyectadas por build_server(): hub (store+wake+config) y
    # registro de pollers activos por fingerprint.
    hub: Any = None
    pollers: Any = None  # dict[fingerprint, int] con lock propio

    # ── utilidades ───────────────────────────────────────────────────
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        print(f"[push] {self.address_string()} {fmt % args}", flush=True)

    def _client_fingerprint(self) -> str | None:
        der = self.connection.getpeercert(binary_form=True)
        if not der:
            return None
        return hashlib.sha256(der).hexdigest()

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _reject(self, fingerprint: str | None, reason: str) -> None:
        print(
            f"[push] RECHAZADO ip={self.address_string()} fp={fingerprint}"
            f" motivo={reason}",
            flush=True,
        )

    def _authorize(self) -> str | None:
        """Devuelve el fingerprint autorizado o None (ya respondió)."""
        fp = self._client_fingerprint()
        if not fp:
            self._reject(None, "sin certificado de cliente")
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "client_cert_required"})
            return None
        if tls_material.is_revoked(fp):
            self._reject(fp, "certificado revocado")
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "revoked"})
            return None
        slot = self.hub.store.slot()
        if slot is None:
            self._reject(fp, "sin slot emparejado")
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "not_paired"})
            return None
        if fp != slot.get("cert_fingerprint"):
            self._reject(fp, "slot ocupado por otro certificado")
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "slot_occupied"})
            return None
        return fp

    # ── rutas ────────────────────────────────────────────────────────
    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/push/v1/health":
                if self._authorize() is None:
                    return
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "server": self.server_version},
                )
                return
            if parsed.path == "/api/push/v1/poll":
                self._handle_poll(parse_qs(parsed.query))
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
            pass
        except Exception as exc:
            try:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)}
                )
            except Exception:
                pass

    def do_POST(self) -> None:  # noqa: N003
        try:
            if urlparse(self.path).path != "/api/push/v1/ack":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            fp = self._authorize()
            if fp is None:
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self._reject(fp, f"body {length}B > {MAX_BODY}B")
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "body_too_large"})
                return
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8"))
                ids = [int(i) for i in (data.get("ids") or [])]
            except (ValueError, UnicodeDecodeError):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "malformed"})
                return
            acked = self.hub.store.ack(ids)
            self.hub.store.touch()
            self._send_json(HTTPStatus.OK, {"acked": acked})
        except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
            pass
        except Exception as exc:
            try:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)}
                )
            except Exception:
                pass

    # ── long-poll ────────────────────────────────────────────────────
    def _handle_poll(self, params: dict[str, list[str]]) -> None:
        fp = self._authorize()
        if fp is None:
            return
        try:
            since = int((params.get("since") or ["0"])[0])
            timeout = min(float((params.get("timeout") or ["0"])[0]), 60.0)
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "malformed"})
            return
        timeout = max(timeout, 0.0)

        # Límites: 1 poller por fingerprint, N conexiones totales.
        with self.pollers["lock"]:
            total = sum(self.pollers["by_fp"].values())
            if self.pollers["by_fp"].get(fp, 0) >= 1 or total >= self.hub.config.max_connections:
                self._reject(fp, "límite de conexiones")
                self._send_json(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    {"error": "too_many_connections"},
                )
                return
            self.pollers["by_fp"][fp] = self.pollers["by_fp"].get(fp, 0) + 1
        try:
            events = self.hub.store.next_batch(since, self.hub.config.max_backlog)
            if not events and timeout > 0:
                deadline = time.monotonic() + timeout
                while not events:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    if self.hub.wake.wait(min(remaining, 5.0)):
                        events = self.hub.store.next_batch(
                            since, self.hub.config.max_backlog
                        )
            if events:
                self.hub.store.mark_sent([e["id"] for e in events])
            self.hub.store.touch()
            self._send_json(
                HTTPStatus.OK,
                {"events": events, "last_id": (events[-1]["id"] if events else since)},
            )
        finally:
            with self.pollers["lock"]:
                self.pollers["by_fp"][fp] -= 1
                if self.pollers["by_fp"][fp] <= 0:
                    self.pollers["by_fp"].pop(fp, None)


def build_server(bind_ip: str, port: int, hub: Any) -> ThreadingHTTPServer:
    """Servidor HTTPS con mTLS: exige cert cliente firmado por nuestra CA."""
    tls_material.init_ca()
    tls_material.ensure_server_cert(bind_ip)
    d = tls_material.material_dir()
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=d / "server.crt", keyfile=d / "server.key")
    context.load_verify_locations(cafile=d / "ca.crt")
    context.verify_mode = ssl.CERT_REQUIRED

    # Inyección de estado compartido en el handler (1 servicio por proceso).
    PushHandler.hub = hub
    PushHandler.pollers = {"lock": threading.Lock(), "by_fp": {}}

    class _Server(ThreadingHTTPServer):
        daemon_threads = True
        # Sin SO_REUSEADDR: en Windows permite doble-bind silencioso (dos
        # procesos compartiendo el puerto). Que el segundo bind falle ruidoso
        # es el comportamiento correcto — start_service_thread reintenta.
        allow_reuse_address = False

    server = _Server((bind_ip, port), PushHandler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    return server
