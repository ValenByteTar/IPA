"""Configuración de IPA Push (DEC-011): configs/notifications.yaml.

Sin secretos en el YAML — todo el material TLS vive en
``outputs/notifications/tls/``. Defaults operativos si falta el archivo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "notifications.yaml"


@dataclass
class PushConfig:
    enabled: bool = True
    bind_ip: str = "auto"          # "auto" resuelve la IP LAN al arranque
    port: int = 8766
    max_connections: int = 5
    poll_timeout_s: int = 25
    max_backlog: int = 50
    retention_days: int = 7
    events: dict[str, bool] = field(default_factory=lambda: {"research": True})

    @classmethod
    def load(cls, path: Path | str | None = None) -> "PushConfig":
        cfg_path = Path(path) if path else DEFAULT_CONFIG
        data: dict[str, Any] = {}
        if cfg_path.exists():
            try:
                data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            except ValueError:
                data = {}
        push = data.get("push") or {}
        events = data.get("events") or {}
        return cls(
            enabled=bool(push.get("enabled", True)),
            bind_ip=str(push.get("bind_ip", "auto")),
            port=int(push.get("port", 8766)),
            max_connections=int(push.get("max_connections", 5)),
            poll_timeout_s=int(push.get("poll_timeout_s", 25)),
            max_backlog=int(push.get("max_backlog", 50)),
            retention_days=int(push.get("retention_days", 7)),
            events={"research": bool(events.get("research", True))},
        )
