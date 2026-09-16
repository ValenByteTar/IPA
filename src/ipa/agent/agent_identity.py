"""Identity loader for the personal agent.

`configs/agent_identity.yaml` is the source of truth of the personality. It is
configuration (not data): its audit trail is the git history. Every episode
records the sha256 of the active identity file (identity_hash) so memory is
linked to the identity version active at each turn (DEC-002).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_IDENTITY_PATH = Path("configs") / "agent_identity.yaml"


@dataclass(frozen=True)
class Identity:
    """Loaded agent identity with its content hash."""

    name: str
    user: str
    language: str
    persona: str
    principles: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    skills: list[dict[str, str]] = field(default_factory=list)
    identity_hash: str = ""

    def system_prompt(self, *, role: str = "general", include_user_model: bool = True) -> str:
        """Build the base system prompt. Extending the YAML changes this
        output without touching code — that is the Fase 0 behavior gate.

        When include_user_model is True, dynamic context layers are appended:
        user model (goals, interests, preferences), strategic principles,
        learned skills, and uncertainty topics. Each layer is independent
        and renders empty if no data is available (no-op on fresh installs).
        """
        role_line = {
            "general": "Soy el asistente personal del usuario en una conversación abierta.",
            "tutor": "Actúo como tutor pedagógico: diagnostico antes de explicar, propongo el próximo paso y evalúo con evidencia.",
        }.get(role, "")
        principles = "\n".join(f"- {principle}" for principle in self.principles)
        parts = [
            f"Soy {self.name}, el agente personal de {self.user}.",
            self.persona.strip(),
            f"Idioma: {self.language}.",
        ]
        if role != "general":
            parts.append(f"Rol activo: {role}. {role_line}")
        if principles:
            parts.append(f"Principios:\n{principles}")
        if self.capabilities:
            parts.append("Capacidades disponibles: " + ", ".join(self.capabilities) + ".")
        if self.skills:
            flows = "\n".join(
                f"- {s.get('name', '?')}: {s.get('flow', '')}" for s in self.skills
            )
            parts.append(f"Flujos típicos (cómo combinar herramientas):\n{flows}")
        if include_user_model:
            dynamic = self._render_dynamic_context()
            if dynamic:
                parts.append(dynamic)
        return "\n\n".join(parts)

    def _render_dynamic_context(self) -> str:
        """Render dynamic context layers (user model, principles, skills, uncertainty).

        Each layer is wrapped in try/except: a failure in one layer never
        breaks the system prompt. Layers render empty if stores are empty
        or unavailable (fresh install, no data yet).
        """
        layers: list[str] = []
        # User model (goals, interests, preferences, facts)
        try:
            from ipa.agent.user_model import UserModelStore, render_user_model_context
            store = UserModelStore()
            try:
                ctx = render_user_model_context(store)
                if ctx:
                    layers.append(ctx)
            finally:
                store.close()
        except Exception:
            pass
        # Strategic principles (learned from usage)
        try:
            from ipa.agent.strategic_memory import StrategicMemoryStore, render_active_principles
            store = StrategicMemoryStore()
            try:
                ctx = render_active_principles(store)
                if ctx:
                    layers.append(ctx)
            finally:
                store.close()
        except Exception:
            pass
        # Learned skills (repeated tool patterns)
        try:
            from ipa.agent.skill_library import SkillLibraryStore, render_active_skills
            store = SkillLibraryStore()
            try:
                ctx = render_active_skills(store)
                if ctx:
                    layers.append(ctx)
            finally:
                store.close()
        except Exception:
            pass
        # Uncertainty (low-confidence topics)
        try:
            from ipa.agent.uncertainty import UncertaintyStore, render_uncertainty_context
            store = UncertaintyStore()
            try:
                ctx = render_uncertainty_context(store)
                if ctx:
                    layers.append(ctx)
            finally:
                store.close()
        except Exception:
            pass
        # Real system state (corpus, user model, tutor, tasks) — grounds
        # self-knowledge: the model sees what it actually has, not just
        # tool names.
        try:
            from ipa.agent.system_state import render_system_state
            ctx = render_system_state()
            if ctx:
                layers.append(ctx)
        except Exception:
            pass
        return "\n\n".join(layers) if layers else ""


def load_identity(path: str | Path | None = None) -> Identity:
    """Load and hash the agent identity. Raises FileNotFoundError if missing."""
    identity_path = Path(path) if path else DEFAULT_IDENTITY_PATH
    raw = identity_path.read_bytes()
    data: dict[str, Any] = yaml.safe_load(raw.decode("utf-8")) or {}
    identity_hash = "sha256:" + hashlib.sha256(raw).hexdigest()
    return Identity(
        name=str(data.get("name", "Personal AGI")),
        user=str(data.get("user", "usuario")),
        language=str(data.get("language", "español")),
        persona=str(data.get("persona", "")),
        principles=[str(item) for item in data.get("principles", [])],
        capabilities=[str(item) for item in data.get("capabilities", [])],
        skills=[dict(item) for item in data.get("skills", []) if isinstance(item, dict)],
        identity_hash=identity_hash,
    )


__all__ = ["DEFAULT_IDENTITY_PATH", "Identity", "load_identity"]
