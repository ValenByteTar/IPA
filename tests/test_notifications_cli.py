"""Tests del CLI de IPA Push (DEC-011): init-ca, pair, devices, revoke, status.

Corren el CLI real como subprocess con IPA_NOTIFICATIONS_DIR aislado —
valida el flujo de emparejamiento completo que ejecutará el usuario.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ipa.notifications.store import Store

CLI = Path(__file__).resolve().parents[1] / "scripts" / "cli" / "notifications.py"


def _run(args: list[str], notif_dir: Path) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env["IPA_NOTIFICATIONS_DIR"] = str(notif_dir)
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


@pytest.fixture()
def notif_dir(tmp_path, monkeypatch):
    d = tmp_path / "notif"
    monkeypatch.setenv("IPA_NOTIFICATIONS_DIR", str(d))
    return d


def test_pair_flow_completo(notif_dir):
    # pair sin slot → ok, imprime material completo
    r = _run(["pair", "--name", "Pixel"], notif_dir)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "Emparejamiento completado" in out
    assert "https://" in out
    assert "huella CA" in out
    assert "client-pixel.p12" in out
    assert "contraseña" in out
    # El slot quedó grabado en la DB aislada (IPA_NOTIFICATIONS_DIR) —
    # verificamos vía el CLI devices (mismo store)
    r2 = _run(["devices"], notif_dir)
    assert r2.returncode == 0
    # pair de nuevo sin --replace → error explícito
    r3 = _run(["pair", "--name", "Otro"], notif_dir)
    assert r3.returncode == 1
    assert "ocupada" in r3.stderr
    # pair --replace → desaloja y re-empareja
    r4 = _run(["pair", "--name", "Otro", "--replace"], notif_dir)
    assert r4.returncode == 0, r4.stderr
    # revoke libera
    r5 = _run(["revoke"], notif_dir)
    assert r5.returncode == 0
    assert "revocado" in r5.stdout
    # devices sin slot → mensaje claro
    r6 = _run(["devices"], notif_dir)
    assert "Sin dispositivo" in r6.stdout


def test_status_y_send(notif_dir):
    assert _run(["init-ca"], notif_dir).returncode == 0
    r = _run(
        ["send", "--event", "research.done", "--title", "prueba", "--body", "hola"],
        notif_dir,
    )
    assert r.returncode == 0, r.stderr
    assert "notification id=1" in r.stdout
    r2 = _run(["status"], notif_dir)
    assert r2.returncode == 0
    assert '"total": 1' in r2.stdout
