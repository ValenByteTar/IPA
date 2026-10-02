"""Tests del material TLS de IPA Push (DEC-011): CA, server cert, client .p12."""

from __future__ import annotations

from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives.serialization import pkcs12

from ipa.notifications import tls_material


@pytest.fixture()
def tls_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("IPA_NOTIFICATIONS_DIR", str(tmp_path / "notif"))
    return tls_material.material_dir()


def test_init_ca_idempotente(tls_dir):
    cert1, ca_crt, ca_key = tls_material.init_ca()
    fp1 = tls_material._fingerprint(cert1)
    assert ca_crt.exists() and ca_key.exists()
    cert2, _, _ = tls_material.init_ca()
    assert tls_material._fingerprint(cert2) == fp1  # no regenera
    assert tls_material.ca_fingerprint() == fp1


def test_init_ca_force_regenera(tls_dir):
    cert1, _, _ = tls_material.init_ca()
    fp1 = tls_material._fingerprint(cert1)
    cert2, _, _ = tls_material.init_ca(force=True)
    assert tls_material._fingerprint(cert2) != fp1


def test_server_cert_san_ip_y_reuso(tls_dir):
    tls_material.init_ca()
    cert1, crt, key = tls_material.ensure_server_cert("127.0.0.1")
    sans = cert1.extensions.get_extension_for_class(
        x509.SubjectAlternativeName
    ).value
    assert "127.0.0.1" in {str(v) for v in sans.get_values_for_type(x509.IPAddress)}
    assert tls_material._fingerprint(cert1) == tls_material._fingerprint(
        x509.load_pem_x509_certificate(crt.read_bytes())
    )
    # Misma IP → reutiliza; IP distinta → regenera
    cert2, _, _ = tls_material.ensure_server_cert("127.0.0.1")
    assert tls_material._fingerprint(cert2) == tls_material._fingerprint(cert1)
    cert3, _, _ = tls_material.ensure_server_cert("192.168.1.50")
    assert tls_material._fingerprint(cert3) != tls_material._fingerprint(cert1)
    # Firmado por la CA (issuer == subject de la CA)
    ca_cert, _, _ = tls_material.init_ca()
    assert cert3.issuer == ca_cert.subject


def test_issue_client_p12_y_huella(tls_dir):
    tls_material.init_ca()
    p12_path, password, fingerprint = tls_material.issue_client("Pixel")
    assert p12_path.exists() and p12_path.name == "client-pixel.p12"
    key, cert, cas = pkcs12.load_key_and_certificates(
        p12_path.read_bytes(), password.encode()
    )
    assert key is not None and cert is not None and cas
    assert tls_material._fingerprint(cert) == fingerprint
    # Contraseña incorrecta no abre
    with pytest.raises(Exception):
        pkcs12.load_key_and_certificates(p12_path.read_bytes(), b"wrong")


def test_revocacion(tls_dir):
    tls_material.init_ca()
    _, _, fingerprint = tls_material.issue_client("Pixel")
    assert not tls_material.is_revoked(fingerprint)
    tls_material.revoke_fingerprint(fingerprint)
    assert tls_material.is_revoked(fingerprint)
    # Persiste en disco
    assert fingerprint in (tls_dir / "revoked.json").read_text()


def test_material_dir_respeta_env(tmp_path, monkeypatch):
    monkeypatch.setenv("IPA_NOTIFICATIONS_DIR", str(tmp_path / "custom"))
    assert tls_material.material_dir() == tmp_path / "custom" / "tls"
    monkeypatch.delenv("IPA_NOTIFICATIONS_DIR")
    assert tls_material.material_dir().name == "tls"
