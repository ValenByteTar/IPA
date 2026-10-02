"""Material TLS propio de IPA Push (DEC-011): CA local + server cert + client .p12.

Todo vive en ``outputs/notifications/tls/`` (cubierto por ``outputs/*`` en
``.gitignore`` — nada de claves en el repo). Generación con ``cryptography``,
ya presente en el venv como dependencia transitiva de google-auth/pdfminer.six
— sin dependencias nuevas.

Modelo de confianza: la app Android fija la huella sha256 de ``ca.crt`` (no el
hostname), así un cambio de IP LAN solo exige regenerar el server cert — la
CA no cambia. La identidad del teléfono es posesión de la clave privada del
cert cliente (nunca sale del dispositivo salvo en el .p12 del emparejamiento).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import ipaddress
import json
import os
import secrets
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import pkcs12
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE = ROOT / "outputs" / "notifications"

CA_CN = "IPA Push CA"
CA_DAYS = 365 * 10
SERVER_DAYS = 365 * 5
CLIENT_DAYS = 365 * 5


def material_dir() -> Path:
    override = os.environ.get("IPA_NOTIFICATIONS_DIR")
    base = Path(override) if override else DEFAULT_BASE
    return base / "tls"


def _fingerprint(cert: x509.Certificate) -> str:
    return hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()


def _name(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def _write(cert_path: Path, key_path: Path, cert, key) -> None:
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )


def _load_ca() -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    d = material_dir()
    cert = x509.load_pem_x509_certificate((d / "ca.crt").read_bytes())
    key = serialization.load_pem_private_key(
        (d / "ca.key").read_bytes(), password=None
    )
    return cert, key


def init_ca(force: bool = False) -> tuple[x509.Certificate, Path, Path]:
    """CA local de IPA Push (idempotente, self-signed). (cert, ca.crt, ca.key)."""
    d = material_dir()
    d.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = d / "ca.crt", d / "ca.key"
    if force:
        cert_path.unlink(missing_ok=True)
        key_path.unlink(missing_ok=True)
    if cert_path.exists() and key_path.exists():
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        return cert, cert_path, key_path
    key = ec.generate_private_key(ec.SECP256R1())
    now = _dt.datetime.now(_dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(_name(CA_CN))
        .issuer_name(_name(CA_CN))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + _dt.timedelta(days=CA_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(key, hashes.SHA256())
    )
    _write(cert_path, key_path, cert, key)
    return cert, cert_path, key_path


def ca_fingerprint() -> str:
    cert, _, _ = init_ca()
    return _fingerprint(cert)


def ensure_server_cert(bind_ip: str) -> tuple[x509.Certificate, Path, Path]:
    """Server cert firmado por la CA con SAN = IP LAN. Si la IP cambió o
    expiró se regenera (la app fija la CA — no el hostname — así que no hay
    que re-emparejar nada)."""
    d = material_dir()
    cert_path, key_path = d / "server.crt", d / "server.key"
    if cert_path.exists() and key_path.exists():
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        try:
            sans = cert.extensions.get_extension_for_class(
                x509.SubjectAlternativeName
            ).value
            ips = {str(v) for v in sans.get_values_for_type(x509.IPAddress)}
        except x509.ExtensionNotFound:
            ips = set()
        expired = cert.not_valid_after_utc < _dt.datetime.now(_dt.timezone.utc)
        if bind_ip in ips and not expired:
            key = serialization.load_pem_private_key(
                key_path.read_bytes(), password=None
            )
            return cert, cert_path, key
        cert_path.unlink()
        key_path.unlink(missing_ok=True)
    ca_cert, ca_key = _load_ca()
    key = ec.generate_private_key(ec.SECP256R1())
    now = _dt.datetime.now(_dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(_name("ipa-push"))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + _dt.timedelta(days=SERVER_DAYS))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.IPAddress(ipaddress.ip_address(bind_ip)),
                    x509.DNSName("ipa-push.local"),
                    x509.DNSName("localhost"),
                ]
            ),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )
    _write(cert_path, key_path, cert, key)
    return cert, cert_path, key_path


def issue_client(name: str) -> tuple[Path, str, str]:
    """Emite el cert cliente del slot y lo empaqueta como .p12.

    Devuelve (p12_path, password_one_time, fingerprint_sha256_hex).
    La contraseña se imprime UNA vez por el CLI de emparejamiento.
    """
    d = material_dir()
    ca_cert, ca_key = _load_ca()
    slug = (
        "".join(ch if ch.isalnum() else "-" for ch in name.lower()).strip("-")
        or "phone"
    )
    p12_path = d / f"client-{slug}.p12"
    key = ec.generate_private_key(ec.SECP256R1())
    now = _dt.datetime.now(_dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(_name(f"IPA Push — {name}"))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + _dt.timedelta(days=CLIENT_DAYS))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )
    password = secrets.token_urlsafe(18)
    p12_path.write_bytes(
        pkcs12.serialize_key_and_certificates(
            slug.encode(),
            key,
            cert,
            cas=[ca_cert],
            encryption_algorithm=serialization.BestAvailableEncryption(
                password.encode()
            ),
        )
    )
    return p12_path, password, _fingerprint(cert)


# ── Revocación ────────────────────────────────────────────────────────
def _revoked_path() -> Path:
    return material_dir() / "revoked.json"


def revoke_fingerprint(fingerprint: str) -> None:
    path = _revoked_path()
    data = []
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except ValueError:
            data = []
    if fingerprint not in data:
        data.append(fingerprint)
    path.write_text(json.dumps(data, indent=2))


def is_revoked(fingerprint: str) -> bool:
    path = _revoked_path()
    if not path.exists():
        return False
    try:
        return fingerprint in json.loads(path.read_text())
    except ValueError:
        return False
