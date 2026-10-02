"""IPA Push CLI (DEC-011): emparejamiento, revocación, servicio y pruebas.

Examples:
    notifications.py init-ca
    notifications.py pair --name "Pixel" [--replace]
    notifications.py revoke
    notifications.py serve
    notifications.py send --event research.done --title "prueba" --body "hola"
    notifications.py devices
    notifications.py status
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ipa.notifications import tls_material  # noqa: E402
from ipa.notifications.config import PushConfig  # noqa: E402
from ipa.notifications.service import PushService, resolve_bind_ip  # noqa: E402
from ipa.notifications.store import Store  # noqa: E402


def cmd_init_ca(_args) -> int:
    cert, cert_path, _ = tls_material.init_ca()
    print(f"CA lista: {cert_path}")
    print(f"  subject : {cert.subject.rfc4514_string()}")
    print(f"  huella  : {tls_material._fingerprint(cert)}")
    return 0


def cmd_pair(args) -> int:
    store = Store()
    slot = store.slot()
    if slot is not None and not args.replace:
        print(
            f"ERROR: la plaza ya está ocupada por '{slot.get('name') or 'un dispositivo'}'"
            f" (fingerprint {slot.get('cert_fingerprint', '')[:16]}...).\n"
            "Para desalojarla explícitamente: notifications.py pair --replace\n"
            "Para liberarla sin reemplazo: notifications.py revoke",
            file=sys.stderr,
        )
        return 1
    tls_material.init_ca()
    bind_ip = resolve_bind_ip(PushConfig.load().bind_ip)
    tls_material.ensure_server_cert(bind_ip)
    p12_path, password, fingerprint = tls_material.issue_client(args.name)
    if slot is not None:
        tls_material.revoke_fingerprint(slot["cert_fingerprint"])
    store.bind_slot(fingerprint, args.name)
    print("Emparejamiento completado.")
    print(f"  dispositivo : {args.name}")
    print(f"  servidor    : https://{bind_ip}:{PushConfig.load().port}")
    print(f"  huella CA   : {tls_material.ca_fingerprint()}")
    print(f"  .p12        : {p12_path}")
    print(f"  contraseña  : {password}   (una sola vez — copiala ahora)")
    print(
        "\nPasos en el teléfono:\n"
        "  1. Copia el .p12 al teléfono (USB) e instálalo:\n"
        "     Ajustes > Seguridad > Instalar certificado > VPN y apps\n"
        "  2. En la app IPA Push: URL del servidor + huella CA de arriba.\n"
        "  3. Concede el acceso al certificado cuando la app lo pida (una vez)."
    )
    return 0


def cmd_revoke(_args) -> int:
    store = Store()
    fingerprint = store.revoke_slot()
    if fingerprint is None:
        print("No hay dispositivo emparejado.")
        return 0
    tls_material.revoke_fingerprint(fingerprint)
    print(f"Plaza liberada y certificado revocado: {fingerprint}")
    return 0


def cmd_serve(args) -> int:
    config = PushConfig.load()
    service = PushService(config=config)
    bind_ip, port = service.start()
    print(f"[push] sirviendo en https://{bind_ip}:{port} (Ctrl+C para parar)")
    try:
        while True:
            import time

            time.sleep(3600)
    except KeyboardInterrupt:
        service.stop()
        print("[push] detenido")
    return 0


def cmd_send(args) -> int:
    store = Store()
    payload = json.loads(args.payload) if args.payload else {}
    nid = store.enqueue(
        args.event,
        args.title,
        args.body,
        payload,
        collapse_key=args.collapse_key,
    )
    print(f"Encolada notification id={nid}")
    return 0


def cmd_devices(_args) -> int:
    store = Store()
    slot = store.slot()
    if slot is None:
        print("Sin dispositivo emparejado (notifications.py pair --name ...)")
    else:
        print(json.dumps(slot, ensure_ascii=False, indent=2))
    return 0


def cmd_status(_args) -> int:
    store = Store()
    print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-ca", help="Crea la CA local si no existe").set_defaults(
        fn=cmd_init_ca
    )
    pair = sub.add_parser("pair", help="Empareja EL dispositivo (slot único)")
    pair.add_argument("--name", required=True)
    pair.add_argument(
        "--replace", action="store_true", help="Desaloja el slot ocupado"
    )
    pair.set_defaults(fn=cmd_pair)
    sub.add_parser("revoke", help="Libera la plaza y revoca el cert").set_defaults(
        fn=cmd_revoke
    )
    sub.add_parser("serve", help="Servidor push en primer plano").set_defaults(
        fn=cmd_serve
    )
    send = sub.add_parser("send", help="Encola un aviso de prueba")
    send.add_argument("--event", required=True)
    send.add_argument("--title", required=True)
    send.add_argument("--body", default="")
    send.add_argument("--payload", default="")
    send.add_argument("--collapse-key", default=None)
    send.set_defaults(fn=cmd_send)
    sub.add_parser("devices", help="Muestra el dispositivo del slot").set_defaults(
        fn=cmd_devices
    )
    sub.add_parser("status", help="Estadísticas de la cola").set_defaults(
        fn=cmd_status
    )

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
