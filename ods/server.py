"""FastAPI app: static UI, class catalog, and a WebSocket detection stream.

WebSocket protocol (one connection per camera session):
  client -> text   {"classes": ["person", "car"], "conf": 0.4}   (re-send any time)
  client -> binary JPEG frame
  server -> text   {"detections": [...], "ms": 42}  or  {"error": "..."}
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .detector import Detector
from .hub import Hub, client_ip
from .registry import ROOT, Registry


def create_app(registry: Registry | None = None, detector: Detector | None = None) -> FastAPI:
    registry = registry or Registry.load()
    detector = detector or Detector(registry)
    app = FastAPI(title="ODS")
    app.state.public_url = None  # set by `ods --share`; what phones should open

    @app.get("/api/info")
    def info():
        return {"public_url": app.state.public_url}

    @app.get("/api/classes")
    def classes():
        return {"classes": registry.catalog(), "default_conf": registry.default_conf}

    @app.websocket("/ws/detect")
    async def ws_detect(ws: WebSocket):
        await ws.accept()
        selected: list[str] = []
        conf: float | None = None
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("text") is not None:
                    cfg = json.loads(msg["text"])
                    selected = list(cfg.get("classes", []))
                    conf = cfg.get("conf")
                    try:
                        registry.group(selected)  # validate early
                    except (KeyError, FileNotFoundError) as e:
                        selected = []
                        await ws.send_text(json.dumps({"error": str(e.args[0])}))
                    continue
                data = msg.get("bytes")
                if data is None or not selected:
                    await ws.send_text(json.dumps({"detections": [], "ms": 0}))
                    continue
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    await ws.send_text(json.dumps({"error": "bad frame"}))
                    continue
                t0 = time.perf_counter()
                dets = await asyncio.to_thread(detector.detect, frame, selected, conf)
                await ws.send_text(json.dumps(
                    {"detections": dets, "ms": round((time.perf_counter() - t0) * 1000)}))
        except WebSocketDisconnect:
            pass

    # Phone-as-camera pairing: see ods/hub.py for the protocol.
    hub = Hub(registry, detector)

    @app.get("/api/hosts")
    def hosts(request: Request, code: str | None = None):
        ip = client_ip(request.headers, request.client.host if request.client else None)
        return {"hosts": hub.list_for(ip, code)}

    @app.websocket("/ws/host")
    async def ws_host(ws: WebSocket, name: str = ""):
        await hub.run_host(ws, client_ip(ws.headers, ws.client.host if ws.client else None), name)

    @app.websocket("/ws/phone")
    async def ws_phone(ws: WebSocket):
        await hub.run_phone(ws, client_ip(ws.headers, ws.client.host if ws.client else None))

    static = ROOT / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.middleware("http")
    async def revalidate_ui(request: Request, call_next):
        # Make browsers check for a newer page/stylesheet each load instead of reusing a stale copy.
        response = await call_next(request)
        if request.url.path in ("/", "/phone") or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/phone")
    def phone():
        return FileResponse(static / "phone.html")

    return app


def _free_port(preferred: int) -> int:
    """`preferred` if it can be bound on loopback, otherwise any free port."""
    import socket

    for candidate in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", candidate))
                return s.getsockname()[1]
            except OSError:
                continue
    return preferred


def _lan_ip() -> str | None:
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; just picks the outbound interface
            return s.getsockname()[0]
    except OSError:
        return None


def _self_signed_cert(lan_ip: str | None) -> tuple[Path, Path]:
    """Create (or reuse) a self-signed cert covering localhost and this machine's LAN IP."""
    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    certs = ROOT / "certs"
    certs.mkdir(exist_ok=True)
    cert_path, key_path = certs / "cert.pem", certs / "key.pem"
    ip_file = certs / "ip.txt"
    if cert_path.exists() and key_path.exists() and ip_file.exists() \
            and ip_file.read_text().strip() == (lan_ip or ""):
        return cert_path, key_path

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ods-local")])
    sans: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
    ]
    if lan_ip:
        sans.append(x509.IPAddress(ipaddress.ip_address(lan_ip)))
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    ip_file.write_text(lan_ip or "")
    return cert_path, key_path


def main():
    import os
    import sys
    import threading

    import uvicorn

    port = int(os.environ.get("PORT", 8000))  # hosting platforms inject PORT
    share = "--share" in sys.argv
    registry = Registry.load()
    detector = Detector(registry)
    # Load models in the background so the server answers immediately but the first frame isn't slow.
    threading.Thread(target=detector.warmup, daemon=True, name="warmup").start()
    app = create_app(registry, detector)

    if share:
        # Public HTTPS URL via a Cloudflare quick tunnel (works behind inbound-blocking firewalls).
        from . import share as tunnel
        proc = None
        wanted, port = port, _free_port(port)  # only the tunnel needs this port, so any free one works
        if port != wanted:
            print(f"Port {wanted} is already in use by another program; using {port} instead "
                  "(you don't need this port - use the public link below).")
        try:
            proc, url = tunnel.open_tunnel(tunnel.tunnel_command(tunnel.find_cloudflared(), port))
            app.state.public_url = url
            tunnel.announce(url)
            # Only the tunnel needs to reach us, so don't listen on the LAN.
            uvicorn.run(app, host="127.0.0.1", port=port)
        except (RuntimeError, OSError) as e:
            sys.exit(f"--share failed: {e}\n"
                     "Install cloudflared manually (https://github.com/cloudflare/cloudflared/releases) "
                     "and make sure it is on your PATH, then retry.")
        finally:
            tunnel.close_tunnel(proc)
        return

    # 0.0.0.0 is a bind address, not browsable; print URLs a browser can open.
    lan_ip = _lan_ip()
    if "--https" in sys.argv:
        # Phones only expose the camera on secure origins, so serve HTTPS with a self-signed cert.
        cert, key = _self_signed_cert(lan_ip)
        print(f"\nOn this PC:   https://localhost:{port}")
        if lan_ip:
            print(f"On your phone (same Wi-Fi): https://{lan_ip}:{port}")
        print("Accept the certificate warning (Advanced -> Proceed) the first time.\n")
        uvicorn.run(app, host="0.0.0.0", port=port,
                    ssl_certfile=str(cert), ssl_keyfile=str(key))
    else:
        print(f"\nOpen http://localhost:{port} in your browser")
        print("Phone camera: run  uv run ods --share  to get a public HTTPS link\n")
        uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
