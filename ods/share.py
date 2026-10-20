"""`ods --share`: publish the local server through a free Cloudflare quick tunnel.

The tunnel client only makes an outbound connection, so it works behind firewalls that block inbound
traffic, and Cloudflare terminates HTTPS (which phones require for camera access). The URL is random,
changes on every run, and works only while the process is running.
"""
from __future__ import annotations

import platform
import re
import shutil
import stat
import subprocess
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

from .registry import ROOT

RELEASE = "https://github.com/cloudflare/cloudflared/releases/latest/download/"
URL_RE = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com", re.I)
TOOLS = ROOT / "tools"


def _asset() -> tuple[str, str]:
    """(release asset name, file name on disk) for this platform."""
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "amd64"
    system = platform.system()
    if system == "Windows":
        return f"cloudflared-windows-{arch}.exe", "cloudflared.exe"
    if system == "Darwin":
        return f"cloudflared-darwin-{arch}.tgz", "cloudflared"
    return f"cloudflared-linux-{arch}", "cloudflared"


def find_cloudflared() -> str:
    """Path to cloudflared: on PATH, already downloaded to tools/, or fetched from the official release."""
    on_path = shutil.which("cloudflared")
    if on_path:
        return on_path
    asset, name = _asset()
    exe = TOOLS / name
    if exe.exists():
        return str(exe)
    print(f"cloudflared not found; downloading {asset} from github.com/cloudflare/cloudflared ...")
    TOOLS.mkdir(exist_ok=True)
    download = TOOLS / asset
    urllib.request.urlretrieve(RELEASE + asset, download)
    if asset.endswith(".tgz"):
        with tarfile.open(download) as tar:
            member = tar.getmember("cloudflared")
            member.name = name  # never trust paths from an archive
            tar.extract(member, TOOLS)
        download.unlink()
    elif download != exe:
        download.replace(exe)
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return str(exe)


def tunnel_command(exe: str, port: int) -> list[str]:
    return [exe, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"]


def open_tunnel(cmd: list[str], timeout: float = 60.0) -> tuple[subprocess.Popen, str]:
    """Start the tunnel process and block until it prints its public URL."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    found: list[str] = []
    tail: list[str] = []

    def pump() -> None:  # keep draining so the child never blocks on a full pipe
        for line in proc.stdout:  # type: ignore[union-attr]
            tail.append(line.rstrip())
            del tail[:-15]
            if not found and (m := URL_RE.search(line)):
                found.append(m.group(0))

    threading.Thread(target=pump, daemon=True, name="tunnel-log").start()
    deadline = time.monotonic() + timeout
    while not found and proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if not found:
        proc.terminate()
        raise RuntimeError("could not open a tunnel; cloudflared said:\n  " + "\n  ".join(tail[-8:]))
    return proc, found[0]


def close_tunnel(proc: subprocess.Popen | None) -> None:
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()


def announce(url: str) -> None:
    print(f"""
  Public URL (HTTPS):  {url}

  1. Open it on this computer and tick "Accept phone cameras".
  2. Open the same URL on your phone: it lists this computer. Tap it, then accept here.
  Keep this window open; press Ctrl+C to stop sharing. The link changes every run.
  Anyone with the link can use the detector; each computer must still accept a phone.
""")

