import sys

import pytest

from ods import share


def _fake(script: str) -> list[str]:
    return [sys.executable, "-c", script]


def test_open_tunnel_returns_url_from_log_output():
    cmd = _fake(
        "import time\n"
        "print('2026-01-01T00:00:00Z INF Requesting new quick Tunnel on trycloudflare.com...', flush=True)\n"
        "print('|  https://quiet-river-1234.trycloudflare.com  |', flush=True)\n"
        "time.sleep(30)\n"
    )
    proc, url = share.open_tunnel(cmd, timeout=15)
    try:
        assert url == "https://quiet-river-1234.trycloudflare.com"
        assert proc.poll() is None  # still running, log is being drained
    finally:
        share.close_tunnel(proc)
    assert proc.poll() is not None


def test_open_tunnel_reports_why_it_failed():
    cmd = _fake("print('ERR failed to dial edge: connection refused', flush=True)")
    with pytest.raises(RuntimeError, match="connection refused"):
        share.open_tunnel(cmd, timeout=15)


def test_tunnel_command_targets_loopback_port():
    cmd = share.tunnel_command("cloudflared", 7860)
    assert cmd[:2] == ["cloudflared", "tunnel"]
    assert cmd[cmd.index("--url") + 1] == "http://127.0.0.1:7860"


def test_asset_names_cover_this_platform():
    asset, name = share._asset()
    assert asset.startswith("cloudflared-") and name.startswith("cloudflared")
