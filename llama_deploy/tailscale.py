"""
Tailscale integration: install, bring up, health-check, and IP retrieval.

Public API
----------
  tailscale_install()        — idempotent: installs tailscale if not present
  tailscale_up(auth_key)     — runs `tailscale up`; auth_key is optional
  tailscale_ip() -> str      — returns the Tailscale IPv4 address (100.x.x.x)
  tailscale_health() -> bool — True when tailscale is running and has an IP

All output goes through sh() / tqdm.write() so it integrates seamlessly with
the existing step/progress infrastructure.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Optional

from llama_deploy.log import die, log_line, sh


_TS_IP_RE = re.compile(r"\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

def tailscale_install() -> None:
    """
    Idempotent Tailscale installation via the official install script.

    Uses the curl-pipe-bash method documented at https://tailscale.com/kb/
    install-ubuntu. Skips if `tailscale` is already on PATH.
    """
    from tqdm import tqdm

    if shutil.which("tailscale"):
        tqdm.write("[TS] tailscale already installed — skipping.")
        log_line("[TS] tailscale already installed.")
        return

    tqdm.write("[TS] Installing Tailscale...")
    log_line("[TS] Installing Tailscale.")
    sh("curl -fsSL https://tailscale.com/install.sh | sh")
    sh("systemctl enable --now tailscaled")


# ---------------------------------------------------------------------------
# Bring up
# ---------------------------------------------------------------------------

def tailscale_up(auth_key: Optional[str] = None) -> None:
    """
    Bring Tailscale up.  If auth_key is provided it is passed as
    --authkey so the node authenticates non-interactively.

    Calls `tailscale up --accept-routes` so advertised subnet routes
    are accepted automatically.
    """
    from tqdm import tqdm

    if not shutil.which("tailscale"):
        die("tailscale is not installed; run tailscale_install() first.")

    status = _ts_status_json()
    if status and status.get("BackendState") == "Running":
        ip = tailscale_ip()
        tqdm.write(f"[TS] Tailscale already running (IP: {ip}); skipping tailscale up.")
        log_line(f"[TS] Already running: {ip}")
        return

    cmd = "tailscale up --accept-routes"
    if auth_key:
        # Auth key passed via TS_AUTHKEY env var — never appears in /proc/<pid>/cmdline
        log_line(f"\n$ {cmd}  # TS_AUTHKEY=<REDACTED>")
        proc = subprocess.run(
            ["bash", "-lc", cmd],
            capture_output=True,
            text=True,
            env={**os.environ, "TS_AUTHKEY": auth_key},
        )
        if proc.returncode != 0:
            log_line(f"[TS] tailscale up failed: {proc.stderr.strip()}")
            die(f"tailscale up failed (exit {proc.returncode}): {proc.stderr.strip()}")
        tqdm.write("[TS] tailscale up completed.")
        log_line("[TS] tailscale up completed.")
    else:
        sh(cmd)


# ---------------------------------------------------------------------------
# IP and health
# ---------------------------------------------------------------------------

def tailscale_ip() -> str:
    """
    Return the Tailscale IPv4 address (100.x.x.x).

    Tries `tailscale ip -4` first, then falls back to parsing
    `tailscale status --json`.  Raises SystemExit via die() if no IP found.
    """
    # Fast path: tailscale ip -4
    try:
        out = subprocess.check_output(
            ["tailscale", "ip", "-4"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        m = _TS_IP_RE.search(out)
        if m:
            return m.group(0)
    except Exception:
        pass

    # Fallback: parse status JSON
    status = _ts_status_json()
    if status:
        self_node = status.get("Self") or {}
        for addr in self_node.get("TailscaleIPs") or []:
            m = _TS_IP_RE.search(str(addr))
            if m:
                return m.group(0)

    die(
        "Could not determine Tailscale IP address.\n"
        "  Run: tailscale status\n"
        "  Then: tailscale up --accept-routes"
    )
    return ""  # unreachable — keeps type-checkers happy


def tailscale_health() -> bool:
    """
    Return True when Tailscale is installed, running, and has a 100.x IP.

    Does not call die() — safe to use as a condition in skip_if lambdas.
    """
    if not shutil.which("tailscale"):
        return False
    status = _ts_status_json()
    if not status:
        return False
    if status.get("BackendState") != "Running":
        return False
    try:
        tailscale_ip()
        return True
    except SystemExit:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_status_json() -> Optional[dict]:
    """Return parsed `tailscale status --json` or None on failure."""
    import json
    try:
        out = subprocess.check_output(
            ["tailscale", "status", "--json"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return json.loads(out)
    except Exception:
        return None
