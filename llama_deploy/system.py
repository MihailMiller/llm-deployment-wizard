"""
OS-level idempotent operations: package installation, swap, firewall, Docker.

All functions are designed to be safe to call multiple times (idempotent).
"""

from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import subprocess
from pathlib import Path
from shlex import quote
from typing import List, Optional

from llama_deploy.log import die, log_line, sh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def backup_file(path: Path) -> None:
    if not path.exists():
        return
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bak = path.with_suffix(path.suffix + f".bak.{ts}")
    shutil.copy2(path, bak)
    from tqdm import tqdm
    tqdm.write(f"[BACKUP] {path} -> {bak}")
    log_line(f"[BACKUP] {path} -> {bak}")


def write_file(path: Path, content: str, *, mode: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup_file(path)
    path.write_text(content, encoding="utf-8")
    if mode is not None:
        os.chmod(path, mode)
    from tqdm import tqdm
    tqdm.write(f"[WRITE] {path}")
    log_line(f"[WRITE] {path}")


# ---------------------------------------------------------------------------
# System detection
# ---------------------------------------------------------------------------

_SUDO_PASSTHROUGH = {
    # App-specific secrets
    "HF_TOKEN", "TAILSCALE_AUTHKEY",
    # Shell essentials (sudo resets PATH by default)
    "HOME", "PATH", "PYTHONPATH",
    # Terminal UX (tqdm needs TERM/COLUMNS)
    "TERM", "COLORTERM", "COLUMNS", "LINES",
    # Locale
    "LANG", "LC_ALL", "LC_CTYPE",
}


def require_root_reexec() -> None:
    if os.geteuid() == 0:
        return
    if shutil.which("sudo") is None:
        die("Must run as root (sudo not found).")
    import sys
    print("[INFO] Re-executing via sudo...", flush=True)
    env_args = [f"{k}={os.environ[k]}" for k in _SUDO_PASSTHROUGH if k in os.environ]
    os.execvp("sudo", ["sudo", "env", *env_args, sys.executable, *sys.argv])


def detect_ubuntu() -> None:
    osr = Path("/etc/os-release")
    if not osr.exists():
        die("/etc/os-release not found.")
    txt = osr.read_text(encoding="utf-8", errors="ignore").lower()
    if "ubuntu" not in txt:
        die("This script is for Ubuntu hosts.")
    if not Path("/run/systemd/system").exists():
        die("systemd not detected; this script assumes systemd.")


def detect_ssh_ports() -> List[int]:
    ports: List[int] = []
    if shutil.which("sshd"):
        try:
            out = subprocess.check_output(
                ["bash", "-lc", "sshd -T 2>/dev/null | awk '$1==\"port\"{print $2}'"],
                text=True,
            )
            for ln in out.splitlines():
                ln = ln.strip()
                if ln.isdigit():
                    ports.append(int(ln))
        except Exception:
            pass
    if not ports:
        try:
            out = subprocess.check_output(
                ["bash", "-lc", "ss -lntp | awk '/sshd/ {print $4}'"],
                text=True,
            )
            for ln in out.splitlines():
                m = re.search(r":(\d+)\s*$", ln.strip())
                if m:
                    ports.append(int(m.group(1)))
        except Exception:
            pass
    return sorted(set(ports)) or [22]


# ---------------------------------------------------------------------------
# Package installation
# ---------------------------------------------------------------------------

def ensure_base_packages() -> None:
    sh("export DEBIAN_FRONTEND=noninteractive; apt-get update -y")
    sh("export DEBIAN_FRONTEND=noninteractive; apt-get install -y ca-certificates curl gnupg ufw jq python3")


def ensure_unattended_upgrades() -> None:
    sh("export DEBIAN_FRONTEND=noninteractive; apt-get install -y unattended-upgrades")
    sh("dpkg-reconfigure -f noninteractive unattended-upgrades", check=False)


# ---------------------------------------------------------------------------
# Swap
# ---------------------------------------------------------------------------

def ensure_swap(gib: int) -> None:
    swaps = Path("/proc/swaps").read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(swaps) > 1:
        from tqdm import tqdm
        tqdm.write("[OK] Swap already active; skipping swap creation.")
        return
    sh(f"fallocate -l {gib}G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count={gib * 1024}")
    sh("chmod 600 /swapfile")
    sh("mkswap /swapfile")
    sh("swapon /swapfile")
    fstab = Path("/etc/fstab").read_text(encoding="utf-8", errors="ignore")
    if "/swapfile none swap sw 0 0" not in fstab:
        sh("echo '/swapfile none swap sw 0 0' >> /etc/fstab")
    sh("free -h", check=False)
    write_file(Path("/etc/sysctl.d/99-swappiness.conf"), "vm.swappiness=10\n", mode=0o644)
    sh("sysctl -p /etc/sysctl.d/99-swappiness.conf", check=False)


# ---------------------------------------------------------------------------
# Firewall
# ---------------------------------------------------------------------------

def ensure_firewall(network) -> None:
    """
    Apply UFW rules based on a NetworkConfig (profile-aware).

    Receives a NetworkConfig rather than scattered keyword arguments so the
    caller does not need to decompose flags.  The skip check
    (configure_ufw=False) is handled by Step.skip_if in orchestrator.py.

    Profile â†’ UFW strategy
    ----------------------
    LOCALHOST    : port not opened; SSH kept open; everything else denied.
    HOME_PRIVATE : port opened only from network.lan_cidr; deny from all else.
    VPN_ONLY     : port not opened in UFW (VPN layer handles access).
    PUBLIC       : if open_firewall=True, open port to all; otherwise same as
                   localhost (NGINX sits on :80/:443 which are separate).
    """
    from llama_deploy.config import AccessProfile
    from tqdm import tqdm

    profile = network.access_profile

    # Always keep SSH reachable first
    for p in detect_ssh_ports():
        sh(f"ufw allow {p}/tcp", check=False)

    if profile == AccessProfile.HOME_PRIVATE:
        if not network.lan_cidr:
            # Guarded by NetworkConfig validation but be explicit
            log_line("[UFW] home-private: lan_cidr missing â€” skipping port rule.")
        else:
            # Deny from everywhere, then allow from LAN CIDR only
            sh(f"ufw deny {network.port}/tcp", check=False)
            sh(f"ufw allow from {quote(network.lan_cidr)} to any port {network.port} proto tcp", check=False)
            tqdm.write(f"[UFW] home-private: port {network.port}/tcp allowed from {network.lan_cidr} only.")
            log_line(f"[UFW] home-private: port {network.port}/tcp allowed from {network.lan_cidr} only.")

    elif profile == AccessProfile.VPN_ONLY:
        # Do not open the port in UFW; VPN routing provides access.
        tqdm.write(f"[UFW] vpn-only: port {network.port}/tcp NOT opened in UFW; VPN handles routing.")
        log_line(f"[UFW] vpn-only: port {network.port}/tcp not exposed via UFW.")

    elif profile == AccessProfile.PUBLIC and network.open_firewall:
        sh(f"ufw allow {network.port}/tcp", check=False)
        tqdm.write(f"[UFW] public: port {network.port}/tcp opened to all.")
        log_line(f"[UFW] public: port {network.port}/tcp opened to all.")

    else:
        # LOCALHOST or PUBLIC without open_firewall â€” port stays closed in UFW
        tqdm.write(f"[UFW] {profile.value}: port {network.port}/tcp not opened (loopback or NGINX only).")
        log_line(f"[UFW] {profile.value}: port {network.port}/tcp not opened in UFW.")

    sh("ufw default deny incoming", check=False)
    sh("ufw default allow outgoing", check=False)
    sh("ufw --force enable", check=False)
    sh("ufw status verbose", check=False)


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

def ensure_docker_daemon_hardening() -> None:
    """
    Write /etc/docker/daemon.json with iptables=false.

    Docker's default iptables=true inserts ACCEPT rules into the FORWARD chain
    that bypass UFW entirely for containers published on 0.0.0.0. Setting
    iptables=false prevents Docker from touching iptables at all.

    Our compose files bind service ports to 127.0.0.1 and use a regular bridge
    network for container-to-container traffic, so there is no functional impact
    on the deployed service - this only removes Docker's iptables auto-management.

    Docker is restarted only if daemon.json actually changes.
    """
    import json
    from tqdm import tqdm

    daemon_path = Path("/etc/docker/daemon.json")
    desired: dict = {"iptables": False}

    current: dict = {}
    if daemon_path.exists():
        try:
            current = json.loads(daemon_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    merged = {**current, **desired}
    if merged == current:
        tqdm.write("[DOCKER] daemon.json already hardened (iptables=false).")
        log_line("[DOCKER] daemon.json already contains iptables=false.")
        return

    backup_file(daemon_path)
    write_file(daemon_path, json.dumps(merged, indent=2) + "\n", mode=0o644)
    tqdm.write("[DOCKER] Written /etc/docker/daemon.json with iptables=false; restarting Docker.")
    log_line("[DOCKER] daemon.json updated with iptables=false; restarting Docker.")
    sh("systemctl restart docker")


def ensure_docker() -> None:
    sh("export DEBIAN_FRONTEND=noninteractive; apt-get update -y")
    sh("export DEBIAN_FRONTEND=noninteractive; apt-get install -y ca-certificates curl gnupg")
    sh("install -m 0755 -d /etc/apt/keyrings")
    sh("curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --batch --yes --no-tty --dearmor -o /etc/apt/keyrings/docker.gpg")
    sh(
        'echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] '
        'https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" '
        '> /etc/apt/sources.list.d/docker.list'
    )
    sh("export DEBIAN_FRONTEND=noninteractive; apt-get update -y")
    sh(
        "export DEBIAN_FRONTEND=noninteractive; apt-get install -y "
        "docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin"
    )
    sh("systemctl enable --now docker")
    sh("docker --version")
    sh("docker compose version")

