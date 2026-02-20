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

def require_root_reexec() -> None:
    if os.geteuid() == 0:
        return
    if shutil.which("sudo") is None:
        die("Must run as root (sudo not found).")
    import sys
    print("[INFO] Re-executing via sudo...", flush=True)
    os.execvp("sudo", ["sudo", "-E", sys.executable, *sys.argv])


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
    Apply UFW rules based on a NetworkConfig.

    Receives a NetworkConfig rather than scattered keyword arguments so the
    caller does not need to decompose flags. The skip check (configure_ufw=False)
    is handled by the Step.skip_if in orchestrator.py, not here.
    """
    for p in detect_ssh_ports():
        sh(f"ufw allow {p}/tcp", check=False)

    if network.open_firewall:
        sh(f"ufw allow {network.port}/tcp", check=False)

    sh("ufw default deny incoming", check=False)
    sh("ufw default allow outgoing", check=False)
    sh("ufw --force enable", check=False)
    sh("ufw status verbose", check=False)


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

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
