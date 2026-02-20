"""
OS-level idempotent operations: package installation, swap, firewall, Docker.

All functions are designed to be safe to call multiple times (idempotent).
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import os
import re
import shutil
import subprocess
from pathlib import Path
from shlex import quote
from typing import List, Optional, Tuple

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

def _parse_ipv4_network(raw: str) -> Optional[ipaddress.IPv4Network]:
    raw = (raw or "").strip()
    if not raw or raw == "default":
        return None
    if "/" not in raw:
        raw = f"{raw}/32"
    try:
        net = ipaddress.ip_network(raw, strict=False)
    except ValueError:
        return None
    if isinstance(net, ipaddress.IPv4Network):
        return net
    return None


def _host_ipv4_routes(*, include_docker_interfaces: bool = True) -> List[ipaddress.IPv4Network]:
    """
    Return IPv4 route destinations from the host routing table.

    We only need destination prefixes (first token of each `ip route` line) to
    detect overlap with Docker bridge subnets.
    """
    routes: List[ipaddress.IPv4Network] = []
    try:
        out = subprocess.check_output(
            ["ip", "-o", "-4", "route", "show"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return routes

    for line in out.splitlines():
        parts = line.split()
        if not parts:
            continue
        dev = None
        if "dev" in parts:
            idx = parts.index("dev")
            if idx + 1 < len(parts):
                dev = parts[idx + 1]

        if (
            not include_docker_interfaces
            and dev
            and (dev == "docker0" or dev.startswith("br-") or dev.startswith("veth"))
        ):
            continue

        net = _parse_ipv4_network(parts[0])
        if net is not None:
            routes.append(net)
    return routes


def _docker_bridge_subnets() -> List[Tuple[str, ipaddress.IPv4Network]]:
    """
    Return (network_name, subnet) tuples for Docker bridge networks.
    """
    items: List[Tuple[str, ipaddress.IPv4Network]] = []
    try:
        names_out = subprocess.check_output(
            ["docker", "network", "ls", "--filter", "driver=bridge", "--format", "{{.Name}}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return items

    for name in [ln.strip() for ln in names_out.splitlines() if ln.strip()]:
        try:
            subnets_out = subprocess.check_output(
                [
                    "docker",
                    "network",
                    "inspect",
                    name,
                    "--format",
                    "{{range .IPAM.Config}}{{.Subnet}}{{\"\\n\"}}{{end}}",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            continue

        for raw in subnets_out.splitlines():
            net = _parse_ipv4_network(raw)
            if net is not None:
                items.append((name, net))
    return items


def _choose_default_address_pools(
    occupied: List[ipaddress.IPv4Network],
    pool_count: int = 2,
) -> List[dict]:
    """
    Choose non-overlapping Docker default-address-pools from preferred ranges.
    """
    chosen: List[dict] = []

    candidates: List[ipaddress.IPv4Network] = [
        *[ipaddress.ip_network(f"10.{octet}.0.0/16") for octet in range(200, 256)],
        ipaddress.ip_network("192.168.240.0/20"),
        ipaddress.ip_network("192.168.224.0/20"),
        ipaddress.ip_network("198.18.0.0/16"),
        ipaddress.ip_network("198.19.0.0/16"),
    ]

    for base in candidates:
        if any(base.overlaps(net) for net in occupied):
            continue
        chosen.append({"base": str(base), "size": 24})
        if len(chosen) >= pool_count:
            return chosen

    die(
        "Could not choose non-overlapping Docker default-address-pools. "
        "Please set /etc/docker/daemon.json default-address-pools manually."
    )


def _bridge_route_conflicts() -> List[Tuple[str, ipaddress.IPv4Network, ipaddress.IPv4Network]]:
    """
    Return bridge subnets that overlap non-default host routes.
    """
    conflicts: List[Tuple[str, ipaddress.IPv4Network, ipaddress.IPv4Network]] = []
    host_routes = _host_ipv4_routes(include_docker_interfaces=False)
    bridge_subnets = _docker_bridge_subnets()

    for bridge_name, bridge_net in bridge_subnets:
        for route_net in host_routes:
            if bridge_net == route_net:
                conflicts.append((bridge_name, bridge_net, route_net))
                continue
            if bridge_net.overlaps(route_net):
                conflicts.append((bridge_name, bridge_net, route_net))
    return conflicts


def _fail_on_bridge_route_conflicts() -> None:
    """
    Abort early when Docker bridge subnets shadow host routes.
    """
    conflicts = _bridge_route_conflicts()
    if not conflicts:
        return

    lines = [
        f"- {name}: {bridge} overlaps host route {route}"
        for name, bridge, route in conflicts[:8]
    ]
    if len(conflicts) > 8:
        lines.append(f"- ... and {len(conflicts) - 8} more")

    detail = "\n".join(lines)
    die(
        "Detected Docker bridge subnet overlap with host routes. "
        "This can blackhole return traffic from LAN/VPN clients.\n"
        f"{detail}\n"
        "Fix: remove conflicting Docker bridge networks and retry "
        "(e.g. `docker network prune -f`, or targeted `docker network rm <name>`)."
    )


def ensure_docker_daemon_hardening() -> None:
    """
    Write /etc/docker/daemon.json with hardened networking defaults.

    Docker's default iptables=true inserts ACCEPT rules into the FORWARD chain
    that bypass UFW entirely for containers published on 0.0.0.0. Setting
    iptables=false prevents Docker from touching iptables at all.

    We also ensure default-address-pools is present to avoid:
      "all predefined address pools have been fully subnetted"
    on hosts that create many Compose networks over time.

    Our compose files bind service ports to 127.0.0.1 and use a regular bridge
    network for container-to-container traffic, so there is no functional impact
    on the deployed service - this only removes Docker's iptables auto-management.

    Docker is restarted only if daemon.json actually changes.
    """
    import json
    from tqdm import tqdm

    daemon_path = Path("/etc/docker/daemon.json")
    current: dict = {}
    if daemon_path.exists():
        try:
            current = json.loads(daemon_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    occupied = _host_ipv4_routes(include_docker_interfaces=False) + [subnet for _, subnet in _docker_bridge_subnets()]
    default_pools = _choose_default_address_pools(occupied)

    merged = dict(current)
    merged["iptables"] = False
    if not merged.get("default-address-pools"):
        merged["default-address-pools"] = default_pools

    if merged == current:
        tqdm.write("[DOCKER] daemon.json already hardened (iptables=false, address pools configured).")
        log_line("[DOCKER] daemon.json already hardened (iptables=false, default-address-pools set).")
        _fail_on_bridge_route_conflicts()
        return

    backup_file(daemon_path)
    write_file(daemon_path, json.dumps(merged, indent=2) + "\n", mode=0o644)
    tqdm.write("[DOCKER] Updated /etc/docker/daemon.json (iptables=false + address pools); restarting Docker.")
    log_line("[DOCKER] daemon.json updated (iptables=false + default-address-pools); restarting Docker.")
    sh("systemctl restart docker")
    _fail_on_bridge_route_conflicts()


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

