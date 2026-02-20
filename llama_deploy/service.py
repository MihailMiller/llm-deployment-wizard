"""
Service configuration: models.ini, docker-compose.yml, and Docker lifecycle.

Token management has moved to tokens.py (TokenStore). The ensure_token_file
helper is no longer needed here.

Two compose layouts are generated depending on cfg.auth_mode:

  Plaintext mode:
    llama-server published at bind_host:port:8080
    --api-key-file /run/secrets/api_keys passed to llama-server

  Hashed mode:
    llama-server published at 127.0.0.1:8081:8080 (NGINX-internal port)
    --api-key-file removed from llama-server command
    llama-auth sidecar added (python:3.12-slim, published at 127.0.0.1:9000:9000)
    NGINX (on host) does auth_request to 127.0.0.1:9000 before proxying to 8081
"""

from __future__ import annotations

from pathlib import Path
from shlex import quote

from llama_deploy.config import AccessProfile, AuthMode, Config, DockerNetworkMode, ModelSpec
from llama_deploy.log import sh
from llama_deploy.system import write_file


# ---------------------------------------------------------------------------
# models.ini (llama-server router preset)
# ---------------------------------------------------------------------------

def write_models_ini(
    preset_path: Path,
    llm: ModelSpec,
    emb: ModelSpec,
    parallel: int,
) -> None:
    """
    Write the llama-server router preset INI file.

    Uses llm.effective_alias and emb.effective_alias as section headers so the
    model names advertised to clients match the actual repos — even when
    --llm-repo is overridden. Fixes Bug 2 (hardcoded "Qwen/Qwen3-8B" strings).
    """
    assert llm.resolved_filename, "LLM ModelSpec must be resolved before writing INI"
    assert emb.resolved_filename, "Embedding ModelSpec must be resolved before writing INI"

    content = f"""version = 1

[*]
parallel = {parallel}
jinja = false
c = {llm.ctx_len}

[{llm.effective_alias}]
model = /models/{llm.resolved_filename}
load-on-startup = true
c = {llm.ctx_len}

[{emb.effective_alias}]
model = /models/{emb.resolved_filename}
load-on-startup = true
embeddings = true
pooling = last
c = {emb.ctx_len}
"""
    write_file(preset_path, content, mode=0o644)


# ---------------------------------------------------------------------------
# Auth sidecar script
# ---------------------------------------------------------------------------

_SIDECAR_SCRIPT = '''\
#!/usr/bin/env python3
"""
llama_deploy auth sidecar.

Called by NGINX via auth_request. Reads active SHA-256 token hashes from
HASHES_FILE (reloaded on every request so revocation is instant without a
container restart). Returns HTTP 200 when the bearer token matches a stored
hash, 401 otherwise.

Never logs the Authorization header to avoid capturing token values.
"""
import hashlib
import http.server
import json
import os
from pathlib import Path

HASHES_FILE = Path(os.getenv("HASHES_FILE", "/run/secrets/token_hashes.json"))


def _load_hashes() -> set:
    if not HASHES_FILE.exists():
        return set()
    try:
        return set(json.loads(HASHES_FILE.read_text(encoding="utf-8")).get("hashes", []))
    except Exception:
        return set()


class _AuthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            if token_hash in _load_hashes():
                self.send_response(200)
                self.end_headers()
                return
        self.send_response(401)
        self.end_headers()

    def log_message(self, fmt, *args) -> None:
        pass  # suppress access logs — never log Authorization headers


if __name__ == "__main__":
    port = int(os.getenv("AUTH_PORT", "9000"))
    server = http.server.HTTPServer(("0.0.0.0", port), _AuthHandler)
    server.serve_forever()
'''


def write_auth_sidecar_script(base_dir: Path) -> None:
    """Write the auth sidecar Python script to <base_dir>/auth_sidecar.py."""
    script_path = base_dir / "auth_sidecar.py"
    write_file(script_path, _SIDECAR_SCRIPT, mode=0o755)


# ---------------------------------------------------------------------------
# docker-compose.yml
# ---------------------------------------------------------------------------

def write_compose(compose_path: Path, cfg: Config) -> None:
    if cfg.auth_mode == AuthMode.HASHED:
        _write_compose_hashed(compose_path, cfg)
    else:
        _write_compose_plaintext(compose_path, cfg)


def _effective_bind_host(cfg: Config) -> str:
    """
    Return the host address that should appear in the Docker port mapping.

    For profiles that should never expose the backend directly (HOME_PRIVATE,
    VPN_ONLY, LOCALHOST) we always pin to 127.0.0.1 even if bind_host is
    "0.0.0.0".  In those profiles the host-level firewall (UFW) or VPN layer
    controls external access; Docker must not punch through on its own.

    PUBLIC is the only profile that may legitimately bind to 0.0.0.0 (when no
    NGINX domain is configured and open_firewall is True).
    """
    from llama_deploy.log import log_line

    net = cfg.network
    if net.access_profile in (
        AccessProfile.HOME_PRIVATE,
        AccessProfile.VPN_ONLY,
        AccessProfile.LOCALHOST,
    ) and net.bind_host == "0.0.0.0":
        log_line(
            f"[COMPOSE] Profile={net.access_profile.value}: "
            f"overriding Docker bind from 0.0.0.0 -> 127.0.0.1 "
            f"(external access via UFW/VPN, not Docker port mapping)."
        )
        return "127.0.0.1"
    return net.bind_host


def _network_mode_block(cfg: Config) -> str:
    if cfg.docker_network_mode == DockerNetworkMode.COMPOSE:
        return ""
    return f"    network_mode: {cfg.docker_network_mode.value}\n"


def _llama_bind_host(cfg: Config) -> str:
    """
    Host passed to llama-server inside the container.

    In host mode, the process binds directly on the host namespace, so we must
    respect cfg.network.bind_host. In bridge/compose mode, container-local
    0.0.0.0 is correct and external exposure is controlled by port mappings.
    """
    if cfg.docker_network_mode == DockerNetworkMode.HOST:
        return cfg.network.bind_host
    return "0.0.0.0"


def _write_compose_plaintext(compose_path: Path, cfg: Config) -> None:
    """Plaintext mode: llama-server owns auth via --api-key-file."""
    net = cfg.network
    effective_host = (
        _effective_bind_host(cfg)
        if cfg.docker_network_mode != DockerNetworkMode.HOST
        else net.bind_host
    )
    network_mode_block = _network_mode_block(cfg)
    bind_host = _llama_bind_host(cfg)
    ports_block = (
        f'    ports:\n      - "{effective_host}:{net.port}:8080"\n'
        if net.publish and cfg.docker_network_mode != DockerNetworkMode.HOST
        else ""
    )
    content = f"""services:
  llama:
    image: {cfg.image}
    container_name: llama-router
    restart: unless-stopped
{network_mode_block}

{ports_block}    volumes:
      - {cfg.base_dir}/models:/models:ro
      - {cfg.base_dir}/presets:/presets:ro
      - {cfg.base_dir}/cache:/root/.cache
      - {cfg.base_dir}/secrets:/run/secrets:ro

    command: >
      --host {bind_host}
      --port 8080
      --api-key-file /run/secrets/api_keys
      --models-dir /models
      --models-preset /presets/models.ini
      --models-max {cfg.models_max}
      --no-webui
      --offline

    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    pids_limit: 512
    read_only: true
    tmpfs:
      - /tmp:rw,noexec,nosuid,size=256m
"""
    write_file(compose_path, content, mode=0o644)


def _write_compose_hashed(compose_path: Path, cfg: Config) -> None:
    """
    Hashed mode: llama-server has no --api-key-file; auth is delegated to
    the llama-auth sidecar via NGINX auth_request.

    llama-server is published on 127.0.0.1:8081 (internal; NGINX proxies to it).
    The auth sidecar is published on 127.0.0.1:9000 (NGINX auth_request target).
    """
    internal_port = cfg.llama_internal_port
    sidecar_port  = cfg.sidecar_port
    network_mode_block = _network_mode_block(cfg)
    bind_host = _llama_bind_host(cfg)
    llama_ports_block = (
        f'    ports:\n      - "127.0.0.1:{internal_port}:8080"\n\n'
        if cfg.docker_network_mode != DockerNetworkMode.HOST
        else ""
    )
    sidecar_ports_block = (
        f'    ports:\n      - "127.0.0.1:{sidecar_port}:9000"\n\n'
        if cfg.docker_network_mode != DockerNetworkMode.HOST
        else ""
    )

    content = f"""services:
  llama:
    image: {cfg.image}
    container_name: llama-router
    restart: unless-stopped
{network_mode_block}

{llama_ports_block}    volumes:
      - {cfg.base_dir}/models:/models:ro
      - {cfg.base_dir}/presets:/presets:ro
      - {cfg.base_dir}/cache:/root/.cache

    command: >
      --host {bind_host}
      --port 8080
      --models-dir /models
      --models-preset /presets/models.ini
      --models-max {cfg.models_max}
      --no-webui
      --offline

    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    pids_limit: 512
    read_only: true
    tmpfs:
      - /tmp:rw,noexec,nosuid,size=256m

  llama-auth:
    image: python:3.12-slim
    container_name: llama-auth
    restart: unless-stopped
{network_mode_block}

{sidecar_ports_block}    volumes:
      - {cfg.base_dir}/auth_sidecar.py:/auth_sidecar.py:ro
      - {cfg.base_dir}/secrets:/run/secrets:ro

    command: python /auth_sidecar.py

    environment:
      - HASHES_FILE=/run/secrets/token_hashes.json
      - AUTH_PORT=9000

    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    pids_limit: 64
    read_only: true
    tmpfs:
      - /tmp:rw,noexec,nosuid,size=32m
"""
    write_file(compose_path, content, mode=0o644)


# ---------------------------------------------------------------------------
# Docker lifecycle
# ---------------------------------------------------------------------------

def docker_pull(image: str) -> None:
    sh(f"docker pull {quote(image)}")


def docker_compose_up(compose_path: Path) -> None:
    sh(f"cd {quote(str(compose_path.parent))} && docker compose up -d")


def docker_compose_down(compose_path: Path) -> None:
    sh(f"cd {quote(str(compose_path.parent))} && docker compose down", check=False)
