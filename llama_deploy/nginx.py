"""
NGINX + certbot TLS termination and/or hashed-token auth proxy.

Architectures
-------------
TLS only (--domain set, plaintext auth):
  Internet → NGINX (:443 HTTPS) → 127.0.0.1:port (llama-server)

Hashed auth, no domain (local only):
  Client → NGINX (:port, loopback) → auth_request → sidecar (:sidecar_port)
                                   → 127.0.0.1:upstream_port (llama-server)

Hashed auth + TLS (--domain set, --auth-mode hashed):
  Internet → NGINX (:443 HTTPS) → auth_request → sidecar (:sidecar_port)
                                → 127.0.0.1:upstream_port (llama-server)

Public entry points
-------------------
    ensure_tls_for_domain(domain, email, upstream_port, configure_ufw,
                          use_auth_sidecar, sidecar_port)
    ensure_local_proxy(bind_host, port, upstream_port, configure_ufw,
                       use_auth_sidecar, sidecar_port)

LLM-specific proxy settings
----------------------------
- proxy_buffering off   : critical for token streaming (SSE / chunked)
- proxy_read_timeout    : generous timeout for long inference runs
- client_max_body_size  : allow large prompt payloads
"""

from __future__ import annotations

from pathlib import Path
import socket
from shlex import quote

from llama_deploy.log import die, log_line, sh


_SITES_AVAILABLE = Path("/etc/nginx/sites-available")
_SITES_ENABLED   = Path("/etc/nginx/sites-enabled")


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def ensure_nginx_certbot() -> None:
    """Install NGINX and the certbot nginx plugin (idempotent)."""
    sh("export DEBIAN_FRONTEND=noninteractive; apt-get update -y")
    sh(
        "export DEBIAN_FRONTEND=noninteractive; apt-get install -y "
        "nginx python3-certbot-nginx"
    )
    sh("systemctl enable --now nginx")


def ensure_nginx() -> None:
    """Install NGINX only — no certbot (for local auth proxy without TLS)."""
    sh("export DEBIAN_FRONTEND=noninteractive; apt-get update -y")
    sh("export DEBIAN_FRONTEND=noninteractive; apt-get install -y nginx")
    sh("systemctl enable --now nginx")


# ---------------------------------------------------------------------------
# NGINX config
# ---------------------------------------------------------------------------

def _config_name(domain_or_host: str) -> str:
    """Filesystem-safe config name derived from a domain or bind address."""
    return domain_or_host.replace(".", "_").replace("-", "_").replace(":", "_")


def _is_bind_port_free(bind_host: str, port: int) -> bool:
    """
    Return True if bind_host:port can be bound right now.
    """
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.bind((bind_host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _pick_free_bind_port(bind_host: str, preferred: int, *, avoid: set[int] | None = None) -> int:
    """
    Pick a free TCP port for the given bind host, preferring the requested one.
    """
    avoid = avoid or set()
    if preferred not in avoid and _is_bind_port_free(bind_host, preferred):
        return preferred

    for candidate in range(preferred + 1, 65536):
        if candidate in avoid:
            continue
        if _is_bind_port_free(bind_host, candidate):
            return candidate

    for candidate in range(1024, preferred):
        if candidate in avoid:
            continue
        if _is_bind_port_free(bind_host, candidate):
            return candidate

    die(f"Could not find a free TCP port on {bind_host} for local NGINX proxy.")


def _webui_location_block(webui_port: int) -> str:
    """
    Return an NGINX location / block that proxies to Open WebUI.

    Includes WebSocket upgrade headers required for Open WebUI's streaming
    responses and real-time features.
    """
    return f"""
    # Compatibility: some Open WebUI flows/routes may land on /chat.
    # Route it to the WebUI root instead of returning a backend 404.
    location = /chat {{
        return 302 /;
    }}

    location = /chat/ {{
        return 302 /;
    }}

    # Open WebUI — served at domain root; protected by Open WebUI's own user auth
    location / {{
        proxy_pass         http://127.0.0.1:{webui_port};
        proxy_http_version 1.1;

        # WebSocket support (required for Open WebUI streaming)
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection "upgrade";

        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_buffering    off;
        proxy_read_timeout 300s;
        proxy_connect_timeout  10s;
        proxy_send_timeout 300s;
    }}
"""


def _auth_request_block() -> str:
    """Return the auth_request directive line."""
    return """\
        # Delegate auth to the llama-auth sidecar (hashed token mode)
        auth_request /auth;

"""


def _auth_location_block(sidecar_port: int) -> str:
    """Return the internal /auth location that proxies to the sidecar."""
    return f"""
    # Internal auth endpoint — forwards the Authorization header to the sidecar
    location = /auth {{
        internal;
        proxy_pass              http://127.0.0.1:{sidecar_port};
        proxy_pass_request_body off;
        proxy_set_header        Content-Length "";
        proxy_set_header        Authorization $http_authorization;
        proxy_set_header        X-Original-URI $request_uri;
    }}
"""


def write_nginx_proxy_config(
    domain: str,
    upstream_port: int,
    *,
    use_auth_sidecar: bool = False,
    sidecar_port: int = 9000,
    webui_port: int = 0,
) -> None:
    """
    Write an initial HTTP-only NGINX site config. certbot will later add the SSL block.

    Without webui_port: single location / proxies all traffic to llama-server (with auth).
    With webui_port: splits routing — /v1/ goes to llama-server (auth enforced),
                     / goes to Open WebUI (its own user auth).
    """
    from llama_deploy.system import backup_file, write_file

    name = _config_name(domain)
    config_path = _SITES_AVAILABLE / name

    auth_req = _auth_request_block() if use_auth_sidecar else ""
    auth_loc = _auth_location_block(sidecar_port) if use_auth_sidecar else ""

    if webui_port:
        # Split routing: API at /v1/ (auth-gated) + Open WebUI at / (own auth)
        api_location = f"""
    # LLM API — bearer token required
    location /v1/ {{
{auth_req}        proxy_pass         http://127.0.0.1:{upstream_port};
        proxy_http_version 1.1;

        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection        "";

        proxy_buffering            off;
        proxy_cache                off;
        proxy_request_buffering    off;

        proxy_read_timeout    300s;
        proxy_connect_timeout  10s;
        proxy_send_timeout    300s;

        client_max_body_size 32m;
    }}
"""
        webui_loc = _webui_location_block(webui_port)
        body = api_location + auth_loc + webui_loc
    else:
        # Single location: all traffic goes to llama-server (with optional auth)
        body = f"""
    # Proxy all requests to llama-server
    location / {{
{auth_req}        proxy_pass         http://127.0.0.1:{upstream_port};
        proxy_http_version 1.1;

        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection        "";

        proxy_buffering            off;
        proxy_cache                off;
        proxy_request_buffering    off;

        proxy_read_timeout    300s;
        proxy_connect_timeout  10s;
        proxy_send_timeout    300s;

        client_max_body_size 32m;
    }}
{auth_loc}"""

    content = f"""# llama-server proxy — managed by llama_deploy
# certbot will add an SSL server block below this one.
server {{
    listen 80;
    listen [::]:80;
    server_name {domain};

    # Let's Encrypt ACME challenge (used by certbot)
    location /.well-known/acme-challenge/ {{
        root /var/www/html;
    }}
{body}}}
"""
    write_file(config_path, content, mode=0o644)

    # Symlink into sites-enabled (idempotent)
    link = _SITES_ENABLED / name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(config_path)

    # Remove the default site if still there (it conflicts on port 80)
    default_link = _SITES_ENABLED / "default"
    if default_link.exists() or default_link.is_symlink():
        default_link.unlink()
        log_line("[NGINX] Removed default site symlink.")

    sh("nginx -t")
    sh("systemctl reload nginx")


def write_nginx_local_config(
    bind_host: str,
    port: int,
    upstream_port: int,
    *,
    use_auth_sidecar: bool = False,
    sidecar_port: int = 9000,
    webui_port: int = 0,
) -> None:
    """
    Write a local (non-TLS) NGINX site config for hashed auth without a domain.

    Without webui_port: all traffic proxied to llama-server (with auth).
    With webui_port: /v1/ → llama-server (auth), / → Open WebUI (own auth).
    """
    from llama_deploy.system import write_file

    name = f"llama_local_{port}"
    config_path = _SITES_AVAILABLE / name

    auth_req = _auth_request_block() if use_auth_sidecar else ""
    auth_loc = _auth_location_block(sidecar_port) if use_auth_sidecar else ""

    if webui_port:
        api_location = f"""
    location /v1/ {{
{auth_req}        proxy_pass         http://127.0.0.1:{upstream_port};
        proxy_http_version 1.1;

        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header Connection        "";

        proxy_buffering            off;
        proxy_cache                off;
        proxy_request_buffering    off;

        proxy_read_timeout    300s;
        proxy_connect_timeout  10s;
        proxy_send_timeout    300s;

        client_max_body_size 32m;
    }}
"""
        body = api_location + auth_loc + _webui_location_block(webui_port)
    else:
        body = f"""
    location / {{
{auth_req}        proxy_pass         http://127.0.0.1:{upstream_port};
        proxy_http_version 1.1;

        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header Connection        "";

        proxy_buffering            off;
        proxy_cache                off;
        proxy_request_buffering    off;

        proxy_read_timeout    300s;
        proxy_connect_timeout  10s;
        proxy_send_timeout    300s;

        client_max_body_size 32m;
    }}
{auth_loc}"""

    content = f"""# llama-server local proxy — managed by llama_deploy
server {{
    listen {bind_host}:{port};
    server_name _;
{body}}}
"""
    write_file(config_path, content, mode=0o644)

    link = _SITES_ENABLED / name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(config_path)

    default_link = _SITES_ENABLED / "default"
    if default_link.exists() or default_link.is_symlink():
        default_link.unlink()
        log_line("[NGINX] Removed default site symlink.")

    sh("nginx -t")
    sh("systemctl reload nginx")


# ---------------------------------------------------------------------------
# Certificate
# ---------------------------------------------------------------------------

def obtain_certificate(domain: str, email: str) -> None:
    """
    Run certbot --nginx to obtain (or renew) a Let's Encrypt certificate.

    certbot modifies the NGINX config in-place to add the SSL server block
    and HTTP→HTTPS redirect. Subsequent calls are idempotent — certbot
    will renew the cert if it is near expiry, or skip if it is still valid.

    Prerequisites: the domain's DNS A record must already point to this
    server's public IP, and port 80 must be reachable from the internet
    (for the ACME HTTP-01 challenge).
    """
    from tqdm import tqdm
    tqdm.write(f"[CERTBOT] Requesting certificate for {domain} (email: {email})")
    tqdm.write("[CERTBOT] Ensure DNS A record for this domain points to this server's IP.")
    sh(
        f"certbot --nginx "
        f"--non-interactive "
        f"--agree-tos "
        f"--redirect "
        f"-m {quote(email)} "
        f"-d {quote(domain)}"
    )
    # Verify the auto-renewal timer is active
    sh("systemctl is-active certbot.timer || systemctl enable --now certbot.timer", check=False)


# ---------------------------------------------------------------------------
# Firewall
# ---------------------------------------------------------------------------

def open_nginx_firewall_ports() -> None:
    """Open ports 80 (HTTP/ACME) and 443 (HTTPS) in UFW."""
    sh("ufw allow 80/tcp",  check=False)
    sh("ufw allow 443/tcp", check=False)
    sh("ufw status verbose", check=False)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def ensure_tls_for_domain(
    domain: str,
    email: str,
    upstream_port: int,
    configure_ufw: bool,
    *,
    use_auth_sidecar: bool = False,
    sidecar_port: int = 9000,
    webui_port: int = 0,
) -> None:
    """
    Full TLS setup pipeline. Safe to call on re-deployment.

    Steps:
    1. Install nginx + python3-certbot-nginx
    2. Write HTTP proxy config (with optional auth_request) + reload nginx
    3. Optionally open UFW ports 80 + 443
    4. Run certbot --nginx (adds SSL block + HTTP→HTTPS redirect)
    """
    from tqdm import tqdm
    from llama_deploy.config import is_valid_domain, normalize_domain

    domain = normalize_domain(domain) or ""
    if not is_valid_domain(domain):
        from llama_deploy.log import die
        die("Invalid domain for TLS setup. Use a bare hostname like api.example.com.")

    tqdm.write(f"[TLS] Setting up NGINX + Let's Encrypt for {domain}")
    log_line(f"[TLS] domain={domain} upstream_port={upstream_port} auth_sidecar={use_auth_sidecar}")

    ensure_nginx_certbot()
    write_nginx_proxy_config(
        domain, upstream_port,
        use_auth_sidecar=use_auth_sidecar,
        sidecar_port=sidecar_port,
        webui_port=webui_port,
    )

    if configure_ufw:
        open_nginx_firewall_ports()

    obtain_certificate(domain, email)

    tqdm.write(f"[TLS] HTTPS endpoint ready: https://{domain}/v1")
    log_line(f"[TLS] https://{domain}/v1 active")


def ensure_local_proxy(
    bind_host: str,
    port: int,
    upstream_port: int,
    configure_ufw: bool,
    *,
    use_auth_sidecar: bool = False,
    sidecar_port: int = 9000,
    webui_port: int = 0,
) -> int:
    """
    Set up NGINX as a local (non-TLS) reverse proxy for hashed auth mode
    when no domain is configured.

    NGINX listens on bind_host:port and proxies to llama-server on upstream_port.
    The auth_request directive delegates token verification to the sidecar.

    Returns the actual proxy port that was configured.
    """
    from tqdm import tqdm

    requested_port = port
    site_name = f"llama_local_{requested_port}"
    has_existing_site = (
        (_SITES_ENABLED / site_name).exists()
        or (_SITES_ENABLED / site_name).is_symlink()
    )
    selected_port = requested_port if has_existing_site else _pick_free_bind_port(bind_host, requested_port)

    if selected_port != requested_port:
        tqdm.write(
            f"[NGINX] Requested local proxy port {requested_port} is busy; "
            f"using {selected_port}."
        )
        log_line(
            f"[NGINX] local proxy port adjusted: "
            f"{bind_host}:{requested_port} -> {bind_host}:{selected_port}"
        )

    tqdm.write(f"[NGINX] Setting up local auth proxy on {bind_host}:{selected_port}")
    log_line(f"[NGINX] local proxy bind={bind_host}:{selected_port} upstream=127.0.0.1:{upstream_port}")

    ensure_nginx()
    write_nginx_local_config(
        bind_host, selected_port, upstream_port,
        use_auth_sidecar=use_auth_sidecar,
        sidecar_port=sidecar_port,
        webui_port=webui_port,
    )

    if configure_ufw and bind_host == "0.0.0.0":
        sh(f"ufw allow {selected_port}/tcp", check=False)
        sh("ufw status verbose", check=False)

    tqdm.write(f"[NGINX] Local proxy ready: http://{bind_host}:{selected_port}/v1")
    log_line(f"[NGINX] http://{bind_host}:{selected_port}/v1 active")
    return selected_port
