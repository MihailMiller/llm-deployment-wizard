"""
Command-line interface: subcommand dispatcher and batch-mode config builder.

Subcommands
-----------
  python -m llama_deploy                         # wizard if TTY, else --help
  python -m llama_deploy deploy                  # same
  python -m llama_deploy deploy --batch [flags]  # non-interactive (CI/scripts)
  python -m llama_deploy tokens list    [--base-dir DIR]
  python -m llama_deploy tokens create  --name NAME  [--base-dir DIR]
  python -m llama_deploy tokens revoke  <id>  [--base-dir DIR]
  python -m llama_deploy tokens show    <id>  [--base-dir DIR]

`build_config(argv)` is the batch-mode parser. It is also the source of truth
for all available flags and their defaults. The wizard hardcodes its own
defaults because it shows human-readable choices, not raw argparse strings.

Batch-mode parameter fixes vs original script
---------------------------------------------
  --public + --bind + --no-publish  →  --bind HOST  (explicit) + --no-publish
  --allow-public-port               →  --open-firewall  (validated)
  --no-ufw                          →  --skip-ufw
  llm_candidates hardcoded          →  --llm-candidates (comma list)
  emb_candidates hardcoded          →  --emb-candidates (comma list)
  (new)                             →  --skip-download
  (new)                             →  --token-name
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import List, Optional

from llama_deploy.config import (
    AccessProfile,
    AuthMode,
    BackendKind,
    Config,
    DockerNetworkMode,
    is_valid_domain,
    ModelSpec,
    NetworkConfig,
    normalize_domain,
)

_DEFAULT_LLM_CANDIDATES = "Q4_K_M,Q5_K_M,Q4_0,Q3_K_M"
_DEFAULT_EMB_CANDIDATES = "Q8_0,F16,Q6_K,Q4_K_M"


# ---------------------------------------------------------------------------
# Batch-mode config builder
# ---------------------------------------------------------------------------

def build_config(argv: Optional[List[str]] = None) -> Config:
    """
    Parse argv (defaults to sys.argv[1:]) and return an immutable Config.
    All cross-argument validation is done here via parser.error().
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m llama_deploy deploy --batch",
        description="Deploy llama.cpp (llama-server) as an OpenAI-compatible Docker service (non-interactive).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = parser.add_argument_group("Paths")
    g.add_argument("--base-dir", default="/opt/llama", metavar="DIR",
                   help="Base directory for models, config, cache, secrets. (default: /opt/llama)")

    g = parser.add_argument_group("Backend")
    g.add_argument("--backend", default="cpu", choices=[b.value for b in BackendKind],
                   help="llama.cpp Docker image variant. (default: cpu)")

    g = parser.add_argument_group("Network")
    g.add_argument("--profile",
                   default=None,
                   choices=[p.value for p in AccessProfile],
                   metavar="PROFILE",
                   help=(
                       "Access profile: one of {%(choices)s}. "
                       "Sets safe defaults for --bind and firewall rules. "
                       "  localhost    — loopback only (default). "
                       "  home-private — LAN access; requires --lan-cidr. "
                       "  vpn-only     — VPN interface access (Tailscale). "
                       "  public       — internet-facing (combine with --domain or --open-firewall)."
                   ))
    g.add_argument("--lan-cidr", default=None, metavar="CIDR",
                   help="LAN source CIDR for home-private profile (e.g. 192.168.1.0/24). "
                        "Required when --profile=home-private.")
    g.add_argument("--bind", default=None, metavar="HOST",
                   help="Override bind address. Inferred from --profile when omitted. "
                        "Use 0.0.0.0 for public. (default: 127.0.0.1)")
    g.add_argument("--port", type=int, default=8080,
                   help="Host port to publish. (default: 8080)")
    g.add_argument("--no-publish", action="store_true",
                   help="Do NOT publish any host port (Docker-network-only).")
    g.add_argument("--open-firewall", action="store_true",
                   help="Open --port in UFW. Requires --bind 0.0.0.0.")
    g.add_argument("--skip-ufw", action="store_true",
                   help="Do not configure UFW at all.")
    g.add_argument("--docker-network-mode",
                   default="bridge",
                   choices=[m.value for m in DockerNetworkMode],
                   metavar="MODE",
                   help="Container network mode: bridge (default), compose, or host.")

    g = parser.add_argument_group("System")
    g.add_argument("--swap-gib", type=int, default=8, metavar="GIB",
                   help="Swap file size if none exists. (default: 8)")
    g.add_argument("--no-auto-optimize", action="store_true",
                   help="Disable host-spec auto tuning (model choice, ctx, memory knobs).")
    g.add_argument("--tailscale-authkey", default=None, metavar="KEY",
                   help="Tailscale auth key for non-interactive 'tailscale up'. "
                        "Only used when --profile=vpn-only. "
                        "Falls back to TAILSCALE_AUTHKEY env var.")

    g = parser.add_argument_group("Authentication")
    g.add_argument("--token", default=None, metavar="TOKEN",
                   help="API token value. If omitted, a random token is generated.")
    g.add_argument("--token-name", default="default", metavar="NAME",
                   help="Label for the first API token. (default: default)")
    g.add_argument("--auth-mode", default="plaintext", choices=[m.value for m in AuthMode],
                   metavar="MODE",
                   help="Token storage strategy: 'plaintext' (llama-server --api-key-file) "
                        "or 'hashed' (SHA-256 stored; NGINX auth_request sidecar). "
                        "(default: plaintext)")
    g.add_argument("--hf-token", default=None, metavar="TOKEN",
                   help="HuggingFace access token. Falls back to HF_TOKEN env var.")

    g = parser.add_argument_group("Models")
    g.add_argument("--llm-repo", default="Qwen/Qwen3-8B-GGUF", metavar="REPO",
                   help="HuggingFace repo for the LLM GGUF. (default: Qwen/Qwen3-8B-GGUF)")
    g.add_argument("--llm-candidates", default=_DEFAULT_LLM_CANDIDATES, metavar="PATTERNS",
                   help=f"Comma-separated GGUF filename patterns for LLM. (default: {_DEFAULT_LLM_CANDIDATES})")
    g.add_argument("--emb-repo", default="Qwen/Qwen3-Embedding-0.6B-GGUF", metavar="REPO",
                   help="HuggingFace repo for the embedding GGUF. (default: Qwen/Qwen3-Embedding-0.6B-GGUF)")
    g.add_argument("--emb-candidates", default=_DEFAULT_EMB_CANDIDATES, metavar="PATTERNS",
                   help=f"Comma-separated GGUF filename patterns for embeddings. (default: {_DEFAULT_EMB_CANDIDATES})")
    g.add_argument("--skip-download", action="store_true",
                   help="Skip HF queries and downloads if model files are already on disk.")
    g.add_argument("--allow-unverified-downloads", action="store_true",
                   help="Allow continuing when SHA256 cannot be verified; non-interactive trust mode.")
    g.add_argument("--skip-health-check", action="store_true",
                   help="Skip the /health wait and smoke tests after container start. "
                        "Useful when the model is slow to load (>5 min) or on re-deploys "
                        "where the service is already known-good.")

    g = parser.add_argument_group("HTTPS / TLS (NGINX + Let's Encrypt)")
    g.add_argument("--domain", default=None, metavar="DOMAIN",
                   help=(
                       "Public domain name (e.g. api.example.com). When set, NGINX is installed "
                       "as a TLS-terminating reverse proxy and certbot obtains a Let's Encrypt "
                       "certificate. The Docker port stays on loopback; NGINX faces the internet. "
                       "Requires --bind 127.0.0.1 (default). DNS A record must point to this host."
                   ))
    g.add_argument("--certbot-email", default=None, metavar="EMAIL",
                   help="Email address for Let's Encrypt certificate renewal notices. Required with --domain.")

    g = parser.add_argument_group("Server tuning")
    g.add_argument("--models-max", type=int, default=2, metavar="N",
                   help="Router: max simultaneously loaded models. (default: 2)")
    g.add_argument("--ctx-llm", type=int, default=3072, metavar="TOKENS",
                   help="Context window for the LLM. (default: 3072)")
    g.add_argument("--ctx-emb", type=int, default=2048, metavar="TOKENS",
                   help="Context window for the embedding model. (default: 2048)")
    g.add_argument("--parallel", type=int, default=1, metavar="N",
                   help="Router parallel slots. (default: 1)")

    raw = parser.parse_args(argv)

    # Resolve access profile and bind address
    profile = AccessProfile(raw.profile) if raw.profile else None

    # Derive bind_host from profile when --bind is not explicitly provided
    if raw.bind is None:
        if profile in (AccessProfile.HOME_PRIVATE, AccessProfile.PUBLIC):
            bind_host = "0.0.0.0"
        else:
            bind_host = "127.0.0.1"
    else:
        bind_host = raw.bind

    # Derive profile from bind_host when --profile is not given (backward compat)
    if profile is None:
        if bind_host == "0.0.0.0":
            profile = AccessProfile.PUBLIC
        else:
            profile = AccessProfile.LOCALHOST

    # Cross-argument validation
    domain = normalize_domain(raw.domain)
    if domain and not is_valid_domain(domain):
        parser.error("--domain must be a bare hostname like api.example.com (no http://, path, or port).")
    if bind_host == "0.0.0.0" and raw.no_publish:
        parser.error("--bind 0.0.0.0 cannot be combined with --no-publish.")
    if raw.open_firewall and bind_host != "0.0.0.0":
        parser.error("--open-firewall requires --bind 0.0.0.0.")
    if domain and bind_host != "127.0.0.1":
        parser.error("--domain requires --bind 127.0.0.1 (NGINX proxies to loopback).")
    if domain and not raw.certbot_email:
        parser.error("--certbot-email is required when --domain is set.")
    if raw.certbot_email and not domain:
        parser.error("--certbot-email has no effect without --domain.")
    if profile == AccessProfile.HOME_PRIVATE and not raw.lan_cidr:
        parser.error("--profile=home-private requires --lan-cidr (e.g. 192.168.1.0/24).")
    if raw.lan_cidr and profile != AccessProfile.HOME_PRIVATE:
        parser.error("--lan-cidr is only valid with --profile=home-private.")
    if raw.docker_network_mode == DockerNetworkMode.HOST.value and raw.no_publish:
        parser.error("--docker-network-mode=host cannot be combined with --no-publish.")
    if raw.docker_network_mode == DockerNetworkMode.HOST.value and raw.auth_mode == AuthMode.HASHED.value:
        parser.error("--docker-network-mode=host is not supported with --auth-mode hashed.")

    hf_token: Optional[str] = raw.hf_token or os.environ.get("HF_TOKEN") or None
    tailscale_authkey: Optional[str] = (
        raw.tailscale_authkey or os.environ.get("TAILSCALE_AUTHKEY") or None
    )

    try:
        network = NetworkConfig(
            bind_host=bind_host,
            port=raw.port,
            publish=not raw.no_publish,
            open_firewall=raw.open_firewall,
            configure_ufw=not raw.skip_ufw,
            access_profile=profile,
            lan_cidr=raw.lan_cidr or None,
        )
    except ValueError as exc:
        parser.error(str(exc))
    llm_spec = ModelSpec(
        hf_repo=raw.llm_repo,
        candidate_patterns=[p.strip() for p in raw.llm_candidates.split(",") if p.strip()],
        ctx_len=raw.ctx_llm,
    )
    emb_spec = ModelSpec(
        hf_repo=raw.emb_repo,
        candidate_patterns=[p.strip() for p in raw.emb_candidates.split(",") if p.strip()],
        ctx_len=raw.ctx_emb,
        is_embedding=True,
    )
    return Config(
        base_dir=Path(raw.base_dir),
        backend=BackendKind(raw.backend),
        network=network,
        swap_gib=raw.swap_gib,
        models_max=raw.models_max,
        parallel=raw.parallel,
        api_token=raw.token,
        api_token_name=raw.token_name,
        hf_token=hf_token,
        skip_download=raw.skip_download,
        llm=llm_spec,
        emb=emb_spec,
        auto_optimize=not raw.no_auto_optimize,
        allow_unverified_downloads=raw.allow_unverified_downloads,
        skip_health_check=raw.skip_health_check,
        domain=domain,
        certbot_email=raw.certbot_email or None,
        auth_mode=AuthMode(raw.auth_mode),
        tailscale_authkey=tailscale_authkey,
        docker_network_mode=DockerNetworkMode(raw.docker_network_mode),
    )


# ---------------------------------------------------------------------------
# `tokens` subcommand handlers
# ---------------------------------------------------------------------------

def _detect_auth_mode(base_dir: Path) -> AuthMode:
    """
    Infer token auth mode from on-disk artifacts.

    Priority:
      1) docker-compose.yml: presence of llama-auth sidecar → hashed
      2) tokens.json record shape (active hashed record wins)
      3) token_hashes.json (hashed) vs api_keys (plaintext) existence
      4) plaintext fallback for empty/new stores

    Checking docker-compose.yml first avoids being misled by a stale api_keys
    file left over from a previous plaintext-mode deploy when the service has
    since been redeployed in hashed mode.
    """
    compose_path = base_dir / "docker-compose.yml"
    if compose_path.exists():
        try:
            if "llama-auth" in compose_path.read_text(encoding="utf-8"):
                return AuthMode.HASHED
        except Exception:
            pass

    secrets = base_dir / "secrets"
    token_file = secrets / "tokens.json"

    if token_file.exists():
        try:
            data = json.loads(token_file.read_text(encoding="utf-8"))
            for rec in data.get("tokens", []):
                if rec.get("hash") and not rec.get("value"):
                    return AuthMode.HASHED
        except Exception:
            pass

    hash_file = secrets / "token_hashes.json"
    key_file  = secrets / "api_keys"
    if hash_file.exists() and not key_file.exists():
        return AuthMode.HASHED

    return AuthMode.PLAINTEXT


def _tokens_list(base_dir: Path, auth_mode: AuthMode) -> None:
    from llama_deploy.tokens import TokenStore
    store = TokenStore(base_dir / "secrets", auth_mode=auth_mode)
    tokens = store.list_tokens()
    if not tokens:
        print("No tokens found.")
        return

    active_count  = sum(1 for t in tokens if not t.revoked)
    revoked_count = len(tokens) - active_count

    col_id    = 20
    col_name  = 18
    col_st    = 9
    col_date  = 10
    header = (
        f"{'ID':<{col_id}}  {'Name':<{col_name}}  {'Status':<{col_st}}  {'Created':<{col_date}}"
    )
    sep = "─" * len(header)
    print(f"\nAPI Tokens  ({base_dir}/secrets/tokens.json)")
    print(sep)
    print(header)
    print(sep)
    for t in tokens:
        date = t.created_at[:10]
        print(f"{t.id:<{col_id}}  {t.name:<{col_name}}  {t.status:<{col_st}}  {date:<{col_date}}")
    print(sep)
    print(f"{active_count} active, {revoked_count} revoked\n")


def _tokens_create(base_dir: Path, name: str, auth_mode: AuthMode) -> None:
    from llama_deploy.tokens import TokenStore
    store = TokenStore(base_dir / "secrets", auth_mode=auth_mode)
    record = store.create_token(name)
    print(f'\nCreated token "{record.name}":')
    print(f"  ID    : {record.id}")
    print(f"  Value : {record.value}")
    print()
    print("  ⚠  Store this value now — it will not be shown again.")
    if store.auth_mode == AuthMode.HASHED:
        print(f"     Hashfile updated: {base_dir}/secrets/token_hashes.json")
        print("     Takes effect immediately (no container restart required).\n")
    else:
        print(f"     Keyfile updated: {base_dir}/secrets/api_keys")
        print(f"     Restart llama-server to pick up the new key:")
        print(f"     docker compose -f {base_dir}/docker-compose.yml restart llama\n")


def _tokens_revoke(base_dir: Path, token_id: str, auth_mode: AuthMode) -> None:
    from llama_deploy.tokens import TokenStore
    store = TokenStore(base_dir / "secrets", auth_mode=auth_mode)
    try:
        record = store.revoke_token(token_id)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    active = len(store.active_tokens())
    print(f'\nRevoked token "{record.name}" ({record.id}).')
    print(f"  Active tokens remaining: {active}")
    if store.auth_mode == AuthMode.HASHED:
        print(f"  Hashfile updated: {base_dir}/secrets/token_hashes.json")
        print("  Revocation takes effect immediately (no container restart required).\n")
    else:
        print(f"  Keyfile updated: {base_dir}/secrets/api_keys")
        print(f"  Restart llama-server to drop the revoked key:")
        print(f"  docker compose -f {base_dir}/docker-compose.yml restart llama\n")
    if active == 0:
        print("  ⚠  No active tokens remain. The API will reject all requests.")
        print(f"     Create a new token: python -m llama_deploy tokens create --name <name>\n")


def _tokens_show(base_dir: Path, token_id: str, auth_mode: AuthMode) -> None:
    from llama_deploy.tokens import TokenStore
    store = TokenStore(base_dir / "secrets", auth_mode=auth_mode)
    try:
        record = store.show_token(token_id)
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f'\n⚠  Displaying token value for "{record.name}" ({record.id}).')
    try:
        answer = input("  Confirm? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    if answer not in ("y", "yes"):
        print("  Aborted.")
        sys.exit(0)
    print(f"\n  Status : {record.status}")
    print(f"  Value  : {record.value}\n")


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def dispatch(argv: Optional[List[str]] = None) -> None:
    """
    Parse the top-level subcommand and hand off to the appropriate handler.
    Called by __main__.py.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m llama_deploy",
        description="llama.cpp deployment and token management tool.",
        add_help=False,
    )
    parser.add_argument("subcommand", nargs="?", default=None,
                        choices=["deploy", "tokens"])
    parser.add_argument("--help", "-h", action="store_true")

    known, remaining = parser.parse_known_args(argv)
    sub = known.subcommand

    if sub is None:
        if known.help or not sys.stdin.isatty():
            _print_top_level_help()
            return
        _run_deploy_wizard()
        return

    if sub == "deploy":
        rem = remaining or []
        if known.help or "-h" in rem or "--help" in rem:
            if "--batch" in rem:
                build_config(["--help"])
            else:
                _print_top_level_help()
            return
        if "--batch" in rem:
            remaining2 = [r for r in rem if r != "--batch"]
            cfg = build_config(remaining2)
            from llama_deploy.orchestrator import run_deploy
            run_deploy(cfg)
        elif known.help:
            _print_top_level_help()
        elif sys.stdin.isatty():
            _run_deploy_wizard()
        else:
            _print_top_level_help()
        return

    if sub == "tokens":
        rem = remaining or []
        if known.help or "-h" in rem or "--help" in rem:
            _dispatch_tokens(["-h"])
            return
        _dispatch_tokens(rem)
        return


def _print_top_level_help() -> None:
    print(
        "Usage: python -m llama_deploy [deploy|tokens] [options]\n\n"
        "Subcommands:\n"
        "  deploy              Interactive wizard (TTY) or --batch for non-interactive\n"
        "  deploy --batch …    Non-interactive deployment (use -h for flag list)\n"
        "  tokens list         List all API tokens\n"
        "  tokens create       Create a new named API token\n"
        "  tokens revoke <id>  Revoke an API token\n"
        "  tokens show   <id>  Display a token value (with confirmation)\n\n"
        "Examples:\n"
        "  python -m llama_deploy\n"
        "  python -m llama_deploy deploy --batch --bind 127.0.0.1 --port 8080\n"
        "  python -m llama_deploy tokens list\n"
        "  python -m llama_deploy tokens create --name my-app\n"
        "  python -m llama_deploy tokens revoke tk_4f9a2b1c3d8e\n"
    )


def _run_deploy_wizard() -> None:
    from llama_deploy.wizard import run_wizard
    from llama_deploy.orchestrator import run_deploy
    cfg = run_wizard()
    run_deploy(cfg)


def _dispatch_tokens(argv: List[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m llama_deploy tokens",
        description="Manage API tokens for the deployed llama-server.",
    )
    parser.add_argument("--base-dir", default="/opt/llama", metavar="DIR",
                        help="Base directory used during deployment. (default: /opt/llama)")
    parser.add_argument("--auth-mode", default=None, choices=[m.value for m in AuthMode],
                        metavar="MODE",
                        help="Override auth mode detection: 'plaintext' or 'hashed'. "
                             "Auto-detected from docker-compose.yml when omitted.")
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("list", help="List all tokens (including revoked).")

    p_create = sub.add_parser("create", help="Create a new named token.")
    p_create.add_argument("--name", required=True, metavar="NAME",
                          help="Human-readable label for the token.")

    p_revoke = sub.add_parser("revoke", help="Revoke a token by ID.")
    p_revoke.add_argument("id", metavar="TOKEN_ID", help="The tk_... ID to revoke.")

    p_show = sub.add_parser("show", help="Display a token's value (requires confirmation).")
    p_show.add_argument("id", metavar="TOKEN_ID", help="The tk_... ID to show.")

    args = parser.parse_args(argv)
    base = Path(args.base_dir)
    auth_mode = AuthMode(args.auth_mode) if args.auth_mode else _detect_auth_mode(base)

    if args.action == "list":
        _tokens_list(base, auth_mode)
    elif args.action == "create":
        _tokens_create(base, args.name, auth_mode)
    elif args.action == "revoke":
        _tokens_revoke(base, args.id, auth_mode)
    elif args.action == "show":
        _tokens_show(base, args.id, auth_mode)
