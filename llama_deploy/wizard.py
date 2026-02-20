"""
Interactive Human-in-the-Loop configuration wizard.

Uses only stdlib input() — no third-party dependencies beyond what the rest
of the package already requires. ANSI escape codes are used for colour and
box-drawing; they degrade gracefully in terminals without colour support.

Public API
----------
    run_wizard() -> Config

The wizard collects all deployment parameters interactively, shows a review
block, and returns a fully validated Config on confirmation. It performs NO
disk I/O — all side effects (token creation, directory creation, Docker pull,
etc.) happen in the orchestrator after this function returns.

Call only when sys.stdin.isatty() is True. Non-interactive callers should use
cli.build_config() directly.
"""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

from llama_deploy.config import (
    AuthMode,
    BackendKind,
    Config,
    ModelSpec,
    NetworkConfig,
)

# ---------------------------------------------------------------------------
# ANSI helpers (degrade silently on dumb terminals)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(t: str)   -> str: return _c("1", t)
def _yellow(t: str) -> str: return _c("33", t)
def _cyan(t: str)   -> str: return _c("36", t)
def _green(t: str)  -> str: return _c("32", t)
def _red(t: str)    -> str: return _c("31", t)
def _dim(t: str)    -> str: return _c("2", t)


# ---------------------------------------------------------------------------
# Low-level UI primitives
# ---------------------------------------------------------------------------

_BOX_WIDTH = 54


def _header(step: int, total: int, title: str) -> None:
    print()
    top = "╔" + "═" * (_BOX_WIDTH - 2) + "╗"
    mid = f"║  {_bold(f'Step {step}/{total} · {title}')}"
    mid += " " * (_BOX_WIDTH - 2 - len(f"  Step {step}/{total} · {title}")) + "║"
    bot = "╚" + "═" * (_BOX_WIDTH - 2) + "╝"
    print(_cyan(top))
    print(_cyan("║") + mid[1:-1] + _cyan("║"))
    print(_cyan(bot))
    print()


def _section(title: str) -> None:
    print()
    print(_bold(title))
    print("─" * len(title))


def _warn(msg: str) -> None:
    print(_yellow(f"  ⚠  WARNING: {msg}"))


def _info(msg: str) -> None:
    print(_dim(f"  {msg}"))


def _prompt(msg: str, default: str = "") -> str:
    """Show a prompt, return stripped input or default on empty enter."""
    hint = _dim(f" [{default}]") if default else ""
    try:
        val = input(f"  {msg}{hint}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return val if val else default


def _prompt_int(msg: str, default: int, min_val: int = 1, max_val: int = 9999) -> int:
    while True:
        raw = _prompt(msg, str(default))
        try:
            v = int(raw)
            if min_val <= v <= max_val:
                return v
        except ValueError:
            pass
        print(_red(f"  Please enter a number between {min_val} and {max_val}."))


def _choose(
    options: List[Tuple[str, str]],
    default: int = 1,
    extra_label: Optional[str] = None,
) -> int:
    """
    Print a numbered menu and return the chosen 1-based index.

    options: list of (short_label, description) pairs
    extra_label: if set, appended as the last option (e.g. "Custom")
    Returns: 1-based index
    """
    all_opts = list(options)
    if extra_label:
        all_opts.append((extra_label, ""))
    for i, (label, desc) in enumerate(all_opts, 1):
        marker = _green(f"[{i}]") if i == default else f"[{i}]"
        suffix = f"  {_dim(desc)}" if desc else ""
        print(f"  {marker} {label}{suffix}")
    while True:
        raw = _prompt("Choice", str(default))
        try:
            v = int(raw)
            if 1 <= v <= len(all_opts):
                return v
        except ValueError:
            pass
        print(_red(f"  Please enter a number between 1 and {len(all_opts)}."))


def _confirm(msg: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = _prompt(msg, hint)
    if raw.upper() in ("Y/N", "Y", "YES", ""):
        return True if raw.upper() in ("Y", "YES", "") else default
    return raw.lower() in ("y", "yes")


# ---------------------------------------------------------------------------
# Wizard steps
# ---------------------------------------------------------------------------

def _step_backend() -> BackendKind:
    _header(1, 5, "Backend")
    options = [
        ("CPU only",     "works everywhere — no GPU required"),
        ("NVIDIA GPU",   "CUDA — requires nvidia-docker"),
        ("AMD GPU",      "ROCm — requires ROCm drivers"),
        ("Vulkan GPU",   "cross-vendor GPU acceleration"),
        ("Intel GPU",    "SYCL / OpenVINO"),
    ]
    idx = _choose(options, default=1)
    kind = list(BackendKind)[idx - 1]
    print(_dim(f"\n  Selected: {kind.value}  →  {kind.docker_image()}"))
    return kind


def _step_models() -> Tuple[ModelSpec, ModelSpec]:
    _header(2, 5, "Models")

    # --- LLM ---
    _section("Text generation model (LLM)")
    llm_presets = [
        ("Ministral-3-14B-Instruct-2512", "unsloth/Ministral-3-14B-Instruct-2512-GGUF · ~8-9 GB · 4-bit"),
        ("Qwen3-14B",    "Qwen/Qwen3-14B-GGUF · ~9 GB · 4-bit"),
    ]
    llm_presets.extend([
        ("Llama 3.1 8B Instruct", "bartowski/Llama-3.1-8B-Instruct-GGUF"),
        ("Mistral 7B Instruct v0.3", "bartowski/Mistral-7B-Instruct-v0.3-GGUF"),
        ("Gemma 2 9B IT", "bartowski/gemma-2-9b-it-GGUF"),
        ("Phi-3.5 mini instruct", "bartowski/Phi-3.5-mini-instruct-GGUF"),
    ])
    llm_idx = _choose(llm_presets, default=1, extra_label="Custom HuggingFace repo")
    if llm_idx <= len(llm_presets):
        llm_repos = [
            ("unsloth/Ministral-3-14B-Instruct-2512-GGUF", ["Q4_K_M", "Q5_K_M", "Q4_0", "Q3_K_M"]),
            ("Qwen/Qwen3-14B-GGUF", ["Q4_K_M", "Q5_K_M", "Q4_0", "Q3_K_M"]),
        ]
        llm_repos.extend([
            ("bartowski/Llama-3.1-8B-Instruct-GGUF", ["Q4_K_M", "Q5_K_M", "Q4_0", "Q3_K_M"]),
            ("bartowski/Mistral-7B-Instruct-v0.3-GGUF", ["Q4_K_M", "Q5_K_M", "Q4_0"]),
            ("bartowski/gemma-2-9b-it-GGUF", ["Q4_K_M", "Q5_K_M", "Q4_0"]),
            ("bartowski/Phi-3.5-mini-instruct-GGUF", ["Q4_K_M", "Q5_K_M", "Q4_0"]),
        ])
        llm_repo, llm_patterns = llm_repos[llm_idx - 1]
    else:
        llm_repo = _prompt("HuggingFace repo (e.g. bartowski/Mistral-7B-GGUF)")
        if not llm_repo:
            llm_repo = "unsloth/Ministral-3-14B-Instruct-2512-GGUF"
            llm_patterns = ["Q4_K_M", "Q5_K_M", "Q4_0", "Q3_K_M"]
        else:
            raw_pats = _prompt("GGUF filename patterns (comma-separated)", "Q4_K_M,Q5_K_M")
            llm_patterns = [p.strip() for p in raw_pats.split(",") if p.strip()]
    llm_ctx = _prompt_int("Context window (tokens)", default=4096, min_val=128, max_val=131072)

    # --- Embedding ---
    _section("Embedding model")
    emb_presets = [
        ("Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B-GGUF · ~0.6 GB"),
        ("Qwen3-Embedding-4B",   "Qwen/Qwen3-Embedding-4B-GGUF · ~3 GB"),
    ]
    emb_presets.extend([
        ("Qwen3-Embedding-8B", "Qwen/Qwen3-Embedding-8B-GGUF"),
        ("BGE base EN v1.5", "lmstudio-community/bge-base-en-v1.5-GGUF"),
        ("BGE small EN v1.5", "lmstudio-community/bge-small-en-v1.5-GGUF"),
    ])
    emb_idx = _choose(emb_presets, default=1, extra_label="Custom HuggingFace repo")
    if emb_idx <= len(emb_presets):
        emb_repos = [
            ("Qwen/Qwen3-Embedding-0.6B-GGUF", ["Q8_0", "F16", "Q6_K", "Q4_K_M"]),
            ("Qwen/Qwen3-Embedding-4B-GGUF",   ["Q8_0", "F16", "Q6_K", "Q4_K_M"]),
        ]
        emb_repos.extend([
            ("Qwen/Qwen3-Embedding-8B-GGUF", ["Q8_0", "F16", "Q6_K", "Q4_K_M"]),
            ("lmstudio-community/bge-base-en-v1.5-GGUF", ["Q8_0", "F16", "Q6_K", "Q4_K_M"]),
            ("lmstudio-community/bge-small-en-v1.5-GGUF", ["Q8_0", "F16", "Q6_K", "Q4_K_M"]),
        ])
        emb_repo, emb_patterns = emb_repos[emb_idx - 1]
    else:
        emb_repo = _prompt("HuggingFace repo")
        if not emb_repo:
            emb_repo = "Qwen/Qwen3-Embedding-0.6B-GGUF"
            emb_patterns = ["Q8_0", "F16", "Q6_K", "Q4_K_M"]
        else:
            raw_pats = _prompt("GGUF filename patterns (comma-separated)", "Q8_0,F16,Q4_K_M")
            emb_patterns = [p.strip() for p in raw_pats.split(",") if p.strip()]
    emb_ctx = _prompt_int("Context window (tokens)", default=2048, min_val=128, max_val=32768)

    llm_spec = ModelSpec(hf_repo=llm_repo, candidate_patterns=llm_patterns, ctx_len=llm_ctx)
    emb_spec = ModelSpec(hf_repo=emb_repo, candidate_patterns=emb_patterns, ctx_len=emb_ctx, is_embedding=True)
    return llm_spec, emb_spec


def _step_network() -> Tuple[NetworkConfig, Optional[str], Optional[str]]:
    """
    Returns (NetworkConfig, domain_or_None, certbot_email_or_None).

    Option 2 (HTTPS + domain) keeps the Docker port on loopback and sets up
    NGINX as the public-facing TLS terminator. The caller passes domain and
    certbot_email through to Config so the orchestrator can run nginx.py.
    """
    _header(3, 5, "Network")
    options = [
        ("Localhost only",  "127.0.0.1 — most secure, for local apps only"),
        ("HTTPS + domain",  "NGINX + Let's Encrypt — recommended for internet exposure"),
        ("All interfaces",  "0.0.0.0 — direct exposure without TLS"),
        ("Docker network",  "no host port — only from other containers"),
    ]
    idx = _choose(options, default=1)

    domain: Optional[str] = None
    certbot_email: Optional[str] = None

    if idx == 1:
        bind, publish, open_fw = "127.0.0.1", True, False

    elif idx == 2:
        # NGINX terminates TLS; llama-server stays on loopback
        bind, publish, open_fw = "127.0.0.1", True, False
        print()
        _info("NGINX will listen on :80 and :443 and proxy to the loopback port.")
        _info("Make sure your domain's DNS A record already points to this server.")
        print()
        while not domain or "." not in domain:
            domain = _prompt("Domain name (e.g. api.example.com)")
            if not domain or "." not in domain:
                print(_red("  Please enter a valid domain name."))
        while not certbot_email or "@" not in certbot_email:
            certbot_email = _prompt("Email for Let's Encrypt renewal notices")
            if not certbot_email or "@" not in certbot_email:
                print(_red("  Please enter a valid email address."))

    elif idx == 3:
        bind, publish, open_fw = "0.0.0.0", True, False
        _warn("The API will be reachable without TLS from any host on your network.")
        open_fw = _confirm("Open this port in UFW?", default=True)

    else:
        bind, publish, open_fw = "127.0.0.1", False, False

    port = 8080
    if publish:
        port = _prompt_int("Internal port", default=8080, min_val=1024, max_val=65535)

    configure_ufw = True
    if idx not in (2, 3):
        configure_ufw = _confirm("Configure UFW firewall? (keeps SSH open, denies rest)", default=True)

    network = NetworkConfig(
        bind_host=bind,
        port=port,
        publish=publish,
        open_firewall=open_fw,
        configure_ufw=configure_ufw,
    )
    return network, domain, certbot_email


def _step_token() -> Tuple[str, AuthMode]:
    _header(4, 5, "API Token")
    print("  You will receive an API key to authenticate requests.")
    print("  Give it a name so you can identify and revoke it later.")
    print()
    name = _prompt("Token name", default="default")

    _section("Token storage")
    _info("Plaintext: llama-server compares tokens directly — simpler, no extra services.")
    _info("Hashed:    only SHA-256 hashes stored; plaintext never written to disk.")
    _info("           NGINX is installed as an auth proxy (required for hashed mode).")
    print()
    options = [
        ("Plaintext", "simpler — works with all network modes"),
        ("Hashed",    "more secure — NGINX auth_request + sidecar; instant revocation"),
    ]
    idx = _choose(options, default=1)
    auth_mode = AuthMode.PLAINTEXT if idx == 1 else AuthMode.HASHED

    if auth_mode == AuthMode.HASHED:
        _info("NGINX will be installed and configured as a local auth-enforcing proxy.")

    return name or "default", auth_mode


def _step_system() -> Tuple[int, str]:
    _header(5, 5, "System")
    swap_gib = _prompt_int("Swap file to create if none exists (GiB)", default=8, min_val=0, max_val=256)
    base_dir = _prompt("Base directory for models, config, secrets", default="/opt/llama")
    return swap_gib, base_dir or "/opt/llama"


# ---------------------------------------------------------------------------
# Review + confirm
# ---------------------------------------------------------------------------

def _review(cfg: Config) -> None:
    print()
    print(_bold("─" * _BOX_WIDTH))
    print(_bold("  Review"))
    print(_bold("─" * _BOX_WIDTH))

    net = cfg.network
    rows = [
        ("Backend",    cfg.image),
        ("LLM",        f"{cfg.llm.effective_alias}  (ctx {cfg.llm.ctx_len})"),
        ("Embedding",  f"{cfg.emb.effective_alias}  (ctx {cfg.emb.ctx_len})"),
        ("Endpoint",   cfg.public_base_url),
        ("Firewall",   "UFW enabled" if net.configure_ufw else "UFW skipped"),
        ("Token name", f'"{cfg.api_token_name}"'),
        ("Auth",       cfg.auth_mode.value),
        ("Base dir",   str(cfg.base_dir)),
    ]
    if cfg.use_tls:
        rows.insert(4, ("TLS domain",  cfg.domain))
        rows.insert(5, ("Cert email",  cfg.certbot_email))
    col = max(len(k) for k, _ in rows) + 2
    for k, v in rows:
        print(f"  {_dim(k.ljust(col))} {v}")
    print(_bold("─" * _BOX_WIDTH))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_wizard() -> Config:
    """
    Run the interactive wizard and return a fully validated Config.
    Performs no disk I/O — all side effects happen in the orchestrator.
    Exits cleanly on Ctrl-C or EOF.
    """
    print()
    print(_cyan("╔" + "═" * (_BOX_WIDTH - 2) + "╗"))
    print(_cyan("║") + _bold(f"  llama.cpp Deployment Wizard".center(_BOX_WIDTH - 2)) + _cyan("║"))
    print(_cyan("║") + "  Configure your OpenAI-compatible AI service".center(_BOX_WIDTH - 2) + _cyan("║"))
    print(_cyan("╚" + "═" * (_BOX_WIDTH - 2) + "╝"))

    backend             = _step_backend()
    llm_spec, emb_spec  = _step_models()
    network, domain, certbot_email = _step_network()
    token_name, auth_mode = _step_token()
    swap_gib, base_dir_str = _step_system()

    from pathlib import Path
    cfg = Config(
        base_dir=Path(base_dir_str),
        backend=backend,
        network=network,
        swap_gib=swap_gib,
        models_max=2,
        parallel=1,
        api_token=None,
        api_token_name=token_name,
        hf_token=None,
        skip_download=False,
        llm=llm_spec,
        emb=emb_spec,
        allow_unverified_downloads=False,
        domain=domain,
        certbot_email=certbot_email,
        auth_mode=auth_mode,
    )

    _review(cfg)

    if not _confirm("Proceed with deployment?", default=True):
        print("\n  Aborted.")
        sys.exit(0)

    return cfg
