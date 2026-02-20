"""
Deployment orchestrator: Step protocol, run_steps runner, and run_deploy().

The Step dataclass replaces the original (label, fn) tuple list that used
string-matching to capture return values (Bug 4). Every step's return value
is stored in step.result and can be read by subsequent steps without re-calling
the function (Bug 1).

Token management uses TokenStore (tokens.py) instead of the old ensure_token_file.
The token value is shown ONCE after deployment if it was newly created.

All steps â€" including HF download, config writing, and Docker startup â€" are
included in a single all_steps list so the progress bar reflects the true
total (Bug 7).

Auth modes
----------
Plaintext (default):
  llama-server uses --api-key-file; tokens stored as plaintext in api_keys.

Hashed:
  NGINX auth_request delegates to the llama-auth sidecar; only SHA-256
  hashes are stored in token_hashes.json; plaintext never touches disk.
  - If cfg.use_tls: the TLS step also configures auth_request in NGINX.
  - Otherwise: a dedicated local_proxy_step installs NGINX without certbot.
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Callable, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Progress helper (tqdm with a graceful fallback)
# ---------------------------------------------------------------------------

class _TqdmStub:
    """Fallback progress helper when tqdm is not available yet."""

    def __init__(self, total: int = 0, desc: str = "", unit: str = "") -> None:
        self.total = total
        self.desc = desc
        self.unit = unit

    @staticmethod
    def write(msg: str) -> None:
        print(msg, flush=True)

    def update(self, _n: int = 1) -> None:
        return

    def __enter__(self) -> "_TqdmStub":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        return False


tqdm = _TqdmStub


def _ensure_tqdm(allow_install: bool = True) -> None:
    """
    Ensure tqdm is importable. On non-root import paths we skip package install
    attempts and fall back to a minimal stub.
    """
    global tqdm

    try:
        from tqdm import tqdm as real_tqdm
        tqdm = real_tqdm
        return
    except Exception:
        pass

    if not allow_install:
        return

    print("[BOOT] Installing tqdm (python3-tqdm)...", flush=True)
    subprocess.run(
        ["bash", "-lc", "apt-get update -y && apt-get install -y python3-tqdm"],
        check=False,
    )
    try:
        from tqdm import tqdm as real_tqdm
        tqdm = real_tqdm
        return
    except Exception:
        print("[BOOT] apt install failed; trying pip...", flush=True)
        subprocess.run(
            ["bash", "-lc", "apt-get update -y && apt-get install -y python3-pip"],
            check=False,
        )
        subprocess.run(
            ["bash", "-lc", "python3 -m pip install --no-cache-dir -U pip tqdm"],
            check=False,
        )
        try:
            from tqdm import tqdm as real_tqdm
            tqdm = real_tqdm
            return
        except Exception as e:
            print(f"[WARN] tqdm unavailable ({e}); continuing without progress bar.", flush=True)
            tqdm = _TqdmStub


_ensure_tqdm(allow_install=False)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """
    One idempotent deployment step.

    result is populated by run_steps() after fn() returns. Downstream steps
    close over the Step object and read step.result rather than re-calling fn().
    """

    label:   str
    fn:      Callable[[], Any]
    skip_if: Optional[Callable[[], bool]] = None
    result:  Any = field(default=None, repr=False)


@dataclass
class TokenRuntime:
    """
    Runtime token material used by deployment validation and summary printing.

    temporary_id is set only when a temporary validation token was created
    (hashed mode re-deploy where existing tokens have no recoverable plaintext).
    """

    value: str
    temporary_id: Optional[str] = None


_LLM_RETRY_PRESETS: List[Tuple[str, str, List[str]]] = [
    ("Qwen3-8B (default)", "Qwen/Qwen3-8B-GGUF", ["Q4_K_M", "Q5_K_M", "Q4_0", "Q3_K_M"]),
    ("Ministral-3-14B-Instruct-2512", "unsloth/Ministral-3-14B-Instruct-2512-GGUF", ["Q4_K_M", "Q5_K_M", "Q4_0", "Q3_K_M"]),
    ("Qwen3-14B", "Qwen/Qwen3-14B-GGUF", ["Q4_K_M", "Q5_K_M", "Q4_0", "Q3_K_M"]),
    ("Llama 3.1 8B Instruct", "bartowski/Llama-3.1-8B-Instruct-GGUF", ["Q4_K_M", "Q5_K_M", "Q4_0", "Q3_K_M"]),
    ("Mistral 7B Instruct v0.3", "bartowski/Mistral-7B-Instruct-v0.3-GGUF", ["Q4_K_M", "Q5_K_M", "Q4_0"]),
    ("Gemma 2 9B IT", "bartowski/gemma-2-9b-it-GGUF", ["Q4_K_M", "Q5_K_M", "Q4_0"]),
    ("Phi-3.5 mini instruct", "bartowski/Phi-3.5-mini-instruct-GGUF", ["Q4_K_M", "Q5_K_M", "Q4_0"]),
]

_EMB_RETRY_PRESETS: List[Tuple[str, str, List[str]]] = [
    ("Qwen3-Embedding-0.6B (default)", "Qwen/Qwen3-Embedding-0.6B-GGUF", ["Q8_0", "F16", "Q6_K", "Q4_K_M"]),
    ("Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-4B-GGUF", ["Q8_0", "F16", "Q6_K", "Q4_K_M"]),
    ("Qwen3-Embedding-8B", "Qwen/Qwen3-Embedding-8B-GGUF", ["Q8_0", "F16", "Q6_K", "Q4_K_M"]),
    ("BGE base EN v1.5", "lmstudio-community/bge-base-en-v1.5-GGUF", ["Q8_0", "F16", "Q6_K", "Q4_K_M"]),
    ("BGE small EN v1.5", "lmstudio-community/bge-small-en-v1.5-GGUF", ["Q8_0", "F16", "Q6_K", "Q4_K_M"]),
]

_DEFAULT_LLM_REPO = "Qwen/Qwen3-8B-GGUF"
_DEFAULT_LLM_PATTERNS = ["Q4_K_M", "Q5_K_M", "Q4_0", "Q3_K_M"]
_MID_LLM_REPO = "Qwen/Qwen3-8B-GGUF"
_MID_LLM_PATTERNS = ["Q4_K_M", "Q5_K_M", "Q4_0", "Q3_K_M"]
_SMALL_LLM_REPO = "bartowski/Phi-3.5-mini-instruct-GGUF"
_SMALL_LLM_PATTERNS = ["Q4_K_M", "Q5_K_M", "Q4_0"]


def _spec_key(spec) -> str:
    return f"{spec.hf_repo}|{','.join(spec.candidate_patterns)}|{int(spec.is_embedding)}|{spec.ctx_len}"


def _prompt_model_retry_choice(kind: str, current_spec, tried: Set[str]):
    """
    Ask the user which model to try next after a failed download.
    Returns a new ModelSpec, or None if the user cancels.
    """
    from llama_deploy.config import ModelSpec

    presets = _EMB_RETRY_PRESETS if current_spec.is_embedding else _LLM_RETRY_PRESETS
    candidates: List[Tuple[str, ModelSpec]] = []
    for label, repo, pats in presets:
        spec = ModelSpec(
            hf_repo=repo,
            candidate_patterns=list(pats),
            ctx_len=current_spec.ctx_len,
            is_embedding=current_spec.is_embedding,
        )
        if _spec_key(spec) not in tried:
            candidates.append((label, spec))

    tqdm.write(f"[CHOICE] Select another {kind} model:")
    for i, (label, spec) in enumerate(candidates, 1):
        tqdm.write(f"  [{i}] {label}  ->  {spec.hf_repo}")
    custom_idx = len(candidates) + 1
    stop_idx = len(candidates) + 2
    tqdm.write(f"  [{custom_idx}] Custom HuggingFace repo")
    tqdm.write(f"  [{stop_idx}] Stop retries")

    while True:
        try:
            raw = input(f"Choice [1-{stop_idx}] (default {custom_idx}): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not raw:
            choice = custom_idx
        elif raw.isdigit():
            choice = int(raw)
        else:
            tqdm.write(f"[WARN] Please enter a number from 1 to {stop_idx}.")
            continue

        if 1 <= choice <= len(candidates):
            return candidates[choice - 1][1]
        if choice == custom_idx:
            default_pats = ",".join(current_spec.candidate_patterns) if current_spec.candidate_patterns else (
                "Q8_0,F16,Q6_K,Q4_K_M" if current_spec.is_embedding else "Q4_K_M,Q5_K_M,Q4_0,Q3_K_M"
            )
            repo = input("HuggingFace repo: ").strip()
            if not repo:
                tqdm.write("[WARN] Repo cannot be empty.")
                continue
            raw_pats = input(f"GGUF filename patterns (comma-separated) [{default_pats}]: ").strip()
            if not raw_pats:
                raw_pats = default_pats
            pats = [p.strip() for p in raw_pats.split(",") if p.strip()]
            if not pats:
                tqdm.write("[WARN] At least one pattern is required.")
                continue
            return ModelSpec(
                hf_repo=repo,
                candidate_patterns=pats,
                ctx_len=current_spec.ctx_len,
                is_embedding=current_spec.is_embedding,
            )
        if choice == stop_idx:
            return None
        tqdm.write(f"[WARN] Please enter a number from 1 to {stop_idx}.")


def _resolve_model_with_retry(cfg, initial_spec, kind: str):
    """
    Try downloading a model; on failure in interactive mode, let the user pick
    another model and retry until success or cancellation.
    """
    from llama_deploy.model import resolve_model

    spec = initial_spec
    tried: Set[str] = set()
    attempt = 0

    while True:
        attempt += 1
        tried.add(_spec_key(spec))
        if attempt > 1:
            tqdm.write(f"[INFO] Retrying {kind} with {spec.hf_repo} ...")
        try:
            return resolve_model(
                spec,
                cfg.models_dir,
                cfg.hf_token,
                allow_unverified_downloads=cfg.allow_unverified_downloads,
            )
        except SystemExit as e:
            if not sys.stdin.isatty():
                raise
            if int(getattr(e, "code", 1) or 1) == 0:
                raise
            tqdm.write(f"[WARN] {kind} model attempt failed for {spec.hf_repo}.")
            try:
                ans = input(f"Switch to a different {kind} model and retry? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                raise
            if ans not in ("y", "yes"):
                raise
            next_spec = _prompt_model_retry_choice(kind, spec, tried)
            if next_spec is None:
                raise
            spec = next_spec


def _detect_mem_total_gib() -> Optional[float]:
    """
    Return total system RAM in GiB from /proc/meminfo, or None if unavailable.
    """
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kib = float(parts[1])
                        return kib / (1024.0 * 1024.0)
    except Exception:
        return None
    return None


def _auto_optimize_cfg(cfg):
    """
    Tune deployment settings using host specs to reduce OOM risk.

    Strategy
    --------
    - Detect RAM and CPU core count.
    - Downshift the default LLM on low-memory hosts.
    - Clamp ctx / swap for safer startup on constrained systems.
    - Keep explicit custom --llm-repo choices intact (unless they match default).

    Override rules
    --------------
    - A setting is only changed when it would still be at its default value.
      If the user already overrode a value (e.g. --ctx-llm 8192 on a 12 GiB
      machine), that choice is respected and flagged with [AUTO-SKIP] so the
      user can see the recommendation without being silently overruled.
    - Every change is logged with [AUTO] old -> new so decisions are auditable.
    - If no /proc/meminfo is available the entire pass is skipped cleanly.
    """
    from llama_deploy.log import log_line

    mem_gib = _detect_mem_total_gib()
    cores = os.cpu_count() or 1
    if mem_gib is None:
        tqdm.write("[AUTO] Could not read /proc/meminfo; skipping auto optimization.")
        return cfg

    tuned = cfg
    changes: list = []   # str entries logged as [AUTO]
    skipped: list = []   # str entries logged as [AUTO-SKIP]

    def _log_change(msg: str) -> None:
        changes.append(msg)
        log_line(f"[AUTO] {msg}")

    def _log_skip(msg: str) -> None:
        skipped.append(msg)
        log_line(f"[AUTO-SKIP] {msg}")

    # ---- models_max --------------------------------------------------------
    # Keep at least 2 because this deployment always configures one LLM preset
    # and one embedding preset. Lowering to 1 leads to startup model-count
    # failures in some llama-server builds.
    if tuned.models_max < 2:
        _log_change(f"models_max: {tuned.models_max} -> 2  (requires LLM + embedding presets)")
        tuned = replace(tuned, models_max=2)

    # ---- swap --------------------------------------------------------------
    if mem_gib < 8:
        rec_swap = 24
    elif mem_gib < 16:
        rec_swap = 16
    else:
        rec_swap = tuned.swap_gib  # no change needed

    if tuned.swap_gib < rec_swap:
        _log_change(f"swap_gib: {tuned.swap_gib} -> {rec_swap}  (OOM guard for {mem_gib:.1f} GiB RAM)")
        tuned = replace(tuned, swap_gib=rec_swap)

    # ---- LLM repo / ctx  ---------------------------------------------------
    # Only auto-switch when the user is running the default LLM.
    using_default_llm = (
        tuned.llm.hf_repo == _DEFAULT_LLM_REPO
        and tuned.llm.candidate_patterns == _DEFAULT_LLM_PATTERNS
    )

    if using_default_llm:
        if mem_gib < 10:
            rec_repo, rec_pats, rec_ctx = _SMALL_LLM_REPO, list(_SMALL_LLM_PATTERNS), min(tuned.llm.ctx_len, 2048)
            label = "Phi-3.5-mini"
        elif mem_gib < 16:
            rec_repo, rec_pats, rec_ctx = _MID_LLM_REPO, list(_MID_LLM_PATTERNS), min(tuned.llm.ctx_len, 3072)
            label = "Qwen3-8B"
        elif mem_gib < 24 and tuned.llm.ctx_len > 3072:
            rec_repo, rec_pats, rec_ctx = tuned.llm.hf_repo, tuned.llm.candidate_patterns, 3072
            label = None
        else:
            rec_repo, rec_pats, rec_ctx, label = None, None, None, None

        if rec_repo is not None and rec_repo != tuned.llm.hf_repo:
            _log_change(
                f"llm_repo: Qwen3-8B -> {label}  (RAM {mem_gib:.1f} GiB)"
            )
            llm = replace(tuned.llm, hf_repo=rec_repo, candidate_patterns=rec_pats,
                          ctx_len=rec_ctx)
            tuned = replace(tuned, llm=llm)
        elif rec_ctx is not None and rec_ctx != tuned.llm.ctx_len:
            _log_change(f"llm_ctx: {tuned.llm.ctx_len} -> {rec_ctx}  (RAM {mem_gib:.1f} GiB < 24 GiB)")
            tuned = replace(tuned, llm=replace(tuned.llm, ctx_len=rec_ctx))
    else:
        # User chose a custom model — report what we would have done without overriding.
        if mem_gib < 10:
            _log_skip(
                f"Would switch LLM to Phi-3.5-mini (RAM {mem_gib:.1f} GiB), "
                f"but keeping user-selected {tuned.llm.hf_repo}."
            )
        elif mem_gib < 16:
            _log_skip(
                f"Would switch LLM to Qwen3-8B (RAM {mem_gib:.1f} GiB), "
                f"but keeping user-selected {tuned.llm.hf_repo}."
            )

    # ---- embedding ctx -----------------------------------------------------
    if mem_gib < 8 and tuned.emb.ctx_len > 1024:
        _log_change(f"emb_ctx: {tuned.emb.ctx_len} -> 1024  (RAM {mem_gib:.1f} GiB < 8 GiB)")
        tuned = replace(tuned, emb=replace(tuned.emb, ctx_len=1024))

    # ---- summary -----------------------------------------------------------
    tqdm.write(f"[AUTO] Host specs: RAM={mem_gib:.1f} GiB, CPU={cores} cores")
    if changes:
        for c in changes:
            tqdm.write(f"[AUTO] {c}")
    else:
        tqdm.write("[AUTO] No tuning changes needed.")
    for s in skipped:
        tqdm.write(f"[AUTO-SKIP] {s}")

    return tuned


def run_steps(steps: List[Step], bar: tqdm) -> None:
    from llama_deploy.log import log_line
    import time

    for step in steps:
        if step.skip_if and step.skip_if():
            tqdm.write(f"[SKIP] {step.label}")
            log_line(f"[SKIP] {step.label}")
            bar.update(1)
            continue

        tqdm.write(f"\n[STEP] {step.label}")
        log_line(f"[STEP] {step.label}")
        t0 = time.time()
        step.result = step.fn()
        elapsed = time.time() - t0
        tqdm.write(f"[DONE] {step.label} ({elapsed:.1f}s)")
        log_line(f"[DONE] {step.label} ({elapsed:.1f}s)")
        bar.update(1)


# ---------------------------------------------------------------------------
# Token helper
# ---------------------------------------------------------------------------

def _ensure_first_token(cfg) -> TokenRuntime:
    """
    Return token material for deployment validation.

    Plaintext mode: reuse existing token if one exists; create on first run.
    Hashed mode: only reuse tokens that were created in hashed mode (have a
    stored hash). Plaintext-mode tokens are incompatible — the sidecar reads
    token_hashes.json and a plaintext token has no entry there. On migration
    from plaintext → hashed, a new hashed token is created automatically.
    """
    from llama_deploy.config import AuthMode
    from llama_deploy.tokens import TokenStore

    store = TokenStore(cfg.secrets_dir, auth_mode=cfg.auth_mode)
    active = store.active_tokens()

    if cfg.auth_mode == AuthMode.HASHED:
        hashed = [t for t in active if t.hash]
        if hashed:
            # Re-deploy in hashed mode: proper tokens exist.
            # Plaintext is unrecoverable; use a temporary smoke-test token.
            smoke = store.create_token("__deploy-smoke-test")
            tqdm.write(f"[OK] Created temporary smoke-test token ({smoke.id}) for validation.")
            return TokenRuntime(value=smoke.value, temporary_id=smoke.id)
        if active:
            tqdm.write(
                f"[WARN] {len(active)} plaintext token(s) found but unusable in hashed mode "
                "(no hash stored) — creating a new hashed token."
            )
        record = store.create_token(cfg.api_token_name, value=cfg.api_token)
        tqdm.write(f"[OK] Created token '{record.name}' ({record.id})")
        return TokenRuntime(value=record.value)

    # Plaintext mode
    if active:
        tqdm.write(f"[OK] Existing token found in store: {active[0].id}")
        if active[0].value:
            return TokenRuntime(value=active[0].value)

    record = store.create_token(cfg.api_token_name, value=cfg.api_token)
    tqdm.write(f"[OK] Created token '{record.name}' ({record.id})")
    return TokenRuntime(value=record.value)


def _is_new_token(cfg) -> bool:
    """Return True if no tokens exist yet (first-run detection for post-deploy message)."""
    from llama_deploy.tokens import TokenStore
    return len(TokenStore(cfg.secrets_dir, auth_mode=cfg.auth_mode).active_tokens()) == 0


# ---------------------------------------------------------------------------
# run_deploy()  â€" the core deployment pipeline
# ---------------------------------------------------------------------------

def run_deploy(cfg) -> None:
    """
    Execute all deployment steps for the given Config.

    Called by both wizard mode (cli._run_deploy_wizard) and batch mode
    (cli.dispatch with --batch flag).
    """
    from llama_deploy.config import AuthMode, DockerNetworkMode
    from llama_deploy.system import (
        require_root_reexec,
        detect_ubuntu,
        ensure_base_packages,
        ensure_unattended_upgrades,
        ensure_docker,
        ensure_docker_daemon_hardening,
        ensure_swap,
        ensure_firewall,
        resolve_hashed_proxy_ports,
    )
    from llama_deploy.tokens import TokenStore
    from llama_deploy.service import (
        write_models_ini,
        write_compose,
        write_auth_sidecar_script,
        docker_pull,
        docker_compose_up,
    )
    from llama_deploy.health import wait_health, curl_smoke_tests, sanity_checks, profile_smoke_checks
    from llama_deploy.log import log_line, redact, LOG_PATH

    require_root_reexec()
    detect_ubuntu()
    _ensure_tqdm(allow_install=True)

    if cfg.auto_optimize:
        cfg = _auto_optimize_cfg(cfg)
    else:
        tqdm.write("[AUTO] Host-spec auto optimization disabled by config.")

    if cfg.auth_mode == AuthMode.HASHED and cfg.docker_network_mode != DockerNetworkMode.HOST:
        llama_port, sidecar_port = resolve_hashed_proxy_ports(
            cfg.llama_internal_port,
            cfg.sidecar_port,
        )
        if llama_port != cfg.llama_internal_port or sidecar_port != cfg.sidecar_port:
            tqdm.write(
                "[AUTO] Hashed mode ports adjusted to avoid conflicts: "
                f"llama {cfg.llama_internal_port}->{llama_port}, "
                f"sidecar {cfg.sidecar_port}->{sidecar_port}"
            )
            log_line(
                "[AUTO] hashed ports adjusted: "
                f"llama {cfg.llama_internal_port}->{llama_port}, "
                f"sidecar {cfg.sidecar_port}->{sidecar_port}"
            )
            cfg = replace(cfg, llama_internal_port=llama_port, sidecar_port=sidecar_port)

    tqdm.write(f"[INFO] Logging to: {LOG_PATH}")
    log_line(f"=== START {dt.datetime.now(dt.timezone.utc).isoformat()}Z ===")
    log_line("Args: " + redact(" ".join(sys.argv)))

    # Ensure directories exist before any step runs
    for d in (cfg.models_dir, cfg.presets_dir, cfg.cache_dir, cfg.secrets_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Detect first-run before the token step runs (for post-deploy message)
    first_run = _is_new_token(cfg)

    # -----------------------------------------------------------------------
    # Phase 1: System preparation
    # -----------------------------------------------------------------------
    token_step = Step(
        label="Create API token",
        fn=lambda: _ensure_first_token(cfg),
    )

    system_steps: List[Step] = [
        Step("Install base packages",      ensure_base_packages),
        Step("Enable unattended upgrades", ensure_unattended_upgrades),
        Step("Install / enable Docker",    ensure_docker),
        Step("Docker daemon hardening",    ensure_docker_daemon_hardening),
        Step("Ensure swap",                lambda: ensure_swap(cfg.swap_gib)),
        Step(
            "Firewall (UFW) hardening",
            fn=lambda: ensure_firewall(cfg.network),
            skip_if=lambda: not cfg.network.configure_ufw,
        ),
        token_step,
    ]

    # -----------------------------------------------------------------------
    # Phase 2: Model resolution and download
    # -----------------------------------------------------------------------
    _unverified_models: list = []  # mutable cell — populated if trust_overridden

    def _resolve_llm():
        spec = _resolve_model_with_retry(cfg, cfg.llm, "LLM")
        if spec.trust_overridden:
            _unverified_models.append(spec.effective_alias)
        return spec

    def _resolve_emb():
        spec = _resolve_model_with_retry(cfg, cfg.emb, "Embedding")
        if spec.trust_overridden:
            _unverified_models.append(spec.effective_alias)
        return spec

    llm_step = Step(
        label="Resolve + download LLM GGUF",
        fn=_resolve_llm,
        skip_if=lambda: cfg.skip_download,
    )
    emb_step = Step(
        label="Resolve + download embedding GGUF",
        fn=_resolve_emb,
        skip_if=lambda: cfg.skip_download,
    )

    def _resolve_from_disk() -> None:
        """When --skip-download, fill step.result from whatever .gguf is on disk."""
        if not cfg.skip_download:
            return
        from llama_deploy.model import sha256_file
        from llama_deploy.log import die

        for step, spec in ((llm_step, cfg.llm), (emb_step, cfg.emb)):
            matches = [
                f for f in cfg.models_dir.glob("*.gguf")
                if any(p in f.name for p in spec.candidate_patterns)
            ]
            if not matches:
                die(
                    f"--skip-download requested but no matching GGUF found for "
                    f"{spec.hf_repo} in {cfg.models_dir}. "
                    f"Patterns tried: {spec.candidate_patterns}"
                )
            chosen = matches[0]
            sha = sha256_file(chosen)
            step.result = spec.with_resolved(
                filename=chosen.name, sha256=sha, size=chosen.stat().st_size
            )
            tqdm.write(f"[SKIP-DL] Using existing {chosen.name} (sha={sha[:12]}...)")

    # -----------------------------------------------------------------------
    # Phase 3: Service configuration and startup
    # -----------------------------------------------------------------------
    def _write_config() -> None:
        write_models_ini(cfg.preset_path, llm_step.result, emb_step.result, cfg.parallel, cfg.models_max)
        write_compose(cfg.compose_path, cfg)
        if cfg.auth_mode == AuthMode.HASHED:
            write_auth_sidecar_script(cfg.base_dir)

    config_step = Step("Write models.ini + docker-compose.yml", _write_config)
    pull_step   = Step("Pull Docker image",    lambda: docker_pull(cfg.image))
    up_step     = Step("Start Docker Compose", lambda: docker_compose_up(cfg.compose_path))

    # -----------------------------------------------------------------------
    # Phase 2b: Tailscale (vpn-only profile)
    # -----------------------------------------------------------------------
    from llama_deploy.config import AccessProfile

    tailscale_ip_holder: list = []  # mutable cell so the summary can read it later

    def _setup_tailscale() -> None:
        from llama_deploy.tailscale import tailscale_install, tailscale_up, tailscale_ip
        tailscale_install()
        tailscale_up(auth_key=cfg.tailscale_authkey)
        ip = tailscale_ip()
        tailscale_ip_holder.clear()
        tailscale_ip_holder.append(ip)
        tqdm.write(f"[TS] VPN endpoint ready: {ip}:{cfg.network.port}")

    tailscale_step = Step(
        label="Tailscale VPN setup",
        fn=_setup_tailscale,
        skip_if=lambda: cfg.network.access_profile != AccessProfile.VPN_ONLY,
    )

    # -----------------------------------------------------------------------
    # Phase 3b: NGINX — local auth proxy (hashed, no domain) OR TLS (domain set)
    # -----------------------------------------------------------------------
    # When hashed + domain: ensure_tls_for_domain handles NGINX (with auth_request).
    # When hashed + no domain: ensure_local_proxy installs NGINX without certbot.
    # Both steps are mutually exclusive via skip_if.

    def _setup_local_proxy() -> None:
        nonlocal cfg
        from llama_deploy.nginx import ensure_local_proxy
        selected_port = ensure_local_proxy(
            bind_host=cfg.network.bind_host,
            port=cfg.network.port,
            upstream_port=cfg.llama_internal_port,
            configure_ufw=cfg.network.configure_ufw,
            use_auth_sidecar=True,
            sidecar_port=cfg.sidecar_port,
        )
        if selected_port != cfg.network.port:
            old_port = cfg.network.port
            cfg = replace(cfg, network=replace(cfg.network, port=selected_port))
            tqdm.write(
                f"[AUTO] Local proxy port adjusted: "
                f"{cfg.network.bind_host}:{old_port} -> {cfg.network.bind_host}:{selected_port}"
            )
            log_line(
                f"[AUTO] local proxy port adjusted: "
                f"{cfg.network.bind_host}:{old_port} -> {cfg.network.bind_host}:{selected_port}"
            )

    local_proxy_step = Step(
        label="NGINX local auth proxy (hashed mode)",
        fn=_setup_local_proxy,
        skip_if=lambda: cfg.auth_mode != AuthMode.HASHED or cfg.use_tls,
    )

    def _setup_tls() -> None:
        from llama_deploy.nginx import ensure_tls_for_domain
        ensure_tls_for_domain(
            domain=cfg.domain,
            email=cfg.certbot_email,
            upstream_port=(
                cfg.llama_internal_port
                if cfg.auth_mode == AuthMode.HASHED
                else cfg.network.port
            ),
            configure_ufw=cfg.network.configure_ufw,
            use_auth_sidecar=cfg.auth_mode == AuthMode.HASHED,
            sidecar_port=cfg.sidecar_port,
        )

    tls_step = Step(
        label=f"NGINX + Let's Encrypt TLS ({cfg.domain})",
        fn=_setup_tls,
        skip_if=lambda: not cfg.use_tls,
    )

    # -----------------------------------------------------------------------
    # Phase 4: Validation
    # -----------------------------------------------------------------------
    # Health check against the loopback port (always reachable, regardless of TLS)
    def _loopback_url() -> str:
        return cfg.network.base_url

    if cfg.network.publish or cfg.auth_mode == AuthMode.HASHED:
        if cfg.auth_mode == AuthMode.HASHED:
            health_fn = lambda: wait_health(
                f"{_loopback_url()}/health",
                timeout_s=300,
                bearer_token=token_step.result.value,
            )
        else:
            health_fn = lambda: wait_health(f"{_loopback_url()}/health", timeout_s=300)

        health_step = Step(
            "Wait for /health",
            health_fn,
            skip_if=lambda: cfg.skip_health_check,
        )
        smoke_step = Step(
            "Smoke tests (OpenAI-compatible routes)",
            lambda: curl_smoke_tests(_loopback_url(), token_step.result.value, llm_step.result, emb_step.result),
            skip_if=lambda: cfg.skip_health_check,
        )
    else:
        def _internal_test() -> None:
            from llama_deploy.config import DockerNetworkMode
            from llama_deploy.log import sh
            sh("docker pull curlimages/curl:8.5.0", check=False)
            if cfg.docker_network_mode == DockerNetworkMode.BRIDGE:
                # In bridge mode, target the container's bridge IP.
                router_target = (
                    subprocess.check_output(
                        [
                            "docker",
                            "inspect",
                            "-f",
                            "{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                            "llama-router",
                        ],
                        text=True,
                    )
                    .strip()
                )
                if not router_target:
                    raise RuntimeError("Could not determine llama-router bridge IP for internal smoke test.")
                runner_network = "bridge"
            elif cfg.docker_network_mode == DockerNetworkMode.COMPOSE:
                # In compose mode, use the project network and service DNS name.
                runner_network = (
                    subprocess.check_output(
                        [
                            "docker",
                            "inspect",
                            "-f",
                            "{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{\"\\n\"}}{{end}}",
                            "llama-router",
                        ],
                        text=True,
                    )
                    .splitlines()[0]
                    .strip()
                )
                if not runner_network:
                    raise RuntimeError("Could not determine compose network for internal smoke test.")
                router_target = "llama-router"
            else:
                raise RuntimeError(
                    "Internal no-publish smoke test is unsupported for docker_network_mode="
                    f"{cfg.docker_network_mode.value}."
                )

            sh(
                f"docker run --rm --network {runner_network} curlimages/curl:8.5.0 "
                f"sh -lc \"curl -fsS http://{router_target}:8080/health && "
                f"curl -fsS http://{router_target}:8080/v1/models "
                f"-H 'Authorization: Bearer {token_step.result.value}' | head -c 800\""
            )

        health_step = Step("Internal network smoke test", _internal_test,
                           skip_if=lambda: cfg.skip_health_check)
        smoke_step  = Step("Sanity checks", lambda: sanity_checks(cfg),
                           skip_if=lambda: cfg.skip_health_check)

    sanity_step        = Step("Sanity checks (ports + logs)", lambda: sanity_checks(cfg))
    profile_check_step = Step("Profile smoke-checks",         lambda: profile_smoke_checks(cfg))

    service_steps = [
        config_step, pull_step, up_step,
        tailscale_step,
        local_proxy_step, tls_step,
        health_step, smoke_step, sanity_step, profile_check_step,
    ]

    all_steps: List[Step] = system_steps + [llm_step, emb_step] + service_steps

    # -----------------------------------------------------------------------
    # Execute
    # -----------------------------------------------------------------------
    try:
        with tqdm(total=len(all_steps), desc="Deploying llama.cpp", unit="step") as bar:
            run_steps(system_steps, bar)
            _resolve_from_disk()
            run_steps([llm_step, emb_step], bar)
            run_steps(service_steps, bar)

        # -------------------------------------------------------------------
        # Post-deploy summary
        # -------------------------------------------------------------------
        _print_summary(cfg, token_step.result.value, first_run, _unverified_models)
        if tailscale_ip_holder:
            ts_ip = tailscale_ip_holder[0]
            tqdm.write(f"  VPN endpoint : {ts_ip}:{cfg.network.port}  (Tailscale)")
    finally:
        runtime = token_step.result
        if isinstance(runtime, TokenRuntime) and runtime.temporary_id:
            store = TokenStore(cfg.secrets_dir, auth_mode=cfg.auth_mode)
            try:
                store.revoke_token(runtime.temporary_id)
                tqdm.write(f"[CLEANUP] Revoked temporary smoke-test token ({runtime.temporary_id}).")
            except Exception as e:
                tqdm.write(
                    f"[WARN] Failed to revoke temporary smoke-test token "
                    f"({runtime.temporary_id}): {e}"
                )
                log_line(
                    f"[WARN] Failed to revoke temporary smoke-test token "
                    f"({runtime.temporary_id}): {e}"
                )

        log_line(f"=== END {dt.datetime.now(dt.timezone.utc).isoformat()}Z ===")


def _print_summary(cfg, api_token: str, first_run: bool, unverified_models: list) -> None:
    from llama_deploy.config import AuthMode

    print()
    print("â•”" + "â•" * 52 + "â•—")
    print("â•‘" + "  Deployment complete".ljust(52) + "â•‘")
    print("â•š" + "â•" * 52 + "â•")
    print()

    if first_run:
        print(f'Your API token "{cfg.api_token_name}":')
        print()
        print(f"  {api_token}")
        print()
        if cfg.auth_mode == AuthMode.HASHED:
            print("  âš   This is shown ONCE. The plaintext is NOT stored on disk.")
            print(f"     Hash stored in: {cfg.secrets_dir}/tokens.json  (mode 600)")
        else:
            print("  âš   This is shown ONCE. Stored in:")
            print(f"     {cfg.secrets_dir}/tokens.json  (mode 600)")
    else:
        print("  Token already existed â€” value unchanged.")
        print(f"  See {cfg.secrets_dir}/tokens.json for token details.")

    print()
    print(f"  Endpoint : {cfg.public_base_url}")
    print(f"  Auth mode: {cfg.auth_mode.value}")
    print(f"  Base dir : {cfg.base_dir}")
    print()
    print("Token management:")
    print("  python -m llama_deploy tokens list")
    print("  python -m llama_deploy tokens create --name <name>")
    print("  python -m llama_deploy tokens revoke <id>")
    print()
    base_url  = cfg.public_base_url.rstrip("/")
    llm_model = cfg.llm.effective_alias
    emb_model = cfg.emb.effective_alias
    # Use the real token in examples on first run so they are copy-pasteable.
    # On re-deploy the smoke-test token is already revoked; show a placeholder.
    tok = api_token if first_run else "<your-token>"

    print("Quick-start  (copy-paste ready)")
    print()
    print("  # List available models")
    print(f"  curl {base_url}/v1/models \\")
    print(f"    -H 'Authorization: Bearer {tok}'")
    print()
    print("  # Chat completion")
    print(f"  curl {base_url}/v1/chat/completions \\")
    print(f"    -H 'Authorization: Bearer {tok}' \\")
    print( "    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"model\":\"{llm_model}\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello!\"}}]}}'")
    print()
    print("  # Embeddings")
    print(f"  curl {base_url}/v1/embeddings \\")
    print(f"    -H 'Authorization: Bearer {tok}' \\")
    print( "    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"model\":\"{emb_model}\",\"input\":\"text to embed\"}}'")
    print()
    print("  # Python (pip install openai)")
    print( "  from openai import OpenAI")
    print(f"  client = OpenAI(base_url=\"{base_url}/v1\", api_key=\"{tok}\")")
    print(f"  r = client.chat.completions.create(model=\"{llm_model}\",")
    print( "        messages=[{\"role\": \"user\", \"content\": \"Hello!\"}])")
    print( "  print(r.choices[0].message.content)")
    print()

    if unverified_models:
        print("*** SECURITY WARNING " + "*" * 33)
        print("The following models were deployed WITHOUT verified checksums:")
        for alias in unverified_models:
            print(f"  - {alias}")
        print("Re-deploy with verified models before using in production.")
        print("*" * 54)
        print()


# ---------------------------------------------------------------------------
# Legacy entry point (direct invocation without subcommands)
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Called by __main__.py when invoked as `python -m llama_deploy`.
    Delegates to cli.dispatch() which handles TTY detection and subcommands.
    """
    from llama_deploy.cli import dispatch
    dispatch()


if __name__ == "__main__":
    main()
