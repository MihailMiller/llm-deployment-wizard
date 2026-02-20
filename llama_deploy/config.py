"""
Immutable configuration dataclasses.

All cross-parameter validation lives in __post_init__ so it fires at
construction time regardless of where Config is built (CLI, tests, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class BackendKind(str, Enum):
    CPU    = "cpu"
    CUDA   = "cuda"
    ROCM   = "rocm"
    VULKAN = "vulkan"
    INTEL  = "intel"

    def docker_image(self) -> str:
        base = "ghcr.io/ggml-org/llama.cpp:server"
        return base if self == BackendKind.CPU else f"{base}-{self.value}"


class AuthMode(str, Enum):
    PLAINTEXT = "plaintext"  # llama-server --api-key-file (tokens stored in plaintext)
    HASHED    = "hashed"     # NGINX auth_request → sidecar (only SHA-256 hashes stored)


class AccessProfile(str, Enum):
    """
    Named access profiles that encode the intended exposure of the API.

    Each profile determines which source addresses may reach the service and
    how the firewall should be configured.  The detailed UFW rule set is
    applied in system.ensure_firewall(); service.write_compose() uses the
    profile to enforce correct bind addresses.

      LOCALHOST    — loopback only; no external access.
      HOME_PRIVATE — LAN/home network access restricted via UFW to lan_cidr.
      VPN_ONLY     — reachable only via VPN interface (e.g. Tailscale).
      PUBLIC       — internet-facing: either NGINX+TLS or a fully open interface.
    """

    LOCALHOST    = "localhost"
    HOME_PRIVATE = "home-private"
    VPN_ONLY     = "vpn-only"
    PUBLIC       = "public"


class DockerNetworkMode(str, Enum):
    """
    Docker networking strategy for the service containers.

      BRIDGE : use Docker's built-in "bridge" network (no per-project network).
      COMPOSE: use Docker Compose managed project network (<project>_default).
      HOST   : share host network namespace (advanced).
    """

    BRIDGE = "bridge"
    COMPOSE = "compose"
    HOST = "host"


@dataclass(frozen=True)
class ModelSpec:
    """
    Specification for one model to be downloaded from HuggingFace and served.

    candidate_patterns: ordered list of GGUF filename substrings to try in
    priority order. The first pattern that matches a file present in the repo
    is used. Replaces the hardcoded llm_candidates / emb_candidates lists that
    previously lived inside main() and were decoupled from --llm-repo.

    effective_alias: the name used as the INI section header AND the model name
    advertised to clients via /v1/models. Derived automatically from hf_repo if
    alias is not set (strips trailing -GGUF suffix). This fixes the bug where
    write_models_ini and curl_smoke_tests hardcoded "Qwen/Qwen3-8B" even when
    --llm-repo was overridden.
    """

    hf_repo: str
    candidate_patterns: List[str]
    ctx_len: int
    alias: Optional[str] = None
    is_embedding: bool = False

    # Populated by model.resolve_model(); not set from CLI
    resolved_filename:  Optional[str]  = field(default=None,  compare=False)
    resolved_sha256:    Optional[str]  = field(default=None,  compare=False)
    resolved_size:      Optional[int]  = field(default=None,  compare=False)
    trust_overridden:   bool           = field(default=False, compare=False)

    def __post_init__(self) -> None:
        if not self.hf_repo:
            raise ValueError("hf_repo must not be empty")
        if not self.candidate_patterns:
            raise ValueError("candidate_patterns must contain at least one entry")
        if self.ctx_len < 128:
            raise ValueError(f"ctx_len={self.ctx_len} is implausibly small")

    @property
    def effective_alias(self) -> str:
        """
        Returns alias if explicitly set; otherwise strips a trailing -GGUF suffix
        from hf_repo.

        Examples:
            "Qwen/Qwen3-8B-GGUF"             -> "Qwen/Qwen3-8B"
            "bartowski/Mistral-7B-Instruct-GGUF" -> "bartowski/Mistral-7B-Instruct"
            "org/model" (no -GGUF)            -> "org/model"
        """
        if self.alias:
            return self.alias
        return re.sub(r"-GGUF$", "", self.hf_repo, flags=re.IGNORECASE)

    def with_resolved(
        self,
        filename: str,
        sha256: str,
        size: int,
        *,
        trust_overridden: bool = False,
    ) -> "ModelSpec":
        """Return a new frozen ModelSpec with resolved download metadata."""
        return ModelSpec(
            hf_repo=self.hf_repo,
            candidate_patterns=self.candidate_patterns,
            ctx_len=self.ctx_len,
            alias=self.alias,
            is_embedding=self.is_embedding,
            resolved_filename=filename,
            resolved_sha256=sha256,
            resolved_size=size,
            trust_overridden=trust_overridden,
        )


@dataclass(frozen=True)
class NetworkConfig:
    """
    Replaces the --public / --bind / --no-publish / --allow-public-port tangle.

    All invariants are enforced at construction time via __post_init__ so that
    invalid combinations are caught immediately at argument-parse time rather
    than partway through main() execution.

    access_profile is the *intent* of the deployment (who should reach it).
    It drives firewall rule generation in system.ensure_firewall() and bind
    address enforcement in service.write_compose().  The low-level bind_host /
    open_firewall fields remain for fine-grained batch-mode control.

    lan_cidr is required when access_profile=home-private and specifies the
    source network for UFW ALLOW rules (e.g. "192.168.1.0/24").
    """

    bind_host: str = "127.0.0.1"
    port: int = 8080
    publish: bool = True          # maps Docker port to host
    open_firewall: bool = False   # open the port in UFW (replaces --allow-public-port)
    configure_ufw: bool = True    # whether to touch UFW at all (replaces --no-ufw)
    access_profile: AccessProfile = AccessProfile.LOCALHOST
    lan_cidr: Optional[str] = None  # required for home-private profile

    def __post_init__(self) -> None:
        if self.bind_host == "0.0.0.0" and not self.publish:
            raise ValueError(
                "--bind 0.0.0.0 cannot be combined with --no-publish: "
                "a publicly-bound service that is not port-mapped is unreachable from the host."
            )
        if self.open_firewall and self.bind_host != "0.0.0.0":
            raise ValueError(
                "--open-firewall only makes sense with --bind 0.0.0.0; "
                "there is nothing to open in UFW for a loopback-only service."
            )
        # Profile-consistency invariants
        if self.access_profile == AccessProfile.HOME_PRIVATE and not self.lan_cidr:
            raise ValueError(
                "access_profile=home-private requires --lan-cidr "
                "(e.g. 192.168.1.0/24) to restrict UFW to your LAN."
            )
        if self.access_profile == AccessProfile.LOCALHOST and self.bind_host == "0.0.0.0":
            raise ValueError(
                "access_profile=localhost cannot bind to 0.0.0.0; "
                "use --bind 127.0.0.1 (the default) for loopback-only access."
            )
        if self.access_profile == AccessProfile.VPN_ONLY and self.open_firewall:
            raise ValueError(
                "access_profile=vpn-only must not open the firewall to all; "
                "remove --open-firewall."
            )
        if self.lan_cidr is not None:
            # Basic CIDR format check (does not validate prefix length range)
            import re as _re
            if not _re.match(
                r"^\d{1,3}(\.\d{1,3}){3}/\d{1,2}$",
                self.lan_cidr,
            ):
                raise ValueError(
                    f"lan_cidr={self.lan_cidr!r} is not a valid CIDR "
                    "(expected format: A.B.C.D/prefix, e.g. 192.168.1.0/24)."
                )
        if self.access_profile == AccessProfile.HOME_PRIVATE and self.open_firewall:
            raise ValueError(
                "access_profile=home-private must not use --open-firewall; "
                "UFW access is restricted to lan_cidr automatically."
            )

    @property
    def is_public(self) -> bool:
        return self.bind_host == "0.0.0.0"

    @property
    def base_url(self) -> str:
        return f"http://{self.bind_host}:{self.port}"

    @property
    def profile_label(self) -> str:
        """Human-readable summary of the access profile for logs and review."""
        labels = {
            AccessProfile.LOCALHOST:    "localhost (loopback only)",
            AccessProfile.HOME_PRIVATE: f"home-private (LAN {self.lan_cidr})",
            AccessProfile.VPN_ONLY:     "vpn-only (Tailscale / VPN)",
            AccessProfile.PUBLIC:       "public (internet-facing)",
        }
        return labels.get(self.access_profile, self.access_profile.value)


@dataclass(frozen=True)
class Config:
    """
    Single immutable source of truth for all deployment parameters.

    Constructed exactly once by cli.build_config(); never mutated after that.
    Every downstream function receives the Config (or a relevant sub-object)
    rather than individual scattered arguments.
    """

    base_dir: Path
    backend: BackendKind
    network: NetworkConfig
    swap_gib: int
    models_max: int
    parallel: int
    api_token: Optional[str]      # None = auto-generate on first run
    api_token_name: str           # label for the first token (used by wizard + token store)
    hf_token: Optional[str]       # None = unauthenticated HuggingFace access
    skip_download: bool           # skip HF queries if model files already verified
    llm: ModelSpec
    emb: ModelSpec
    auto_optimize: bool = True            # tune model/runtime defaults from detected host specs
    allow_unverified_downloads: bool = False  # permit trusting downloads when upstream SHA cannot be proven
    domain: Optional[str] = None          # public domain for NGINX + Let's Encrypt TLS
    certbot_email: Optional[str] = None   # email for Let's Encrypt renewal notices
    auth_mode: AuthMode = AuthMode.PLAINTEXT  # token storage strategy
    tailscale_authkey: Optional[str] = None   # auth key for non-interactive tailscale up (vpn-only profile)
    docker_network_mode: DockerNetworkMode = DockerNetworkMode.BRIDGE  # Docker network mode for containers

    def __post_init__(self) -> None:
        if self.docker_network_mode == DockerNetworkMode.HOST:
            if not self.network.publish:
                raise ValueError(
                    "docker_network_mode=host cannot be combined with --no-publish."
                )
            if self.auth_mode == AuthMode.HASHED:
                raise ValueError(
                    "docker_network_mode=host is not supported with --auth-mode hashed."
                )

    # Internal ports used in hashed mode (not user-configurable)
    @property
    def llama_internal_port(self) -> int:
        """Port llama-server is published on when NGINX fronts it (hashed mode)."""
        return 8081

    @property
    def sidecar_port(self) -> int:
        """Port the auth sidecar listens on (always loopback)."""
        return 9000

    @property
    def use_tls(self) -> bool:
        """True when NGINX + certbot should be configured."""
        return bool(self.domain)

    @property
    def use_nginx(self) -> bool:
        """True when NGINX is needed (TLS or hashed auth mode)."""
        return self.use_tls or self.auth_mode == AuthMode.HASHED

    @property
    def public_base_url(self) -> str:
        """Human-readable endpoint URL shown in the post-deploy summary."""
        if self.use_tls:
            return f"https://{self.domain}/v1"
        if self.network.publish or self.auth_mode == AuthMode.HASHED:
            return f"{self.network.base_url}/v1"
        return "(Docker network only)"

    # Derived paths (computed properties for convenience)
    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"

    @property
    def presets_dir(self) -> Path:
        return self.base_dir / "presets"

    @property
    def cache_dir(self) -> Path:
        return self.base_dir / "cache"

    @property
    def secrets_dir(self) -> Path:
        return self.base_dir / "secrets"

    @property
    def compose_path(self) -> Path:
        return self.base_dir / "docker-compose.yml"

    @property
    def token_file(self) -> Path:
        return self.secrets_dir / "api_keys"

    @property
    def preset_path(self) -> Path:
        return self.presets_dir / "models.ini"

    @property
    def image(self) -> str:
        return self.backend.docker_image()
