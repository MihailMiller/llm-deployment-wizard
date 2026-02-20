# LLM Deployment Wizard: Beginner-Friendly Private LLM API / Open WebUI Deployment

Deploy **llama.cpp** (`llama-server`) as a private, OpenAI-compatible REST API on Ubuntu with one command and a guided wizard.

If you are new to self-hosting, this project gives you a safe default path:
- Private by default (`127.0.0.1`, firewall-aware, no accidental public exposure)
- Guided setup (wizard chooses sane defaults and explains networking choices)
- Production basics included (Docker, NGINX/TLS options, token lifecycle, logs)

```
POST /v1/chat/completions   model="Qwen/Qwen3-8B"
POST /v1/embeddings         model="Qwen/Qwen3-Embedding-0.6B"
GET  /v1/models
```

The server runs in Docker, uses hardened container settings, and is never exposed to the internet unless you explicitly choose it.

## Why use this? (Value proposition)

Without this tool, a typical deployment means wiring together Docker, model downloads, token handling, firewall rules, reverse proxy config, TLS certificates, and health checks yourself.

`llama_deploy` combines that into one flow so you can:
- Get from zero to a working `/v1/chat/completions` endpoint quickly
- Keep a safer default security posture as a beginner
- Re-deploy safely without rebuilding everything by hand
- Rotate/revoke named API tokens without manual file surgery

## Start here (first deployment)

```bash
git clone <repo-url> /opt/llama_deploy
cd /opt/llama_deploy
sudo python3 -m llama_deploy
```

When deployment finishes, copy the token shown once and test:

```bash
curl http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer <token>"
```

If this returns JSON model metadata, your local API is up.

---

## Access profiles (choose exposure level)

Access profiles let you deploy for different use cases without needing to know the underlying UFW/bind details:

| Profile | Who can reach the API | No port-forwarding needed |
|---|---|---|
| `localhost` | This machine only (default) | yes |
| `home-private` | Your LAN (UFW-restricted to a CIDR) | yes |
| `vpn-only` | Tailscale network peers | yes |
| `public` | Internet (NGINX+TLS or direct) | requires open port |

---

## What this does for you

| Step | What happens |
|---|---|
| System prep | Updates apt, installs Docker CE + Compose plugin, creates swap if needed, configures UFW |
| Model download | Fetches GGUF files from HuggingFace, verifies SHA-256, resumes interrupted downloads |
| Configuration | Writes `models.ini` (llama-server router preset) and `docker-compose.yml` |
| Service start | Pulls the llama.cpp Docker image and starts the container |
| Validation | Waits for `/health`, runs three smoke tests (models list, embeddings, chat) |
| Token | Generates a named API token, stores it in `secrets/tokens.json`, shows it once |

Everything is logged to `/var/log/llamacpp_deploy.log`.

---

## Prerequisites

### Host

| Requirement | Notes |
|---|---|
| Ubuntu 22.04 or 24.04 | Tested on both. Must have `systemd`. |
| `sudo` or root access | The script re-executes itself via `sudo` if needed. |
| Python 3.8+ | Already present on Ubuntu. `tqdm` is installed automatically if missing. |
| 15–25 GB free disk | Depends on the models you choose (default: ~10 GB total). |
| 4 GB RAM minimum | 12+ GB recommended for the default Qwen3-8B model. |
| Internet access | To download the Docker image and GGUF files from HuggingFace. |

### Optional: GPU

| Backend | Additional requirement |
|---|---|
| `cuda` | NVIDIA GPU with CUDA 12+, `nvidia-docker2` / NVIDIA Container Toolkit installed |
| `rocm` | AMD GPU with ROCm drivers installed |
| `vulkan` | Vulkan-capable GPU and Vulkan drivers |
| `intel` | Intel GPU with OpenVINO / SYCL drivers |

If you choose CPU (the default) no GPU drivers are needed.

### Optional: Private HuggingFace models

If you use a private or gated model repository (e.g. Meta-Llama), set the `HF_TOKEN` environment variable or pass `--hf-token`:

```bash
export HF_TOKEN=hf_your_token_here
```

The default models (Qwen3-8B, Qwen3-Embedding-0.6B) are public and do not require a token.
On low-memory hosts, deployment auto-detects RAM/CPU and can downshift to a smaller LLM profile automatically to reduce OOM risk.
The wizard includes a larger preset catalog (Qwen, Llama, Mistral, Gemma, Phi, BGE) and can prompt you to switch models and retry if a download fails.

---

## Installation

Clone the repository to your Ubuntu server:

```bash
git clone <repo-url> /opt/llama_deploy
cd /opt/llama_deploy
```

No `pip install` is required — the package runs directly. `tqdm` is the only non-stdlib dependency and is installed automatically on first run.

---

## Quick start: interactive wizard

Run without arguments on any terminal that has a TTY:

```bash
sudo python3 -m llama_deploy
```

You will be guided through six steps:

```
╔════════════════════════════════════════════════════╗
║         llama.cpp Deployment Wizard                ║
║   Configure your OpenAI-compatible AI service      ║
╚════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════╗
║  Step 1/6 · Backend                               ║
╚════════════════════════════════════════════════════╝

What hardware acceleration do you have?
  [1] CPU only        works everywhere — no GPU required
  [2] NVIDIA GPU      CUDA — requires nvidia-docker
  [3] AMD GPU         ROCm — requires ROCm drivers
  [4] Vulkan GPU      cross-vendor GPU acceleration
  [5] Intel GPU       SYCL / OpenVINO

Choice [1]:

╔════════════════════════════════════════════════════╗
║  Step 3/6 · Network                               ║
╚════════════════════════════════════════════════════╝

  Choose who should be able to reach the API:

  [1] Localhost only        127.0.0.1 — local apps on this machine only
  [2] Home / LAN network    LAN CIDR — private network access, UFW-restricted
  [3] VPN only (Tailscale)  VPN interface — reachable only via Tailscale or other VPN
  [4] Internet / Public     NGINX + Let's Encrypt TLS — or direct 0.0.0.0
  [5] Docker internal       no host port — only reachable from other containers

Choice [1]:
```

After step 6, a review summary is shown and you confirm before anything is written to disk.

At the end of deployment your API token is printed **once**:

```
╔════════════════════════════════════════════════════╗
║  Deployment complete                               ║
╚════════════════════════════════════════════════════╝

Your API token "my-app":

  sk-AbCdEfGhIjKlMnOpQrStUvWxYz...

  ⚠  This is shown ONCE. Stored in:
     /opt/llama/secrets/tokens.json  (mode 600)

Token management:
  python -m llama_deploy tokens list
  python -m llama_deploy tokens create --name "another-app"
  python -m llama_deploy tokens revoke tk_4f9a2b1c3d8e
```

---

## Non-interactive (CI / scripts)

Pass `--batch` to skip the wizard and use flags directly:

```bash
sudo python3 -m llama_deploy deploy --batch \
  --base-dir /opt/llama \
  --backend cpu \
  --bind 127.0.0.1 \
  --port 8080 \
  --token-name "my-server" \
  --swap-gib 8
```

With hashed token storage:

```bash
sudo python3 -m llama_deploy deploy --batch \
  --bind 127.0.0.1 \
  --port 8080 \
  --auth-mode hashed \
  --token-name "my-server"
```

See all available flags:

```bash
python3 -m llama_deploy deploy --batch --help
```

<details>
<summary>All flags</summary>

```
Paths:
  --base-dir DIR        Base directory for models, config, cache, secrets.
                        (default: /opt/llama)

Backend:
  --backend {cpu,cuda,rocm,vulkan,intel}
                        llama.cpp Docker image variant. (default: cpu)

Network:
  --profile PROFILE     Access profile: localhost (default), home-private,
                        vpn-only, or public. Sets safe bind/firewall defaults.
  --lan-cidr CIDR       LAN source CIDR for home-private (e.g. 192.168.1.0/24).
                        Required when --profile=home-private.
  --bind HOST           Override bind address (inferred from --profile when
                        omitted). Use 0.0.0.0 for public. (default: 127.0.0.1)
  --port PORT           Host port to publish. (default: 8080)
  --no-publish          Do NOT publish any host port (Docker-network-only).
  --open-firewall       Open --port in UFW. Requires --bind 0.0.0.0.
  --skip-ufw            Do not configure UFW at all.
  --docker-network-mode MODE
                        Container network mode: bridge (default), compose, or host.
                        host is advanced and not supported with --auth-mode hashed.

System:
  --swap-gib GIB        Swap file size if none exists. (default: 8)
  --no-auto-optimize    Disable host-spec auto tuning (model/ctx/memory knobs).
  --tailscale-authkey KEY
                        Auth key for non-interactive tailscale up.
                        Only used with --profile=vpn-only.
                        Falls back to TAILSCALE_AUTHKEY env var.

Authentication:
  --token TOKEN         API token value. Randomly generated if omitted.
  --token-name NAME     Label for the first API token. (default: default)
  --auth-mode MODE      Token storage: plaintext (default) or hashed.
                        hashed stores only SHA-256; requires NGINX + sidecar.
  --hf-token TOKEN      HuggingFace access token (or use HF_TOKEN env var).

Models:
  --llm-repo REPO       HuggingFace repo for the LLM GGUF.
                        (default: Qwen/Qwen3-8B-GGUF)
  --llm-candidates PAT  Comma-separated filename patterns for LLM GGUF.
                        (default: Q4_K_M,Q5_K_M,Q4_0,Q3_K_M)
  --emb-repo REPO       HuggingFace repo for the embedding GGUF.
                        (default: Qwen/Qwen3-Embedding-0.6B-GGUF)
  --emb-candidates PAT  Comma-separated filename patterns for embedding GGUF.
                        (default: Q8_0,F16,Q6_K,Q4_K_M)
  --skip-download       Skip HF queries/downloads if files are already on disk.
  --allow-unverified-downloads
                        Continue on SHA mismatch/missing checksum without prompt.

HTTPS / TLS (NGINX + Let's Encrypt):
  --domain DOMAIN       Public domain name (e.g. api.example.com).
                        Installs NGINX as a TLS-terminating reverse proxy and
                        runs certbot to obtain a Let's Encrypt certificate.
                        The Docker port stays on loopback; NGINX faces the internet.
                        Use a bare hostname only (no http://, path, or port).
                        DNS A record for DOMAIN must already point to this host.
  --certbot-email EMAIL Email for Let's Encrypt renewal notices.
                        Required when --domain is set.

Server tuning:
  --models-max N        Max simultaneously loaded models. (default: 2)
  --ctx-llm TOKENS      Context window for the LLM. (default: 3072)
  --ctx-emb TOKENS      Context window for the embedding model. (default: 2048)
  --parallel N          Router parallel slots. (default: 1)
```

</details>

---

## Token management

Tokens are stored in `<base-dir>/secrets/tokens.json` (mode `0600`, root-only).
The flat `api_keys` file that llama-server reads is **automatically regenerated** every time you create or revoke a token.

### List all tokens

```bash
python3 -m llama_deploy tokens list
```

```
API Tokens  (/opt/llama/secrets/tokens.json)
────────────────────────────────────────────────────────────
ID                    Name                Status    Created
────────────────────────────────────────────────────────────
tk_4f9a2b1c3d8e      my-app              active    2026-02-20
tk_7e1d5a9b2c4f      my-laptop           active    2026-02-21
tk_1a3c5e7d9b2f      old-client          REVOKED   2026-02-19
────────────────────────────────────────────────────────────
2 active, 1 revoked
```

### Create a new token

```bash
python3 -m llama_deploy tokens create --name "my-new-app"
```

The token value is printed **once** and never shown again. In plaintext mode you can later reveal it with `tokens show`; in hashed mode this is impossible by design.

### Revoke a token

```bash
python3 -m llama_deploy tokens revoke tk_7e1d5a9b2c4f
```

Revocation behavior depends on auth mode:

- `plaintext`: `api_keys` is rewritten. Restart `llama` to reload keys.
- `hashed`: `token_hashes.json` is rewritten. Revocation is effective immediately.

### Show a token value again

```bash
python3 -m llama_deploy tokens show tk_4f9a2b1c3d8e
```

Requires interactive confirmation. In hashed mode, `tokens show` returns an error because plaintext token values are never stored on disk.

---

## Using the API

Replace `<token>` with the value printed at the end of deployment. In plaintext mode you can retrieve it from `tokens.json`; in hashed mode it is never stored.

### Test the connection

```bash
curl http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer <token>"
```

### Chat completion

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 200
  }'
```

### Embeddings

```bash
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": ["Hello world", "Another sentence"]
  }'
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="<token>",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

---

## Using a different model

Point `--llm-repo` at any GGUF repository on HuggingFace and supply matching filename patterns.
The model name advertised on `/v1/models` and used in requests is derived automatically from the repo name (trailing `-GGUF` is stripped).

```bash
sudo python3 -m llama_deploy deploy --batch \
  --llm-repo bartowski/Mistral-7B-Instruct-v0.3-GGUF \
  --llm-candidates "Q4_K_M,Q5_K_M" \
  --emb-repo Qwen/Qwen3-Embedding-0.6B-GGUF
```

Chat requests would then use `"model": "bartowski/Mistral-7B-Instruct-v0.3"`.

---

## Home / LAN network access (no port-forwarding)

Use the `home-private` profile to make the API reachable anywhere on your home LAN without exposing it to the internet. UFW restricts inbound traffic to your LAN CIDR — the Docker container itself stays bound to loopback.

```
[LAN clients: 192.168.1.x] ──HTTP──► UFW allows ──► 127.0.0.1:8080 (Docker)
[WAN / internet]            ──HTTP──► UFW blocks
```

### Via the wizard

Select **[2] Home / LAN network** in Step 3/6 · Network / Access Profile:

```
  [2] Home / LAN network   LAN CIDR — private network access, UFW-restricted

  LAN CIDR (e.g. 192.168.1.0/24): 192.168.1.0/24
  Port [8080]:
```

### Via batch mode

```bash
sudo python3 -m llama_deploy deploy --batch \
  --profile home-private \
  --lan-cidr 192.168.1.0/24 \
  --port 8080
```

### Connecting from another LAN device

Use the server's LAN IP (e.g. `192.168.1.10`):

```bash
curl http://192.168.1.10:8080/v1/models \
  -H "Authorization: Bearer <token>"
```

Python OpenAI SDK:

```python
client = OpenAI(base_url="http://192.168.1.10:8080/v1", api_key="<token>")
```

### Security notes

- Docker port is bound to `127.0.0.1` — Docker itself does not expose the port to `0.0.0.0`. UFW handles LAN access.
- LAN CIDR is enforced by UFW: `ufw allow from 192.168.1.0/24 to any port 8080`.
- No TLS. Consider hashed token mode (`--auth-mode hashed`) if your LAN includes untrusted devices.
- To verify: `ufw status numbered` — you should see a rule allowing your CIDR and a deny for `8080/tcp` from all else.

---

## VPN-only access via Tailscale (no port-forwarding, no LAN exposure)

The `vpn-only` profile keeps the API on loopback and uses Tailscale to create a private WireGuard mesh. Only Tailscale peers can reach it — no firewall holes in UFW, no open LAN port.

```
[Tailscale peer: 100.x.x.x] ──WireGuard──► Tailscale IP on this host
                                               ──► 127.0.0.1:8080 (Docker)
[Internet / LAN (no Tailscale)] ──► blocked (port never opened)
```

### Prerequisites

- A free [Tailscale account](https://tailscale.com/) (sign in with Google/GitHub/email).
- The devices that will call the API must also be in your Tailscale network.
- Optional: a Tailscale auth key for non-interactive setup ([generate one here](https://login.tailscale.com/admin/settings/keys)).

### Via the wizard

Select **[3] VPN only (Tailscale)** in Step 3/6 · Network / Access Profile. In Step 6 (System) you will be prompted for an optional auth key.

### Via batch mode

```bash
# With pre-generated auth key (fully automated)
sudo python3 -m llama_deploy deploy --batch \
  --profile vpn-only \
  --tailscale-authkey tskey-auth-xxxxxxxxxxxx-xxxxxxxxxxxxxxxx \
  --port 8080

# Without auth key (tailscale up will print an interactive login URL)
sudo python3 -m llama_deploy deploy --batch \
  --profile vpn-only \
  --port 8080
```

### What happens during deployment

1. Tailscale is installed via the official install script if not already present.
2. `tailscale up --accept-routes` is run (with auth key if provided).
3. The Tailscale IP (`100.x.x.x`) is shown in the deployment summary.
4. UFW is **not** asked to open port 8080 — Tailscale's kernel WireGuard interface handles routing.

### Connecting from another Tailscale device

Use the server's Tailscale IP from the summary (e.g. `100.64.0.1`):

```bash
curl http://100.64.0.1:8080/v1/models \
  -H "Authorization: Bearer <token>"
```

Python OpenAI SDK:

```python
client = OpenAI(base_url="http://100.64.0.1:8080/v1", api_key="<token>")
```

### Security notes

- `ss -lntp` will show port 8080 bound to `127.0.0.1` only — no LAN/WAN exposure.
- `ufw status` will show no rule for port 8080 — Tailscale's WireGuard tunnel is outside UFW scope.
- To check VPN health: `tailscale status`.
- Traffic between Tailscale nodes is end-to-end encrypted via WireGuard.

---

## Exposing publicly with HTTPS (NGINX + Let's Encrypt)

When you want the API reachable over the internet, the recommended path is NGINX as a TLS-terminating reverse proxy with a free Let's Encrypt certificate. The Docker container stays bound to loopback — it is never directly exposed.

```
Internet ──HTTPS 443──► NGINX ──HTTP──► 127.0.0.1:8080 (Docker)
```

### Prerequisites

| Requirement | Notes |
|---|---|
| A domain name | DNS A record must already point to this server's IP before you run the deploy command. |
| Ports 80 and 443 open | certbot needs port 80 for the initial HTTP-01 challenge. Port 443 serves the API. |
| No existing NGINX config on port 443 | The installer writes its own server block and removes the NGINX default site. |

### Via the wizard

Select **[4] Internet / Public** → **[1] HTTPS + domain** in Step 3/6 · Network / Access Profile:

```
  [4] Internet / Public     NGINX + Let's Encrypt TLS — or direct 0.0.0.0

  [1] HTTPS + domain   NGINX + Let's Encrypt — recommended
  [2] All interfaces   0.0.0.0 plain HTTP — no TLS

Choice [1]:

  NGINX will listen on :80 and :443 and proxy to the loopback port.
  Make sure your domain's DNS A record already points to this server.

  Domain name (e.g. api.example.com): api.myserver.com
  Email for Let's Encrypt renewal notices: me@example.com
  Port [8080]:
```

### Via batch mode

```bash
sudo python3 -m llama_deploy deploy --batch \
  --domain api.myserver.com \
  --certbot-email me@example.com \
  --bind 127.0.0.1 \
  --port 8080
```

`--bind 127.0.0.1` is the default and is required when `--domain` is set (NGINX faces the internet; Docker stays on loopback).

### What the deploy step does

1. Installs `nginx` and `python3-certbot-nginx` via apt.
2. Writes `/etc/nginx/sites-available/api.myserver.com` — a reverse-proxy server block with streaming-safe settings (`proxy_buffering off`, `proxy_read_timeout 300s`, `client_max_body_size 32m`).
3. Enables the site and reloads NGINX.
4. Runs `certbot --nginx` — obtains a certificate, rewrites the server block for HTTPS, and sets up an HTTP → HTTPS redirect automatically.
5. Enables the `certbot.timer` systemd unit for automatic renewal (checks twice daily, renews when < 30 days remain).
6. Opens ports 80 and 443 in UFW (port 8080 is not opened).

### After deployment

Your endpoint is:

```
https://api.myserver.com/v1
```

Test it:

```bash
curl https://api.myserver.com/v1/models \
  -H "Authorization: Bearer <token>"
```

Python OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.myserver.com/v1",
    api_key="<token>",
)
```

### Certificate renewal

Renewal is handled automatically by systemd. To check status:

```bash
systemctl status certbot.timer
certbot certificates
```

To force a manual renewal test:

```bash
certbot renew --dry-run
```

---

## File layout

After deployment, `<base-dir>` (default `/opt/llama`) contains:

**Plaintext auth mode (default):**

```
/opt/llama/
├── models/
│   ├── Qwen3-8B-Q4_K_M.gguf
│   └── Qwen3-Embedding-0.6B-Q8_0.gguf
├── presets/
│   └── models.ini
├── cache/
├── secrets/
│   ├── tokens.json                    # token metadata (mode 0600)
│   └── api_keys                       # plaintext keyfile for llama-server (mode 0600)
└── docker-compose.yml
```

**Hashed auth mode (`--auth-mode hashed`):**

```
/opt/llama/
├── models/  presets/  cache/          # same as above
├── auth_sidecar.py                    # stdlib Python auth server (run by Docker)
├── secrets/
│   ├── tokens.json                    # metadata; value=null, hash=SHA-256 (mode 0600)
│   └── token_hashes.json              # active hashes for the sidecar (mode 0600)
└── docker-compose.yml                 # includes llama-auth sidecar service
```

Log file: `/var/log/llamacpp_deploy.log`

---

## Security model

- The container runs with `cap_drop: ALL`, `no-new-privileges: true`, a read-only root filesystem, and a 512-process limit.
- The `secrets/` directory is mounted read-only inside the container.
- The API is bound to `127.0.0.1` by default — not reachable from the network.
- UFW is configured to deny all incoming traffic except SSH (port detected automatically) and optionally the API port.
- In **plaintext mode**: `api_keys` is `0600` root-only; token values are stored in plaintext (required for llama-server comparison) but never logged or printed after first display.
- In **hashed mode**: only SHA-256 hashes are stored in `token_hashes.json`; the plaintext is never written to disk. llama-server runs without `--api-key-file`. Auth is enforced by the `llama-auth` Docker sidecar via NGINX `auth_request`.
- Choosing `--bind 0.0.0.0` shows a warning in the wizard and requires `--open-firewall` to be explicitly passed in batch mode to add a UFW rule.
- When `--domain` is used, NGINX terminates TLS publicly; the Docker container stays bound to `127.0.0.1`. Only ports 80 and 443 are opened in UFW — port 8080 is never exposed.

### Two token storage modes

#### Plaintext (default)

llama-server owns the auth check. It reads `api_keys` line by line and does a direct string comparison against the `Authorization: Bearer` header:

```
Client → Bearer sk-abc123 → llama-server reads api_keys line by line
                              → string compare: "sk-abc123" == line?
```

This requires the raw token value on disk. It is the simpler option and works without NGINX.

#### Hashed (`--auth-mode hashed`)

llama-server's built-in auth is disabled. NGINX delegates every request to a tiny auth sidecar via `auth_request`:

```
Client → Bearer sk-abc123
  → NGINX auth_request /auth
    → llama-auth sidecar:
        SHA-256("sk-abc123") == hash in token_hashes.json?  → 200 / 401
  → if 200: proxy to llama-server (127.0.0.1:<auto-selected-port>, no --api-key-file)
```

By default, hashed mode prefers loopback ports `8081` (llama) and `9000` (sidecar).
If either port is already in use, deployment now auto-selects free loopback ports
and writes the same values into both Docker Compose and NGINX.

Only the SHA-256 hash is stored. The plaintext value is shown once at creation and never written to disk. Revocation removes the hash from `token_hashes.json` and takes effect on the next request — no container restart needed. The trade-off is that NGINX and the auth sidecar are required.

| | Plaintext | Hashed |
|---|---|---|
| Plaintext on disk | `api_keys` | never |
| Stored as | raw value | SHA-256 hex |
| Revocation takes effect | after container restart | immediately |
| Extra services | none | NGINX + `llama-auth` sidecar |
| `tokens show <id>` | works | error (value not stored) |

---

## Security checklist

Run through this after every deployment to confirm your exposure matches your intent.

### All profiles

- [ ] `docker ps` shows container as `Up` and healthy.
- [ ] `docker inspect llama-router | grep -A5 Ports` shows the expected port binding (host IP should be `127.0.0.1` unless profile=`public`).
- [ ] `cat <base-dir>/secrets/tokens.json | python3 -m json.tool` — confirm token exists and is `active`.
- [ ] Container is hardened: `docker inspect llama-router | grep -E 'ReadonlyRootfs|NoNewPrivileges'` → both `true`.

### `localhost` profile

- [ ] `ss -lntp | grep 8080` shows `127.0.0.1:8080` — **not** `0.0.0.0:8080`.
- [ ] `curl http://192.168.1.x:8080/v1/models` from another LAN device → connection refused.

### `home-private` profile

- [ ] `ss -lntp | grep 8080` shows `127.0.0.1:8080` (Docker bind is loopback; UFW handles LAN).
- [ ] `ufw status numbered | grep 8080` shows `ALLOW from <lan-cidr>` AND `DENY 8080/tcp`.
- [ ] `curl http://<server-lan-ip>:8080/v1/models -H "Authorization: Bearer <token>"` → 200 from a LAN device.
- [ ] `curl http://<server-lan-ip>:8080/v1/models` from a device **outside** the LAN CIDR → connection refused or timeout.

### `vpn-only` profile

- [ ] `tailscale status` shows this node as `online`.
- [ ] `ss -lntp | grep 8080` shows `127.0.0.1:8080` — **not** `0.0.0.0:8080`.
- [ ] `ufw status | grep 8080` → **no rule** (port is not opened; Tailscale handles routing).
- [ ] `curl http://<tailscale-ip>:8080/v1/models -H "Authorization: Bearer <token>"` from a Tailscale peer → 200.
- [ ] `curl http://<tailscale-ip>:8080/v1/models` from a non-Tailscale device → connection refused.

### `public` (NGINX + TLS) profile

- [ ] `curl https://<domain>/v1/models -H "Authorization: Bearer <token>"` → 200.
- [ ] `curl http://<domain>/v1/models` → 301 redirect to HTTPS.
- [ ] `ss -lntp | grep 8080` shows `127.0.0.1:8080` (Docker behind NGINX).
- [ ] `ufw status | grep 8080` → **no rule** (only 80 and 443 are open).
- [ ] `certbot certificates` → certificate valid, expiry > 30 days.

---

## Re-deploying / updating

Running the deploy command again is safe — all system steps are idempotent:

- If Docker is already installed it is not reinstalled.
- If swap already exists it is not recreated.
- If a token already exists in `tokens.json` it is kept; no new token is generated.
- GGUF files with a passing SHA-256 check are not re-downloaded.

To force a clean model re-download, delete the `.gguf` files from `<base-dir>/models/`.

---

## Development checks

Run a quick sanity check locally:

```bash
python3 -m compileall -q llama_deploy
python3 -m unittest discover -s tests -v
```

GitHub Actions runs the same checks on push/PR (`.github/workflows/ci.yml`).

---

## Package structure

```
llama_deploy/
├── __init__.py       package version
├── __main__.py       entry point → cli.dispatch()
├── cli.py            subcommand dispatcher + batch-mode argparse
├── config.py         Config, ModelSpec, NetworkConfig, BackendKind dataclasses
├── log.py            redact(), log_line(), die(), sh()
├── model.py          HuggingFace API, download, SHA-256 verify, resolve_model()
├── nginx.py          NGINX install, proxy config, certbot, UFW ports 80+443
├── orchestrator.py   Step, run_steps(), run_deploy()
├── service.py        write_models_ini(), write_compose(), docker_*()
├── health.py         wait_health(), curl_smoke_tests(), sanity_checks(), profile_smoke_checks()
├── system.py         ensure_*() idempotent OS operations (UFW rules are profile-aware)
├── tailscale.py      Tailscale install, up, IP retrieval, health check
├── tokens.py         TokenRecord, TokenStore (create / revoke / sync keyfile)
└── wizard.py         interactive 6-step HITL wizard → Config
```

---

## Troubleshooting

**The wizard does not start, I see `--help` output.**
The wizard requires an interactive terminal (TTY). Use a real SSH session, not a piped command. For automation use `deploy --batch`.

**`docker: command not found` after the script runs.**
Reload your shell or log out and back in: `exec bash -l`. Docker was just installed; the PATH needs refreshing.

**`failed to create network ... all predefined address pools have been fully subnetted`.**
If you use `--docker-network-mode compose`, Docker Compose creates project networks and the host can run out of subnets.
Use `--docker-network-mode bridge` (default) to avoid per-project network creation.

Prune unused bridge networks, then retry:
```bash
docker network prune -f
```
If needed, remove specific conflicting networks instead of global prune:
```bash
docker network ls --filter driver=bridge
docker network inspect <network_name>
docker network rm <network_name>
```
During deploy, a preflight check now aborts early when Docker bridge subnets overlap host routes so the failure is explicit.

**`Bind for 127.0.0.1:8081 failed: port is already allocated`.**
In hashed mode, `llama_deploy` now auto-selects free loopback ports for the internal
llama upstream and auth sidecar, and configures matching NGINX upstream/auth ports.
Re-run deploy to regenerate `docker-compose.yml` and NGINX config with free ports.

**Download fails or SHA-256 mismatch.**
In the interactive wizard flow, deployment now offers a retry loop: you can switch to another preset model (or enter a custom repo/pattern) and retry until one succeeds. In batch mode (non-interactive), retries are not prompted; set model flags explicitly or rerun with a different model repo.

**The container starts but `/health` times out.**
The model is still loading into memory. For large models on slow disks this can take several minutes. Increase the timeout by editing the `wait_health` call in `orchestrator.py`, or watch progress with:
```bash
docker logs -f llama-router
```

**I revoked a token but requests still succeed.**
In `plaintext` mode, the `api_keys` file was updated, but the running container must be restarted to reload it:
```bash
docker compose -f /opt/llama/docker-compose.yml restart llama
```
In `hashed` mode, revocation is immediate and no restart is required.

**I want to expose the API over the network — which profile should I use?**

| Goal | Profile | Command |
|---|---|---|
| Access from this machine only | `localhost` (default) | no extra flags |
| Access from home LAN devices | `home-private` | `--profile home-private --lan-cidr 192.168.1.0/24` |
| Access from anywhere via VPN | `vpn-only` | `--profile vpn-only [--tailscale-authkey KEY]` |
| Access from internet with TLS | `public` + domain | `--domain api.example.com --certbot-email me@x.com` |
| Access from internet bare HTTP | `public` + open port | `--profile public --open-firewall` |

**Port 8080 is accessible from my LAN even though I chose `home-private`.**
Verify the Docker bind: `ss -lntp | grep 8080` should show `127.0.0.1:8080`, not `0.0.0.0:8080`. Then verify the UFW rule: `ufw status numbered` should show `ALLOW from 192.168.1.0/24 to any port 8080` and `DENY 8080/tcp`. If the Docker port shows `0.0.0.0`, re-deploy — the bind-policy enforcement in `service.py` pins `home-private` to loopback.

**Tailscale says "needs authentication" during deployment.**
If you did not provide a `--tailscale-authkey`, `tailscale up` will print a login URL to the terminal. Open it in a browser on any device, authenticate, and deployment will continue automatically.

**`ss -lntp` shows port 8080 on 0.0.0.0 after a VPN-only deploy.**
This should not happen — `vpn-only` pins the Docker bind to `127.0.0.1`. If you see it, check `docker-compose.yml` under `<base-dir>/` and confirm the ports line reads `"127.0.0.1:8080:8080"`. Re-run deploy to regenerate it.

**certbot fails: "Could not find a usable domain name".**
The DNS A record for your domain must already point to this server's IP address before running the deploy command. Verify with `dig +short api.myserver.com` or `nslookup api.myserver.com`.

**certbot fails: connection refused on port 80.**
Port 80 must be reachable from the internet for the Let's Encrypt HTTP-01 challenge. Check that UFW is not blocking it (`ufw status`) and that no upstream firewall (e.g. cloud provider security group) blocks port 80.

**NGINX returns 502 Bad Gateway.**
The llama-server container is not running or is still loading the model. Check with:
```bash
docker compose -f /opt/llama/docker-compose.yml ps
docker logs llama-router
```

**NGINX returns 500 in local hashed mode.**
`llama_deploy` now keeps existing NGINX sites untouched and auto-selects a free
local proxy port when the requested one is already in use. Re-run deploy so
`docker-compose.yml` and NGINX are regenerated with matching ports.

