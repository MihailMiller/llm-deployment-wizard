# llama_deploy

Deploy **llama.cpp** (`llama-server`) as a private, OpenAI-compatible REST API on an Ubuntu server — with one command, a guided wizard, and named API tokens you can revoke at any time.

```
POST /v1/chat/completions   model="unsloth/Ministral-3-14B-Instruct-2512"
POST /v1/embeddings         model="Qwen/Qwen3-Embedding-0.6B"
GET  /v1/models
```

The server runs inside Docker, is bound to localhost by default, uses a hardened read-only container, and is never exposed to the internet unless you explicitly choose it.

---

## What this does

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
| 4 GB RAM minimum | 16+ GB recommended for the default Ministral-3-14B model. |
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

The default models (Ministral-3-14B-Instruct-2512, Qwen3-Embedding-0.6B) are public and do not require a token.
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

You will be guided through five steps:

```
╔════════════════════════════════════════════════════╗
║         llama.cpp Deployment Wizard                ║
║   Configure your OpenAI-compatible AI service      ║
╚════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════╗
║  Step 1/5 · Backend                               ║
╚════════════════════════════════════════════════════╝

What hardware acceleration do you have?
  [1] CPU only        works everywhere — no GPU required
  [2] NVIDIA GPU      CUDA — requires nvidia-docker
  [3] AMD GPU         ROCm — requires ROCm drivers
  [4] Vulkan GPU      cross-vendor GPU acceleration
  [5] Intel GPU       SYCL / OpenVINO

Choice [1]:

╔════════════════════════════════════════════════════╗
║  Step 3/5 · Network                               ║
╚════════════════════════════════════════════════════╝

  [1] Localhost only   127.0.0.1 — most secure, for local apps only
  [2] HTTPS + domain   NGINX + Let's Encrypt — recommended for internet exposure
  [3] All interfaces   0.0.0.0 — direct exposure without TLS
  [4] Docker network   no host port — only from other containers

Choice [1]: 2

  Domain name (e.g. api.example.com): api.myserver.com
  Email for Let's Encrypt renewal notices: me@example.com
  Internal port [8080]:
```

After step 5, a review summary is shown and you confirm before anything is written to disk.

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
  --bind HOST           Host address to bind the published port to.
                        Use 0.0.0.0 to expose publicly. (default: 127.0.0.1)
  --port PORT           Host port to publish. (default: 8080)
  --no-publish          Do NOT publish any host port (Docker-network-only).
  --open-firewall       Open --port in UFW. Requires --bind 0.0.0.0.
  --skip-ufw            Do not configure UFW at all.

System:
  --swap-gib GIB        Swap file size if none exists. (default: 8)

Authentication:
  --token TOKEN         API token value. Randomly generated if omitted.
  --token-name NAME     Label for the first API token. (default: default)
  --auth-mode MODE      Token storage: plaintext (default) or hashed.
                        hashed stores only SHA-256; requires NGINX + sidecar.
  --hf-token TOKEN      HuggingFace access token (or use HF_TOKEN env var).

Models:
  --llm-repo REPO       HuggingFace repo for the LLM GGUF.
                        (default: unsloth/Ministral-3-14B-Instruct-2512-GGUF)
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
                        DNS A record for DOMAIN must already point to this host.
  --certbot-email EMAIL Email for Let's Encrypt renewal notices.
                        Required when --domain is set.

Server tuning:
  --models-max N        Max simultaneously loaded models. (default: 2)
  --ctx-llm TOKENS      Context window for the LLM. (default: 4096)
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
    "model": "unsloth/Ministral-3-14B-Instruct-2512",
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
    model="unsloth/Ministral-3-14B-Instruct-2512",
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

Select **[2] HTTPS + domain** in Step 3/5 · Network:

```
╔════════════════════════════════════════════════════╗
║  Step 3/5 · Network                               ║
╚════════════════════════════════════════════════════╝

  [1] Localhost only   127.0.0.1 — most secure, for local apps only
  [2] HTTPS + domain   NGINX + Let's Encrypt — recommended for internet exposure
  [3] All interfaces   0.0.0.0 — direct exposure without TLS
  [4] Docker network   no host port — only from other containers

Choice [1]: 2

  NGINX will listen on :80 and :443 and proxy to the loopback port.
  Make sure your domain's DNS A record already points to this server.

  Domain name (e.g. api.example.com): api.myserver.com
  Email for Let's Encrypt renewal notices: me@example.com
  Internal port [8080]:
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
│   ├── Ministral-3-14B-Instruct-2512-Q4_K_M.gguf
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
  → if 200: proxy to llama-server (127.0.0.1:8081, no --api-key-file)
```

Only the SHA-256 hash is stored. The plaintext value is shown once at creation and never written to disk. Revocation removes the hash from `token_hashes.json` and takes effect on the next request — no container restart needed. The trade-off is that NGINX and the auth sidecar are required.

| | Plaintext | Hashed |
|---|---|---|
| Plaintext on disk | `api_keys` | never |
| Stored as | raw value | SHA-256 hex |
| Revocation takes effect | after container restart | immediately |
| Extra services | none | NGINX + `llama-auth` sidecar |
| `tokens show <id>` | works | error (value not stored) |

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
├── health.py         wait_health(), curl_smoke_tests(), sanity_checks()
├── system.py         ensure_*() idempotent OS operations
├── tokens.py         TokenRecord, TokenStore (create / revoke / sync keyfile)
└── wizard.py         interactive 5-step HITL wizard → Config
```

---

## Troubleshooting

**The wizard does not start, I see `--help` output.**
The wizard requires an interactive terminal (TTY). Use a real SSH session, not a piped command. For automation use `deploy --batch`.

**`docker: command not found` after the script runs.**
Reload your shell or log out and back in: `exec bash -l`. Docker was just installed; the PATH needs refreshing.

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

**I want to expose the API over the network.**
Use `--bind 0.0.0.0 --open-firewall` in batch mode, or choose "All interfaces" in the wizard. Make sure you understand the security implications — the API will be reachable by anyone who can reach the host. For a production setup, prefer option [2] HTTPS + domain in the wizard instead.

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

