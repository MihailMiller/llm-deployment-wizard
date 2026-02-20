"""
Health polling and smoke tests.

curl_smoke_tests receives ModelSpec objects and uses effective_alias for model
names in request bodies — no hardcoded "Qwen/Qwen3-8B" strings.
"""

from __future__ import annotations

import json
import time
import urllib.request

from llama_deploy.config import ModelSpec
from llama_deploy.log import die, log_line, sh


def wait_health(url: str, timeout_s: int = 300) -> None:
    """
    Poll /health until it returns HTTP 200 or the deadline expires.

    Diagnostic hints are emitted at regular intervals so a stalled step is
    immediately actionable rather than silently timing out.  Hint intervals:
      30 s  — suggest checking docker logs
      60 s  — show live container status
      120 s — dump recent logs + port bindings
    """
    from tqdm import tqdm

    HINT_INTERVALS = {
        30:  "[WAIT] Still waiting… Check container logs: docker logs --tail 40 llama-router",
        60:  "[WAIT] Still waiting… Running: docker ps",
        120: "[WAIT] Still waiting… Dumping recent logs and port state for diagnosis:",
    }
    hint_shown: set = set()

    deadline = time.time() + timeout_s
    start = time.time()
    last_tick = start
    with tqdm(total=timeout_s, desc="Waiting for /health", unit="s") as bar:
        while time.time() < deadline:
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=3) as resp:
                    if resp.status == 200:
                        tqdm.write("[OK] /health returned 200")
                        log_line("[OK] /health returned 200")
                        return
            except Exception:
                pass

            elapsed = int(time.time() - start)
            for threshold, hint in HINT_INTERVALS.items():
                if elapsed >= threshold and threshold not in hint_shown:
                    hint_shown.add(threshold)
                    tqdm.write(hint)
                    log_line(hint)
                    if threshold == 60:
                        sh("docker ps --format 'table {{.Names}}\\t{{.Status}}\\t{{.Ports}}'",
                           check=False)
                    elif threshold == 120:
                        sh("docker logs --tail 60 llama-router", check=False)
                        sh("ss -lntp | head -30", check=False)

            now = time.time()
            step = int(now - last_tick)
            if step > 0:
                bar.update(step)
                last_tick = now
            time.sleep(1)

    # Deadline expired — emit rich diagnostics before dying
    tqdm.write("[ERROR] /health did not return 200 within the timeout.")
    log_line(f"[ERROR] /health timeout after {timeout_s}s")
    tqdm.write("[DIAG] Container status:")
    sh("docker ps -a --format 'table {{.Names}}\\t{{.Image}}\\t{{.Status}}\\t{{.Ports}}'",
       check=False)
    tqdm.write("[DIAG] Recent logs (last 80 lines):")
    sh("docker logs --tail 80 llama-router", check=False)
    tqdm.write("[DIAG] Port bindings:")
    sh("ss -lntp", check=False)
    die(
        f"Service did not become healthy within {timeout_s}s.\n"
        "Next steps:\n"
        "  1. docker logs llama-router          — check for model load errors / OOM\n"
        "  2. docker inspect llama-router        — verify volume mounts\n"
        "  3. ss -lntp                           — verify port is bound\n"
        "  4. Increase swap or reduce --ctx-llm if you see OOM messages."
    )


def curl_smoke_tests(
    base_url: str,
    token: str,
    llm: ModelSpec,
    emb: ModelSpec,
) -> None:
    """
    Run three smoke tests against the OpenAI-compatible API.

    Model names come from spec.effective_alias so they match what the server
    advertises in /v1/models — regardless of which HF repo was used.
    Fixes Bug 2 (hardcoded Qwen model names in the original script).
    """
    log_line(f"[SMOKE] Starting smoke tests against {base_url}")

    # 1. Model listing
    sh(
        f'curl -fsS "{base_url}/v1/models" '
        f'-H "Authorization: Bearer {token}" | head -c 800'
    )

    # 2. Embeddings
    emb_payload = json.dumps({"model": emb.effective_alias, "input": ["hello world"]})
    sh(
        f'curl -fsS "{base_url}/v1/embeddings" '
        f'-H "Authorization: Bearer {token}" '
        f'-H "Content-Type: application/json" '
        f"-d '{emb_payload}' | head -c 800"
    )

    # 3. Chat completion
    chat_payload = json.dumps({
        "model": llm.effective_alias,
        "messages": [{"role": "user", "content": "Say hello in 5 words."}],
        "max_tokens": 64,
        "temperature": 0.2,
    })
    sh(
        f'curl -fsS "{base_url}/v1/chat/completions" '
        f'-H "Authorization: Bearer {token}" '
        f'-H "Content-Type: application/json" '
        f"-d '{chat_payload}' | head -c 800"
    )


def profile_smoke_checks(cfg) -> None:
    """
    Reproducible smoke-checks verifying access profile invariants:

      LOCALHOST / VPN_ONLY  : port NOT reachable from LAN (checked via ss)
      HOME_PRIVATE          : same Docker bind check; UFW rule separately visible
      PUBLIC                : port binding present as expected

    Results are logged but never fatal — this is diagnostic, not blocking.
    """
    from llama_deploy.config import AccessProfile
    from tqdm import tqdm

    net = cfg.network
    profile = net.access_profile

    tqdm.write(f"[SMOKE] Profile check: profile={profile.value}  port={net.port}")
    log_line(f"[SMOKE] Profile check: profile={profile.value}  port={net.port}")

    # Check 1: Docker port binding
    rc_all = sh(
        f"ss -lntp | grep -E '0\\.0\\.0\\.0:{net.port}\\b|\\[::\\]:{net.port}\\b'",
        check=False,
    )
    rc_lo = sh(
        f"ss -lntp | grep -E '127\\.0\\.0\\.1:{net.port}\\b'",
        check=False,
    )

    if profile in (AccessProfile.LOCALHOST, AccessProfile.VPN_ONLY, AccessProfile.HOME_PRIVATE):
        if rc_all == 0:
            tqdm.write(f"[SMOKE] FAIL: port {net.port} exposed on 0.0.0.0 — should be loopback only.")
            log_line(f"[SMOKE] FAIL: port {net.port} exposed on 0.0.0.0 (profile={profile.value}).")
        elif rc_lo == 0:
            tqdm.write(f"[SMOKE] OK: port {net.port} bound to 127.0.0.1 only.")
            log_line(f"[SMOKE] OK: port {net.port} on loopback (profile={profile.value}).")
        else:
            tqdm.write(f"[SMOKE] WARN: port {net.port} not found on any interface — container may not be up yet.")

    elif profile == AccessProfile.PUBLIC:
        if net.open_firewall and rc_all == 0:
            tqdm.write(f"[SMOKE] OK: port {net.port} exposed on 0.0.0.0 (public profile, open_firewall=True).")
            log_line(f"[SMOKE] OK: port {net.port} public as expected.")
        elif rc_lo == 0:
            tqdm.write(f"[SMOKE] OK: port {net.port} on loopback (public profile with NGINX).")
            log_line(f"[SMOKE] OK: port {net.port} on loopback for NGINX proxy.")
        else:
            tqdm.write(f"[SMOKE] WARN: port {net.port} not found — check Docker compose logs.")

    # Check 2: UFW status (informational only)
    sh("ufw status numbered 2>/dev/null | head -30 || true", check=False)


def sanity_checks(cfg) -> None:
    """
    Docker status, recent logs, and port-binding verification.

    For each access profile the check verifies that the backend port is NOT
    reachable from unintended sources:
      LOCALHOST / VPN_ONLY : port must NOT appear on 0.0.0.0 or [::]
      HOME_PRIVATE         : same — Docker bind is pinned to 127.0.0.1 by
                             service._effective_bind_host(); UFW handles LAN.
      PUBLIC               : port may appear on 0.0.0.0 (intentional), but
                             we still log what ss sees for the operator.
    """
    from llama_deploy.config import AccessProfile

    sh(
        "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' "
        "| sed -n '1,30p'",
        check=False,
    )
    sh("docker logs --tail 200 llama-router || true", check=False)
    sh("ss -lntp | sed -n '1,200p'", check=False)

    net = cfg.network
    profile = net.access_profile

    if profile in (AccessProfile.LOCALHOST, AccessProfile.VPN_ONLY, AccessProfile.HOME_PRIVATE):
        # Verify no accidental 0.0.0.0 / [::] exposure (grep exits 0 when it finds a match)
        rc = sh(
            f"ss -lntp | grep -E '0\\.0\\.0\\.0:{net.port}\\b|\\[::\\]:{net.port}\\b'",
            check=False,
        )
        if rc == 0:
            log_line(
                f"[WARN] sanity: port {net.port} appears on 0.0.0.0/[::] "
                f"despite profile={profile.value}. Check Docker port mapping."
            )
            from tqdm import tqdm
            tqdm.write(
                f"[WARN] Port {net.port} is exposed on 0.0.0.0 — "
                f"unexpected for profile={profile.value}. "
                "Verify docker-compose.yml ports binding."
            )
        else:
            log_line(f"[OK] sanity: port {net.port} not exposed on 0.0.0.0 (profile={profile.value}).")

    elif profile == AccessProfile.PUBLIC:
        log_line(f"[OK] sanity: profile=public — port {net.port} exposure is intentional.")
