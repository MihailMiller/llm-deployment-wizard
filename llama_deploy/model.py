"""
HuggingFace model resolution and download.

Public entry point: resolve_model(spec, dst_dir, hf_token) -> ModelSpec
Returns a new ModelSpec with resolved_filename / resolved_sha256 / resolved_size
populated, and the GGUF file verified on disk.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from llama_deploy.config import ModelSpec
from llama_deploy.log import die, log_line

_METADATA_FALLBACK_REPOS = {
    # Some mirrors expose LFS metadata even when the upstream repo does not.
    "Qwen/Qwen3-8B-GGUF": "Aldaris/Qwen3-8B-Q4_K_M-GGUF",
}


# ---------------------------------------------------------------------------
# HuggingFace API
# ---------------------------------------------------------------------------

def hf_model_metadata(repo: str, hf_token: Optional[str]) -> Dict[str, Any]:
    url = f"https://huggingface.co/api/models/{repo}"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "llamacpp-paranoid-deploy/1.0")
    if hf_token:
        req.add_header("Authorization", f"Bearer {hf_token}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        return json.loads(data.decode("utf-8"))
    except urllib.error.HTTPError as e:
        die(f"HuggingFace API error for {repo}: HTTP {e.code}")
    except Exception as e:
        die(f"Failed to query HuggingFace API for {repo}: {e}")


def pick_hf_file(meta: Dict[str, Any], spec: ModelSpec) -> Tuple[str, int, str]:
    """
    Return (filename, size_bytes, sha256) by matching spec.candidate_patterns
    against the files listed in the HF repo metadata.

    candidate_patterns are matched as substrings of the filename. The first
    pattern that matches an existing repo file is used.

    Fixes Bug 3: previously candidate lists were hardcoded Qwen names in main().
    Now each ModelSpec carries its own candidate_patterns so changing --llm-repo
    to a non-Qwen repo and providing matching --llm-candidates works correctly.
    """
    siblings = meta.get("siblings") or []
    by_name: Dict[str, Dict[str, Any]] = {
        s.get("rfilename"): s
        for s in siblings
        if isinstance(s, dict) and s.get("rfilename")
    }

    for pattern in spec.candidate_patterns:
        # First try exact match, then substring match
        matches = [name for name in by_name if pattern in name]
        if not matches:
            continue
        # Prefer shorter name (less quantization suffixes) among substring matches
        chosen = sorted(matches, key=len)[0]
        s = by_name[chosen]
        lfs = s.get("lfs") or {}
        sha = lfs.get("sha256") or lfs.get("oid")
        size = lfs.get("size") or s.get("size")
        if not sha or not size:
            raise ValueError(
                f"HF metadata for '{chosen}' is missing sha256/size. "
                f"The repo may not expose LFS metadata publicly."
            )
        return chosen, int(size), str(sha)

    raise ValueError(
        f"None of the candidate patterns matched any file in repo '{spec.hf_repo}'. "
        f"Tried patterns: {spec.candidate_patterns}. "
        f"Available files: {sorted(by_name.keys())}"
    )


# ---------------------------------------------------------------------------
# Checksums and disk
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_disk_space(dst_dir: Path, required_bytes: int, headroom: float = 1.20) -> None:
    usage = shutil.disk_usage(dst_dir)
    need = int(required_bytes * headroom)
    if usage.free < need:
        die(
            f"Not enough free disk at {dst_dir}. "
            f"Need ~{need / 1e9:.1f} GB free, have {usage.free / 1e9:.1f} GB."
        )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_hf_file(
    repo: str,
    filename: str,
    dst: Path,
    expected_sha256: str,
    expected_size: int,
    hf_token: Optional[str],
) -> None:
    """
    Download a file from HuggingFace with tqdm progress, verify size and sha256.
    Resumes a partial .part file if one exists (HTTP Range request).
    """
    from tqdm import tqdm

    dst.parent.mkdir(parents=True, exist_ok=True)

    # If file already exists and sha matches, keep it
    if dst.exists():
        got = sha256_file(dst)
        if got.lower() == expected_sha256.lower():
            tqdm.write(f"[OK] {dst.name} exists and sha256 matches.")
            return
        ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        bad = dst.with_suffix(dst.suffix + f".BADSHA.{ts}")
        shutil.move(dst, bad)
        tqdm.write(f"[WARN] Existing {dst.name} sha mismatch; moved aside to {bad.name}")

    ensure_disk_space(dst.parent, expected_size)

    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    part = dst.with_suffix(dst.suffix + ".part")
    resume_from = part.stat().st_size if part.exists() else 0

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "llamacpp-paranoid-deploy/1.0")
    if hf_token:
        req.add_header("Authorization", f"Bearer {hf_token}")
    if resume_from > 0:
        req.add_header("Range", f"bytes={resume_from}-")

    tqdm.write(f"[DL] {url} -> {dst} (resume_from={resume_from})")
    log_line(f"[DL] {url} -> {dst} (resume_from={resume_from})")

    h = hashlib.sha256()
    if resume_from > 0:
        with part.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)

    try:
        with urllib.request.urlopen(req, timeout=60) as resp, \
             part.open("ab") as f, \
             tqdm(
                 total=expected_size,
                 initial=resume_from,
                 unit="B",
                 unit_scale=True,
                 unit_divisor=1024,
                 desc=f"Downloading {dst.name}",
             ) as bar:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                bar.update(len(chunk))
    except Exception as e:
        die(f"Download failed for {dst.name}: {e}")

    got_size = part.stat().st_size
    if got_size != expected_size:
        die(f"Size mismatch for {dst.name}. Expected {expected_size}, got {got_size} bytes.")

    got_sha = h.hexdigest()
    if got_sha.lower() != expected_sha256.lower():
        die(
            f"SHA256 mismatch for {dst.name}\n"
            f"Expected: {expected_sha256}\n"
            f"Got:      {got_sha}"
        )

    import os
    part.replace(dst)
    os.chmod(dst, 0o644)
    tqdm.write(f"[OK] Downloaded {dst.name}, sha256 verified.")


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def resolve_model(spec: ModelSpec, dst_dir: Path, hf_token: Optional[str]) -> ModelSpec:
    """
    Single public entry point for model resolution.

    1. Queries HF API for repo metadata.
    2. Picks the best GGUF file using spec.candidate_patterns.
    3. Downloads and verifies the file.
    4. Returns a new ModelSpec with resolved_* fields populated.

    The orchestrator closes over the returned ModelSpec so write_models_ini
    and curl_smoke_tests receive spec.effective_alias (derived from hf_repo,
    not a hardcoded string).
    """
    from tqdm import tqdm

    resolved_repo = spec.hf_repo
    tqdm.write(f"[HF] Querying metadata for {resolved_repo}")
    meta = hf_model_metadata(resolved_repo, hf_token)

    try:
        filename, size, sha256 = pick_hf_file(meta, spec)
    except ValueError as e:
        fallback_repo = _METADATA_FALLBACK_REPOS.get(spec.hf_repo)
        if fallback_repo and "missing sha256/size" in str(e):
            tqdm.write(f"[WARN] {e}")
            tqdm.write(f"[HF] Retrying metadata from fallback repo: {fallback_repo}")
            log_line(f"[HF] Retrying metadata from fallback repo: {fallback_repo}")

            fallback_spec = ModelSpec(
                hf_repo=fallback_repo,
                candidate_patterns=spec.candidate_patterns,
                ctx_len=spec.ctx_len,
                alias=spec.effective_alias,  # keep original served model alias stable
                is_embedding=spec.is_embedding,
            )
            resolved_repo = fallback_repo
            meta = hf_model_metadata(resolved_repo, hf_token)
            try:
                filename, size, sha256 = pick_hf_file(meta, fallback_spec)
            except ValueError as e2:
                die(f"{e} Fallback repo '{fallback_repo}' also failed: {e2}")
        else:
            die(str(e))

    tqdm.write(
        f"[HF] Resolved from {resolved_repo}: {filename} "
        f"({size / 1e9:.2f} GB, sha256={sha256[:12]}...)"
    )

    dst = dst_dir / filename
    download_hf_file(resolved_repo, filename, dst, sha256, size, hf_token)

    return spec.with_resolved(filename=filename, sha256=sha256, size=size)
