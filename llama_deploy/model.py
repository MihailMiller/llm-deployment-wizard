"""
HuggingFace model resolution and download.

Public entry point:
  resolve_model(spec, dst_dir, hf_token, allow_unverified_downloads=False)
  -> ModelSpec
Returns a new ModelSpec with resolved_filename / resolved_sha256 / resolved_size
populated, and the GGUF file present on disk (strictly verified when upstream
metadata includes sha256/size).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import shutil
import sys
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
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _normalize_sha256(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    v = raw.strip()
    if v.startswith("W/"):
        v = v[2:].strip()
    v = v.strip('"').strip("'")
    if ":" in v:
        # Handle "sha256:<hex>" style values.
        _, tail = v.split(":", 1)
        v = tail.strip()
    if _SHA256_RE.fullmatch(v):
        return v.lower()
    return None


def _int_or_none(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _confirm_trust_unverified(
    *,
    repo: str,
    revision: str,
    filename: str,
    reason: str,
    allow_unverified_downloads: bool,
) -> bool:
    from tqdm import tqdm

    if allow_unverified_downloads:
        tqdm.write(
            "[WARN] Proceeding with unverified download because "
            "--allow-unverified-downloads is set."
        )
        return True

    tqdm.write(f"[WARN] {reason}")
    if not sys.stdin.isatty():
        tqdm.write(
            "[WARN] Non-interactive run: refusing unverified download. "
            "Re-run with --allow-unverified-downloads to proceed."
        )
        return False

    tqdm.write(f"[WARN] File: {repo}@{revision[:12]} / {filename}")
    while True:
        try:
            ans = input("Trust this download and continue? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if ans in ("y", "yes"):
            return True
        if ans in ("", "n", "no"):
            return False


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


def probe_hf_resolve_metadata(
    repo: str,
    revision: str,
    filename: str,
    hf_token: Optional[str],
) -> Tuple[Optional[int], Optional[str]]:
    """
    Best-effort metadata probe via /resolve headers.

    Some public HF repos omit lfs.sha256/size in /api/models payload. We try to
    recover size and checksum from response headers (if exposed).
    """
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("User-Agent", "llamacpp-paranoid-deploy/1.0")
    if hf_token:
        req.add_header("Authorization", f"Bearer {hf_token}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            headers = resp.headers
            size = (
                _int_or_none(headers.get("x-linked-size"))
                or _int_or_none(headers.get("content-length"))
            )
            sha = (
                _normalize_sha256(headers.get("x-linked-etag"))
                or _normalize_sha256(headers.get("etag"))
            )
            return size, sha
    except Exception:
        return None, None


def pick_hf_file(meta: Dict[str, Any], spec: ModelSpec) -> Tuple[str, Optional[int], Optional[str]]:
    """
    Return (filename, size_bytes?, sha256?) by matching spec.candidate_patterns
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
        pattern_l = pattern.lower()
        # Case-insensitive substring match to handle repos with lowercase filenames.
        matches = [name for name in by_name if pattern_l in name.lower()]
        if not matches:
            continue
        # Prefer shorter name (less quantization suffixes) among substring matches
        chosen = sorted(matches, key=len)[0]
        s = by_name[chosen]
        lfs = s.get("lfs") or {}
        sha = lfs.get("sha256") or lfs.get("oid")
        size = lfs.get("size") or s.get("size")
        resolved_size = int(size) if size else None
        resolved_sha = _normalize_sha256(str(sha) if sha else None)
        return chosen, resolved_size, resolved_sha

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
    revision: str,
    filename: str,
    dst: Path,
    expected_sha256: Optional[str],
    expected_size: Optional[int],
    hf_token: Optional[str],
    *,
    allow_unverified_downloads: bool = False,
) -> Tuple[str, int]:
    """
    Download a file from HuggingFace with tqdm progress.
    If expected metadata is present, verify size and sha256 strictly.
    Resumes a partial .part file if one exists (HTTP Range request).
    """
    from tqdm import tqdm

    dst.parent.mkdir(parents=True, exist_ok=True)

    # If file already exists and it can be validated, keep it.
    if dst.exists():
        got = sha256_file(dst)
        got_size = dst.stat().st_size
        if expected_sha256:
            if got.lower() == expected_sha256.lower():
                tqdm.write(f"[OK] {dst.name} exists and sha256 matches.")
                return got, got_size
        elif expected_size is not None and got_size == expected_size:
            tqdm.write(f"[OK] {dst.name} exists and size matches.")
            tqdm.write(f"[WARN] Upstream sha256 unavailable; using local sha256={got}.")
            return got, got_size
        elif expected_size is None:
            tqdm.write(f"[WARN] Upstream sha256/size unavailable; reusing existing {dst.name}.")
            tqdm.write(f"[WARN] Local sha256 for reference: {got}")
            return got, got_size

        ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        bad = dst.with_suffix(dst.suffix + f".BADMETA.{ts}")
        shutil.move(dst, bad)
        tqdm.write(f"[WARN] Existing {dst.name} could not be validated; moved aside to {bad.name}")

    if expected_size is not None:
        ensure_disk_space(dst.parent, expected_size)

    part = dst.with_suffix(dst.suffix + ".part")
    if part.exists() and expected_size is not None and expected_sha256:
        part_size = part.stat().st_size
        if part_size > expected_size:
            part.unlink(missing_ok=True)
        elif part_size == expected_size:
            part_sha = sha256_file(part)
            if part_sha.lower() == expected_sha256.lower():
                import os
                part.replace(dst)
                os.chmod(dst, 0o644)
                tqdm.write(f"[OK] Reused complete {part.name}, sha256 verified.")
                return part_sha, expected_size
            tqdm.write(f"[WARN] Discarding stale {part.name}; checksum does not match expected revision.")
            part.unlink(missing_ok=True)

    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    resume_from = part.stat().st_size if part.exists() else 0
    if expected_size is None and resume_from > 0:
        # Without reliable upstream size, avoid risking a bad append if Range is ignored.
        part.unlink(missing_ok=True)
        resume_from = 0

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
    if expected_size is not None and got_size != expected_size:
        die(f"Size mismatch for {dst.name}. Expected {expected_size}, got {got_size} bytes.")

    got_sha = h.hexdigest()
    if expected_sha256 and got_sha.lower() != expected_sha256.lower():
        _, probe_sha = probe_hf_resolve_metadata(repo, revision, filename, hf_token)
        if probe_sha and got_sha.lower() == probe_sha.lower():
            tqdm.write(
                f"[WARN] Expected sha256 ({expected_sha256[:12]}...) mismatched, "
                f"but resolve header sha256 matches downloaded file ({probe_sha[:12]}...). "
                f"Accepting pinned file."
            )
            expected_sha256 = probe_sha
        else:
            reason = (
                f"SHA256 mismatch for {dst.name}. "
                f"Expected {expected_sha256}, got {got_sha}. "
                "Resolve headers did not provide a matching checksum."
            )
            if not _confirm_trust_unverified(
                repo=repo,
                revision=revision,
                filename=filename,
                reason=reason,
                allow_unverified_downloads=allow_unverified_downloads,
            ):
                die(
                    f"SHA256 mismatch for {dst.name}\n"
                    f"Expected: {expected_sha256}\n"
                    f"Got:      {got_sha}"
                )

    import os
    part.replace(dst)
    os.chmod(dst, 0o644)
    if expected_sha256:
        tqdm.write(f"[OK] Downloaded {dst.name}, sha256 verified.")
    else:
        tqdm.write(f"[OK] Downloaded {dst.name}.")
        tqdm.write(f"[WARN] Upstream sha256 unavailable; local sha256={got_sha}")
    return got_sha, got_size


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def resolve_model(
    spec: ModelSpec,
    dst_dir: Path,
    hf_token: Optional[str],
    *,
    allow_unverified_downloads: bool = False,
) -> ModelSpec:
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

    def _resolve_from_repo(repo: str, pick_spec: ModelSpec) -> Tuple[str, Optional[int], Optional[str], str]:
        tqdm.write(f"[HF] Querying metadata for {repo}")
        meta = hf_model_metadata(repo, hf_token)
        revision = str(meta.get("sha") or "main")
        filename, size, sha256 = pick_hf_file(meta, pick_spec)
        probed_size, probed_sha = probe_hf_resolve_metadata(repo, revision, filename, hf_token)
        if probed_size is not None:
            if size is not None and size != probed_size:
                tqdm.write(
                    f"[WARN] HF API size ({size}) disagrees with resolve header ({probed_size}); "
                    f"using resolve header."
                )
            size = probed_size
        if probed_sha is not None:
            if sha256 is not None and sha256.lower() != probed_sha.lower():
                tqdm.write(
                    f"[WARN] HF API sha256 ({sha256[:12]}...) disagrees with resolve header "
                    f"({probed_sha[:12]}...); using resolve header."
                )
            sha256 = probed_sha
        return filename, size, sha256, revision

    resolved_repo = spec.hf_repo
    primary: Optional[Tuple[str, Optional[int], Optional[str], str]] = None
    primary_err: Optional[ValueError] = None
    try:
        primary = _resolve_from_repo(spec.hf_repo, spec)
    except ValueError as e:
        primary_err = e

    fallback_repo = _METADATA_FALLBACK_REPOS.get(spec.hf_repo)
    chosen = primary

    if fallback_repo and (primary_err is not None or (primary and (primary[1] is None or primary[2] is None))):
        reason = str(primary_err) if primary_err else f"HF metadata for '{primary[0]}' is missing sha256/size."
        tqdm.write(f"[WARN] {reason}")
        tqdm.write(f"[HF] Retrying metadata from fallback repo: {fallback_repo}")
        log_line(f"[HF] Retrying metadata from fallback repo: {fallback_repo}")
        fallback_spec = ModelSpec(
            hf_repo=fallback_repo,
            candidate_patterns=spec.candidate_patterns,
            ctx_len=spec.ctx_len,
            alias=spec.effective_alias,  # keep served model alias stable
            is_embedding=spec.is_embedding,
        )
        try:
            fb = _resolve_from_repo(fallback_repo, fallback_spec)
            if primary is None:
                chosen = fb
                resolved_repo = fallback_repo
            else:
                primary_score = int(primary[1] is not None) + int(primary[2] is not None)
                fb_score = int(fb[1] is not None) + int(fb[2] is not None)
                if fb_score >= primary_score:
                    chosen = fb
                    resolved_repo = fallback_repo
        except ValueError as e2:
            if primary_err is not None:
                die(f"{primary_err} Fallback repo '{fallback_repo}' also failed: {e2}")
            tqdm.write(f"[WARN] Fallback repo '{fallback_repo}' did not improve resolution: {e2}")

    if primary_err is not None and chosen is None:
        die(str(primary_err))
    assert chosen is not None
    filename, size, sha256, revision = chosen

    if size is None:
        tqdm.write(f"[WARN] Size unavailable for {filename}; progress may not show total bytes.")
    if sha256 is None:
        if not _confirm_trust_unverified(
            repo=resolved_repo,
            revision=revision,
            filename=filename,
            reason=f"No sha256 metadata is available for {filename}.",
            allow_unverified_downloads=allow_unverified_downloads,
        ):
            die(
                f"Refusing unverified download for {filename}. "
                "Approve trust interactively or re-run with --allow-unverified-downloads."
            )
        tqdm.write(f"[WARN] sha256 unavailable for {filename}; proceeding by user trust.")

    size_txt = f"{size / 1e9:.2f} GB" if size is not None else "size=?"
    sha_txt = f"{sha256[:12]}..." if sha256 else "sha256=?"
    tqdm.write(f"[HF] Resolved from {resolved_repo}@{revision[:12]}: {filename} ({size_txt}, {sha_txt})")

    dst = dst_dir / filename
    local_sha, local_size = download_hf_file(
        resolved_repo,
        revision,
        filename,
        dst,
        sha256,
        size,
        hf_token,
        allow_unverified_downloads=allow_unverified_downloads,
    )

    return spec.with_resolved(filename=filename, sha256=local_sha, size=local_size)
