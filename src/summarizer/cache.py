from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional


def _safe_slug(repo_url: str) -> str:
    # Keep it filesystem-friendly.
    slug = repo_url.strip().rstrip("/")
    for prefix in ("https://", "http://", "git@"):
        if slug.startswith(prefix):
            slug = slug[len(prefix) :]
    slug = slug.replace(":", "_").replace("/", "_")
    return slug


def default_cache_dir() -> Path:
    # Allow override (useful for deployments).
    env = os.getenv("GITCONNECT_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".gitconnect" / "cache").resolve()


def cache_key(
    *,
    repo_url: str,
    commit_sha: str,
    model: Optional[str],
    use_grounding: bool,
    user_prompt: str,
) -> str:
    """Stable key for a repo at a given commit.

    We include prompt/model/grounding so callers can request different summaries
    without overwriting a cached result for the same commit.
    """

    prompt_norm = user_prompt.strip().encode("utf-8")
    h = hashlib.sha256()
    h.update(commit_sha.encode("utf-8"))
    h.update(b"\n")
    h.update((model or "").encode("utf-8"))
    h.update(b"\n")
    h.update(b"1" if use_grounding else b"0")
    h.update(b"\n")
    h.update(prompt_norm)
    digest = h.hexdigest()[:16]
    return f"{commit_sha[:12]}_{digest}"


def cache_path(
    *,
    repo_url: str,
    commit_sha: str,
    model: Optional[str],
    use_grounding: bool,
    user_prompt: str,
    cache_dir: Optional[Path] = None,
) -> Path:
    cache_dir = cache_dir or default_cache_dir()
    slug = _safe_slug(repo_url)
    key = cache_key(
        repo_url=repo_url,
        commit_sha=commit_sha,
        model=model,
        use_grounding=use_grounding,
        user_prompt=user_prompt,
    )
    return cache_dir / slug / f"{key}.json"


def load_cached_json(path: Path) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_cache_json(path: Path, obj: dict) -> None:
    """Write cache file if it does not already exist.

    This guarantees a cached result for a given key won't be overwritten by a later run.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        # Cache should never break core functionality.
        return
