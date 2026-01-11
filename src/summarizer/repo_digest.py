from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from git import Repo  # GitPython

from .schemas import DigestDocument, RepoDigest, RepoMeta


README_CANDIDATES = [
    "README",
    "README.md",
    "README.MD",
    "README.rst",
    "README.txt",
]

DOC_DIR_CANDIDATES = [
    "docs",
    "doc",
    "documentation",
    "guide",
    "guides",
]

MANIFEST_FILES = [
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "Pipfile",
    "Pipfile.lock",
    "poetry.lock",
    "Gemfile",
    "Gemfile.lock",
    "go.mod",
    "go.sum",
    "Cargo.toml",
    "Cargo.lock",
    "composer.json",
    "composer.lock",
]

# Common patterns to skip for safety and noise
SKIP_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
}

SKIP_FILE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".mp4",
    ".mov",
    ".avi",
    ".exe",
    ".dll",
    ".so",
}

# Naive secret patterns (best-effort scrub)
SECRET_PATTERNS = [
    re.compile(r'(?i)api[_-]?key\s*[:=]\s*[\"\']?[A-Za-z0-9_\-]{16,}[\"\']?'),
    re.compile(r'(?i)secret\s*[:=]\s*[\"\']?.{8,}[\"\']?'),
    re.compile(r'(?i)access[_-]?token\s*[:=]\s*[\"\']?[A-Za-z0-9_\-]{16,}[\"\']?'),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS Access Key ID pattern
]


def _scrub_secrets(text: str) -> str:
    """Best-effort redaction of obvious secrets."""
    for pat in SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def _read_text_file(path: Path, max_chars: int) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    data = _scrub_secrets(data)
    if len(data) > max_chars:
        data = data[:max_chars] + "\n[TRUNCATED]"
    return data


def _is_text_candidate(path: Path) -> bool:
    if path.is_dir():
        return False
    if path.suffix.lower() in SKIP_FILE_EXTS:
        return False
    if path.name.startswith(".") and path.name not in {".github", ".gitignore"}:
        return False
    return True


def _detect_language_counts(repo_root: Path, max_files: int = 8000) -> Dict[str, int]:
    """Counts file extensions; used as a rough language signal."""
    counts: Dict[str, int] = {}
    n = 0
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            n += 1
            if n > max_files:
                return counts
            ext = Path(fn).suffix.lower() or "(no_ext)"
            counts[ext] = counts.get(ext, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:30])


def _safe_rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except Exception:
        return str(path)


def build_repo_tree(repo_root: Path, max_lines: int = 2500, max_depth: int = 6) -> str:
    """Build a compact ASCII tree (best-effort)."""
    lines: List[str] = []

    def walk(curr: Path, depth: int) -> None:
        if len(lines) >= max_lines or depth > max_depth:
            return
        try:
            entries = sorted(curr.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except Exception:
            return
        for e in entries:
            if e.name in SKIP_DIRS:
                continue
            if e.is_file() and e.suffix.lower() in SKIP_FILE_EXTS:
                continue
            prefix = "  " * depth
            lines.append(f"{prefix}- {e.name}{'' if e.is_file() else '/'}")
            if e.is_dir():
                walk(e, depth + 1)
            if len(lines) >= max_lines:
                return

    walk(repo_root, 0)
    out = "\n".join(lines)
    if len(lines) >= max_lines:
        out += "\n[TRUNCATED TREE]"
    return out


def _scrub_secrets(text: str) -> str:
    """Best-effort redaction of obvious secrets."""
    for pat in SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def _read_text_file(path: Path, max_chars: int) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    data = _scrub_secrets(data)
    if len(data) > max_chars:
        data = data[:max_chars] + "\n[TRUNCATED]"
    return data


def _detect_languages(repo_root: Path, max_files: int = 5000) -> Dict[str, int]:
    """Rough language detection via file extensions."""
    counts: Dict[str, int] = {}
    n = 0
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            n += 1
            if n > max_files:
                return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:30])
            ext = Path(fn).suffix.lower() or "(no_ext)"
            counts[ext] = counts.get(ext, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:30])


def _safe_rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except Exception:
        return str(path)


def _get_git_meta(repo_root: Path) -> Tuple[Optional[str], Optional[str]]:
    """Return (latest_commit_sha, latest_commit_date_iso) if repo is git."""
    try:
        repo = Repo(repo_root)
        commit = repo.head.commit
        return commit.hexsha, commit.committed_datetime.isoformat()
    except Exception:
        return None, None


def build_digest_for_gemini(
    *,
    repo_url: str,
    repo_root: Path,
    max_total_chars: int = 90_000,
    max_file_chars: int = 12_000,
    max_code_files: int = 8,
    max_doc_files: int = 5,
) -> RepoDigest:
    """Build a compact, safe digest from a repo directory.

    This is intentionally filesystem-based so it can be swapped later with
    a Moorcheh-based ContextProvider without changing the summarizer.
    """
    warnings: List[str] = []

    # Basic repo stats
    latest_sha, latest_date = _get_git_meta(repo_root)
    language_counts = _detect_languages(repo_root)

    file_count = 0
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        file_count += len([f for f in files if Path(f).suffix.lower() not in SKIP_FILE_EXTS])

    meta = RepoMeta(
        url=repo_url,
        local_path=str(repo_root),
        latest_commit_sha=latest_sha,
        latest_commit_date_iso=latest_date,
        detected_languages=language_counts,
        file_count=file_count,
    )

    docs: List[DigestDocument] = []

    # Tree
    tree = build_repo_tree(repo_root)
    docs.append(
        DigestDocument(
            id="repo_tree",
            type="tree",
            path=None,
            text=tree,
            metadata={"max_depth": 6},
        )
    )

    # Metadata summary
    meta_text = (
        f"Repo URL: {repo_url}\n"
        f"Latest commit: {latest_sha or 'unknown'}\n"
        f"Latest commit date: {latest_date or 'unknown'}\n"
        f"File count (approx): {file_count}\n"
        f"Top extensions: {list(language_counts.items())[:10]}\n"
    )
    docs.append(DigestDocument(id="repo_metadata", type="metadata", text=meta_text, metadata={}))

    # README
    readme_path = None
    for cand in README_CANDIDATES:
        p = repo_root / cand
        if p.exists() and p.is_file():
            readme_path = p
            break
    if readme_path:
        docs.append(
            DigestDocument(
                id="readme",
                type="readme",
                path=_safe_rel(readme_path, repo_root),
                text=_read_text_file(readme_path, max_file_chars),
                metadata={},
            )
        )
    else:
        warnings.append("README not found (or not in common locations).")

    # Manifests / lockfiles / config
    manifest_hits: List[Path] = []
    for fn in MANIFEST_FILES:
        p = repo_root / fn
        if p.exists() and p.is_file():
            manifest_hits.append(p)
    for p in manifest_hits:
        doc_type = "manifest"
        if p.name.endswith(".lock") or p.name.endswith("lock.json") or "lock" in p.name:
            doc_type = "lockfile"
        docs.append(
            DigestDocument(
                id=f"manifest::{p.name}",
                type=doc_type,  # type: ignore[arg-type]
                path=_safe_rel(p, repo_root),
                text=_read_text_file(p, max_file_chars),
                metadata={},
            )
        )

    # Docs (a few top files)
    doc_files: List[Path] = []
    for dname in DOC_DIR_CANDIDATES:
        dpath = repo_root / dname
        if dpath.exists() and dpath.is_dir():
            for p in sorted(dpath.rglob("*.md")):
                if p.suffix.lower() in SKIP_FILE_EXTS:
                    continue
                doc_files.append(p)
    doc_files = doc_files[:max_doc_files]
    for p in doc_files:
        docs.append(
            DigestDocument(
                id=f"docs::{_safe_rel(p, repo_root)}",
                type="docs",
                path=_safe_rel(p, repo_root),
                text=_read_text_file(p, max_file_chars),
                metadata={},
            )
        )

    # Sample code files (best-effort: pick smaller, top-level-ish)
    code_candidates: List[Tuple[int, Path]] = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            p = Path(root) / fn
            if p.suffix.lower() in SKIP_FILE_EXTS:
                continue
            if p.name in README_CANDIDATES:
                continue
            if p.name in MANIFEST_FILES:
                continue
            # only text-ish extensions
            if p.suffix.lower() in {".py", ".ts", ".js", ".tsx", ".jsx", ".java", ".go", ".rs", ".cs", ".cpp", ".c", ".h", ".php", ".rb"}:
                try:
                    code_candidates.append((p.stat().st_size, p))
                except Exception:
                    continue
    code_candidates.sort(key=lambda t: t[0])
    for _, p in code_candidates[:max_code_files]:
        docs.append(
            DigestDocument(
                id=f"code::{_safe_rel(p, repo_root)}",
                type="code",
                path=_safe_rel(p, repo_root),
                text=_read_text_file(p, max_file_chars),
                metadata={},
            )
        )

    # Enforce total cap
    total = 0
    capped_docs: List[DigestDocument] = []
    for d in docs:
        if total >= max_total_chars:
            break
        remaining = max_total_chars - total
        text = d.text
        if len(text) > remaining:
            text = text[:remaining] + "\n[TRUNCATED DIGEST]"
            warnings.append("Digest exceeded max_total_chars; truncated.")
        capped_docs.append(d.model_copy(update={"text": text}))
        total += len(text)

    return RepoDigest(meta=meta, documents=capped_docs, warnings=warnings)
