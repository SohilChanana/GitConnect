from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .schemas import RepoDigest, ViabilityGrade, ViabilitySignals


# -----------------------------
# Filesystem-derived signals
# -----------------------------

_LICENSE_CANDIDATES = {
    "license",
    "license.md",
    "license.txt",
    "copying",
    "copying.md",
    "copying.txt",
}

_LINT_CONFIG_CANDIDATES = {
    ".prettierrc",
    ".prettierrc.json",
    ".prettierrc.yml",
    ".prettierrc.yaml",
    ".prettierrc.js",
    ".eslintrc",
    ".eslintrc.json",
    ".eslintrc.yml",
    ".eslintrc.yaml",
    ".eslintrc.js",
    ".editorconfig",
    "ruff.toml",
    ".pre-commit-config.yaml",
    ".clang-format",
    ".clang-tidy",
    "rustfmt.toml",
    "checkstyle.xml",
    "pmd.xml",
}

_TEST_DIR_CANDIDATES = {"test", "tests", "__tests__", "spec", "specs"}

_DOCKER_CANDIDATES = {"dockerfile", "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"}


def _has_any_file(repo_root: Path, names_lower: set[str]) -> bool:
    try:
        for p in repo_root.iterdir():
            if p.is_file() and p.name.lower() in names_lower:
                return True
    except Exception:
        return False
    return False


def _has_ci(repo_root: Path) -> bool:
    wf = repo_root / ".github" / "workflows"
    if not wf.exists() or not wf.is_dir():
        return False
    try:
        return any(p.suffix.lower() in {".yml", ".yaml"} for p in wf.iterdir() if p.is_file())
    except Exception:
        return False


def _has_tests(repo_root: Path) -> bool:
    # quick directory check
    try:
        for p in repo_root.iterdir():
            if p.is_dir() and p.name.lower() in _TEST_DIR_CANDIDATES:
                return True
    except Exception:
        pass

    # fallback: scan a small set of files for common test naming patterns
    patterns = ("test_", "_test.", ".spec.", ".test.")
    try:
        n = 0
        for p in repo_root.rglob("*"):
            if n > 2500:
                break
            n += 1
            if not p.is_file():
                continue
            name = p.name.lower()
            if any(tok in name for tok in patterns):
                return True
    except Exception:
        return False

    return False


def derive_filesystem_signals(repo_root: Path, digest: Optional[RepoDigest] = None) -> ViabilitySignals:
    # README is usually captured in digest; still check filesystem as well.
    has_readme = False
    if digest:
        has_readme = any(d.type == "readme" for d in digest.documents)
    if not has_readme:
        try:
            has_readme = any(p.is_file() and p.name.lower().startswith("readme") for p in repo_root.iterdir())
        except Exception:
            has_readme = False

    has_license = _has_any_file(repo_root, _LICENSE_CANDIDATES)
    has_ci = _has_ci(repo_root)
    has_tests = _has_tests(repo_root)
    lint_config_present = _has_any_file(repo_root, _LINT_CONFIG_CANDIDATES)

    return ViabilitySignals(
        has_readme=has_readme,
        has_license=has_license,
        has_ci=has_ci,
        has_tests=has_tests,
        lint_config_present=lint_config_present,
    )


# -----------------------------
# Activity / time helpers
# -----------------------------

def _parse_iso_date(date_iso: str) -> Optional[_dt.date]:
    # Supports YYYY-MM-DD or full ISO timestamps
    try:
        return _dt.date.fromisoformat(date_iso[:10])
    except Exception:
        return None


def infer_activity_bucket(latest_commit_date_iso: Optional[str], *, today: Optional[_dt.date] = None) -> str:
    if not latest_commit_date_iso:
        return "unknown"
    d = _parse_iso_date(latest_commit_date_iso)
    if not d:
        return "unknown"
    today = today or _dt.date.today()
    delta_days = (today - d).days
    # Heuristic buckets
    if delta_days <= 180:
        return "active"
    if delta_days <= 540:
        return "stale"
    return "archived"


# -----------------------------
# Scoring
# -----------------------------

_DEP_RISK_PENALTY = {0: 0, 1: 8, 2: 16, 3: 25}
_SEC_RISK_PENALTY = {0: 0, 1: 10, 2: 20, 3: 35}
_MAINT_PENALTY = {0: 0, 1: 5, 2: 10, 3: 15}
_ACTIVITY_DELTA = {"active": 10, "stale": -10, "archived": -30, "unknown": -5}


def _grade(score: float) -> ViabilityGrade:
    if score >= 85:
        return "strong"
    if score >= 70:
        return "good"
    if score >= 50:
        return "caution"
    return "risky"


def _coerce_level(v: Optional[int]) -> Optional[int]:
    if v is None:
        return None
    try:
        v = int(v)
    except Exception:
        return None
    return v if 0 <= v <= 3 else None


def compute_viability_score(signals: ViabilitySignals) -> Tuple[float, Dict[str, Any]]:
    """Deterministic Viability Score v1.

    Returns:
      (score, breakdown_dict)

    The score is evidence-driven and *intentionally* conservative when key signals are unknown.
    """

    base = 40.0
    bonuses: Dict[str, float] = {}
    penalties: Dict[str, float] = {}
    unknowns: list[str] = []

    def bonus(name: str, pts: float, cond: Optional[bool]):
        if cond is True:
            bonuses[name] = pts

    def penalty(name: str, pts: float, cond: bool):
        if cond:
            penalties[name] = pts

    # Docs / UX
    bonus("readme_present", 8, signals.has_readme)
    bonus("install_instructions", 6, signals.install_instructions_present)
    bonus("run_instructions", 6, signals.run_instructions_present)
    bonus("usage_examples", 4, signals.usage_examples_present)

    # Hygiene
    bonus("tests_present", 8, signals.has_tests)
    # Test command is language-agnostic, but only meaningful if the repo actually has tests (or CI running tests).
    test_cmd_ok = signals.test_command_present is True and (signals.has_tests is True or signals.has_ci is True)
    if test_cmd_ok:
        bonuses["test_command_present"] = 4
    bonus("ci_present", 6, signals.has_ci)
    bonus("lint_or_format_config", 4, signals.lint_config_present)

    # Legal (license absence is a notable adoption risk)
    if signals.has_license is True:
        bonuses["license_present"] = 5
    elif signals.has_license is False:
        penalties["license_missing"] = 10
    else:
        unknowns.append("has_license")

    # Maintenance recency
    act = signals.recent_activity_bucket or "unknown"
    if act not in _ACTIVITY_DELTA:
        act = "unknown"
    delta = _ACTIVITY_DELTA[act]
    if delta >= 0:
        bonuses[f"activity_{act}"] = float(delta)
    else:
        penalties[f"activity_{act}"] = float(-delta)

    if signals.recent_activity_bucket == "unknown":
        unknowns.append("recent_activity_bucket")

    # Risk levels
    dep = _coerce_level(signals.dependency_risk)
    sec = _coerce_level(signals.security_red_flags)
    maint = _coerce_level(signals.maintainability_notes_count)

    if dep is None:
        unknowns.append("dependency_risk")
        penalties["dependency_risk_unknown"] = 5
    else:
        p = _DEP_RISK_PENALTY[dep]
        if p:
            penalties[f"dependency_risk_{dep}"] = float(p)

    if sec is None:
        unknowns.append("security_red_flags")
        penalties["security_risk_unknown"] = 5
    else:
        p = _SEC_RISK_PENALTY[sec]
        if p:
            penalties[f"security_red_flags_{sec}"] = float(p)

    if maint is None:
        unknowns.append("maintainability_notes_count")
        penalties["maintainability_unknown"] = 2
    else:
        p = _MAINT_PENALTY[maint]
        if p:
            penalties[f"maintainability_{maint}"] = float(p)

    score = base + sum(bonuses.values()) - sum(penalties.values())
    score = max(0.0, min(100.0, score))
    grade = _grade(score)

    breakdown: Dict[str, Any] = {
        "formula_version": "v1",
        "base": base,
        "bonuses": bonuses,
        "penalties": penalties,
        "unknowns": unknowns,
        "score": round(score, 1),
        "grade": grade,
        "signals_used": signals.model_dump(),
    }
    return round(score, 1), breakdown


def merge_signals(*, llm: ViabilitySignals, fs: ViabilitySignals, inferred_activity: Optional[str] = None) -> ViabilitySignals:
    """Merge LLM signals with filesystem signals.

    Filesystem-derived values are treated as authoritative for:
      has_readme, has_license, has_ci, has_tests, lint_config_present
    """

    fs_preferred = {
        "has_readme",
        "has_license",
        "has_ci",
        "has_tests",
        "lint_config_present",
    }

    merged = llm.model_dump()
    fs_dict = fs.model_dump()
    for k, v in fs_dict.items():
        if k in fs_preferred:
            merged[k] = v
        else:
            if merged.get(k) is None and v is not None:
                merged[k] = v

    if (merged.get("recent_activity_bucket") in (None, "unknown")) and inferred_activity:
        merged["recent_activity_bucket"] = inferred_activity

    return ViabilitySignals(**merged)
