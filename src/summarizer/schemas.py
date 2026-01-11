from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


DigestDocType = Literal[
    "readme",
    "docs",
    "manifest",
    "lockfile",
    "config",
    "code",
    "tree",
    "metadata",
    "other",
]


class DigestDocument(BaseModel):
    """A single chunk of context to send to an LLM."""

    id: str
    type: DigestDocType
    path: Optional[str] = None
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RepoMeta(BaseModel):
    url: str
    local_path: str
    default_branch: Optional[str] = None
    latest_commit_sha: Optional[str] = None
    latest_commit_date_iso: Optional[str] = None
    detected_languages: Dict[str, int] = Field(default_factory=dict)
    file_count: int = 0


class RepoDigest(BaseModel):
    """Structured digest built from a repo (filesystem based)."""

    meta: RepoMeta
    documents: List[DigestDocument] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class Citation(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None


ViabilityVerdict = Literal["viable", "risky", "obsolete", "unknown"]
ActivityBucket = Literal["active", "stale", "archived", "unknown"]
ViabilityGrade = Literal["strong", "good", "caution", "risky"]


class ViabilitySignals(BaseModel):
    """Normalized signals used to compute a deterministic Viability Score (v1)."""

    # Mostly file/structure signals (some are also auto-detected from filesystem)
    has_readme: Optional[bool] = None
    has_license: Optional[bool] = None
    has_ci: Optional[bool] = None
    has_tests: Optional[bool] = None
    lint_config_present: Optional[bool] = None

    # Semantic signals (typically LLM-derived from README/docs)
    install_instructions_present: Optional[bool] = None
    run_instructions_present: Optional[bool] = None
    usage_examples_present: Optional[bool] = None
    test_command_present: Optional[bool] = None

    # Risk metrics (0=none, 3=severe); may be null if unknown
    dependency_risk: Optional[int] = Field(default=None, ge=0, le=3)
    security_red_flags: Optional[int] = Field(default=None, ge=0, le=3)
    maintainability_notes_count: Optional[int] = Field(default=None, ge=0, le=3)

    # Maintenance recency; may be inferred from latest commit date if unknown
    recent_activity_bucket: ActivityBucket = "unknown"


class ViabilityEvidence(BaseModel):
    """Short, digest-grounded evidence snippets supporting signals/verdict."""

    claim: str
    path: Optional[str] = None
    snippet: Optional[str] = None


class SummaryReport(BaseModel):
    repo_url: str
    summary_bullets: List[str] = Field(default_factory=list)
    summary_paragraph: str = ""
    tech_stack: Dict[str, Any] = Field(default_factory=dict)
    viability_verdict: ViabilityVerdict = "unknown"
    viability_reasons: List[str] = Field(default_factory=list)
    viability_signals: ViabilitySignals = Field(default_factory=ViabilitySignals)
    viability_evidence: List[ViabilityEvidence] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    limitations: List[str] = Field(default_factory=list)

    # Room for Phase 3 + later scoring
    retrieval_provider: Optional[str] = None  # e.g., "filesystem" | "moorcheh"
    viability_score: Optional[float] = None
    viability_score_breakdown: Dict[str, Any] = Field(default_factory=dict)
