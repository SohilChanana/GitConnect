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


class SummaryReport(BaseModel):
    repo_url: str
    summary_bullets: List[str] = Field(default_factory=list)
    summary_paragraph: str = ""
    tech_stack: Dict[str, Any] = Field(default_factory=dict)
    viability_verdict: ViabilityVerdict = "unknown"
    viability_reasons: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    limitations: List[str] = Field(default_factory=list)

    # Room for Phase 3 + later scoring
    retrieval_provider: Optional[str] = None  # e.g., "filesystem" | "moorcheh"
    viability_score: Optional[float] = None
    viability_score_breakdown: Dict[str, Any] = Field(default_factory=dict)
