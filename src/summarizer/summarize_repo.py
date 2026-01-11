from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from .gemini_client import gemini_generate_json
from .prompts import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_PROMPT_TEMPLATE
from .repo_digest import build_digest_for_gemini
from .schemas import RepoDigest, SummaryReport
from .cache import cache_path, load_cached_json, write_cache_json
from .viability_score import (
    compute_viability_score,
    derive_filesystem_signals,
    infer_activity_bucket,
    merge_signals,
)


def digest_to_text(digest: RepoDigest) -> str:
    """Flatten digest documents into a single string for the LLM."""
    parts = []
    for doc in digest.documents:
        header = f"\n=== {doc.type.upper()} :: {doc.path or doc.id} ===\n"
        parts.append(header + doc.text)
    if digest.warnings:
        parts.append("\n=== WARNINGS ===\n" + "\n".join(digest.warnings))
    return "\n\n".join(parts)


def summarize_repo_once(
    *,
    repo_url: str,
    repo_root: Path,
    user_prompt: str = "",
    use_grounding: bool = True,
    model: Optional[str] = None,
) -> SummaryReport:
    """One-shot: build digest, call Gemini, parse into SummaryReport.

    After Gemini returns a structured report (summary + signals), we compute a
    deterministic Viability Score (v1) locally from the extracted signals.
    """
    digest = build_digest_for_gemini(repo_url=repo_url, repo_root=repo_root)

    # Cache per-commit so the same repo state yields the same final score/report.
    commit_sha = (digest.meta.latest_commit_sha or "").strip()
    if commit_sha:
        cpath = cache_path(
            repo_url=repo_url,
            commit_sha=commit_sha,
            model=model,
            use_grounding=use_grounding,
            user_prompt=user_prompt,
        )
        cached = load_cached_json(cpath)
        if cached is not None:
            try:
                return SummaryReport.model_validate(cached)
            except Exception:
                # Ignore invalid cache and proceed.
                pass
    digest_text = digest_to_text(digest)

    user_prompt_full = SUMMARY_USER_PROMPT_TEMPLATE.format(
        user_prompt=user_prompt.strip(),
        digest_text=digest_text,
    )

    raw = gemini_generate_json(
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        user_prompt=user_prompt_full,
        model=model,
        use_grounding=use_grounding,
    )

    # Parse into our schema; if it fails, do a single repair pass.
    try:
        report = SummaryReport.model_validate(raw)
    except Exception:
        print("[gitconnect-summary] JSON invalid; running one repair pass...", file=sys.stderr)
        repair_user_prompt = (
            "Fix the previous response into STRICT valid JSON.\n"
            "Return ONLY one JSON object that matches the SummaryReport schema.\n"
            "Required top-level keys:\n"
            "repo_url, summary_bullets, summary_paragraph, tech_stack, viability_verdict, viability_reasons, "
            "viability_signals, viability_evidence, citations, confidence, limitations\n"
            "No markdown. No trailing commas.\n\n"
            f"Previous response:\n{json.dumps(raw, ensure_ascii=False)}"
        )

        raw2 = gemini_generate_json(
            system_prompt=SUMMARY_SYSTEM_PROMPT,
            user_prompt=repair_user_prompt,
            model=model,
            use_grounding=use_grounding,
        )
        report = SummaryReport.model_validate(raw2)

    # Always annotate the source of the digest
    report.retrieval_provider = "filesystem"

    # Merge digest warnings into limitations (dedupe)
    if digest.warnings:
        report.limitations = list(dict.fromkeys(report.limitations + digest.warnings))

    # Compute deterministic Viability Score (v1)
    fs_signals = derive_filesystem_signals(repo_root=repo_root, digest=digest)
    inferred_activity = infer_activity_bucket(digest.meta.latest_commit_date_iso)
    merged_signals = merge_signals(llm=report.viability_signals, fs=fs_signals, inferred_activity=inferred_activity)

    score, breakdown = compute_viability_score(merged_signals)
    report.viability_signals = merged_signals
    report.viability_score = score
    report.viability_score_breakdown = breakdown

    # Persist cache (non-overwriting) keyed by commit + prompt/model settings.
    if commit_sha:
        try:
            write_cache_json(cpath, report.model_dump())
        except Exception:
            pass

    return report
