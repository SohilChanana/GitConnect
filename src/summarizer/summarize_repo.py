from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .gemini_client import gemini_generate_json
from .prompts import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_PROMPT_TEMPLATE
from .repo_digest import build_digest_for_gemini
from .schemas import RepoDigest, SummaryReport


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
    """One-shot: build digest, call Gemini, parse into SummaryReport."""
    digest = build_digest_for_gemini(repo_url=repo_url, repo_root=repo_root)
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
        # Ask the model to fix its own output into STRICT schema JSON.
        print("[gitconnect-summary] JSON invalid; running one repair pass...", file=sys.stderr)
        repair_user_prompt = (
            "Fix the previous response into STRICT valid JSON.\n"
            "Return ONLY one JSON object with exactly these keys:\n"
            "repo_url, summary_bullets, summary_paragraph, tech_stack, viability_verdict, viability_reasons, citations, confidence, limitations, retrieval_provider, viability_score, viability_score_breakdown\n"
            "tech_stack must contain keys: languages, frameworks, libraries, tools, platforms (arrays).\n"
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

    report.retrieval_provider = "filesystem"
    if digest.warnings:
        report.limitations = list(dict.fromkeys(report.limitations + digest.warnings))
    return report
