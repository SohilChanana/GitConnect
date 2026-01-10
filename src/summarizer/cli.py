from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from src.github_fetcher import GitHubFetcher, GitHubFetchError

from .summarize_repo import summarize_repo_once


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gitconnect-summary",
        description="One-shot repo summary + viability verdict using Gemini.",
    )
    parser.add_argument("repo_url", help="GitHub repo URL (public or private with token)")
    parser.add_argument("--prompt", default="", help="Extra user prompt for the summary")
    parser.add_argument(
        "--no-grounding",
        action="store_true",
        help="Disable Google Search grounding (repo-only summary).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override Gemini model (else uses GEMINI_MODEL env var).",
    )
    args = parser.parse_args(argv)

    try:
        with GitHubFetcher() as fetcher:
            repo_path = fetcher.clone_repository(args.repo_url)
            report = summarize_repo_once(
                repo_url=args.repo_url,
                repo_root=repo_path,
                user_prompt=args.prompt,
                use_grounding=not args.no_grounding,
                model=args.model,
            )
            print(report.model_dump_json(indent=2))
        return 0
    except GitHubFetchError as e:
        print(f"[gitconnect-summary] GitHub fetch error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[gitconnect-summary] Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
