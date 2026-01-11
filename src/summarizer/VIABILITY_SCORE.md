# Viability Score (v1)

This repository uses a **deterministic** scoring function to produce a `viability_score` (0–100) and a `viability_score_breakdown`.
Gemini provides a structured `viability_signals` object (plus short `viability_evidence` snippets), and **the host computes the score locally**.

To keep results consistent across repeated runs, the summarizer also **caches the full computed report per commit SHA** (see "Caching").

## Signals

`viability_signals` fields:

### File / hygiene signals (some auto-detected from filesystem)
- `has_readme` (bool)
- `has_license` (bool)
- `has_ci` (bool) — GitHub Actions workflows present
- `has_tests` (bool) — tests folder / common test filename patterns
- `lint_config_present` (bool) — common lint/format config files

### Semantic signals (LLM-derived from README/docs when available)
- `install_instructions_present` (bool)
- `run_instructions_present` (bool)
- `usage_examples_present` (bool)
- `test_command_present` (bool)

Signal definitions are **language-agnostic**. For example, `test_command_present=true` means there is an explicit runnable test command shown in docs or CI (e.g., npm test, mvn test, ./gradlew test, cargo test, pytest). It does not assume any language-specific tooling.

### Risk levels (0–3)
- `dependency_risk` (int 0–3, or null if unknown)
- `security_red_flags` (int 0–3, or null if unknown)
- `maintainability_notes_count` (int 0–3, or null if unknown)

### Activity bucket
- `recent_activity_bucket` ∈ {`active`, `stale`, `archived`, `unknown`}

The pipeline will try to **infer** `recent_activity_bucket` from `latest_commit_date_iso`:
- `active`  ≤ 180 days
- `stale`   ≤ 540 days
- `archived` > 540 days (heuristic label; not necessarily GitHub “archived”)

## Formula

Start at **40** and apply bonuses and penalties.

### Bonuses
Docs / UX:
- `readme_present` +8
- `install_instructions` +6
- `run_instructions` +6
- `usage_examples` +4

Engineering hygiene:
- `tests_present` +8
- `test_command_present` +4 (only if tests or CI exist; see notes)
- `ci_present` +6
- `lint_or_format_config` +4

Legal:
- `license_present` +5

Activity:
- `activity_active` +10

### Penalties
Legal:
- `license_missing` −10

Activity:
- `activity_stale` −10
- `activity_archived` −30
- `activity_unknown` −5

Risk levels:
- Dependency risk penalties:
  - 0 → 0
  - 1 → −8
  - 2 → −16
  - 3 → −25
- Security red flags penalties:
  - 0 → 0
  - 1 → −10
  - 2 → −20
  - 3 → −35
- Maintainability notes penalties:
  - 0 → 0
  - 1 → −5
  - 2 → −10
  - 3 → −15

Unknown handling (conservative):
- If `dependency_risk` is null → −5
- If `security_red_flags` is null → −5
- If `maintainability_notes_count` is null → −2
- If `has_license` is null → no bonus/penalty here (but it will show up in `unknowns`)

Finally:
- Clamp to **[0, 100]**
- Round to 1 decimal place

### Notes on fairness

- The score does **not** depend on programming language. It rewards *reproducibility and maintenance signals* that are relevant in any ecosystem.
- `lint_config_present` is a **small bonus only** (never a penalty), because linting conventions vary by ecosystem.
- `test_command_present` is only awarded when it is meaningful: the repo must also have tests or CI present.

## Caching

The summarizer caches a report per commit so that re-running the same repo state does not overwrite the score/report.

- Cache key includes: repo URL, commit SHA, model, grounding flag, and user prompt.
- Default cache location:
  - $HOME/.gitconnect/cache
  - Override with environment variable: GITCONNECT_CACHE_DIR

## Grade mapping

- **Strong**: 85–100
- **Good**: 70–84.9
- **Caution**: 50–69.9
- **Risky**: 0–49.9

## Notes / Intended use

- v1 is a **repo health + adoption** score, meant for “should I use/extend this repo?”
- A future v2 can add Moorcheh-derived graph metrics (breakage radius, unresolved symbols, cycles, hotspot analysis, etc.).
