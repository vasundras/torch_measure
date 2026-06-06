# Routing decision for Branch feat/caimira (vasundras/torch_measure)

## Inputs observed

- **scope_mode**: `pr` (synthetic — no GitHub PR exists; treating `main..feat/caimira` local diff as PR-shaped)
- **ticket_type**: `feature` (commit-message prefix "Add ..." across 4 commits; no explicit type tag)
- **change_size**: `large` (4523 insertions across 19 files; PR-bucket threshold `<5000` = large, `>=5000` = massive — we are just under the massive cutoff)
- **change_classification**: `mixed` (Python source `.py` + tests + shell `.sh` + docs `.md` + notebook `.ipynb` + manifest `.txt` + gitignore — code-dominant)
- **risk_tags**: **none** of the canonical 5 (no `auth`, `security`, `payments`, `data-migration`, `privacy` keywords match in PR labels, file paths, or commit messages)
  - **Note (informational, NOT a routing trigger):** the submission code IS competition-bound and has a known history of silent regression per the sibling repo's CLAUDE.md 2026-05-22 transfer postmortem. The canonical risk-tag inference does NOT cover this class. If the reviewer wants C6+C7 defense-in-depth, re-invoke with `--always-on-meta` or `--single-side-c6`.
- **stack**: `python` (extensions `.py` + marker `pyproject.toml` + `torch_measure` Python package)
- **round**: 1 (first-pass — no prior `REVIEW_*.md` siblings in this directory)
- **is_fork**: **true** (parent: `aims-foundations/torch_measure`); see `_PRIMER.md` `## Source` and "Reviewer-side guidance: this is a fork"

## Rule fired

**R7** — `pr`, `large`, `code`/`mixed`, no canonical risk_tags. Selects:
- 6 always-on personas (correctness, testing, maintainability, project-standards, agent-native, learnings-research)
- + stack-specific (`kieran-python`)
- + `adversarial-reviewer` (large-diff conditional per ce-code-review persona-selection rules)
- Full 5-critic charter (C1+C2+C3+C4+C5)
- No paired C6+C7 (no risk_tags trigger)

**Persona narrowing applied (parent decision, recorded for audit):**
- Dropped `agent-native` from the dispatch set. Justification: the diff is library code + a Codabench competition submission entry point. The skill's `ce-agent-native-reviewer` covers "any action a user can take, an agent can also take" — UI / CLI / system-prompt agent parity. The diff has no UI surface, no agent-facing tool definitions, no system prompts. Dropping `agent-native` reduces wave-budget without dropping a relevant lens. The parent owns this narrowing per `adaptive-routing.md` R7's persona-selection authority.

## Wave 1 personas selected

- **ce-correctness-reviewer**: 4523 new lines of substantive logic across CAIMIRA + EB + Codabench predict path; correctness is the dominant axis.
- **ce-testing-reviewer**: ~1207 LoC of new tests; verify coverage, weak assertions, brittle implementation-coupled tests, missing edge cases.
- **ce-maintainability-reviewer**: 3 new top-level public classes + 1 new public helper exported from `torch_measure.models`; surface area is large, abstractions cross-cut.
- **ce-project-standards-reviewer**: CONTRIBUTING.md mandates Sphinx docs for new public APIs — diff adds 3 new public classes + 1 helper with **NO** `docs/source/` additions. Also commit-message style violation (multi-line vs single-line upstream preference). Also lint workflow does not cover `submission/`.
- **ce-learnings-researcher**: Look up parent CS321M `docs/solutions/` for prior patterns: D-3 distinguishable-defensive-fallbacks, D-7 binarization, D-9 stress-test gate, silent-NCF-head-load post-mortem, labeling NaN-bomb post-mortem, two-tier-error-reporting pattern. Cross-reference whether this diff respects each.
- **ce-kieran-python-reviewer**: Python stack persona. Strict bar for type hints, Pythonic clarity, and maintainability.
- **ce-adversarial-reviewer**: Large diff (4523 lines) + new public-library surface + competition-bound code with prior known-failure history → adversarial defense-in-depth. Actively construct failure scenarios.

Active width: **7 personas** (well under the 10-cap).

Dispatched in **parallel via single multi-Agent message** per Operational Rule for active-width-budgeted parallel dispatch.

## Wave 2 critics selected

- **C1 anchor_honesty**: always
- **C2 fp_catalog**: always for code-bearing scopes (round 1 has no prior findings, so will mostly produce `NEW` verdicts)
- **C3 fn_catalog**: always for code-bearing scopes — fresh-sweep methodology
- **C4 external_claims**: always; FULL external lane (no `--no-external-research` flag passed). Note: parent assembled the primer's `## External research` section by dispatching `ce-best-practices-researcher` in background during primer assembly (Step 7); C4's job is to VERIFY claims, not to re-do research.
- **C5 patch_mechanical**: fires if ≥1 reviewer proposed a `suggested_fix` patch block (reviewers are allowed to propose patches; dispatch C5 conditionally after Wave 1 completes)
- **C6 adversarial_defender**: SKIPPED — no canonical risk_tags fired
- **C7 fn_amplifier**: SKIPPED — paired with C6

Active width: **4–5 critics** (C1, C2, C3, C4 fire; C5 may or may not fire depending on Wave 1 patch-proposal output).

## Overrides applied

- None. No `--always-on-meta`, no `--single-side-c6`, no `--no-external-research`, no `--proceed-on-massive` (diff is large, not massive).

## Audit hooks (informational — NOT machine-parsed)

- **F6-A check** (synthesis Step 3): the **MACHINE-PARSED** source is the `## Wave 2 critics selected` section above. Parent will rerun the F6-A parse-pseudocode from `synthesis.md` § Step 3 against this section after the critic wave completes. Expected set: `{C1, C2, C3, C4}` plus `C5` IF Wave 1 proposes any patches. The pseudocode filters `\bskipped\b` lines so C6 / C7 will not be expected.
- **Synthesis Verifier C** (synthesis Step 4): triangulation fires conditionally — at least one of (UNCLEAR ≥1, round N≥2 overturned, high-risk, `--always-on-meta`). Round 1 + no canonical high-risk + no `--always-on-meta` → triangulation fires ONLY if Wave 1 produces UNCLEAR findings (likely for the CAIMIRA paper-faithfulness claims and the D-9 sub-gate (c) FAIL interpretation).

## Massive-warning result

None. Diff is `large` (4523 lines), not `massive` (≥5000). No prompt fired.

## Round-numbering context

- First-pass (round 1).
- Artifact-id directory: `reviews/Branch_feat-caimira_vasundras__torch_measure/` — created during Round 0.
- No `_PRIOR_FINDINGS_INDEX.md` (round 1, no prior siblings).
