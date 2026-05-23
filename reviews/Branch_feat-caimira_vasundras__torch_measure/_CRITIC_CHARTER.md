# Critic charter for Branch feat/caimira (vasundras/torch_measure), round 1

## Scope

- **scope_mode**: pr (synthetic — `main..feat/caimira` local diff; no actual GitHub PR exists for this branch)
- **artifact_id**: `Branch feat-caimira (vasundras/torch_measure)` (filesystem-safe: `Branch_feat-caimira_vasundras__torch_measure/`)
- **round**: 1
- **is_fork**: true (parent: `aims-foundations/torch_measure`)
- **stack**: Python
- **change_size**: large (4523 lines / 19 files)
- **risk_tags**: none of the canonical 5

## Critics firing this round

| Critic | Lane | Firing? | Notes |
|---|---|---|---|
| C1 anchor_honesty | FP | **fires** | Always — mechanical anchor check against current artifact files. |
| C2 fp_catalog | FP | **fires** | Round 1 — most findings will be classified `NEW` (no prior round to map against); still required for code-bearing scope. |
| C3 fn_catalog (fresh-sweep) | FN | **fires** | Cover-to-cover sweep BEFORE reading any REVIEW_*.md per Operational Rule 3. |
| C4 external_claims | FP / FN ground truth | **fires** | Full external lane (no `--no-external-research`). Verifies CAIMIRA paper claims, PyTorch APIs, HF dataset references, Codabench platform contract refs, prior-art citations against authoritative sources. |
| C5 patch_mechanical | FP | **conditional** | Fires after Wave 1 completes IF any REVIEW_*.md contains a `suggested_fix` block; verifies before-text uniqueness, after-text syntactic validity, applicability. |
| C5_alt internal_consistency | FN | skipped | Not docs-only diff. |
| C6 adversarial_defender | FP | skipped | No canonical risk_tags. |
| C7 fn_amplifier | FN | skipped | Paired with C6. |

## Lane vocabulary

- **FP-lane critics**: C1, C2, C4 (refutation lane), C5
- **FN-lane critics**: C3, C4 (corroboration lane)
- (C6 / C7 / C5_alt not firing)

## FP/FN ratio

**4 FP-lane : 1 FN-lane** (canonical baseline; C4 is dual-lane but counted on the FP side per upstream convention). No `--single-side-c6` compounding.

## Active-width budget

- Wave 1 active width: **7 personas** dispatched concurrently.
- Wave 2 active width: **4 critics** at minimum (C1, C2, C3, C4); **5 critics** if C5 fires after Wave 1 completes.
- Wave 3 active width: 0 unless triangulation fires (Step 4 conditional). Triangulation will fire if Wave 1 + Wave 2 produce any UNCLEAR per-finding verdicts.

## Inputs each critic receives

Every critic reads (in this order):

1. `_PRIMER.md` (with `## External research` section populated from the background `ce-best-practices-researcher` dispatch if completed; or with placeholder + `_EXTERNAL_RESEARCH.md` sidecar if pending).
2. `_FULL_DIFF.patch` — the raw 4656-line diff (available for grep + line-specific lookups).
3. `_CRITIC_CHARTER.md` (this file).
4. `_ROUTING_DECISION.md` — for F6-A audit reference.
5. All `REVIEW_*.md` files in this directory (except C3, which reads them LAST per Rule 3).

## Cross-critic dependencies

- **C3 → C4**: C3's `genuine_fns` (with factual claims) feed C4's `promoted_from_fn_corroborations` array. C4 must be dispatched AFTER C3.
- **C5 → fires last** (after Wave 1 completes) since it depends on Wave 1's patch-proposal output.

## Dispatch order

1. **Wave 2 first dispatch (parallel)**: C1, C2, C3 (fires immediately after Wave 1 completes; respects Rule 3 sequencing internally).
2. **Wave 2 second dispatch (after C3 returns)**: C4 (reads C3's output).
3. **Wave 2 final dispatch (conditional)**: C5 if Wave 1 produced patches.

Practically: dispatch C1 + C2 + C3 in parallel as soon as Wave 1 completes; THEN dispatch C4 once C3 returns; THEN dispatch C5 if any Wave 1 reviewer attached a `suggested_fix`.

## Severity rubric (shared across all critics + reviewers)

- **P0**: correctness bug, security vulnerability, data-loss path, broken contract that will surface in production
- **P1**: high-impact maintainability gap, missing critical test coverage, contract violation that may not yet have surfaced
- **P2**: style/idiom/perf concern with measurable cost; doc gap that meaningfully impedes contribution
- **P3**: nice-to-have / readability / minor naming / "would be nicer if"

## Fork visibility constraint (load-bearing for synthesis comment-publication gate)

`vasundras/torch_measure` is a fork of `aims-foundations/torch_measure`. Any comment posted to the fork is publicly visible AND the upstream parent + sibling forks will see the activity. Findings whose subject is the new model files (`caimira.py`, `cold_start_lookup.py`, `llm_judge_irt.py`) may be upstream-eligible after the fork-only CS321M competition cycle ends; the reviewer may want to NOT post early to keep the upstream PR clean. Findings whose subject is the `submission/` directory are fork-only (CS321M-specific) and posting on the fork is appropriate.

The Wave 3 synthesis will distinguish three classes of finding in `_FINAL_VERDICT.md` `## Fork-specific guidance`:
1. `fork_only_caimira_branch` — the `submission/` scaffold + the `.gitignore` additions
2. `upstream_eligible` — the 3 new model files + the `__init__.py` exports
3. `applies_to_both` — anything in tests/ that exercises the new model files
