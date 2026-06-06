# Final verdict for Branch feat/caimira (vasundras/torch_measure), round 1

## Verdict

**Ready with fixes** (fork-only Codabench submission path) | **Not ready** (upstream PR to `aims-foundations/torch_measure`)

The two-track interpretation is load-bearing:

- For **fork-only Codabench-bound use** of `submission/` (CS321M competition 15934): there is no merge gate — the artifact is "dormant" (not yet uploaded). The 14+ P1 findings reduce to **must-fix-before-real-upload**: the D-9 transfer-audit gate must pass on a full-trained artifact (currently FAILed on smoke), the NaN-to-0.9999 silent path (CRIT-ADV-1/8) and the Platt-shift calibration mismatch (CRIT-ADV-5) are operationally consequential, and the train.py multi-condition row collapse (CRIT-CORR-1) means CAIMIRA trains on a randomized condition subset. Fix-or-acknowledge these before consuming a daily Codabench quota slot. **Not blocking for further iteration on the fork branch.**

- For **upstream PR to `aims-foundations/torch_measure`** (of the 3 new model files + `__init__.py` additions): two CONTRIBUTING.md violations are hard blockers (CRIT-STD-1 missing Sphinx entries; CRIT-STD-2 missing doctest `Examples` on `LLMJudgeIRT` + `build_difficulty_prompt`), plus CRIT-MAINT-3 (`ColdStartLookupPredictor` breaks `Predictor` base contract), plus CRIT-ADV-2 (the pre-existing `__all__` TabPFNPredictor missing-import that this branch's `__init__.py` edits perpetuate — should be remediated in the same PR), plus the CAIMIRA paper-faithfulness questions in §"What was not validated" below. **Not ready.**

## Confidence

**Moderate**

Why not High: triangulation did not fire (no UNCLEAR per-finding classifications, no high-risk routing, no `--always-on-meta`). Phase D external-context re-validation also did not fire. Per `synthesis.md` rubric, the confidence ceiling without triangulation is Moderate.

Why not Low: F6-A/B/C all PASSed. Phase A bookend: no primer drift. C1: 0 NOT-FOUND anchors across 80 findings. C2: 0 ADDRESSED-FP across 80 findings (every "X is missing" was mechanically verified absent). C3: fresh-sweep methodology completed per Rule 3. C4: 9 VERIFIED claims via independent commands (including the load-bearing `_logit(NaN)=16.118` empirical reproduction). Every finding has ≥2-lane critic coverage.

## Counts

- Findings surfaced by Wave 1: **80** (CORR=6, TEST=11, MAINT=10, STD=14, LEARN=9, PY=16, ADV=14)
- Plus C3 GENUINE-FN promotions: **3** (FN-V1-1, FN-V1-11, FN-V1-12)
- **Grand total candidates:** **83**
- After dedup (per C2's summary): 3 dedup pairs collapse to 80 distinct findings + 3 PROMOTED-FROM-FN = **80 + 3 = 83 → effectively 77 distinct concerns** after dedup
- **UPHELD: 80** (every reviewer finding survives C1+C2+C4+C5; 2 of them are SPLIT-with-patch-broken)
- **SUPPRESSED (FP): 0**
- **SPLIT: 2** (CRIT-PY-6, CRIT-PY-13 — patch shape broken; underlying findings stand)
- **PROMOTED FROM FN: 3** (FN-V1-1, FN-V1-11, FN-V1-12 — all C4-CORROBORATED)
- **UNCLEAR: 0**

### Severity distribution (P0/P1/P2/P3) — UPHELD findings

| Severity | Count | Top examples |
|---|---|---|
| P0 | **0** | — |
| P1 | **~15** | NaN→0.9999 silent path (CRIT-ADV-1/8/CORR-2), Platt-shift calibration mismatch (CRIT-ADV-5), train.py multi-condition collapse (CRIT-CORR-1), missing Sphinx entries (CRIT-STD-1), missing doctest Examples (CRIT-STD-2), Predictor base-contract drift (CRIT-MAINT-3), CAIMIRA↔CAIMIRALite forward-parity untested (CRIT-TEST-4), train.py untested (CRIT-TEST-1), model.py production path never exercised in CI (CRIT-TEST-2), labeling.py stateful globals untested (CRIT-TEST-3), 6 duplicated symbols with no parity test (CRIT-MAINT-1), PAIEC-PREDICT-002 mis-cited (CRIT-LEARN-6), `__all__` TabPFNPredictor missing import (CRIT-ADV-2; pre-existing), `float(label)` raises on non-numeric reveal (CRIT-ADV-11), LLMJudgeIRT NaN-from-judge bypass (CRIT-CORR-4) |
| P2 | ~24 | EB priors lack provenance (CRIT-MAINT-5), `or`-chain falsy fallthrough on 0.0 priors (CRIT-CORR-5/ADV-6), `_lookup_p` private access (CRIT-PY-1), submission/ invisible to CI lint (CRIT-STD-3), D-9 gate coverage (CRIT-LEARN-5), case-insensitive dict overwrite (CRIT-ADV-7), Platt cache by `len(labeled)` stale-vulnerable (CRIT-ADV-10), SimHash regex strips non-ASCII (FN-V1-1), Platt-shift `mean_y` boundary semantics, build_zip.sh awk filename-with-spaces (CRIT-ADV-4), metadata_bonus 256-cap inversion (CRIT-ADV-3), … |
| P3 | ~41 | Style consistency, type-hint tightening, doctest coverage, test marker hygiene, naming, … |

## Per-finding table (P1 findings — full set)

> Full per-finding records are in the 7 `REVIEW_*.md` files. Critic verdicts are in the 5 `CRITIC_*.md` files. The table below is a P1-only roll-up for the merge decision; lower-severity findings remain in the individual review files.

| ID | Source | Title | Sev | Status | Scope | Critics voting |
|---|---|---|---|---|---|---|
| CRIT-ADV-1 / -8 / CORR-2 | adversarial + correctness | NaN-input to `_logit` silently returns `±16.118` (verified empirically Python 3.11); single NaN in CAIMIRA forward → confident `0.9999` for every in-vocab item; D-3 invariant broken | **P1** | UPHELD | both (3 files define _logit) | C1:HONEST C2:NOT-ADDRESSED C3:CORROBORATED C4:VERIFIED C5:APPLIES-CLEAN |
| CRIT-ADV-5 | adversarial | Per-benchmark Platt-shift is fit on EB-only logits but applied to the CAIMIRA-EB blended logit; calibration mismatch is structurally the same class as the 2026-05-22 transfer-postmortem signature | **P1** | UPHELD | fork-only (submission/) | C1:HONEST C2:NOT-ADDRESSED C3:CORROBORATED C4:N/A (design claim) |
| CRIT-CORR-1 | correctness | `submission/train.py` `_build_wide_response` collapses multi-condition `(s,i)` rows to a single non-deterministic label via wide-form overwrite; CAIMIRA trains on randomized condition subset while EB sees the full distribution | **P1** | UPHELD | fork-only | C1:HONEST C2:NOT-ADDRESSED C3:CORROBORATED |
| CRIT-CORR-4 | correctness | `LLMJudgeIRT.predict` try/except on `judge_fn` doesn't catch a NaN-returning judge; NaN propagates through `_sigmoid`+min/max to the return → contract violation | **P1** | UPHELD | upstream (but NOT-FOR-PRODUCTION module) | C1:HONEST C2:NOT-ADDRESSED C3:CORROBORATED C5:APPLIES-CLEAN |
| CRIT-MAINT-1 | maintainability | 6 top-level symbols (`_logit`, `_sigmoid`, `_clip`, `_DEFAULT_PROVIDER_PREFIXES`, `parse_subject_name`, `resolve_subject_name`) duplicated character-for-character between `submission/caimira_lite.py` and `src/torch_measure/models/cold_start_lookup.py`; no parity test | **P1** | UPHELD | both | C1:HONEST C2:NOT-ADDRESSED C3:CORROBORATED C4:VERIFIED |
| CRIT-MAINT-3 | maintainability | `ColdStartLookupPredictor` exported alongside Rasch/TwoPL/AmortizedIRT but does NOT inherit `Predictor` base; `predict(record: dict, labeled)` takes string-keyed dict, not `(subject_idx, item_idx)` tensor query of `Predictor.predict(query)` | **P1** | UPHELD | upstream-eligible | C1:HONEST C2:NOT-ADDRESSED |
| CRIT-TEST-1 | testing | `submission/train.py` (458 LoC; produces 3 load-bearing artifacts) has ZERO tests | **P1** | UPHELD | fork-only | C1:HONEST C2:NOT-ADDRESSED |
| CRIT-TEST-2 | testing | `submission/model.py` production path (CAIMIRA+EB blend, OOV fallback, Platt shift) is NEVER executed in CI — every test under LOCAL_SMOKE=1 short-circuits at line 220 to fixed `0.5` | **P1** | UPHELD | fork-only | C1:HONEST C2:NOT-ADDRESSED |
| CRIT-TEST-3 | testing | `submission/labeling.py` stateful globals + reservoir sampling + `_MAX_STRATA` cap + empty-tokens branch are untested | **P1** | UPHELD | fork-only | C1:HONEST C2:NOT-ADDRESSED C3:CORROBORATED |
| CRIT-TEST-4 | testing | `TestCAIMIRALiteStateDictParity` tests STORAGE parity (state_dict key/value match) but never FORWARD-OUTPUT parity between upstream-CAIMIRA's `predict()` and lite's `caimira_logit()` after save+load — the exact gap the silent-NCF-head-load post-mortem says structural tests must close | **P1** | UPHELD | both | C1:HONEST C2:NOT-ADDRESSED C3:CORROBORATED |
| CRIT-STD-1 | project-standards | No Sphinx entries in `docs/source/api/models.rst` for the 4 new public symbols (`CAIMIRA`, `ColdStartLookupPredictor`, `LLMJudgeIRT`, `build_difficulty_prompt`) → CONTRIBUTING.md:36 violation | **P1** | UPHELD | upstream-eligible | C1:HONEST C2:NOT-ADDRESSED C4:VERIFIED |
| CRIT-STD-2 | project-standards | `LLMJudgeIRT` class and `build_difficulty_prompt` function lack `Examples` doctest blocks → CONTRIBUTING.md:35 violation | **P1** | UPHELD | upstream-eligible | C1:HONEST C2:NOT-ADDRESSED |
| CRIT-LEARN-6 | learnings-research | `submission/model.py:17-27` + `submission/README.md:135-136` claim module-init failures surface as `[PAIEC-PREDICT-002]`; canonical pattern doc explicitly says PAIEC-PREDICT-002 is for `predict()` output violations, not module-init (which falls to generic fallback) | **P1** | UPHELD | fork-only (docs accuracy) | C1:HONEST C2:NOT-ADDRESSED C4:VERIFIED |
| CRIT-ADV-2 | adversarial | `__all__` declares `TabPFNPredictor` at `__init__.py:47` but the symbol is NEVER imported in the file → `from torch_measure.models import *` AttributeError. Pre-existing in upstream merge but this branch's `__init__.py` edits perpetuate without remediation | **P1** | UPHELD | upstream-eligible | C1:HONEST C2:NOT-ADDRESSED C4:VERIFIED |
| CRIT-ADV-11 | adversarial | `EB.fit_platt` line ~431 does `float(ex["label"])`; one corrupted reveal with a non-numeric `label` raises ValueError, breaking the entire round mid-stream | **P1** | UPHELD | fork-only | C1:HONEST C2:NOT-ADDRESSED |

## Promoted-from-FN (3 findings)

| ID | Title | Sev | Status | Critics voting |
|---|---|---|---|---|
| FN-V1-1 | `submission/labeling.py:43` `_TOKEN_RE = r'[a-z0-9]+'` strips all non-ASCII content; CJK/Cyrillic/Arabic items collapse to identical `<empty>` SimHash signature → diversity score = 0 between non-Latin items | P2 | PROMOTED FROM FN | C3:GENUINE-FN C4:CORROBORATED |
| FN-V1-11 | `CAIMIRALite.skill = nn.Parameter(torch.zeros(...))` at submission/caimira_lite.py:221 vs upstream `nn.Parameter(torch.randn(...) * 0.01)` at src/torch_measure/models/caimira.py:126 — divergent untrained behavior; masked by state-dict load in production but `test_caimira_logit_returns_float` passes trivially on zeros-skill | P3 | PROMOTED FROM FN | C3:GENUINE-FN C4:CORROBORATED |
| FN-V1-12 | `tests/test_models/test_caimira.py:127-149` `test_predict_matches_manual_caimira_equation` is tautological — both test sides invoke the same `compute_item_params()` on the same model; verifies self-consistency not paper-equation faithfulness | P3 | PROMOTED FROM FN | C3:GENUINE-FN C4:CORROBORATED |

## Why this verdict

- **Anchor honesty is uniformly high** (C1: 0 NOT-FOUND across 80) — the reviewers were not fabricating.
- **Zero false positives across 80 findings** (C2: 0 ADDRESSED) — every "X is missing" claim verified absent at the cited file:line. This is unusual and reflects strong reviewer discipline, but it also means the synthesis cannot trim findings via FP-suppression; the full P1 set goes forward.
- The diff contains substantive new public-API surface (CAIMIRA, ColdStartLookupPredictor, LLMJudgeIRT) with **strong adherence to most modern best practices** (PyTorch state_dict pattern, weights_only=True under 2.6+, HF SHA pinning, register_buffer for inference invariants, NumPy docstrings on 2 of 3 new classes, D-3 / no-broad-except discipline, SimHash canonical port, EB shrinkage standard form) but with **two CONTRIBUTING.md hard violations** for upstream-PR scope (missing Sphinx + missing doctest Examples).
- The submission/ directory is **operationally risky** under the 2026-05-22 transfer-postmortem context: 14 of 15 P1 findings concentrate in submission/, and the D-9 gate has not been full-trained-passed. The CAIMIRA+EB hybrid is a STRUCTURAL counter to the postmortem's failure signature but is empirically UNVERIFIED on the hidden leaderboard.
- The `_logit(NaN) = 16.118` silent path (verified empirically by both adversarial reviewer and C4) is the **single most consequential** P1 — it's the exact "silent-NCF-head-load-failure" class the project's own design-pattern doc was written to defend against, and it survives D-3 discipline only because D-3 forbids `try/except → 0.5`, not because the math itself is NaN-safe.

## What was rejected

**Nothing.** Zero SUPPRESSED-FP findings.

This is unusual for a multi-wave review. The two paths to a SUPPRESSED-FP verdict are:
1. C1 NOT-FOUND (anchor missing): 0 occurrences.
2. C2 ADDRESSED-FP (current artifact addresses concern): 0 occurrences.

The synthesis accepts this outcome at face value but notes that **the absence of FP suppressions could itself be a confidence-warning** — either every reviewer was unusually rigorous, OR the critics didn't search hard enough for refutations. Phase A bookend + F6-A/B/C PASSing + the specific empirical refutation attempts in C4 (9 VERIFIED via independent commands) all argue for the former interpretation.

## What remains uncertain

- **CAIMIRA paper-faithfulness on zero-centering protocol (paper Eq. 7) and `skill_reg` coefficient (paper Section 4.3 λ_s=1e-5 vs diff default 1e-4).** The external research subagent surfaced these as plausible deviations; C4 attempted to fetch arXiv:2410.06524 directly and was permission-denied. UNVERIFIABLE-IN-SESSION on the paper-text side. The user should spot-check the paper PDF directly before opening an upstream PR or before claiming "paper-faithful" in any external-facing doc.
- **Hidden-leaderboard transfer behavior.** The CAIMIRA+EB hybrid has never been run against a real Codabench submission. The D-9 transfer-audit gate failed on a smoke-trained artifact (degenerate constant-0.604 predictor that never exercised the CAIMIRA logit path). UNVERIFIABLE-IN-SESSION; verifiable only by actually consuming a quota slot.
- **Empirical justification for `_BLEND_LAMBDA=0.6`.** Documented as "conservative default per Wave-2 Lane-B recommendation" in submission/README.md; the D-9 ablation candidates `{0.3, 0.5, 0.7, 0.9}` are listed but not run.
- **Whether the duplicated symbol set between caimira_lite.py and cold_start_lookup.py is byte-for-byte identical NOW** vs at the moment of port. No automated drift detection. CRIT-MAINT-1 is UPHELD on this exact concern.

## What validation passed

- F6-A (expected critic count): PASS — `{C1, C2, C3, C4, C5}` actual matches expected.
- F6-B (every finding has ≥2-lane coverage): PASS — every reviewer finding has C1+C2 plus C3 cross-check.
- F6-C (C3 fresh-sweep + C2 full coverage): PASS — C3 declared `fresh_sweep_complete: true` per Rule 3; C2 mapped all 80 findings.
- F6-D (triangulation behavior matches conditional logic): PASS — triangulation correctly skipped given no UNCLEAR / no high-risk / no `--always-on-meta`.
- Phase A bookend (primer drift check): PASS — HEAD SHA unchanged at `4d7ef2f6c21c569d0e03dcecd21b83c3e5c6a408`; 3-of-3 file:line spot-checks resolve verbatim.

## What was not validated

- **3-verifier triangulation (Step 4) did not fire.** Rationale: strict trigger logic per `synthesis.md` — no UNCLEAR per-finding classifications, no high-risk routing tags, no `--always-on-meta` flag. Confidence ceiling is therefore Moderate (not High/Very-High) per the rubric. If the user wants the synthesis upgraded to High confidence on a future round, re-invoke with `--always-on-meta`.
- **Phase D bookend (Step 6) post-verdict external-context re-validation did not fire** — same conditional logic as Step 4.
- **C6/C7 paired adversarial-defender + fn-amplifier did not fire** — no canonical risk_tags (auth/security/payments/data-migration/privacy) detected in routing. The submission is competition-bound and historically known to regress on hidden, but those operational risks don't align with the canonical risk-tag set. If the user wants C6+C7 defense-in-depth specifically, re-invoke with `--always-on-meta`.
- **Paper-text comparison against arXiv:2410.06524 sections 3.3 / 4.3 / Eq. 7 / Eq. 10.** C4 attempted WebFetch on arxiv.org, aclanthology.org, and github.com/Lalor-Lab/CAIMIRA — all returned permission-denied or 404. The user has the paper PDF locally (per the CLAUDE.md context); a direct re-verification of the centering protocol and `λ_s` is the recommended next step before any "paper-faithful" upstream claim.
- **scikit-learn 1.8 deprecations were declared in the primer's external research but C4 could not independently verify** (WebSearch denied). Not load-bearing — the diff doesn't use the deprecated APIs.

## What would change the verdict

To move from "Ready with fixes (fork)" / "Not ready (upstream)" to **Ready to merge (upstream)**:
1. Add Sphinx entries to `docs/source/api/models.rst` for CAIMIRA, ColdStartLookupPredictor, LLMJudgeIRT, build_difficulty_prompt (closes CRIT-STD-1).
2. Add `Examples` doctest blocks to LLMJudgeIRT class docstring and build_difficulty_prompt function (closes CRIT-STD-2).
3. Resolve CRIT-MAINT-3: either make ColdStartLookupPredictor inherit `Predictor` (with a `predict(query)` shim) OR move it to a `torch_measure.contrib.` namespace until the base-class contract is harmonized.
4. Patch the `_logit(NaN) = 16.118` silent path (CRIT-ADV-1/8/CORR-2): the C5-VERIFIED patch in REVIEW_correctness.md applies cleanly; add an analogous NaN guard at the `_logit` callsites.
5. Fix the pre-existing `__all__` `TabPFNPredictor` missing import (CRIT-ADV-2): one-line `from torch_measure.models.tabpfn_predictor import TabPFNPredictor` in `__init__.py`.
6. PDF spot-check the CAIMIRA paper to either confirm or correct the "paper is silent on zero-centering" hedge AND the `skill_reg=1e-4` default vs paper's `1e-5`.

To move from "Ready with fixes (fork)" to **safe-to-upload to Codabench**:
1. Address the submission/ P1s: train.py multi-condition collapse (CRIT-CORR-1), Platt-shift calibration mismatch (CRIT-ADV-5), `float(label)` raises on bad reveal (CRIT-ADV-11), train.py untested (CRIT-TEST-1), model.py production-path untested (CRIT-TEST-2), labeling.py state untested (CRIT-TEST-3), CAIMIRA↔CAIMIRALite forward-parity untested (CRIT-TEST-4), 6 duplicated symbols without parity test (CRIT-MAINT-1).
2. Re-train the CAIMIRA artifact on real data (909 subjects × all binary benchmarks) and rerun the D-9 gate. Confirm sub-gates (a)/(c)/(d) PASS with a NON-degenerate predictor (verify `main_distribution.std > 0.05`, per the README's documented sub-gate (a) coverage gap).
3. Fix the PAIEC error-code citation (CRIT-LEARN-6) in submission/model.py + submission/README.md.

To move DOWN from "Ready with fixes (fork)" to **block the submission entirely**:
- Any additional UNCLEAR finding promoted to UPHELD that hits the predict-path silent-failure surface.
- Discovery that the duplicated symbol set has actually drifted (currently UNVERIFIED — see "What remains uncertain").

## Provenance

- Synthesis started: 2026-05-22
- Synthesis ended: 2026-05-22
- HEAD SHA at start (primer assembly): `4d7ef2f6c21c569d0e03dcecd21b83c3e5c6a408`
- HEAD SHA at synthesis (Phase A re-fetch): `4d7ef2f6c21c569d0e03dcecd21b83c3e5c6a408`
- Phase A drift result: **none**
- F6 results: A=PASS, B=PASS, C=PASS, D=PASS
- Triangulation: **skipped** (no triggers per conditional logic; documented in F6-D)
- Addendum applied: **no** (round 1)
- Phase D re-validation: **skipped** (same conditional logic as triangulation)
- Wave 1 active width: 7 personas dispatched in parallel
- Wave 2 active width: 5 critics dispatched (C1, C2, C3, C5 parallel; C4 sequenced after C3)
- External research lane: completed at primer-assembly Step 7; brief at `_EXTERNAL_RESEARCH.md`

## Fork-specific guidance

- **Fork-only findings (must address before Codabench upload; do NOT need to surface upstream):** CRIT-CORR-1, CRIT-ADV-5, CRIT-ADV-11, CRIT-TEST-1, CRIT-TEST-2, CRIT-TEST-3, CRIT-LEARN-6, CRIT-LEARN-5, CRIT-LEARN-8, CRIT-MAINT-2, CRIT-MAINT-5, CRIT-STD-3, CRIT-STD-7, CRIT-ADV-3, CRIT-ADV-4, CRIT-ADV-9, CRIT-ADV-10, CRIT-ADV-13, all `submission/`-anchored findings.

- **Upstream-eligible findings (candidates for upstream PR to `aims-foundations/torch_measure`; satisfy CONTRIBUTING.md first):** CRIT-CORR-4 (LLMJudgeIRT NaN), CRIT-CORR-5 (or-chain falsy), CRIT-MAINT-1 (duplicated symbols, partial — apply parity test at fork edge), CRIT-MAINT-3 (Predictor contract), CRIT-MAINT-4 (LLMJudgeIRT NOT-FOR-PRODUCTION in `__all__`), CRIT-STD-1 (Sphinx docs), CRIT-STD-2 (doctest Examples), CRIT-ADV-1/8 (NaN in `_logit` at cold_start_lookup.py), CRIT-ADV-2 (TabPFNPredictor in __all__ without import — pre-existing in upstream but this PR should fix it), CRIT-PY-1 (rename `_lookup_p` → `lookup_p`), FN-V1-1 (SimHash non-ASCII regex — affects upstream's labeling-style code if ever ported back), FN-V1-12 (tautological test in `tests/test_models/test_caimira.py`).

- **Applies-to-both findings:** CRIT-MAINT-1 (the duplicated-symbols claim itself applies to both files), CRIT-TEST-4 (forward-output parity), FN-V1-11 (skill init divergence), CRIT-PY-6, CRIT-PY-13, several others.

The upstream parent `aims-foundations/torch_measure` will see PR activity on the fork; the team may want to delay opening any upstream-bound PR until after the CS321M competition cycle closes, to keep the upstream PR scoped to library work (not entangled with submission/ scaffolding).

## See also

- `_PRIMER.md` — full primer with 6 sections (Source / Change / Comments / Project standards / External research / Provenance).
- `_EXTERNAL_RESEARCH.md` — full external-best-practices research brief.
- `_ROUTING_DECISION.md` — routing inputs + rule fired (R7) + Wave 1/Wave 2 selection.
- `_CRITIC_CHARTER.md` — Wave 2 critic charters with lane assignments.
- `_F6_CHECKS.md` — F6-A/B/C/D structural meta-checks.
- `_FULL_DIFF.patch` — full 4656-line diff of `main..feat/caimira`.
- 7 `REVIEW_*.md` files — Wave 1 reviewer outputs.
- 5 `CRITIC_*.md` files — Wave 2 critic outputs.
- `_DRAFT_REVIEW_COMMENTS.md` — paste-ready review comments with publication approval gate.
