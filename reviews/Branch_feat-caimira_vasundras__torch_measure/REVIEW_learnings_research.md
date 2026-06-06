# REVIEW_learnings_research — Branch feat/caimira (vasundras/torch_measure), round 1

**Lane:** Learnings research (parent CS321M `docs/solutions/` cross-application).
**Findings schema:** `CRIT-LEARN-N`.
**Provenance:** searched `/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/docs/solutions/` (7 categories, 27 entries); intersected against `submission/{model.py,labeling.py,train.py,build_zip.sh,caimira_lite.py,README.md}` (read end-to-end) and the 4 commit messages on `feat/caimira`. The fork has no `docs/solutions/` of its own; the parent-repo store is the shared knowledge base.

**Lane scope reminder:** This review surfaces past learnings that apply to THIS diff and flags any place where the diff appears to under-apply or violate one. "Properly applies pattern X" is surfaced as informational P3 because it gives the Wave 3 synthesis confidence to UPHOLD other findings that hinge on it.

**Severity rubric (shared with the charter):** P0 correctness/security/contract-break; P1 high-impact maintainability or unsurfaced-contract gap; P2 measurable style/perf/doc cost; P3 nice-to-have. Compliance findings (no-action-needed) are P3 informational.

**Forensics summary table** (build per direct-intersection check; sources are absolute paths in the parent CS321M `docs/solutions/`):

| ID | Pattern | Diff site | Verdict | Severity |
|---|---|---|---|---|
| CRIT-LEARN-1 | `runtime-errors/silent-ncf-head-load-failure-via-full-module-pickle-2026-05-17.md` | `submission/train.py:384` + `:397-404`; `submission/model.py:153` | **properly applies (compliance)** | P3 informational |
| CRIT-LEARN-2 | `design-patterns/distinguishable-defensive-fallbacks-2026-05-18.md` | `submission/model.py` (no try/except → 0.5) + `submission/labeling.py:220-222` | **properly applies (compliance)** | P3 informational |
| CRIT-LEARN-3 | `runtime-errors/labeling-py-nan-fallback-bomb-2026-05-17.md` | `submission/labeling.py` (single `except` path; no `ENCODER is None` branch; stdlib-only) | **properly applies (compliance) + structural immunity** | P3 informational |
| CRIT-LEARN-4 | `performance-issues/codabench-adaptive-labeling-simhash-acquisition-2026-05-20.md` | `submission/labeling.py` vs parent `predictive-eval-competition/submission/labeling.py` | **properly applies (verbatim port)** | P3 informational |
| CRIT-LEARN-5 | `architecture-patterns/pre-submission-transfer-audit-stress-test-gate-2026-05-22.md` | `submission/README.md` "D-9 gate run history" | **partially applies + a substantive coverage-gap claim worth verifying** | P2 |
| CRIT-LEARN-6 | `design-patterns/codabench-two-tier-error-reporting-paiec-system-2026-05-19.md` | `submission/model.py:17-27` docstring + `submission/README.md:135-136` | **properly applies; one P1 documentation accuracy gap** | P1 |
| CRIT-LEARN-7 | `design-patterns/explicit-allowlist-for-bundle-artifacts-2026-05-19.md` (**not cited by parent CLAUDE.md**) | `submission/build_zip.sh:54-127` | **properly applies (textbook implementation)** | P3 informational |
| CRIT-LEARN-8 | `architecture-patterns/codabench-sprint-build-gate-submit-orchestrator-2026-05-21.md` (**not cited by parent CLAUDE.md**) | `submission/build_zip.sh` (Gate G + H only); missing A/D/E/F | **under-applies — only 2 of 8 sprint gates implemented** | P2 |
| CRIT-LEARN-9 | `architecture-patterns/staff-errata-propagation-supersedes-live-platform-text-2026-05-21.md` | `submission/model.py:36, :211` + `submission/README.md` (no sample-size mention; `K=5` claim) | **partial — `K=5` is correct per Ed #160; sample-size silent (acceptable; not propagated either way)** | P3 informational |

Notes on file scope: this review touches `submission/` for runtime-contract checks. The diff also adds three model files (`src/torch_measure/models/caimira.py`, `cold_start_lookup.py`, `llm_judge_irt.py`) targeted at upstream — the parent's `docs/solutions/` does not carry IRT-implementation learnings, so those files do not intersect any past learning in scope. C4 (external claims researcher) and the other Wave 1 reviewers cover library-side correctness.

---

## CRIT-LEARN-1 — Silent NCF Head Load Failure pattern: properly applied (P3 informational)

**Pattern:** `/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/docs/solutions/runtime-errors/silent-ncf-head-load-failure-via-full-module-pickle-2026-05-17.md`

**Pattern rule:** trainers must save `state_dict()`, not the full module; loaders must use `weights_only=True`; the trainer must round-trip-load through the EXACT consumer class before shipping; the architecture class must live in a real importable module both sides share.

**Diff sites (verified):**

- Trainer save site: `submission/train.py:384` — `torch.save(model_cpu.state_dict(), pt_path)` (state_dict, not full module). The model is moved to CPU and `.eval()`-ed first (line 383).
- Trainer round-trip verification: `submission/train.py:387-404` — loads back into a fresh `CAIMIRALite` (the standalone consumer class, NOT the trainer's `torch_measure.models.CAIMIRA`), uses `weights_only=True`, asserts `(missing, unexpected) == ([], [])`, raises `SystemExit` on mismatch.
- Consumer load site: `submission/model.py:153` — `torch.load(_HEAD_PATH, map_location=DEVICE, weights_only=True)`.
- Shared importable module for the consumer class: `submission/caimira_lite.py` — class `CAIMIRALite` lives in this module (line 7 of the docstring frames it as the "SHIPPABLE STANDALONE port"). Both `train.py` and `model.py` `from caimira_lite import CAIMIRALite`.
- Architecture-version pin: `submission/caimira_lite.py:74` — `CAIMIRA_ARCH_VERSION = 1`. (The pattern doc recommends this; the diff carries it.)

**Verdict:** The pattern is implemented to-the-letter — and notably better than the original NCF case because the round-trip-load goes through `CAIMIRALite` (the actual consumer class), not just through a fresh instance of the trainer's `CAIMIRA` class. This catches one extra class of failure (state_dict-shape drift between the trainer-side and ship-side classes) that the canonical post-mortem warns about generically but the original NCF fix only handled within a single class.

**Caveat (P3, not raised as a finding):** The pattern doc's "Prevention" section recommends an "import-time probe before zipping" (`from submission import model; assert m.NCF_HEAD is not None; assert m.predict(...) != 0.5`). The diff's `build_zip.sh` does NOT run this probe — that's the gap CRIT-LEARN-8 is built around (the sprint-orchestrator's Gate E/F). Re-flagging here so the synthesis can correlate.

**Class:** `applies_to_both` (the round-trip pattern is general engineering discipline; if these model files PR upstream to `aims-foundations/torch_measure`, the trainer's round-trip pattern is worth carrying, though the upstream library doesn't ship a single consumer artifact).

**Severity:** P3 informational (compliance — surface so synthesis can UPHOLD).

---

## CRIT-LEARN-2 — Distinguishable defensive fallbacks (D-3 rule): properly applied (P3 informational)

**Pattern:** `/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/docs/solutions/design-patterns/distinguishable-defensive-fallbacks-2026-05-18.md`

**Pattern rule (the four options):** defensive fallbacks must either (1) return values OUTSIDE the legitimate output domain (distinguishable offsets), (2) signal via a non-value side-channel surviving the runtime, (3) re-raise loudly, or (4) be made impossible by producer-side artifact verification.

**`submission/model.py` — `predict()` (option 3 + option 4):**

- Diff site: `submission/model.py:17-27` docstring explicitly cites the D-3 / no-broad-except discipline by name and names the silent-`0.5` post-mortem.
- Grep verification: `grep -n "except Exception" submission/model.py` → 0 matches. `grep -n "return 0.5"` → 1 match at `:221` inside the LOCAL_SMOKE branch (`if _LOCAL_SMOKE: return 0.5`) — that is **the kit's own local-smoke contract**, not a fallback, gated on `PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1` which the hosted runtime does NOT set (verified at `:66`, `:136-138`, `:220-221`).
- Module init failure path (option 3): `torch.load(..., weights_only=True)` at `:153`, `json.loads(_META_PATH.read_text())` at `:141`, `EBLookup.from_json(_EB_PATH)` at `:159` — all unguarded. Any failure propagates to module-init, which the Codabench platform surfaces as `[PAIEC-PREDICT-002]` per CRIT-LEARN-6. This matches the pattern doc's "re-raise loudly" option.
- Producer-side verification (option 4): the `train.py` round-trip check (per CRIT-LEARN-1) makes the silent-load failure mode structurally impossible — a broken artifact never leaves the trainer.

**`submission/labeling.py` — `acquisition_function()` (option 1, distinguishable offset):**

- Diff site: `submission/labeling.py:182-204` docstring + `:220-222` the lone `except Exception: return 0.0`.
- Legitimate-domain claim verified: legitimate score = `_clamp_score(diversity + metadata + tie_break)` where `diversity ∈ [0, 1]` (line 154-158), `metadata = 0.20 / (1 + count) ∈ [0.20/256, 0.20] ≈ [7.8e-4, 0.20]` (line 131-134, with `_MAX_STRATA=256`), `tie_break ∈ [0, _TIE_EPSILON] = [0, 0.01]` (line 213 + `_TIE_EPSILON=0.01` at line 42). Minimum legitimate sum is therefore `0 + 7.8e-4 + 0 ≈ 7.8e-4`, which is strictly greater than the `0.0` sentinel. The docstring's claim that the legitimate floor is "essentially never zero in practice" is more cautious than necessary — the metadata bonus is structurally non-zero on every call. Either way, the sentinel `0.0` is distinguishable from any successful path's return.
- The `_clamp_score(value)` (line 174-177) helper also returns `0.0` if `value` is non-finite, which means a NaN/inf bug inside the try block STILL collapses to the sentinel `0.0` rather than escaping. Belt-and-suspenders defense against the canonical NaN-fallback bomb (see CRIT-LEARN-3).
- Pattern doc rule (1) literal text: *"Move the fallback value outside the legitimate output domain. For a probability output, return `0.5 + 1e-3` from the 'no head' branch."* The labeling.py implementation follows the same rule shape (sentinel outside legitimate domain) — and goes further by also satisfying the pattern doc's structural-immunity advice via `_clamp_score`.

**`submission/caimira_lite.py`:** no `except Exception` blocks (Grep verified, line-260 caimira_logit has its only `return float(logit)` at line 243, no fallback). The clip helper at line 117 is a value bounder, not a defensive fallback.

**Verdict:** The diff implements all four options of the D-3 pattern in concert — option 3 for module init, option 4 for the producer side, option 1 for the labeling.py sentinel. Both `submission/model.py:17-27` docstring AND `submission/README.md:129-140` cite the pattern by name and reference the canonical bug post-mortem. The README's "Failure mode" row of the differences table at `:54` ("Distinguishable EB fallback (varies with input)") explicitly distinguishes the new design from the 2026-05-22 `<UNK>`-fallback antipattern.

**Class:** `fork_only_caimira_branch` for the submission code; `upstream_eligible` portion is `caimira_lite.py`'s discipline (no broad-except, no fallback-bomb shape) which travels to the upstream PR if the standalone file is ever pulled in — though `caimira_lite.py` is intentionally fork-only per the README's "Why standalone" section.

**Severity:** P3 informational (compliance).

---

## CRIT-LEARN-3 — `labeling.py` NaN-fallback-bomb pattern: structural immunity (P3 informational)

**Pattern:** `/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/docs/solutions/runtime-errors/labeling-py-nan-fallback-bomb-2026-05-17.md`

**Pattern rule:** the canonical bug had TWO independent NaN-return paths — the obvious `except` branch AND a structured `if ENCODER is None: return float("nan")` early-return — and the kit contract escalates a single NaN/inf/raise/timeout from `acquisition_function()` to discarding ALL acquisition scores for the whole round. The fix audited ALL return statements and the prevention rules mandate force-injection unit tests per fault path.

**Diff sites (verified):**

- `grep -nE "return [0-9]|return float|except" submission/labeling.py` enumerates EVERY return site (lines 66, 79, 134, 156, 176, 177, 222). None return `float('nan')` or `float('inf')`.
- The only `except` is at line 220, returning `0.0` (line 222) — finite, low-priority, in-domain-distinguishable per CRIT-LEARN-2.
- **No `from model import ENCODER`** anywhere in the file (Grep `from model import` → 0 matches). The SimHash rewrite removed the cross-module dependency that produced the structured Path 2 in the canonical bug. The pattern doc's Path 2 is therefore **structurally inapplicable** in this diff — there is no shared state between `model.py` load failure and `labeling.py` returning NaN.
- `_clamp_score` at line 174-177 is the structural NaN-immunity layer — even if `diversity + metadata + tie_break` ever evaluated to NaN/inf (the integer arithmetic on bit-counts makes this strictly impossible, but defense-in-depth), the helper substitutes `0.0`.

**Verdict:** Two independent reasons the NaN-bomb pattern's bug class cannot manifest in this `labeling.py`:

1. The encoder dependency that enabled Path 2 in the canonical bug has been entirely removed (stdlib-only — `hashlib.blake2b`, `re`, `math`).
2. The `_clamp_score` helper plus the single `except Exception → 0.0` sentinel guarantee finite return on every code path, including any future code that might re-introduce floating-point arithmetic.

**Class:** `fork_only_caimira_branch` (this file is a submission entry point; not upstream-eligible).

**Severity:** P3 informational (compliance — and stronger than mere compliance: structural immunity).

---

## CRIT-LEARN-4 — SimHash adaptive-labeling pattern: verbatim port (P3 informational)

**Pattern:** `/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/docs/solutions/performance-issues/codabench-adaptive-labeling-simhash-acquisition-2026-05-20.md`

**Pattern rule:** the canonical SimHash implementation in the parent repo's `predictive-eval-competition/submission/labeling.py` is **64-bit signatures, `_MAX_SEEN=128` reservoir, `_MAX_TOKENS=256`, `_MAX_STRATA=256`, stdlib-only (`hashlib.blake2b`/`re`/`math`), single `except Exception → 0.0` sentinel, no `model`/`ENCODER`/`numpy`/`torch`/`sentence-transformers` imports**. Performance contract: ~0.2 ms p95 vs ~57 ms for encoder-backed variants on a 5,000-candidate stream (benchmark `8106`).

**Diff site:** `submission/labeling.py` claims to be a port from `predictive-eval-competition/submission/labeling.py` (docstring `:24-29`).

**Verification (read both files end-to-end + diff-by-eye):**

- `_BITS = 64`, `_MAX_TOKENS = 256`, `_MAX_SEEN = 128`, `_MAX_STRATA = 256`, `_TIE_EPSILON = 0.01` — **identical** to parent (parent: lines 19-23; fork: lines 38-42).
- `_visible_text`, `_tokens`, `_hash_u64`, `_stable_unit_interval`, `_simhash`, `_coarse_subject_bucket`, `_stratum_key`, `_metadata_bonus`, `_update_stratum`, `_candidate_key`, `_diversity_score`, `_update_reservoir`, `_clamp_score`, `acquisition_function` — all 14 functions present in BOTH files with byte-identical bodies (modulo whitespace).
- The single difference is the module-level docstring header. Parent docstring (`:1-10`) is concise SimHash narrative; fork docstring (`:1-30`) adds: (a) the kit-contract failure cascade reminder (the NaN-bomb teaching), (b) bullet-point summary of the three guarantees (stdlib-only, bounded cost, bulletproof), (c) explicit port-note citing the parent repo. The port note exists because the fork's submission code may diverge from the parent's submission code in the future — without the port note, the fork's `labeling.py` would lose its provenance link.
- The function-level docstring on `acquisition_function` (fork `:182-204` vs parent `:165-187`) has cosmetic wording differences but identical semantics. The parent cites the pattern doc path directly (`../docs/solutions/design-patterns/distinguishable-defensive-fallbacks-2026-05-18.md`); the fork omits the inline path link (which is correct because the fork doesn't have `docs/solutions/` at sibling depth) — instead the docstring's last sentence appeals to "the project's distinguishable-defensive-fallbacks rule" by name.

**Verdict:** verbatim port (modulo intentional docstring divergence to make the port self-documenting). The pattern doc's "Prevention" rules — `acquisition_function()` is microsecond-to-low-millisecond; no `model`/`ENCODER`/`torch`/`numpy`/`sentence-transformers` imports; bounded module-level state — are ALL satisfied.

**Class:** `fork_only_caimira_branch`. The SimHash entry point is competition-specific; not upstream-eligible.

**Severity:** P3 informational (compliance — and the verbatim-port discipline gives the synthesis confidence that the parent-repo benchmarks at `8106` carry over).

---

## CRIT-LEARN-5 — D-9 pre-submission stress-test gate: partially applies, with substantive coverage-gap claim (P2)

**Pattern:** `/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/docs/solutions/architecture-patterns/pre-submission-transfer-audit-stress-test-gate-2026-05-22.md`

**Pattern rule:** four sub-gates run BEFORE every non-byte-identical Codabench submission: (a) probability-distribution audit, (b) packaged-runtime equivalence check, (c) benchmark-heldout stress, (d) calibration probes. Sub-gate (a)'s explicit text per the canonical doc: *"mean(p) within `±0.05` of the validation positive rate; fraction outside `[1e-3, 1-1e-3]` < `0.05`; per-benchmark mean(p) tracks per-benchmark positive rate to within `±0.10`; no benchmark/condition cell has > `0.50` of probability mass within `0.05` of a hard endpoint."* Pattern doc also says sub-gate (d) "produces a TABLE, not a single pass/fail" (verbatim, line 136 of the canonical doc).

**Diff site:** `submission/README.md:142-159` carries the "D-9 gate run history" table. Smoke run from 2026-05-22: OVERALL FAIL; sub-gates (a) PASS, (b) PASS, (c) FAIL (stress_nll=0.759 > 0.50), (d) PASS. The README's addendum at `:149-159` flags the substantive finding: sub-gate (a) checks `mean(p) ∈ [pos_rate ± 0.05]` and `frac_extreme < 0.05` but does NOT penalize **mid-range constancy**, so a degenerate constant-`0.604` predictor with `std = 1.1e-16` passed.

**Verification of the claim:**

- Re-reading the canonical pattern doc's sub-gate (a) text: the explicit pass criteria are mean(p) match, frac-extreme threshold, per-benchmark mean tracking, and no-single-cell-mass-near-endpoint. **The canonical doc does NOT explicitly require any minimum `std(p)` or quantile-spread check.** The diff's empirical finding (a constant predictor satisfies (a) but is obviously broken) is consistent with the canonical doc's stated criteria.
- The pattern doc's "Pass criteria" section at line 64 of the canonical doc does mention "per-benchmark mean(p) tracks per-benchmark positive rate to within `±0.10`" — for a constant predictor, this would only fail if some benchmark's positive rate differs from `0.604` by more than `0.10`. On a smoke run with only one subject and 905 items dropping to EB-Level-4/6, this could plausibly NOT fail per benchmark even though the global predictor is degenerate. So the README's diagnosis (the gate's (a) verdict is misleading on a constant predictor) appears correct on its own terms.
- The README's `:153-158` paragraph correctly cites the canonical pattern doc by relative path (`../docs/solutions/architecture-patterns/pre-submission-transfer-audit-stress-test-gate-2026-05-22.md`) AND correctly cites the canonical doc's "audit produces a TABLE, not a single pass/fail" framing — that exact phrasing IS at line 136 of the canonical pattern doc (in the sub-gate (d) Pass criteria description). The README applies it to sub-gate (a) instead of (d), which is a slight extension of the canonical phrasing's original scope but is consistent with the doc's broader stance that all four sub-gates produce evidence to be cross-checked, not single-bit verdicts.

**Caveat on the claim's strength:** the README addendum frames the mid-range-constancy gap as "a documented coverage gap of the gate," which is one accurate way to describe it. A reviewer could equally well describe it as: the smoke artifact's `subject_to_idx` containing only `deepseek-coder-v2` means every audit-sample row falls through to EB Level-4/6, so what the gate measured is the EB fallback path's saturation rather than CAIMIRA's distribution. Both framings are correct; the README leans on the gap framing. The C4 external-claims critic should verify the canonical pattern doc's lack of a `std(p)` threshold against the actual file content (the file is a single source of truth at line 50-68 of the canonical doc, and at the time of this review carries no minimum-std requirement).

**Under-application risk:** the diff's `build_zip.sh` does NOT invoke `python ../predictive-eval-competition/scripts/run_d9_gate.py submission_caimira.zip` (per the README's "Run flow" step 5). The gate is a manual operator step, not a build-time enforcement — meaning a future user could `bash build_zip.sh && [upload to Codabench]` without ever running the gate. The pattern doc's "When to Apply" section is unambiguous: *"BEFORE every submission that consumes a daily-quota slot, when the candidate is NOT a byte-identical reupload."* The README does say the gate must pass first; the build_zip.sh script does not enforce this. See CRIT-LEARN-8 for the broader sprint-gate under-application.

**Class:** `fork_only_caimira_branch` (the D-9 gate is competition-specific).

**Severity:** P2 — the diff has done the right things at the documentation layer (the gap is flagged accurately and propagates a useful empirical finding back to the parent-repo's pattern surface), but the under-application risk (build_zip.sh does not enforce the gate) leaves the door open for a future operator to skip the gate. The README's "Open follow-ups" section names this gap, which is partial mitigation.

**Recommendation:** the README's coverage-gap finding is worth carrying back to the parent-repo's `docs/solutions/architecture-patterns/pre-submission-transfer-audit-stress-test-gate-2026-05-22.md` as an UPDATE (sub-gate (a) should be amended to include a min-std or quantile-spread check). This is the kind of cross-cutting back-propagation that historically-vs-current-truth-propagation-for-audit-docs-2026-05-19.md anticipates. Not a blocker for this diff; a follow-up.

---

## CRIT-LEARN-6 — PAIEC two-tier error reporting: properly applied; one P1 doc-accuracy gap

**Pattern:** `/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/docs/solutions/design-patterns/codabench-two-tier-error-reporting-paiec-system-2026-05-19.md`

**Pattern rule:** Codabench surfaces SPECIFIC `[PAIEC-*-NNN]` diagnostics for some failure classes (observed: `[PAIEC-PREDICT-002]` for invalid `predict()` output; `[PAIEC-HF-DECLARED]` for missing-models.txt declarations) and a GENERIC fallback (no error code) for others. The pattern doc explicitly says NEW PAIEC codes can appear in Ed or platform output and that "the durable gap is that the kit does not document the code list, the covered stages, or which failures still collapse to the opaque generic tier."

**Diff sites:**

- `submission/model.py:17-27` docstring claims that module-init failures surface as `[PAIEC-PREDICT-002]`.
- `submission/README.md:135-136` repeats the claim: *"Module-init failures propagate to the platform as a `[PAIEC-PREDICT-002]` error before any predictions run."*

**Verification against the canonical pattern doc:**

- Canonical pattern doc line 49-53 (verbatim quote of submission `741617`'s output): the only documented `[PAIEC-PREDICT-002]` error message is *"Invalid predict() output: predict() must return a finite float in [0, 1]"* — that is, the `[PAIEC-PREDICT-002]` code specifically tags **`predict()` return-value contract violations**, not module-init failures.
- The canonical pattern doc does NOT document the module-init failure code. Module-init failures fall into the canonical pattern doc's generic fallback tier — *"the platform did not expose a participant-actionable code for the failing stage. The failure may be module-init, package-shape, dependency/cache, pre-fetch, queue/backlog, or another stage the visible classifier did not identify"* (verbatim from the canonical doc's *"Generic fallback"* section).
- Looking at the canonical doc's "Example 2" (lines 135-146): the team's actual full-package submissions `740083` and `740453` — both of which were live, real submissions including the encoder, NCF head, models.txt, labeling.py — failed with the **generic** banner ("No additional details are safe to show"), NOT with a PAIEC-coded message. The canonical doc explicitly cautions against assuming module-init failures get a `[PAIEC-PREDICT-002]` code.

**Verdict:** the diff's documentation makes a claim that the canonical pattern doc explicitly refutes. Specifically:

- `submission/model.py:17-27` says: *"If module init fails (import error, missing artifact, ``torch.load`` failure with ``weights_only=True``), the platform surfaces a ``[PAIEC-PREDICT-002]`` error before any predictions run."*
- The pattern doc says: `[PAIEC-PREDICT-002]` is for `predict()` output violations (NaN/inf/non-float-in-[0,1]); module-init failures historically surface as the GENERIC fallback.

The factual error is non-trivial: a future operator triaging a `Failed` submission from this scaffolding might look for `[PAIEC-PREDICT-002]` in the logs, fail to find it (because the actual surface is the generic banner), and conclude the failure is platform-side rather than artifact-side — exactly the misroute the canonical pattern doc was written to prevent.

**Class:** `fork_only_caimira_branch` (docstring + README — submission-specific). The 4523-line diff's only PAIEC-coded references are these two sites.

**Severity:** **P1** — documentation gap that may not yet have surfaced but will affect triage if the artifact ever fails. P1 (not P0) because (a) the broader D-3 / no-broad-except discipline is correctly applied (the module init WILL fail loudly), and (b) the README also names the canonical pattern doc by URI path so an operator can find the corrected information one click away. But the inline claim should be amended.

**Recommended fix (informational, not normative):**

- Replace `[PAIEC-PREDICT-002]` in `submission/model.py:17-27` and `submission/README.md:135-136` with either (a) a more accurate phrasing such as *"a platform-level error (specific `[PAIEC-*]` code if classified, generic banner otherwise — see ../docs/solutions/design-patterns/codabench-two-tier-error-reporting-paiec-system-2026-05-19.md for triage)"* OR (b) preserve the `[PAIEC-PREDICT-002]` reference but scope it specifically to "if `predict()` returns a non-finite or out-of-domain value at runtime" rather than to module-init failures.
- The Wave 3 synthesis should NOT propose a `suggested_fix` patch for this without first cross-checking against `predictive-eval-competition/handoffs/bug_reproductions_2026-05-19/OBSERVATIONS.md` (the canonical pattern doc's source) — if the PAIEC code family has expanded since 2026-05-19, the right fix is a more precise reference, not just an apology.

---

## CRIT-LEARN-7 — Explicit allowlist for bundle artifacts: properly applied (P3 informational) — NOT cited by parent CLAUDE.md

**Pattern:** `/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/docs/solutions/design-patterns/explicit-allowlist-for-bundle-artifacts-2026-05-19.md`

**Pattern rule:** for any build flow that packages a directory containing tracked-but-shouldn't-ship files, prefer an explicit positional allowlist over a recursive-with-exclusions pattern. Pair the recipe with output-size validation, pre-upload contents inspection (`unzip -l`), and inline rationale.

**Note:** the parent's CLAUDE.md cites the broader `design-patterns/distinguishable-defensive-fallbacks-2026-05-18.md` but does NOT cite this allowlist doc explicitly. The agent charter calls out searching for unstated learnings; this is one. The pattern is canonically referenced from the silent-NCF post-mortem (`runtime-errors/silent-ncf-head-load-failure-via-full-module-pickle-2026-05-17.md` line 164) and the sprint-orchestrator doc (Gate G) — so it transitively appears in the cited corpus.

**Diff site:** `submission/build_zip.sh:54-127`.

**Verification against the canonical pattern doc:**

- Line 54-63: explicit positional `REQUIRED_FILES` array — exactly the pattern doc's "Fixed recipe (explicit allowlist)" shape.
- Line 65-81: refusal-to-build if any required file is missing — the pattern doc's "loud omission" property.
- Line 83-86: refusal-to-overwrite without `--force` — a stricter discipline than the pattern doc requires; defends against silent re-bundling of stale artifacts.
- Line 88-99: single-`.pt` safety check — directly mirrors the pattern doc's example output-size-validation discipline (the pattern doc cites the canonical case of 17MB-vs-1.7MB-bloat-from-per-seed-`.pt`-leakage). Worth noting that the diff's single-`.pt` check operates on the ALLOWLIST not on the eventual zip — it's verifying the recipe author hasn't typoed two `.pt` files into the list, which is one notch upstream from the canonical pattern.
- Line 101: `rm -f "${OUTPUT_ZIP}"` before `zip -j ...` — prevents zip-append-mode accumulation, a class of bug not in the pattern doc but adjacent.
- Line 105-108: `zip -j` strips paths — produces the flat ZIP shape the kit pre-validator requires.
- Line 110-116: post-build NESTED-PATH check — verifies the build matched intent.
- Line 118-127: post-build ALLOWLIST-vs-ACTUAL comparison — the pattern doc's "Inspect contents before upload" gate built into the script itself.
- `set -euo pipefail` at line 26 — strict-mode bash, which protects against the canonical bug in `logic-errors/bash-retry-loop-silent-success-pipefail-without-errexit-2026-05-21.md` (which the parent CLAUDE.md DOES cite). This script has no retry loops, so the canonical pipefail-trap doesn't apply, but the strict-mode hygiene is the right baseline.
- Line 5-15 inline rationale: explicitly cites the silent-`.pt`-leakage bug from the canonical pattern doc by reference (`scripts/check_submission.sh:101-116`).

**Verdict:** textbook implementation. The diff's build_zip.sh is arguably more careful than the canonical pattern doc's example — it adds three properties (refusal-to-overwrite, nested-path post-check, allowlist-vs-actual post-check) that the canonical doc doesn't require but recommends in spirit.

**Class:** `fork_only_caimira_branch` (the script is competition-specific).

**Severity:** P3 informational (compliance — exemplary).

---

## CRIT-LEARN-8 — Sprint build-gate-submit orchestrator: under-applies (only Gates G + H present, missing A/D/E/F) — NOT cited by parent CLAUDE.md (P2)

**Pattern:** `/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/docs/solutions/architecture-patterns/codabench-sprint-build-gate-submit-orchestrator-2026-05-21.md`

**Pattern rule:** the canonical Codabench submission orchestrator runs EIGHT pre-submit gates A-H (per the pattern doc's lines 80-94), ordered cheapest-first, short-circuit on first failure:
- **A**: static-AST no-network — forbid `requests`/`urllib`/`socket`/`subprocess`/`huggingface_hub`/`datasets` imports; require `SentenceTransformer(...)` and `from_pretrained(...)` to carry `local_files_only=True`.
- **B**: models.txt byte-identical check.
- **C**: no `requirements.txt`.
- **D**: offline dynamic import — run `model.predict(...)` in a subprocess with `HF_HUB_OFFLINE=1` + monkeypatched `socket.create_connection`/`urllib.request.urlopen` raising `NETWORK_BLOCK_TRAP`. Pass `labeled=None`, `labeled=[]`, and `labeled=[dict(x, label=0/1) × 4]`.
- **E**: local-smoke — re-run `predict()` with `PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1`.
- **F**: labeling smoke — `acquisition_function(...)` 10× on distinct items; every return finite Python float.
- **G**: allowlist build + audit (the CRIT-LEARN-7 layer).
- **H**: forbidden ZIP scan — `.env`, `.git`, `secret`, `token`, `.planning`, `runs/`, `scripts/`, `tmp/`, `requirements.txt`, multiple `.pt`; confirms `model.py` is at root.
- (2026-05-22 amendment) **Gate I**: transfer-audit stress gate (CRIT-LEARN-5).

**Note:** the parent's CLAUDE.md does NOT cite this orchestrator doc explicitly. It is transitively referenced through the D-9 pattern doc (which is cited) and through the CAIMIRA work's grandparent rationale. The agent charter mandates surfacing 0-3 unstated learnings; this is one.

**Diff site:** `submission/build_zip.sh` (139 lines).

**Verification against the canonical pattern doc:**

- **Gate A (static-AST no-network):** NOT IMPLEMENTED. The script does not inspect `model.py` / `labeling.py` / `caimira_lite.py` for forbidden imports. The canonical orchestrator's Gate A is the static guard against network-trap regressions; without it, a future commit could add `from huggingface_hub import snapshot_download` to `model.py` and the build_zip.sh would happily ship it.
- **Gate B (models.txt byte-identical):** PARTIALLY IMPLEMENTED. The script INCLUDES `models.txt` in the allowlist (line 58), but does not verify the file content matches a canonical reference. The fork's `models.txt` (per primer) is one line: `sentence-transformers/all-mpnet-base-v2`. The canonical-orchestrator gate compares against a known-good reference; the diff has no such reference.
- **Gate C (no requirements.txt):** IMPLEMENTED IMPLICITLY. The allowlist doesn't include `requirements.txt`, so `zip -j` will not pick it up — but the allowlist doesn't have a check that `requirements.txt` ISN'T present in `submission/`. If a future commit drops a requirements.txt next to model.py, build_zip.sh silently ignores it. The canonical Gate C is a `Path.exists()` check that would FAIL the build.
- **Gate D (offline dynamic import + network trap):** NOT IMPLEMENTED. The script does not run `model.predict(...)` in any subprocess. The canonical gate is the load-bearing safety net for the silent-NCF-head-load failure class (the CRIT-LEARN-1 pattern's "import-time probe before zipping" prevention rule); without it, a broken artifact could be packaged without ever exercising the consumer code path.
- **Gate E (local-smoke):** NOT IMPLEMENTED. The script does not run `PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1 python -c "from model import predict; predict({...})"`. The fork has `tests/test_submission/test_model_contract.py` (per primer line 60), but the test is invoked as part of pytest, NOT as part of build_zip.sh — so a user running only `bash build_zip.sh` skips it.
- **Gate F (labeling smoke):** NOT IMPLEMENTED. No `acquisition_function(...)` finite-return check in the build script.
- **Gate G (allowlist build + audit):** FULLY IMPLEMENTED per CRIT-LEARN-7.
- **Gate H (forbidden ZIP scan):** PARTIALLY IMPLEMENTED. The post-build allowlist-vs-actual check at line 118-127 verifies the ZIP contains EXACTLY the allowlist — that catches the canonical "what got bundled?" question. The canonical Gate H also scans for forbidden TOKENS within the bundled files (e.g., `.env`, `secret`, `token` literal-string content), which this script does not do. For a one-time competition submission the practical risk is low because the allowlist is short and human-vetted, but the canonical orchestrator's belt-and-suspenders posture is defensive against secret-leak regressions.
- **Gate I (transfer-audit stress gate, 2026-05-22 amendment):** NOT IMPLEMENTED. See CRIT-LEARN-5.

**Score:** out of 8 canonical gates (A-H), the diff fully implements 1 (G), partially implements 2 (B, H), implements 1 implicitly (C), and skips 4 (A, D, E, F). Plus skips the 2026-05-22 amendment Gate I.

**Verdict:** the diff's `build_zip.sh` is a competent allowlist-builder (Gate G) plus minimal post-build sanity (partial Gate H), but does NOT implement the broader pre-submit-gate pattern from the canonical sprint orchestrator. The README does claim to expect operators to manually run validators in steps 3-5 of the "Run flow" — partial mitigation but not equivalent to the canonical gate-and-fail-loud architecture.

**Class:** `fork_only_caimira_branch` (submission-specific).

**Severity:** P2 — this is an under-application, not a violation. The runtime artifacts (the 1.7MB ZIP) will pass the kit's pre-validator and the kit's local smoke test if the operator runs `scripts/check_submission.sh` and `scripts/smoke_test_submission.sh` from the parent repo (per the README's "Run flow"). But the canonical orchestrator pattern is to bake the gates INTO the build script so a one-step `bash build_zip.sh` produces either a ready-to-upload ZIP or a loud failure — the diff requires the operator to assemble a 5-step run-flow manually, and a future operator could shortcut steps 3-5 under deadline pressure.

**Recommendation (informational):** consider adding a `build_zip.sh --strict` mode that runs Gate A (AST scan via `python -c "import ast; ..."`), Gate D (subprocess `predict()` call with offline env vars), Gate E (`PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1 predict()` smoke), and Gate F (`acquisition_function()` finite-return smoke) inside the build script itself. The diff's clean two-script split (build vs gate-and-submit) is also defensible; either way, the absence of the gates in `bash build_zip.sh` should be flagged in the README's "Open follow-ups" alongside the existing D-9 gate caveat.

---

## CRIT-LEARN-9 — Staff-errata propagation: `K=5` correctly applied; sample-size silent (P3 informational)

**Pattern:** `/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/docs/solutions/architecture-patterns/staff-errata-propagation-supersedes-live-platform-text-2026-05-21.md`

**Pattern rule:** staff Ed errata supersedes live Codabench page text when they disagree. Two specific load-bearing facts the parent CS321M repo's `llm_wiki/contract/current-ground-truth.md` carries:
- **K=5 per data category** for `labeled` reveals (kit and Ed agree).
- **Sample size**: 10,000 per Ed #160 staff reply 2026-05-21 (live Codabench page text + static 2026-05-17 kit/PDF still say 5,000; Ed wins per source-precedence).

**Diff sites:**

- `submission/model.py:36` ("`K=5 ``labeled`` reveals from the platform's K=5 ``labeled`` reveals"). Verified: matches the pattern doc's canonical statement.
- `submission/model.py:211` (`Optional list of K=5-per-category revealed examples`). Verified: matches.
- `submission/README.md` doesn't carry sample-size claims at all (no `5,000`/`10,000` mentions; `grep` confirms 0 matches).

**Verification against pattern:**

- The diff is CORRECT about K=5 — that fact is stable across both the 2026-05-17 kit and the 2026-05-21 Ed #160 errata.
- The diff is SILENT about sample size — it doesn't say "5,000" (which would be wrong) and doesn't say "10,000" (which would be right). For a SUBMISSION ENTRY POINT, this is actually defensible: sample size doesn't appear in the `predict(input, labeled)` contract, so there's no need to mention it. The pattern doc's concern is about OPERATIONAL ASSUMPTIONS leaking into downstream docs.
- One adjacent caveat: `submission/labeling.py:8-12` mentions "the streaming 10K-candidate-per-round path" — that's the staff-errata-correct value (10,000 = the new sample size). The fork's `labeling.py` matches the parent's, which was updated post-2026-05-21 to reflect the errata. So both the K=5 and the 10K claims in the fork are consistent with the canonical errata-aware ground truth.

**Verdict:** the diff propagates the K=5 fact correctly AND the 10K-candidates-per-round fact correctly. The pattern doc's concern (silent propagation of stale 5,000 claims) does not manifest.

**Class:** `fork_only_caimira_branch` (submission-specific docs).

**Severity:** P3 informational (compliance).

---

## Surface-level applicability of secondary patterns (not finding-grade)

For completeness — these were checked but produce no actionable finding:

- **Soft-gate harness adoption** (`soft-gate-harness-adoption-with-public-surface-fallback-2026-05-18.md`): not applicable. The fork does not adopt the parent's `cursor_codabench_hf_skill` harness. The diff's submission flow is self-contained and operator-driven (per `README.md` "Run flow"). If the fork ever needs to integrate the harness, the soft-gate pattern would apply — but that's a future architectural decision.

- **Pre-flight dynamic source pull** (`pre-flight-dynamic-source-pull-2026-05-19.md`): not applicable. The fork has no `.session_context/`, no `scripts/sync_dynamic_sources.sh`, no `cursor_codabench_hf_skill/`. The diff is on a torch_measure fork, not the predictive-eval-competition repo. If the fork is later merged with the parent's dynamic-source-pull pipeline, the submission code would benefit — but that's not in scope for this diff.

- **Bash retry loop pipefail trap** (`bash-retry-loop-silent-success-pipefail-without-errexit-2026-05-21.md`): the diff's `submission/build_zip.sh` uses `set -euo pipefail` (strict mode minimum) and has no retry loops. The canonical bug pattern does not apply. Compliance noted; no finding.

- **Keychain-backed dynamic source refresh** (`keychain-backed-dynamic-source-refresh-pipeline-2026-05-21.md`): not applicable. The fork doesn't carry the predictive-eval-competition's credential pipeline.

- **Historical-vs-current-truth propagation** (`historical-vs-current-truth-propagation-for-audit-docs-2026-05-19.md`): partially relevant to CRIT-LEARN-6 (the PAIEC doc-accuracy gap is exactly the kind of stale-current-state-claim the pattern doc warns about), but the diff isn't an audit-doc artifact, so the canonical workflow doesn't directly apply. The CRIT-LEARN-6 fix should follow the pattern doc's discipline (preserve historical narrative; refresh current-state claims), but that's mechanism for fixing CRIT-LEARN-6, not a separate finding.

- **`codabench-poller-wrong-status-endpoint.md` / `codabench-repeat-submitter-idempotency-guard.md`**: not applicable. The fork has no polling or idempotency-checking submission tooling; it's a manual operator-driven flow.

- **Modal-related patterns** (`modal-*.md`): not applicable. The fork has no Modal integration.

- **Gold-track current-state CoVe ledger** (`gold-track-current-state-cove-ledger-2026-05-21.md`): the fork doesn't carry the parent's `llm_wiki/contract/current-ground-truth.md` ledger. The pattern doc's "When to apply" condition (recording cached-vs-live source discrepancies) isn't triggered by anything in this diff.

---

## Recommendations for Wave 2 critics + Wave 3 synthesis

1. **CRIT-LEARN-6 is the only P1 in this lane.** A future operator triaging a `Failed` submission from this scaffolding will look for `[PAIEC-PREDICT-002]` in the platform logs based on the README's claim, fail to find it (because module-init failures actually surface as the generic banner), and misroute the triage. The Wave 3 synthesis should propose a doc patch that either (a) replaces `[PAIEC-PREDICT-002]` with a more accurate phrasing, or (b) scopes the existing reference to `predict()` output violations only. C4 should cross-check this finding against `predictive-eval-competition/handoffs/bug_reproductions_2026-05-19/OBSERVATIONS.md` before any fix lands.

2. **CRIT-LEARN-5 and CRIT-LEARN-8 are sibling P2s** — both flag that the diff has done the documentation right (D-9 gate run history table; D-3 / no-broad-except docstring) but the BUILD SCRIPT does not enforce the gates. A future operator running `bash build_zip.sh && [upload to Codabench]` could skip both the D-9 gate AND the sprint-orchestrator Gates A/D/E/F. The Wave 3 synthesis should treat these together — either both are accepted-with-followup or both get a single combined remediation.

3. **CRIT-LEARN-1 through CRIT-LEARN-4 are compliance findings** — they give the synthesis confidence to UPHOLD the diff's broader claims about silent-load defense, D-3 discipline, NaN-bomb immunity, and SimHash performance. If any Wave 2 critic raises a FP about these patterns, the synthesis should weight these compliance findings heavily in the C2 FP catalog.

4. **CRIT-LEARN-7 is a compliance finding for an UNSTATED-by-parent pattern** — the explicit-allowlist build script is textbook. If a future operator tries to "simplify" build_zip.sh to `zip -r ../submission.zip .`, they will regress to the canonical 17MB-vs-1.7MB bug. The inline rationale at line 5-15 mitigates this risk. Worth surfacing in the synthesis as a positive — the fork has done one thing the parent CLAUDE.md doesn't explicitly require but should.

5. **CRIT-LEARN-9 is a compliance finding for the staff-errata pattern** — both load-bearing facts (K=5 and 10K) are propagated correctly. No action.

6. **Cross-lane note for the C4 external-claims critic:** the CAIMIRA paper citations in `caimira_lite.py:53-55` and `src/torch_measure/models/caimira.py` (per primer) refer to arXiv:2410.06524 (Lalor et al., EMNLP 2024). The parent's `docs/solutions/` does not carry CAIMIRA-paper-specific learnings, so this lane has nothing to contribute to that check; C4 owns it.

7. **Cross-lane note for the C3 fresh-sweep critic:** the diff has 4523 added lines; this learnings lane covered the submission-side surface (~1500 lines of the diff: model.py, labeling.py, train.py, caimira_lite.py, build_zip.sh, README.md). C3 should sweep the remaining ~3000 lines (the 3 model files in src/torch_measure/models/ + the 5 test files + the tutorial notebook) for cross-cutting concerns this lane did not address — particularly the upstream-eligibility constraint that CONTRIBUTING.md requires Sphinx docs under `docs/source/`, which the diff does not add (per the primer).
