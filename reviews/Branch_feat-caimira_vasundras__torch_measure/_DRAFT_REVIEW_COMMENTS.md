# Draft review comments — Branch feat/caimira (vasundras/torch_measure)

## Publication status

- **Status:** DRAFTED (awaiting user approval — see `## Approval prompt` at bottom)
- **Posting target (default):** `Local-only — do not post`
  - **Why default to local:** there is no open PR for `feat/caimira` on `github.com/vasundras/torch_measure` (verified via `gh pr list`). Comments cannot attach to a PR that doesn't exist. The skill's options for posting are:
    1. **Local-only handoff** (default): keep this file as the deliverable; user reads + applies fixes manually. **No external write.**
    2. **Open PR first, then post comments on it.** User opens `vasundras:feat/caimira → vasundras:main` (intra-fork PR; doesn't trigger upstream-parent notifications) OR `vasundras:feat/caimira → aims-foundations:main` (cross-fork upstream PR — public, visible to upstream maintainers). After PR exists, this file's comments can be attached as `gh pr review` line comments.
    3. **Open Issue on `vasundras/torch_measure`** for tracker continuity.
  - **Recommendation:** start with option 1 (local handoff). The 14 P1 findings include some that aren't ready for public discussion (e.g., the CAIMIRA paper-faithfulness questions need a PDF spot-check; opening an upstream PR before that check would surface uncertain claims to the upstream maintainers).
- **Visibility note for forks:** posting any comment to `vasundras/torch_measure` is **publicly visible**. The upstream parent `aims-foundations/torch_measure` will see fork activity. Other forks of the same parent will see it too. Whatever gets posted is searchable and ranks on the public discussion of the upstream project's ecosystem.

## Comments

Below are 6 draft comment families. Each family rolls up multiple finding IDs from `_FINAL_VERDICT.md`. The "Target" line names the destination IF the user chooses to post; the default destination is `Local-only — do not post`.

---

### Comment 1 — Silent NaN propagation in `_logit` breaks D-3 invariant

**Target (if approved for posting):** PR-level review comment on `submission/caimira_lite.py:97-100`, with cross-reference to `submission/model.py:97-103` and `src/torch_measure/models/cold_start_lookup.py:63-66`.

**Severity:** P1 (operational — empirically verified)

**Supporting findings:** CRIT-ADV-1, CRIT-ADV-8, CRIT-CORR-2; C4 VERIFIED via `python3` empirical reproduction.

> The `_logit` helper at `caimira_lite.py:97-100` (and duplicated in `cold_start_lookup.py:63-66` and `submission/model.py:97-103`) is documented as "numerically safe" but silently returns `±16.118` for NaN inputs: `max(1e-7, min(1 - 1e-7, nan))` returns `1e-7` under Python's NaN-comparison semantics (since `1e-7 < nan` is False), and `math.log(1e-7 / (1 - 1e-7)) ≈ -16.118` is finite.
>
> Consequence: if any model parameter, encoder output, or intermediate value becomes NaN during training or inference, the subsequent CAIMIRA forward will produce a confident `sigmoid(±16.118) ≈ 0.9999` prediction for every in-vocab item — the exact "silent constant prediction" signature that the project's own `docs/solutions/runtime-errors/silent-ncf-head-load-failure-via-full-module-pickle-2026-05-17.md` post-mortem was written to defend against. The D-3 / no-broad-except discipline only forbids `try/except → 0.5` swallowing; it does NOT cover NaN-invariant violations in the math itself.
>
> Suggested direction (C5 verified the patch applies cleanly):
> ```python
> def _logit(p: float) -> float:
>     if not math.isfinite(p):
>         raise ValueError(f"_logit requires finite input, got {p}")
>     p = max(1e-7, min(1 - 1e-7, p))
>     return math.log(p / (1 - p))
> ```
> Three call-site copies need the same patch (caimira_lite, cold_start_lookup, submission/model). Consider the CRIT-MAINT-1 fix (single-source-of-truth + parity test) alongside.

---

### Comment 2 — Platt-shift calibration mismatch (CAIMIRA-EB blend)

**Target:** PR-level review comment on `submission/model.py:252-256`.

**Severity:** P1 (operational; structurally same class as 2026-05-22 transfer postmortem)

**Supporting findings:** CRIT-ADV-5

> The per-benchmark intercept-only Platt shift is fit against EB-only logits (`EB.fit_platt` walks the labeled examples through `EB.lookup_p`, not through the hybrid blend), then applied to the BLENDED logit at `submission/model.py:254-256` (`p = _sigmoid(slope * _logit(p) + intercept)` where `p` is the post-`_blend_logits` value).
>
> If CAIMIRA and EB systematically disagree on a benchmark (which is the entire reason for blending them), the shift's expected behavior (move EB toward observed label-mean) doesn't directly improve the blended prediction. This is structurally the same calibration-mismatch class as the 2026-05-22 transfer postmortem's failure signature (local best val_nll, worst hidden score) — the model and the calibrator were fit on incompatible logit bases.
>
> Options:
> 1. Fit Platt directly against the HYBRID logit (compute `_blend_logits(caimira_p, eb_p, lambda_)` for each labeled example, take the mean, derive shift).
> 2. Apply the Platt shift to `eb_p` BEFORE blending, then blend.
> 3. Document the mismatch as deliberate (with a hypothesis for WHY it improves the hidden score) and run the D-9 gate to verify.
>
> Currently the README doesn't address this. Recommend option 1 or 3 before D-9 full-train run.

---

### Comment 3 — `submission/train.py` collapses multi-condition labels via wide-form overwrite

**Target:** Line comment on `submission/train.py:444-454` (`_build_wide_response`); cross-reference `submission/train.py:152-187` (`collect_binary_examples`) and `:190-201` (`build_indices`).

**Severity:** P1 (correctness)

**Supporting findings:** CRIT-CORR-1

> `collect_binary_examples` iterates over all responses and emits one `examples` row per response — so the same `(subject_id, item_id)` pair under different `test_condition` values produces N rows, each with potentially different labels. `_build_wide_response` then projects them into a `(n_subjects, n_items)` matrix where `wide[s, i] = y` overwrites — the surviving label is the LAST one processed, which depends on iteration order from the HF parquet (non-deterministic across pyarrow versions).
>
> Net effect: CAIMIRA trains on a randomized condition-subset; EB tables (built from the same `examples` list with `||condition` keys) see the full distribution. The hybrid blend at predict time combines a "biased toward random condition" CAIMIRA signal with an EB signal that has full information — not a coherent ensemble.
>
> Two paths:
> 1. **Long-form fit:** CAIMIRA's `fit()` accepts long-form data per its base class; pass `(subject_idx, item_idx, response)` triples directly without wide-form intermediate. The current `_build_wide_response → fit` round-trip is unnecessary.
> 2. **Pre-collapse:** before `_build_wide_response`, group `examples` by `(subject_id, item_id)` and take the mean label (or majority vote for binary). Document the collapse policy.
>
> Option 1 is cleaner. Either way, fix before any real-data D-9 run, otherwise the gate's verdict reflects an artifact of pyarrow row order, not model behavior.

---

### Comment 4 — CONTRIBUTING.md violations for upstream PR scope

**Target:** PR-level review comment OR PR body checklist item (if user opens an upstream PR).

**Severity:** P1 (CONTRIBUTING.md hard requirements)

**Supporting findings:** CRIT-STD-1, CRIT-STD-2

> If this branch is being prepared for an upstream PR to `aims-foundations/torch_measure`, the diff currently fails CONTRIBUTING.md:36 — there are no entries in `docs/source/api/models.rst` for the 4 new public symbols `CAIMIRA`, `ColdStartLookupPredictor`, `LLMJudgeIRT`, `build_difficulty_prompt` (`grep -E '(CAIMIRA|ColdStartLookupPredictor|LLMJudgeIRT|build_difficulty_prompt)' docs/source/api/models.rst` returns zero matches; C4 VERIFIED).
>
> Additionally, CONTRIBUTING.md:35 mandates `Examples` blocks. `CAIMIRA` and `ColdStartLookupPredictor` have them; `LLMJudgeIRT` class and `build_difficulty_prompt` function do not.
>
> If the intent is an upstream PR, both are pre-requisites. If the intent is fork-only Codabench use, both are non-blocking.

---

### Comment 5 — `__all__` declares `TabPFNPredictor` without importing it (pre-existing bug perpetuated)

**Target:** Line comment on `src/torch_measure/models/__init__.py:47`.

**Severity:** P1 (broken at HEAD; pre-existing — pre-dates this branch)

**Supporting findings:** CRIT-ADV-2, verified via grep at `__init__.py`

> `__init__.py:47` lists `"TabPFNPredictor"` in `__all__` but there is no corresponding `from torch_measure.models.tabpfn_predictor import TabPFNPredictor` in the import block (lines 5-26). `from torch_measure.models import *` would raise `AttributeError` on this name. The file `src/torch_measure/models/tabpfn_predictor.py` does exist.
>
> This pre-dates `feat/caimira` (the merge commit `777a3b1` brought it in from upstream/main per the commit history). It's not THIS branch's bug, but this branch's `__init__.py` edits perpetuate it — and the diff is the natural place to fix it since the file is already being edited.
>
> One-line fix: add `from torch_measure.models.tabpfn_predictor import TabPFNPredictor` to the import block. Worth surfacing to the upstream maintainers via Issue or as part of an upstream PR.

---

### Comment 6 — CAIMIRA paper-faithfulness needs spot-check

**Target:** Branch-level note (no specific anchor); included in any upstream PR description as a "verification needed" callout.

**Severity:** P1 (verification gap; not a confirmed bug)

**Supporting findings:** External research surfaced 2 candidate deviations; C4 UNVERIFIABLE-IN-SESSION (arxiv WebFetch denied).

> Two CAIMIRA paper-faithfulness claims need direct PDF verification before any "paper-faithful" external-facing claim:
>
> 1. **Zero-centering protocol.** `caimira.py:69-90` docstring says: *"The paper specifies that difficulty is zero-centred over the training item bank but does not specify whether the centering mean is recomputed every batch / epoch or held fixed."* External research suggests paper Eq. 7 (`d_j := d'_j - (1/n_q) Σ_j d'_j`) actually specifies the **frozen-bank** mean — not silent on it. The implementation's choice (dynamic during training, frozen at inference) may be a deliberate variation rather than a "silent paper" interpretation. Recommend updating the docstring to either confirm the paper's specification or document the variation explicitly.
>
> 2. **`skill_reg` default.** Diff defaults `skill_reg=1e-4` (`caimira.py:257`); paper Section 4.3 reports `λ_s = 1e-5`. **10× larger than paper.** Was this set by sweep on team data, or is it a copy-paste from `difficulty_reg`? Either explain or align to the paper.
>
> External research could not verify either claim from arxiv directly (WebFetch denied). User has the paper PDF locally; please spot-check.

---

## Lower-severity comments (informational; group as "nice-to-have" in any PR)

These come from REVIEW_kieran_python.md (P2/P3) + REVIEW_maintainability.md (P2/P3) + REVIEW_testing.md (P2/P3) + the 3 PROMOTED-FROM-FN findings. They are NOT included as separate posting candidates by default — they belong in `_FINAL_VERDICT.md` for the next iteration. Examples:

- CRIT-PY-1: rename `_lookup_p` → `lookup_p` (LLMJudgeIRT accesses private member; sibling EBLookup already exposes it as public)
- CRIT-PY-2 / CRIT-MAINT-8: `Literal["auto", "dynamic", "frozen"]` on `compute_item_params(center=...)`
- CRIT-PY-3: lift `import math` to module-level in `submission/model.py`
- FN-V1-1: SimHash regex `[a-z0-9]+` strips non-ASCII — multilingual benchmarks may collapse to identical signatures
- FN-V1-11: `CAIMIRALite.skill` initializes to zeros vs upstream's `randn * 0.01` — divergence is masked by state-dict load in production
- FN-V1-12: `test_predict_matches_manual_caimira_equation` is tautological

---

## Approval prompt (verbatim — to be presented to user)

The 6 comment families above + the lower-severity notes summarize 80 reviewer findings + 3 promoted-from-FN candidates.

**Choose what to do with them:**

1. **`local_handoff_only`** (default; **recommended**) — keep `_DRAFT_REVIEW_COMMENTS.md` as the handoff artifact; you read + apply fixes locally; **no external write**. No PR opened, no comments posted, no Issues created.

2. **`open_intra_fork_pr_and_post`** — open `vasundras:feat/caimira → vasundras:main` PR (does NOT notify upstream maintainers of `aims-foundations/torch_measure` directly, but is publicly visible on the fork). Then attach the 6 comment families as line comments. You preview the PR body before it's posted.

3. **`open_upstream_pr_and_post`** — open `vasundras:feat/caimira → aims-foundations/torch_measure:main` (UPSTREAM cross-fork PR). Highly visible to upstream maintainers. **Do NOT choose this until comments 4 and 6 are addressed first** (CONTRIBUTING.md violations + paper-faithfulness spot-check) — otherwise the upstream PR ships uncertain claims to the maintainers.

4. **`open_issue_on_fork`** — open an Issue on `vasundras/torch_measure` summarizing the P1 findings; useful for tracker continuity without committing to a PR. Visible on the fork.

5. **`approve_subset`** — let me know which specific comment families to post + where.

6. **`decline`** — same as 1 (local-only); explicitly close out the publication gate.

**The skill will NOT call `gh pr review`, `gh pr comment`, `gh issue create`, or `gh issue comment` until you explicitly approve one of options 2-5 above.**

## Publication status (final)

- Status: AWAITING USER APPROVAL
- Posting target: `Local-only — do not post` (default until approved)
- Commands run for posting: none
