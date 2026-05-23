# REVIEW_correctness — Wave 1 reviewer output

## Summary

Tracing the diff end-to-end against the kit contract and the trainer↔ship-side state-dict parity, I found one P1 silent-collision bug in the trainer's wide-form construction (multi-condition responses for the same `(subject_id, item_id)` collapse non-deterministically), one P2 NaN-propagation gap in `clip_for_predict` (a NaN-shaped trained weight or judge output would bypass the clip and reach the kit's hard-rejector), and one P3 meta-vs-reality lie in `train.py`'s embed_dim field. The CAIMIRA paper-faithfulness claim (bias asymmetry, zero-centering, dot-product response equation), the state_dict parity (trainer-side `CAIMIRA` ↔ ship-side `CAIMIRALite`), the EB 6-level hierarchy, the reservoir-sampling acquisition function, the Platt cap, and the no-broad-except discipline in `predict()` all check out cleanly. The hybrid hot path (line 250 in `submission/model.py`) is numerically safe because `_logit` clips to `[1e-7, 1-1e-7]` and `_sigmoid` is branched-stable; the only NaN reachability is via NaN-shaped model weights (P2) or via the LLMJudgeIRT judge_fn returning NaN (P1, but NOT-FOR-PRODUCTION).

## Findings

### CRIT-CORR-1 — Trainer silently collapses multi-condition responses for the same `(subject_id, item_id)` to a non-deterministic single label
- **Severity:** P1
- **Anchor:** `submission/train.py:444-454` (`_build_wide_response`) + `submission/train.py:190-201` (`build_indices`) + `submission/train.py:152-187` (`collect_binary_examples`)
- **Evidence quote:** 
  - `train.py:69` declares `_RESPONSE_COLS = ("subject_id", "item_id", "benchmark_id", "test_condition", "response")` — the response table HAS a `test_condition` column, so the same `(subject_id, item_id)` can appear under multiple conditions.
  - `train.py:174-178` emits one example dict PER row (no de-dup by `(subject_id, item_id)`); the example dict stores `subject_name` and `item_content` (NOT `item_id` / `subject_id`).
  - `train.py:198-200`: `item_to_idx` is keyed by `item_content` (every item with the same content collapses into one row).
  - `train.py:452-453`:
    ```
    wide = torch.full((n_subj, n_items), float("nan"))
    wide[s, i] = y
    ```
- **Rationale:** Walk through the data flow when item 42 has both `test_condition="zero-shot"` (label 0) and `test_condition="cot"` (label 1) for subject 7. `collect_binary_examples` produces TWO example dicts with the same `subject_name="gpt-4"` and the same `item_content="What is 2+2?"`. `build_indices` maps both to the same `(s=7, i=42)` long-form triple. `build_long_form` produces `s=[..., 7, 7, ...]`, `i=[..., 42, 42, ...]`, `y=[..., 0.0, 1.0, ...]`. Then `wide[s, i] = y` with duplicate indices — per PyTorch's advanced-indexing assignment semantics, this is non-deterministic on CUDA (the docs say "if the indices contain duplicates, the result is non-deterministic"); on CPU it takes the last value but the iteration order through `by_bench[...]` is dict-insertion order, which depends on parquet row order. The result is that CAIMIRA trains on an arbitrary one of the per-condition labels, silently losing condition signal. The EB tables built downstream (`build_eb_tables`, lines 242-308) DO carry condition info (`sbc[subject||benchmark||condition]`), so the EB fallback sees the full distribution, but the CAIMIRA forward (which the hybrid blend weights at 0.6) trains on a collapsed view. The downstream effect is a real (if subtle) reduction in CAIMIRA signal quality for benchmarks that ship multiple `test_condition`s per `(subject, item)` pair. The bug is currently MASKED whenever each `item_id` is queried under exactly one `test_condition` value across all subjects — verify against the HF dataset shape before judging severity escalation.
- **Suggested fix** (no patch; needs design discussion):
  Either (a) include `test_condition` in the item-index key (`item_to_idx[(item_content, condition)] = ...`), which doubles or triples `n_items` but preserves condition signal, OR (b) explicitly aggregate duplicate `(s, i)` cells by `.mean()` before building the wide matrix, OR (c) drop duplicates and emit a `[train_caimira] N collisions resolved by aggregation` log line so the operator knows the collapse occurred. Option (b) preserves binary-label property after rounding; option (a) is the principled fix.
- **Note:** Applies to fork-only / `submission/` directory — `torch_measure.models.CAIMIRA` itself is generic and condition-agnostic; the bug is in the trainer's data plumbing.

### CRIT-CORR-2 — `clip_for_predict` does not defend against NaN; NaN-shaped trained weights would propagate to a contract violation
- **Severity:** P2
- **Anchor:** `submission/caimira_lite.py:115-117` (`clip_for_predict`) and `submission/model.py:258` (final return)
- **Evidence quote:**
  ```python
  def clip_for_predict(p: float) -> float:
      """Final clip applied at the kit boundary, narrower than ``_clip``."""
      return float(max(_PREDICT_LO, min(_PREDICT_HI, float(p))))
  ```
- **Rationale:** Python's `min(nan, finite)` and `max(nan, finite)` propagate NaN (NaN comparisons are always False, so the `<` comparison inside `min/max` returns False, and the implementation returns NaN in both cases). Trace: if CAIMIRA training diverged and the saved state_dict contains NaN weights, then `self.relevance_head(emb)` → NaN, `torch.softmax(NaN, dim=-1)` → NaN, `caimira_logit` is NaN, `caimira_p = _sigmoid(NaN)` — let me trace this. `_sigmoid`'s branch `if x >= 0` evaluates `NaN >= 0` as False; falls to `else` branch: `e = math.exp(NaN) = NaN`, returns `NaN / (1 + NaN) = NaN`. So `caimira_p = NaN`. Then `_logit(NaN)` calls `max(1e-7, min(1-1e-7, NaN))` — same NaN propagation — returns NaN. Then `_blend_logits` returns NaN. Then `clip_for_predict` returns NaN. The platform's pre-validator (per `starting_kit/README.md:225-229`) catches this and fails the submission with the verbatim `Invalid predict() output: predict() must return a finite float in [0, 1].` error. So the failure is loud, not silent — but it consumes a daily submission quota slot AND triggers `[PAIEC-PREDICT-002]` only AFTER the platform has loaded the artifact. A producer-side defensive `if not math.isfinite(p): p = 0.5` in `clip_for_predict` (or a `torch.isfinite(self.state_dict()[...].sum())` check before `torch.save` in `train.py`) would catch the divergence pre-flight, consistent with the project's distinguishable-defensive-fallbacks rule (option 4 — "be made impossible by producer-side artifact verification"). Currently the producer-side verification is `train.py:399-403` (state-dict round-trip), which checks shape parity but NOT weight finiteness.
- **Suggested fix**:
  ```python
  # before (caimira_lite.py:115-117):
  def clip_for_predict(p: float) -> float:
      """Final clip applied at the kit boundary, narrower than ``_clip``."""
      return float(max(_PREDICT_LO, min(_PREDICT_HI, float(p))))
  # after:
  def clip_for_predict(p: float) -> float:
      """Final clip applied at the kit boundary, narrower than ``_clip``.

      NaN-defensive: NaN input maps to the midpoint (0.5) rather than
      propagating to the kit's pre-validator. The platform's distinguishable
      fallback is ``-ln(0.5) = 0.69`` NLL which is the canonical broken-model
      signature; the alternative ``Invalid predict() output`` rejection
      consumes a daily quota slot without producing leaderboard signal.
      """
      p = float(p)
      if not math.isfinite(p):
          return 0.5
      return float(max(_PREDICT_LO, min(_PREDICT_HI, p)))
  ```
  Requires adding `import math` (it's already imported via line 64).
- **Note:** Fork-only (`submission/`). Upstream `torch_measure` does not own this clip helper. The pattern doc `distinguishable-defensive-fallbacks-2026-05-18.md` notes the design tradeoff explicitly: silent-0.5 is the constant-broken signature, NOT distinguishable from a legitimate prediction. So this is a deliberate design tension — current code prefers loud kit-rejection over silent-0.5, which is the safer default per D-3. If the team prefers loud over silent, leave as-is and document the rationale at the call site.

### CRIT-CORR-3 — `train.py` writes a constant `EMBED_DIM=768` to `meta.json` instead of `embeddings.shape[1]`, allowing meta to drift from the actual artifact
- **Severity:** P3
- **Anchor:** `submission/train.py:413` and `submission/train.py:357`
- **Evidence quote:**
  ```python
  # train.py:357 — TRUE embedding dim from the encoder output:
  model = CAIMIRA(
      n_subjects=len(subject_to_idx),
      n_items=len(item_to_idx),
      embedding_dim=embeddings.shape[1],
      latent_dim=args.latent_dim,
  )
  ...
  # train.py:413 — meta.json field uses the HARDCODED constant:
  metadata = {
      ...
      "embed_dim": EMBED_DIM,  # always 768; does not reflect actual encoder output
      ...
  }
  ```
- **Rationale:** `EMBED_DIM` in `caimira_lite.py` is hardcoded `768` (the all-mpnet-base-v2 output dim). If a future user changes `ENCODER_REPO` to a different SentenceTransformer (e.g., `all-MiniLM-L6-v2` which is 384-d), the trainer builds CAIMIRA with `embedding_dim=384` (correct, from `embeddings.shape[1]`), saves a state_dict with `relevance_head.weight` shape `(latent, 384)`, but writes `"embed_dim": 768` to meta.json. Then `model.py:144` reads `embed_dim = int(META.get("embed_dim", EMBED_DIM))` → 768, builds `CAIMIRALite(embedding_dim=768)`, and `load_state_dict` raises shape-mismatch — which is LOUD, not silent. So the consequence is "load fails noisily at module init" → `[PAIEC-PREDICT-002]`. Not a wrong-answer bug. But the meta.json field is misleading and any operator inspecting the JSON to debug a load failure would be misled.
- **Suggested fix**:
  ```python
  # before (train.py:413):
          "embed_dim": EMBED_DIM,
  # after:
          "embed_dim": int(embeddings.shape[1]),
  ```
- **Note:** Fork-only. Mechanical fix. The before-text appears exactly once in the file (verified via grep) so the patch is safe.

### CRIT-CORR-4 — `LLMJudgeIRT.predict()` returns NaN when `judge_fn` returns NaN (not when it raises)
- **Severity:** P1 (within the LLMJudgeIRT contract; mitigated by the NOT-FOR-PRODUCTION marker on the module)
- **Anchor:** `src/torch_measure/models/llm_judge_irt.py:142-151`
- **Evidence quote:**
  ```python
  try:
      judge_logit = float(self.judge_fn(item_content, benchmark))
  except Exception:
      judge_logit = 0.0

  delta = self.alpha * judge_logit + self.beta
  z = max(-self.logit_cap, min(self.logit_cap, theta - delta))
  p = _sigmoid(z)
  lo, hi = self.lookup.output_clip
  return max(lo, min(hi, float(p)))
  ```
- **Rationale:** The `try/except Exception` guards against `judge_fn` raising, but `float(nan_returning_fn(...))` does NOT raise — it returns `float('nan')`. Then `judge_logit = NaN`, `delta = self.alpha * NaN + self.beta = NaN`, `z = max(-cap, min(cap, theta - NaN))`. Trace `min(cap, NaN)`: Python's `min` returns the second arg when comparison fails, and `cap < NaN` is False — so `min(cap, NaN) = NaN`. Similarly `max(-cap, NaN) = NaN`. Then `_sigmoid(NaN)`: as traced in CRIT-CORR-2, returns NaN. Then `max(lo, min(hi, NaN)) = NaN`. The function returns NaN, violating its docstring contract (`Returns: float — Predicted probability of correctness, clipped to ``output_clip``.`). Any caller using this in a pipeline that doesn't NaN-check would propagate the NaN downstream.
  
  The mitigation is that `LLMJudgeIRT` is marked NOT-FOR-PRODUCTION in the module docstring (lines 16-37). But the class is exported from `torch_measure.models.__init__` and could be picked up by a downstream user who skipped the negative-result disclosure.
- **Suggested fix**:
  ```python
  # before (llm_judge_irt.py:142-145):
      try:
          judge_logit = float(self.judge_fn(item_content, benchmark))
      except Exception:
          judge_logit = 0.0
  # after:
      try:
          judge_logit = float(self.judge_fn(item_content, benchmark))
          if not math.isfinite(judge_logit):
              judge_logit = 0.0
      except Exception:
          judge_logit = 0.0
  ```
  Requires confirming `math` is imported (it is, line 41).
- **Note:** Upstream-eligible (`src/torch_measure/models/llm_judge_irt.py`). If upstreaming, the docstring also needs a "Returns: float — finite, clipped to output_clip; NaN judge_fn outputs degrade to logit=0.0" addendum.

### CRIT-CORR-5 — `EBLookup.lookup_p` and `ColdStartLookupPredictor._lookup_p` use `or` to combine truthy/None lookups, masking legitimate `0.0` priors
- **Severity:** P2
- **Anchor:** `submission/caimira_lite.py:335-336` and `src/torch_measure/models/cold_start_lookup.py:320-321`
- **Evidence quote:**
  ```python
  # caimira_lite.py:335-336:
  subj_p = self.subj.get(subj_name) or self._subj_ci.get(subj_name.lower())
  bench_p = self.bench.get(benchmark) or self._bench_ci.get(benchmark.lower())
  # cold_start_lookup.py:320-321 — same pattern:
  subj_p = self.subj.get(subj_name) or self._subj_ci.get(subj_name.lower())
  bench_p = self.bench.get(benchmark) or self._bench_ci.get(benchmark.lower())
  ```
- **Rationale:** Python's `or` short-circuits on falsy values, which includes `0.0`. If `subj_name` resolves to an exact key in `self.subj` with value `0.0` (i.e., a subject empirically scored zero in training), `self.subj.get(subj_name) or self._subj_ci.get(...)` skips the direct hit and falls through to the case-insensitive lookup. In the present code, this is MASKED because `_table_clip` (in `train.py:238-239`) clamps all stored values to `[0.05, 0.95]`, so no entry can be exactly `0.0`. But the constraint is implicit — a future trainer that writes raw means OR a user who hand-constructs an `EBLookup` with the documented schema (e.g., `subj={"weak_model": 0.0}`) would silently lose the direct hit. The downstream effect is that a level-5 fallback may use the case-insensitive entry (which could differ) instead of the exact match.
  
  Verify with a focused trace: suppose `subj = {"weak_model": 0.0}` and `_subj_ci = {"weak_model": 0.0}` (built from `subj` at construction). Then `self.subj.get("weak_model")` returns `0.0` (falsy), `or` short-circuits to `self._subj_ci.get("weak_model".lower())` which is `0.0`. Same value — no harm. But suppose `subj = {"Weak_Model": 0.0, "weak_model": 0.4}` (case-distinct entries). Then `self.subj.get("Weak_Model")` returns `0.0`, `or` short-circuits to `self._subj_ci.get("weak_model")` which is `0.4`. The exact-case `0.0` was masked.
  
  The current `_table_clip` guarantee mitigates this in practice, but it's a logic bug in the lookup helper.
- **Suggested fix** (no patch; needs careful before/after — both `caimira_lite.py` AND `cold_start_lookup.py` need the same change, and the comparison shape matters):
  ```python
  # before:
  subj_p = self.subj.get(subj_name) or self._subj_ci.get(subj_name.lower())
  # after:
  subj_p = self.subj.get(subj_name)
  if subj_p is None:
      subj_p = self._subj_ci.get(subj_name.lower())
  ```
  Same change for `bench_p` two lines below. Apply to both files.
- **Note:** `submission/caimira_lite.py` is fork-only; `src/torch_measure/models/cold_start_lookup.py` is upstream-eligible. If the project standard is to keep the two files in lock-step (`caimira_lite.py` is a verbatim port per its module docstring), both should be fixed together.

### CRIT-CORR-6 — `set_embeddings` writes self._embeddings then `predict()` reads relevance/difficulty from FULL bank; cold-start predict() at inference would IndexError silently if META has an `item_idx` exceeding `n_items`
- **Severity:** P3
- **Anchor:** `src/torch_measure/models/caimira.py:223-246`
- **Evidence quote:**
  ```python
  def predict(self, query: dict[str, torch.Tensor]) -> torch.Tensor:
      s = query["subject_idx"]
      i = query["item_idx"]
      relevance, difficulty = self.compute_item_params()
      diff = self.skill[s] - difficulty[i]
      logit = (diff * relevance[i]).sum(dim=-1)
      return torch.sigmoid(logit)
  ```
- **Rationale:** `compute_item_params()` with default `embeddings=None` returns relevance/difficulty over the FULL training bank `(n_items, latent_dim)`. Then `relevance[i]` and `difficulty[i]` index into this — if `i` contains values `>= n_items` (e.g., from a cold-start caller who confused the API), this raises `IndexError`. This is loud failure, NOT silent. Not a correctness bug per se — the caller is responsible for passing in-bank indices for `predict()`, and `compute_item_params(new_embeddings)` is the documented cold-start path. No fix required; flagging only because the API surface has TWO cold-start paths (`predict()` for in-bank items, `compute_item_params(new_embeddings)` for out-of-bank items), and the contract is in the docstring but not enforced. A caller who passes a query with out-of-bank `item_idx` would get a clear IndexError, so this is self-documenting failure.
- **Suggested fix**: (no patch; the current behavior is correct and self-documenting.)
- **Note:** Upstream-eligible. Documented design choice, not a bug. Flagging for completeness; the testing reviewer may want to verify the cold-start path is exercised in the test suite.

## Residual risks

These were investigated and found to be non-bugs, but the reasoning is non-obvious and worth recording:

- **R1: `_blend_logits` numerical stability.** Both `_logit` calls clip inputs to `[1e-7, 1-1e-7]`, giving logit output in `[-16.118, 16.118]`. With `lambda_=0.6` the weighted sum is in `[-16.118, 16.118]` (max coefficient sum is 1.0). `_sigmoid` of values in this range is finite and well-behaved (no overflow because of the branched implementation). Final `clip_for_predict` to `[1e-4, 1-1e-4]` is the kit-contract bound. No NaN/inf path here unless one of the inputs to `_logit` is NaN, which would require `caimira_p` or `eb_p` to be NaN, which is covered by CRIT-CORR-2.
- **R2: `EBLookup.fit_platt` `len(labeled)` cache key collision risk.** Within a Codabench round the `labeled` list is monotonic (the platform appends K=5 reveals per data category). Same-length implies same-content. If the platform ever changes to per-category passing of disjoint lists with the same length, the cache would return stale calibration. Currently fine; would surface as a P0 if the contract changes.
- **R3: `_update_reservoir` correctness as a reservoir-sampling algorithm.** Verified against Vitter Algorithm R. The probability that the n-th candidate replaces an existing slot is `_MAX_SEEN / n = 128/n` for `n > 128`, which matches the textbook. The use of `hash(candidate_key + counter)` as a deterministic-pseudorandom source is fine for diversity acquisition (it's a pure-function reservoir, not a uniform-random one). Note: the determinism means the reservoir contents are reproducible given the candidate order, which is desirable for offline replay.
- **R4: `caimira_logit` `subject_idx` out-of-range.** `subject_idx` is sourced from `SUBJECT_TO_IDX.get(subj_name)` in `model.py:235-238` and is `None` if not found. The `None` case triggers the EB-only path. If `META["subject_to_idx"]` is corrupted such that a value is `>= n_subjects`, `self.skill[subject_idx]` raises `IndexError` — loud, not silent. Not a bug; just worth noting that this depends on meta consistency.
- **R5: `set_embeddings` accepts float64 silently.** Float32 model params × float64 input raises a dtype mismatch error from PyTorch (loud, not silent). Not a correctness bug.
- **R6: `compute_item_params(center='auto')` during training-time eval.** When `model.train()` is True (during a fit-time validation loss inside the mle_fit loop), `center='auto'` resolves to `"dynamic"`, which recomputes the mean each forward pass over the FULL training bank. This is correct for gradient flow during fitting but means held-out validation rows during fit would have non-frozen centering. The paper-faithfulness claim is preserved: dynamic centering during gradient updates, frozen centering at inference (after `fit()` calls `self.eval()` + `_refresh_difficulty_mean()`).

## Testing gaps

These are CORRECTNESS-flavored gaps that the testing reviewer should be aware of — I'm flagging them here rather than writing tests because that's not my lane:

- **T1**: The `tests/test_models/test_caimira.py` suite does NOT exercise the duplicate `(subject_idx, item_idx)` case in `_normalize_fit_inputs` (the inherited base-class method). A test that feeds a wide-form matrix with multiple condition-specific values for the same `(s, i)` would surface CRIT-CORR-1's blast radius.
- **T2**: `tests/test_submission/test_model_contract.py` runs in `LOCAL_SMOKE` mode (returns fixed `0.5`), so it does NOT exercise the hot path through `_blend_logits` → `_sigmoid` → `clip_for_predict`. A fixture-based test that loads a TINY known artifact (e.g., a `CAIMIRALite` with all-zeros weights and a hand-crafted `eb_tables.json`) would catch CRIT-CORR-2 if NaN weights were introduced.
- **T3**: No test exercises `_lookup_p` with a legitimate `0.0` prior in `subj` or `bench` (CRIT-CORR-5). The lookup-table-build clip at `[0.05, 0.95]` masks this in practice but a regression that loosens the clip would surface this.
- **T4**: `LLMJudgeIRT.predict()` is tested only against `judge_fn`s that return finite floats; no test covers the NaN-returning case (CRIT-CORR-4).
