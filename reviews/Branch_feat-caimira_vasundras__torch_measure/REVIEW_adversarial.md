# REVIEW_adversarial — Branch feat/caimira (vasundras/torch_measure), round 1

**Reviewer:** ce-adversarial-reviewer (Wave 1)
**Charter:** construct concrete failure scenarios for the diff's hot paths; surface 8-15 substantive scenarios with reproduction recipes and the BEST DEFENSE the code already has.
**Method:** brute-force iteration over diff's adversarial-input surfaces + cross-component cascade tracing + empirical verification with Python 3.11 / PyTorch 2.10.0 on disk where load-bearing.

Depth bucket: **Deep** (4523 changed lines; competition-bound submission code with documented hidden-leaderboard regression history; module-level state; cross-process state-dict transfer; ZIP packaging via shell). All four hunting modes (assumption violation, composition failures, cascade construction, abuse cases) exercised below.

Severity rubric per `_CRITIC_CHARTER.md`: P0 (production correctness / data-loss path), P1 (high-impact, may-not-yet-have-surfaced), P2 (style/perf/doc), P3 (nice-to-have). Confidence per the anchored rubric in this persona's charter.

---

## CRIT-ADV-1 — NaN in CAIMIRA forward silently coerces to confident 0.9999 (P1 / D-3 violation)

**Severity:** P1
**Confidence:** 75 (concrete scenario, exact mechanism verified empirically)
**Component:** `submission/model.py`, `submission/caimira_lite.py`
**Anchor:** `submission/caimira_lite.py:97-101` (`_logit` clip), `submission/caimira_lite.py:115-117` (`clip_for_predict`), `submission/model.py:115-117` (`_blend_logits`)

**Constructed scenario.** CAIMIRA's `caimira_logit(subject_idx, item_embedding)` returns a Python float. If any of `skill[subject_idx]`, `difficulty_head(emb)`, or `relevance_head(emb)` contains NaN (e.g., a corrupted state_dict download, or downstream numerical instability from a Long-running A100 retrain that produced subtly-non-finite weights), the returned logit is NaN. Trace through the predict path:

```python
caimira_logit = float('nan')           # CAIMIRA caimira_logit returns nan
caimira_p     = _sigmoid(nan)          # = nan (verified)
p             = _blend_logits(caimira_p, eb_p, 0.6)
                # = _sigmoid(0.6 * _logit(nan) + 0.4 * _logit(eb_p))
                # _logit(nan): max(1e-7, nan) returns 1e-7 (Python max/min with NaN
                # returns first non-NaN arg); min(1-1e-7, nan) returns 1-1e-7.
                # → _logit produces log((1-1e-7) / 1e-7) ≈ 16.118
                # _sigmoid(0.6 * 16.118 + 0.4 * logit(0.5))
                # = _sigmoid(9.67 + 0) ≈ 0.99994
p_final       = clip_for_predict(p)    # = 0.9999 (clipped to 1-1e-4)
```

Empirically verified with Python 3.11 / PyTorch 2.10.0:

```
$ python3 -c "import math
def _logit(p): p=max(1e-7,min(1-1e-7,p)); return math.log(p/(1-p))
print(_logit(float('nan')))"
16.11809555148467
```

**Consequence.** A NaN intermediate value flows through the blend and emerges as a confident, extreme prediction (0.9999) for every in-vocab item in the round. This is the canonical "silent regression" pattern that the parent CS321M project's `silent-ncf-head-load-failure-via-full-module-pickle-2026-05-17.md` post-mortem and `distinguishable-defensive-fallbacks-2026-05-18.md` design rule were created to defend against. The README's `## D-3 / no-broad-except discipline` section explicitly cites those rules. The current implementation has a NaN→constant-0.9999 silent fallback that LOOKS like CAIMIRA is extremely confident, but is actually broken.

**Best defense the code has.** None at the in-vocab path. The EB-only path is NaN-immune because `EB.lookup_p` never returns NaN by construction (all table values are pre-clipped to `[0.05, 0.95]` at table build, and the `_sigmoid(logit(...) + ... )` blend at level 3 uses pre-clipped values). The defense lives only in `EB.fit_platt`'s `mean_y <= 0 or mean_y >= 1` guard, which doesn't address upstream NaN propagation. The `clip_for_predict` clamp is the silent-failure path, not a defense.

**Possible fix sketch (advisory):** insert a `math.isfinite(caimira_logit)` check between `CAIMIRA.caimira_logit(...)` and `_sigmoid(caimira_logit)`. On non-finite, raise (D-3 loud-fail) OR fall through to the EB-only path (graceful degradation; the EB fallback is distinguishable from constant-0.9999 because it varies with input). The README's `## Cold-start strategy` section already documents a structural distinction between in-vocab and EB-only paths — extending the in-vocab path to detect NaN and demote to EB-only would preserve the README's design intent.

---

## CRIT-ADV-2 — `__all__` declares `TabPFNPredictor` but no import statement adds it (P1, pre-existing in upstream merge but perpetuated by CAIMIRA commit)

**Severity:** P1
**Confidence:** 100 (mechanically verifiable)
**Component:** `src/torch_measure/models/__init__.py`
**Anchor:** `src/torch_measure/models/__init__.py:5-26` (imports), `:47` (`"TabPFNPredictor"` in `__all__`)

**Constructed scenario.** A consumer runs:

```python
from torch_measure.models import TabPFNPredictor  # ImportError
# OR
from torch_measure.models import *                # AttributeError on TabPFNPredictor
```

Verified the AttributeError mechanism on a minimal module reproduction:

```
$ python3 -c "import types; m=types.ModuleType('test'); m.__all__=['Foo']
for name in m.__all__:
    value = getattr(m, name)" 2>&1
AttributeError: module 'test' has no attribute 'Foo'
```

The `__init__.py` lacks `from torch_measure.models.tabpfn_predictor import TabPFNPredictor`, but `tabpfn_predictor.py` exists in the same directory (`ls src/torch_measure/models/tabpfn_predictor.py` confirms). The test file `tests/test_models/test_tabpfn_predictor.py` imports it via the submodule path (`from torch_measure.models.tabpfn_predictor import TabPFNPredictor`), so the test suite passes — but external consumers and anyone using `from torch_measure.models import *` will hit AttributeError.

**Provenance.** Pre-existing in upstream merge `777a3b1` per `git show 777a3b1:src/torch_measure/models/__init__.py`. The CAIMIRA commit `1e4ec1c` added the `"CAIMIRA"` entry to `__all__` immediately above the broken `"TabPFNPredictor"` line, so it's noisy at git-blame time but the author did not introduce the bug.

**Consequence.** Any package consumer doing the common `from package import *` glob import fails. For an upstream PR to `aims-foundations/torch_measure`, this is a CI failure waiting to happen. The CONTRIBUTING.md doesn't require a `from torch_measure.models import *` test, but the broken `__all__` violates the documented intent ("Measurement models: IRT, factor models, network models, and rotation utilities" module docstring).

**Best defense the code has.** None — `__all__` is purely declarative; Python doesn't validate it at module import. The existing tests don't exercise the glob-import path.

**Possible fix sketch (advisory):** add `from torch_measure.models.tabpfn_predictor import TabPFNPredictor` after the `bradley_terry` import. Note for the synthesis: this fix is **upstream-eligible** (applies to the new model files area) but the bug is pre-existing in upstream `aims-foundations/main` per the `git show 777a3b1` evidence. Whether to upstream-PR it depends on whether the upstream project deems it a regression.

---

## CRIT-ADV-3 — `metadata_bonus` overflow asymmetry favors strata 256+ over early strata (P2)

**Severity:** P2
**Confidence:** 100 (mechanically verifiable; exhibited below)
**Component:** `submission/labeling.py`
**Anchor:** `submission/labeling.py:131-141` (`_metadata_bonus`, `_update_stratum`), `:41` (`_MAX_STRATA = 256`)

**Constructed scenario.** The `_stratum_counts` dict has a hard cap of `_MAX_STRATA = 256` unique keys. The `_update_stratum` function only adds new keys when `len(_stratum_counts) < _MAX_STRATA`:

```python
def _update_stratum(ex):
    key = _stratum_key(ex)
    if key in _stratum_counts or len(_stratum_counts) < _MAX_STRATA:
        _stratum_counts[key] = _stratum_counts.get(key, 0) + 1
```

The `_metadata_bonus` function reads counts from the dict:

```python
def _metadata_bonus(ex):
    key = _stratum_key(ex)
    count = _stratum_counts.get(key, 0)
    return 0.20 / float(1 + count)
```

Result: once 256 unique strata exist in the dict, any subsequent unique stratum gets a `count = 0` → metadata_bonus = 0.20 (the MAXIMUM possible bonus). Meanwhile, the strata in the dict have count ≥ 1 → bonus ≤ 0.10. Empirically:

```
$ python3 << 'EOF'
_stratum_counts = {}
def upd(k):
    if k in _stratum_counts or len(_stratum_counts) < 256:
        _stratum_counts[k] = _stratum_counts.get(k, 0) + 1
def bonus(k):
    return 0.20 / float(1 + _stratum_counts.get(k, 0))
for i in range(256): upd(f"key_{i}")
upd("key_256")  # ignored — cap reached
print(f"bonus for key_0 (in dict, count=1): {bonus('key_0')}")   # 0.10
print(f"bonus for key_256 (not in dict):    {bonus('key_256')}")  # 0.20
EOF
bonus for key_0 (in dict, count=1): 0.1
bonus for key_256 (not in dict):    0.2
```

**Consequence.** In a 10K-candidate round where the platform's stratum diversity is high, the FIRST 256 strata get penalized over time while the 257th+ strata get rewarded. The acquisition function systematically favors LATE-arriving novel strata. Since hidden Codabench rounds may have category orderings that depend on the platform's internal sampling, this could systematically bias which examples get selected for labeling. The K=5-per-data-category constraint may or may not hit this cap (depends on the data category count), but the kit's spec is "10K items per round per Ed #160", so 256 strata is plausibly hit.

**Best defense the code has.** None — the cap is intentional (bounded memory) but the cap-behavior asymmetry is unintentional.

**Possible fix sketch (advisory):** when the dict is at cap, return the MEAN bonus across existing entries (not the max-by-default), OR initialize the count for overflow strata to the median existing count. Defer until empirically motivated.

---

## CRIT-ADV-4 — `build_zip.sh` awk filename parser fails on filenames with spaces (P2)

**Severity:** P2
**Confidence:** 100 (empirically verified)
**Component:** `submission/build_zip.sh`
**Anchor:** `submission/build_zip.sh:111` (nested check), `:119` (allowlist diff)

**Constructed scenario.** The post-build sanity check uses `awk 'NR > 3 && NF >= 4 {print $NF}'` to extract filenames from `unzip -l` output. `$NF` is the LAST field. For filenames containing spaces, `unzip -l` emits the filename with embedded spaces, and awk's field-splitting fragments it. Empirically:

```
$ echo "data" > "with spaces.py" && zip /tmp/s.zip "with spaces.py"
$ unzip -l /tmp/s.zip | awk 'NR > 3 && NF >= 4 {print $NF}'
spaces.py     # NOT "with spaces.py" — only the last whitespace-delimited token
```

**Consequence.** None of the 7 currently-required files (`model.py`, `labeling.py`, `models.txt`, `caimira_lite.py`, `caimira_lite.pt`, `caimira_lite.meta.json`, `eb_tables.json`) contain spaces, so the current allowlist diff passes. But a future maintainer adding a required file with a space in its name (or accidentally renaming an existing file) would cause the post-build sanity check to fail with a confusing "ZIP file list differs from allowlist" error showing fragmented names. The allowlist check is then non-helpful: it would say "expected: model.py, ..., expected_name.py" vs. "actual: model.py, ..., name.py" — leaving the operator to figure out the off-by-prefix.

Additionally, `awk 'NR > 3 && /\// {print $NF}'` on line 111 has the same flaw: a filename with spaces AND a `/` (e.g., a subdirectory escaping the `zip -j` flat behavior) would print only the last token, possibly omitting the slash and reporting a misleading "no nested paths" result.

**Best defense the code has.** Single-`.pt`-file check (`pt_count != 1` on line 96-99) is robust to space-bearing filenames because it works on the allowlist array directly, not on awk-parsed output.

**Possible fix sketch (advisory):** use `python -c "import zipfile; print('\n'.join(zipfile.ZipFile(sys.argv[1]).namelist()))"` instead of awk-parsed `unzip -l`. The Python `zipfile.ZipFile.namelist()` returns the exact internal names, no whitespace ambiguity.

---

## CRIT-ADV-5 — Platt shift is computed from EB-only logits but applied to CAIMIRA-EB blended logits — calibration mismatch (P1)

**Severity:** P1
**Confidence:** 75 (concrete scenario; consequence depends on CAIMIRA-vs-EB rank correlation per benchmark)
**Component:** `submission/model.py`, `submission/caimira_lite.py`
**Anchor:** `submission/model.py:252-256` (Platt application), `submission/caimira_lite.py:346-387` (`fit_platt`)

**Constructed scenario.** Inside `fit_platt`, the per-labeled-example logit is computed from the EB-only path:

```python
# submission/caimira_lite.py:368-369
p = self.lookup_p(subj_name, bench, cond)  # EB-only, no CAIMIRA blend
by_bench.setdefault(bench, []).append((_logit(p), float(ex["label"])))
```

The shift is then computed as:

```python
mean_y         = sum(ys) / len(ys)
mean_x         = sum(xs) / len(xs)        # mean of EB-ONLY logits
target_logit   = math.log(mean_y / (1 - mean_y))
shift          = target_logit - mean_x    # difference computed over EB-only baseline
```

At predict time, the shift is applied to the BLENDED `p`:

```python
# submission/model.py:250-256
item_embedding = _encode_item(item_content)
caimira_logit  = CAIMIRA.caimira_logit(subject_idx, item_embedding)
caimira_p      = _sigmoid(caimira_logit)
p = _blend_logits(caimira_p, eb_p, _BLEND_LAMBDA)    # blended p
# ...
slope, intercept = EB._platt[benchmark]
p = _sigmoid(slope * _logit(p) + intercept)          # shift applied to BLENDED p
```

If CAIMIRA's logit and EB's logit differ for the in-vocab subjects in `labeled`, the BLEND mean logit ≠ the EB-only mean logit. The shift is `target_logit - mean_x_EB`, which is the right magnitude to move EB-only predictions toward `mean_y`. Applied to the BLENDED prediction, it overshoots or undershoots depending on the sign of the CAIMIRA-vs-EB disagreement.

Quantitative example: suppose for benchmark X, the K=5 reveals all have label=1 (e.g., easy items from a strong model). EB-only logits are ~ logit(0.5) = 0 (mean prior). CAIMIRA-blended logits are ~ logit(0.85) = 1.73 (because CAIMIRA correctly predicts the strong model gets these right). target_logit for label=1 → guard skips (mean_y == 1.0). OK, that case is degenerate-protected.

Take a non-degenerate case: K=5 reveals with mean_y = 0.6. EB-only mean_x = logit(0.5) = 0. target_logit = logit(0.6) = 0.405. shift = 0.405 - 0 = 0.405. Now apply to BLENDED p: if blended_p for the in-vocab item is 0.85 (logit ≈ 1.73), the corrected logit is 1.73 + 0.405 = 2.135 → sigmoid → 0.894. The model now predicts 89.4% for an item where CAIMIRA already said 85% — the EB-derived "we're under-predicting" shift is applied to an already-confident CAIMIRA prediction, OVER-correcting.

The shift cap of ±1.5 logit limits the damage but does not eliminate it.

**Consequence.** Per the 2026-05-22 transfer postmortem, prior CAIMIRA m=5 submissions had the best local val_nll AND the worst hidden score. One hypothesis (per `pre-submission-transfer-audit-stress-test-gate-2026-05-22.md` hypothesis F) is that the K=5 reveal regime interacts with calibration in a way that hurts hidden score. The calibration-mismatch design here is a concrete instance of that hypothesis — the per-benchmark Platt is fit on EB-only logits but applied to the blended (CAIMIRA-leaning) prediction. The D-9 gate has not validated this on real-trained CAIMIRA artifacts; only on a smoke degenerate predictor.

**Best defense the code has.** The `±1.5` shift cap bounds the damage. The `mean_y <= 0 or mean_y >= 1` guard prevents the most degenerate case. The CAIMIRA-leaning blend weight `λ=0.6` is the conservative default per the README's Wave-2 Lane-B recommendation but is empirically unvalidated for transfer.

**Possible fix sketch (advisory):** fit the Platt shift using the BLENDED logit at the labeled examples (re-compute the blended `p` for each labeled example before fitting), OR use a different per-row calibration where in-vocab and OOV examples are calibrated separately. Defer until D-9 ablation produces empirical motivation; surface as `## Residual risk` for the synthesis.

---

## CRIT-ADV-6 — `or`-chain falsy fallthrough silently drops zero-valued priors (P2)

**Severity:** P2
**Confidence:** 100 (verified)
**Component:** `submission/caimira_lite.py`, `src/torch_measure/models/cold_start_lookup.py`
**Anchor:** `submission/caimira_lite.py:335-336` (lookup_p), `src/torch_measure/models/cold_start_lookup.py:320-321`

**Constructed scenario.** `EBLookup.lookup_p` uses Python's `or` short-circuit:

```python
subj_p  = self.subj.get(subj_name) or self._subj_ci.get(subj_name.lower())
bench_p = self.bench.get(benchmark) or self._bench_ci.get(benchmark.lower())
```

If `self.subj["X"] == 0.0`, the `or` evaluates `0.0` as falsy and falls through to the case-insensitive map. The lookup returns the case-insensitive value (potentially a DIFFERENT subject with the same lowercase name), not the legitimate `0.0`. Empirically:

```
$ python3 -c "d={'X': 0.0}; ci={'x': 0.7}; print(d.get('X') or ci.get('X'.lower()))"
0.7
```

**Consequence.** Bounded in practice by the table-build-time `_TABLE_CLIP_LO = 0.05` clip in `submission/train.py:266-269`, which writes probabilities clipped to `[0.05, 0.95]`. So the `or`-chain bug cannot fire via the production EB JSON path. BUT:

1. A hand-edited `eb_tables.json` (e.g., an ablation, a manual experiment, a debug session) with `subj["X"] = 0.0` would trigger silent fallthrough.
2. `name_aliases` mapping with `{X: ""}` would produce an empty-string canonical name. The aliases are stored as-is; if a build pipeline produces a zero/empty entry, the `or`-chain hides it.
3. A future maintainer who relaxes the `_TABLE_CLIP_LO` (e.g., for a fine-grained near-zero benchmark) would expose the bug.

**Best defense the code has.** The `_TABLE_CLIP_LO = 0.05` at table-build time prevents the production case. The `_CLIP_LO = 0.05` constant at predict time mirrors this. The bug is latent, not active.

**Possible fix sketch (advisory):** use `if subj_name in self.subj else self._subj_ci.get(subj_name.lower())` (or equivalently a chained `.get` with explicit `None` sentinel). Defer because the production path is structurally protected, but worth flagging because `cold_start_lookup.py` is upstream-eligible and may have callers who don't enforce the same table-build clip.

---

## CRIT-ADV-7 — Case-insensitive map overwrites with last-iterated value when subject names collide on lowercase (P2)

**Severity:** P2
**Confidence:** 100 (verified)
**Component:** `submission/caimira_lite.py`, `src/torch_measure/models/cold_start_lookup.py`
**Anchor:** `submission/caimira_lite.py:288-291` (case-insensitive view construction)

**Constructed scenario.** The constructor builds case-insensitive views by dict comprehension:

```python
self._subj_ci = {k.lower(): v for k, v in self.subj.items()}
```

If `self.subj` has two keys that differ only in case (e.g., `"GPT-4"` and `"gpt-4"`), the second iteration overwrites the first in `_subj_ci`. Empirically:

```
$ python3 -c "d={'GPT-4': 0.75, 'gpt-4': 0.50}; print({k.lower():v for k,v in d.items()})"
{'gpt-4': 0.5}
```

A `lookup_p("Gpt-4", ...)` query would:
1. Direct lookup `self.subj.get("Gpt-4")` → None.
2. Case-insensitive lookup `self._subj_ci.get("gpt-4")` → 0.50 (the overwritten value).

The correct answer (per the source-of-truth dict) is ambiguous, but `0.75` is what the user might expect because `"GPT-4"` was added first. The bug is that the case-insensitive view silently picks one of the colliding values based on iteration order — Python dict order is insertion-order-preserving, so the LATER-added entry wins.

**Consequence.** Bounded by the assumption that `subjects.parquet` doesn't have case-colliding display names. For the CS321M 909-subject catalog, this is plausibly true but not guaranteed. A future drift in the dataset (e.g., a benchmark adds "gpt-4" while another already has "GPT-4") would silently swap which prior is used.

**Best defense the code has.** The training pipeline uses `subject_display` from the HF registry as the canonical name. Plausibly stable across train/test per the README's "Cold-start strategy" section. But there's no explicit assertion enforcing case-uniqueness.

**Possible fix sketch (advisory):** add a `_validate_case_uniqueness(self.subj)` step in `__init__` that raises `ValueError` if multiple keys lowercase to the same value. Loud failure preferred over silent overwriting.

---

## CRIT-ADV-8 — `_logit(NaN)` and `_sigmoid(NaN)` produce silent finite outputs via Python max/min NaN semantics (P1 / D-3 violation)

**Severity:** P1
**Confidence:** 100 (verified)
**Component:** `submission/caimira_lite.py`, `submission/model.py`, `submission/train.py`
**Anchor:** `submission/caimira_lite.py:97-101` (`_logit`), `:103-108` (`_sigmoid`), `:115-117` (`clip_for_predict`)

**Constructed scenario.** Python's `max(1e-7, nan)` returns `1e-7` (the first non-NaN argument) due to NaN-comparison semantics (`nan > x` is always False). The `_logit` function:

```python
def _logit(p: float) -> float:
    p = max(1e-7, min(1 - 1e-7, p))    # NaN gets coerced to 1-1e-7 (via min) and then 1e-7 (via max)
    # Actually: min(1-1e-7, nan) returns 1-1e-7; max(1e-7, 1-1e-7) returns 1-1e-7
    # So p = 1 - 1e-7
    return math.log(p / (1 - p))       # = log((1-1e-7) / 1e-7) ≈ 16.118
```

Verified:

```
$ python3 -c "import math
def _logit(p): p=max(1e-7,min(1-1e-7,p)); return math.log(p/(1-p))
print(_logit(float('nan')))"
16.11809555148467
```

`clip_for_predict(nan)` similarly returns `0.9999`:

```
$ python3 -c "
def clip_for_predict(p):
    return float(max(1e-4, min(1 - 1e-4, float(p))))
print(clip_for_predict(float('nan')))"
0.9999
```

**Composition with CRIT-ADV-1.** A NaN in the predict path silently produces 0.9999. This is the cross-component instance of CRIT-ADV-1's broader claim — every helper that calls `max(lo, min(hi, x))` for clamping treats NaN as a constant-fallback.

**Consequence.** Any NaN that emerges (corrupted weights, inf in difficulty due to softmax saturating with bad weights, NaN in `_difficulty_mean` buffer if it was unrefreshed before save) silently produces 0.9999. The D-9 gate's sub-gate (a) would CATCH this (mean(p) ≈ 0.9999 deviates from base rate by > 0.05), but the README's gate run history explicitly notes sub-gate (a) "PASSes on a degenerate constant-0.604 predictor" — a flaw in the gate's `mean(p)` check that this finding compounds.

**Best defense the code has.** Sub-gate (a) of the D-9 gate (if run) would catch a constant-0.9999. But (a) does not catch this if NaN is sporadic rather than systematic (e.g., NaN only on rare subjects with bad skill vectors). Also: sub-gate (b) (packaged-runtime equivalence check) would catch this because the trainer's saved artifact would also produce NaN on the same input → no divergence; the divergence-based check passes despite the bug.

**Possible fix sketch (advisory):** add `if not math.isfinite(p): raise ValueError(f"Non-finite prediction: {p}")` immediately before `clip_for_predict`'s clamp, in BOTH `submission/model.py:258` and `submission/caimira_lite.py:117`. This converts the silent failure into a loud fail and forces the D-3 discipline at every clamping site. Alternative: use `math.nan` propagation via `if math.isnan(p): return float("nan")` paths — but the kit explicitly rejects NaN outputs, so loud-raise is correct.

---

## CRIT-ADV-9 — `acquisition_function` and `predict()` have asymmetric NaN/exception handling — same bad input causes silent low-priority labeling but loud predict crash (P2 / contract divergence)

**Severity:** P2
**Confidence:** 75
**Component:** `submission/labeling.py`, `submission/model.py`
**Anchor:** `submission/labeling.py:206-222` (try/except), `submission/model.py:185-258` (no try/except)

**Constructed scenario.** The kit calls `acquisition_function(input)` BEFORE `predict(input, labeled)` for each candidate. If a candidate has an input field whose `str()` raises (e.g., the synthetic `_Boom` class in `test_acquisition_function_bulletproof_sentinel_on_exception`):

- `acquisition_function`: catches via `except Exception: return 0.0` → distinguishable sentinel → candidate ranked LAST for labeling, NOT labeled.
- `predict(input, labeled)`: re-runs against the same candidate at predict time. `parse_subject_name(Boom())` → `re.match` raises `TypeError`. No try/except → exception propagates → submission fails.

Test reproduction (the test itself shows this):

```python
class _Boom:
    def __str__(self):
        raise RuntimeError("synthetic")
out = labeling.acquisition_function({"benchmark": _Boom()})
assert out == 0.0   # labeling returns sentinel
# But: model.predict({"benchmark": _Boom(), ...}) would re-raise TypeError.
```

**Cascade construction.** In a hosted round:
1. Platform streams a candidate with corrupted `benchmark` field (e.g., a Unicode encoding error in the upstream pipeline).
2. `acquisition_function` returns 0.0 → candidate is least-priority for labeling.
3. The platform may still issue `predict()` for this candidate (acquisition_function only ranks labeling priority, NOT whether to predict).
4. `predict()` raises → round fails → all subsequent predictions in the round are forfeit.

**Consequence.** A single bad-input candidate breaks the entire round. The D-3 discipline (loud-fail in predict) is the correct policy per the project's design pattern, but the asymmetry with `acquisition_function` (silent-sentinel) means an attacker controlling the input stream could selectively trigger predict() failures with inputs that the labeling phase would have skipped over.

**Best defense the code has.** Per kit contract, `input` is platform-controlled and always has the four string fields. A non-string field would be a platform bug, not a user attack. The D-3 discipline says loud-fail in predict() is correct because silent fallback to 0.5 historically masked a 3-week leaderboard regression.

**Possible fix sketch (advisory):** none — the asymmetry is intentional per the D-3 discipline and the distinguishable-defensive-fallbacks rule. This finding is mostly a "be aware" note for the synthesis: if hidden-leaderboard runs ever fail with `[PAIEC-PREDICT-002]` for some specific items, the asymmetry-with-acquisition could be a debugging signal.

---

## CRIT-ADV-10 — `_platt_fit_key` cache is by length only, not content — silently stale when label content drifts at same length (P2)

**Severity:** P2
**Confidence:** 50 (depends on whether kit contract guarantees monotonic labeled-list growth; defended by current kit docs but undocumented for edge cases)
**Component:** `submission/caimira_lite.py`, `src/torch_measure/models/cold_start_lookup.py`
**Anchor:** `submission/caimira_lite.py:353-356` (length-cache), `src/torch_measure/models/cold_start_lookup.py:412-415`

**Constructed scenario.** The Platt-fit cache invalidates only by `len(labeled)`:

```python
new_key = len(labeled)
if new_key == self._platt_fit_key:
    return
```

Per kit contract, `labeled` is monotonically growing (K=5 reveals per data category accumulated). But if the platform ever:
- Re-emits the same length with different content (e.g., a queue retry that re-sends an old labeled-list snapshot followed by a new one of the same length),
- OR a future kit revision changes labeled-list semantics to non-monotonic,

the cache would return STALE Platt shifts. Predictions for the new content would be calibrated against the OLD labeled-list's mean. The K=5 size means the calibration is already noisy; stale calibration is silently incorrect.

**Consequence.** Unverifiable from this diff. The kit docs at `starting_kit/README.md` describe the K=5-per-category monotonic accumulator model. If that holds, length-based caching is sound. If the platform ever ships a non-monotonic labeled stream, the cache silently corrupts.

**Best defense the code has.** None — the cache is faithful to the documented kit contract, but offers no defense against undocumented platform changes.

**Possible fix sketch (advisory):** cache by `(len(labeled), hash(tuple((ex.get("benchmark"), ex.get("label")) for ex in labeled)))`. Adds O(K) per fit_platt call, negligible vs. the existing per-fit O(K) loop. Defer until the platform contract surfaces a non-monotonic case.

---

## CRIT-ADV-11 — `fit_platt` raises ValueError on non-numeric `label` field — one corrupted reveal breaks the entire round (P1)

**Severity:** P1
**Confidence:** 100 (verified)
**Component:** `submission/caimira_lite.py`, `src/torch_measure/models/cold_start_lookup.py`
**Anchor:** `submission/caimira_lite.py:369` (`float(ex["label"])`), `src/torch_measure/models/cold_start_lookup.py:431`

**Constructed scenario.** Per the kit contract, `ex["label"]` is `∈ {0, 1}`. But the code does:

```python
by_bench.setdefault(bench, []).append((_logit(p), float(ex["label"])))
```

If `ex["label"]` is a string `"true"` or any non-numeric token (kit bug, platform pipeline corruption), `float(...)` raises `ValueError`. Empirically:

```
$ python3 -c "print(float('true'))"
ValueError: could not convert string to float: 'true'
```

**Cascade.** `fit_platt` is called from inside `predict()` (`submission/model.py:253`). If `fit_platt` raises, `predict()` raises. Per D-3, the round fails immediately, but `[PAIEC-PREDICT-002]` is generated for the FIRST predict() call that had a `labeled` list — which may be hundreds of predictions into the round. All preceding predictions are lost. The error message will reference the corrupted reveal, but the platform's two-tier error reporting (per `codabench-two-tier-error-reporting-paiec-system-2026-05-19.md`) may surface as the generic "No additional details are safe to show" depending on which stage the ValueError surfaces in.

**Consequence.** Bounded by the kit's strict label format. But the D-3 discipline says we PREFER loud-fail over silent corruption — so this is by design. The concern is amplification: one bad labeled row destroys all subsequent predict() calls in the round.

**Best defense the code has.** Per kit contract, the platform validates `labeled` before passing it. The `if "label" not in ex: continue` guard handles missing-label gracefully. The float() raise is the canonical loud-fail.

**Possible fix sketch (advisory):** add `try: y = float(ex["label"]) except (ValueError, TypeError): continue` to skip corrupted rows without breaking the round. This trades D-3 loud-fail for graceful degradation — design decision belongs to the synthesis. Note: per `distinguishable-defensive-fallbacks-2026-05-18.md` rule (1), the design preference is loud-fail. So this fix is anti-pattern under the current discipline; flag for awareness only.

---

## CRIT-ADV-12 — Test order leaks `submission/labeling.py` module state across tests; only one test resets state (P3 / testing gap)

**Severity:** P3
**Confidence:** 75
**Component:** `tests/test_submission/test_model_contract.py`, `tests/test_submission/test_caimira_lite.py`
**Anchor:** `tests/test_submission/test_model_contract.py:107-176` (`TestModelAcquisitionContract`)

**Constructed scenario.** `submission/labeling.py` has three module-level mutables:

```python
_seen_signatures: list[int] = []
_stratum_counts: dict[tuple[str, str, str], int] = {}
_candidate_count = 0
```

These are mutated by every `acquisition_function` call. The test class `TestModelAcquisitionContract` has four tests:
- `test_acquisition_function_signature` — calls nothing.
- `test_acquisition_function_returns_finite_real` — calls once.
- `test_acquisition_function_bulletproof_sentinel_on_exception` — calls once (with Boom input).
- `test_acquisition_function_distinguishable_from_legitimate_floor` — manually resets state, calls once.

Test ordering depends on pytest's collection order. If `test_acquisition_function_returns_finite_real` runs first (which it would per source order), it adds one entry to `_seen_signatures`. The second test (`test_acquisition_function_bulletproof_sentinel_on_exception`) doesn't reset — but its Boom input raises before mutating state, so state is unchanged. The third test (`test_acquisition_function_distinguishable_from_legitimate_floor`) DOES reset, but it's the LAST test, so its reset doesn't protect prior tests.

The asserting test (`> 0.0`) in `test_acquisition_function_distinguishable_from_legitimate_floor` would still pass even if `_seen_signatures` was non-empty, because the diversity score is `min(...) / 64.0 ≥ 0`. The bigger concern is determinism: if a future test adds an assertion on the diversity score's EXACT value, the leaked state would make it test-order-dependent.

**Consequence.** Minor — current tests don't observably fail. But the leak is a flaky-test risk for any future test that adds exact-value assertions on the score.

**Best defense the code has.** The `_load_model_module` helper in `test_model_contract.py:23-33` reloads `submission.model`; if a parallel test reload pattern existed for `labeling`, leaks would be eliminated. None of the tests use this pattern.

**Possible fix sketch (advisory):** add a `pytest fixture` with `autouse=True, scope="function"` that resets all three module-level globals before each test in `TestModelAcquisitionContract`. Or change `submission/labeling.py` to encapsulate state in a class instance.

---

## CRIT-ADV-13 — `caimira_logit` returns silent IndexError on stale meta.json subject_to_idx (P2)

**Severity:** P2
**Confidence:** 100 (verified)
**Component:** `submission/caimira_lite.py`, `submission/model.py`
**Anchor:** `submission/caimira_lite.py:226-243` (`caimira_logit`), `submission/model.py:235-250`

**Constructed scenario.** `submission/model.py:235-237`:

```python
subject_idx = SUBJECT_TO_IDX.get(subj_name)
if subject_idx is None:
    subject_idx = SUBJECT_TO_IDX.get(raw_name)
```

If `meta.json`'s `subject_to_idx` is STALE relative to the `.pt` file (e.g., a partial trainer run produced a `.pt` with `n_subjects=10` but a meta.json with `subject_to_idx` mapping 20 keys), the lookup could return `subject_idx = 15`, which is OUT OF RANGE for `self.skill[subject_idx]`:

```python
# submission/caimira_lite.py:241
skill = self.skill[subject_idx]
```

`torch.Tensor.__getitem__` with an out-of-range index raises `IndexError`. Verified:

```
$ python3 -c "import torch; t=torch.zeros(10,5); print(t[15])"
IndexError: index 15 is out of bounds for dimension 0 with size 10
```

**Consequence.** Per D-3, the round fails loudly. This is CORRECT D-3 behavior — but it surfaces as `[PAIEC-PREDICT-002]` and may consume a daily Codabench quota slot. The round-trip step in `submission/train.py:399-403` catches state_dict KEY mismatches but does NOT validate meta.json consistency with the state_dict's actual tensor shapes. A trainer crash mid-run (e.g., OOM after saving .pt but before writing meta.json) would produce an inconsistent pair.

**Best defense the code has.** The trainer writes both files in the SAME `main()` call. Atomicity is OS-level (not transactional). If a SIGKILL between `torch.save(model_cpu.state_dict(), pt_path)` (line 384) and `meta_path.write_text(json.dumps(metadata, indent=2))` (line 432) occurs, the artifacts are inconsistent.

**Possible fix sketch (advisory):** in `submission/model.py` module init, add a consistency assert:

```python
n_subjects_in_state = state["skill"].shape[0]
assert n_subjects == n_subjects_in_state, (
    f"meta.json says n_subjects={n_subjects} but state_dict has skill.shape[0]={n_subjects_in_state}"
)
assert max(SUBJECT_TO_IDX.values()) < n_subjects, "subject_to_idx references out-of-range index"
```

This converts a runtime IndexError on the Nth prediction into a loud module-init failure, preserving D-3 discipline AND localizing the error to module-init (PAIEC-PREDICT-002 fires before any predictions, which the kit/platform surface as a more diagnostic error than a mid-round failure).

---

## CRIT-ADV-14 — `fit_alpha_beta` returns positive log-likelihood named `fit_nll` (P3 / naming bug)

**Severity:** P3
**Confidence:** 100 (verified)
**Component:** `src/torch_measure/models/llm_judge_irt.py`
**Anchor:** `src/torch_measure/models/llm_judge_irt.py:215-221`

**Constructed scenario.** The method's docstring (`:175`) and return-variable name (`fit_nll`) say "NLL" (negative log-likelihood), but the computation is:

```python
# llm_judge_irt.py:220
fit_nll = float(np.mean(y_arr * np.log(p) + (1 - y_arr) * np.log(1 - p)))
```

This is the MEAN LOG-LIKELIHOOD (positive value, no leading negative sign). The actual NLL would be `-float(np.mean(...))`. The class is labeled experimental + NOT-FOR-PRODUCTION (per the module docstring's "Negative-result disclosure"), so no production code consumes this return value. But the docstring contract is wrong — a future caller treating `fit_nll` as a true NLL would interpret e.g. `-0.6` as "good fit" when it's actually "moderate fit" (a true NLL of `-0.6` is a fairly poor fit; the actual mean LL is `-0.6`, equivalent to NLL = `+0.6`).

**Consequence.** Documentation drift. The neg-result disclosure already says `+0.002 NLL lift` vs M3, but the actual return is a positive LL, so the +0.002 interpretation may also be inverted. No tests assert against the value.

**Best defense the code has.** Tests in `test_llm_judge_irt.py:227-230, 244-247` ignore the third return value with `_`. The model is marked experimental and NOT recommended for deployment.

**Possible fix sketch (advisory):** rename to `fit_ll` OR negate the value to produce true NLL. Either way, update the docstring. Defer because experimental.

---

## Residual risks

These are not concrete failure-scenario findings, but constructed risks that surface from the diff's design choices and merit synthesis attention:

- **R-1: `_BLEND_LAMBDA = 0.6` is empirically unvalidated for the hidden leaderboard.** Per the parent CS321M `2026-05-22 transfer postmortem`, prior CAIMIRA submissions had best local val_nll AND worst hidden score. The blend is a structural counter, but the lambda is fixed at the README's "conservative default per Wave-2 Lane-B recommendation" with explicit acknowledgment that D-9 ablation candidates `{0.3, 0.5, 0.7, 0.9}` have not been swept. Submitting without sweeping wastes daily Codabench quota.

- **R-2: D-9 gate's documented coverage gap on sub-gate (a) compounds with CRIT-ADV-1 and CRIT-ADV-8.** The README's gate run history explicitly notes that sub-gate (a) PASSes on degenerate constant-0.604 predictors. If a future training run produces a NaN-emitting artifact (the CRIT-ADV-1/8 scenario), sub-gate (a) would only catch it if the resulting near-constant 0.9999 prediction's mean deviates from base rate by more than 0.05. Per Codabench's typical base rates (~0.55), 0.9999 - 0.55 = 0.45 > 0.05 → sub-gate (a) WOULD catch this specific scenario. But for partial-NaN (NaN on some rows, valid on others), the mean might fall within tolerance while many rows are silently 0.9999. Sub-gate (c) stress NLL would catch the resulting bad-NLL, but per the README's D-9 gate run history, sub-gate (c) has only been smoke-validated against degenerate predictors.

- **R-3: Symbolic links in `submission/` would be followed by `zip -j`.** If a future maintainer creates a symlink (e.g., `submission/caimira_lite.pt` → `/some/large/checkpoint.pt`), `zip -j` follows symlinks by default. There's no `--no-symlinks` flag or pre-build check. Not a security vulnerability in a personal dev env, but a confusing failure mode for a future operator who symlinks for convenience.

- **R-4: `SentenceTransformer(ENCODER_REPO, device=str(DEVICE))` at module init has no network fallback.** Per kit contract, the encoder is pre-fetched into the HF cache by `models.txt`. But if the HF cache fetch fails silently (e.g., the network was briefly down during pre-fetch), module init raises at the encoder load, surfacing as `[PAIEC-PREDICT-002]`. This is the correct D-3 behavior, but the failure mode is non-obvious: the platform's stderr suppression policy (per `bug 21 in CONCERNS.md`) means an operator may not see the underlying HF cache miss.

- **R-5: `_seen_signatures` reservoir privileges the first 128 candidates over later ones.** Empirically verified (~6% replacement rate at 10K calls). For a round where the platform streams candidates in non-randomized order (e.g., sorted by category), this means early-category candidates dominate the diversity reservoir, and later-category candidates compute their diversity score against an unbalanced set. The acquisition function would then systematically prefer late-category candidates (because they're far from the early-category-heavy reservoir). For the kit's documented K=5-per-category model, this could subtly bias which categories get the most label reveals.

---

## Testing gaps

- **T-1: No test exercises NaN propagation through the predict path.** CRIT-ADV-1 and CRIT-ADV-8 surface a structural defect that no current test catches. A test that monkey-patches `CAIMIRA.caimira_logit` to return `float('nan')` and asserts that `predict()` either raises OR returns a non-extreme value would catch the silent-0.9999 path.

- **T-2: No test exercises stale-meta.json scenarios.** CRIT-ADV-13's IndexError path isn't covered. A test that loads a state_dict with mismatched `subject_to_idx` range would catch it. The current `TestCAIMIRALiteStateDictParity` only validates KEY-set parity, not VALUE-range consistency.

- **T-3: No test exercises labeled-list corruption.** CRIT-ADV-11's `float("true")` ValueError path is uncovered. A test that passes `[{..., "label": "true"}]` to `predict()` would catch the round-breaking behavior.

- **T-4: No test exercises `_MAX_STRATA = 256` cap behavior.** CRIT-ADV-3's overflow-favors-late-strata behavior is unverified by the test suite. A test that adds 257 unique strata and asserts that the 257th gets the same metadata_bonus as the 1st would catch the asymmetry.

- **T-5: `from torch_measure.models import *` is not in the test matrix.** CRIT-ADV-2's `TabPFNPredictor` AttributeError would be caught by a one-line glob-import test.

- **T-6: Build_zip.sh post-build sanity is not tested end-to-end.** The awk-parsing brittleness (CRIT-ADV-4) is uncovered. A bash integration test that builds a fake submission/ with a spaces-in-filename and asserts the post-build check fails LOUDLY (not silently passes the fragmented allowlist diff) would catch the regression.

---

## Output schema

```json
{
  "reviewer": "adversarial",
  "findings": [
    {"id": "CRIT-ADV-1",  "severity": "P1", "confidence": 75,  "title": "NaN in CAIMIRA forward silently coerces to confident 0.9999 (D-3 violation)",                                                "owner": "human", "autofix_class": "advisory"},
    {"id": "CRIT-ADV-2",  "severity": "P1", "confidence": 100, "title": "__all__ declares TabPFNPredictor but no import statement adds it (pre-existing upstream, perpetuated by CAIMIRA commit)",   "owner": "human", "autofix_class": "manual"},
    {"id": "CRIT-ADV-3",  "severity": "P2", "confidence": 100, "title": "metadata_bonus overflow asymmetry favors strata 256+ over early strata",                                                    "owner": "human", "autofix_class": "advisory"},
    {"id": "CRIT-ADV-4",  "severity": "P2", "confidence": 100, "title": "build_zip.sh awk filename parser fails on filenames with spaces",                                                          "owner": "human", "autofix_class": "advisory"},
    {"id": "CRIT-ADV-5",  "severity": "P1", "confidence": 75,  "title": "Platt shift is computed from EB-only logits but applied to CAIMIRA-EB blended logits — calibration mismatch",              "owner": "human", "autofix_class": "advisory"},
    {"id": "CRIT-ADV-6",  "severity": "P2", "confidence": 100, "title": "or-chain falsy fallthrough silently drops zero-valued priors",                                                              "owner": "human", "autofix_class": "advisory"},
    {"id": "CRIT-ADV-7",  "severity": "P2", "confidence": 100, "title": "Case-insensitive map overwrites with last-iterated value when subject names collide on lowercase",                          "owner": "human", "autofix_class": "advisory"},
    {"id": "CRIT-ADV-8",  "severity": "P1", "confidence": 100, "title": "_logit(NaN) and _sigmoid(NaN) produce silent finite outputs via Python max/min NaN semantics (D-3 violation)",              "owner": "human", "autofix_class": "advisory"},
    {"id": "CRIT-ADV-9",  "severity": "P2", "confidence": 75,  "title": "acquisition_function and predict() have asymmetric NaN/exception handling — silent labeling sentinel vs loud predict crash","owner": "human", "autofix_class": "advisory"},
    {"id": "CRIT-ADV-10", "severity": "P2", "confidence": 50,  "title": "_platt_fit_key cache is by length only, not content — silently stale when label content drifts at same length",            "owner": "human", "autofix_class": "advisory"},
    {"id": "CRIT-ADV-11", "severity": "P1", "confidence": 100, "title": "fit_platt raises ValueError on non-numeric label field — one corrupted reveal breaks the entire round",                    "owner": "human", "autofix_class": "advisory"},
    {"id": "CRIT-ADV-12", "severity": "P3", "confidence": 75,  "title": "Test order leaks submission/labeling.py module state across tests; only one test resets state",                            "owner": "human", "autofix_class": "manual"},
    {"id": "CRIT-ADV-13", "severity": "P2", "confidence": 100, "title": "caimira_logit returns silent IndexError on stale meta.json subject_to_idx",                                                "owner": "human", "autofix_class": "advisory"},
    {"id": "CRIT-ADV-14", "severity": "P3", "confidence": 100, "title": "fit_alpha_beta returns positive log-likelihood named fit_nll (naming bug)",                                                "owner": "human", "autofix_class": "manual"}
  ],
  "residual_risks": [
    "R-1: _BLEND_LAMBDA=0.6 is empirically unvalidated for hidden leaderboard",
    "R-2: D-9 gate sub-gate (a) coverage gap compounds with CRIT-ADV-1/8 NaN cascade",
    "R-3: zip -j follows symlinks by default; no pre-build check in build_zip.sh",
    "R-4: SentenceTransformer load at module init has no network fallback; surfaces as opaque PAIEC-PREDICT-002 under platform stderr suppression",
    "R-5: _seen_signatures reservoir privileges first 128 candidates; non-randomized stream order systematically biases acquisition"
  ],
  "testing_gaps": [
    "T-1: No test exercises NaN propagation through the predict path",
    "T-2: No test exercises stale-meta.json scenarios (subject_to_idx range vs state_dict skill.shape[0])",
    "T-3: No test exercises labeled-list corruption (non-numeric label field)",
    "T-4: No test exercises _MAX_STRATA=256 cap behavior",
    "T-5: from torch_measure.models import * is not in the test matrix (would catch TabPFNPredictor)",
    "T-6: build_zip.sh post-build sanity is not tested end-to-end (awk-parsing brittleness)"
  ]
}
```
