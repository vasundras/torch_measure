# CRITIC_3 — FN catalog (fresh-sweep, round 1)

```json
{
  "critic": "C3_fn_catalog",
  "round": 1,
  "artifact_id": "Branch feat/caimira (vasundras/torch_measure)",
  "fresh_sweep_complete": true,
  "candidate_fns": [
    {
      "fn_id": "FN-V1-1",
      "title": "_TOKEN_RE = r'[a-z0-9]+' strips all non-ASCII content from SimHash signatures; CJK/Cyrillic/Arabic items collapse to identical '<empty>' signature",
      "severity": "P2",
      "anchor": "submission/labeling.py:43 (`_TOKEN_RE = re.compile(r'[a-z0-9]+')`) + :65-66 (`_tokens` uses `_TOKEN_RE.findall(...)`) + :82-97 (`_simhash` uses `_tokens(ex)`) + :85-86 (`if not toks: toks = ['<empty>']`)",
      "rationale": "Mental execution: an item_content like a Chinese-language math question ('如果x+y=10且x-y=2,求x') is lowercased then run through `_TOKEN_RE.findall(...)`. The regex `[a-z0-9]+` matches only ASCII a-z and 0-9 — NO Unicode word characters. The result is `[]` (empty list). Then `_simhash` falls to the `['<empty>']` sentinel branch — EVERY non-ASCII-bearing item gets the same SimHash signature. Pairwise Hamming distance is 0, so `_diversity_score(signature)` returns 0/64 = 0.0 between any two non-ASCII items. Acquisition score collapses to `0 + metadata_bonus + tie_break` ≈ 0.01-0.20, which the platform may rank below ASCII items. The aims-foundations/measurement-db dataset includes benchmarks with non-Latin content (e.g., MMLU has multilingual splits historically); for those rows, the diversity signal is null. Per the parent CS321M plan, the 10K-candidate-per-round path is precisely where this matters — diversity-weighted labeling is the contract. Not corroborated by any reviewer.",
      "classification": "GENUINE-FN",
      "corroborating_reviewer": null
    },
    {
      "fn_id": "FN-V1-2",
      "title": "submission/train.py _build_wide_response silently collapses multi-condition responses for the same (subject_id, item_id) — last-write-wins on duplicate index assignment",
      "severity": "P1",
      "anchor": "submission/train.py:444-454 (`_build_wide_response`) + :190-201 (`build_indices` keys items by content) + :171-178 (`collect_binary_examples` emits row-per-condition)",
      "rationale": "Mental execution: for benchmark X, subject 'gpt-4' answers item 'What is 2+2?' under conditions {'zero-shot': 0, 'cot': 1}. `collect_binary_examples` produces TWO examples with identical `(subject_name, item_content)` but different `(condition, label)`. `build_indices` maps both to the same `(s, i)`. `_build_wide_response` does `wide[s, i] = y` with duplicate indices — PyTorch's docs say this is non-deterministic on CUDA; on CPU it takes the last value. The order depends on parquet row-iteration, which is platform-defined. The CAIMIRA training data is silently halved (or worse) for items measured under multiple conditions. EB tables DO carry condition signal (via `sbc[s||b||c]` keys), but CAIMIRA — the load-bearing predictor at λ=0.6 — trains on collapsed data.",
      "classification": "CORROBORATED-BY-REVIEWER",
      "corroborating_reviewer": "correctness (CRIT-CORR-1)"
    },
    {
      "fn_id": "FN-V1-3",
      "title": "NaN propagation through clip_for_predict and _logit produces a silent confident-0.9999 output (Python max/min NaN semantics)",
      "severity": "P1",
      "anchor": "submission/caimira_lite.py:97-101 (`_logit`) + :115-117 (`clip_for_predict`) + submission/model.py:258 (final return path)",
      "rationale": "Mental execution: `_logit(nan)` calls `max(1e-7, min(1-1e-7, nan))`. Per Python docs, `min(a, b)` with NaN returns the first non-NaN argument; so `min(1-1e-7, nan) = 1-1e-7`, `max(1e-7, 1-1e-7) = 1-1e-7`. `_logit` then computes `log((1-1e-7)/1e-7) ≈ 16.118`. So NaN coerces to a CONFIDENT FINITE LOGIT. `_blend_logits(nan_caimira_p, eb_p, 0.6)` yields `_sigmoid(0.6 * 16.118 + 0.4 * _logit(eb_p)) ≈ 0.9999`. Then `clip_for_predict(0.9999) = 0.9999`. A NaN in trained weights silently produces extreme confident predictions for every in-vocab item — the exact silent-failure pattern the D-3 discipline + the silent-NCF-head-load post-mortem warn against. Not specific to the EB-only fallback (which is structurally NaN-immune via table clipping).",
      "classification": "CORROBORATED-BY-REVIEWER",
      "corroborating_reviewer": "correctness (CRIT-CORR-2), adversarial (CRIT-ADV-1, CRIT-ADV-8)"
    },
    {
      "fn_id": "FN-V1-4",
      "title": "EBLookup.lookup_p `or`-chain falsy fallthrough silently drops legitimate 0.0 priors",
      "severity": "P2",
      "anchor": "submission/caimira_lite.py:335-336 (and src/torch_measure/models/cold_start_lookup.py:320-321) — `subj_p = self.subj.get(subj_name) or self._subj_ci.get(subj_name.lower())`",
      "rationale": "Mental execution: if `self.subj['X'] = 0.0` (legitimate empirical prior for a never-correct subject), `self.subj.get('X')` returns 0.0, the `or` short-circuits to `self._subj_ci.get('x')`, which may return a DIFFERENT value (case-distinct entry) or None. Bounded in production by the `_TABLE_CLIP_LO = 0.05` at table-build time, but the helper is upstream-eligible (ColdStartLookupPredictor is in __init__.__all__), so a third-party caller constructing an EBLookup with `subj={'X': 0.0}` would silently lose the direct hit. Same bug duplicated across two files — fixing one without the other adds drift.",
      "classification": "CORROBORATED-BY-REVIEWER",
      "corroborating_reviewer": "correctness (CRIT-CORR-5), adversarial (CRIT-ADV-6)"
    },
    {
      "fn_id": "FN-V1-5",
      "title": "submission/model.py defines its own _logit/_sigmoid with per-call `import math`, duplicating caimira_lite versions already imported in the same file",
      "severity": "P3",
      "anchor": "submission/model.py:72-78 (imports `EMBED_DIM, CAIMIRALite, EBLookup, clip_for_predict, parse_subject_name` from caimira_lite) + :97-112 (defines own `_logit`, `_sigmoid` each doing `import math` inside the function body)",
      "rationale": "Three sites carry the same `_logit`/`_sigmoid` math: cold_start_lookup.py:63-74, caimira_lite.py:97-108, and model.py:97-112. Each pair of files is byte-identical. The model.py version uniquely does `import math` INSIDE the function body — a code smell suggesting the author was unsure whether `math` was top-level imported (it isn't; model.py imports only json/os/sys/pathlib/typing/torch at the top). Per-call import is cheap (just a `sys.modules` dict lookup), but signals reader confusion. A future contributor who adds `math.isnan` to predict() may re-import or move to module-level, exacerbating the inconsistency.",
      "classification": "CORROBORATED-BY-REVIEWER",
      "corroborating_reviewer": "maintainability (CRIT-MAINT-2), kieran-python (CRIT-PY-3)"
    },
    {
      "fn_id": "FN-V1-6",
      "title": "META['subject_to_idx'] could be stale relative to saved skill.shape[0] — no consistency check at module init",
      "severity": "P2",
      "anchor": "submission/model.py:141-161 (META load + CAIMIRALite init + state load) + submission/train.py:384-432 (writes .pt at line 384, meta.json at line 432; no atomicity guarantee)",
      "rationale": "Mental execution: trainer SIGKILL or partial-write between `torch.save(state_dict)` (line 384) and `meta_path.write_text(json.dumps(metadata))` (line 432) produces an inconsistent pair. At load time, `META['subject_to_idx']` could reference idx=15 while state['skill'].shape[0]=10. The lookup at runtime would do `SUBJECT_TO_IDX.get('subject_15') -> 15`, then `CAIMIRA.skill[15]` raises IndexError at the Nth prediction. Per D-3, the round fails loudly — but only mid-round, consuming a daily quota slot. A consistency assert at module init would surface the failure during the (cheap) load phase, before the platform begins iterating predictions.",
      "classification": "CORROBORATED-BY-REVIEWER",
      "corroborating_reviewer": "adversarial (CRIT-ADV-13)"
    },
    {
      "fn_id": "FN-V1-7",
      "title": "EBLookup.fit_platt cache key is `len(labeled)` only — same length but different content silently reuses stale calibrator",
      "severity": "P2",
      "anchor": "submission/caimira_lite.py:353-357 (`new_key = len(labeled); if new_key == self._platt_fit_key: return`) and src/torch_measure/models/cold_start_lookup.py:412-415",
      "rationale": "Mental execution: kit guarantees monotonic growth of `labeled` per round (K=5 reveals appended per category). But the docstring claims 'idempotent if called with the same labeled list' — actually it's idempotent on SAME LENGTH. If the platform ever passes two different lists of the same length (queue retry, kit revision changes semantics, test harness reuses a list ref), the calibrator silently stays stale. The docstring is misleading: 'cached by length' would be more precise. Within current kit contract this is benign — but the documented contract leans on the K=5 monotonicity assumption that isn't re-verified by tests.",
      "classification": "CORROBORATED-BY-REVIEWER",
      "corroborating_reviewer": "adversarial (CRIT-ADV-10), kieran-python (CRIT-PY-12)"
    },
    {
      "fn_id": "FN-V1-8",
      "title": "_metadata_bonus overflow asymmetry favors strata 256+ over existing strata once cap is reached",
      "severity": "P2",
      "anchor": "submission/labeling.py:131-140 (`_metadata_bonus` reads count; `_update_stratum` skips new keys at cap) + :41 (`_MAX_STRATA = 256`)",
      "rationale": "Mental execution: `_update_stratum` only inserts new keys when `len(_stratum_counts) < _MAX_STRATA`. Once 256 strata exist, the 257th+ never enters the dict; `_metadata_bonus` for them returns `0.20 / (1 + 0) = 0.20` — the maximum possible bonus. Existing strata at count >= 1 get bonus <= 0.10. Acquisition systematically favors LATE-arriving novel strata over early-arriving ones. For 10K-candidate rounds streamed in some platform-defined order, this biases label budget toward later candidates — a subtle quality issue.",
      "classification": "CORROBORATED-BY-REVIEWER",
      "corroborating_reviewer": "adversarial (CRIT-ADV-3)"
    },
    {
      "fn_id": "FN-V1-9",
      "title": "fit_platt's `float(ex['label'])` raises ValueError on non-numeric label — one corrupted reveal in `labeled` breaks the entire round at mid-flight",
      "severity": "P1",
      "anchor": "submission/caimira_lite.py:369 (`by_bench.setdefault(...).append((_logit(p), float(ex['label'])))`) + src/torch_measure/models/cold_start_lookup.py:431",
      "rationale": "Mental execution: kit contract is `label ∈ {0, 1}`. But if the platform ever surfaces a string label ('true', 'yes') or `None`, `float(...)` raises. fit_platt is called from inside `model.predict()` (model.py:253) on EVERY predict call where `labeled` is truthy. The first call after the bad row enters `labeled` raises ValueError → submission fails mid-round → all subsequent predictions lost. Per D-3 this is correct loud-fail, but the amplification is uniquely bad: one corrupted reveal destroys all remaining predict() outputs. Trivial defensive guard (try/except around the `float(...)` per row) would scope the damage to a single skipped reveal.",
      "classification": "CORROBORATED-BY-REVIEWER",
      "corroborating_reviewer": "adversarial (CRIT-ADV-11)"
    },
    {
      "fn_id": "FN-V1-10",
      "title": "Platt shift fit from EB-only logits is applied to the CAIMIRA+EB blended logit — calibration mismatch when CAIMIRA and EB rank-disagree",
      "severity": "P1",
      "anchor": "submission/caimira_lite.py:346-387 (`fit_platt` uses `self.lookup_p` — EB-only logit) + submission/model.py:252-256 (applies shift to the BLENDED `p`)",
      "rationale": "Mental execution: fit_platt computes `mean_x` from EB-only logits at the K=5 reveals. Shift = `target_logit - mean_x_EB`. At predict time, the shift is applied to the CAIMIRA-blended logit. If CAIMIRA's per-row logit differs from EB's per-row logit (which is the WHOLE POINT of the hybrid — CAIMIRA adds item-content signal that EB lacks), the shift miscalibrates the blended prediction. The ±1.5 cap bounds the damage but does not eliminate it. This is exactly hypothesis F ('K=5 reveal regime interaction') from the D-9 transfer-audit pattern doc the parent CLAUDE.md cites — and the 2026-05-22 postmortem signature (best local val_nll, worst hidden score) is consistent with this class of bug.",
      "classification": "CORROBORATED-BY-REVIEWER",
      "corroborating_reviewer": "adversarial (CRIT-ADV-5)"
    },
    {
      "fn_id": "FN-V1-11",
      "title": "CAIMIRALite.skill initializes as torch.zeros(...) but upstream CAIMIRA uses torch.randn(...) * 0.01 — divergent untrained behavior masked by state-dict-load overwrite",
      "severity": "P3",
      "anchor": "submission/caimira_lite.py:221 (`self.skill = nn.Parameter(torch.zeros(n_subjects, latent_dim))`) vs src/torch_measure/models/caimira.py:126 (`nn.Parameter(torch.randn(n_subjects, latent_dim, device=self._device) * 0.01)`)",
      "rationale": "Mental execution: an untrained CAIMIRALite produces `skill[i] = 0` for all i, so `(skill - difficulty) * relevance` reduces to `-difficulty * relevance`. The forward output is determined entirely by item content (difficulty + relevance), not by subject identity. Tests at `test_caimira_logit_returns_float` (line 73-79) instantiate without loading state and pass trivially because the assertion is just `-100 < out < 100`. The state-dict parity test at `test_state_dict_round_trips` (line 46-63) DOES overwrite with `fill_(0.3)` and asserts the lite class's state matches, but never asserts that an UNTRAINED CAIMIRA's forward equals an UNTRAINED CAIMIRALite's forward — they would differ because of the init divergence. Not load-bearing in production (always preceded by state_dict load) but the test-fixture choice masks the design divergence. No reviewer surfaced this.",
      "classification": "GENUINE-FN",
      "corroborating_reviewer": null
    },
    {
      "fn_id": "FN-V1-12",
      "title": "Test fixture `test_predict_matches_manual_caimira_equation` uses zeros-buffer (untrained) — passes by self-consistency on zeros, doesn't validate paper-equation against a published reference",
      "severity": "P3",
      "anchor": "tests/test_models/test_caimira.py:127-149 (`test_predict_matches_manual_caimira_equation`)",
      "rationale": "Mental execution: the test does `model.eval()` BEFORE training, so `_difficulty_mean` buffer is its constructor-default `torch.zeros(latent_dim)`. The `compute_item_params(center='frozen')` subtracts zero → equivalent to no centering. The test verifies that `predict()` matches `sigmoid((skill - difficulty) * relevance).sum(-1)` row by row — but BOTH sides compute the same equation by the same code path, so this is essentially a tautology checking that `predict()` calls `compute_item_params()` and applies the equation. To verify paper-faithfulness, the test would need to compare against an independent implementation (e.g., a hand-coded numpy reference of `(s_i - d_j)^T r_j`) OR test on a non-trivial buffer (post-fit). Per External Research §5b, the paper Eq. 7 may specify FROZEN-bank centering as the paper-faithful default — but the diff's docstring claims the paper is silent on dynamic-vs-frozen. A test against a paper-faithful manual computation would be the structural defense. No reviewer surfaced this specifically as a paper-equation-verification gap (testing reviewer covered the broader 'forward parity' gap but not the self-consistency-tautology framing).",
      "classification": "GENUINE-FN",
      "corroborating_reviewer": null
    },
    {
      "fn_id": "FN-V1-13",
      "title": "TabPFNPredictor in __all__ but no `from .tabpfn_predictor import TabPFNPredictor` — `from torch_measure.models import *` raises AttributeError",
      "severity": "P1",
      "anchor": "src/torch_measure/models/__init__.py:5-26 (imports) + :47 (`'TabPFNPredictor'` listed in __all__)",
      "rationale": "Mental execution: Python's `from module import *` iterates `__all__` and does `getattr(module, name)` for each. If a name is in `__all__` but no corresponding import exists, AttributeError is raised. `tabpfn_predictor.py` exists in the directory (verified via `ls`); the test file imports it via submodule path (`from torch_measure.models.tabpfn_predictor import TabPFNPredictor`). External consumers doing `from torch_measure.models import *` would fail. This bug is PRE-EXISTING on `main` (verified via `git show main:.../models/__init__.py`) but the CAIMIRA commit `1e4ec1c` added the `'CAIMIRA'` entry immediately above the broken `'TabPFNPredictor'` line, perpetuating the bug visibly. Not strictly introduced by this diff, but visible in the diff's changes to __init__.py.",
      "classification": "CORROBORATED-BY-REVIEWER",
      "corroborating_reviewer": "adversarial (CRIT-ADV-2)"
    },
    {
      "fn_id": "FN-V1-14",
      "title": "submission/model.py PAIEC-PREDICT-002 docstring claim is wrong — that code tags predict() output violations, not module-init failures",
      "severity": "P1",
      "anchor": "submission/model.py:17-27 (docstring claims module init failures surface as `[PAIEC-PREDICT-002]`) + submission/README.md:135-136 (same claim)",
      "rationale": "Per the parent CS321M `docs/solutions/design-patterns/codabench-two-tier-error-reporting-paiec-system-2026-05-19.md`, the verbatim Codabench output for `[PAIEC-PREDICT-002]` is `'Invalid predict() output: predict() must return a finite float in [0, 1].'` — this code is the PRE-VALIDATOR error for `predict()` return-value violations (NaN/inf/non-float/out-of-domain). Module-init failures (missing artifact, torch.load failure, JSON parse failure) historically surface as the GENERIC banner (`'No additional details are safe to show'`), not `[PAIEC-PREDICT-002]`. A future operator triaging a `Failed` submission from this scaffolding would search the platform logs for `[PAIEC-PREDICT-002]`, fail to find it on a module-init failure, and misroute the triage. The docstring + README both need amending. Has confirmed paper-trail in the canonical pattern doc.",
      "classification": "CORROBORATED-BY-REVIEWER",
      "corroborating_reviewer": "learnings_research (CRIT-LEARN-6)"
    },
    {
      "fn_id": "FN-V1-15",
      "title": "Missing Sphinx doc entries for CAIMIRA, ColdStartLookupPredictor, LLMJudgeIRT, build_difficulty_prompt — CONTRIBUTING.md hard requirement violation if upstreaming",
      "severity": "P1",
      "anchor": "src/torch_measure/models/__init__.py:13, 14, 17 (new public exports) — no corresponding additions in docs/source/api/models.rst",
      "rationale": "Per CONTRIBUTING.md (cited in primer), 'All new public APIs need: A numpy-style docstring with a short usage example; A corresponding entry in the Sphinx docs under `docs/source/`'. The diff adds 4 new public symbols to `__all__` but does NOT add any `.. autoclass::` / `.. autofunction::` blocks to `docs/source/api/models.rst`. Mechanically verifiable absence; CONTRIBUTING.md hard requirement. Blocks an upstream PR to aims-foundations/torch_measure.",
      "classification": "CORROBORATED-BY-REVIEWER",
      "corroborating_reviewer": "project_standards (CRIT-STD-1, CRIT-STD-2)"
    }
  ],
  "summary": {
    "candidates_total": 15,
    "genuine_fns": 3,
    "corroborated": 12
  }
}
```

## Fresh-sweep methodology notes

The fresh sweep covered the full new artifact (15 source/test files; ~2530 LoC of new Python + 139 LoC of bash + 183 LoC of README) before reading any REVIEW_*.md output. Mental execution focused on:

- **Cross-file invariants** — the duplicated `_logit`/`_sigmoid` triplet (model.py, caimira_lite.py, cold_start_lookup.py) with three independent implementations.
- **Numerical stability** — NaN coercion via Python `max(lo, min(hi, nan))` semantics; the `_clamp_score` infinity-guard in labeling.py; the table-clip lower bound that masks the `or`-chain bug.
- **Concurrency / reentrancy** — `_item_cache`, `_seen_signatures`, `_stratum_counts`, `_candidate_count` module-level globals safe under the single-threaded Codabench container.
- **Error propagation** — fit_platt's `float(ex['label'])` raises mid-round; CAIMIRA→clip_for_predict silently coerces NaN to 0.9999.
- **Reproducibility** — `set_seed` in train.py covers random/numpy/torch; deterministic reservoir sampling in labeling.py.
- **Backwards compatibility** — state_dict layout parity (verified by tests); meta.json field drift (`embed_dim` uses constant instead of `embeddings.shape[1]`).
- **Documentation vs implementation drift** — PAIEC-PREDICT-002 claim contradicts canonical pattern doc; `_BLEND_LAMBDA` referenced three ways; fit_platt cache docstring vs actual behavior.
- **Test fixture quality** — `test_predict_matches_manual_caimira_equation` self-consistent but tautological; `test_acquisition_function_*` doesn't reset module-level globals across tests.
- **Edge cases** — Unicode handling in `_TOKEN_RE`; legitimate 0.0 priors blocked by `or` chains; collision when `subject_to_idx` is stale vs `state['skill'].shape[0]`.

### Genuine FNs (not surfaced by any reviewer)

- **FN-V1-1** (P2): Non-ASCII content collapses to a single `<empty>` signature in `_simhash`. Material for multilingual benchmarks.
- **FN-V1-11** (P3): `CAIMIRALite.skill` initialized as zeros while upstream uses small-randn. Divergent untrained behavior masked by state-dict load.
- **FN-V1-12** (P3): `test_predict_matches_manual_caimira_equation` is tautological (both sides compute the same equation by the same code path; doesn't validate against an independent paper-equation reference).

### Corroborated candidates (overlap with reviewers)

Twelve of fifteen candidates were independently surfaced by Wave 1 reviewers, confirming the fresh-sweep's calibration against the reviewer corpus. The strongest convergence was on:

- **NaN-to-confident-0.9999 silent path** — surfaced by correctness, adversarial (twice), reinforcing P0/P1 severity.
- **Multi-condition response collapse in trainer** — surfaced by correctness (CRIT-CORR-1) with the strongest evidence trace.
- **Triplicated `_logit`/`_sigmoid` helpers** — independently raised by maintainability and kieran-python.
- **Platt shift calibration mismatch (EB-only fit, blended apply)** — uniquely raised by adversarial (CRIT-ADV-5); the most consequential silent-degradation hypothesis given the 2026-05-22 transfer postmortem signature.
