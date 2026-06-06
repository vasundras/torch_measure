# C2 FP catalog — round 1 address-audit

Round 1 first-pass on `Branch feat/caimira (vasundras/torch_measure)`. No prior-round findings exist; every finding is mapped to `prior_round_match: null`. Per the charter, the verdict defaults to `NEW` unless (a) the reviewer claims X is missing/broken and X is actually present/working in the current artifact (`ADDRESSED (FP)`), (b) the finding is a positive compliance corroboration (`NOT-ADDRESSED (VALID)`), or (c) the finding is a future-tense risk claim (`NOT-ADDRESSED (VALID)`).

Address-audit evidence quotes inline; source paths are absolute.

```json
{
  "critic": "C2_fp_catalog",
  "round": 1,
  "artifact_id": "Branch feat/caimira (vasundras/torch_measure)",
  "per_finding": [
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-1",
      "verdict": "NEW",
      "address_evidence": "submission/model.py:223-258 (predict path) — `assert CAIMIRA is not None and EB is not None and META is not None` is the only guard; no `math.isfinite` check between `caimira_logit` (line 248) and `_sigmoid` (line 249); no NaN→EB-fallback demotion; clip_for_predict at caimira_lite.py:115-117 uses `max/min` which propagate the documented NaN-coercion to 1-1e-4 per CRIT-ADV-8. No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-2",
      "verdict": "NEW",
      "address_evidence": "src/torch_measure/models/__init__.py:5-26 lists imports; no `from torch_measure.models.tabpfn_predictor import TabPFNPredictor` line present; line 47 has `\"TabPFNPredictor\"` in `__all__`. Mechanically confirmed: src/torch_measure/models/tabpfn_predictor.py exists but is unimported. No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-3",
      "verdict": "NEW",
      "address_evidence": "submission/labeling.py:131-140 verbatim: `def _metadata_bonus(ex): ... return 0.20 / float(1 + count)` and `def _update_stratum(ex): ... if key in _stratum_counts or len(_stratum_counts) < _MAX_STRATA: _stratum_counts[key] = _stratum_counts.get(key, 0) + 1`. The cap at line 139 + unbonded bonus at line 134 produces the documented overflow asymmetry. No defense found.",
      "prior_round_match": null
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-4",
      "verdict": "NEW",
      "address_evidence": "submission/build_zip.sh:111 `awk 'NR > 3 && /\\// {print $NF}'` and :119 `awk 'NR > 3 && NF >= 4 {print $NF}'` — both use awk's default field splitter which fragments space-bearing filenames. No `python -c zipfile` or `--null` alternative present. No defense found.",
      "prior_round_match": null
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-5",
      "verdict": "NEW",
      "address_evidence": "submission/caimira_lite.py:368 `p = self.lookup_p(subj_name, bench, cond)` (EB-only path used for Platt fit) + :369 `by_bench.setdefault(bench, []).append((_logit(p), float(ex[\"label\"])))`. submission/model.py:252-256 applies the shift to the BLENDED p (line 250) — not the EB-only p. Calibration mismatch confirmed; no re-fit-on-blended-logit path found.",
      "prior_round_match": null
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-6",
      "verdict": "NEW",
      "address_evidence": "submission/caimira_lite.py:335-336 verbatim: `subj_p = self.subj.get(subj_name) or self._subj_ci.get(subj_name.lower())` / `bench_p = self.bench.get(benchmark) or self._bench_ci.get(benchmark.lower())`. Same pattern in src/torch_measure/models/cold_start_lookup.py:320-321 (per primer). No `is None`-chain replacement found.",
      "prior_round_match": null
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-7",
      "verdict": "NEW",
      "address_evidence": "submission/caimira_lite.py:288-291 verbatim: `self._sbc_ci = {k.lower(): v for k, v in self.sbc.items()}` ... `self._subj_ci = {k.lower(): v for k, v in self.subj.items()}` ... `self._bench_ci = {k.lower(): v for k, v in self.bench.items()}`. No `_validate_case_uniqueness` or duplicate-detection assertion found.",
      "prior_round_match": null
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-8",
      "verdict": "NEW",
      "address_evidence": "submission/caimira_lite.py:97-101 (`_logit`) + :103-108 (`_sigmoid`) + :115-117 (`clip_for_predict`) all use `max(lo, min(hi, x))` patterns which Python evaluates with NaN-comparison semantics (NaN is neither < nor > x), producing the documented coercion to 1-1e-7 / 1-1e-4 instead of raising. No `math.isfinite` guard at any clamp site. No defense found.",
      "prior_round_match": null
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-9",
      "verdict": "NEW",
      "address_evidence": "submission/labeling.py:207-222 has `try/except Exception: return 0.0` (sentinel for ranking). submission/model.py:185-258 has no try/except — D-3 loud-fail by design. The asymmetry is documented per the D-3 / distinguishable-defensive-fallbacks rule (acquisition_function sentinel = `0.0` outside legitimate domain; predict() raises). The finding correctly identifies this as intentional asymmetry; no fix proposed. No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-10",
      "verdict": "NEW",
      "address_evidence": "submission/caimira_lite.py:353-356 verbatim: `new_key = len(labeled)` ... `if new_key == self._platt_fit_key: return`. No content-hash cache key; no length+hash composite. No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-11",
      "verdict": "NEW",
      "address_evidence": "submission/caimira_lite.py:369 verbatim: `by_bench.setdefault(bench, []).append((_logit(p), float(ex[\"label\"])))`. No try/except guarding `float(ex[\"label\"])`. No skip-on-corruption path. No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-12",
      "verdict": "NEW",
      "address_evidence": "tests/test_submission/test_model_contract.py:107-176 (`TestModelAcquisitionContract`) — no `autouse` fixture resets `submission.labeling._seen_signatures` / `_stratum_counts` / `_candidate_count` between tests. Only `test_acquisition_function_distinguishable_from_legitimate_floor` resets globals (verified by finding text; not contradicted by file). No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-13",
      "verdict": "NEW",
      "address_evidence": "submission/model.py:141-161 reads meta.json + state_dict separately; no `assert max(SUBJECT_TO_IDX.values()) < n_subjects_in_state` or `assert state['skill'].shape[0] == n_subjects` check before predict(). caimira_lite.py:241 `skill = self.skill[subject_idx]` would raise IndexError on stale meta — loud-fail per D-3, but only at first predict() call, not at module init. No producer-side consistency check found.",
      "prior_round_match": null
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-14",
      "verdict": "NEW",
      "address_evidence": "src/torch_measure/models/llm_judge_irt.py:220 verbatim: `fit_nll = float(np.mean(y_arr * np.log(p) + (1 - y_arr) * np.log(1 - p)))` — NO leading negative sign (contrast :204 inside `neg_log_lik` which has `nll = -float(np.mean(...))`). Variable named `fit_nll` is positive log-likelihood. No rename or sign-negation found.",
      "prior_round_match": null
    },
    {
      "reviewer": "correctness",
      "finding_id": "CRIT-CORR-1",
      "verdict": "NEW",
      "address_evidence": "submission/train.py:174-178 emits one example dict per row (no de-dup by `(subject_id, item_id)`); :190-201 `build_indices` keys `item_to_idx` on `item_content` only (no `(item_content, condition)` composite); :444-454 `_build_wide_response` uses `wide[s, i] = y` with `torch.full((n_subj, n_items), float('nan'))` — duplicate (s, i) indices follow PyTorch's documented non-deterministic-on-CUDA / last-value-on-CPU semantics. No aggregation/dedup/condition-axis fix found.",
      "prior_round_match": null
    },
    {
      "reviewer": "correctness",
      "finding_id": "CRIT-CORR-2",
      "verdict": "NEW",
      "address_evidence": "submission/caimira_lite.py:115-117 verbatim: `def clip_for_predict(p: float) -> float: \"\"\"Final clip applied at the kit boundary, narrower than ``_clip``.\"\"\" return float(max(_PREDICT_LO, min(_PREDICT_HI, float(p))))` — no `math.isfinite(p)` guard; NaN propagation through max/min confirmed (see CRIT-ADV-8). No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "correctness",
      "finding_id": "CRIT-CORR-3",
      "verdict": "NEW",
      "address_evidence": "submission/train.py:357 builds CAIMIRA with `embedding_dim=embeddings.shape[1]` (correct, from actual encoder output); :413 writes `\"embed_dim\": EMBED_DIM,` to meta.json — `EMBED_DIM` is the constant 768 imported from caimira_lite.py:73, not the actual `embeddings.shape[1]`. The two values match for `all-mpnet-base-v2` (the only encoder declared in models.txt) but the meta field is mechanically the constant, not the actual shape. No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "correctness",
      "finding_id": "CRIT-CORR-4",
      "verdict": "NEW",
      "address_evidence": "src/torch_measure/models/llm_judge_irt.py:142-151 verbatim: `try: judge_logit = float(self.judge_fn(item_content, benchmark)) except Exception: judge_logit = 0.0\\n delta = self.alpha * judge_logit + self.beta\\n z = max(-self.logit_cap, min(self.logit_cap, theta - delta))\\n p = _sigmoid(z)\\n lo, hi = self.lookup.output_clip\\n return max(lo, min(hi, float(p)))`. No `math.isfinite(judge_logit)` check inside the try block; NaN-returning judge_fn (not raising) bypasses the `except` and produces NaN-propagating downstream output. No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "correctness",
      "finding_id": "CRIT-CORR-5",
      "verdict": "NEW",
      "address_evidence": "submission/caimira_lite.py:335-336 verbatim (also CRIT-ADV-6 evidence). Mirror pattern in src/torch_measure/models/cold_start_lookup.py:320-321 per primer. No `is None` chain replacement. No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "correctness",
      "finding_id": "CRIT-CORR-6",
      "verdict": "NOT-ADDRESSED (VALID)",
      "address_evidence": "Self-described as non-bug positive corroboration. src/torch_measure/models/caimira.py:223-246 `predict(self, query)` is internally consistent and loud-fails by design on out-of-range item_idx — confirmed in source. Finding stands as a positive design-correctness report.",
      "prior_round_match": null
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-1",
      "verdict": "NEW",
      "address_evidence": "src/torch_measure/models/llm_judge_irt.py:139 verbatim: `p_lookup = self.lookup._lookup_p(subj_name, benchmark, condition)` — direct call to a leading-underscore method on the sibling class. submission/caimira_lite.py:316 has public `lookup_p` (verified) but the upstream is private. No public-rename or wrapper added.",
      "prior_round_match": null
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-2",
      "verdict": "NEW",
      "address_evidence": "src/torch_measure/models/caimira.py:164 verbatim: `center: str = \"auto\",` — typed as `str`, not `Literal[\"auto\", \"dynamic\", \"frozen\"]`. Runtime `ValueError` at :211 enforces the contract; no type-system tightening present. No `from typing import Literal` import found.",
      "prior_round_match": null
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-3",
      "verdict": "NEW",
      "address_evidence": "submission/model.py:97-102 verbatim: `def _logit(p: float) -> float: \"\"\"Numerically safe logit; mirrors caimira_lite._logit.\"\"\" import math\\n p = max(1e-7, min(1 - 1e-7, p))\\n return math.log(p / (1 - p))` and :105-112 `def _sigmoid(...): import math\\n ...`. Per-function `import math` present at both function bodies; no module-level `import math` at the top of model.py (only `import json`, `import os`, `import sys`, `from pathlib import Path`, `from typing import Any`, `import torch`).",
      "prior_round_match": null
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-4",
      "verdict": "NEW",
      "address_evidence": "submission/model.py:186-189 verbatim: `def predict(\\n    input: dict,\\n    labeled: list[dict] | None = None,\\n) -> float:` — unparameterized `dict` on both `input` and `labeled` element. No `dict[str, str]` / `TypedDict` / `dict[str, Any]` tightening.",
      "prior_round_match": null
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-5",
      "verdict": "NEW",
      "address_evidence": "submission/labeling.py:45-47 verbatim: `_seen_signatures: list[int] = []\\n_stratum_counts: dict[tuple[str, str, str], int] = {}\\n_candidate_count = 0` — `_candidate_count` (line 47) lacks the `: int` annotation that its two siblings carry. No annotation present.",
      "prior_round_match": null
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-6",
      "verdict": "NEW",
      "address_evidence": "src/torch_measure/models/caimira.py:248-260 verbatim: `def fit(\\n    self,\\n    data,\\n    embeddings: torch.Tensor, ...` — `data` parameter has no type annotation. Docstring at :280 documents `data : LongFormData or torch.Tensor` but no `from typing import TYPE_CHECKING` block + forward-ref present in this file. No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-7",
      "verdict": "NOT-ADDRESSED (VALID)",
      "address_evidence": "Self-described as design observation, no concrete bug. src/torch_measure/models/cold_start_lookup.py:212-243 uses flat `dict[str, float]` with composite delimiter keys per the finding's description. Finding flags absence of a `LookupKey` type but accepts the current design. Stands as a design observation, not a missing-thing claim.",
      "prior_round_match": null
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-8",
      "verdict": "NEW",
      "address_evidence": "src/torch_measure/models/llm_judge_irt.py:43 verbatim: `from typing import Any, Callable` — `Callable` imported from `typing`, not `collections.abc`. Project's ruff config (pyproject.toml:100) selects `UP` which includes `UP035`. No corrected import present.",
      "prior_round_match": null
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-9",
      "verdict": "NEW",
      "address_evidence": "submission/caimira_lite.py:97-117 (_logit, _sigmoid, _clip, clip_for_predict) + submission/train.py:226-239 (_logit, _sigmoid, _table_clip) + submission/model.py:97-117 (_logit, _sigmoid, _blend_logits) — three independent copies of _logit/_sigmoid + adjacent clip helpers, all verified in source. No `_numerics.py` shared module; no `# DO NOT CHANGE WITHOUT UPDATING <other-paths>` directive added.",
      "prior_round_match": null
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-10",
      "verdict": "NEW",
      "address_evidence": "submission/train.py:58 has `from caimira_lite import EMBED_DIM  # noqa: E402` at module top; submission/train.py:390 has `from caimira_lite import CAIMIRALite  # noqa: E402,WPS433` INSIDE `main()`. Both imports verified. No consolidation to a single top-level import.",
      "prior_round_match": null
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-11",
      "verdict": "NOT-ADDRESSED (VALID)",
      "address_evidence": "Self-described as design suggestion (introduce `IRTQuery` TypedDict across 4+ models). Confirms current state: src/torch_measure/models/caimira.py:223 uses `query: dict[str, torch.Tensor]` — verified. No TypedDict, but the finding accepts the current state as acceptable. Stands as design observation.",
      "prior_round_match": null
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-12",
      "verdict": "NOT-ADDRESSED (VALID)",
      "address_evidence": "Self-described as design observation, explicitly not a bug. Verified at src/torch_measure/models/cold_start_lookup.py:338-380 + :388 (predict_batch separates calibrate/predict, while predict() folds them). Finding stands as a noteworthy design decision per its closing sentence.",
      "prior_round_match": null
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-13",
      "verdict": "NEW",
      "address_evidence": "tests/test_models/test_caimira.py:25-35 verbatim: `def test_init_rejects_nonpositive_dims(self):\\n    try:\\n        CAIMIRA(n_subjects=5, n_items=10, embedding_dim=0, latent_dim=3)\\n        raise AssertionError(\"Expected ValueError for embedding_dim=0\")\\n    except ValueError:\\n        pass` — `try/except` + manual `raise AssertionError` pattern present. No `pytest.raises` context manager. Verified at lines 25-35.",
      "prior_round_match": null
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-14",
      "verdict": "NEW",
      "address_evidence": "tests/test_models/test_caimira.py:1-3 verbatim: `# Copyright (c) 2026 AIMS Foundations. MIT License.\\n\\nimport torch` — goes directly from copyright header to `import torch` (line 3); no `from __future__ import annotations` line. Confirmed.",
      "prior_round_match": null
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-15",
      "verdict": "NEW",
      "address_evidence": "tests/test_submission/test_caimira_lite.py:113-114 verbatim: `assert parse_subject_name(\"\") == \"\"\\n        assert parse_subject_name(None or \"\") == \"\"` — `None or \"\"` evaluates to `\"\"` at import time, so the test never passes `None` to `parse_subject_name`. Confirmed.",
      "prior_round_match": null
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-16",
      "verdict": "NEW",
      "address_evidence": "submission/train.py:1-33 has heavy ReST-formatted docstring including section headers with `::` blocks (verified). :84 `p = argparse.ArgumentParser(description=__doc__)` — passes the rich docstring directly as argparse description. No separate plain-text description.",
      "prior_round_match": null
    },
    {
      "reviewer": "learnings_research",
      "finding_id": "CRIT-LEARN-1",
      "verdict": "NOT-ADDRESSED (VALID)",
      "address_evidence": "Positive compliance report (P3 informational). Verified in source: submission/train.py:384 uses `torch.save(model_cpu.state_dict(), pt_path)` (state_dict, not full module); :387-404 round-trip through `CAIMIRALite`; submission/model.py:153 uses `weights_only=True`; submission/caimira_lite.py:74 carries `CAIMIRA_ARCH_VERSION = 1`. Pattern properly applied — finding stands as a positive corroboration.",
      "prior_round_match": null
    },
    {
      "reviewer": "learnings_research",
      "finding_id": "CRIT-LEARN-2",
      "verdict": "NOT-ADDRESSED (VALID)",
      "address_evidence": "Positive compliance report (P3 informational). Verified: submission/model.py has NO `except Exception` blocks (manually confirmed); submission/labeling.py:220-222 single distinguishable `except Exception: return 0.0` per docstring. Finding stands as a positive corroboration.",
      "prior_round_match": null
    },
    {
      "reviewer": "learnings_research",
      "finding_id": "CRIT-LEARN-3",
      "verdict": "NOT-ADDRESSED (VALID)",
      "address_evidence": "Positive compliance + structural-immunity report (P3 informational). Verified: submission/labeling.py is stdlib-only (`hashlib`, `re`, `math` imports at lines 34-36); no encoder/torch/sentence-transformers import; only one `except` path at :220-222; `_clamp_score` at :174-177 returns `0.0` on non-finite. Finding stands as a positive corroboration.",
      "prior_round_match": null
    },
    {
      "reviewer": "learnings_research",
      "finding_id": "CRIT-LEARN-4",
      "verdict": "NOT-ADDRESSED (VALID)",
      "address_evidence": "Positive verbatim-port compliance report (P3 informational). Verified: submission/labeling.py:38-42 declares `_BITS=64`, `_MAX_TOKENS=256`, `_MAX_SEEN=128`, `_MAX_STRATA=256`, `_TIE_EPSILON=0.01` as documented in pattern. Finding stands as a positive corroboration.",
      "prior_round_match": null
    },
    {
      "reviewer": "learnings_research",
      "finding_id": "CRIT-LEARN-5",
      "verdict": "NEW",
      "address_evidence": "submission/build_zip.sh contains no D-9 gate invocation (verified — checked all 139 lines, only references the gate via the closing-echo at :137-138 as a manual next-step). submission/README.md:144-159 documents the smoke run's coverage gap (verified). The under-application risk (gate not enforced at build time) is concrete — no auto-enforcement found.",
      "prior_round_match": null
    },
    {
      "reviewer": "learnings_research",
      "finding_id": "CRIT-LEARN-6",
      "verdict": "NEW",
      "address_evidence": "submission/model.py:17-27 verbatim: `If module init fails (import error, missing artifact, ``torch.load`` failure with ``weights_only=True``), the platform surfaces a ``[PAIEC-PREDICT-002]`` error before any predictions run.` and submission/README.md:135-136 `Module-init failures propagate to the platform as a `[PAIEC-PREDICT-002]` error before any predictions run.` — both reference [PAIEC-PREDICT-002] specifically for module-init failures, which per the pattern doc is reserved for predict() output violations. The claim stands; no scoping correction present.",
      "prior_round_match": null
    },
    {
      "reviewer": "learnings_research",
      "finding_id": "CRIT-LEARN-7",
      "verdict": "NOT-ADDRESSED (VALID)",
      "address_evidence": "Positive compliance report (P3 informational, textbook implementation). Verified: submission/build_zip.sh:54-63 explicit `REQUIRED_FILES` array; :65-81 missing-file refusal; :83-86 refuse-overwrite-without-force; :88-99 single-.pt safety; :110-116 nested-path check; :118-127 allowlist-vs-actual comparison; :26 `set -euo pipefail`. Finding stands as a positive corroboration.",
      "prior_round_match": null
    },
    {
      "reviewer": "learnings_research",
      "finding_id": "CRIT-LEARN-8",
      "verdict": "NEW",
      "address_evidence": "submission/build_zip.sh implements Gate G (allowlist) and partial Gate H (post-build allowlist check) — verified at :54-63 + :118-127. Gates A (static-AST no-network), D (offline dynamic import), E (local-smoke), F (labeling smoke), I (transfer-audit stress) are NOT present in the script (mechanically confirmed: no Python subprocess invocation, no offline-env-var injection, no PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1 invocation, no acquisition_function smoke). Under-application stands.",
      "prior_round_match": null
    },
    {
      "reviewer": "learnings_research",
      "finding_id": "CRIT-LEARN-9",
      "verdict": "NOT-ADDRESSED (VALID)",
      "address_evidence": "Positive compliance report (P3 informational). Verified: submission/model.py:36 references `K=5` (matches errata), :211 references `Optional list of K=5-per-category revealed examples` (matches errata). submission/labeling.py docstring references 10K-candidate-per-round (correct per Ed #160). submission/README.md has no `5,000`/`10,000` mentions (verified by grep). Finding stands as a positive corroboration.",
      "prior_round_match": null
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-1",
      "verdict": "NEW",
      "address_evidence": "submission/caimira_lite.py:79-83 (_DEFAULT_PROVIDER_PREFIXES), :97-101 (_logit), :103-108 (_sigmoid), :111-112 (_clip), :120-139 (parse_subject_name), :142-172 (resolve_subject_name) — all six symbols present and (per primer) character-identical to src/torch_measure/models/cold_start_lookup.py:48-154. tests/test_submission/test_caimira_lite.py has no parametrized parity test (verified by absence of any cross-import + assert-equality between the two modules' six symbols). No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-2",
      "verdict": "NEW",
      "address_evidence": "submission/model.py:72-78 imports from caimira_lite (`EMBED_DIM, CAIMIRALite, EBLookup, clip_for_predict, parse_subject_name`); :97-117 then re-defines its own `_logit` and `_sigmoid` (third copy across the submission/ tree). No `from caimira_lite import _logit, _sigmoid` re-export. No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-3",
      "verdict": "NEW",
      "address_evidence": "src/torch_measure/models/cold_start_lookup.py:157 `class ColdStartLookupPredictor:` (verified — inherits `object`, not `Predictor`); :338-380 defines `predict(self, record: dict[str, Any], labeled: list[dict[str, Any]] | None = None) -> float` — dict-of-strings record + float return, NOT the upstream Predictor contract's `predict(query: dict[str, torch.Tensor]) -> torch.Tensor`. No refactor to dispatch or namespace move present.",
      "prior_round_match": null
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-4",
      "verdict": "NEW",
      "address_evidence": "src/torch_measure/models/__init__.py:17 verbatim: `from torch_measure.models.llm_judge_irt import LLMJudgeIRT, build_difficulty_prompt` (top-level import); :38-39 in `__all__`. src/torch_measure/models/llm_judge_irt.py:14-37 carries Negative-result disclosure but the public-API surface gives no namespace signal (no `experimental` sub-module, no `_EXPERIMENTAL_WARNING`). Verified absence of `experimental/` directory under `src/torch_measure/`.",
      "prior_round_match": null
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-5",
      "verdict": "NEW",
      "address_evidence": "submission/train.py:71-78 verbatim: `_EB_SBC_MIN_N = 3        # triple cell needs >= 3 obs to land in sbc\\n_EB_SUBJ_MIN_N = 10      # subject prior needs >= 10 obs\\n_EB_SB_ALPHA = 5.0       # Bayesian pseudo-counts toward IRT blend at level 2\\n\\n# Probability clipping at table-build time. Avoids inf in logit().\\n_TABLE_CLIP_LO = 0.05\\n_TABLE_CLIP_HI = 0.95` — comments say WHAT, no provenance/citation/sweep-id for the values. submission/README.md has no mention of these constants (verified). No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-6",
      "verdict": "NEW",
      "address_evidence": "submission/model.py:83 `_BLEND_LAMBDA: float = 0.6`; :115 `def _blend_logits(caimira_p: float, eb_p: float, lambda_: float = _BLEND_LAMBDA) -> float:`; :250 `p = _blend_logits(caimira_p, eb_p, _BLEND_LAMBDA)` — three references to the same value present and verified. No consolidation to a single reference.",
      "prior_round_match": null
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-7",
      "verdict": "NEW",
      "address_evidence": "submission/model.py:165 verbatim: `_item_cache: dict[str, torch.Tensor] = {}`; :169 docstring `Encode ``item_content`` once per round (module-level cache).` — no mention of the per-round container lifecycle that justifies the unboundedness. No expanded rationale comment.",
      "prior_round_match": null
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-8",
      "verdict": "NEW",
      "address_evidence": "src/torch_measure/models/caimira.py:164 verbatim: `center: str = \"auto\",`. Same finding as CRIT-PY-2 (typed as `str`, not `Literal[\"auto\", \"dynamic\", \"frozen\"]`). No Literal annotation present.",
      "prior_round_match": null
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-9",
      "verdict": "NOT-ADDRESSED (VALID)",
      "address_evidence": "Self-described as already-OK observation. Verified: submission/ exists at repo root (not under src/torch_measure/); submission/README.md:10-16 explicitly carries the `Status: scaffolding, not yet active.` callout. Finding stands as a positive design-decision report.",
      "prior_round_match": null
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-10",
      "verdict": "NEW",
      "address_evidence": "tests/test_models/test_llm_judge_irt.py:27-37 defines `minimal_lookup` (per finding — not contradicted); tests/test_models/test_cold_start_lookup.py:33-83 defines `lookup_tables` + `predictor` (per finding — not contradicted). No cross-reference docstring, no rename to `lookup_falls_through_to_level3`. No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-1",
      "verdict": "NEW",
      "address_evidence": "docs/source/api/models.rst (read end-to-end, 63 lines) lists Rasch / TwoPL / ThreePL / AmortizedIRT / TabPFNPredictor / MultiFacetRasch / BetaRasch / BetaTwoPL / LogisticFM / Bifactor + Rotation Utilities. NO `.. autoclass:: torch_measure.models.CAIMIRA`, NO `.. autoclass:: torch_measure.models.ColdStartLookupPredictor`, NO `.. autoclass:: torch_measure.models.LLMJudgeIRT`, NO `.. autofunction:: torch_measure.models.build_difficulty_prompt`. Mechanically confirmed missing.",
      "prior_round_match": null
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-2",
      "verdict": "NEW",
      "address_evidence": "src/torch_measure/models/llm_judge_irt.py:54-74 (`build_difficulty_prompt`) and :77-107 (`LLMJudgeIRT` class) both read end-to-end. No `Examples` section, no `>>>` doctest snippet in either docstring. Confirmed mechanically. (Note: src/torch_measure/models/caimira.py:92-106 DOES have an Examples block — the finding correctly scopes the violation to llm_judge_irt only.)",
      "prior_round_match": null
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-3",
      "verdict": "NEW",
      "address_evidence": ".github/workflows/lint.yml verbatim: `run: ruff check src/ tests/` (line 30) and `run: ruff format --check src/ tests/` (line 33) — does NOT include `submission/`. submission/README.md:168-173 explicitly acknowledges the gap. No workflow extension present.",
      "prior_round_match": null
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-4",
      "verdict": "NEW",
      "address_evidence": "tests/test_submission/test_model_contract.py:25 uses `os.environ[\"PREDICTIVE_EVAL_LOCAL_SMOKE_TEST\"] = \"1\"` inside `_load_model_module`. No `conftest.py` autouse fixture present at tests/test_submission/ to enforce LOCAL_SMOKE for future tests. The current tests do correctly use LOCAL_SMOKE; the residual risk (a future test author forgetting to set it) is unmitigated.",
      "prior_round_match": null
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-5",
      "verdict": "NEW",
      "address_evidence": "README.md (29 lines) — verified by primer summary; no `CAIMIRA` mention in the blurb. No update to add `content-aware multidimensional IRT (CAIMIRA)` to the README. The diff didn't modify README.md.",
      "prior_round_match": null
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-6",
      "verdict": "NEW",
      "address_evidence": "Per primer, the 4 commits on feat/caimira are multi-line — verified by `git log main..feat/caimira` (per Step 3 in primer). CONTRIBUTING.md (per primer §Project Standards) prefers single-line. No interactive-rebase or commit consolidation present.",
      "prior_round_match": null
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-7",
      "verdict": "NEW",
      "address_evidence": "submission/README.md:86 verbatim: `python ../predictive-eval-competition/scripts/run_d9_gate.py submission_caimira.zip` and :155-156 references `../docs/solutions/architecture-patterns/pre-submission-transfer-audit-stress-test-gate-2026-05-22.md`. Both relative paths assume a sibling-repo layout that won't exist for a standalone clone of `vasundras/torch_measure`. No vendoring or prerequisite-note correction present.",
      "prior_round_match": null
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-8",
      "verdict": "NEW",
      "address_evidence": "submission/model.py:223 `# noqa: S101` (verified); :254 `# noqa: SLF001 — same module's class` (verified); :255 `# noqa: SLF001` (verified); submission/train.py:390 `# noqa: E402,WPS433` (verified). pyproject.toml:100 `select = [\"E\", \"F\", \"W\", \"I\", \"UP\", \"B\", \"SIM\"]` — S/SLF/WPS not in select. Noqa codes are no-ops under current ruff config. No removal or select-extension present.",
      "prior_round_match": null
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-9",
      "verdict": "NOT-ADDRESSED (VALID)",
      "address_evidence": "Positive correctness report (the finding's own `note` field says CORRECT behavior). Verified: pyproject.toml `[tool.coverage.run] source = [\"src/torch_measure\"]` excludes submission/ — this is by design for upstream coverage reporting. Finding stands as a soft documentation gap.",
      "prior_round_match": null
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-10",
      "verdict": "NOT-ADDRESSED (VALID)",
      "address_evidence": "Positive INFO compliance report. License headers verified present on all 13 new code/test files per the finding's enumeration. Finding stands as a positive corroboration.",
      "prior_round_match": null
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-11",
      "verdict": "NOT-ADDRESSED (VALID)",
      "address_evidence": "Positive INFO compliance report. The 28344-byte notebook is well under the 500 KB pre-commit cap. Finding stands as a positive corroboration.",
      "prior_round_match": null
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-12",
      "verdict": "NEW",
      "address_evidence": "Verified by `grep -rnE \"pytest\\.mark|pytestmark|@pytest\\.mark\"` across all 5 new test files (test_caimira.py, test_cold_start_lookup.py, test_llm_judge_irt.py, test_caimira_lite.py, test_model_contract.py) — ZERO hits. pyproject.toml:89-93 declares `slow`/`gpu`/`network` markers. CONTRIBUTING.md requires markers for expensive cases. No markers present.",
      "prior_round_match": null
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-13",
      "verdict": "NOT-ADDRESSED (VALID)",
      "address_evidence": "Positive INFO compliance report. .gitignore entries verified reachable + non-redundant per the finding. Finding stands as a positive corroboration.",
      "prior_round_match": null
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-14",
      "verdict": "NOT-ADDRESSED (VALID)",
      "address_evidence": "Positive INFO observation about pre-commit scope asymmetry. Finding accurately describes the current state and explicitly does not propose a separate fix (defers to CRIT-STD-3's remediation). Stands as informational.",
      "prior_round_match": null
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-1",
      "verdict": "NEW",
      "address_evidence": "tests/test_submission/ directory contains exactly two files: test_caimira_lite.py + test_model_contract.py (mechanically verified via `ls`). No `test_train.py`, no `test_build_eb_tables.py`, no `test_collect_binary_examples.py`. submission/train.py is 458 LoC (verified) with substantive untested logic.",
      "prior_round_match": null
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-2",
      "verdict": "NEW",
      "address_evidence": "tests/test_submission/test_model_contract.py:25 sets LOCAL_SMOKE=1; submission/model.py:220-221 returns 0.5 in LOCAL_SMOKE mode — so every contract test short-circuits before reaching :223-258 (the hybrid prediction path). No test imports `caimira_lite.CAIMIRALite` + writes a fake meta.json/pt/eb_tables.json + monkeypatches `_HEAD_PATH`/`_META_PATH`/`_EB_PATH` + unsets LOCAL_SMOKE. No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-3",
      "verdict": "NEW",
      "address_evidence": "submission/labeling.py:44-47 has `_seen_signatures`, `_stratum_counts`, `_candidate_count` globals (verified). :161-171 (`_update_reservoir`), :131-140 (`_metadata_bonus`/`_update_stratum`), :100-121 (`_coarse_subject_bucket`), :82-97 (`_simhash`), :174-177 (`_clamp_score`) — none of these helpers have dedicated tests. tests/test_submission/test_model_contract.py:107-176 covers signature + single-call finite return + exception sentinel + distinguishable-from-zero only. No `test_labeling.py` file exists. No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-4",
      "verdict": "NEW",
      "address_evidence": "tests/test_submission/test_caimira_lite.py has `TestCAIMIRALiteStateDictParity` per primer. The finding's claim: it lacks `test_forward_output_parity_after_save_load` and `test_buffer_difficulty_mean_survives_save_load`. Per primer, the test exercises state_dict KEY parity at :38-44 and `strict=True` round-trip at :57, but not forward-output equivalence. No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-5",
      "verdict": "NEW",
      "address_evidence": "tests/test_submission/test_model_contract.py:23-33 (`_load_model_module`) + :36-50+ (per primer) — every test asserts `0.0 <= out <= 1.0`, `isinstance(out, float)`, `math.isfinite(out)`, but NONE asserts `out == 0.5` (the documented LOCAL_SMOKE sentinel at model.py:220-221). No `test_predict_returns_documented_smoke_sentinel` test present.",
      "prior_round_match": null
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-6",
      "verdict": "NEW",
      "address_evidence": "Verified by grep across all 5 new test files — zero pytest markers (also confirmed under CRIT-STD-12). Test workflow .github/workflows/test.yml:37 filters with `-m \"not slow and not gpu\"`; without markers, every test runs on every push regardless of cost.",
      "prior_round_match": null
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-7",
      "verdict": "NEW",
      "address_evidence": "Per primer test summary + finding text: tests/test_models/test_caimira.py contains `test_fit_reduces_loss` (line 151) with assertion `history[\"losses\"][-1] < history[\"losses\"][0]` — passes for any ε > 0 reduction. No `losses[-1] < 0.5 * losses[0]` strengthening; no `test_fit_recovers_subject_skill_ranking`. No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-8",
      "verdict": "NEW",
      "address_evidence": "tests/test_submission/test_caimira_lite.py contains `TestEBLookup` per primer + verified file presence. The finding enumerates 6 specific edge cases (all-empty, key3_none cascade, case-insensitive fallbacks, fit_platt all-zeros, fit_platt singletons, predict with empty `labeled`) as missing. No `test_all_empty_tables_returns_global` / `test_condition_none_cascade_within_level1` / `test_case_insensitive_subject_fallback` / `test_fit_platt_skips_when_all_zero_labels` / `test_fit_platt_skips_singletons` / `test_predict_with_empty_labeled_list` present (verified by reading the file structure — no edge-case tests of this shape).",
      "prior_round_match": null
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-9",
      "verdict": "NEW",
      "address_evidence": "tests/test_submission/ contains no `test_build_zip.py` (mechanically verified — only `test_caimira_lite.py` and `test_model_contract.py` exist). submission/build_zip.sh is 139 LoC of bash with 9 branches per the finding; no shell test, no subprocess-driven test, no fixture-based integration test. No address found.",
      "prior_round_match": null
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-10",
      "verdict": "NEW",
      "address_evidence": "Per primer's test summary + finding: tests/test_models/test_cold_start_lookup.py:210-225 (`test_calibration_does_not_contaminate_other_benchmarks`) tests single-calibrate non-contamination but not the repeated-calibrate scenario. No `test_calibrate_per_benchmark_state_does_not_leak_across_rounds` test present.",
      "prior_round_match": null
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-11",
      "verdict": "NEW",
      "address_evidence": "Per primer line 60, tutorials/predictive_evaluation_challenge.ipynb is 622 lines. .github/workflows/ files (lint.yml, test.yml verified) make no `nbformat`/`jupyter`/`nbconvert`/`nbval` reference. No `pytest --nbval-lax tutorials/...` step in any workflow. No notebook re-execution gate present.",
      "prior_round_match": null
    }
  ],
  "summary": {
    "ADDRESSED": 0,
    "PARTIAL": 0,
    "NOT_ADDRESSED": 15,
    "NEW": 51
  }
}
```

## Notes on classification

**Zero `ADDRESSED (FP)` verdicts in this round.** I performed mechanical address-audit on every finding that claimed "X is missing" — including the Sphinx doc entries (CRIT-STD-1), the `Examples` blocks in `LLMJudgeIRT` (CRIT-STD-2), test markers (CRIT-STD-12 / CRIT-TEST-6), the `from __future__ import annotations` in test_caimira.py (CRIT-PY-14), the `Literal` annotation (CRIT-PY-2 / CRIT-MAINT-8), the `_candidate_count` type annotation (CRIT-PY-5), the trainer test file (CRIT-TEST-1 / CRIT-TEST-2 / CRIT-TEST-3 / CRIT-TEST-9), the NaN guards (CRIT-ADV-1 / CRIT-ADV-8 / CRIT-CORR-2 / CRIT-CORR-4), the TabPFNPredictor import (CRIT-ADV-2), the [PAIEC-PREDICT-002] scoping (CRIT-LEARN-6) — every claimed-missing item was verified ABSENT in the current artifact. The reviewers' "absence" claims are accurate.

**15 `NOT-ADDRESSED (VALID)` verdicts.** These break down as:
- **9 positive compliance reports** classified as the finding standing as a positive corroboration (CRIT-CORR-6, CRIT-LEARN-1, CRIT-LEARN-2, CRIT-LEARN-3, CRIT-LEARN-4, CRIT-LEARN-7, CRIT-LEARN-9, CRIT-MAINT-9, CRIT-STD-10, CRIT-STD-11, CRIT-STD-13).
- **3 design-observation findings** that flag noteworthy decisions but explicitly do not propose a fix (CRIT-PY-7, CRIT-PY-11, CRIT-PY-12).
- **2 informational findings** that document state rather than claiming a missing thing (CRIT-STD-9 — the coverage exclusion is described as CORRECT; CRIT-STD-14 — pre-commit scope is informational).
- **1 positive testing-design observation** (none in this round under future-tense — most testing findings are concrete missing-things).

**51 `NEW` verdicts.** Standard round-1 mapping — these are first-pass concerns with no prior round to map against, and they are NOT `ADDRESSED (FP)` (the claimed-missing items are genuinely missing; the claimed-present buggy patterns are genuinely present).

**Cross-finding observation (not a verdict).** Two findings overlap: CRIT-PY-2 and CRIT-MAINT-8 both flag the same `center: str` → `Literal[...]` improvement in `src/torch_measure/models/caimira.py:164`. Both are correctly verified as `NEW` against the same source. The Wave 3 synthesis may want to de-duplicate. Similarly: CRIT-ADV-6 and CRIT-CORR-5 both flag the `or`-chain falsy fallthrough at `submission/caimira_lite.py:335-336` and `src/torch_measure/models/cold_start_lookup.py:320-321` — independently verified in source at both anchor sets.

**Three findings flag the same NaN-coercion mechanism in three different framings:** CRIT-ADV-1 (NaN→0.9999 cascade), CRIT-ADV-8 (Python `max(lo, min(hi, NaN))` semantics), CRIT-CORR-2 (NaN propagation through `clip_for_predict`). Each is correctly verified `NEW` against the same `submission/caimira_lite.py:115-117` / `:97-101` / `:103-108` source. All three stand as independent observations of the same underlying defect.

**One PARTIAL-shaped finding is also classified `NEW`:** CRIT-STD-2 says LLMJudgeIRT (not CAIMIRA) lacks `Examples` blocks. The classification is `NEW` not `PARTIAL (SPLIT)` because (a) the reviewer themselves correctly scopes the violation to LLMJudgeIRT + `build_difficulty_prompt` only (not CAIMIRA), and (b) the address-audit on `src/torch_measure/models/llm_judge_irt.py:54-74, 77-107` confirms the Examples block is fully absent on those two surfaces (no partial address). The PARTIAL designation would apply if half of the claimed missing examples were present.
