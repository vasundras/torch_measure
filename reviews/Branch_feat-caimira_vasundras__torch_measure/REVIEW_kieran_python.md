```json
{
  "reviewer": "kieran-python",
  "findings": [
    {
      "id": "CRIT-PY-1",
      "severity": "P2",
      "confidence": 80,
      "title": "LLMJudgeIRT.predict reaches into ColdStartLookupPredictor._lookup_p (private cross-class access)",
      "where": {
        "file": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/src/torch_measure/models/llm_judge_irt.py",
        "lines": "139"
      },
      "explanation": "`LLMJudgeIRT.predict()` calls `self.lookup._lookup_p(subj_name, benchmark, condition)` — a leading-underscore method on a sibling class. The underscore signals 'subject to change without notice', so any future refactor of `ColdStartLookupPredictor` that touches `_lookup_p`'s signature silently breaks `LLMJudgeIRT` even though they live in the same public surface. The module also re-imports the same module's `_logit`, `_sigmoid`, `parse_subject_name`, and `resolve_subject_name` from `cold_start_lookup` (lines 45-51) — three of those are also underscore-prefixed (`_logit`, `_sigmoid`). The cleanest fix is to promote `_lookup_p` to `lookup_p` on `ColdStartLookupPredictor` since it is the actual deterministic raw-lookup primitive (the public `predict` adds Platt calibration + clipping, which `LLMJudgeIRT` deliberately skips). The sibling `submission/caimira_lite.py` already does this right — `EBLookup.lookup_p` is public (line 316). The upstream class should mirror that naming so the two implementations of the same hierarchy converge. Before:\n\n```python\n# llm_judge_irt.py:139\np_lookup = self.lookup._lookup_p(subj_name, benchmark, condition)\n```\n\nAfter (with corresponding rename on `ColdStartLookupPredictor`):\n\n```python\np_lookup = self.lookup.lookup_p(subj_name, benchmark, condition)\n```\n\nKeep the old `_lookup_p` as a one-line wrapper for backward compatibility if anyone outside the repo has started importing it.",
      "anchors": [
        "src/torch_measure/models/llm_judge_irt.py:139",
        "src/torch_measure/models/cold_start_lookup.py:288",
        "submission/caimira_lite.py:316"
      ],
      "evidence_kind": "diff"
    },
    {
      "id": "CRIT-PY-2",
      "severity": "P2",
      "confidence": 80,
      "title": "CAIMIRA.compute_item_params `center` arg should be `Literal[\"auto\", \"dynamic\", \"frozen\"]`",
      "where": {
        "file": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/src/torch_measure/models/caimira.py",
        "lines": "161-213"
      },
      "explanation": "The `center` parameter accepts exactly three string values (`\"auto\"`, `\"dynamic\"`, `\"frozen\"`) and raises `ValueError` on anything else (line 211). The signature types it as `str`, which gives the caller and the type checker no help. Tightening to `Literal[\"auto\", \"dynamic\", \"frozen\"]` would let an IDE auto-complete the three modes, would surface bad call sites at type-check time instead of `ValueError` at runtime, and would document the enum in the signature itself rather than buried in the docstring's `Parameters` section. The runtime `ValueError` should stay as a defense against dynamic dispatch from JSON / CLI / Pyro samplers. Before:\n\n```python\ndef compute_item_params(\n    self,\n    embeddings: torch.Tensor | None = None,\n    center: str = \"auto\",\n) -> tuple[torch.Tensor, torch.Tensor]:\n```\n\nAfter:\n\n```python\nfrom typing import Literal\n\n_CenterMode = Literal[\"auto\", \"dynamic\", \"frozen\"]\n\ndef compute_item_params(\n    self,\n    embeddings: torch.Tensor | None = None,\n    center: _CenterMode = \"auto\",\n) -> tuple[torch.Tensor, torch.Tensor]:\n```\n\nThis is a clarity P2, not a P1 — the `ValueError` already prevents bad inputs at runtime.",
      "anchors": [
        "src/torch_measure/models/caimira.py:164",
        "src/torch_measure/models/caimira.py:211"
      ],
      "evidence_kind": "diff"
    },
    {
      "id": "CRIT-PY-3",
      "severity": "P2",
      "confidence": 75,
      "title": "Two `import math` calls inside `_logit` / `_sigmoid` in submission/model.py — lift to module-level",
      "where": {
        "file": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/submission/model.py",
        "lines": "97-112"
      },
      "explanation": "`_logit` (line 97) and `_sigmoid` (line 105) each do `import math` inside the function body. `math` is a stdlib C-extension module — the import is cheap, but the per-call import is paying for the `sys.modules` lookup on every call, and Codabench will hit `predict()` once per item over 10K items. More importantly, function-level imports of stdlib modules are a code smell unless they're guarding a deferred-heavy import (e.g., `import torch` inside a function called rarely). For `math`, the convention is module-level. Sibling module `submission/caimira_lite.py` (line 64) does this correctly. Before:\n\n```python\n# submission/model.py:97-103\ndef _logit(p: float) -> float:\n    \"\"\"Numerically safe logit; mirrors caimira_lite._logit.\"\"\"\n    import math\n\n    p = max(1e-7, min(1 - 1e-7, p))\n    return math.log(p / (1 - p))\n```\n\nAfter:\n\n```python\n# submission/model.py — top of file alongside `import json`, `import os`\nimport math\n\n# ...\n\ndef _logit(p: float) -> float:\n    \"\"\"Numerically safe logit; mirrors caimira_lite._logit.\"\"\"\n    p = max(1e-7, min(1 - 1e-7, p))\n    return math.log(p / (1 - p))\n```\n\nSeparately: `submission/model.py` re-implements `_logit` and `_sigmoid` even though `caimira_lite.py` already exports the same functions (lines 97-108 there). The submission could `from caimira_lite import _logit, _sigmoid` instead — but those are underscore-private. Either promote them to public in `caimira_lite.py` (preferred) or keep two copies (current state, but lift the imports to module scope).",
      "anchors": [
        "submission/model.py:97",
        "submission/model.py:105",
        "submission/caimira_lite.py:64"
      ],
      "evidence_kind": "diff"
    },
    {
      "id": "CRIT-PY-4",
      "severity": "P3",
      "confidence": 70,
      "title": "submission/model.py predict() typed as `input: dict` instead of `dict[str, str]`",
      "where": {
        "file": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/submission/model.py",
        "lines": "186-219"
      },
      "explanation": "The `predict` signature on line 187 types the first argument as `input: dict` (unparametrized). The contract (documented in the docstring `Parameters` section on lines 206-209) is that the four keys are `benchmark`, `condition`, `subject_content`, `item_content`, all strings. The matching kit contract docstring on lines 11-13 also says `predict(input: dict, labeled: list[dict] | None = None) -> float`. Tightening to `dict[str, str]` on the parameter (and the `labeled` element type to `dict[str, str | int | float]` to accommodate the `label` key) would document the data shape at the type-checker boundary. The contract test (`tests/test_submission/test_model_contract.py:53-59`) constructs `dict[str, str]` literals already. Before:\n\n```python\ndef predict(\n    input: dict,\n    labeled: list[dict] | None = None,\n) -> float:\n```\n\nAfter:\n\n```python\ndef predict(\n    input: dict[str, str],\n    labeled: list[dict[str, Any]] | None = None,\n) -> float:\n```\n\n(Use `dict[str, Any]` for `labeled[i]` because each element is a 5-key dict where the `label` value is float-or-int while the other four are strings — a `TypedDict` would be tighter still but is overkill for a single call site.) This is P3 because the kit contract itself uses unparametrized `dict` in the staff docstring (per parent CLAUDE.md), so the project's prevailing-style argument is mixed.",
      "anchors": [
        "submission/model.py:187",
        "tests/test_submission/test_model_contract.py:53"
      ],
      "evidence_kind": "diff"
    },
    {
      "id": "CRIT-PY-5",
      "severity": "P3",
      "confidence": 75,
      "title": "submission/labeling.py `_candidate_count` missing type annotation",
      "where": {
        "file": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/submission/labeling.py",
        "lines": "45-47"
      },
      "explanation": "The three module-level globals are inconsistently annotated:\n\n```python\n_seen_signatures: list[int] = []          # line 45 — annotated\n_stratum_counts: dict[tuple[str, str, str], int] = {}  # line 46 — annotated\n_candidate_count = 0                      # line 47 — NOT annotated\n```\n\n`_candidate_count` is reassigned via `global _candidate_count` inside `acquisition_function` (line 205) so its inferred type matters to any static analyzer the team adds in the future. Add the annotation. Before:\n\n```python\n_candidate_count = 0\n```\n\nAfter:\n\n```python\n_candidate_count: int = 0\n```\n\nP3 because the inferred type is correct today and the file is exempt from CI's ruff path (`.github/workflows/lint.yml` runs against `src/ tests/` only), but the inconsistency with its two siblings is a small readability tax.",
      "anchors": [
        "submission/labeling.py:47",
        "submission/labeling.py:205"
      ],
      "evidence_kind": "diff"
    },
    {
      "id": "CRIT-PY-6",
      "severity": "P3",
      "confidence": 70,
      "title": "CAIMIRA.fit `data` parameter missing type annotation (project-style mismatch vs. _base.IRTModel.fit)",
      "where": {
        "file": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/src/torch_measure/models/caimira.py",
        "lines": "248-260"
      },
      "explanation": "`CAIMIRA.fit(self, data, embeddings: torch.Tensor, ...)` is missing the annotation on `data`. The docstring (line 280-281) documents it as `data : LongFormData or torch.Tensor`. The parent class `IRTModel.fit` types this correctly as `data: LongFormData | torch.Tensor` (per `src/torch_measure/models/_base.py:32` with the `TYPE_CHECKING` import on line 14 to avoid the circular import). Sibling `AmortizedIRT.fit` (`src/torch_measure/models/amortized.py:135`) ALSO omits the annotation — so the inconsistency is project-wide, but the right fix is to mirror the base class. Before:\n\n```python\ndef fit(\n    self,\n    data,\n    embeddings: torch.Tensor,\n    ...,\n) -> dict:\n```\n\nAfter:\n\n```python\nfrom typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from torch_measure.datasets._long_form import LongFormData\n\nclass CAIMIRA(IRTModel):\n    def fit(\n        self,\n        data: LongFormData | torch.Tensor,\n        embeddings: torch.Tensor,\n        ...,\n    ) -> dict:\n```\n\nP3 because the prevailing project style is split — fixing CAIMIRA only would create asymmetry with `AmortizedIRT`. The right Kieran-bar answer is to fix both. Mention in the same PR if upstreaming.",
      "anchors": [
        "src/torch_measure/models/caimira.py:250",
        "src/torch_measure/models/_base.py:32",
        "src/torch_measure/models/amortized.py:135"
      ],
      "evidence_kind": "diff"
    },
    {
      "id": "CRIT-PY-7",
      "severity": "P3",
      "confidence": 60,
      "title": "ColdStartLookupPredictor uses `dict[str, float]` for unrestricted ID-keyed maps — consider TypedDict / named keys",
      "where": {
        "file": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/src/torch_measure/models/cold_start_lookup.py",
        "lines": "212-243"
      },
      "explanation": "The constructor takes five `dict[...]` parameters (`sbc`, `sb`, `subj`, `bench`, `name_aliases`, `name_lc`) each with a documented composite-key format (`'<subject>||<benchmark>||<condition>'`, etc.). The key shape is enforced only by docstring discipline — a typo at any of the 4 build sites in `submission/train.py` (line 260-263) produces a silent miss at predict time. This is the canonical case for either (a) a small dataclass that owns the key construction (`LookupKey(subject, benchmark, condition).encode() -> str`), or (b) using nested dicts `dict[str, dict[str, dict[str, float]]]` for the three-level table. Both are heavier than the current flat-dict + delimiter design — Kieran's bar accepts the current design as 'explicit and obvious' (the keys are constructed via f-strings at every consumer site so any typo is visible at code review). Flagging as P3 because the EB-table consumer in `submission/caimira_lite.py:316-344` ALSO constructs keys via the same f-string pattern, so the duplication is structural; consolidating is a refactor not a fix. The current code is good enough — but the absence of any `LookupKey` type means a future contributor adding a 7th lookup level has no compile-time check that they got the delimiter right.",
      "anchors": [
        "src/torch_measure/models/cold_start_lookup.py:212",
        "submission/train.py:260",
        "submission/caimira_lite.py:316"
      ],
      "evidence_kind": "design"
    },
    {
      "id": "CRIT-PY-8",
      "severity": "P3",
      "confidence": 65,
      "title": "LLMJudgeIRT uses `typing.Callable` instead of `collections.abc.Callable`",
      "where": {
        "file": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/src/torch_measure/models/llm_judge_irt.py",
        "lines": "43"
      },
      "explanation": "`from typing import Any, Callable` (line 43) — `typing.Callable` was deprecated in favor of `collections.abc.Callable` per PEP 585 and ruff's `UP035` rule (which is in the project's `select = [\"UP\", ...]` list, so this MIGHT actually be a lint regression that slipped through). The fix is mechanical:\n\nBefore:\n\n```python\nfrom typing import Any, Callable\n```\n\nAfter:\n\n```python\nfrom collections.abc import Callable\nfrom typing import Any\n```\n\nThe sibling `cold_start_lookup.py` already imports `Iterable` from `collections.abc` (line 42) — so the project convention is established; `llm_judge_irt.py` just missed it. Re-run `ruff check src/torch_measure/models/llm_judge_irt.py --select UP035` to confirm; if `UP035` is in fact catching `Callable` from `typing` here, this is a P2-not-P3 because CI will start failing the next time anyone touches the file.",
      "anchors": [
        "src/torch_measure/models/llm_judge_irt.py:43",
        "src/torch_measure/models/cold_start_lookup.py:42",
        "pyproject.toml:[tool.ruff.lint].select"
      ],
      "evidence_kind": "lint-tool"
    },
    {
      "id": "CRIT-PY-9",
      "severity": "P3",
      "confidence": 70,
      "title": "Two `_table_clip` / `_clip` / `_PREDICT_LO` triplets duplicated across submission/caimira_lite.py, submission/train.py, and cold_start_lookup.py — consider one shared helper",
      "where": {
        "file": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/submission/caimira_lite.py",
        "lines": "97-117"
      },
      "explanation": "The same `_logit`, `_sigmoid`, `_clip`, and constant-clip-bounds pattern appears in:\n\n- `src/torch_measure/models/cold_start_lookup.py:63-78` (`_logit`, `_sigmoid`, `_clip`, `_CLIP_LO=0.05`, `_CLIP_HI=0.95`)\n- `submission/caimira_lite.py:97-117` (identical `_logit`, `_sigmoid`, `_clip`, plus `clip_for_predict` with `_PREDICT_LO=1e-4`, `_PREDICT_HI=1.0-1e-4`)\n- `submission/train.py:226-239` (`_logit`, `_sigmoid`, `_table_clip` with `_TABLE_CLIP_LO=0.05`, `_TABLE_CLIP_HI=0.95`)\n- `submission/model.py:97-112` (`_logit`, `_sigmoid`, `_blend_logits`)\n\nThis is intentional duplication — `caimira_lite.py` cannot import from `torch_measure` because the runtime is sandboxed and ZIP-rooted. But the train-side and the cold_start_lookup-side could share. Worth noting that all four `_logit` implementations use the SAME clip constant `1e-7` and the SAME `math.log(p / (1 - p))` formula, and all four `_sigmoid` implementations use the SAME branch-on-sign overflow-safe form. If any one of them is changed (e.g., to `1e-9`), the others silently disagree — and the round-trip-state-dict-load check in `submission/train.py:399` does NOT catch numerical-tolerance drift in helpers.\n\nThe pragmatic fix is to add a top-level header comment at each duplicate site naming the canonical source and adding a `# DO NOT CHANGE WITHOUT UPDATING <other-paths>` directive. The Kieran-bar fix is to extract `_numerics.py` (logit / sigmoid / clip with shared constants) that lives in the package AND is copied verbatim into the submission ZIP via the build script. Flagging as P3 because the current state is the deliberate design (sandbox forces some duplication) — but the absence of any drift-prevention mechanism is a future-bug shape.",
      "anchors": [
        "submission/caimira_lite.py:97",
        "submission/train.py:226",
        "submission/model.py:97",
        "src/torch_measure/models/cold_start_lookup.py:63"
      ],
      "evidence_kind": "design"
    },
    {
      "id": "CRIT-PY-10",
      "severity": "P3",
      "confidence": 65,
      "title": "submission/train.py inline-imports `CAIMIRALite` inside `main()` — split init and import for testability",
      "where": {
        "file": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/submission/train.py",
        "lines": "390"
      },
      "explanation": "`from caimira_lite import CAIMIRALite  # noqa: E402,WPS433` is done INSIDE `main()` on line 390 — after the `sys.path.insert` on line 56 and an earlier `from caimira_lite import EMBED_DIM` on line 58. The pattern works (the path-mutation guarantees `caimira_lite` is importable), but the in-function import is confusing because the constant import at line 58 already covers the path-mutation. A reader scanning `main()` will see the second `from caimira_lite import ...` and assume there's a runtime-mutability reason (there isn't). \n\nMove the `CAIMIRALite` import next to the `EMBED_DIM` import at line 58:\n\nBefore (line 58):\n```python\nfrom caimira_lite import EMBED_DIM  # noqa: E402\n```\n\nAfter (line 58):\n```python\nfrom caimira_lite import EMBED_DIM, CAIMIRALite  # noqa: E402\n```\n\nAnd remove the in-`main()` import + its `WPS433` noqa. P3 because the current behavior is correct; this is purely readability. The `WPS433` (nested import) noqa suggests someone DID think about it and decided in-function was intentional — would be worth a comment explaining why if the in-function form must stay.",
      "anchors": [
        "submission/train.py:58",
        "submission/train.py:390"
      ],
      "evidence_kind": "diff"
    },
    {
      "id": "CRIT-PY-11",
      "severity": "P3",
      "confidence": 60,
      "title": "CAIMIRA.predict only takes a 2-key dict (subject_idx, item_idx) — sibling models use the same shape but have no shared TypedDict",
      "where": {
        "file": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/src/torch_measure/models/caimira.py",
        "lines": "223-246"
      },
      "explanation": "`CAIMIRA.predict(self, query: dict[str, torch.Tensor]) -> torch.Tensor` takes a dict that the docstring says must contain `subject_idx` and `item_idx` (lines 231-234). The same shape is used by `Rasch.predict` (`rasch.py:37`), `TwoPL.predict` (`twopl.py:42`), and `AmortizedIRT.predict` (`amortized.py:121`). None of them type the dict — it's `dict[str, torch.Tensor]` everywhere, with the two-key contract documented in docstrings. A small `TypedDict` would document the shape at the type-check boundary:\n\n```python\nfrom typing import TypedDict\n\nclass IRTQuery(TypedDict):\n    subject_idx: torch.Tensor\n    item_idx: torch.Tensor\n\nclass CAIMIRA(IRTModel):\n    def predict(self, query: IRTQuery) -> torch.Tensor:\n        ...\n```\n\nThis is a Kieran-bar P3 — the current `dict[str, torch.Tensor]` typing works fine; tightening would be a multi-file refactor across 4+ models. Worth doing IF this branch ever PRs the new CAIMIRA upstream, since it'd be the right place to introduce `IRTQuery` once for all factor-model predictors. Not a fork-only concern.",
      "anchors": [
        "src/torch_measure/models/caimira.py:223",
        "src/torch_measure/models/rasch.py:37",
        "src/torch_measure/models/twopl.py:42",
        "src/torch_measure/models/amortized.py:121"
      ],
      "evidence_kind": "design"
    },
    {
      "id": "CRIT-PY-12",
      "severity": "P3",
      "confidence": 70,
      "title": "ColdStartLookupPredictor.predict mutates self (calibrate side-effect) — surprising for callers passing labeled",
      "where": {
        "file": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/src/torch_measure/models/cold_start_lookup.py",
        "lines": "338-380"
      },
      "explanation": "`ColdStartLookupPredictor.predict(record, labeled)` calls `self.calibrate(labeled)` (line 374) — which writes `self._platt` and `self._platt_fit_key` as a side effect. From the caller's perspective, `predict` LOOKS pure (single record in, float out), but it actually mutates the predictor's calibration state. The idempotency cache (`_platt_fit_key` keyed by `len(labeled)`) protects against re-fitting on identical-length calls, but if a caller passes two DIFFERENT labeled lists of the same length sequentially, the second one is silently dropped. \n\nThe contract isn't broken — `EBLookup.fit_platt` in `submission/caimira_lite.py:346` has the exact same caching behavior, so the design is intentional. But Kieran's bar would either:\n\n1. Rename `predict(record, labeled)` → `predict_and_calibrate(record, labeled)` to surface the side effect, OR\n2. Move calibration entirely outside `predict` so the caller does `predictor.calibrate(labeled); predictor.predict(record)` explicitly.\n\nThe second option is what the docstring at line 387-390 (`predict_batch`) actually does — it calls `self.calibrate(labeled)` once up front, then passes `None` into per-record `self.predict(r)`. So the design ALREADY has the clean pattern; `predict()`'s `labeled` arg is just convenience that hides a mutation. \n\nFlagging as P3 because both the kit contract (which submits `predict(input, labeled)`) and the upstream `Predictor.predict` shape encourage the surface that's currently there. A future refactor that separates `calibrate()` calls explicitly from `predict()` would be a public-API change. Mention in passing as a noteworthy design decision; not a bug.",
      "anchors": [
        "src/torch_measure/models/cold_start_lookup.py:374",
        "src/torch_measure/models/cold_start_lookup.py:388",
        "submission/caimira_lite.py:401"
      ],
      "evidence_kind": "design"
    },
    {
      "id": "CRIT-PY-13",
      "severity": "P3",
      "confidence": 75,
      "title": "Tests use `try/except + raise AssertionError` instead of `pytest.raises` (mixed across files)",
      "where": {
        "file": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/tests/test_models/test_caimira.py",
        "lines": "25-74"
      },
      "explanation": "`tests/test_models/test_caimira.py` uses the verbose `try/except` + manual-AssertionError-fallthrough pattern in five places (lines 25-35, 52-58, 60-66, 68-74). The sibling `test_cold_start_lookup.py` uses idiomatic `pytest.raises(ValueError)` context managers. The `test_caimira_lite.py` file also uses `pytest.raises`. The mixed style is a minor maintenance tax — the `try/except` pattern is harder to read at a glance, and (more importantly) it can silently pass if the code under test raises a DIFFERENT exception class than expected. Before:\n\n```python\ndef test_init_rejects_nonpositive_dims(self):\n    try:\n        CAIMIRA(n_subjects=5, n_items=10, embedding_dim=0, latent_dim=3)\n        raise AssertionError(\"Expected ValueError for embedding_dim=0\")\n    except ValueError:\n        pass\n```\n\nAfter:\n\n```python\ndef test_init_rejects_nonpositive_dims(self):\n    with pytest.raises(ValueError, match=\"embedding_dim\"):\n        CAIMIRA(n_subjects=5, n_items=10, embedding_dim=0, latent_dim=3)\n    with pytest.raises(ValueError, match=\"latent_dim\"):\n        CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=0)\n```\n\nThe `match=` argument also asserts the error message — catching regressions where the wrong-validation-clause raises. P3 because every test currently passes; this is a clarity boost.",
      "anchors": [
        "tests/test_models/test_caimira.py:25",
        "tests/test_models/test_caimira.py:52",
        "tests/test_models/test_caimira.py:60",
        "tests/test_models/test_caimira.py:68"
      ],
      "evidence_kind": "diff"
    },
    {
      "id": "CRIT-PY-14",
      "severity": "P3",
      "confidence": 80,
      "title": "tests/test_models/test_caimira.py missing `from __future__ import annotations` (inconsistent with rest of project)",
      "where": {
        "file": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/tests/test_models/test_caimira.py",
        "lines": "1-8"
      },
      "explanation": "Every new source file in the diff opens with `from __future__ import annotations`, and the sibling new test files `test_cold_start_lookup.py:14`, `test_llm_judge_irt.py:10`, `test_caimira_lite.py:11`, and `test_model_contract.py:11` all have it. `test_caimira.py` does NOT — it goes straight from the copyright header to `import torch`. The file as written works on Python 3.10+ because it doesn't use any PEP 604 union syntax or PEP 585 generics outside annotations, but the inconsistency is a small readability tax. Add the import:\n\nBefore (line 1-3):\n```python\n# Copyright (c) 2026 AIMS Foundations. MIT License.\n\nimport torch\n```\n\nAfter:\n```python\n# Copyright (c) 2026 AIMS Foundations. MIT License.\n\nfrom __future__ import annotations\n\nimport torch\n```\n\nP3 with confidence 80 because it's mechanical, low-cost, and consistent with all the other new files in this diff.",
      "anchors": [
        "tests/test_models/test_caimira.py:1",
        "tests/test_models/test_cold_start_lookup.py:14"
      ],
      "evidence_kind": "diff"
    },
    {
      "id": "CRIT-PY-15",
      "severity": "P3",
      "confidence": 65,
      "title": "test_caimira_lite.py:114 `parse_subject_name(None or \"\")` — confusing test for None handling",
      "where": {
        "file": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/tests/test_submission/test_caimira_lite.py",
        "lines": "112-114"
      },
      "explanation": "Line 114 reads:\n\n```python\nassert parse_subject_name(None or \"\") == \"\"\n```\n\nThe expression `None or \"\"` evaluates to `\"\"` at import time — so this is exactly the same test as line 113 (`parse_subject_name(\"\") == \"\"`), just with extra noise. The intent appears to be 'verify None input is handled', but the `or \"\"` short-circuits BEFORE the call, so `parse_subject_name` never actually sees `None`. \n\nSibling test `test_cold_start_lookup.py:168` does this correctly:\n\n```python\nassert parse_subject_name(None) == \"\"  # type: ignore[arg-type]\n```\n\nThe `# type: ignore[arg-type]` acknowledges the parameter is typed `str` and the call is deliberately violating the contract. \n\nBefore:\n```python\nassert parse_subject_name(None or \"\") == \"\"\n```\n\nAfter:\n```python\nassert parse_subject_name(None) == \"\"  # type: ignore[arg-type]\n```\n\nOR delete the redundant line entirely. P3 because the test passes today and the test ABOVE it (line 113) covers the empty-string case — the duplicate is just confusing.",
      "anchors": [
        "tests/test_submission/test_caimira_lite.py:114",
        "tests/test_models/test_cold_start_lookup.py:168"
      ],
      "evidence_kind": "diff"
    },
    {
      "id": "CRIT-PY-16",
      "severity": "P3",
      "confidence": 55,
      "title": "submission/model.py top-level docstring uses heavy reST blocks but `__doc__` is consumed by argparse downstream",
      "where": {
        "file": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/submission/train.py",
        "lines": "1-33, 84"
      },
      "explanation": "`submission/train.py` line 84 (`p = argparse.ArgumentParser(description=__doc__)`) passes the module docstring directly as argparse's `description`. The docstring includes ReST `::` blocks and section underlines (`Differences from`, `Smoke run`, `Full run`) on lines 12-33 that render fine in Sphinx but display as raw `::` and underline characters in `--help` output at the terminal. \n\nThe pragmatic fix: either keep the docstring rich (for Sphinx / IDE hover) and pass a separate plain-text `description` to argparse, OR drop the ReST markup. Verified by mentally executing `python submission/train.py --help` — the `description` line would show literal `::` markers. Before:\n\n```python\np = argparse.ArgumentParser(description=__doc__)\n```\n\nAfter:\n\n```python\np = argparse.ArgumentParser(\n    description=\"Offline trainer for the CAIMIRA+EB Codabench submission.\",\n)\n```\n\nP3 / confidence 55 because the impact is purely a one-time ugly `--help` output; nobody's been bitten by it in the diff history. Mention if the team values `--help` polish.",
      "anchors": [
        "submission/train.py:1",
        "submission/train.py:84"
      ],
      "evidence_kind": "design"
    }
  ],
  "residual_risks": [
    {
      "title": "submission/ is invisible to ruff CI — drift between src/ style and submission/ style",
      "summary": "The project's `.github/workflows/lint.yml` runs `ruff check src/ tests/` only. The 4 new submission/ files (`model.py`, `caimira_lite.py`, `labeling.py`, `train.py`) are NOT linted. Several of the findings above (CRIT-PY-3 stdlib-import-inside-function, CRIT-PY-5 missing annotation on `_candidate_count`, CRIT-PY-8 `Callable` from `typing`) would be auto-caught by the existing ruff `select = ['UP', 'E', 'F']` rules if `submission/` were added to the lint path. Recommend adding `submission/` to the lint workflow OR moving the linter discipline to a pre-commit hook that catches submission/ on the contributor side. The submission README's 'Open follow-ups' section acknowledges this gap but doesn't track when it'll be closed. Risk: as the submission/ codebase grows, style drift between src/ and submission/ will accumulate silently — and the team's audit gates (D-9) cannot catch type-hint or naming regressions.",
      "anchors": [".github/workflows/lint.yml", "submission/README.md", "pyproject.toml:[tool.ruff.lint]"]
    },
    {
      "title": "Test files are exempt from typing discipline (`def test_foo(self):` with no return type) — fine, but fixture functions are inconsistently typed",
      "summary": "Spot-check of `tests/test_models/test_cold_start_lookup.py:33-82` shows fixtures `lookup_tables() -> dict` (line 34) and `predictor(lookup_tables: dict) -> ColdStartLookupPredictor` (line 73) are well-typed. But `tests/test_models/test_llm_judge_irt.py:28` types `minimal_lookup() -> ColdStartLookupPredictor` correctly, while `tests/test_submission/test_caimira_lite.py:144-152` has `def lookup(self) -> EBLookup:` typed but the class-level test methods (line 154+) take `lookup` as an untyped fixture parameter (`def test_level1_sbc_triple(self, lookup):` — no annotation on `lookup`). This is the prevailing pytest style and arguably fine, but inconsistent within the same file. Not a finding — a residual stylistic risk worth one minute of cleanup if the team's bar wants test-fixture parameters typed.",
      "anchors": ["tests/test_models/test_llm_judge_irt.py:28", "tests/test_submission/test_caimira_lite.py:144"]
    },
    {
      "title": "`from __future__ import annotations` everywhere but no `from typing import TYPE_CHECKING` in any new file that uses forward refs",
      "summary": "The new `caimira.py` has `from __future__ import annotations` (line 17) and uses `data` as an unannotated parameter (CRIT-PY-6) rather than introducing a `TYPE_CHECKING` block to forward-ref `LongFormData`. Sibling `_base.py:13-14` shows the canonical pattern for this. The diff misses three opportunities to use `TYPE_CHECKING` blocks (caimira.py, cold_start_lookup.py, llm_judge_irt.py) and instead falls back to either `typing.Any` (for `record: dict[str, Any]`) or unannotated parameters. If the project ever adopts mypy strict mode, these will surface as errors. Residual risk: the typing discipline that the upstream maintainers added in `_base.py` and `_network_base.py` (both have `TYPE_CHECKING` blocks) is not being mirrored in the new module files.",
      "anchors": ["src/torch_measure/models/caimira.py:17", "src/torch_measure/models/_base.py:13"]
    },
    {
      "title": "Doctest in CAIMIRA docstring (caimira.py:92-106) is not executed by CI",
      "summary": "The `Examples` section in `CAIMIRA`'s class docstring (lines 92-106) uses `>>> ` doctest syntax. The project's `pyproject.toml` does NOT enable `--doctest-modules` in `[tool.pytest.ini_options]`. The doctest is therefore documentation-only — if a refactor breaks the example, nothing catches it. Sibling `Rasch`, `TwoPL`, `AmortizedIRT` docstrings do NOT use doctest syntax, so the inconsistency is also stylistic. Either enable doctest-modules in pytest config OR rewrite the `Examples` block to be illustrative-only (e.g., remove the `>>> ` prefix and present as Python pseudocode). Residual risk: a future code change that breaks the example will go unnoticed until a human reads the docstring.",
      "anchors": ["src/torch_measure/models/caimira.py:92", "pyproject.toml:[tool.pytest.ini_options]"]
    }
  ],
  "testing_gaps": [
    {
      "title": "No test exercises the CAIMIRA + EB hybrid `_blend_logits` path in submission/model.py",
      "summary": "`submission/model.py` defines `_blend_logits(caimira_p, eb_p, lambda_=0.6) -> float` (line 115-117) — the core hybrid prediction primitive. The contract test `tests/test_submission/test_model_contract.py` exercises `predict()` only in `LOCAL_SMOKE=1` mode (returns fixed `0.5`, line 220-221), so neither the CAIMIRA path nor the EB path nor the blend ever runs in the test suite. The blend's numerical behavior (lambda=1 → pure CAIMIRA, lambda=0 → pure EB, intermediate values produce logit-space interpolation) is asserted nowhere. Add a unit test that:\n\n1. Mocks `CAIMIRA.caimira_logit` and `EB.lookup_p` to return known values.\n2. Calls `_blend_logits(0.7, 0.3, lambda_=0.6)`.\n3. Asserts the result equals `_sigmoid(0.6 * logit(0.7) + 0.4 * logit(0.3))`.\n\nThis covers a P0-class regression risk: if anyone changes `_BLEND_LAMBDA` or `_blend_logits`'s formula, the contract test won't catch it.",
      "anchors": ["submission/model.py:115", "tests/test_submission/test_model_contract.py:23"]
    },
    {
      "title": "No test exercises the out-of-vocab subject fallback branch (lines 239-245 of submission/model.py)",
      "summary": "`submission/model.py` lines 239-245 contain the EB-only fallback path:\n\n```python\nif subject_idx is None or not item_content:\n    p = eb_p  # EB-only path\nelse:\n    item_embedding = _encode_item(item_content)\n    caimira_logit = CAIMIRA.caimira_logit(subject_idx, item_embedding)\n    caimira_p = _sigmoid(caimira_logit)\n    p = _blend_logits(caimira_p, eb_p, _BLEND_LAMBDA)\n```\n\nThe `subject_idx is None` branch (out-of-vocab subject) is critical for cold-start subjects that didn't exist in `META['subject_to_idx']`. The contract test never exercises this — it always runs in LOCAL_SMOKE which returns `0.5` before reaching this branch. Add a non-LOCAL_SMOKE integration test with a synthetic `CAIMIRA + EB + META` triple that confirms the branch picks `eb_p` for an unknown subject.",
      "anchors": ["submission/model.py:239"]
    },
    {
      "title": "Test fixtures use degenerate sbc/sb data — no test validates the `_lookup_p` levels in the order they actually fire",
      "summary": "`tests/test_models/test_cold_start_lookup.py:40-69` constructs a hand-crafted lookup where each level is reachable by some test case. But the data is small (3 subjects, 3 benchmarks) and the test ordering is by NAME (`test_level1_triple_match`, `test_level2_pair_match`, ...), not by data-driven scenario. There's no test that asserts the LEVEL-RESOLUTION ORDER is preserved when multiple levels could fire — e.g., if `(subj, bench, cond)` is in both `sbc` AND `sb`, the test would silently pass either way. The 6-level fallback contract is the load-bearing design — adding an explicit `test_levels_resolve_in_priority_order` that injects ALL six matches for the same query and asserts level 1 wins, level 2 if 1 absent, etc., would catch any future reorder regression.",
      "anchors": ["tests/test_models/test_cold_start_lookup.py:100", "src/torch_measure/models/cold_start_lookup.py:288"]
    },
    {
      "title": "No test forces `acquisition_function` reservoir to roll over past `_MAX_SEEN = 128`",
      "summary": "`submission/labeling.py:161-171` implements bounded-reservoir SimHash diversity with `_MAX_SEEN = 128`. The hash-of-(candidate_key, count) → slot logic on line 166-171 is the load-bearing eviction primitive — but no test in `test_model_contract.py` or anywhere else feeds it more than 128 candidates and asserts (a) the reservoir size stays at 128, (b) old signatures get evicted, (c) signatures are evicted UNIFORMLY (not biased toward later candidates). At 10K candidates per round, this code runs ~10K times. Add a test that pushes 256 distinct candidates through `acquisition_function` and asserts `len(labeling._seen_signatures) == 128` and that the reservoir is approximately uniform across the 256 inputs (within statistical tolerance).",
      "anchors": ["submission/labeling.py:161", "tests/test_submission/test_model_contract.py:139"]
    },
    {
      "title": "`CAIMIRA.compute_item_params(center=\"invalid\")` raises ValueError but no test asserts it",
      "summary": "`src/torch_measure/models/caimira.py:210-211` raises `ValueError(f\"Unknown center mode {center!r}; ...\")` for unrecognized `center` values. `test_caimira.py` covers the three valid modes (`auto`, `dynamic`, `frozen`) but never asserts the rejection path. If CRIT-PY-2 is acted on (tightening to `Literal[...]`), this gap closes via the type checker — but in the meantime, a small `pytest.raises(ValueError)` test would also document the runtime contract. Note: this is exactly the kind of edge-case that a `Literal[\"auto\", \"dynamic\", \"frozen\"]` annotation would make redundant at the type-check layer; if you fix CRIT-PY-2, keep the runtime check + add a test simultaneously.",
      "anchors": ["src/torch_measure/models/caimira.py:210", "tests/test_models/test_caimira.py:10"]
    }
  ]
}
```
