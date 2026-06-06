---
reviewer: maintainability
artifact_id: Branch feat-caimira (vasundras/torch_measure)
round: 1
date: 2026-05-22
---

# Maintainability review — feat/caimira

```json
{
  "reviewer": "maintainability",
  "findings": [
    {
      "id": "CRIT-MAINT-1",
      "title": "Duplicated stdlib helpers across submission/caimira_lite.py and src/torch_measure/models/cold_start_lookup.py with no mechanical parity test",
      "severity": "P1",
      "confidence": 90,
      "anchor_kind": "code",
      "applies_to": "applies_to_both",
      "files": [
        {
          "path": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/submission/caimira_lite.py",
          "lines": "79-172"
        },
        {
          "path": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/src/torch_measure/models/cold_start_lookup.py",
          "lines": "48-154"
        },
        {
          "path": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/tests/test_submission/test_caimira_lite.py",
          "lines": "35-138"
        }
      ],
      "evidence": "Six top-level symbols are duplicated character-for-character between the two files: `_DEFAULT_PROVIDER_PREFIXES` (caimira_lite.py:79-83 vs cold_start_lookup.py:48-52), `_logit` (caimira_lite.py:97-100 vs cold_start_lookup.py:63-66), `_sigmoid` (caimira_lite.py:103-108 vs cold_start_lookup.py:69-74), `_clip` (caimira_lite.py:111-112 vs cold_start_lookup.py:77-78), `parse_subject_name` (caimira_lite.py:120-139 vs cold_start_lookup.py:81-99), and `resolve_subject_name` (caimira_lite.py:142-172 vs cold_start_lookup.py:102-154). The standalone copy is intentional and well-justified — the file-level docstring at caimira_lite.py:17-20 explains: *\"The hosted Codabench container ships an organizer-supplied `torch_measure` package that may NOT track this fork's `feat/caimira` branch. The ZIP must carry its own self-contained implementation.\"* But there is NO test that compares the two implementations mechanically. `TestCAIMIRALiteStateDictParity` (tests/test_submission/test_caimira_lite.py:35-63) covers the `CAIMIRALite` ↔ `CAIMIRA` state_dict surface, but it does NOT verify that `submission.caimira_lite.parse_subject_name` produces the same output as `torch_measure.models.cold_start_lookup.parse_subject_name` for any input, nor that `_DEFAULT_PROVIDER_PREFIXES` matches, nor that the EB lookup hierarchy returns identical values across `EBLookup` and `ColdStartLookupPredictor` for the same lookup tables. Drift between the two helpers will silently produce a competition submission that returns one prediction at training time and a different prediction at inference time, with no test failure to catch it. The submission/README.md \"Why standalone\" section (lines 116-127) describes the situation but does not call out the test gap.",
      "why_it_matters": "Code duplication without a parity test is the classic two-implementations-of-one-contract maintainability failure mode. If a future contributor fixes a bug in `torch_measure.models.cold_start_lookup.parse_subject_name` (say, to handle a new provider prefix like `mistralai-foundation/`), the submission code at runtime will NOT pick up the fix unless they also edit `submission/caimira_lite.py` AND remember to run the submission ZIP build to regenerate the artifact. The duplication is justified by the standalone-ZIP constraint; the missing parity test is not.",
      "suggested_fix": "Add a single parametrized test in `tests/test_submission/test_caimira_lite.py` (or a new `tests/test_submission/test_helper_parity.py`) that asserts byte-equivalence of the six duplicated symbols. Concrete shape: (1) `assert submission.caimira_lite._DEFAULT_PROVIDER_PREFIXES == torch_measure.models.cold_start_lookup._DEFAULT_PROVIDER_PREFIXES`; (2) `pytest.mark.parametrize` over a list of representative inputs (empty string, `\"Name: gpt-4\"`, `\"meta-llama/Llama-2-7b-chat\"`, multi-line content, etc.) and assert `submission.caimira_lite.parse_subject_name(x) == torch_measure.models.cold_start_lookup.parse_subject_name(x)` and same for `resolve_subject_name` with a fixed-fixture lookup table; (3) for the numerical helpers `_logit` and `_sigmoid`, parametrize over `[1e-9, 0.05, 0.5, 0.95, 1 - 1e-9]` and `[-10, -1, 0, 1, 10]` respectively and assert exact-equality (these are pure-stdlib so the equality is genuinely defined). The new test sits in test_submission/ so it only runs in the fork's CI matrix; it is fork-only and does NOT need to ship upstream.",
      "applies_to_class": "fork_only_caimira_branch",
      "rationale": "The duplication itself lives in fork-only `submission/` — but the parity test it requires only makes sense WHILE the duplication exists, which is for the lifetime of the fork's Codabench submission. If the fork is later structured to export `caimira_lite.py` as a vendored copy of `cold_start_lookup.py` symbols (or vice versa), the parity test guards the vendoring discipline."
    },
    {
      "id": "CRIT-MAINT-2",
      "title": "Triplicated _logit/_sigmoid stubs in submission/model.py duplicate caimira_lite imports already in the same file",
      "severity": "P2",
      "confidence": 85,
      "anchor_kind": "code",
      "applies_to": "fork_only_caimira_branch",
      "files": [
        {
          "path": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/submission/model.py",
          "lines": "97-117"
        },
        {
          "path": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/submission/caimira_lite.py",
          "lines": "97-108"
        }
      ],
      "evidence": "`submission/model.py` already imports five symbols from `caimira_lite` at the top of the file (model.py:72-78: `EMBED_DIM, CAIMIRALite, EBLookup, clip_for_predict, parse_subject_name`). But then defines its OWN `_logit` (model.py:97-102) and `_sigmoid` (model.py:105-112) at module level — both of which are character-identical to `caimira_lite._logit` and `caimira_lite._sigmoid`. Each of these two functions ALSO does a per-call `import math` at function body (model.py:99 and model.py:107) — not a module-level import. This is a third copy of `_logit`/`_sigmoid` in the submission/ tree, alongside the two already noted in CRIT-MAINT-1.",
      "why_it_matters": "Three independent copies of two simple math helpers across the same 408-line submission/ directory is more cognitive load than the problem warrants. The per-function `import math` is also unusual: `math` is imported nowhere else in model.py (model.py:58-64 imports json/os/sys/pathlib/typing/torch only), so a reader scanning the imports cannot tell that `math` is used until they read the function bodies. If a future contributor decides to use `math.isnan` or `math.isfinite` defensively in `predict()`, they will either re-import math at the function level (continuing the pattern) or move it to module-level — at which point the two function-level imports become visibly redundant. The simplest fix removes both ambiguities at once.",
      "suggested_fix": "Replace model.py:97-112 (the two helper definitions) with: `from caimira_lite import _logit as _logit, _sigmoid as _sigmoid  # re-export for local readability` at the top of model.py near the existing caimira_lite import block (model.py:72-78). Drop the `import math` lines entirely. This eliminates the third copy and makes the dependency on caimira_lite's helpers explicit. The leading underscore on the imported names is OK as a fork-only import (they are explicitly private in caimira_lite but the import lives in the same submission/ directory). If the reviewer prefers not to import private symbols, an alternative is to lift `_logit` and `_sigmoid` in caimira_lite from `_underscore` to public (rename to `logit_safe` / `sigmoid_safe` or similar) since they are pure-stdlib and already documented.",
      "applies_to_class": "fork_only_caimira_branch",
      "rationale": "All edits land in submission/, which is fork-only by design (per CONTRIBUTING.md the package is flat-by-domain under src/torch_measure/; submission/ is not part of the upstream surface)."
    },
    {
      "id": "CRIT-MAINT-3",
      "title": "ColdStartLookupPredictor breaks the documented Predictor contract — dict-shaped predict() instead of (subject_idx, item_idx) tensor query — and inherits from `object` rather than `Predictor`",
      "severity": "P1",
      "confidence": 70,
      "anchor_kind": "code",
      "applies_to": "upstream_eligible",
      "files": [
        {
          "path": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/src/torch_measure/models/cold_start_lookup.py",
          "lines": "157-211"
        },
        {
          "path": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/src/torch_measure/models/_predictor.py",
          "lines": "14-65"
        },
        {
          "path": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/src/torch_measure/models/_base.py",
          "lines": "17-28"
        },
        {
          "path": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/src/torch_measure/models/tabpfn_predictor.py",
          "lines": "23-43"
        }
      ],
      "evidence": "`Predictor` is documented as the abstract base for any model producing P(correct) over (subject, item) cells (`_predictor.py:14-25`): subclasses implement `predict(query: dict[str, torch.Tensor]) -> torch.Tensor` returning a 1-D probability tensor over rows. `IRTModel.fit` docstring (`_base.py:26-28`) explicitly carves out the path for non-factor predictors: *\"For non-factor predictors (TabPFN-style, neural baselines), inherit `Predictor` directly instead.\"* `TabPFNPredictor` follows this path (`tabpfn_predictor.py:23` `class TabPFNPredictor(Predictor)` with `predict(query) -> torch.Tensor`). `ColdStartLookupPredictor`, however, inherits from `object` (`cold_start_lookup.py:157` `class ColdStartLookupPredictor:`) and defines `predict(self, record: dict[str, Any], labeled: list[dict[str, Any]] | None = None) -> float` (`cold_start_lookup.py:338-380`) — the record is a single competition-format dict with `subject_content`/`benchmark`/`condition`/`item_content` STRING keys, and the return is a single Python float, not a tensor.",
      "why_it_matters": "The class is exported as `torch_measure.models.ColdStartLookupPredictor` (`models/__init__.py:14, __all__:37`) and lives in the public API alongside Rasch / TwoPL / AmortizedIRT / TabPFNPredictor — but it has a fundamentally different shape than every other predictor in the package. A library user who reads CONTRIBUTING.md or `_predictor.py` will expect `predict(query)` to take long-form tensors. They will write code like `predict_dense(coldstart_predictor)` or `coldstart_predictor.predict({\"subject_idx\": ..., \"item_idx\": ...})` — both of which will fail with `KeyError`. The class is also not an `nn.Module` (no parameters, no `.to(device)`, no `state_dict()`), so `predict_dense` cannot be wrapped on it either. The library effectively gains TWO incompatible `predict()` signatures with the same module-level export pattern, and the difference is only documented in the class docstring (`cold_start_lookup.py:158-209`) rather than at the type-system level. The judgment call here is real: the cold-start setting GENUINELY operates on text records, not integer indices (the docstring's explanation at `cold_start_lookup.py:6-13` is correct that *\"at competition test time, item IDs are NEW (never seen during training), so integer-based lookup is impossible\"*). So either: (a) the class should NOT inherit from `Predictor` AND should NOT export from `torch_measure.models` (move to `torch_measure.experimental.cold_start` or similar), OR (b) it should adopt the `Predictor(nn.Module, ABC)` contract by accepting `query` dicts and dispatching internally to the dict-shaped lookup. Option (b) is the cleaner fix because it preserves the public-API consistency. The same comment applies to `LLMJudgeIRT` (`llm_judge_irt.py:123-151` — same `predict(record) -> float` shape).",
      "suggested_fix": "Option (b): refactor `ColdStartLookupPredictor.predict(record, labeled)` to `predict(query: dict[str, Tensor | list]) -> Tensor` that accepts the canonical Predictor contract AND falls back to record-shape when the query carries text fields. Concrete shape: define a new method `predict_record(record: dict, labeled: list[dict] | None) -> float` that contains the current `predict()` body verbatim, then make `predict(query)` dispatch on the type of `query['subject_idx']` (LongTensor → integer lookup path raising NotImplementedError since the class is cold-start by design; list/None → loop predict_record over `query.get('records', [])`). Update `tests/test_models/test_cold_start_lookup.py` to test BOTH shapes. Same edit pattern applies to `LLMJudgeIRT`. ALTERNATIVELY, option (a): rename the import to `torch_measure.experimental.cold_start.ColdStartLookupPredictor` and drop it from `models/__all__`; the class is then visibly NOT-a-Predictor and the user is on notice that it does not follow the IRT/factor-model contract. The decision between (a) and (b) is a library-design call; I lean toward (b) for consistency, but a maintainer who values experimentality may prefer (a). Either way, NOT-status-quo is the right move before the upstream PR, because the class is the first non-Predictor export in `torch_measure.models`.",
      "applies_to_class": "upstream_eligible",
      "rationale": "The non-Predictor contract change is in `src/torch_measure/models/cold_start_lookup.py` and `src/torch_measure/models/llm_judge_irt.py` — both upstream-eligible files. The decision-record cost (option a vs option b) is also a public-API decision. The fork-only `submission/` code does not need to change either way."
    },
    {
      "id": "CRIT-MAINT-4",
      "title": "LLMJudgeIRT exported as public API despite being marked NOT-FOR-PRODUCTION and never recommended by its own docstring",
      "severity": "P2",
      "confidence": 80,
      "anchor_kind": "code",
      "applies_to": "upstream_eligible",
      "files": [
        {
          "path": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/src/torch_measure/models/llm_judge_irt.py",
          "lines": "1-37"
        },
        {
          "path": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/src/torch_measure/models/__init__.py",
          "lines": "17-39"
        }
      ],
      "evidence": "`llm_judge_irt.py:1-37` module docstring includes a `Negative-result disclosure` section: *\"Across three rigorously validated iterations on the competition data ... the judge consistently extracted a real but very small signal\"* and *\"The model is therefore documented here for completeness and reproducibility but is NOT the recommended deployment target — the simpler `ColdStartLookupPredictor` is preferred for its robustness.\"* The class docstring at `llm_judge_irt.py:104-106` reinforces this: *\"See module docstring for the negative-result disclosure. Production code should prefer `ColdStartLookupPredictor` directly.\"* But `__init__.py:17` imports both `LLMJudgeIRT` and `build_difficulty_prompt` from this module, and `__init__.py:38-39` adds both to `__all__`. So `from torch_measure.models import LLMJudgeIRT` works at the same import-namespace surface as `from torch_measure.models import Rasch`/`TwoPL`/`AmortizedIRT` — no namespace signal that the class is experimental.",
      "why_it_matters": "A library user reading the `__init__.py` `__all__` list (the canonical public API enumeration) will see `LLMJudgeIRT` alongside the production-grade IRT models and reasonably conclude that it's a recommended option. The disclaimer is only visible after they `import` the class and read the docstring — three steps later. This is the textbook pattern that motivates `experimental` sub-namespaces in NumPy / PyTorch / scikit-learn: keeping non-production code at the top-level export surface invites adoption, then breakage when the negative result motivates removal. Build_difficulty_prompt is even more borderline — it's a string formatter for one specific judge prompt, not a model. The fact that it's at the same export tier as `Rasch` is structural noise.",
      "suggested_fix": "Move `LLMJudgeIRT` + `build_difficulty_prompt` to a new module `src/torch_measure/experimental/llm_judge_irt.py` (or `src/torch_measure/models/_experimental/llm_judge_irt.py` for a soft variant). Update `__init__.py` to NOT import them, and add `src/torch_measure/experimental/__init__.py` that re-exports them with an explicit `__experimental__ = True` flag. Update the tutorial notebook (`tutorials/predictive_evaluation_challenge.ipynb`) and `tests/test_models/test_llm_judge_irt.py` to import from the new path. Document the experimental namespace policy in CONTRIBUTING.md (it's currently silent on the topic). If the maintainer prefers to keep top-level exports for discoverability, the alternative is to add a `_EXPERIMENTAL_WARNING` `UserWarning` raised once per import: `warnings.warn(\"LLMJudgeIRT is experimental — see module docstring for negative-result disclosure\", UserWarning, stacklevel=2)` at module bottom. This is less invasive but doesn't fix the `__all__` enumeration.",
      "applies_to_class": "upstream_eligible",
      "rationale": "The decision lives in `src/torch_measure/models/llm_judge_irt.py` and the package `__init__.py` — both upstream-eligible. The tutorial notebook (also added in this diff) would need to follow."
    },
    {
      "id": "CRIT-MAINT-5",
      "title": "Module-level magic constants in submission/train.py lack provenance citations (where do _EB_SB_ALPHA=5.0, _EB_SBC_MIN_N=3, _EB_SUBJ_MIN_N=10 come from?)",
      "severity": "P2",
      "confidence": 75,
      "anchor_kind": "code",
      "applies_to": "fork_only_caimira_branch",
      "files": [
        {
          "path": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/submission/train.py",
          "lines": "71-78"
        }
      ],
      "evidence": "`submission/train.py:71-78` declares four module-level constants used in `build_eb_tables`: `_EB_SBC_MIN_N = 3` (\"triple cell needs >= 3 obs to land in sbc\"), `_EB_SUBJ_MIN_N = 10` (\"subject prior needs >= 10 obs\"), `_EB_SB_ALPHA = 5.0` (\"Bayesian pseudo-counts toward IRT blend at level 2\"), and `_TABLE_CLIP_LO / _TABLE_CLIP_HI = 0.05, 0.95` (\"Avoids inf in logit()\"). The comments tell you WHAT each does, but not WHERE the value came from. Is `_EB_SB_ALPHA=5.0` (a) the value from a published Bayesian-shrinkage reference (cite it), (b) the value the team validated on a held-out NLL sweep (cite the sweep — date / commit / wandb-run-id), or (c) a tuning intuition (\"~5 prior observations per shrinkage target makes the prior worth one shrinkage cycle\")? Same for `_EB_SUBJ_MIN_N=10` and `_EB_SBC_MIN_N=3` — these affect the partition between EB Level 1/2 vs Level 4/5/6, which is load-bearing for the predict-time behavior. The submission/README.md does not mention these constants either.",
      "why_it_matters": "Magic numbers without provenance are the single most common source of \"why is this value 5.0?\" debugging sessions in ML pipelines. A future contributor who wants to lift the `_EB_SUBJ_MIN_N` threshold to 20 (to be more conservative) cannot tell whether they would be (a) re-tuning a free hyperparameter that should be on a validation sweep, (b) deviating from a paper-recommended setting, or (c) breaking a contract with the EB lookup-table format that downstream code depends on. The D-9 transfer-audit-gate context (per the parent CLAUDE.md) makes provenance even more load-bearing: when sub-gate (c) FAILS on a real-train artifact, the first debugging move is to sweep these constants — and without a provenance trail, the contributor wastes time re-deriving values that the author already knew.",
      "suggested_fix": "Replace the four constant declarations at submission/train.py:71-78 with constants that carry inline provenance comments. Concrete shape: `_EB_SB_ALPHA = 5.0  # Pseudo-count for the Bayesian-shrinkage step at EB level 2.\\n# Rationale: ~5 prior observations per (subject, benchmark) cell is the\\n# threshold at which the IRT-blend prior contributes equally to the\\n# observed mean (under the Beta(alpha, alpha) symmetry). Empirically\\n# validated in [PR-NUMBER or sweep-id] on the binary-benchmark fold of\\n# aims-foundations/measurement-db@589ccfdb...`. Same pattern for the two _MIN_N constants. If the values are genuinely unmotivated (i.e. defaults from a notebook), say so explicitly: `# UNCHECKED DEFAULT — not validated by sweep; consider tuning if D-9 sub-gate (c) FAILS`. The point is that a future reader can distinguish at-a-glance between provenance-backed vs unchecked-default constants. The constants in submission/labeling.py (`_BITS, _MAX_TOKENS, _MAX_SEEN, _MAX_STRATA, _TIE_EPSILON`) are already self-documenting (their values are mechanically derived from the algorithm: 64-bit SimHash → _BITS=64; bounded-cost reservoir → _MAX_SEEN). The submission/train.py constants are different in kind because they affect numerical fitting decisions.",
      "applies_to_class": "fork_only_caimira_branch",
      "rationale": "submission/train.py is fork-only Codabench-submission code; the upstream `torch_measure.models.ColdStartLookupPredictor` does NOT have these constants because it loads EB tables from JSON rather than building them. The decision-record cost lives entirely inside the fork."
    },
    {
      "id": "CRIT-MAINT-6",
      "title": "_BLEND_LAMBDA is both a module-level constant AND a parameter default, with redundant explicit-default passing at the call site",
      "severity": "P3",
      "confidence": 90,
      "anchor_kind": "code",
      "applies_to": "fork_only_caimira_branch",
      "files": [
        {
          "path": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/submission/model.py",
          "lines": "80-83, 115-117, 250"
        }
      ],
      "evidence": "`submission/model.py:80-83` declares `_BLEND_LAMBDA: float = 0.6` with a docstring-comment about the conservative default. `model.py:115-117` declares `def _blend_logits(caimira_p: float, eb_p: float, lambda_: float = _BLEND_LAMBDA) -> float:` — the parameter defaults to the module constant. Then `model.py:250` calls `_blend_logits(caimira_p, eb_p, _BLEND_LAMBDA)` — explicitly passing the same default value. So three references exist to the same value: the constant declaration, the parameter default, and the explicit call-site argument. The two co-existing patterns (parameter w/ default + call-site explicit pass) suggest the author was either (a) preparing for D-9 ablation (per the comment at model.py:82: \"D-9 ablation candidates: {0.3, 0.5, 0.7, 0.9}\") and would later call `_blend_logits(...; lambda_=0.5)`, OR (b) defensive-coded against accidental future re-imports.",
      "why_it_matters": "When `_BLEND_LAMBDA` is touched (e.g. for the D-9 ablation candidates: {0.3, 0.5, 0.7, 0.9}), the question is: where does the contributor change the value? The current code has three answers — the constant, the parameter default, or the call site — and any single edit changes the value at one but not all. A reader cannot tell from `predict()` at line 250 whether `lambda_=_BLEND_LAMBDA` is being passed because the author specifically wanted the module-level constant OR because the call-site default WAS originally a different value that was later reset. The fix is cheap and clarifying.",
      "suggested_fix": "Pick one of two patterns: (1) DELETE the parameter default and always read from the module constant — change `def _blend_logits(caimira_p, eb_p, lambda_):` and document at the call site that this is fork-only Codabench code; OR (2) DROP the explicit `_BLEND_LAMBDA` argument at model.py:250 — change to `_blend_logits(caimira_p, eb_p)` and let the parameter default fire. Option (2) is the more common Python convention (\"the default is for the public-API call site\"). Either edit reduces the three-way reference to two-way.",
      "applies_to_class": "fork_only_caimira_branch",
      "rationale": "submission/model.py is fork-only by design; the constants and helper are not part of the upstream surface."
    },
    {
      "id": "CRIT-MAINT-7",
      "title": "submission/model.py module-level `_item_cache` lacks the comment that explains its unboundedness is safe (per-round container lifecycle)",
      "severity": "P3",
      "confidence": 80,
      "anchor_kind": "code",
      "applies_to": "fork_only_caimira_branch",
      "files": [
        {
          "path": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/submission/model.py",
          "lines": "165-183"
        }
      ],
      "evidence": "`model.py:165` declares `_item_cache: dict[str, torch.Tensor] = {}` at module scope. `_encode_item` (model.py:168-183) populates it on every cache miss — there is no eviction policy, no size cap, no LRU/TTL. The comment at model.py:169 says \"Encode `item_content` once per round (module-level cache)\" — but does NOT explain the load-bearing reason this is OK: the Codabench container is restarted between rounds, so the module-level cache is reset when the process dies, which IS the cap. A reader who sees an unbounded `dict[str, Tensor]` populated by a hot-path encoder might reasonably worry about memory growth across rounds, or wonder whether to add an LRU. They have to read the parent CLAUDE.md or the kit docs to know that the container is single-use.",
      "why_it_matters": "Unbounded caches are P0 maintainability red-flags in long-running services; they are P3 in single-use sandboxed containers. The current code is the second; the comment doesn't say so. A reader assessing whether to port this code to a long-running service (e.g. a CI-hosted submission validation harness) needs to know that the unboundedness is a load-bearing constraint, not a defect to fix.",
      "suggested_fix": "Expand the inline comment at model.py:169 to: `\"\"\"Encode ``item_content`` once per round.\\n\\nThe cache is module-level and intentionally unbounded — the hosted\\nCodabench container is single-use per round (see\\n``starting_kit/README.md:293-297`` for the per-round-container lifecycle),\\nso the process restart between rounds IS the eviction policy. Each\\nround processes ~10K items; at 768-dim float32 encoder output that's\\n~30 MB, well below the container's memory cap. Do NOT add an LRU here\\nwithout first verifying the round-lifecycle assumption against the\\ncurrent kit. \\\"\"\"`. This makes the safety argument visible.",
      "applies_to_class": "fork_only_caimira_branch",
      "rationale": "submission/model.py is fork-only Codabench-submission code; the lifecycle assumption it depends on is platform-specific."
    },
    {
      "id": "CRIT-MAINT-8",
      "title": "CAIMIRA.compute_item_params `center: str` parameter is stringly-typed where a Literal[\"auto\", \"dynamic\", \"frozen\"] would give IDE support",
      "severity": "P3",
      "confidence": 95,
      "anchor_kind": "code",
      "applies_to": "upstream_eligible",
      "files": [
        {
          "path": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/src/torch_measure/models/caimira.py",
          "lines": "161-211"
        }
      ],
      "evidence": "`caimira.py:164` declares `center: str = \"auto\"`. The docstring at `caimira.py:174-185` enumerates the three valid values: `\"auto\"`, `\"dynamic\"`, `\"frozen\"`. The function body at `caimira.py:203-211` dispatches on the three values and raises `ValueError` for anything else: `raise ValueError(f\"Unknown center mode {center!r}; expected 'auto', 'dynamic', or 'frozen'.\")`. So the contract IS \"one of three string literals\" — but the type system sees only `str`. A reader using a modern IDE / mypy / pyright would benefit from `Literal[\"auto\", \"dynamic\", \"frozen\"]`: auto-complete in `model.compute_item_params(center=\"<TAB>\")` would propose only the three valid values; passing `center=\"FROZEN\"` (uppercase typo) would be flagged at type-check time, not runtime.",
      "why_it_matters": "Stringly-typed enums are a common Python anti-pattern that the type system can catch with one annotation change. The cost of the annotation is one import (`from typing import Literal`); the benefit is IDE auto-completion + static checking for callers. Same pattern applies (informationally) to a handful of arg names in `submission/train.py` (e.g. `args.device: str = None` could be `str | None` and the resolution at train.py:316 could be cleaner) but those are fork-only.",
      "suggested_fix": "Change `caimira.py:164` from `center: str = \"auto\",` to `center: Literal[\"auto\", \"dynamic\", \"frozen\"] = \"auto\",`. Add `from typing import Literal` near the top of caimira.py (currently only `torch` and `from torch import nn` are imported). Keep the existing runtime validation at lines 203-211 — `Literal` is enforced at type-check time only, so the runtime guard remains useful for callers passing non-literal `str` values. Update the docstring to mention the new annotation (single-line addition).",
      "applies_to_class": "upstream_eligible",
      "rationale": "src/torch_measure/models/caimira.py is the upstream-PR target. The `Literal` annotation is a small, focused improvement that signals modern-typing discipline to the upstream reviewers."
    },
    {
      "id": "CRIT-MAINT-9",
      "title": "submission/ directory is at the repo root, not under src/torch_measure/ — but the README correctly identifies this as fork-only scaffolding",
      "severity": "P3",
      "confidence": 90,
      "anchor_kind": "code",
      "applies_to": "fork_only_caimira_branch",
      "files": [
        {
          "path": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/submission/README.md",
          "lines": "1-19"
        }
      ],
      "evidence": "CONTRIBUTING.md (per the primer §Project Standards) says: *\"The package is organized **flat-by-domain** under `src/torch_measure/` — each domain (`models/`, `metrics/`, `fitting/`, `data/`, `datasets/`, `cat/`, `viz/`) has a matching `tests/test_<domain>/` directory.\"* The new `submission/` directory at repo root is NOT under `src/torch_measure/` and is NOT on the listed domains. submission/README.md:1-19 explicitly identifies the directory as Codabench-bound scaffolding rather than library code, and the file/directory layout (mirrored by tests/test_submission/) is internally consistent. The structural mismatch is acknowledged by the README so it does NOT propagate into upstream-PR confusion.",
      "why_it_matters": "On its own this is a minor architectural choice — competition-submission code is a different lifecycle from library code, and putting it at repo root rather than inside the package keeps it from being imported via the library's namespace. The risk would have been the upstream PR accidentally including submission/ alongside the new model files; the diff structure (submission/ is its own logical sibling to src/) makes that easy to avoid. The README's `> **Status: scaffolding, not yet active.**` callout at lines 12-16 is the right discipline.",
      "suggested_fix": "No change required. Optional: add a single `# Not part of the upstream library — see submission/README.md` comment to the .gitignore lines that gitignore submission artifacts (already covered by the `*.pt` global pattern and the explicit submission entries at .gitignore L40-44 per the primer). Optional: if/when the fork upstreams the model files, add an explicit `.upstreamignore` or CONTRIBUTING-side note saying \"submission/ is fork-only and must NOT be included in the upstream PR\".",
      "applies_to_class": "fork_only_caimira_branch",
      "rationale": "submission/ is the canonical example of fork-only code — Codabench-competition scaffolding does not belong in the upstream library. The structural decision is correct; flagged here only because a future contributor might propose moving it without realizing the design intent."
    },
    {
      "id": "CRIT-MAINT-10",
      "title": "Test fixture `minimal_lookup` defined in test_llm_judge_irt.py duplicates the shape of the larger `predictor` fixture in test_cold_start_lookup.py",
      "severity": "P3",
      "confidence": 60,
      "anchor_kind": "code",
      "applies_to": "applies_to_both",
      "files": [
        {
          "path": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/tests/test_models/test_llm_judge_irt.py",
          "lines": "27-37"
        },
        {
          "path": "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/tests/test_models/test_cold_start_lookup.py",
          "lines": "33-83"
        }
      ],
      "evidence": "`tests/test_models/test_llm_judge_irt.py:27-37` defines `minimal_lookup` — a 3-line `ColdStartLookupPredictor` constructed in-line with `subj={\"gpt-4\": 0.75, ...}`, `bench={\"mmlupro\": 0.50, ...}`, `global_mean=0.645`. `tests/test_models/test_cold_start_lookup.py:33-83` defines `lookup_tables` (dict) + `predictor` (ColdStartLookupPredictor) — a much larger fixture with 3 subjects, 3 benchmarks, and an `sbc / sb` table for level-1/2 lookups. The two fixtures cover different test concerns (`minimal_lookup` always falls through to level 3+; `predictor` exercises levels 1-2). They are not the same — but the naming convention is identical (both are local-to-module pytest fixtures returning `ColdStartLookupPredictor`), and a reader cross-referencing the two test files might wonder why `minimal_lookup` isn't a shared `conftest.py` fixture used by both.",
      "why_it_matters": "Fixture duplication that isn't documented invites two failure modes: (a) future contributors editing one fixture without realizing the other exists, drifting the test corpus; (b) future contributors moving fixtures to conftest.py and accidentally changing the contract of one (e.g. adding an sbc entry to `minimal_lookup` so it no longer falls through to level 3). The current state is fine — the two fixtures are deliberately different — but the difference would be more durable if it were named.",
      "suggested_fix": "Rename `minimal_lookup` in test_llm_judge_irt.py:27-37 to `lookup_falls_through_to_level3` (or `lookup_no_sbc_sb`) to make the load-bearing property explicit. Either keep as-is OR add a one-line docstring change to the existing fixture: `\"\"\"A tiny lookup that ALWAYS falls through to a level-3 IRT blend.\\n\\nDeliberately different from tests/test_models/test_cold_start_lookup.py::predictor,\\nwhich exercises levels 1-2 via populated sbc/sb tables. Tests in this\\nfile target the IRT-blend interaction with the judge_fn delta.\\\"\"\"`. The cross-reference makes the fixture-choice intentional and durable across future test refactors.",
      "applies_to_class": "applies_to_both",
      "rationale": "Tests in tests/test_models/ are upstream-eligible (they ship with the upstream model files). The cross-reference comment is upstream-friendly; the rename is style preference and could go either way."
    }
  ],
  "residual_risks": [
    {
      "id": "RES-MAINT-1",
      "summary": "ColdStartLookupPredictor's dict-shaped predict() is the cleanest API for the cold-start setting (text records, not integer indices), and may justifiably violate the Predictor contract — but the public-API consistency tradeoff is a maintainer judgment call. CRIT-MAINT-3 lays out two refactor paths; either is defensible. The reviewer's lane (maintainability) flags it; the synthesis-stage maintainer decides whether to accept the inconsistency or refactor.",
      "files": [
        "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/src/torch_measure/models/cold_start_lookup.py",
        "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/src/torch_measure/models/_predictor.py"
      ]
    },
    {
      "id": "RES-MAINT-2",
      "summary": "submission/train.py imports `CAIMIRALite` AT THE BOTTOM of main() for round-trip verification (train.py:390 `from caimira_lite import CAIMIRALite  # noqa: E402,WPS433`). This is a deliberate function-local import — the docstring at train.py:387-389 says the round-trip catches state_dict-shape mismatches between trainer and submission-runtime classes. The function-local import is OK (it sidesteps a circular-import concern in some test orderings), but a reader might mistake it for missed top-level import organization. Adding a comment that explicitly cites the test-ordering rationale (or moving the import to the top with a `# noqa` rationale) would harden the maintainability story.",
      "files": [
        "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/submission/train.py"
      ]
    },
    {
      "id": "RES-MAINT-3",
      "summary": "The submission/README.md `D-9 gate run history` table is hand-maintained markdown. A future agent regenerating the table from a JSON log (e.g. `submission/d9_history.jsonl`) would harden the provenance story. Currently the README is the single source of truth for the gate-run history; the JSON output at `/tmp/d9_caimira_smoke/gate_result.json` referenced at README:146 is in /tmp/ and will not survive a reboot.",
      "files": [
        "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/submission/README.md"
      ]
    }
  ],
  "testing_gaps": [
    {
      "id": "GAP-MAINT-1",
      "summary": "No mechanical parity test between submission/caimira_lite.py helpers and src/torch_measure/models/cold_start_lookup.py helpers (CRIT-MAINT-1). The `TestCAIMIRALiteStateDictParity` covers the state_dict layout but NOT the six character-identical top-level symbols (_logit, _sigmoid, _clip, _DEFAULT_PROVIDER_PREFIXES, parse_subject_name, resolve_subject_name). See CRIT-MAINT-1 for the proposed test shape.",
      "files": [
        "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/tests/test_submission/test_caimira_lite.py"
      ]
    },
    {
      "id": "GAP-MAINT-2",
      "summary": "No test verifies that `CAIMIRA.compute_item_params(center=\"INVALID\")` raises ValueError with the documented message (caimira.py:211). The runtime guard exists but isn't exercised; coverage would catch a future refactor that swallows or changes the error.",
      "files": [
        "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/tests/test_models/test_caimira.py"
      ]
    },
    {
      "id": "GAP-MAINT-3",
      "summary": "No test verifies that submission/model.py's `_blend_logits(caimira_p=1.0, eb_p=0.0, lambda_=1.0)` and `_blend_logits(0.0, 1.0, 0.0)` recover the unblended endpoints — the documented `lambda_=1` and `lambda_=0` edge cases at model.py:116. Coverage would catch a sign or weighting regression on the blend during a future tuning refactor.",
      "files": [
        "/Users/ankit.aggarwal/Dropbox/Stanford/CS321M/predictive-eval-competition/vasundras_torch_measure/tests/test_submission/"
      ]
    }
  ]
}
```

## Findings notes

### Severity distribution

- **P1 (2):** CRIT-MAINT-1 (helper-duplication without parity test, applies-to-both) and CRIT-MAINT-3 (ColdStartLookupPredictor breaks the Predictor contract, upstream-eligible).
- **P2 (3):** CRIT-MAINT-2 (triplicated _logit/_sigmoid in model.py, fork-only), CRIT-MAINT-4 (LLMJudgeIRT exported as public API despite NOT-FOR-PRODUCTION, upstream-eligible), CRIT-MAINT-5 (magic constants without provenance in submission/train.py, fork-only).
- **P3 (5):** CRIT-MAINT-6 (redundant `_BLEND_LAMBDA` passing, fork-only), CRIT-MAINT-7 (`_item_cache` unboundedness comment, fork-only), CRIT-MAINT-8 (Literal[] annotation for `center=` parameter, upstream-eligible), CRIT-MAINT-9 (submission/ directory placement, fork-only — flagged as already-OK), CRIT-MAINT-10 (test-fixture naming, applies-to-both).

### Fork-vs-upstream classification

- **upstream_eligible (3):** CRIT-MAINT-3, CRIT-MAINT-4, CRIT-MAINT-8 — all touch the 3 new model files (caimira.py / cold_start_lookup.py / llm_judge_irt.py) or `__init__.py`.
- **fork_only_caimira_branch (5):** CRIT-MAINT-2, CRIT-MAINT-5, CRIT-MAINT-6, CRIT-MAINT-7, CRIT-MAINT-9 — all touch only `submission/`.
- **applies_to_both (2):** CRIT-MAINT-1 (the duplication LIVES in the fork but the upstream side of the parity test is `src/torch_measure/models/cold_start_lookup.py`), CRIT-MAINT-10 (test fixtures across both upstream-eligible and fork-only test files).

### Charter-fit summary

- **Premature abstraction:** None flagged — the abstractions in this diff (`Predictor` base, `CAIMIRALite` standalone, `EBLookup`) all have at least one concrete implementor or load-bearing reason. The `Predictor` base now has 14+ implementors (Rasch, TwoPL, ThreePL, BetaRasch, BetaTwoPL, AmortizedIRT, CAIMIRA, MultiFacetRasch, MultiFacet2PL, TestletRasch, LogisticFM, Bifactor, NCF, TabPFNPredictor, ColdStartLookupPredictor, LLMJudgeIRT) so the abstraction earns its keep.
- **Unnecessary indirection:** CRIT-MAINT-2 (triple-copy of `_logit`/`_sigmoid`) and CRIT-MAINT-6 (`_BLEND_LAMBDA` three-way reference) are the closest matches. Neither rises to "layer of delegation"; both are pattern-noise that is cheap to clean up.
- **Dead or unreachable code:** None flagged. The `LLMJudgeIRT` class is documented-as-experimental but not dead (the tutorial notebook references it, and the tests in test_llm_judge_irt.py exercise it).
- **Coupling between unrelated modules:** CRIT-MAINT-1 (helper duplication WITHOUT parity test) is the closest match — the duplication isn't coupling per se, but the lack of a test means a future edit on one side silently changes behavior on the other. The fork-only `submission/` directory IS structurally decoupled from the upstream library (correctly per CRIT-MAINT-9); the failure mode is at the helper-symbol level, not the module level.
- **Naming that obscures intent:** CRIT-MAINT-8 (`center: str` should be `Literal[...]`) and CRIT-MAINT-10 (`minimal_lookup` fixture name doesn't say what's minimal about it). Neither rises to "obscures intent in a load-bearing way" but CRIT-MAINT-8 is a free type-system win.
