# Primer — Branch feat/caimira (vasundras/torch_measure)

## Source

**Scope:** synthetic-pr (branch-vs-main local diff; no GitHub PR exists for this branch yet).

| Field | Value |
|---|---|
| Host | github.com |
| Repo | vasundras/torch_measure |
| Repo kind | **Fork** of `aims-foundations/torch_measure` (MIT licensed; "PyTorch-native toolkit for predictive evaluation of AI systems") |
| Default branch | main |
| Active branch | feat/caimira |
| Head commit | `4d7ef2f6c21c569d0e03dcecd21b83c3e5c6a408` |
| Commit range | `main..feat/caimira` (4 commits) |
| Diff stats | 19 files changed, 4523 insertions(+), 0 deletions(-) |
| Reviewer's gh account | `ankaggarwal94` (distinct from the fork owner `vasundras`) |
| Open PR | **None** for this branch as of 2026-05-22 |

**Reviewer-side context (from sibling `CS321M/predictive-eval-competition/CLAUDE.md`):**

- The fork owner `vasundras` is a teammate whose `torch_measure` repo has been outscoring the main campaign's submissions on the CS321M Predictive Evaluation Challenge hidden leaderboard (reference scores `-0.61 / 0.68` per the parent CLAUDE.md "Pre-submission transfer-audit pattern" section).
- The reviewer (`ankaggarwal94`) has been adding CAIMIRA + cold-start work and a Codabench submission scaffold ON this fork. The intent (per the four commit messages) is two-track: (a) contribute CAIMIRA + cold_start_lookup + llm_judge_irt back upstream to `aims-foundations/torch_measure`, and (b) ship a Codabench submission for CS321M competition 15934 from this scaffolding.
- Co-authored-by: Cursor (agent-pair-programming flow).

### Commit list (oldest first)

| SHA | One-line |
|---|---|
| `6b5eb31` | Add cold-start predictive evaluation models for CS321M challenge (existed before merge; `ColdStartLookupPredictor`, `LLMJudgeIRT`, tutorial notebook) — present on `main`'s descendants but absent from `aims-foundations/main`. Carried into `feat/caimira` through the merge. |
| `777a3b1` | Merge upstream/main into my-project — picks up the upstream `AmortizedIRT`, `TabPFNPredictor`, multifaceted 2PL refactor, etc. |
| `1e4ec1c` | Add CAIMIRA content-aware multidimensional IRT model |
| `1e889f2` | Add Codabench submission scaffolding (CAIMIRA + EB hybrid back-stop) |
| `89b1d61` | Gitignore submission/ trainer artifacts + Codabench ZIPs |
| `4d7ef2f` | Document D-9 smoke-gate run + coverage gap on sub-gate (a) |

Note: only the last 4 commits are unique to `feat/caimira` vs `main`. The first two are reachable via the merge commit and are NOT in `main..feat/caimira`, BUT the new files added by `6b5eb31` ARE in the diff because `main` (descendant of `aims-foundations/main`) does not contain them. This is documented faithfully in the diff `--name-status` output.

### Files added or modified

```
M  .gitignore                                                  +8  -0
M  src/torch_measure/models/__init__.py                        +7  -0
A  src/torch_measure/models/caimira.py                       +344  -0
A  src/torch_measure/models/cold_start_lookup.py             +445  -0
A  src/torch_measure/models/llm_judge_irt.py                 +221  -0
A  submission/README.md                                      +183  -0
A  submission/build_zip.sh                                   +139  -0
A  submission/caimira_lite.py                                +408  -0
A  submission/labeling.py                                    +222  -0
A  submission/model.py                                       +258  -0
A  submission/models.txt                                       +1  -0
A  submission/train.py                                       +458  -0
A  tests/test_models/test_caimira.py                         +177  -0
A  tests/test_models/test_cold_start_lookup.py               +355  -0
A  tests/test_models/test_llm_judge_irt.py                   +247  -0
A  tests/test_submission/__init__.py                           +0  -0
A  tests/test_submission/test_caimira_lite.py                +252  -0
A  tests/test_submission/test_model_contract.py              +176  -0
A  tutorials/predictive_evaluation_challenge.ipynb           +622  -0
                                                          ─────────────
                                                          +4523  -0
```

## Change

The full diff is persisted to `_FULL_DIFF.patch` (4656 lines). The compressed inline summary below covers the key surfaces; reviewers should `Read` the actual source files (paths above) for full content.

### Library additions (3 new top-level classes exported from `torch_measure.models`)

**`CAIMIRA` (`src/torch_measure/models/caimira.py`, 344 lines):**

```python
class CAIMIRA(IRTModel):
    def __init__(self, n_subjects, n_items, embedding_dim, latent_dim=5, device="cpu")
    def set_embeddings(self, embeddings: torch.Tensor) -> None
    def compute_item_params(self, embeddings=None, center="auto") -> tuple[Tensor, Tensor]
    def _refresh_difficulty_mean(self) -> None
    def predict(self, query: dict[str, torch.Tensor]) -> torch.Tensor
    def fit(self, data, embeddings, mask=None, max_epochs=1000, lr=1e-3, ...) -> dict
```

Key invariants per the docstring and the four commit messages:

1. **Bias asymmetry:** `relevance_head = nn.Linear(D, K, bias=True)`; `difficulty_head = nn.Linear(D, K, bias=False)`. Claimed paper-faithful per `arXiv:2410.06524` §3.3 footnote 5 + Appendix B.
2. **Zero-centering protocol:** paper is silent on dynamic vs frozen; this impl uses dynamic during `train()`, frozen-snapshot in `_difficulty_mean` buffer at `eval()` / after `fit()`. Documented explicitly.
3. **Response equation:** `P = sigmoid((s_i - d_j)^T r_j)`. Implemented row-wise in `predict()`.
4. **Loss:** Bernoulli NLL + L1 on centered difficulty + L1 on subject skill. Plumbed through `mle_fit`'s `loss_fn` hook.
5. **Default `latent_dim=5`** with explicit note: "the paper does not declare 5 as a universal default and tuning per task may help."

**`ColdStartLookupPredictor` (`src/torch_measure/models/cold_start_lookup.py`, 445 lines):**

```python
class ColdStartLookupPredictor:
    def __init__(self, sbc, sb, subj, bench, global_mean, name_aliases=None, name_lc=None, ...)
    @classmethod from_lookup_json(cls, path, **overrides)
    def to_lookup_json(self, path)
    def _lookup_p(self, subj_name, benchmark, condition) -> float   # 6-level hierarchy
    def predict(self, record, labeled=None) -> float
    def predict_batch(self, records, labeled=None) -> list[float]
    def calibrate(self, labeled) -> None                            # intercept-only Platt
```

Plus module-level helpers `parse_subject_name`, `resolve_subject_name`, `_logit`, `_sigmoid`, `_clip`. Carries the leaderboard score footnote in its module docstring: "Deployed as the final competition submission with leaderboard NLL = -0.59 (tied #6 of 14)."

**`LLMJudgeIRT` (`src/torch_measure/models/llm_judge_irt.py`, 221 lines):** experimental; marked NOT-FOR-PRODUCTION in module docstring with a "Negative-result disclosure" section. `predict()` returns `_sigmoid(theta - delta)` where `theta = logit(lookup.predict(record))` and `delta = alpha * judge_logit + beta`. `fit_alpha_beta` uses scipy.optimize.minimize on a Nelder-Mead simplex with L2 regularization on `alpha` only.

### Submission scaffolding (Codabench competition 15934)

**`submission/model.py` (258 lines):** kit `predict(input, labeled)` entry point. Hybrid CAIMIRA + EB:

- In-vocab subject (resolved name lands in `META["subject_to_idx"]`): encode item once via `all-mpnet-base-v2` cached at module-level, compute `caimira_logit`, `_blend_logits(caimira_p, eb_p, lambda_=0.6)`.
- Out-of-vocab subject OR missing item content: EB-only path.
- Per-benchmark intercept-only Platt shift from `labeled` (slope fixed `1.0`, shift cap `±1.5` logit).
- `PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1` env var bypasses heavy loads → returns fixed `0.5`.
- Final clip: `[1e-4, 1 - 1e-4]`.
- D-3 / no-broad-except discipline: documented; no `try/except → 0.5` blocks in the prediction path.
- Module init crashes loudly if `caimira_lite.pt` / `eb_tables.json` / `caimira_lite.meta.json` missing OR if `torch.load(..., weights_only=True)` fails — surfaces as platform `[PAIEC-PREDICT-002]`.

**`submission/caimira_lite.py` (408 lines):** standalone (does NOT import `torch_measure`). Mirrors the upstream `CAIMIRA` state_dict layout byte-for-byte:
- `CAIMIRALite` nn.Module with `skill`, `relevance_head.weight/bias`, `difficulty_head.weight` (no bias), `_difficulty_mean` buffer.
- `caimira_logit(subject_idx, item_embedding) -> float` — always frozen-mean centering.
- `EBLookup` class with the 6-level fallback hierarchy + intercept-only Platt + JSON persistence.
- Helpers: `parse_subject_name`, `resolve_subject_name`, `clip_for_predict`, `_logit`, `_sigmoid`, `_clip`.

**`submission/train.py` (458 lines):** offline trainer.
- Pulls registries (`subjects.parquet`, `items.parquet`, `benchmarks.parquet`) + response parquets from `aims-foundations/measurement-db` at pinned revision `589ccfdb8e82e6e0b5e35e9d23cd83a6df85018f`.
- Filters to binary `{0,1}` benchmarks (`is_binary = all(v in (0.0, 1.0) for v in labels)`).
- Encodes unique items via `sentence-transformers/all-mpnet-base-v2`.
- Builds long-form → wide-form (NaN-masked) tensor → fits `torch_measure.models.CAIMIRA`.
- Builds EB tables from the SAME binary training rows (Bayesian shrinkage at level 2 with α=5).
- Round-trips the saved state_dict through `caimira_lite.CAIMIRALite` before saving — verifies no missing/unexpected keys.
- Writes `caimira_lite.pt`, `caimira_lite.meta.json`, `eb_tables.json`.
- `--smoke` flag: single benchmark (mmlupro), 5 epochs, ~30s CPU.

**`submission/labeling.py` (222 lines):** `acquisition_function(input) -> float`. Stdlib-only.
- 64-bit SimHash signatures (`hashlib.blake2b`) over visible text fields.
- Reservoir-sampled 128-entry seen set; per-call cost bounded.
- Score = `diversity + metadata_bonus + tie_break`; clamped to `[0.0, 2.0]`.
- Bulletproof `except Exception → 0.0` (distinguishable sentinel outside legitimate domain `(0, 2]`).
- Verbatim port from parent `predictive-eval-competition/submission/labeling.py`.

**`submission/build_zip.sh` (139 lines):** bash strict-mode (`set -euo pipefail`). Explicit-allowlist packaging:
- Required files: `model.py`, `labeling.py`, `models.txt`, `caimira_lite.py`, `caimira_lite.pt`, `caimira_lite.meta.json`, `eb_tables.json`.
- Refuses to overwrite without `--force`. Refuses to build if any artifact missing.
- Single-`.pt` safety check.
- `zip -j` strips paths → flat ZIP at repo root.
- Post-build sanity: nested-path check + allowlist-vs-actual file list comparison.

**`submission/README.md` (183 lines):** operator-facing doc. Includes a "D-9 gate run history" table reporting the 2026-05-22 smoke run: OVERALL **FAIL**, with sub-gates (a) PASS / (b) PASS / (c) FAIL (stress_nll=0.759 > 0.50) / (d) PASS. The README explicitly disclaims that sub-gate (a) PASSed on a degenerate constant-`0.604` predictor and surfaces this as a documented coverage gap of the D-9 gate.

**`submission/models.txt`:** one line, `sentence-transformers/all-mpnet-base-v2`.

### Tests

| Test file | LoC | Classes | Notes |
|---|---|---|---|
| `tests/test_models/test_caimira.py` | 177 | `TestCAIMIRA` | 17 tests per commit message: init/shape; bias asymmetry; softmax simplex; difficulty zero-centered; frozen-mean buffer; cold-start embedding path; manual response-equation match; `fit` reduces loss; eval-mode frozen centering. |
| `tests/test_models/test_cold_start_lookup.py` | 355 | `TestLookupLevels`, `TestNameResolution`, `TestCalibration`, `TestIRTBlend`, `TestPersistence`, `TestNumericalSafety` | All 6 fallback levels independently. Platt cap enforcement. IRT-blend closed-form expectations. |
| `tests/test_models/test_llm_judge_irt.py` | 247 | `TestDifficultyPrompt`, `TestPredict`, `TestFitAlphaBeta` | Including alpha-zero / graceful-degradation contracts. |
| `tests/test_submission/test_caimira_lite.py` | 252 | `TestCAIMIRALiteStateDictParity`, `TestCAIMIRALiteForward`, `TestParseSubjectName`, `TestResolveSubjectName`, `TestEBLookup`, `TestEBLookupPlatt`, `TestClipForPredict` | Critical: state_dict parity between trainer-side `CAIMIRA` and ship-side `CAIMIRALite`. |
| `tests/test_submission/test_model_contract.py` | 176 | `TestModelContract`, `TestModelAcquisitionContract` | Loads `model.py` under `PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1`; signature + native-float-return + finite-[0,1] + empty-labeled + kwarg-labeled checks. |

### `.gitignore` additions (8 lines)

```
submission/caimira_lite.meta.json
submission/eb_tables.json
/submission_*.zip
```

(Plus comments.) `caimira_lite.pt` is already covered by the pre-existing `*.pt` global rule.

### `__init__.py` additions (7 lines)

```python
+from torch_measure.models.caimira import CAIMIRA
+from torch_measure.models.cold_start_lookup import ColdStartLookupPredictor
+from torch_measure.models.llm_judge_irt import LLMJudgeIRT, build_difficulty_prompt
```

Plus four new entries in `__all__`: `ColdStartLookupPredictor`, `LLMJudgeIRT`, `build_difficulty_prompt`, `CAIMIRA`.

### Tutorial notebook

`tutorials/predictive_evaluation_challenge.ipynb` (622 lines) — reproducibility notebook for the M1→M3→M4.5 iteration log. CPU-only. Includes synthetic-data simulation reproducing the LLMJudgeIRT convergent-negative-result pattern.

## Comments / Context

### Tracker context

No GitHub Issue is referenced in any of the 4 commit messages. The CS321M course context is documented in the sibling `predictive-eval-competition/CLAUDE.md` (D-7, D-8, D-9 decision records; the 2026-05-22 transfer postmortem cited as motivation for the hybrid back-stop).

### PR/MR comments

**N/A** — no PR exists for `feat/caimira` on `vasundras/torch_measure` (verified via `gh pr list --repo vasundras/torch_measure --state all --head feat/caimira` returning `[]`).

## Project standards

### README.md

`README.md` is minimal (29 lines): badges, one-paragraph blurb, install instructions, link to CONTRIBUTING + Discord. Establishes the upstream's intended scope: "PyTorch-native toolkit for predictive evaluation of AI systems. Benchmark scores increasingly gate deployment decisions but rarely predict how a model will behave in production. `torch_measure` treats evaluation itself as a predictive modeling problem: latent-variable models infer a system's capability directly from sparse benchmark observations and predict its performance on unseen tasks."

### CONTRIBUTING.md

Key rules (the upstream's, since this is a fork that may aim for upstream PRs):

- "The package is organized **flat-by-domain** under `src/torch_measure/` — each domain (`models/`, `metrics/`, `fitting/`, `data/`, `datasets/`, `cat/`, `viz/`) has a matching `tests/test_<domain>/` directory."
- **All new public APIs need:**
  - "A numpy-style docstring with a short usage example"
  - "A corresponding entry in the Sphinx docs under `docs/source/`" — **NOTE:** the diff adds 3 new public classes (`CAIMIRA`, `ColdStartLookupPredictor`, `LLMJudgeIRT`) + 1 helper (`build_difficulty_prompt`) but does NOT add any Sphinx doc files under `docs/source/`. This is a CONTRIBUTING.md violation if upstreaming.
  - "At least one test, with appropriate pytest markers (`slow`, `gpu`, `network`) for expensive cases."
- "**No committed data files** — data belongs in [measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db)." — the diff respects this; `.pt` / `.meta.json` / `eb_tables.json` are all gitignored.
- "**Commit messages are short, single-line, and describe the change**" — **NOTE:** the four commits on `feat/caimira` are MULTI-LINE (very long bodies). Upstream prefers single-line. This is a style mismatch for upstream PRs.
- "**We squash-merge PRs by default.**" — so single-line commits matter less if upstream squashes.

### pyproject.toml

- License MIT.
- `requires-python = ">=3.10"`.
- Deps: `torch>=2.0`, `numpy>=1.24`, `scipy>=1.10`, `scikit-learn>=1.3`, `pandas>=2.3`, `pyarrow>=0.12`, `pyro-ppl>=1.8`, `huggingface_hub>=0.16`, `matplotlib>=3.7`, `seaborn>=0.12`, `tueplots>=0.0.14`, `tabpfn>=2.2,<3`, `sentence-transformers>=5.4`.
- `[tool.ruff] line-length = 120`; `[tool.ruff.lint] select = ["E", "F", "W", "I", "UP", "B", "SIM"]`; ignore `E501`.
- `[tool.pytest.ini_options]` declares markers `slow`, `gpu`, `network` (CONTRIBUTING.md mandates these for expensive tests).
- `[tool.coverage.run] source = ["src/torch_measure"]`.

### .pre-commit-config.yaml

- pre-commit-hooks: `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-added-large-files --maxkb=500`.
- ruff: `ruff-check --fix --show-fixes`; `ruff-format`.

### .github/workflows

- `lint.yml` — `ruff check src/ tests/` + `ruff format --check src/ tests/`. **NOTE:** does NOT lint `submission/`. The submission scaffolding code passes ruff manually (per submission/README.md "Open follow-ups") but is invisible to CI.
- `test.yml` — pytest matrix on Python `3.10` / `3.11` / `3.12` with `-m "not slow and not gpu"`. Coverage uploaded for `3.11`. Network-marked tests are NOT skipped — they appear to run.
- `slow-tests.yml`, `publish-test.yml`, `publish.yml` — present but not material to this review.

### AGENTS.md / CLAUDE.md / GEMINI.md / .cursor/rules

**None present in this fork's working tree.** (The reviewer's parent `predictive-eval-competition` repo has CLAUDE.md; this fork does not.)

## External research

> **Full digest:** `_EXTERNAL_RESEARCH.md` in this directory. Summary below; substantive findings carried forward to Wave 1 + Wave 2.

### Phase 1.5 deprecation check (top-of-stack)

| Dep | Decl. min | Latest (2026-05) | Deprecation status |
|---|---|---|---|
| `torch` | `>=2.0` | 2.12 | **NOT deprecated.** 2.6 flipped `torch.load(weights_only=)` default to `True` (backward-incompatible). Diff's `submission/model.py:153` correctly uses `weights_only=True`. |
| `sentence-transformers` | `>=5.4` | 5.5.1 | NOT deprecated. |
| `huggingface_hub` | `>=0.16` | 1.16.1 | NOT deprecated; `>=0.16` is permissive but valid. Diff pins dataset revision (best practice). |
| `pyarrow` | `>=0.12` | 24.0.0 | NOT deprecated; floor is 7y stale. |
| `pyro-ppl` | `>=1.8` | 1.9.1 (2024) | NOT deprecated. |
| `scikit-learn` | `>=1.3` | 1.8.0 | NOT sunset; 1.8 deprecations of `LogisticRegression.penalty` and `.n_jobs` (removed in 1.10) — diff doesn't use these. |

### CAIMIRA paper-faithfulness — substantive findings

Verified against arXiv:2410.06524 (Lalor et al., EMNLP 2024). Three findings the Wave 1 reviewers and Wave 2 critics must verify:

1. **Zero-centering claim** — diff's docstring claims paper is silent on dynamic-vs-frozen; researcher's reading of paper Eq. 7 (`d_j := d'_j - (1/n_q) Σ_j d'_j`) suggests it specifies frozen-bank centering. The diff implements **dynamic at train time, frozen at inference**. May be a deviation from paper, not a "silence" — needs PDF spot-check. (Authority: **Academic-preprint** + reviewer-side PDF spot-check needed.)
2. **`skill_reg` default** — diff `caimira.py:325` defaults `skill_reg=1e-4`; paper Section 4.3 reports `λ_s = 1e-5`. **10× larger than paper.** May be a tuning artifact or copy-paste from difficulty default. (Authority: **Academic-preprint**.)
3. **Response equation + bias asymmetry + L1 form + `latent_dim=5` default** — all verified faithful to paper Eq. 3, Eq. 5, Eq. 6 footnote 5, Eq. 10, Section 4.3. (Authority: **Academic-preprint**.)

### Citation framing — non-blocking but worth surfacing

- **`cold_start_lookup.py:16-22` calls level-3 formula "the classical 1-parameter Rasch formulation… rewritten in logit space"** — the formula `sigmoid(logit(subj) + logit(bench) - logit(global))` is more directly a **Bradley-Terry-style** (Bradley & Terry 1952, *Biometrika*) two-way logit additive model than the literal Rasch (1960). Math is the same; citation choice has consequences. (Authority: **Academic-peer-reviewed**.)

### Idiomatic-pattern compliance — positive findings

The diff respects the documented idioms (Authority: **Community-canonical** / Official-docs):
- `state_dict()` saving (anti-pattern of full-module pickle correctly avoided)
- `register_buffer` for the `_difficulty_mean` zero-centering snapshot
- Pinned HF revision (40-char SHA `589ccfdb...`)
- Empirical-Bayes shrinkage formula at level 2 (matches Carlin & Louis 2008 conjugate-posterior form)
- SimHash 64-bit signatures + Hamming distance (Charikar 2002 canonical form)
- Standalone shippable code path (Kaggle/Codabench/DrivenData "vendor your model definition" convention)
- D-3 / no-broad-except discipline applied consistently in `submission/model.py` and `submission/labeling.py`

### Reviewer questions to surface

- Was an L2-on-`beta` ablation tried for `LLMJudgeIRT`? The negative-result narrative (|r|=0.257 small) could be consistent with the unregularized intercept absorbing systematic judge bias, masking a real `alpha` signal.
- Why `_CLIP_LO=0.05` / `_CLIP_HI=0.95` for EB tables vs `_PREDICT_LO=1e-4` / `_PREDICT_HI=1-1e-4` at the kit boundary? Empirical vs convention?
- Where is `_EB_SB_ALPHA = 5.0` (the EB prior strength) sourced from?
- Is `caimira_lite.py`'s forward path **numerically identical** to `src/torch_measure/models/caimira.py`'s forward path? (Test for `max_abs_diff < 1e-6` on same input batch.)
- Does **every** `torch.load` in the diff pass `weights_only=True`?

## Provenance

| Field | Value |
|---|---|
| Timestamp | 2026-05-22 (assembly) |
| Step 0 (tool check) | PASS — `gh` `/opt/homebrew/bin/gh`, `git` `/usr/bin/git` |
| Step 1 (gh auth) | PASS — `gh auth status --hostname github.com` confirms `Logged in to github.com account ankaggarwal94` with scopes `gist, read:org, repo, workflow` |
| Step 1 (repo reachability) | PASS — `gh api repos/vasundras/torch_measure` returns `{fork: true, parent: aims-foundations/torch_measure, default_branch: main, private: false, visibility: public}` |
| Step 2 (input parse) | `scope_mode=pr` (synthetic; no PR URL was passed — the user said "the changes in this codebase"; inferred as the `main..feat/caimira` diff since the working tree is on `feat/caimira` with 4 commits ahead and substantial uncommitted-feature shape; treated as PR-shaped review with local-git diff source). Conflict note: this is a SYNTHETIC pr-scope because no PR exists yet. `_DRAFT_REVIEW_COMMENTS.md` will explicitly tag posting target as "Local-only — no PR exists to comment on; if user wants to open a PR first, the comments are ready to attach." |
| Step 3 (artifact fetch) | `git diff main..feat/caimira` (local; no `gh pr diff` since no PR exists). Persisted to `_FULL_DIFF.patch` (4656 lines / 19 files / 4523 insertions). `git log main..feat/caimira` captured (4 commits + 2 prior commits visible via the merge but already on main). |
| Step 4 (issue / discussion) | None referenced in commit bodies. Not fetched. |
| Step 5 (PR comments) | N/A — no PR exists. |
| Step 6 (project standards) | README.md ✓, CONTRIBUTING.md ✓, pyproject.toml ✓, .pre-commit-config.yaml ✓, .github/workflows/*.yml ✓, Makefile ✗ (absent), AGENTS.md / CLAUDE.md / GEMINI.md / DECISIONS.md / ARCHITECTURE.md / .cursor/rules ✗ (all absent — this fork doesn't carry them). |
| Step 7 (external research) | DISPATCHED ce-best-practices-researcher in background (agent ID redacted) at 2026-05-22; result will be appended as `_EXTERNAL_RESEARCH.md`. Phase 1.5 deprecation check requested for `torch>=2.0`, `sentence-transformers>=5.4`, `huggingface_hub>=0.16`, `pyarrow>=0.12`, `pyro-ppl>=1.8`, `scikit-learn>=1.3`. |
| Step 8 (compression) | NOT applied. Diff is `large` (4523 added lines, between 2000 and 5000) but below the massive threshold (5000) and just above the optional-compression threshold (2000). Reviewers `Read` individual source files; full diff in `_FULL_DIFF.patch`. |
| Errored sources | None. |
| Skipped sources | (1) No PR comments (no PR exists). (2) No tracker (no Issue referenced). (3) No `gh pr diff` (no PR; local git diff used instead). |

### Reviewer-side guidance: this is a fork

The repo is a fork of `aims-foundations/torch_measure`. `_DRAFT_REVIEW_COMMENTS.md` MUST distinguish three classes of finding:

1. **Fork-only findings** (apply only to `feat/caimira` work; OK to post on the fork or leave local).
2. **Upstream-eligible findings** (apply to the new model files `caimira.py` / `cold_start_lookup.py` / `llm_judge_irt.py` which the reviewer may want to PR upstream to `aims-foundations/torch_measure` — must satisfy CONTRIBUTING.md including Sphinx docs).
3. **Submission-bound findings** (apply only to `submission/` which is fork-specific scaffolding for the CS321M competition — not upstream-eligible at all).

### Reviewer-side guidance: load-bearing decision history (parent CLAUDE.md)

The reviewer's parent `predictive-eval-competition/CLAUDE.md` documents prior decisions that constrain interpretation of THIS diff:

- **D-3 / no-broad-except discipline:** the diff respects this in `submission/model.py` and `submission/caimira_lite.py`. The ONE deliberate `except Exception → 0.0` in `submission/labeling.py` is the documented distinguishable-sentinel pattern (output outside legitimate `(0, 2]` domain).
- **D-7 / drop-nonbinary binarization:** `submission/train.py` filters to binary `{0, 1}` benchmarks via `all(v in (0.0, 1.0) for v in labels)`. Aligned with D-7.
- **D-9 / pre-submission stress-test gate:** the diff's last commit (`4d7ef2f`) documents that the D-9 gate has been smoke-run but NOT full-run; sub-gate (c) FAILed on the smoke artifact. The D-9 gate is a HARD PREREQUISITE for any Codabench upload per the parent CLAUDE.md "Pre-submission transfer-audit pattern" — but the diff's submission/ scaffolding is "DORMANT" (not yet uploaded), so this is documented gap, not active risk.
- **2026-05-22 transfer postmortem:** prior CAIMIRA m=5 submissions had **best local val_nll** AND **worst hidden score**. The hybrid CAIMIRA+EB back-stop with EB Platt calibration in THIS diff is a STRUCTURAL counter to the prior failure mode, but has NOT been hidden-leaderboard verified.

These three decisions are inherited context for the Wave 1 reviewers and Wave 2 critics. They are not findings — they are the policy frame.
