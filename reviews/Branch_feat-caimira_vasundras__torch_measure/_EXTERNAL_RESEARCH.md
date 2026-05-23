# External research brief — `feat/caimira` review

> **Source:** `ce-best-practices-researcher` agent dispatched during primer assembly Step 7. Background-completed 2026-05-22.
> **Authority levels** are tagged inline: Official-docs / Academic-peer-reviewed / Academic-preprint / Community-canonical / Community-anecdotal.
> **This file is the canonical External Research input for Wave 1 reviewers and the C4 external_claims critic.** The primer's `## External research` section mirrors the key findings; this sidecar carries the full ≤2500-word digest.

---

## 1. Domain & purpose

The diff sits at the intersection of **PyTorch psychometrics** (the `aims-foundations/torch_measure` upstream is an MIT-licensed IRT/MIRT/CAT toolkit for AI evaluation, declared "Alpha" in `pyproject.toml:26`) and the **Stanford CS321M Predictive Evaluation Challenge** (Codabench competition 15934). The competition's hosted runtime imports `submission/model.py` once per container, then calls `predict(input, labeled) -> float` once per hidden item; in the platform's cold-start regime, AI subjects are known across train and test (909 subjects per `cold_start_lookup.py:11`) but items are NEW at test time, so integer item indices are unusable and the predictor must operate on `(benchmark, condition, subject_content, item_content)` text quadruples. The branch adds (a) a paper-faithful CAIMIRA implementation to the library's `models/` directory, (b) a 6-level hierarchical empirical-Bayes cold-start predictor, (c) an experimental LLM-judge variant marked as a documented negative result, and (d) a fully-scaffolded Codabench submission package that hybridizes CAIMIRA with the EB fallback at logit-space weight `lambda_=0.6`. Authority: **Community-canonical**.

## 2. Stack & key dependencies — Phase 1.5 deprecation check

| Dep | Decl. min | Latest (May 2026) | Deprecation status | Authority |
|---|---|---|---|---|
| `torch` | `>=2.0` | 2.12 (2026-05-13) | **NOT deprecated.** PyTorch 2.6 flipped `torch.load`'s `weights_only` default from `False` to `True` (backward-compatibility-breaking). The submission's `model.py:21` already names this transition and uses `weights_only=True`. | Official-docs (PyTorch 2.6 blog) |
| `sentence-transformers` | `>=5.4` | 5.5.1 (2026-05-20) | **NOT deprecated.** Submission `models.txt` pre-fetches `all-mpnet-base-v2`. | Official-docs (PyPI) |
| `huggingface_hub` | `>=0.16` | **1.16.1** (2026-05-21) | **NOT deprecated**, but `>=0.16` is now extremely permissive — current major is 1.x. Submission pins dataset revision (best practice). | Official-docs (PyPI) |
| `pyarrow` | `>=0.12` | 24.0.0 (2026-04-21) | **NOT deprecated**, but `>=0.12` is a 7-year-old floor — effectively meaningless as a constraint. | Official-docs (PyPI) |
| `pyro-ppl` | `>=1.8` | 1.9.1 (2024-06-02) | **NOT deprecated**, but most recent release is ~2 years old. | Official-docs (PyPI) |
| `scikit-learn` | `>=1.3` | 1.8.0 (2025-12-10) | **NOT sunset**, but **two relevant deprecations in 1.8**: (1) `LogisticRegression.penalty` is deprecated, to be removed in 1.10, replaced by `l1_ratio`; (2) `LogisticRegression.n_jobs` is deprecated, to be removed in 1.10. `CalibratedClassifierCV` is **not deprecated** and now supports `method="temperature"` and Array API. | Official-docs (1.8 changelog) |

**PyTorch 2.6+ `weights_only=True` verdict.** The submission's `submission/model.py:21` explicitly documents the new default and uses it. The pattern is correct; reviewer should still verify that **every** `torch.load` call site in the diff (not just `model.py`) passes `weights_only=True` and saves via `state_dict()`.

## 3. Idiomatic patterns load-bearing for this review

**a. Save `state_dict()`, not the full `nn.Module`.** Diff's `submission/train.py` uses `model_cpu.state_dict()`. **Authority: Official-docs.** **Reviewer question:** Does **every** save path in the diff use `state_dict()`?

**b. `register_buffer` for inference-time invariants.** `CAIMIRA.register_buffer("_difficulty_mean", ...)` at `caimira.py:136` is the canonical pattern. **Authority: Community-canonical.** **Reviewer question:** When `fit()` is called twice (warm-restart), does the snapshot at line 343 correctly overwrite the prior buffer? Is there a regression test for the persisted buffer round-tripping through `state_dict` save/load?

**c. HuggingFace dataset / hub commit-SHA pinning.** The diff pins `aims-foundations/measurement-db@589ccfdb…`. HF docs require the full 40-character hash. **Authority: Official-docs.** **Reviewer question:** Is `589ccfdb…` the full 40-character hash in every string literal in the diff?

**d. Empirical Bayes shrinkage for sparse-cell prediction.** `(n * raw_p + alpha * prior) / (n + alpha)` is the standard Beta-Binomial conjugate posterior mean. Canonical: Carlin & Louis 2008; Efron 2010. **Authority: Academic-peer-reviewed.** **Reviewer question:** Where is `alpha` set, and is the value (`_EB_SB_ALPHA = 5.0` in `submission/train.py:74`) justified empirically vs. picked by convention?

**e. SimHash (Charikar 2002) for stdlib-only diversity sampling.** 64-bit signatures + Hamming-distance is canonical (Manku-Jain-Sarma 2007 for Google web-page dedup). **Authority: Academic-peer-reviewed.** **Reviewer question:** Does the implementation match Charikar's 64-bit signature exactly, or does it pre-shingle to k-grams?

**f. Standalone vs. library-coupled submission packaging.** `submission/caimira_lite.py` deliberately re-implements the forward path. Standard Kaggle / Codabench / DrivenData pattern. **Authority: Community-canonical.** **Reviewer question:** Is `caimira_lite.py`'s forward path **numerically identical** to `src/torch_measure/models/caimira.py`'s forward path (same softmax axis, same dot-product order, same dtype promotion)? Reviewers should ask for a unit test that loads the trained `state_dict` through both implementations and asserts `max_abs_diff < 1e-6` on the same input batch.

## 4. Anti-patterns to flag

**a. Silent `try/except → 0.5` fallback in `predict()`.** Avoided in `submission/model.py` (D-3 discipline). Reviewer should **grep the entire `submission/` tree** for any `except Exception` that returns a fixed value in `[0,1]`. **Authority: Community-canonical** (the user's own `docs/solutions/runtime-errors/silent-ncf-head-load-failure...` and `labeling-py-nan-fallback-bomb...`).

**b. `torch.save(model, path)` instead of `torch.save(model.state_dict(), path)`.** Under PyTorch 2.6+ default `weights_only=True`, full-module pickles raise `UnpicklingError`. The diff's `train.py` looks correct; reviewers should verify there is **no** path where a non-`state_dict` object is saved. **Authority: Official-docs.**

**c. L2-regularizing only `alpha` (the judge slope) but NOT `beta` (the intercept) in `LLMJudgeIRT`.** `llm_judge_irt.py:163-205` regularizes only `alpha`. Comment at line 170 makes this intentional. This is the standard IRT-fitting convention. **HOWEVER**: if the LLM judge produces a systematically biased logit (always log-likes "yes" for short prompts), an unregularized `beta` can absorb that bias and still leave `alpha` near zero, **masking a real signal**. The negative-result narrative (small effect, |r|=0.257) is consistent with that failure mode. **Authority: Community-canonical.** **Reviewer question:** Was an L2-on-`beta` ablation tried? If not, does the documented convergent negative result actually rule out the LLM-judge approach, or only rule out the unregularized-`beta` parameterization?

**d. Subprocess output parsing in submission bash scripts.** The 2026-05-21 bash retry-loop gotcha (a `set -o pipefail` without `set -e` masks failure) is the canonical bash trap. **Authority: Community-canonical** (the user's own `bash-retry-loop-silent-success-pipefail-without-errexit-2026-05-21.md`). `build_zip.sh:26` uses `set -euo pipefail` — strict mode, good.

**e. HF dataset revision drift.** Reviewers should ask whether the `aims-foundations/measurement-db` ID appears anywhere in the diff **without** the revision pin (notebook cell, README snippet, unit-test fixture). Mixed-pinning is a silent reproducibility leak. **Authority: Official-docs.**

**f. Mixed probability-clipping bounds across layers.** `caimira_lite.py:86-92` declares `_CLIP_LO=0.05` / `_CLIP_HI=0.95` (the EB-table internal clip) AND `_PREDICT_LO=1e-4` / `_PREDICT_HI=1-1e-4` (the kit-boundary clip). **Reviewer question:** Why `0.05` and not `0.01` or `1e-3` for the EB table? Was the bound chosen empirically or by convention? Interaction with the D-9 calibration probes at `λ ∈ {0.3, 0.5, 0.7, 0.9}` merits explicit testing.

## 5. Academic / research literature for CAIMIRA paper-faithfulness

Verified against the CAIMIRA paper (Lalor et al., EMNLP 2024, arXiv:2410.06524).

**(a) `latent_dim` default.** Paper Section 4.3 says: *"We ablate the number of latent dimensions, m. Validation loss plateaus beyond m=5 (Fig 4). We thus train a 5-dimensional caimira model…"* The diff's `caimira.py:114` `latent_dim: int = 5` matches **the paper's reported experimental setting**, not a paper-declared "universal default." Diff's docstring (`caimira.py:60-64`) honestly distinguishes this. **Verdict: faithful, with appropriate hedge.**

**(b) Zero-centering: per-batch dynamic vs. frozen snapshot.** Paper Equation 7 is `d_j := d'_j - (1/n_q) Σ_j d'_j` — *"the mean difficulty across all n_q questions in the dataset"*. **This is the frozen-bank centering, NOT per-batch dynamic.** The diff's `caimira.py:69-90` docstring claims the paper is silent on dynamic-vs-frozen, and the implementation chooses **dynamic during training, frozen at inference**. **The paper appears to specify frozen-bank in Eq. 7**; the diff's "paper is silent" hedge is **arguably incorrect**. **Reviewer question (high-priority):** Confirm the paper's Eq. 7 against the source PDF, then decide whether the diff's training-time dynamic centering is a paper-faithful variation or a deviation. **Authority: Academic-preprint.**

**(c) L1 vs. L2 on subject skill.** Paper Eq. 10: `L_reg = λ_d Σ ||d_j||_1 + λ_s Σ ||s_i||_1`, hyperparameter `λ_s = 1e-5` (Section 4.3). **L1 confirmed.** Diff's `caimira.py:325` uses `self.skill.abs().mean()`. Diff defaults `skill_reg=1e-4` — **one order of magnitude larger than the paper's `1e-5`**. **Verdict: L1 form faithful; default coefficient is 10× the paper.** **Reviewer question:** Is `1e-4` from a sweep on the team's data, or copy-paste from the difficulty default? **Authority: Academic-preprint.**

**(d) Bias terms.** Paper Eq. 5: `r'_j := f_R(q_j) = W_R e_j + b_R` (relevance has bias `b_R`). Paper Eq. 6 (difficulty) has no bias, footnote 5: *"We skip the bias term for d'_j since it is mean-centered."* Diff matches exactly. **Verdict: faithful.**

**(e) Response equation.** Paper Eq. 3: `p(U_{i,j}=1) = σ((s_i - d_j)^T r_j)`. Diff `caimira.py:244-246` exactly implements this. **Verdict: faithful.**

**(f) Rasch model for the 1-PL IRT blend.** Diff's `cold_start_lookup.py:16-22` calls `sigmoid(logit(subj) + logit(bench) - logit(global))` *"the classical 1-parameter Rasch formulation… rewritten in logit space against the global mean as a reference."* The formula is more directly a **log-linear / Bradley-Terry-style** two-way logit additive model (Bradley & Terry 1952, *Biometrika*) than the literal Rasch (1960) `P = sigmoid(theta - delta)`. **Reviewer question:** Is the "rewritten Rasch" framing accurate, or is this more honestly Bradley-Terry-style? The math is the same; the citation choice has consequences. **Authority: Academic-peer-reviewed.**

**(g) SimHash.** Canonical: Charikar 2002 STOC; Manku-Jain-Sarma 2007 WWW. 64-bit + Hamming. **Authority: Academic-peer-reviewed.**

**(h) Platt 1999.** Standard form is **two-parameter** (slope `A`, intercept `B`): `P = sigmoid(A * f + B)`. Diff's per-benchmark **intercept-only** Platt at `cold_start_lookup.py:59-60` (cap `1.5`) is a **constrained** Platt that fits **only** the intercept. **Reviewer question:** Is the intercept-only choice justified by K=5 sample size? **Authority: Academic-peer-reviewed.**

## 6. Authority levels (consolidated)

- **Official-docs:** PyTorch 2.6 release blog (weights_only flip), PyTorch saving/loading tutorial, huggingface_hub Quickstart (revision pinning), HF datasets Loading guide, PyPI release histories for `sentence-transformers`/`huggingface_hub`/`pyarrow`/`pyro-ppl`/`scikit-learn`, scikit-learn 1.8 changelog.
- **Academic-peer-reviewed:** Rasch 1960; Bradley & Terry 1952; Charikar 2002 (STOC); Platt 1999; Manku, Jain, Sarma 2007 (WWW); Carlin & Louis 2008; Efron 2010; Embretson & Reise 2000.
- **Academic-preprint:** arXiv:2410.06524 (CAIMIRA, EMNLP 2024).
- **Community-canonical:** PyTorch `register_buffer` idiom; Codabench/Kaggle "vendor your model definition" convention; the user's own `docs/solutions/` patterns.
- **Community-anecdotal:** None used.

## 7. What was NOT looked up

(a) **`tabpfn>=2.2,<3`, `tueplots`/`seaborn`/`matplotlib`** — transitive upstream dependencies that the `feat/caimira` diff does not touch directly. Skipped to keep brief tight.

(b) **The `aims-foundations/measurement-db` HF dataset's schema evolution.** Live-state question, better answered by `hf_hub_download` from a session with network access. CS321M project's dynamic-source-pull pipeline is the canonical owner.

(c) **The full CAIMIRA reference implementation** (the Lalor lab's own released code, if any). I could not load `github.com/Lalor-Lab/CAIMIRA` (404) or the canonical aclanthology page (permission denied). The paper-faithfulness verdicts in §5 are grounded in the paper text, not the original codebase. **A reviewer with access to the Lalor lab's reference implementation should re-verify points (a)-(e) of §5 against that code.**
