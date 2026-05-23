# Codabench submission scaffolding — CAIMIRA + EB hybrid

This directory is a **Codabench-ready submission entry point** for the
CS321M Predictive AI Evaluation Challenge (competition 15934). It wraps
[`torch_measure.models.CAIMIRA`](../src/torch_measure/models/caimira.py)
with an empirical-Bayes cold-start fallback ported from
[`torch_measure.models.ColdStartLookupPredictor`](../src/torch_measure/models/cold_start_lookup.py),
plus a stdlib-only SimHash adaptive-labeling acquisition function.

> **Status: scaffolding, not yet active.** The kit-contract entry points
> (`predict()` and `acquisition_function()`) are functional, but **no Codabench
> upload has been performed from this scaffolding**. Before any real
> submission, the D-9 pre-submission transfer-audit gate must pass — the
> 2026-05-22 transfer postmortem documents that prior CAIMIRA m=5 ZIPs had
> the best local val_nll AND the worst hidden score, so transfer safety
> requires explicit validation.

## Layout

```
submission/
├── README.md            # this file
├── model.py             # kit entry point: predict(input, labeled)
├── labeling.py          # kit entry point: acquisition_function(input)
├── models.txt           # HF repos to pre-fetch (one per line, cap=5)
├── caimira_lite.py      # standalone CAIMIRA forward + EB fallback (no torch_measure import)
├── train.py             # offline trainer; produces .pt + .meta.json + eb_tables.json
└── build_zip.sh         # explicit-allowlist packaging script

# Produced by train.py (gitignored — *.pt is in .gitignore line 40):
├── caimira_lite.pt          # CAIMIRA state_dict
├── caimira_lite.meta.json   # training provenance + subject_to_idx
└── eb_tables.json           # 6-level EB hierarchy + name lookup tables
```

## Prerequisites

This scaffolding assumes a parent CS321M directory layout (see the team's
operator-facing `CLAUDE.md` for the canonical reference). Required sibling
repositories alongside this fork:

* `../starting_kit/` — the Codabench-supplied starting kit. Steps 3-4 of the
  Run flow invoke `../starting_kit/tools/check_submission_zip.py` and
  `../starting_kit/tools/run_smoke_test.py` from that directory.
* `../predictive-eval-competition/` — the team's CS321M working repo. Step
  5 of the Run flow invokes `../predictive-eval-competition/scripts/run_d9_gate.py`
  to run the pre-submission transfer-audit gate.

Without these siblings present, steps 3-5 of the Run flow are NOT executable
from this fork alone — they are documented here as part of the canonical
operator flow, not as self-contained reproducibles.

## How it differs from the team's reference flow

The `predictive-eval-competition` repo at
`predictive-eval-competition/submission/` ships an NCF head as the active
predictor and has earlier shipped CAIMIRA variants ephemerally from
`tmp/tomorrow_submit/`. This directory is the **package-side equivalent**
that lives inside the `torch_measure` repo, so the upstream library can
be the source of CAIMIRA training as well as the submission framework
in one place.

Differences from the 2026-05-22 ephemeral CAIMIRA submissions
documented in the transfer postmortem:

| Concern | 2026-05-22 m=5 submissions | This scaffolding |
|---|---|---|
| Cold-start for unseen `subject` / `benchmark` / `condition` | `<UNK>` row at index 0 (never gradient-updated) | 6-level EB hierarchy + per-bench intercept-only Platt |
| Use of `labeled` reveals at predict time | Discarded (`del labeled`) | Per-benchmark Platt shift (slope=1, ±1.5 cap) |
| In-vocab prediction | CAIMIRA only | CAIMIRA blended with EB (logit-space, λ=0.6) |
| Failure mode | Silent 0.5 fallback on non-finite logit | Distinguishable EB fallback (varies with input) |
| Subject-name resolution | Bare string match → UNK on provider-prefixed names | Strip provider prefix + case-insensitive fallback |

## Run flow

```bash
# 0. From the repo root, activate the dev environment.
source .venv/bin/activate          # or: uv pip install -e .

# 1. Train CAIMIRA + build EB tables. Produces caimira_lite.pt,
#    caimira_lite.meta.json, eb_tables.json in this directory.
#
#    Smoke (single benchmark, ~30s on CPU):
python submission/train.py --smoke
#    Full run (all binary benchmarks, default 200 epochs, ~hours on CPU):
python submission/train.py

# 1b. (Alternate) Remote training via Modal for GPU acceleration.
#     See the "Remote training via Modal" section below for setup.
#     Smoke (plumbing check, ~5 epochs):
modal run modal_train.py::main --kind smoke --latent-dim 5 --seed 42 --epochs 5
#     Full training (CAIMIRA m=5, 100 epochs, --skip-pull to keep artifacts
#     remote-only):
modal run modal_train.py::main --kind train --latent-dim 5 --seed 7 --epochs 100

# 2. Package a flat ZIP (writes submission_caimira.zip at the repo root).
bash submission/build_zip.sh

# 3. Validate the ZIP with the kit's pre-validator (lives in the
#    parent CS321M repo's starting_kit/).
python ../starting_kit/tools/check_submission_zip.py submission_caimira.zip

# 4. (Optional) Run the kit's local smoke test against this directory.
PYTHONPATH=submission \
PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1 \
python ../starting_kit/tools/run_smoke_test.py submission/

# 5. Run the D-9 pre-submission transfer-audit gate before upload.
#    (Lives in the parent repo; runs the four-sub-gate audit on a held-out
#    benchmark before consuming Codabench quota. The signature requires
#    three named arguments per the script's argparse block — see
#    `../predictive-eval-competition/scripts/run_d9_gate.py:64-66`.)
python ../predictive-eval-competition/scripts/run_d9_gate.py \
    --candidate submission_caimira.zip \
    --name caimira \
    --out runs/d9/caimira_gate_result.json

# 6. Upload to Codabench competition 15934 only after D-9 passes.
```

## Remote training via Modal

`modal_train.py` (at the repo root) is an artifact-only Modal wrapper around
`submission/train.py`. It produces `caimira_lite.pt`, `caimira_lite.meta.json`,
and `eb_tables.json` under `runs/modal/<run_id>/` (mirrored from a Modal
Volume named `torch-measure-caimira-artifacts`). The wrapper REFUSES to start
if any `*codabench*` env vars are present in the local environment — this
lane is strictly artifact production, not submission.

### Modal setup

```bash
# One-time, on either Device 1 (Mac) or Device 2 (Win11 + RTX 5090).
pip install modal
modal setup                    # or: modal token new
```

The wrapper requests an H100 / A100-80GB / L40S GPU (in that fallback order)
and persists artifacts in the Modal Volume `torch-measure-caimira-artifacts`,
which is mirrored locally on `--skip-pull=false` (the default).

### Usage

```bash
# Plumbing-only smoke check (5-epoch single-benchmark train + artifact pull):
modal run modal_train.py::main --kind smoke --latent-dim 5 --seed 42 --epochs 5

# Full training run with explicit Platt blend lambdas for D-9 ablation:
modal run modal_train.py::main --kind train --latent-dim 5 --seed 7 \
    --epochs 100 --blend-lambdas 0.3,0.5,0.6,0.7,0.9

# Dry-run plan emission (prints JSON, performs no Modal call):
python modal_train.py --kind train --latent-dim 5 --seed 7 --epochs 100 --dry-run
```

Pulled artifacts land at `runs/modal/<run_id>/`. Pass `--skip-pull` to keep
the artifacts only on the Modal Volume (useful for very large multi-seed
sweeps where local disk pressure matters).

## Cold-start strategy

The kit's `predict(input, labeled)` is called once per hidden item. Each
`input` dict has four string fields: `benchmark` (identifier, not display
name), `condition`, `subject_content` (starts with `Name: <display>`),
`item_content`. CAIMIRA needs an integer `subject_idx` and an item
embedding — the former is the load-bearing constraint:

* **Subject-name parsing.** `parse_subject_name(subject_content)` strips
  the `Name:` prefix and returns the bare display name; that name is
  then `resolve_subject_name`-d through the EB tables' `name_aliases`
  + `name_lc` maps + a stripped provider-prefix candidate (e.g.
  `meta-llama/Llama-2-7b-chat` → `Llama-2-7b-chat`).
* **In-vocab subject** (resolved name lands in `META["subject_to_idx"]`):
  encode `item_content` via `all-mpnet-base-v2` (cached per round),
  compute CAIMIRA's logit via `CAIMIRALite.caimira_logit(subject_idx,
  item_embedding)`, blend with the EB logit in logit space at
  `λ=0.6`, then apply per-benchmark Platt shift from `labeled`.
* **Out-of-vocab subject** (no `subject_idx`): use EB only via the
  6-level hierarchy `sbc → sb → IRT-blend → bench → subj → global`,
  then Platt-shift if `labeled` covers the same benchmark.

The blend weight `λ=0.6` is the conservative default per the Wave-2
Lane-B recommendation. D-9 ablation candidates: `{0.3, 0.5, 0.7, 0.9}`.

## Why standalone (no `torch_measure` import in `model.py`)

The hosted Codabench container provides an organizer-supplied
`torch_measure` package that may **not** track this fork's `feat/caimira`
branch — so `from torch_measure.models import CAIMIRA` may fail (or load
a different class than the trainer wrote). The submission code therefore
ships a **standalone** `caimira_lite.py` that mirrors the upstream
`CAIMIRA` class's predict path and state_dict layout byte-for-byte. The
trainer in `train.py` DOES use `from torch_measure.models import CAIMIRA`
because it runs in the local venv where the package is installed; the
state_dict it produces loads cleanly into `caimira_lite.CAIMIRALite`
(verified by `train.py`'s round-trip step before saving).

## D-3 / no-broad-except discipline

This scaffolding has NO `try/except → 0.5` blocks anywhere in the
prediction path. The only `try/except` is in `labeling.py`'s
`acquisition_function`, which returns the distinguishable sentinel `0.0`
(outside the legitimate `(0, 2]` output domain). Module-init failures
fall into Codabench's GENERIC fallback tier ("No additional details are
safe to show") — loud (the submission errors) but not deeply diagnosed
by the platform. Only `predict()` output-shape violations (NaN/inf,
non-float, out-of-range) surface as the specific `[PAIEC-PREDICT-002]`
code. See
`../docs/solutions/design-patterns/codabench-two-tier-error-reporting-paiec-system-2026-05-19.md`
for the full two-tier taxonomy. The canonical bug pattern these rules
defend against is the April 2026 silent NCF load failure that produced
3 weeks of constant-0.5 leaderboard signal — see the design-pattern
docs in the parent CS321M repo's `docs/solutions/` directory.

## D-9 gate run history

| Date | Artifact | Overall | (a) dist | (b) runtime | (c) stress | (d) calib | Notes |
|---|---|---|---|---|---|---|---|
| 2026-05-22 | smoke train (1 subj × 905 items × 5 ep) | **FAIL** | PASS | PASS | FAIL (0.759 > 0.50) | PASS | Plumbing-only run — sub-gate (b) 176/176 rows green confirms scaffolding works end-to-end; sub-gates (a)/(d) PASS but were measured on a degenerate constant-0.604 predictor (the smoke model has only `deepseek-coder-v2` in `subject_to_idx` so every validation row falls to EB Level-4/6); sub-gate (c) FAIL is expected for a barely-trained model. The CAIMIRA logit path was NEVER exercised on the audit row sample — a real-trained artifact (~909 subjects) is required for a meaningful (a)/(c)/(d) verdict. JSON output: `/tmp/d9_caimira_smoke/gate_result.json` (zip_sha256 `8d0abdea1f93…`). |
| (pending) | full train (all binary benchmarks, ≥909 subjects) | — | — | — | — | — | Required before any Codabench upload per the postmortem-driven D-9 discipline. Will exercise the CAIMIRA+EB hybrid blend on the audit rows (in-vocab subjects). |

**Important finding from the 2026-05-22 smoke run:** the D-9 gate's sub-gate (a)
checks `mean(p) ∈ [pos_rate ± 0.05]` and `frac_extreme < 0.05` — it does NOT
penalize mid-range constancy. A constant predictor with the right mean PASSES
sub-gate (a). For this scaffolding, that means the smoke artifact's degenerate
output (literally `p = 0.604` for every row, `std=1.1e-16`) did not raise the
gate's (a) flag. This is a documented coverage gap of the gate per
`../docs/solutions/architecture-patterns/pre-submission-transfer-audit-stress-test-gate-2026-05-22.md`
("audit produces a TABLE, not a single pass/fail"), not a CAIMIRA-specific
bug — but readers of the gate's PASS verdict on (a) should cross-check the
`main_distribution` block (specifically `std` and the quantile spread) before
treating the verdict as evidence of a well-shaped distribution.

## Open follow-ups

* **D-9 transfer-audit gate has been smoke-run for plumbing validation
  but NOT full-validated for transfer correctness** (see the gate run
  history table above). The 2026-05-22 postmortem signature (best local
  val_nll, worst hidden score) means a CAIMIRA submission without a
  full-train D-9 PASS is a known quota-burning anti-pattern.
* **CI lint coverage.** The repo's `.github/workflows/lint.yml:30,33`
  only runs `ruff check src/ tests/` — this `submission/` directory is
  invisible to CI lint. A follow-up should either (a) extend the lint
  workflow or (b) document the contributor-side `ruff check submission/`
  step. The submission code currently passes ruff manually but won't
  get caught if it regresses.
* **Subject cold-start coverage.** If hidden-test subjects are drawn
  from the 909-subject CS321M catalog (which IS stable across train/
  test), the in-vocab path dominates. If the hidden set adds new
  subjects, the EB-only path takes over — and its calibration depends
  on training-corpus subject priors. The 2026-05-22 transfer regression
  hints that the in-vocab signal alone overfits; the hybrid blend is
  a structural counter but needs hidden-leaderboard verification.
* **Blend weight `λ`.** Currently fixed at 0.6 in `model.py:_BLEND_LAMBDA`.
  Could be made per-benchmark or uncertainty-weighted (`λ = 1 - 1/(1+n_obs)`).
  Defer until D-9 ablation results.
