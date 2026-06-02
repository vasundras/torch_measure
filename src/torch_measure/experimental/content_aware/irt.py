# Copyright (c) 2026 AIMS Foundations. MIT License.
"""Per-benchmark IRT fitting for the Content-Aware Ensemble pipeline.

This module is Stage 1 of the Content-Aware Ensemble approach explored in
the Stanford CS321M Predictive AI Evaluation Challenge. It fits Item
Response Theory (IRT) models independently on each benchmark's training
responses to extract two key quantities:

- **Subject ability** (theta): how capable a given AI subject is on this
  benchmark. Higher = stronger subject.
- **Item difficulty** (delta): how hard a given item is on this benchmark.
  Higher = harder item.

These fitted parameters feed directly into Stage 2
(:mod:`torch_measure.experimental.content_aware.difficulty_regressor`),
where a text encoder + regressor learns to predict item difficulty from
item content alone — enabling predictions on unseen items at test time.

Models supported
----------------
- **Rasch (1PL)**: one parameter per item (difficulty only).
- **2PL**: two parameters per item (difficulty + discrimination).

Both are fitted and logged. Stability metrics (ability standard errors,
observation density) are computed per benchmark to flag unreliable fits.
Flagged benchmarks receive shrinkage treatment in Stage 3
(:mod:`torch_measure.experimental.content_aware.calibration`).

Negative-result context
-----------------------
This module was part of a larger ensemble that showed real positive signal
(Spearman r=+0.139 on 970K val rows, p<1e-300) but did not improve the
final leaderboard score over the simpler
:class:`torch_measure.models.ColdStartLookupPredictor`. The IRT fitting
itself was stable — the difficulty regressor that consumes its output is
where generalization to the hidden test set broke down.

See ``tutorials/content_aware_ensemble_experiment.ipynb`` for the full
experimental log.

References
----------
- Rasch, G. (1960). Probabilistic models for some intelligence and
  attainment tests. Danish Institute for Educational Research.
- Lord, F. M. (1980). Applications of item response theory to practical
  testing problems. Erlbaum.
"""

import pandas as pd
import torch

from torch_measure.models import Rasch, TwoPL
from torch_measure.metrics import ability_standard_errors

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

IRT_MODELS = ["rasch", "twopl"]  # models to fit — both run in one job
MAX_EPOCHS = 1000                 # max MLE fitting epochs
LR         = 0.05                 # MLE learning rate
VERBOSE    = False                # suppress per-epoch output

# Stability warning thresholds — logging only, NOT hard cutoffs
# Hard fallback logic lives in calibration.py after analyzing stability table
SE_WARNING_THRESHOLD  = 0.5   # mean ability SE above this = unstable
DENSITY_WARNING       = 0.05  # density below this = very sparse, potentially unstable

# ---------------------------------------------------------------------------
# Device setup
# IRT fitting uses torch autograd but is not GPU-bound.
# CPU is appropriate — T4 GPU won't meaningfully speed this up.
# ---------------------------------------------------------------------------

def get_device() -> str:
    device = "cpu"
    print(f"[device] Using: {device}")
    return device

# ---------------------------------------------------------------------------
# Core: fit one IRT model on one benchmark
# ---------------------------------------------------------------------------

def fit_benchmark(
    df_bench: pd.DataFrame,
    benchmark: str,
    irt_model_name: str,
    device: str,
) -> dict:
    """
    Fit one IRT model on one benchmark's training responses.

    Uses long-form tensors directly — no wide-form matrix conversion.
    This keeps memory usage low and is consistent with our data pipeline.

    Stability metrics computed:
    - mean_ability_se : cheap tensor-only computation, no wide-form needed
    - density, n_subjects, n_items, n_observed : count-based, zero cost
    - final_loss : MLE convergence signal

    Parameters
    ----------
    df_bench : pd.DataFrame
        Rows for this benchmark only.
        Required columns: subject_id, item_id, label.
    benchmark : str
        Benchmark name (e.g. "mmlupro").
    irt_model_name : str
        "rasch" or "twopl".
    device : str
        Device for torch tensors.

    Returns
    -------
    dict with keys:
        abilities       : {subject_id: float}
        difficulties    : {item_id: float}
        discriminations : {item_id: float} (2PL only, else None)
        stability       : dict of stability metrics
        loss            : final MLE loss
    """
    print(f"\n[irt] Fitting {irt_model_name.upper()} on {benchmark}...")
    print(f"[irt]   Rows: {len(df_bench):,}")

    # Build integer index maps from IDs
    subject_ids = sorted(df_bench["subject_id"].unique().tolist())
    item_ids    = sorted(df_bench["item_id"].unique().tolist())
    n_subjects  = len(subject_ids)
    n_items     = len(item_ids)
    n_observed  = len(df_bench)

    print(f"[irt]   Subjects: {n_subjects}, Items: {n_items}")

    subj_to_idx = {s: i for i, s in enumerate(subject_ids)}
    item_to_idx = {it: i for i, it in enumerate(item_ids)}

    # Build long-form tensors — consistent with rest of pipeline
    subject_idx = torch.tensor(
        [subj_to_idx[s] for s in df_bench["subject_id"]], dtype=torch.long
    )
    item_idx = torch.tensor(
        [item_to_idx[it] for it in df_bench["item_id"]], dtype=torch.long
    )
    response = torch.tensor(
        df_bench["label"].values, dtype=torch.float32
    )

    # Initialize IRT model
    if irt_model_name == "rasch":
        model = Rasch(n_subjects=n_subjects, n_items=n_items, device=device)
    elif irt_model_name == "twopl":
        model = TwoPL(n_subjects=n_subjects, n_items=n_items, device=device)
    else:
        raise ValueError(f"Unknown IRT model: {irt_model_name!r}")

    # Fit via MLE on long-form tensors
    from torch_measure.fitting.mle import mle_fit
    history    = mle_fit(
        model,
        subject_idx=subject_idx,
        item_idx=item_idx,
        response=response,
        max_epochs=MAX_EPOCHS,
        lr=LR,
        verbose=VERBOSE,
    )
    final_loss = history["losses"][-1]
    print(f"[irt]   Final loss: {final_loss:.4f}")

    # Extract fitted parameters as plain tensors
    abilities_tensor       = model.ability.detach()
    difficulties_tensor    = model.difficulty.detach()
    discriminations_tensor = (
        model.discrimination.detach()
        if irt_model_name == "twopl" else None
    )

    # Build output dicts keyed by original IDs
    abilities = {
        sid: abilities_tensor[i].item()
        for sid, i in subj_to_idx.items()
    }
    difficulties = {
        iid: difficulties_tensor[i].item()
        for iid, i in item_to_idx.items()
    }
    discriminations = (
        {iid: discriminations_tensor[i].item() for iid, i in item_to_idx.items()}
        if discriminations_tensor is not None else None
    )

    # ------------------------------------------------------------------
    # Stability metrics
    # ------------------------------------------------------------------
    # We use only cheap metrics that don't require wide-form matrices:
    #   - ability_standard_errors: tensor-only, O(n_subjects x n_items) math
    #   - density, counts: zero extra compute
    #   - final_loss: already computed
    #
    # infit/outfit and difficulty SE skipped — they require building a
    # full (n_subjects x n_items) dense matrix which risks OOM at scale.
    # We will revisit if stability analysis shows we need them.
    # ------------------------------------------------------------------

    ability_se     = ability_standard_errors(
        ability=abilities_tensor,
        difficulty=difficulties_tensor,
        discrimination=discriminations_tensor,
    )
    mean_ability_se = ability_se.mean().item()

    density = n_observed / (n_subjects * n_items)

    # Stability warnings — logged, not enforced
    warnings = []
    if mean_ability_se > SE_WARNING_THRESHOLD:
        warnings.append(f"HIGH ability SE: {mean_ability_se:.3f}")
    if density < DENSITY_WARNING:
        warnings.append(f"LOW density: {density:.4f}")
    if n_items < 100:
        warnings.append(f"FEW items: {n_items}")

    stability = {
        "mean_ability_se": mean_ability_se,
        "density":         density,
        "n_subjects":      n_subjects,
        "n_items":         n_items,
        "n_observed":      n_observed,
        "final_loss":      final_loss,
        "warnings":        warnings,
    }

    if warnings:
        print(f"[irt]   ⚠️  Warnings: {warnings}")
    else:
        print(f"[irt]   ✅ Stable fit")

    print(f"[irt]   Mean ability SE: {mean_ability_se:.3f}")
    print(f"[irt]   Density:         {density:.4f}")

    return {
        "abilities":       abilities,
        "difficulties":    difficulties,
        "discriminations": discriminations,
        "stability":       stability,
        "loss":            final_loss,
    }

# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------

def train(
    device: str,
    data_dir: str,
    checkpoint_dir: str,
    train_file: str = "train_split_with_ids.parquet",
) -> dict:
    """
    Fit Rasch and 2PL models on all benchmarks in the training data.

    For each benchmark and each IRT model:
    - Fits on training responses (long-form, memory efficient)
    - Computes stability metrics
    - Logs stability table

    Parameters
    ----------
    device : str
        "cpu" — IRT fitting is CPU-bound
    data_dir : str
        Directory containing train_split_with_ids.parquet
    checkpoint_dir : str
        Included for interface compatibility; not used for saving in this
        library version (saving is handled by the caller).

    Returns
    -------
    dict with per-model stability tables and benchmark counts
    """
    df_train   = pd.read_parquet(f"{data_dir}/{train_file}")
    print(f"[data] Loaded {len(df_train):,} rows")

    benchmarks = sorted(df_train["benchmark"].unique().tolist())
    print(f"\n[irt] Benchmarks: {benchmarks}")

    all_results = {}

    for irt_model_name in IRT_MODELS:
        print(f"\n{'='*60}")
        print(f"[irt] Model: {irt_model_name.upper()}")
        print(f"{'='*60}")

        all_abilities       = {}
        all_difficulties    = {}
        all_discriminations = {}
        stability_table     = {}

        for benchmark in benchmarks:
            df_bench = df_train[
                df_train["benchmark"] == benchmark
            ].copy().reset_index(drop=True)

            result = fit_benchmark(
                df_bench=df_bench,
                benchmark=benchmark,
                irt_model_name=irt_model_name,
                device=device,
            )

            all_abilities[benchmark]    = result["abilities"]
            all_difficulties[benchmark] = result["difficulties"]
            stability_table[benchmark]  = result["stability"]

            if result["discriminations"] is not None:
                all_discriminations[benchmark] = result["discriminations"]

        # Print stability summary table
        print(f"\n[irt] {irt_model_name.upper()} Stability Summary:")
        print(f"{'Benchmark':<20} {'n_subj':>7} {'n_items':>8} "
              f"{'density':>8} {'ability_SE':>10} {'loss':>8} {'status'}")
        print("-" * 75)
        for bench, s in stability_table.items():
            status = "⚠️" if s["warnings"] else "✅"
            print(
                f"{bench:<20} {s['n_subjects']:>7} {s['n_items']:>8} "
                f"{s['density']:>8.4f} {s['mean_ability_se']:>10.3f} "
                f"{s['final_loss']:>8.4f} {status}"
            )

        all_results[irt_model_name] = {
            "stability_table": stability_table,
            "n_benchmarks":    len(benchmarks),
        }

    return all_results
