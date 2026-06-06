# Copyright (c) 2026 AIMS Foundations. MIT License.
"""Offline calibration for the Content-Aware Ensemble pipeline.

This module fits a two-step calibrator on a held-out ``val_calibration``
split (10% of val, never used for model selection) to correct systematic
over/underconfidence in the IRT pipeline's raw probabilities.

Even a model with good ranking signal (high AUC) can have poorly calibrated
probabilities. The log-loss metric severely punishes confident wrong
predictions — calibration is critical for leaderboard performance.

Two-step calibration
--------------------
**Step 1 — Temperature scaling (global, all benchmarks):**

.. code-block:: text

    logit   = norm_ability - norm_diff     # from IRT + difficulty regressor
    P_T     = sigmoid(logit / T)           # T > 1 = less confident
    T is fit by maximizing val_loglik on val_calibration

**Step 2 — Shrinkage (only for stability_flag=True benchmarks):**

.. code-block:: text

    P_final = (1 - alpha_b) * P_T + alpha_b * pass_rate_b
    alpha_b tuned per flagged benchmark on val_calibration rows for that benchmark

**Step 3 — Clip:**

.. code-block:: text

    P_final = clip(P_final, 0.001, 0.999)

Why this order: temperature fixes global calibration first (monotone
transform, preserves AUC). Shrinkage then fixes benchmark-specific
unreliability for unstable benchmarks.

Empirical results
-----------------
Fitted values from the CS321M competition run:

- Temperature T = 1.4002 (T > 1 → model was overconfident)
- Shrinkage: androidworld alpha=0.7, cybench alpha=0.7

Negative-result context
-----------------------
This module was part of a larger ensemble that showed real positive signal
(Spearman r=+0.139 on 970K val rows, p<1e-300) but did not improve the
final leaderboard score over the simpler
:class:`torch_measure.models.ColdStartLookupPredictor`.

See ``tutorials/content_aware_ensemble_experiment.ipynb`` for the full
experimental log.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIP_MIN          = 0.001
CLIP_MAX          = 0.999
ALPHA_CANDIDATES  = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
T_BOUNDS          = (0.1, 10.0)   # search range for temperature

# ---------------------------------------------------------------------------
# Compute logits for val_calibration
# ---------------------------------------------------------------------------

def compute_logits(
    df: pd.DataFrame,
    stage1a: dict,
    stage2a: dict,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute raw logits for each row in df.

    logit = norm_ability - norm_diff

    Where:
        norm_ability = benchmark-specific normalized ability from Stage 1A
                       (OOD proxy if benchmark unknown, global mean if subject unknown)
        norm_diff    = regressor.predict(encoder(item_text))

    Parameters
    ----------
    df : pd.DataFrame
        Val calibration rows.
    stage1a : dict
        Stage 1A output — abilities, pass rates, stability flags.
    stage2a : dict
        Stage 2A output — regressor, subject_lookup, ood_ability_lookup, global_mean.

    Returns
    -------
    np.ndarray of shape (len(df),) — raw logits
    """
    regressor          = stage2a["regressor"]
    subject_lookup     = stage2a["subject_lookup"]
    ood_ability_lookup = stage2a["ood_ability_lookup"]
    global_mean        = stage2a["global_mean"]
    encoder_name       = stage2a["encoder_model"]

    # Encode items
    print(f"\n[calib] Loading encoder: {encoder_name}")
    encoder = SentenceTransformer(encoder_name, device=device)

    df = df.copy().reset_index(drop=True)
    df["input_text"] = df.apply(
        lambda r: f"Benchmark: {r['benchmark']}\n{r['item_content']}", axis=1
    )

    unique_texts = df["input_text"].unique().tolist()
    print(f"[calib] Encoding {len(unique_texts):,} unique val_calibration items...")
    embeddings = encoder.encode(
        unique_texts,
        convert_to_numpy=True,
        batch_size=256,
        show_progress_bar=True,
        device=device,
    )
    text_to_idx    = {t: i for i, t in enumerate(unique_texts)}
    indices        = [text_to_idx[t] for t in df["input_text"]]
    X              = embeddings[indices]

    # Predict normalized difficulties
    pred_norm_diff = regressor.predict(X)

    # Ability lookup — three-level fallback
    n_l1, n_l2, n_l3 = 0, 0, 0
    norm_abilities = []
    for b, s in zip(df["benchmark"], df["subject_content"]):
        bench_dict = subject_lookup.get(b, {})
        if s in bench_dict:
            norm_abilities.append(bench_dict[s])
            n_l1 += 1
        elif s in ood_ability_lookup:
            norm_abilities.append(ood_ability_lookup[s])
            n_l2 += 1
        else:
            norm_abilities.append(global_mean)
            n_l3 += 1

    norm_abilities = np.array(norm_abilities)
    logits         = norm_abilities - pred_norm_diff

    print(f"[calib] Logit stats: mean={logits.mean():.4f}, std={logits.std():.4f}, "
          f"min={logits.min():.4f}, max={logits.max():.4f}")
    print(f"[calib] Fallback usage: L1={n_l1:,} L2={n_l2:,} L3={n_l3:,}")

    return logits


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def val_loglik(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean log-likelihood — higher is better."""
    probs     = np.clip(probs, CLIP_MIN, CLIP_MAX)
    log_likes = labels * np.log(probs) + (1 - labels) * np.log(1 - probs)
    return float(np.mean(log_likes))


def compute_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, probs))
    except Exception:
        return float("nan")


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    Bins predictions, measures gap between mean predicted prob and actual pass rate.
    Lower ECE = better calibration.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece       = 0.0
    n         = len(probs)

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc  = labels[mask].mean()
        ece     += (mask.sum() / n) * abs(bin_conf - bin_acc)

    return float(ece)


def compute_all_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    df: pd.DataFrame,
) -> dict:
    """Compute val_loglik, AUC, ECE overall and per benchmark."""
    overall = {
        "val_loglik": val_loglik(probs, labels),
        "auc":        compute_auc(probs, labels),
        "ece":        compute_ece(probs, labels),
    }

    per_benchmark = {}
    for b in sorted(df["benchmark"].unique()):
        mask = (df["benchmark"] == b).values
        if mask.sum() == 0:
            continue
        per_benchmark[b] = {
            "val_loglik": val_loglik(probs[mask], labels[mask]),
            "auc":        compute_auc(probs[mask], labels[mask]),
            "ece":        compute_ece(probs[mask], labels[mask]),
            "n_rows":     int(mask.sum()),
        }

    return {**overall, "per_benchmark": per_benchmark}


# ---------------------------------------------------------------------------
# Step 1 — Temperature scaling (global)
# ---------------------------------------------------------------------------

def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Fit global temperature T by maximizing val_loglik on val_calibration.

    Optimizes: max_T mean(log sigmoid(logit / T))
    Using scipy scalar minimizer on [-val_loglik].

    Returns
    -------
    float: optimal T
    """
    def neg_loglik(T):
        probs = sigmoid(logits / T)
        return -val_loglik(probs, labels)

    result = minimize_scalar(
        neg_loglik,
        bounds=T_BOUNDS,
        method="bounded",
        options={"xatol": 1e-6},
    )

    T_opt = float(result.x)
    print(f"\n[calib] Temperature scaling:")
    print(f"  T = {T_opt:.4f}")
    print(f"  T > 1 → predictions softened toward 0.5 (less confident)")
    print(f"  T < 1 → predictions sharpened away from 0.5 (more confident)")
    print(f"  val_loglik before: {val_loglik(sigmoid(logits), labels):.4f}")
    print(f"  val_loglik after:  {val_loglik(sigmoid(logits / T_opt), labels):.4f}")

    return T_opt


# ---------------------------------------------------------------------------
# Step 2 — Shrinkage (per flagged benchmark)
# ---------------------------------------------------------------------------

def fit_shrinkage(
    probs_T: np.ndarray,
    labels: np.ndarray,
    df: pd.DataFrame,
    stage1a: dict,
) -> dict:
    """
    Fit shrinkage alpha per stability_flag=True benchmark.

    For each flagged benchmark:
        P_final = (1 - alpha) * P_T + alpha * pass_rate
    where P_T = sigmoid(logit / T) from Step 1.

    Alpha is tuned on val_calibration rows for that benchmark only.
    Uses ALPHA_CANDIDATES grid search.

    Parameters
    ----------
    probs_T  : temperature-scaled probabilities for all val_calibration rows
    labels   : true labels
    df       : val_calibration dataframe
    stage1a  : Stage 1A output (pass_rate, stability_flag per benchmark)

    Returns
    -------
    dict: {benchmark: {"alpha": float, "pass_rate": float, "alpha_search": {...}}}
    """
    flagged    = {b: v for b, v in stage1a.items() if v["stability_flag"]}
    shrinkage  = {}

    if not flagged:
        print("\n[calib] No flagged benchmarks — skipping shrinkage step")
        return shrinkage

    print(f"\n[calib] Shrinkage tuning for {len(flagged)} flagged benchmark(s):")

    for benchmark, bench_data in flagged.items():
        pass_rate  = bench_data["pass_rate"]
        mask       = (df["benchmark"] == benchmark).values
        n_rows     = mask.sum()

        if n_rows == 0:
            print(f"  {benchmark}: no rows in val_calibration — skipping")
            continue

        bench_probs  = probs_T[mask]
        bench_labels = labels[mask]

        print(f"\n  {benchmark} (pass_rate={pass_rate:.3f}, n_rows={n_rows}):")

        alpha_logliks = []
        for alpha in ALPHA_CANDIDATES:
            p_shrunk = (1 - alpha) * bench_probs + alpha * pass_rate
            ll       = val_loglik(p_shrunk, bench_labels)
            alpha_logliks.append(ll)
            print(f"    alpha={alpha:.1f} → val_loglik={ll:.4f}")

        best_idx   = int(np.argmax(alpha_logliks))
        best_alpha = ALPHA_CANDIDATES[best_idx]
        print(f"  Best alpha: {best_alpha:.1f} (val_loglik={alpha_logliks[best_idx]:.4f})")

        shrinkage[benchmark] = {
            "alpha":      best_alpha,
            "pass_rate":  pass_rate,
            "alpha_search": {
                "candidates":  ALPHA_CANDIDATES,
                "val_logliks": alpha_logliks,
                "best_alpha":  best_alpha,
            },
        }

    return shrinkage


# ---------------------------------------------------------------------------
# Apply calibration
# ---------------------------------------------------------------------------

def apply_calibration(
    logits: np.ndarray,
    benchmarks: np.ndarray,
    T: float,
    shrinkage: dict,
) -> np.ndarray:
    """
    Apply full calibration pipeline to logits.

    Step 1: Temperature scaling (global)
        P_T = sigmoid(logit / T)

    Step 2: Shrinkage (flagged benchmarks only)
        P_final = (1 - alpha_b) * P_T + alpha_b * pass_rate_b

    Step 3: Clip to [CLIP_MIN, CLIP_MAX]

    Parameters
    ----------
    logits     : raw logits (norm_ability - norm_diff) per row
    benchmarks : benchmark name per row
    T          : fitted temperature
    shrinkage  : {benchmark: {"alpha": float, "pass_rate": float}}

    Returns
    -------
    np.ndarray of calibrated probabilities
    """
    # Step 1 — Temperature scaling
    probs = sigmoid(logits / T)

    # Step 2 — Shrinkage for flagged benchmarks
    probs = probs.copy()
    for i, b in enumerate(benchmarks):
        if b in shrinkage:
            alpha     = shrinkage[b]["alpha"]
            pass_rate = shrinkage[b]["pass_rate"]
            probs[i]  = (1 - alpha) * probs[i] + alpha * pass_rate

    # Step 3 — Clip
    probs = np.clip(probs, CLIP_MIN, CLIP_MAX)

    return probs


# ---------------------------------------------------------------------------
# fit_calibration()
# ---------------------------------------------------------------------------

def fit_calibration(
    data_dir: str,
    checkpoint_dir: str,
    device: str = "cpu",
) -> dict:
    """
    Fit full calibration pipeline on val_calibration.

    Pipeline:
    1. Load Stage 1A + Stage 2A artifacts
    2. Load val_calibration
    3. Compute logits for all val_calibration rows
    4. Compute baseline metrics (no calibration)
    5. Fit temperature T
    6. Fit shrinkage alphas for flagged benchmarks
    7. Compute final metrics (after both steps)

    Parameters
    ----------
    data_dir       : directory with val_calibration_with_ids.parquet
    checkpoint_dir : directory with Stage 1A + 2A pkl outputs

    Returns
    -------
    dict: calibration output (T, shrinkage, metrics). Caller handles saving.
    """
    import os
    import pickle
    import io
    import torch

    # ------------------------------------------------------------------
    # Load artifacts
    # ------------------------------------------------------------------
    print("[calib] Loading Stage 1A output...")
    with open(os.path.join(checkpoint_dir, "irt_stage1a_output.pkl"), "rb") as f:
        stage1a = pickle.load(f)

    print("[calib] Loading Stage 2A output...")

    class CpuUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu',
                                            weights_only=False)
            return super().find_class(module, name)

    with open(os.path.join(checkpoint_dir, "difficulty_reg_stage2a_output.pkl"), "rb") as f:
        stage2a = CpuUnpickler(f).load()

    print(f"[calib] Encoder: {stage2a['encoder_model']}")
    print(f"[calib] Regressor: {type(stage2a['regressor']).__name__}")

    flagged = [b for b, v in stage1a.items() if v["stability_flag"]]
    print(f"[calib] Flagged benchmarks (for shrinkage): {flagged}")

    # ------------------------------------------------------------------
    # Load val_calibration
    # ------------------------------------------------------------------
    val_file = "val_calibration_16_with_ids.parquet"
    path     = os.path.join(data_dir, val_file)
    df       = pd.read_parquet(path)
    print(f"[calib] val_calibration: {len(df):,} rows")

    required = ["benchmark", "subject_content", "item_content", "item_id", "label"]
    for col in required:
        assert col in df.columns, f"FAIL: '{col}' missing from val_calibration"
    print("[calib] ✅ Required columns present")

    print(f"[calib] Benchmark distribution:")
    for b, n in df.groupby("benchmark").size().items():
        print(f"  {b:<20} {n:>6,} rows")

    labels     = df["label"].values.astype(float)
    benchmarks = df["benchmark"].values

    # ------------------------------------------------------------------
    # Compute logits
    # ------------------------------------------------------------------
    logits = compute_logits(df, stage1a, stage2a, device=device)

    # ------------------------------------------------------------------
    # Baseline metrics (no calibration — T=1, no shrinkage)
    # ------------------------------------------------------------------
    probs_baseline = np.clip(sigmoid(logits), CLIP_MIN, CLIP_MAX)
    metrics_before = compute_all_metrics(probs_baseline, labels, df)

    print(f"\n[calib] Baseline metrics (no calibration):")
    print(f"  val_loglik: {metrics_before['val_loglik']:.4f}")
    print(f"  AUC:        {metrics_before['auc']:.4f}")
    print(f"  ECE:        {metrics_before['ece']:.4f}")

    # ------------------------------------------------------------------
    # Step 1 — Fit temperature T
    # ------------------------------------------------------------------
    T       = fit_temperature(logits, labels)
    probs_T = np.clip(sigmoid(logits / T), CLIP_MIN, CLIP_MAX)

    metrics_after_T = compute_all_metrics(probs_T, labels, df)
    print(f"\n[calib] After temperature scaling (T={T:.4f}):")
    print(f"  val_loglik: {metrics_after_T['val_loglik']:.4f} "
          f"(Δ={metrics_after_T['val_loglik'] - metrics_before['val_loglik']:+.4f})")
    print(f"  AUC:        {metrics_after_T['auc']:.4f} "
          f"(should be flat)")
    print(f"  ECE:        {metrics_after_T['ece']:.4f} "
          f"(Δ={metrics_after_T['ece'] - metrics_before['ece']:+.4f})")

    # Verify AUC stays flat
    auc_delta = abs(metrics_after_T["auc"] - metrics_before["auc"])
    if auc_delta > 0.005:
        print(f"  ⚠️  AUC changed by {auc_delta:.4f} after temperature scaling")
        print(f"     Temperature scaling should be monotone — investigate")
    else:
        print(f"  ✅ AUC stayed flat (Δ={auc_delta:.4f}) — temperature scaling is monotone")

    # ------------------------------------------------------------------
    # Step 2 — Fit shrinkage alphas
    # ------------------------------------------------------------------
    shrinkage = fit_shrinkage(probs_T, labels, df, stage1a)

    # Apply full calibration
    probs_final   = apply_calibration(logits, benchmarks, T, shrinkage)
    metrics_final = compute_all_metrics(probs_final, labels, df)

    print(f"\n[calib] After temperature + shrinkage:")
    print(f"  val_loglik: {metrics_final['val_loglik']:.4f} "
          f"(Δ={metrics_final['val_loglik'] - metrics_before['val_loglik']:+.4f} vs baseline)")
    print(f"  AUC:        {metrics_final['auc']:.4f}")
    print(f"  ECE:        {metrics_final['ece']:.4f} "
          f"(Δ={metrics_final['ece'] - metrics_before['ece']:+.4f} vs baseline)")

    # ------------------------------------------------------------------
    # Per-benchmark comparison
    # ------------------------------------------------------------------
    print(f"\n[calib] Per-benchmark val_loglik comparison:")
    print(f"{'Benchmark':<20} {'Before':>10} {'After T':>10} {'After T+S':>10} {'Flag'}")
    print("-" * 58)
    for b in sorted(df["benchmark"].unique()):
        before  = metrics_before["per_benchmark"].get(b, {}).get("val_loglik", float("nan"))
        after_t = metrics_after_T["per_benchmark"].get(b, {}).get("val_loglik", float("nan"))
        after_f = metrics_final["per_benchmark"].get(b, {}).get("val_loglik", float("nan"))
        flag    = " ⚠️" if stage1a.get(b, {}).get("stability_flag", False) else ""
        print(f"{b:<20} {before:>10.4f} {after_t:>10.4f} {after_f:>10.4f}{flag}")

    # ------------------------------------------------------------------
    # Build output (caller handles saving)
    # ------------------------------------------------------------------
    output = {
        # What gets baked into model.py
        "temperature": T,
        "shrinkage":   shrinkage,

        # Full metrics for analysis
        "metrics": {
            "before":                       metrics_before,
            "after_temperature":            metrics_after_T,
            "after_temperature_shrinkage":  metrics_final,
        },

        # Metadata
        "encoder_model":      stage2a["encoder_model"],
        "n_calibration_rows": len(df),
        "flagged_benchmarks": [
            b for b, v in stage1a.items() if v["stability_flag"]
        ],
    }

    return output
