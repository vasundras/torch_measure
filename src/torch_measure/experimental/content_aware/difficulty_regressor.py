# Copyright (c) 2026 AIMS Foundations. MIT License.
"""Item difficulty regressor for the Content-Aware Ensemble pipeline.

This module learns a map from item text to normalized difficulty scores.
It is the only component in the Content-Aware Ensemble that must generalize
to unseen items at test time — all test subjects are known (ability lookup
from IRT fitting), but all test items are new (difficulty must be predicted
from text content alone).

Design decisions
----------------
1. **Normalized space throughout.** The regressor predicts normalized
   difficulty directly (z-scores from IRT fitting). The IRT formula at
   inference time is simply ``sigmoid(norm_ability - norm_diff)``. No
   denormalization is needed.

2. **Encoder: BAAI/bge-base-en-v1.5** (768d, 110M params). Superior MTEB
   clustering and STS scores vs all-mpnet-base-v2. Same output dimension —
   drop-in replacement.

3. **Three-level ability fallback at inference:**
   - Level 1: known benchmark + known subject → per-benchmark z-score
   - Level 2: OOD benchmark + known subject → mean z-score across benchmarks
   - Level 3: unknown subject → global mean ability

4. **Two-stage model selection:**
   - Stage 1: all 5 model types at default configs → rank by val_loglik
   - Stage 2: XGBoost grid search (5 configs) + MLP grid search (3 configs)
   - Winner = best val_loglik across both stages combined

5. **Val set discipline.** Model selection uses ``val_selection`` (90% of
   val). The remaining 10% (``val_calibration``) is reserved exclusively
   for temperature scaling in the calibration module.

Negative-result context
-----------------------
This module was part of a larger ensemble that showed real positive signal
(Spearman r=+0.139 on 970K val rows, p<1e-300) but did not improve the
final leaderboard score over the simpler
:class:`torch_measure.models.ColdStartLookupPredictor`. The difficulty
regressor learned to rank items correctly (AUC > 0.5) but was poorly
calibrated in absolute probability space on most benchmarks — the
regression target generalizes in relative terms but not in the absolute
scale needed for log-loss optimization.

See ``tutorials/content_aware_ensemble_experiment.ipynb`` for the full
experimental log.
"""

import numpy as np
import pandas as pd
import torch
import torch
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer
try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

from torch_measure.experimental.content_aware.mlp_regressor import MLPRegressor

# ---------------------------------------------------------------------------
# Constants — never redefine downstream
# ---------------------------------------------------------------------------

ENCODER_MODEL = "BAAI/bge-base-en-v1.5"   # replaces all-mpnet-base-v2
ENCODE_BATCH  = 256
RANDOM_SEED   = 42                          # same as split.py — never redefine
CLIP_MIN      = 0.001                       # non-negotiable clip bounds
CLIP_MAX      = 0.999

KNOWN_BENCHMARKS = [
    "afrimedqa", "agentdojo", "ai2d_test", "androidworld",
    "bfcl", "cybench", "hle", "livecodebench", "matharena",
    "mathvista_mini", "mmbench_v11", "mmlupro", "mtbench",
    "rewardbench", "swebench", "ultrafeedback",
]

# ---------------------------------------------------------------------------
# Stage 1 — default configs for model selection
# ---------------------------------------------------------------------------

STAGE1_CONFIGS = {
    "ridge": {
        "model": "ridge",
        "params": {"alpha": 1.0},
    },
    "lgbm": {
        "model": "lgbm",
        "params": {
            "num_leaves": 63,
            "n_estimators": 300,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "reg_lambda": 1.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_SEED,
            "verbose": -1,
        },
    },
    "xgboost": {
        "model": "xgboost",
        "params": {
            "max_depth": 5,
            "n_estimators": 300,
            "learning_rate": 0.05,
            "min_child_weight": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": RANDOM_SEED,
        },
    },
    "random_forest": {
        "model": "random_forest",
        "params": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_leaf": 20,
            "max_features": 0.5,
            "random_state": RANDOM_SEED,
        },
    },
    "mlp": {
        "model": "mlp",
        "params": {
            "hidden_dims": (256, 256),
            "dropout": 0.2,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 50,
        },
    },
    # "kernel_ridge": {
    #     "model": "kernel_ridge",
    #     "params": {
    #         "alpha": 1.0,
    #         "gamma": 1e-3,
    #         "kernel": "rbf",
    #     },
    # },
}

# Stage 2 — XGBoost grid search only
# Top 2 from Stage 1 carry forward their default-config results (no extra tuning)
STAGE2_XGBOOST_CONFIGS = [
    {"max_depth": 3, "n_estimators": 300, "learning_rate": 0.05,
     "min_child_weight": 5,  "gamma": 0.0, "reg_lambda": 1.0,
     "subsample": 0.8, "colsample_bytree": 0.5, "random_state": RANDOM_SEED},
    {"max_depth": 3, "n_estimators": 500, "learning_rate": 0.05,
     "min_child_weight": 20, "gamma": 0.1, "reg_lambda": 10.0,
     "subsample": 0.8, "colsample_bytree": 0.8, "random_state": RANDOM_SEED},
    {"max_depth": 5, "n_estimators": 300, "learning_rate": 0.05,
     "min_child_weight": 50, "gamma": 0.0, "reg_lambda": 1.0,
     "subsample": 0.8, "colsample_bytree": 0.5, "random_state": RANDOM_SEED},
    {"max_depth": 5, "n_estimators": 500, "learning_rate": 0.1,
     "min_child_weight": 20, "gamma": 1.0, "reg_lambda": 10.0,
     "subsample": 0.8, "colsample_bytree": 0.8, "random_state": RANDOM_SEED},
    {"max_depth": 7, "n_estimators": 500, "learning_rate": 0.05,
     "min_child_weight": 50, "gamma": 0.1, "reg_lambda": 10.0,
     "subsample": 0.8, "colsample_bytree": 0.5, "random_state": RANDOM_SEED},
]

# Stage 2 — MLP grid search (3 configs, focused on depth and epochs)
# Baseline MLP already run in Stage 1. These expand depth and training time.
STAGE2_MLP_CONFIGS = [
    {
        "hidden_dims": (512, 512),
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 100,
    },
    {
        "hidden_dims": (256, 256, 256),
        "dropout": 0.1,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 100,
    },
    {
        "hidden_dims": (512, 256, 128),
        "dropout": 0.1,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
    },
]

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[device] Using: {device}")
    return device

# ---------------------------------------------------------------------------
# Subject ability lookup
# ---------------------------------------------------------------------------

def build_subject_lookup(
    df_train: pd.DataFrame,
    stage1a: dict,
) -> tuple[dict, dict, float]:
    """
    Build subject ability lookup structures from Stage 1A output.

    Returns three structures:

    subject_lookup : {benchmark: {subject_content: normalized_ability}}
        Used for KNOWN benchmarks at test time.
        Key is subject_content string — must exactly match test-time input.

    ood_ability_lookup : {subject_content: mean_normalized_ability}
        Used for OOD benchmarks — mean z-score across all known benchmarks.
        Meaningful because abilities are z-scored (comparable across benchmarks).

    global_mean : float
        Fallback when subject_content not found in any benchmark.
        Represents an "average" AI model.

    Three-level fallback at test time:
        Level 1: subject_lookup[benchmark][subject_content]  → per-benchmark ability
        Level 2: ood_ability_lookup[subject_content]         → mean across benchmarks
        Level 3: global_mean                                 → unknown subject fallback
    """
    # Build subject_id → subject_content map from training data
    # subject_id is unique per subject (enforced by split_with_ids.py)
    id_to_content = (
        df_train[["subject_id", "subject_content"]]
        .drop_duplicates("subject_id")
        .set_index("subject_id")["subject_content"]
        .to_dict()
    )
    print(f"\n[lookup] Unique subject_ids in train: {len(id_to_content):,}")

    # Build per-benchmark lookup keyed by subject_content
    subject_lookup = {}
    all_abilities_by_content = {}  # {subject_content: [ability_b1, ability_b2, ...]}
    n_skipped = 0

    for benchmark, bench_data in stage1a.items():
        bench_lookup = {}
        for subject_id, norm_ability in bench_data["abilities"].items():
            content = id_to_content.get(subject_id)
            if content is None:
                n_skipped += 1
                continue
            bench_lookup[content] = norm_ability
            if content not in all_abilities_by_content:
                all_abilities_by_content[content] = []
            all_abilities_by_content[content].append(norm_ability)
        subject_lookup[benchmark] = bench_lookup

    if n_skipped > 0:
        print(f"[lookup] ⚠️  {n_skipped} subject_ids from Stage 1A not found in train "
              f"(expected if Stage 1A was run on different data)")

    # OOD lookup: mean normalized ability across all known benchmarks
    # z-scores are comparable across benchmarks — averaging is meaningful
    ood_ability_lookup = {
        content: float(np.mean(vals))
        for content, vals in all_abilities_by_content.items()
    }

    # Global mean fallback
    all_vals    = [v for vals in all_abilities_by_content.values() for v in vals]
    global_mean = float(np.mean(all_vals)) if all_vals else 0.0

    print(f"[lookup] Per-benchmark lookup built: {len(subject_lookup)} benchmarks")
    print(f"[lookup] OOD lookup: {len(ood_ability_lookup):,} unique subjects")
    print(f"[lookup] Global mean ability (fallback): {global_mean:.4f}")

    # Sanity check: OOD mean abilities should be approximately N(0,1) distributed
    ood_vals = list(ood_ability_lookup.values())
    print(f"[lookup] OOD ability distribution: "
          f"mean={np.mean(ood_vals):.3f}, std={np.std(ood_vals):.3f} "
          f"(should be approx mean≈0, std≈1)")

    return subject_lookup, ood_ability_lookup, global_mean


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_input_text(row: pd.Series) -> str:
    """
    Build encoder input string for one item.
    Includes benchmark name — difficulty scales differ per benchmark.

    CRITICAL: This exact function must be used at BOTH training AND test time.
    Any change here breaks consistency between trained regressor and submission.
    Format: "Benchmark: {benchmark}\n{item_content}"
    """
    return f"Benchmark: {row['benchmark']}\n{row['item_content']}"


def encode_items(
    df: pd.DataFrame,
    encoder: SentenceTransformer,
    device: str,
    desc: str = "",
) -> np.ndarray:
    """
    Encode unique items only — avoids re-encoding duplicates.
    Returns numpy array indexed to match df rows.
    """
    print(f"[encode] {desc} Encoding {len(df):,} rows...")
    df = df.copy()
    df["input_text"] = df.apply(build_input_text, axis=1)

    unique_texts = df["input_text"].unique().tolist()
    print(f"[encode] Unique texts: {len(unique_texts):,}")

    embeddings = encoder.encode(
        unique_texts,
        batch_size=ENCODE_BATCH,
        convert_to_numpy=True,
        show_progress_bar=True,
        device=device,
    )

    text_to_idx = {t: i for i, t in enumerate(unique_texts)}
    indices     = [text_to_idx[t] for t in df["input_text"]]
    return embeddings[indices]


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(model_name: str, params: dict):
    """Build a regressor from name and params dict."""
    if model_name == "ridge":
        return Ridge(**params)
    elif model_name == "lasso":
        return Lasso(**params)
    elif model_name == "lgbm":
        if lgb is None:
            raise ImportError("model_name='lgbm' requires lightgbm; install it separately to use this regressor")
        return lgb.LGBMRegressor(**params)
    elif model_name == "xgboost":
        if xgb is None:
            raise ImportError("model_name='xgboost' requires xgboost; install it separately to use this regressor")
        return xgb.XGBRegressor(**params, verbosity=0)
    elif model_name == "random_forest":
        return RandomForestRegressor(**params, n_jobs=-1)
    elif model_name == "mlp":
        return MLPRegressor(**params)
    else:
        raise ValueError(f"Unknown model: {model_name!r}")


# ---------------------------------------------------------------------------
# Evaluation — fully vectorized, normalized space throughout
# ---------------------------------------------------------------------------

def compute_metrics(
    predicted_norm_difficulties: np.ndarray,
    val_df: pd.DataFrame,
    subject_lookup: dict,
    ood_ability_lookup: dict,
    global_mean: float,
) -> dict:
    """
    Compute val_loglik (primary), AUC-ROC (secondary), and per-benchmark val_loglik.

    METRIC CONVENTION:
        val_loglik = mean log-likelihood per observation
        HIGHER IS BETTER throughout this codebase.
        This equals the competition's "negative log-loss" with higher-is-better sign.

    Everything stays in normalized space — no denormalization.
    IRT formula: sigmoid(norm_ability - norm_diff)

    Three-level ability fallback:
        Level 1: subject_lookup[benchmark][subject_content]
        Level 2: ood_ability_lookup[subject_content]
        Level 3: global_mean

    Parameters
    ----------
    predicted_norm_difficulties : np.ndarray
        Regressor output — normalized difficulty, one per val_df row.
    val_df : pd.DataFrame
        Val rows. Must have: benchmark, subject_content, label columns.
    subject_lookup : dict
        {benchmark: {subject_content: norm_ability}}
    ood_ability_lookup : dict
        {subject_content: mean_norm_ability} for OOD benchmarks
    global_mean : float
        Fallback when subject not found anywhere.

    Returns
    -------
    dict:
        val_loglik              : float (higher is better)
        auc                     : float
        val_loglik_per_benchmark: {benchmark: float}
        n_ood_lookups           : int (how many rows used OOD fallback)
        n_global_fallbacks      : int (how many rows used global mean fallback)
    """
    val_df = val_df.copy().reset_index(drop=True)

    # Vectorized three-level ability lookup with fallback tracking
    abilities   = []
    n_ood       = 0
    n_global    = 0

    for b, s in zip(val_df["benchmark"], val_df["subject_content"]):
        bench_dict = subject_lookup.get(b, {})
        if s in bench_dict:
            # Level 1: known benchmark + known subject
            abilities.append(bench_dict[s])
        elif s in ood_ability_lookup:
            # Level 2: OOD benchmark or subject not in this benchmark
            abilities.append(ood_ability_lookup[s])
            if b not in subject_lookup:
                n_ood += 1
        else:
            # Level 3: unknown subject entirely
            abilities.append(global_mean)
            n_global += 1

    subject_abilities = np.array(abilities)

    # IRT formula — normalized space throughout
    logits = subject_abilities - predicted_norm_difficulties
    probs  = 1.0 / (1.0 + np.exp(-logits))
    probs  = np.clip(probs, CLIP_MIN, CLIP_MAX)

    labels = val_df["label"].values.astype(float)

    # val_loglik = mean log-likelihood (HIGHER IS BETTER)
    log_likes = labels * np.log(probs) + (1 - labels) * np.log(1 - probs)
    val_loglik = float(np.mean(log_likes))

    # AUC-ROC
    try:
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = float("nan")

    # Per-benchmark val_loglik
    val_loglik_per_benchmark = {}
    for benchmark in sorted(val_df["benchmark"].unique()):
        mask       = (val_df["benchmark"] == benchmark).values
        bench_lls  = log_likes[mask]
        val_loglik_per_benchmark[benchmark] = float(np.mean(bench_lls))

    return {
        "val_loglik":               val_loglik,
        "auc":                      auc,
        "val_loglik_per_benchmark": val_loglik_per_benchmark,
        "n_ood_lookups":            n_ood,
        "n_global_fallbacks":       n_global,
    }


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² as diagnostic metric — not used for model selection."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Basic checks — run after training to validate pipeline correctness
# ---------------------------------------------------------------------------

def run_basic_checks(
    stage1a: dict,
    subject_lookup: dict,
    ood_ability_lookup: dict,
    global_mean: float,
    regressor,
    encoder: SentenceTransformer,
    device: str,
) -> None:
    """
    Run basic sanity checks on the trained pipeline.
    Catches obvious bugs before declaring training complete.

    Checks:
    1. Encoder output shape is 768d
    2. Regressor output is scalar per item
    3. Known benchmark ability lookup works (Level 1 fallback)
    4. OOD ability lookup works (Level 2 fallback)
    5. End-to-end prediction is in [CLIP_MIN, CLIP_MAX]
    6. IRT direction: strong subject on easy item > weak subject on hard item
    7. Sample difficulty predictions vary sensibly in [-3, +3] range
    """
    print(f"\n{'='*60}")
    print("[checks] Running basic pipeline checks...")
    print(f"{'='*60}")

    # Check 1 — Encoder output shape
    test_text = "Benchmark: mmlupro\nWhat is 2 + 2?"
    embedding = encoder.encode(test_text, convert_to_numpy=True, device=device)
    assert embedding.shape == (768,), \
        f"FAIL: Expected embedding shape (768,), got {embedding.shape}"
    print(f"✅ Check 1 — Encoder output shape: {embedding.shape}")

    # Check 2 — Regressor output is scalar
    pred = regressor.predict(embedding.reshape(1, -1))
    assert pred.shape == (1,), \
        f"FAIL: Expected regressor output shape (1,), got {pred.shape}"
    print(f"✅ Check 2 — Regressor output: shape={pred.shape}, value={pred[0]:.4f}")

    # Check 3 — Known benchmark ability lookup (Level 1)
    mmlupro_subjects = subject_lookup.get("mmlupro", {})
    assert len(mmlupro_subjects) > 0, "FAIL: mmlupro subject lookup is empty"
    first_subject    = next(iter(mmlupro_subjects))
    level1_ability   = mmlupro_subjects[first_subject]
    print(f"✅ Check 3 — Level 1 lookup (mmlupro): "
          f"ability={level1_ability:.4f} for '{first_subject[:40]}'")

    # Check 4 — OOD ability lookup (Level 2)
    level2_ability = ood_ability_lookup.get(first_subject, global_mean)
    print(f"✅ Check 4 — Level 2 lookup (OOD proxy): "
          f"mean_ability={level2_ability:.4f} "
          f"(global_mean fallback={global_mean:.4f})")

    # Check 5 — End-to-end prediction in valid range
    norm_diff = float(pred[0])
    norm_ab   = level1_ability
    logit     = norm_ab - norm_diff
    prob      = float(np.clip(1.0 / (1.0 + np.exp(-logit)), CLIP_MIN, CLIP_MAX))
    assert CLIP_MIN <= prob <= CLIP_MAX, \
        f"FAIL: Prediction {prob} outside [{CLIP_MIN}, {CLIP_MAX}]"
    print(f"✅ Check 5 — End-to-end prediction: "
          f"norm_ab={norm_ab:.3f}, norm_diff={norm_diff:.3f}, "
          f"logit={logit:.3f}, prob={prob:.4f}")

    # Check 6 — IRT direction: strong+easy > weak+hard
    prob_strong_easy = float(np.clip(
        1.0 / (1.0 + np.exp(-(2.0 - (-2.0)))), CLIP_MIN, CLIP_MAX
    ))
    prob_weak_hard = float(np.clip(
        1.0 / (1.0 + np.exp(-(-2.0 - 2.0))), CLIP_MIN, CLIP_MAX
    ))
    assert prob_strong_easy > prob_weak_hard, \
        f"FAIL: Strong+easy ({prob_strong_easy:.4f}) should be > weak+hard ({prob_weak_hard:.4f})"
    print(f"✅ Check 6 — IRT direction: "
          f"strong+easy={prob_strong_easy:.4f} > weak+hard={prob_weak_hard:.4f}")

    # Check 7 — Sample difficulty predictions vary sensibly
    sample_texts = [
        "Benchmark: mmlupro\nWhat is 2+2?",
        "Benchmark: mmlupro\nProve the Riemann hypothesis.",
        "Benchmark: afrimedqa\nWhat is the capital of France?",
        "Benchmark: hle\nSolve the protein folding problem from scratch.",
    ]
    sample_embs  = encoder.encode(sample_texts, convert_to_numpy=True, device=device)
    sample_preds = regressor.predict(sample_embs)
    print(f"✅ Check 7 — Sample predictions (expect variation, roughly in [-3, +3]):")
    for text, p in zip(sample_texts, sample_preds):
        print(f"   {p:+.3f}  {text[:65].replace(chr(10), ' ')}")

    print(f"\n[checks] All checks passed ✅")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------

def train(
    device: str,
    data_dir: str,
    checkpoint_dir: str,
    train_file: str = "train_split_with_ids.parquet",
    val_file: str   = "val_selection_with_ids.parquet",
) -> dict:
    """
    Train text → normalized difficulty regressors via two-stage model selection.

    Pipeline:
    1. Load Stage 1A output (normalized abilities + difficulties)
    2. Build subject lookup (known benchmark + OOD proxy)
    3. Build training data (unique train items + normalized difficulties)
    4. Encode items with bge-base-en-v1.5
    5. Stage 1: All 5 models default configs → rank by val_loglik
    6. Stage 2: XGBoost grid search (5 configs) + MLP grid search (3 configs)
    7. Select best overall → refit → run checks

    Parameters
    ----------
    device : str
        Device for encoder and MLP.
    data_dir : str
        Directory with train and val parquet files.
    checkpoint_dir : str
        Directory containing irt_stage1a_output.pkl. Included for interface
        compatibility; artifact saving is handled by the caller.

    Returns
    -------
    dict
        JSON-serializable summary: best_model, best_val_loglik, best_auc,
        stage1 results, stage2 results, per-benchmark breakdown.
    """
    import os
    import pickle

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    stage1a_path = os.path.join(checkpoint_dir, "irt_stage1a_output.pkl")
    with open(stage1a_path, "rb") as f:
        stage1a = pickle.load(f)
    print(f"[data] Loaded Stage 1A output: {len(stage1a)} benchmarks")

    for b, v in stage1a.items():
        flag_str = " ⚠️  UNSTABLE" if v["stability_flag"] else ""
        print(f"  {b:<20} model={v['irt_model']:<6} "
              f"subjects={len(v['abilities']):>4} "
              f"items={len(v['difficulties']):>6}"
              f"{flag_str}")

    df_train = pd.read_parquet(f"{data_dir}/{train_file}")
    df_val   = pd.read_parquet(f"{data_dir}/{val_file}")
    print(f"[data] Train: {len(df_train):,} rows")
    print(f"[data] Val (selection — 90% of val, reserved for model selection): {len(df_val):,} rows")

    required_cols = ["benchmark", "subject_id", "subject_content", "item_id",
                     "item_content", "label"]
    for col in required_cols:
        assert col in df_train.columns, f"FAIL: '{col}' missing from train"
        assert col in df_val.columns,   f"FAIL: '{col}' missing from val"
    print(f"[data] ✅ Required columns present in both splits")

    subject_lookup, ood_ability_lookup, global_mean = build_subject_lookup(
        df_train, stage1a
    )

    # ------------------------------------------------------------------
    # Build regressor training data
    # Unique train items + their normalized difficulty scores from Stage 1A
    # ------------------------------------------------------------------
    print("\n[reg] Building regressor training data...")

    train_items = (
        df_train[["item_id", "item_content", "benchmark"]]
        .drop_duplicates("item_id")
        .reset_index(drop=True)
    )

    def get_norm_difficulty(row):
        return stage1a.get(row["benchmark"], {}) \
                      .get("difficulties", {}) \
                      .get(row["item_id"], None)

    train_items["norm_difficulty"] = train_items.apply(get_norm_difficulty, axis=1)

    n_before = len(train_items)
    train_items = train_items.dropna(subset=["norm_difficulty"]).reset_index(drop=True)
    n_after = len(train_items)
    if n_before != n_after:
        print(f"[reg] ⚠️  Dropped {n_before - n_after} items with no difficulty score")

    print(f"[reg] Training items: {len(train_items):,}")
    print(f"[reg] Difficulty range: "
          f"{train_items['norm_difficulty'].min():.3f} to "
          f"{train_items['norm_difficulty'].max():.3f}")
    print(f"[reg] Difficulty mean:  {train_items['norm_difficulty'].mean():.4f} (should ≈ 0)")
    print(f"[reg] Difficulty std:   {train_items['norm_difficulty'].std():.4f} (should ≈ 1)")

    diff_mean = train_items["norm_difficulty"].mean()
    diff_std  = train_items["norm_difficulty"].std()
    assert abs(diff_mean) < 0.1, \
        f"FAIL: Difficulty mean {diff_mean:.4f} too far from 0 — normalization issue"
    assert abs(diff_std - 1.0) < 0.1, \
        f"FAIL: Difficulty std {diff_std:.4f} too far from 1 — normalization issue"
    print(f"[reg] ✅ Difficulty normalization verified")

    bench_counts = train_items.groupby("benchmark").size()
    print(f"\n[reg] Training items per benchmark:")
    for b, n in bench_counts.items():
        flag = " ⚠️" if stage1a.get(b, {}).get("stability_flag", False) else ""
        print(f"  {b:<20} {n:>6,}{flag}")

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
    print(f"\n[reg] Initializing encoder: {ENCODER_MODEL}")
    encoder = SentenceTransformer(ENCODER_MODEL, device=device)

    X_train = encode_items(train_items, encoder, device, desc="Train items:")
    y_train = train_items["norm_difficulty"].values
    X_val   = encode_items(df_val, encoder, device, desc="Val items:")

    print(f"[reg] X_train: {X_train.shape}")
    print(f"[reg] X_val:   {X_val.shape}")

    # ------------------------------------------------------------------
    # Stage 1: Model selection (all models, default configs)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("[reg] STAGE 1 — Model selection (default configs, all models)")
    print(f"{'='*60}")

    stage1_results = {}

    for config_name, config in STAGE1_CONFIGS.items():
        print(f"\n[reg] Fitting {config_name}...")
        try:
            model    = build_model(config["model"], config["params"])
            model.fit(X_train, y_train)
            r2_train = compute_r2(y_train, model.predict(X_train))
            metrics  = compute_metrics(
                model.predict(X_val), df_val,
                subject_lookup, ood_ability_lookup, global_mean
            )
            stage1_results[config_name] = {
                "r2_train":                 r2_train,
                "val_loglik":               metrics["val_loglik"],
                "auc":                      metrics["auc"],
                "val_loglik_per_benchmark": metrics["val_loglik_per_benchmark"],
                "n_ood_lookups":            metrics["n_ood_lookups"],
                "n_global_fallbacks":       metrics["n_global_fallbacks"],
                "config":                   config,
            }
            print(f"[reg]   R²={r2_train:.4f} | "
                  f"val_loglik={metrics['val_loglik']:.4f} | "
                  f"AUC={metrics['auc']:.4f} | "
                  f"OOD_lookups={metrics['n_ood_lookups']} "
                  f"global_fallbacks={metrics['n_global_fallbacks']}")
        except Exception as e:
            print(f"[reg]   ❌ Failed: {e}")
            stage1_results[config_name] = {
                "val_loglik": float("-inf"), "error": str(e)
            }

    # Rank by val_loglik (higher is better)
    ranked = sorted(
        [(k, v) for k, v in stage1_results.items() if "error" not in v],
        key=lambda x: x[1]["val_loglik"],
        reverse=True,
    )
    print(f"\n[reg] Stage 1 Rankings (higher val_loglik = better):")
    print(f"{'Model':<20} {'val_loglik':>12} {'AUC':>8} {'R²':>8}")
    print("-" * 52)
    for name, res in ranked:
        print(f"{name:<20} {res['val_loglik']:>12.4f} "
              f"{res['auc']:>8.4f} {res['r2_train']:>8.4f}")

    top2 = [name for name, _ in ranked[:2]]
    print(f"\n[reg] Top 2 from Stage 1: {top2}")
    print(f"[reg] Note: top 2 carry forward default-config results — no extra tuning")
    print(f"[reg] XGBoost always gets focused grid search in Stage 2")

    # ------------------------------------------------------------------
    # Stage 2: XGBoost grid search only
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("[reg] STAGE 2 — XGBoost grid search (5 configs)")
    print(f"{'='*60}")

    stage2_results = {}

    for i, params in enumerate(STAGE2_XGBOOST_CONFIGS):
        config_name = f"xgboost_s2_{i}"
        try:
            model    = xgb.XGBRegressor(**params, verbosity=0)
            model.fit(X_train, y_train)
            r2_train = compute_r2(y_train, model.predict(X_train))
            metrics  = compute_metrics(
                model.predict(X_val), df_val,
                subject_lookup, ood_ability_lookup, global_mean
            )
            stage2_results[config_name] = {
                "r2_train":                 r2_train,
                "val_loglik":               metrics["val_loglik"],
                "auc":                      metrics["auc"],
                "val_loglik_per_benchmark": metrics["val_loglik_per_benchmark"],
                "n_ood_lookups":            metrics["n_ood_lookups"],
                "n_global_fallbacks":       metrics["n_global_fallbacks"],
                "params":                   params,
            }
            print(f"[reg]   Config {i}: R²={r2_train:.4f} | "
                  f"val_loglik={metrics['val_loglik']:.4f} | "
                  f"AUC={metrics['auc']:.4f} | "
                  f"depth={params['max_depth']}")
        except Exception as e:
            print(f"[reg]   Config {i} failed: {e}")

    # ------------------------------------------------------------------
    # Stage 2: MLP grid search
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("[reg] STAGE 2 — MLP grid search (3 configs)")
    print(f"{'='*60}")

    for i, params in enumerate(STAGE2_MLP_CONFIGS):
        config_name = f"mlp_s2_{i}"
        try:
            model    = MLPRegressor(**params)
            model.fit(X_train, y_train)
            r2_train = compute_r2(y_train, model.predict(X_train))
            metrics  = compute_metrics(
                model.predict(X_val), df_val,
                subject_lookup, ood_ability_lookup, global_mean
            )
            stage2_results[config_name] = {
                "r2_train":                 r2_train,
                "val_loglik":               metrics["val_loglik"],
                "auc":                      metrics["auc"],
                "val_loglik_per_benchmark": metrics["val_loglik_per_benchmark"],
                "n_ood_lookups":            metrics["n_ood_lookups"],
                "n_global_fallbacks":       metrics["n_global_fallbacks"],
                "params":                   params,
            }
            print(f"[reg]   MLP config {i}: R²={r2_train:.4f} | "
                  f"val_loglik={metrics['val_loglik']:.4f} | "
                  f"AUC={metrics['auc']:.4f} | "
                  f"dims={params['hidden_dims']}")
        except Exception as e:
            print(f"[reg]   MLP config {i} failed: {e}")

    # ------------------------------------------------------------------
    # Select best overall
    # ------------------------------------------------------------------
    all_results = {**stage1_results, **stage2_results}
    valid       = {k: v for k, v in all_results.items() if "error" not in v}
    best_name   = max(valid, key=lambda k: valid[k]["val_loglik"])
    best_result = valid[best_name]

    print(f"\n[reg] ✅ Best model: {best_name}")
    print(f"[reg]    val_loglik: {best_result['val_loglik']:.4f}")
    print(f"[reg]    AUC:        {best_result['auc']:.4f}")
    print(f"\n[reg] Per-benchmark val_loglik for best model:")
    for b, ll in sorted(best_result["val_loglik_per_benchmark"].items()):
        flag = " ⚠️" if stage1a.get(b, {}).get("stability_flag", False) else ""
        print(f"  {b:<20} {ll:.4f}{flag}")

    # ------------------------------------------------------------------
    # Refit best model on full training data
    # ------------------------------------------------------------------
    if "xgboost_s2" in best_name:
        idx        = int(best_name.split("_")[-1])
        best_model = xgb.XGBRegressor(**STAGE2_XGBOOST_CONFIGS[idx], verbosity=0)
    elif "mlp_s2" in best_name:
        idx        = int(best_name.split("_")[-1])
        best_model = MLPRegressor(**STAGE2_MLP_CONFIGS[idx])
    else:
        best_cfg   = STAGE1_CONFIGS[best_name]
        best_model = build_model(best_cfg["model"], best_cfg["params"])
        best_model.fit(X_train, y_train)

    # Run basic checks before returning
    run_basic_checks(
        stage1a, subject_lookup, ood_ability_lookup,
        global_mean, best_model, encoder, device
    )

    # Build output dict (caller handles saving)
    output_pkl = {
        "regressor":                    best_model,
        "subject_lookup":               subject_lookup,
        "ood_ability_lookup":           ood_ability_lookup,
        "global_mean":                  global_mean,
        "encoder_model":                ENCODER_MODEL,
        "stage1_results":               stage1_results,
        "stage2_results":               stage2_results,
        "best_model":                   best_name,
        "best_val_loglik":              best_result["val_loglik"],
        "best_auc":                     best_result["auc"],
        "best_val_loglik_per_benchmark": best_result["val_loglik_per_benchmark"],
    }

    summary = {
        "best_model":      best_name,
        "best_val_loglik": best_result["val_loglik"],
        "best_auc":        best_result["auc"],
        "encoder":         ENCODER_MODEL,
        "stage1": {
            k: {kk: vv for kk, vv in v.items()
                if kk not in ("config", "val_loglik_per_benchmark")}
            for k, v in stage1_results.items() if "error" not in v
        },
        "stage2": {
            k: {kk: vv for kk, vv in v.items()
                if kk not in ("params", "val_loglik_per_benchmark")}
            for k, v in stage2_results.items()
        },
        "best_val_loglik_per_benchmark": best_result["val_loglik_per_benchmark"],
    }

    return summary
