# Copyright (c) 2026 AIMS Foundations. MIT License.
"""Level-dependent ensemble blend for the Content-Aware Ensemble pipeline.

This module implements the final prediction stage of the Content-Aware
Ensemble: a level-dependent blend of
:class:`torch_measure.models.ColdStartLookupPredictor` (the lookup model)
with the PGE pipeline (IRT abilities + text-based difficulty regressor).

The blend weight lambda depends on which fallback level fired in the lookup
model's 6-level hierarchy. When the lookup model has direct empirical data
(levels 1-2), it dominates completely. When it falls back to uninformative
priors (level 6 — global mean), the content-based model takes over fully.

Lambda function
---------------
Empirically derived from 970,675 validation rows across 16 benchmarks:

.. code-block:: text

    λ(level) = 0.114 * exp(0.740 * (level - 3))   for level in {3, 4, 5, 6}
    λ(level) = 0.000                                for level in {1, 2}

Anchored at two high-confidence empirical data points:
- (level=5, λ*=0.477) — 23,766 val rows
- (level=6, λ*=1.000) — 160,447 val rows

Hard zero at levels 1-2: directional agreement analysis showed that the
content-based model pushes predictions in the wrong direction relative to
the lookup model at these levels. The exponential form was selected over
linear and power forms (RMSE=0.059 vs 0.157 and 0.076 respectively).

Blend formula
-------------
.. code-block:: text

    p_final = p_lookup + λ(level) * (p_mine - p_lookup)

Negative-result context
-----------------------
This ensemble showed real positive signal on the validation set:
Spearman r=+0.139 between content-model corrections and ground truth
(p<1e-300, n=970,675). Val NLL improved by ~0.013 over the lookup model
alone. However, it did not improve the leaderboard score on the hidden
test set — the content-based difficulty regressor did not generalize well
enough to the test distribution to provide net benefit.

See ``tutorials/content_aware_ensemble_experiment.ipynb`` for the full
experimental log including the lambda derivation, residual analysis, and
per-benchmark breakdown.
"""

import math

import numpy as np
from sentence_transformers import SentenceTransformer

from torch_measure.models.cold_start_lookup import (
    parse_subject_name,
    resolve_subject_name,
    _logit as _cs_logit,
    _sigmoid as _cs_sigmoid,
    _clip as _cs_clip,
)

# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------

TEMPERATURE = 1.4002
CLIP_MIN    = 0.001
CLIP_MAX    = 0.999

SHRINKAGE = {
    "androidworld": {"alpha": 0.7,  "pass_rate": 0.8720538720538721},
    "cybench":      {"alpha": 0.7,  "pass_rate": 0.2726190476190476},
}

# Ensemble lambda — exponential fitted on domain [3,6]
# λ(level) = 0.114 * exp(0.740 * (level - 3))
# Anchored at empirical val data points:
#   (level=5, λ*=0.477) and (level=6, λ*=1.000)
# Hard zero at levels 1-2: directional agreement is FALSE at these levels
LAMBDA = {
    1: 0.000,
    2: 0.000,
    3: 0.114,
    4: 0.239,
    5: 0.477,
    6: 1.000,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _get_ability(
    benchmark: str,
    subject_content: str,
    subject_lookup: dict,
    ood_ability_lookup: dict,
    global_mean: float,
) -> float:
    bench_dict = subject_lookup.get(benchmark, {})
    if subject_content in bench_dict:
        return bench_dict[subject_content]
    if subject_content in ood_ability_lookup:
        return ood_ability_lookup[subject_content]
    return global_mean


def _predict_lookup_with_level(
    subject_content: str,
    benchmark: str,
    condition: str,
    lookup,
) -> tuple[float, int]:
    """Returns (p_lookup, level) from ColdStartLookupPredictor."""
    condition = (condition or "none").strip() or "none"
    raw_name  = parse_subject_name(subject_content)
    subj_name = resolve_subject_name(
        raw_name, lookup.subj, lookup.sb, lookup.sbc,
        lookup.name_aliases, lookup.name_lc,
    )
    # Level 1
    key3 = f"{subj_name}||{benchmark}||{condition}"
    if key3 in lookup.sbc:
        return float(np.clip(lookup.sbc[key3], *lookup.output_clip)), 1
    if key3.lower() in lookup._sbc_ci:
        return float(np.clip(lookup._sbc_ci[key3.lower()], *lookup.output_clip)), 1
    key3_none = f"{subj_name}||{benchmark}||none"
    if key3_none in lookup.sbc:
        return float(np.clip(lookup.sbc[key3_none], *lookup.output_clip)), 1
    # Level 2
    key2 = f"{subj_name}||{benchmark}"
    if key2 in lookup.sb:
        return float(np.clip(lookup.sb[key2], *lookup.output_clip)), 2
    if key2.lower() in lookup._sb_ci:
        return float(np.clip(lookup._sb_ci[key2.lower()], *lookup.output_clip)), 2
    # Level 3
    subj_p  = lookup.subj.get(subj_name) or lookup._subj_ci.get(subj_name.lower())
    bench_p = lookup.bench.get(benchmark) or lookup._bench_ci.get(benchmark.lower())
    if subj_p is not None and bench_p is not None:
        p = _cs_clip(_cs_sigmoid(
            _cs_logit(subj_p) + _cs_logit(bench_p) - _cs_logit(lookup.global_mean)
        ))
        return float(np.clip(p, *lookup.output_clip)), 3
    # Level 4
    if bench_p is not None:
        return float(np.clip(bench_p, *lookup.output_clip)), 4
    # Level 5
    if subj_p is not None:
        return float(np.clip(subj_p, *lookup.output_clip)), 5
    # Level 6
    return float(np.clip(lookup.global_mean, *lookup.output_clip)), 6


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

def predict(
    input: dict,
    labeled: list | None,
    lookup,
    regressor,
    encoder: SentenceTransformer,
    subject_lookup: dict,
    ood_ability_lookup: dict,
    global_mean: float,
    temperature: float = TEMPERATURE,
    shrinkage: dict = SHRINKAGE,
    lambda_table: dict = LAMBDA,
) -> float:
    """
    Predict P(subject passes item) using the level-dependent ensemble blend.

    Parameters
    ----------
    input : dict
        Competition-format record with keys: ``benchmark``, ``subject_content``,
        ``item_content``, ``condition``.
    labeled : list or None
        Revealed ground-truth labels for adaptive calibration. Passed to
        ``lookup.calibrate()`` if non-empty.
    lookup : ColdStartLookupPredictor
        Fitted lookup model providing (p_lookup, level).
    regressor : sklearn-compatible regressor
        Fitted difficulty regressor (from difficulty_regressor.py).
    encoder : SentenceTransformer
        Sentence encoder used to embed item text.
    subject_lookup : dict
        ``{benchmark: {subject_content: norm_ability}}``
    ood_ability_lookup : dict
        ``{subject_content: mean_norm_ability}`` for OOD benchmarks.
    global_mean : float
        Global mean ability fallback.
    temperature : float
        Temperature scaling parameter T. Default: 1.4002 (fitted on val).
    shrinkage : dict
        Per-benchmark shrinkage config. Default: fitted values.
    lambda_table : dict
        Level → blend weight mapping. Default: empirically derived LAMBDA.

    Returns
    -------
    float
        Blended probability in [CLIP_MIN, CLIP_MAX].
    """
    benchmark       = input["benchmark"]
    subject_content = input["subject_content"]
    item_content    = input["item_content"]
    condition       = input.get("condition", "none") or "none"

    # --- Lookup model prediction + level ---
    if labeled:
        lookup.calibrate(labeled)
    p_lookup, level = _predict_lookup_with_level(
        subject_content, benchmark, condition, lookup
    )
    if labeled and benchmark in lookup._platt:
        slope, intercept = lookup._platt[benchmark]
        p_lookup = _cs_sigmoid(slope * _cs_logit(p_lookup) + intercept)
        p_lookup = float(np.clip(p_lookup, *lookup.output_clip))

    # --- Content-based model prediction ---
    text      = f"Benchmark: {benchmark}\n{item_content}"
    embedding = encoder.encode(
        text, convert_to_tensor=False, show_progress_bar=False
    ).reshape(1, -1)
    norm_diff    = float(regressor.predict(embedding)[0])
    norm_ability = _get_ability(
        benchmark, subject_content,
        subject_lookup, ood_ability_lookup, global_mean,
    )
    logit = norm_ability - norm_diff
    P_T   = _sigmoid(logit / temperature)
    if benchmark in shrinkage:
        alpha     = shrinkage[benchmark]["alpha"]
        pass_rate = shrinkage[benchmark]["pass_rate"]
        P_T       = (1 - alpha) * P_T + alpha * pass_rate
    p_mine = float(np.clip(P_T, CLIP_MIN, CLIP_MAX))

    # --- Blend ---
    lam     = lambda_table.get(level, 0.0)
    p_final = float(np.clip(p_lookup + lam * (p_mine - p_lookup), CLIP_MIN, CLIP_MAX))
    return p_final


# ---------------------------------------------------------------------------
# acquisition_function()
# ---------------------------------------------------------------------------

def acquisition_function(input: dict, lookup) -> float:
    """
    Score how desirable it is to reveal the ground-truth label for this input.

    Uses uncertainty sampling on the lookup model: inputs where
    ColdStartLookupPredictor is most uncertain (p closest to 0.5) are
    most valuable to label, because the K=5 revealed labels are used
    primarily for the lookup model's Platt calibration shift.

    Parameters
    ----------
    input : dict
        Competition-format record.
    lookup : ColdStartLookupPredictor
        Used to get the lookup model's raw prediction.

    Returns
    -------
    float
        Higher = more desirable to label. Most uncertain = highest score.
    """
    p = lookup.predict(input)
    return float(-(abs(p - 0.5)))
