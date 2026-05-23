# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Regression test for ColdStartLookupPredictor.predict_batch calibration.

The original ``predict_batch`` called ``self.calibrate(labeled)`` to populate
``self._platt`` but then invoked ``self.predict(r)`` without forwarding
``labeled``. Inside ``predict()`` the Platt application is guarded by
``if labeled:`` — defaulting to ``None`` — so the calibration was fit and
then never applied. Calling ``predict_batch(records, labeled=...)`` was
functionally identical to calling it without ``labeled`` at all.
"""

from __future__ import annotations

from torch_measure.models.cold_start_lookup import ColdStartLookupPredictor


def _record() -> dict:
    return {
        "benchmark": "mmlupro",
        "condition": "x",
        "subject_content": "Name: m",
        "item_content": "q",
    }


def _labeled() -> list[dict]:
    return [
        {
            "benchmark": "mmlupro",
            "condition": "x",
            "subject_content": f"Name: m{idx}",
            "item_content": f"q{idx}",
            "label": 1.0 if idx % 2 == 0 else 0.0,
        }
        for idx in range(8)
    ]


def _make_predictor() -> ColdStartLookupPredictor:
    return ColdStartLookupPredictor(
        sbc={},
        sb={},
        subj={},
        bench={"mmlupro": 0.30},
        global_mean=0.50,
        name_aliases={},
        name_lc={},
    )


def test_predict_batch_with_labeled_applies_platt_shift():
    """predict_batch must apply Platt to each prediction when labeled is given.

    Setup: bench prior 0.30, labeled mean = 0.5 (4/8 positives) → Platt shift
    is positive (mean_y_logit 0 minus mean_x_logit logit(0.30) ≈ +0.847).
    After fix, predict_batch(records, labeled=...) returns a probability
    HIGHER than predict_batch(records) (no labeled).
    """
    pred = _make_predictor()
    uncalibrated = pred.predict_batch([_record()])[0]

    pred_with_labels = _make_predictor()
    calibrated = pred_with_labels.predict_batch([_record()], labeled=_labeled())[0]

    assert calibrated > uncalibrated, (
        f"predict_batch ignored labeled: calibrated={calibrated} vs "
        f"uncalibrated={uncalibrated} (expected calibrated > uncalibrated due "
        "to positive Platt shift from labeled mean 0.5 > prior 0.30)"
    )


def test_predict_batch_with_labeled_matches_predict_with_labeled():
    """predict_batch must produce the same result as a loop over predict()."""
    pred_batch = _make_predictor()
    batch_result = pred_batch.predict_batch([_record()], labeled=_labeled())[0]

    pred_loop = _make_predictor()
    loop_result = pred_loop.predict(_record(), labeled=_labeled())

    assert batch_result == loop_result, (
        f"predict_batch={batch_result} differs from predict={loop_result} "
        "when both receive the same labeled list"
    )
