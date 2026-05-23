# Copyright (c) 2026 AIMS Foundations. MIT License.

"""NaN/Inf safety for _logit and LLMJudgeIRT.

The Python `max/min` ladder used by every `_logit` implementation collapses
NaN to the lower clip via `1e-7 < nan == False`, producing a finite
``-16.118`` and `sigmoid` ≈ ``0.9999`` downstream — the "constant-prediction"
signature defended against by
``docs/solutions/runtime-errors/silent-ncf-head-load-failure-via-full-module-pickle-2026-05-17.md``.
These tests pin all 3 `_logit` copies plus `LLMJudgeIRT.predict` to refuse
non-finite input loudly.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = REPO_ROOT / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))


@pytest.fixture(autouse=True)
def _local_smoke_env(monkeypatch):
    monkeypatch.setenv("PREDICTIVE_EVAL_LOCAL_SMOKE_TEST", "1")


class TestCaimiraLiteLogitRejectsNonFinite:
    def test_nan_raises(self):
        import importlib

        cl = importlib.reload(importlib.import_module("submission.caimira_lite"))
        with pytest.raises(ValueError):
            cl._logit(float("nan"))

    def test_pos_inf_raises(self):
        import importlib

        cl = importlib.reload(importlib.import_module("submission.caimira_lite"))
        with pytest.raises(ValueError):
            cl._logit(float("inf"))

    def test_neg_inf_raises(self):
        import importlib

        cl = importlib.reload(importlib.import_module("submission.caimira_lite"))
        with pytest.raises(ValueError):
            cl._logit(float("-inf"))


class TestModelLogitRejectsNonFinite:
    def test_nan_raises(self):
        import importlib

        model = importlib.reload(importlib.import_module("submission.model"))
        with pytest.raises(ValueError):
            model._logit(float("nan"))


class TestColdStartLookupLogitRejectsNonFinite:
    def test_nan_raises(self):
        from torch_measure.models import cold_start_lookup

        with pytest.raises(ValueError):
            cold_start_lookup._logit(float("nan"))

    def test_inf_raises(self):
        from torch_measure.models import cold_start_lookup

        with pytest.raises(ValueError):
            cold_start_lookup._logit(float("inf"))


class TestLLMJudgeIRTHandlesNonFiniteJudge:
    def _make(self, judge_fn):
        from torch_measure.models import LLMJudgeIRT
        from torch_measure.models.cold_start_lookup import ColdStartLookupPredictor

        lookup = ColdStartLookupPredictor(
            sbc={},
            sb={},
            subj={"gpt-4": 0.7},
            bench={"mmlupro": 0.5},
            global_mean=0.6,
            name_aliases={},
            name_lc={"gpt-4": "gpt-4"},
        )
        return LLMJudgeIRT(lookup=lookup, judge_fn=judge_fn, alpha=0.5, beta=0.0)

    def _record(self):
        return {
            "subject_content": "Name: gpt-4",
            "benchmark": "mmlupro",
            "condition": "cot",
            "item_content": "q",
        }

    def test_nan_judge_logit_falls_back_to_zero(self):
        model = self._make(lambda *_args, **_kwargs: float("nan"))
        p = model.predict(self._record())
        assert math.isfinite(p)
        lo, hi = model.lookup.output_clip
        assert lo <= p <= hi

    def test_pos_inf_judge_logit_falls_back_to_zero(self):
        model = self._make(lambda *_args, **_kwargs: float("inf"))
        p = model.predict(self._record())
        assert math.isfinite(p)
        lo, hi = model.lookup.output_clip
        assert lo <= p <= hi

    def test_neg_inf_judge_logit_falls_back_to_zero(self):
        model = self._make(lambda *_args, **_kwargs: float("-inf"))
        p = model.predict(self._record())
        assert math.isfinite(p)
        lo, hi = model.lookup.output_clip
        assert lo <= p <= hi
