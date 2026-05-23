# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Kit-contract tests for submission/model.py.

Runs the module in LOCAL_SMOKE mode (PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1)
so we exercise the import + signature surface without requiring the
trained artifact or the HF cache. The kit's own check_submission_zip.py
also uses this mode (per starting_kit/tools/check_submission_zip.py:18,131).
"""

from __future__ import annotations

import importlib
import numbers
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = REPO_ROOT / "submission"


def _load_model_module():
    """Reload submission.model with LOCAL_SMOKE=1 so we skip artifact loads."""
    os.environ["PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"] = "1"
    if str(SUBMISSION_DIR) not in sys.path:
        sys.path.insert(0, str(SUBMISSION_DIR))
    if "model" in sys.modules:
        del sys.modules["model"]
    if "caimira_lite" in sys.modules:
        # Reload to pick up the LOCAL_SMOKE env-var change in model.py.
        del sys.modules["caimira_lite"]
    return importlib.import_module("model")


class TestModelContract:
    def test_module_imports_in_local_smoke_mode(self):
        mod = _load_model_module()
        assert hasattr(mod, "predict")
        assert callable(mod.predict)

    def test_predict_signature(self):
        mod = _load_model_module()
        import inspect

        sig = inspect.signature(mod.predict)
        params = list(sig.parameters.keys())
        assert params[0] == "input"
        assert "labeled" in params
        assert sig.parameters["labeled"].default is None

    def test_predict_returns_native_float(self):
        mod = _load_model_module()
        sample_input = {
            "benchmark": "mmlupro",
            "condition": "cot",
            "subject_content": "Name: gpt-4\nOrganization: openai",
            "item_content": "What is 2+2?",
        }
        out = mod.predict(sample_input, labeled=None)
        assert isinstance(out, float)
        # Kit checker rejects bool — confirm we don't sneak one through.
        assert not isinstance(out, bool)
        # Also verify it's specifically a numbers.Real per check_submission_zip.py:154.
        assert isinstance(out, numbers.Real)

    def test_predict_returns_finite_in_unit_interval(self):
        mod = _load_model_module()
        sample_input = {
            "benchmark": "mmlupro",
            "condition": "cot",
            "subject_content": "Name: gpt-4",
            "item_content": "What is 2+2?",
        }
        out = mod.predict(sample_input)
        import math

        assert math.isfinite(out)
        assert 0.0 <= out <= 1.0

    def test_predict_accepts_empty_labeled(self):
        mod = _load_model_module()
        sample_input = {
            "benchmark": "mmlupro",
            "condition": "cot",
            "subject_content": "Name: gpt-4",
            "item_content": "What is 2+2?",
        }
        out = mod.predict(sample_input, labeled=[])
        assert isinstance(out, float)
        assert 0.0 <= out <= 1.0

    def test_predict_accepts_labeled_as_kwarg(self):
        """Kit checker calls predict(dict(SMOKE_INPUT), labeled=[])."""
        mod = _load_model_module()
        sample_input = {
            "benchmark": "mmlupro",
            "condition": "cot",
            "subject_content": "Name: gpt-4",
            "item_content": "What is 2+2?",
        }
        out = mod.predict(dict(sample_input), labeled=[])
        assert 0.0 <= out <= 1.0


class TestModelAcquisitionContract:
    def test_acquisition_function_signature(self):
        if "labeling" in sys.modules:
            del sys.modules["labeling"]
        if str(SUBMISSION_DIR) not in sys.path:
            sys.path.insert(0, str(SUBMISSION_DIR))
        import inspect

        import labeling

        sig = inspect.signature(labeling.acquisition_function)
        assert list(sig.parameters.keys())[0] == "input"

    def test_acquisition_function_returns_finite_real(self):
        if "labeling" in sys.modules:
            del sys.modules["labeling"]
        if str(SUBMISSION_DIR) not in sys.path:
            sys.path.insert(0, str(SUBMISSION_DIR))
        import labeling

        sample = {
            "benchmark": "mmlupro",
            "condition": "cot",
            "subject_content": "Name: gpt-4",
            "item_content": "What is 2+2?",
        }
        out = labeling.acquisition_function(sample)
        import math

        assert isinstance(out, float)
        assert math.isfinite(out)
        assert not isinstance(out, bool)

    def test_acquisition_function_bulletproof_sentinel_on_exception(self):
        if "labeling" in sys.modules:
            del sys.modules["labeling"]
        if str(SUBMISSION_DIR) not in sys.path:
            sys.path.insert(0, str(SUBMISSION_DIR))
        import labeling

        # Pass a value that breaks _visible_text's str() — e.g., an object
        # whose __str__ raises. The bare-except path must catch and return
        # exactly 0.0.
        class _Boom:
            def __str__(self):
                raise RuntimeError("synthetic")

        out = labeling.acquisition_function({"benchmark": _Boom()})
        assert out == 0.0  # the distinguishable sentinel

    def test_acquisition_function_distinguishable_from_legitimate_floor(self):
        if "labeling" in sys.modules:
            del sys.modules["labeling"]
        if str(SUBMISSION_DIR) not in sys.path:
            sys.path.insert(0, str(SUBMISSION_DIR))
        import labeling

        # Reset module-level state for a clean run.
        labeling._seen_signatures.clear()
        labeling._stratum_counts.clear()
        labeling._candidate_count = 0
        sample = {
            "benchmark": "mmlupro",
            "condition": "cot",
            "subject_content": "Name: gpt-4",
            "item_content": "What is 2+2?",
        }
        score = labeling.acquisition_function(sample)
        # Legitimate floor includes tie_break > 0 — score should be strictly > 0.
        assert score > 0.0
        assert score <= 2.0  # _clamp_score's upper bound
