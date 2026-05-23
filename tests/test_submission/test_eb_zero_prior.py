# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Regression test for the level-3 IRT-blend or-chain on 0.0 priors.

Both ``cold_start_lookup.ColdStartLookupPredictor._lookup_p`` and
``caimira_lite.EBLookup.lookup_p`` build ``subj_p`` / ``bench_p`` via
``self.X.get(key) or self._X_ci.get(key.lower())``. If ``self.X.get`` returns
a legitimate ``0.0`` prior, the ``or`` short-circuits to the case-insensitive
view, masking the real prior. ``_table_clip`` currently floors stored priors
at ``0.05`` so the bug is structurally masked, but the contract should hold
regardless of clip choice.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = REPO_ROOT / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))


def _force_ci_divergence(pred, *, subj_key=None, bench_key=None, subj_ci_value=None, bench_ci_value=None):
    """Force the case-insensitive views to differ from the canonical dicts.

    The constructors build the CI views from the canonical dicts, so a real
    deployment will never see divergence; but the or-chain bug only manifests
    when the CI view differs from the canonical value. Poking the CI dict
    directly is the most direct way to drive the bug without invoking dict-
    overwrite gymnastics that depend on insertion order.
    """
    if subj_key is not None:
        pred._subj_ci[subj_key.lower()] = subj_ci_value
    if bench_key is not None:
        pred._bench_ci[bench_key.lower()] = bench_ci_value


def test_cold_start_zero_subj_prior_not_overridden_by_ci_view():
    from torch_measure.models.cold_start_lookup import ColdStartLookupPredictor

    pred = ColdStartLookupPredictor(
        sbc={},
        sb={},
        subj={"gpt-4": 0.0},
        bench={},
        global_mean=0.5,
        name_aliases={},
        name_lc={},
    )
    _force_ci_divergence(pred, subj_key="gpt-4", subj_ci_value=0.9)
    # Level 5: subject-only path. With the or-chain bug, `0.0 or 0.9` == 0.9.
    p = pred._lookup_p("gpt-4", "unknown_bench", "x")
    assert p == 0.0


def test_cold_start_zero_bench_prior_not_overridden_by_ci_view():
    from torch_measure.models.cold_start_lookup import ColdStartLookupPredictor

    pred = ColdStartLookupPredictor(
        sbc={},
        sb={},
        subj={},
        bench={"mmlupro": 0.0},
        global_mean=0.5,
        name_aliases={},
        name_lc={},
    )
    _force_ci_divergence(pred, bench_key="mmlupro", bench_ci_value=0.9)
    p = pred._lookup_p("unknown_subject", "mmlupro", "x")
    assert p == 0.0


def test_cold_start_zero_priors_in_irt_blend_not_overridden_by_ci_view():
    from torch_measure.models.cold_start_lookup import ColdStartLookupPredictor

    pred = ColdStartLookupPredictor(
        sbc={},
        sb={},
        subj={"gpt-4": 0.0},
        bench={"mmlupro": 0.0},
        global_mean=0.5,
        name_aliases={},
        name_lc={},
    )
    _force_ci_divergence(
        pred,
        subj_key="gpt-4",
        subj_ci_value=0.9,
        bench_key="mmlupro",
        bench_ci_value=0.9,
    )
    p = pred._lookup_p("gpt-4", "mmlupro", "x")
    # With the or-chain bug, level 3 fires with (0.9, 0.9, 0.5) → ~0.92.
    # With the fix, level 3 fires with (0.0, 0.0, 0.5), which after the
    # _logit guard clips both 0.0 inputs and produces a tiny blended value
    # → output_clip floor 0.05.
    assert p < 0.5, f"expected level-3 to use 0.0 priors, got {p}"


def test_cold_start_none_condition_triple_uses_ci_view():
    from torch_measure.models.cold_start_lookup import ColdStartLookupPredictor

    pred = ColdStartLookupPredictor(
        sbc={"gpt-4||mmlupro||none": 0.9},
        sb={"gpt-4||mmlupro": 0.2},
        subj={},
        bench={},
        global_mean=0.5,
        name_aliases={},
        name_lc={},
    )
    p = pred._lookup_p("GPT-4", "MMLUPRO", "cot")
    assert p == 0.9


def test_eb_lookup_zero_subj_prior_not_overridden_by_ci_view():
    import importlib

    cl = importlib.import_module("caimira_lite")
    eb = cl.EBLookup(
        sbc={},
        sb={},
        subj={"gpt-4": 0.0},
        bench={},
        global_mean=0.5,
        name_aliases={},
        name_lc={},
    )
    eb._subj_ci["gpt-4"] = 0.9
    p = eb.lookup_p("gpt-4", "unknown_bench", "x")
    assert p == 0.0


def test_eb_lookup_zero_bench_prior_not_overridden_by_ci_view():
    import importlib

    cl = importlib.import_module("caimira_lite")
    eb = cl.EBLookup(
        sbc={},
        sb={},
        subj={},
        bench={"mmlupro": 0.0},
        global_mean=0.5,
        name_aliases={},
        name_lc={},
    )
    eb._bench_ci["mmlupro"] = 0.9
    p = eb.lookup_p("unknown_subject", "mmlupro", "x")
    assert p == 0.0


def test_eb_lookup_zero_priors_in_irt_blend_not_overridden_by_ci_view():
    import importlib

    cl = importlib.import_module("caimira_lite")
    eb = cl.EBLookup(
        sbc={},
        sb={},
        subj={"gpt-4": 0.0},
        bench={"mmlupro": 0.0},
        global_mean=0.5,
        name_aliases={},
        name_lc={},
    )
    eb._subj_ci["gpt-4"] = 0.9
    eb._bench_ci["mmlupro"] = 0.9
    p = eb.lookup_p("gpt-4", "mmlupro", "x")
    assert p < 0.5, f"expected level-3 to use 0.0 priors, got {p}"


def test_eb_lookup_none_condition_triple_uses_ci_view():
    import importlib

    cl = importlib.import_module("caimira_lite")
    eb = cl.EBLookup(
        sbc={"gpt-4||mmlupro||none": 0.9},
        sb={"gpt-4||mmlupro": 0.2},
        subj={},
        bench={},
        global_mean=0.5,
        name_aliases={},
        name_lc={},
    )
    p = eb.lookup_p("GPT-4", "MMLUPRO", "cot")
    assert p == 0.9
