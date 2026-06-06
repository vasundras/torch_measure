# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Regression tests for lightweight helpers in submission/train.py."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = REPO_ROOT / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))


def test_build_wide_response_averages_duplicate_subject_item_cells():
    import train

    s = torch.tensor([0, 0, 1])
    i = torch.tensor([0, 0, 1])
    y = torch.tensor([0.0, 1.0, 1.0])

    wide = train._build_wide_response(s, i, y, n_subj=2, n_items=2)

    assert wide[0, 0].item() == 0.5
    assert wide[1, 1].item() == 1.0
    assert torch.isnan(wide[0, 1])


def test_build_eb_tables_name_lc_includes_sparse_sb_subjects():
    import train

    tables = train.build_eb_tables(
        [
            {
                "subject_name": "Sparse-Subject",
                "benchmark": "mmlupro",
                "condition": "none",
                "label": 1.0,
            }
        ]
    )

    assert "sparse-subject" in tables["name_lc"]
    assert tables["name_lc"]["sparse-subject"] == "Sparse-Subject"


def test_build_eb_tables_threshold_filters_sbc():
    """Triples with fewer than ``_EB_SBC_MIN_N`` observations must not land
    in the ``sbc`` table (sparse cells would otherwise dominate noise)."""
    import train

    n_below = train._EB_SBC_MIN_N - 1  # default 2
    examples = [
        {
            "subject_name": "Subj-A",
            "benchmark": "mmlupro",
            "condition": "none",
            "label": 1.0,
        }
        for _ in range(n_below)
    ]
    tables = train.build_eb_tables(examples)
    assert "Subj-A||mmlupro||none" not in tables["sbc"]

    # Now bump above the threshold and verify the cell appears.
    examples += [
        {
            "subject_name": "Subj-A",
            "benchmark": "mmlupro",
            "condition": "none",
            "label": 1.0,
        }
    ]
    tables = train.build_eb_tables(examples)
    assert "Subj-A||mmlupro||none" in tables["sbc"]


def test_build_eb_tables_bayesian_shrinkage_pulls_toward_prior():
    """Verify the closed-form shrunk = (n*raw + alpha*prior) / (n+alpha)
    against a hand-computed expectation for the level-2 sb cell."""
    import math

    import train

    # Construct a mid-range scenario that survives the table clip:
    # 10 subj-A rows with label=1, 10 subj-A rows with label=0 across two
    # benchmarks. Subj-A gets _EB_SUBJ_MIN_N=10 obs split 5/5 between the
    # two benchmarks: subj_p_raw = 10/20 = 0.5. Bench-X gets 5 correct from
    # Subj-A + 5 correct from Subj-B (a different subject) = 10/10 = 1.0,
    # clipped to 0.95. global_mean = 15/20 = 0.75. Subj-A's sb cell on
    # Bench-X has n=5 (all 1.0), raw_p = 1.0 → shrinks toward
    # sigmoid(logit(0.5) + logit(0.95) - logit(0.75)) = ~0.81 ≈ prior_p.
    examples = []
    # 5 Subj-A rows on Bench-X, all correct.
    for _ in range(5):
        examples.append({"subject_name": "Subj-A", "benchmark": "Bench-X", "condition": "none", "label": 1.0})
    # 5 Subj-A rows on Bench-Y, all wrong.
    for _ in range(5):
        examples.append({"subject_name": "Subj-A", "benchmark": "Bench-Y", "condition": "none", "label": 0.0})
    # 5 Subj-B rows on Bench-X, all correct.
    for _ in range(5):
        examples.append({"subject_name": "Subj-B", "benchmark": "Bench-X", "condition": "none", "label": 1.0})
    # 5 Subj-B rows on Bench-Y, all correct. (gives Subj-B n=10 obs >= MIN_N)
    for _ in range(5):
        examples.append({"subject_name": "Subj-B", "benchmark": "Bench-Y", "condition": "none", "label": 1.0})

    tables = train.build_eb_tables(examples)

    # Subj-A's marginal correctness: 5/10 = 0.5 (clipped passes through).
    assert "Subj-A" in tables["subj"]
    assert abs(tables["subj"]["Subj-A"] - 0.5) < 1e-6
    # Bench-X is fully correct (10/10) → clips to _TABLE_CLIP_HI.
    assert tables["bench"]["Bench-X"] == train._TABLE_CLIP_HI
    # Compute expected sb cell for ("Subj-A","Bench-X"): n=5, raw_p=1.0;
    # prior_p = sigmoid(logit(0.5) + logit(0.95) - logit(global_mean))
    global_mean = tables["global"]
    subj_p = tables["subj"]["Subj-A"]
    bench_p = tables["bench"]["Bench-X"]
    prior_logit = (
        math.log(subj_p / (1 - subj_p)) + math.log(bench_p / (1 - bench_p)) - math.log(global_mean / (1 - global_mean))
    )
    prior_p = 1.0 / (1.0 + math.exp(-prior_logit))
    expected_shrunk = (5 * 1.0 + train._EB_SB_ALPHA * prior_p) / (5 + train._EB_SB_ALPHA)
    # Clip the expectation the same way the implementation does.
    expected_shrunk_clipped = max(train._TABLE_CLIP_LO, min(train._TABLE_CLIP_HI, expected_shrunk))
    sb_p = tables["sb"]["Subj-A||Bench-X"]
    assert abs(sb_p - expected_shrunk_clipped) < 1e-6, f"sb_p={sb_p} expected={expected_shrunk_clipped}"


def test_build_eb_tables_global_mean_clipped_to_table_range():
    """``global_mean`` must be clipped to ``[_TABLE_CLIP_LO, _TABLE_CLIP_HI]``;
    e.g. all-correct training rows should not produce 1.0 (would explode logit)."""
    import train

    # All labels 1.0 → raw mean = 1.0, must clip to _TABLE_CLIP_HI = 0.95.
    all_correct = [
        {
            "subject_name": f"Subj-{i}",
            "benchmark": "mmlupro",
            "condition": "none",
            "label": 1.0,
        }
        for i in range(5)
    ]
    tables_hi = train.build_eb_tables(all_correct)
    assert tables_hi["global"] == train._TABLE_CLIP_HI

    # All labels 0.0 → raw mean = 0.0, must clip to _TABLE_CLIP_LO = 0.05.
    all_wrong = [
        {
            "subject_name": f"Subj-{i}",
            "benchmark": "mmlupro",
            "condition": "none",
            "label": 0.0,
        }
        for i in range(5)
    ]
    tables_lo = train.build_eb_tables(all_wrong)
    assert tables_lo["global"] == train._TABLE_CLIP_LO
