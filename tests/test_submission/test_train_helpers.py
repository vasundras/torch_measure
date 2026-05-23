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
