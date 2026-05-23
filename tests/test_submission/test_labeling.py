# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Stateful-globals coverage for submission/labeling.py.

The acquisition function carries three module-level globals
(`_seen_signatures`, `_stratum_counts`, `_candidate_count`) that bound cost
and govern diversity. The autouse fixture resets them between tests so
order does not pollute outcomes.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = REPO_ROOT / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))


@pytest.fixture(autouse=True)
def reset_labeling_globals():
    labeling = importlib.import_module("submission.labeling")
    labeling._seen_signatures.clear()
    labeling._stratum_counts.clear()
    labeling._candidate_count = 0
    yield labeling
    labeling._seen_signatures.clear()
    labeling._stratum_counts.clear()
    labeling._candidate_count = 0


def _row(idx: int) -> dict:
    return {
        "benchmark": f"bench_{idx % 17}",
        "condition": f"cond_{idx % 5}",
        "subject_content": f"Name: model-{idx % 41}",
        "item_content": f"Question text {idx} with token {idx % 97}",
    }


def test_reservoir_replaces_after_max_seen(reset_labeling_globals):
    labeling = reset_labeling_globals
    for idx in range(130):
        labeling.acquisition_function(_row(idx))
    assert len(labeling._seen_signatures) == labeling._MAX_SEEN
    assert labeling._candidate_count == 130


def test_metadata_bonus_decays_per_stratum(reset_labeling_globals):
    labeling = reset_labeling_globals
    ex = _row(0)
    bonuses = []
    for _ in range(10):
        bonuses.append(labeling._metadata_bonus(ex))
        labeling._update_stratum(ex)
    # Monotonic non-increasing. ``strict=False`` because the sliding window
    # is intentionally one element shorter than ``bonuses``.
    for prev, curr in zip(bonuses, bonuses[1:], strict=False):
        assert curr <= prev, f"bonus increased: {prev} -> {curr}"
    assert bonuses[0] > bonuses[-1]


def test_distinguishable_sentinel_on_exception(reset_labeling_globals, monkeypatch):
    labeling = reset_labeling_globals

    def boom(_ex):
        raise RuntimeError("simulated _simhash failure")

    monkeypatch.setattr(labeling, "_simhash", boom)
    score = labeling.acquisition_function(_row(0))
    assert score == 0.0


def test_legitimate_score_is_strictly_positive(reset_labeling_globals):
    labeling = reset_labeling_globals
    for idx in range(1000):
        score = labeling.acquisition_function(_row(idx))
        assert score > 0.0


def test_no_state_leak_across_module_reload(reset_labeling_globals):
    labeling = reset_labeling_globals
    for idx in range(50):
        labeling.acquisition_function(_row(idx))
    assert labeling._candidate_count == 50

    reloaded = importlib.reload(labeling)
    assert reloaded._candidate_count == 0
    assert reloaded._seen_signatures == []
    assert reloaded._stratum_counts == {}


@pytest.mark.parametrize(
    "text",
    [
        "你好 世界",
        "Привет мир",
        "مرحبا بالعالم",
        "नमस्ते दुनिया",
        "héllo wörld",
        "café",
    ],
)
def test_simhash_tokens_handle_multilingual_when_metadata_blank(text):
    """Pure non-Latin item_content with blank metadata: tokens must be non-empty.

    Real failure mode: a multilingual benchmark sends rows where all visible
    text is non-Latin. The current ``r"[a-z0-9]+"`` regex returns ``[]``, so
    ``_simhash`` falls back to ``["<empty>"]`` and every such row gets the
    same signature → zero diversity inside the round.
    """
    labeling = importlib.reload(importlib.import_module("submission.labeling"))
    ex = {
        "benchmark": "",
        "condition": "",
        "subject_content": "",
        "item_content": text,
    }
    tokens = labeling._tokens(ex)
    assert tokens, f"expected non-empty tokens for {text!r}, got {tokens}"


def test_metadata_bonus_overflow_stratum_does_not_outrank_early(reset_labeling_globals):
    """After _MAX_STRATA distinct strata are seen, a NEW stratum must not
    receive the MAX bonus 0.20 while frequently-seen early strata have decayed.
    """
    labeling = reset_labeling_globals
    early_ex = {
        "benchmark": "bench_early",
        "condition": "cond_early",
        "subject_content": "Name: model_early",
        "item_content": "early",
    }
    # Seed early_ex into the dict at count=1.
    labeling._update_stratum(early_ex)
    # Fill the rest with distinct seeds.
    for idx in range(labeling._MAX_STRATA - 1):
        seed_ex = {
            "benchmark": f"bench_seed_{idx}",
            "condition": "cond",
            "subject_content": f"Name: model_seed_{idx}",
            "item_content": "ignored",
        }
        labeling._update_stratum(seed_ex)
    assert len(labeling._stratum_counts) == labeling._MAX_STRATA
    # Drive early_ex's count up; it remains in the dict because it was
    # already inserted, so further updates increment it.
    for _ in range(49):
        labeling._update_stratum(early_ex)
    assert labeling._stratum_counts[labeling._stratum_key(early_ex)] == 50

    overflow_ex = {
        "benchmark": "bench_overflow",
        "condition": "cond_overflow",
        "subject_content": "Name: model_overflow",
        "item_content": "ignored",
    }
    assert labeling._stratum_key(overflow_ex) not in labeling._stratum_counts
    early_bonus = labeling._metadata_bonus(early_ex)
    overflow_bonus = labeling._metadata_bonus(overflow_ex)
    assert overflow_bonus <= early_bonus, (
        f"overflow stratum bonus ({overflow_bonus}) outranks frequently-seen "
        f"early stratum bonus ({early_bonus}); late candidates win by accident"
    )
