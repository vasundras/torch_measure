# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Regression test for ``submission.model._fit_base_logit_platt`` caching.

The function was being invoked on every ``predict()`` call when ``labeled``
was non-empty. With ~10,000 hidden items per round and ``len(labeled) = K``
labeled rows, the predict loop spent O(N * K) extra base-predictor calls
just refitting the same Platt shifts. Each base-predictor call runs a
CAIMIRA forward pass + EB lookup, so the wasted cost is non-trivial.

The sibling implementations
(:meth:`torch_measure.models.cold_start_lookup.ColdStartLookupPredictor.calibrate`,
:meth:`submission.caimira_lite.EBLookup.fit_platt`) both cache by
``len(labeled)``; this test pins the same behavior on the module-level
``_fit_base_logit_platt``.
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
def _local_smoke_env(monkeypatch):
    monkeypatch.setenv("PREDICTIVE_EVAL_LOCAL_SMOKE_TEST", "1")


def _labeled() -> list[dict]:
    return [
        {
            "benchmark": "b1",
            "condition": "x",
            "subject_content": f"Name: m{idx}",
            "item_content": f"q{idx}",
            "label": 1.0 if idx % 2 == 0 else 0.0,
        }
        for idx in range(8)
    ]


def test_fit_base_logit_platt_caches_across_repeated_calls():
    model = importlib.reload(importlib.import_module("model"))
    counter = {"n": 0}

    def counting(row):
        counter["n"] += 1
        return 0.5

    labeled = _labeled()
    p1 = model._fit_base_logit_platt(labeled, counting)
    first_calls = counter["n"]
    assert first_calls == len(labeled), (
        f"expected {len(labeled)} base calls on cold cache, got {first_calls}"
    )

    p2 = model._fit_base_logit_platt(labeled, counting)
    second_calls = counter["n"] - first_calls
    assert second_calls == 0, (
        f"expected cache hit (0 new base calls); got {second_calls} — "
        "every Codabench predict() call would refit Platt for every hidden item"
    )
    assert p1 == p2


def test_fit_base_logit_platt_cache_misses_when_labeled_length_changes():
    model = importlib.reload(importlib.import_module("model"))
    counter = {"n": 0}

    def counting(row):
        counter["n"] += 1
        return 0.5

    short = _labeled()[:4]
    long_ = _labeled()
    model._fit_base_logit_platt(short, counting)
    after_short = counter["n"]
    assert after_short == len(short)

    model._fit_base_logit_platt(long_, counting)
    after_long = counter["n"] - after_short
    assert after_long == len(long_), (
        f"different labeled length should invalidate cache; got {after_long} new calls"
    )


def test_initialize_runtime_resets_platt_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("PREDICTIVE_EVAL_LOCAL_SMOKE_TEST", raising=False)
    import json

    import torch
    from caimira_lite import CAIMIRALite

    n_subjects, n_items, embed_dim, latent_dim = 1, 1, 768, 5
    head = CAIMIRALite(
        n_subjects=n_subjects,
        n_items=n_items,
        embedding_dim=embed_dim,
        latent_dim=latent_dim,
    )
    head_path = tmp_path / "caimira_lite.pt"
    meta_path = tmp_path / "caimira_lite.meta.json"
    eb_path = tmp_path / "eb_tables.json"
    torch.save(head.state_dict(), head_path)
    meta_path.write_text(
        json.dumps(
            {
                "n_subjects": n_subjects,
                "n_items": n_items,
                "embed_dim": embed_dim,
                "latent_dim": latent_dim,
                "subject_to_idx": {"m": 0},
            }
        )
    )
    eb_path.write_text(
        json.dumps(
            {
                "sbc": {},
                "sb": {},
                "subj": {"m": 0.5},
                "bench": {"b1": 0.5},
                "global": 0.5,
                "name_aliases": {},
                "name_lc": {},
            }
        )
    )

    class FakeEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, *_args, **_kwargs):
            return torch.zeros(embed_dim)

    model = importlib.reload(importlib.import_module("model"))
    model._initialize_runtime(
        head_path=head_path,
        meta_path=meta_path,
        eb_path=eb_path,
        encoder_factory=FakeEncoder,
        device=torch.device("cpu"),
    )
    # Prime the cache.
    counter = {"n": 0}

    def counting(row):
        counter["n"] += 1
        return 0.5

    labeled = _labeled()
    model._fit_base_logit_platt(labeled, counting)
    primed = counter["n"]
    assert primed == len(labeled)

    # Re-init: cache should be cleared so the next call recomputes.
    model._initialize_runtime(
        head_path=head_path,
        meta_path=meta_path,
        eb_path=eb_path,
        encoder_factory=FakeEncoder,
        device=torch.device("cpu"),
    )
    counter["n"] = 0
    model._fit_base_logit_platt(labeled, counting)
    after_reset = counter["n"]
    assert after_reset == len(labeled), (
        f"_initialize_runtime did not clear the Platt cache; got {after_reset} calls"
    )
