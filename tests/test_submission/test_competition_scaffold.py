# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Regression tests for the CAIMIRA competition scaffold.

These tests stay offline: they exercise helper seams and stdlib-only
acquisition behavior without loading the real encoder or training artifacts.
"""

from __future__ import annotations

import importlib
import math
import socket
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = REPO_ROOT / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))


def test_training_item_key_includes_benchmark_condition_and_content():
    train = importlib.import_module("train")
    examples = [
        {
            "benchmark": "bench_a",
            "condition": "cot",
            "subject_name": "model",
            "item_content": "shared item",
            "label": 1.0,
        },
        {
            "benchmark": "bench_b",
            "condition": "direct",
            "subject_name": "model",
            "item_content": "shared item",
            "label": 0.0,
        },
    ]

    _subject_to_idx, item_to_idx, ordered_items = train.build_indices(examples)
    assert len(item_to_idx) == 2
    assert ordered_items == [
        "Benchmark: bench_a\nCondition: cot\nItem:\nshared item",
        "Benchmark: bench_b\nCondition: direct\nItem:\nshared item",
    ]


def test_training_long_form_uses_composite_item_key():
    train = importlib.import_module("train")
    examples = [
        {"benchmark": "b", "condition": "c1", "subject_name": "s", "item_content": "x", "label": 1.0},
        {"benchmark": "b", "condition": "c2", "subject_name": "s", "item_content": "x", "label": 0.0},
    ]
    subject_to_idx, item_to_idx, _ordered_items = train.build_indices(examples)

    _s, item_idx, _y = train.build_long_form(examples, subject_to_idx, item_to_idx)
    assert item_idx.tolist() == [0, 1]


def test_model_renders_benchmark_condition_item_text_for_encoder_cache(monkeypatch):
    monkeypatch.setenv("PREDICTIVE_EVAL_LOCAL_SMOKE_TEST", "1")
    model = importlib.reload(importlib.import_module("model"))

    example = {"benchmark": "mmlu", "condition": "cot", "item_content": "What is 2+2?"}
    assert model._render_item_text(example) == "Benchmark: mmlu\nCondition: cot\nItem:\nWhat is 2+2?"


def test_encoder_loader_requests_local_files_only(monkeypatch):
    monkeypatch.setenv("PREDICTIVE_EVAL_LOCAL_SMOKE_TEST", "1")
    model = importlib.reload(importlib.import_module("model"))
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, repo: str, **kwargs):
            calls.append((repo, kwargs))

    encoder = model._load_encoder(FakeSentenceTransformer, device="cpu")

    assert isinstance(encoder, FakeSentenceTransformer)
    assert calls == [
        (
            model.ENCODER_REPO,
            {
                "device": "cpu",
                "local_files_only": True,
            },
        )
    ]


def test_local_smoke_import_and_predict_do_not_need_network(monkeypatch):
    monkeypatch.setenv("PREDICTIVE_EVAL_LOCAL_SMOKE_TEST", "1")

    def fail_network(*_args, **_kwargs):
        raise AssertionError("network access is forbidden in local smoke mode")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    model = importlib.reload(importlib.import_module("model"))

    out = model.predict(
        {
            "benchmark": "mmlu",
            "condition": "direct",
            "subject_content": "Name: model",
            "item_content": "Question?",
        },
        labeled=None,
    )
    assert out == 0.5


def test_adaptive_calibration_fits_on_base_predictor_logits(monkeypatch):
    monkeypatch.setenv("PREDICTIVE_EVAL_LOCAL_SMOKE_TEST", "1")
    model = importlib.reload(importlib.import_module("model"))
    labeled = [
        {"benchmark": "b1", "label": 1.0, "item_content": "a"},
        {"benchmark": "b1", "label": 0.0, "item_content": "b"},
    ]
    seen = []

    def base_predictor(row):
        seen.append(row["item_content"])
        return 0.2

    platt = model._fit_base_logit_platt(labeled, base_predictor)

    assert seen == ["a", "b"]
    assert "b1" in platt
    slope, intercept = platt["b1"]
    assert slope == 1.0
    assert intercept > 0.0


def test_acquisition_scores_have_1000_row_distribution():
    labeling = importlib.reload(importlib.import_module("labeling"))
    scores = [
        labeling.acquisition_function(
            {
                "benchmark": f"bench_{idx % 17}",
                "condition": f"cond_{idx % 5}",
                "subject_content": f"Name: model-{idx % 41}",
                "item_content": f"Question text {idx} with token {idx % 97}",
            }
        )
        for idx in range(1000)
    ]

    assert all(math.isfinite(score) for score in scores)
    assert min(scores) > 0.0
    assert float(np.std(scores)) > 0.01
    assert len({round(score, 4) for score in scores}) > 100
