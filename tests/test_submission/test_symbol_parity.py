# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Parity test for the 6 helpers duplicated between caimira_lite.py and
cold_start_lookup.py.

The standalone-ZIP constraint justifies the duplication: the hosted Codabench
container ships an organizer-supplied ``torch_measure`` that may not track
``feat/caimira``, so ``submission/caimira_lite.py`` cannot import the
upstream helpers at runtime. This test catches silent drift between the
two copies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = REPO_ROOT / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))


@pytest.fixture(scope="module")
def lib():
    from torch_measure.models import cold_start_lookup

    return cold_start_lookup


@pytest.fixture(scope="module")
def ship():
    import importlib

    return importlib.import_module("submission.caimira_lite")


@pytest.mark.parametrize("p", [0.01, 0.1, 0.5, 0.9, 0.99, 1e-9, 1 - 1e-9])
def test_logit_parity(lib, ship, p):
    assert lib._logit(p) == ship._logit(p)


def test_logit_nan_parity(lib, ship):
    with pytest.raises(ValueError):
        lib._logit(float("nan"))
    with pytest.raises(ValueError):
        ship._logit(float("nan"))


@pytest.mark.parametrize("x", [-30.0, -3.0, -0.5, 0.0, 0.5, 3.0, 30.0])
def test_sigmoid_parity(lib, ship, x):
    assert lib._sigmoid(x) == ship._sigmoid(x)


@pytest.mark.parametrize("p", [-1.0, 0.0, 0.5, 0.97, 1.0, 5.0])
def test_clip_parity(lib, ship, p):
    assert lib._clip(p) == ship._clip(p)


def test_provider_prefixes_parity(lib, ship):
    assert lib._DEFAULT_PROVIDER_PREFIXES == ship._DEFAULT_PROVIDER_PREFIXES


@pytest.mark.parametrize(
    "subject_content",
    [
        "Name: gpt-4",
        "Name: meta-llama/Llama-2-7b-chat\nOrganization: Meta",
        "",
        "   ",
        "no_name_line",
        "Name: Claude-3-Opus\nOrganization: Anthropic\nParameters: 175B",
    ],
)
def test_parse_subject_name_parity(lib, ship, subject_content):
    assert lib.parse_subject_name(subject_content) == ship.parse_subject_name(subject_content)


@pytest.mark.parametrize(
    "raw_name",
    [
        "gpt-4",
        "GPT-4",
        "meta-llama/Llama-2-7b-chat",
        "openai/gpt-4",
        "unknown-model",
        "",
    ],
)
def test_resolve_subject_name_parity(lib, ship, raw_name):
    subj = {"gpt-4": 0.7, "Llama-2-7b-chat": 0.6}
    sb = {}
    sbc = {}
    aliases = {"some-prefix-name": "Canonical"}
    name_lc = {"gpt-4": "gpt-4", "llama-2-7b-chat": "Llama-2-7b-chat"}
    assert lib.resolve_subject_name(raw_name, subj, sb, sbc, aliases, name_lc) == ship.resolve_subject_name(
        raw_name, subj, sb, sbc, aliases, name_lc
    )
