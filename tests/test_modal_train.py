# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Regression tests for the Modal artifact-factory CLI/entrypoint resolution.

The ``main`` Modal entrypoint originally declared ``smoke: bool = True`` and
only forced ``smoke = True`` when ``kind == "smoke"`` — there was no
``else`` branch, so ``--kind train`` silently ran in smoke mode. The
documented example
``modal run modal_train.py::main --kind train --epochs 100`` would have
produced a smoke artifact under that signature. This test pins the
resolution to: ``kind="smoke"`` forces ``smoke=True`` and caps epochs at 5;
``kind="train"`` preserves the caller's explicit ``smoke`` value.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def modal_train():
    return importlib.import_module("modal_train")


@pytest.mark.parametrize(
    ("kind", "smoke_in", "epochs_in", "smoke_out", "epochs_out"),
    [
        # kind=smoke forces smoke True and caps epochs at 5.
        ("smoke", False, 5, True, 5),
        ("smoke", False, 100, True, 5),
        ("smoke", True, 5, True, 5),
        ("smoke", True, 100, True, 5),
        # kind=train preserves the caller's smoke; epochs unchanged.
        ("train", False, 100, False, 100),
        ("train", False, 5, False, 5),
        ("train", True, 50, True, 50),  # explicit user override
    ],
)
def test_resolve_smoke_and_epochs(modal_train, kind, smoke_in, epochs_in, smoke_out, epochs_out):
    assert modal_train._resolve_smoke_and_epochs(kind, smoke_in, epochs_in) == (smoke_out, epochs_out)


def test_cli_train_kind_does_not_force_smoke(modal_train):
    """End-to-end: ``--kind train`` (no --smoke) must produce a non-smoke plan."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = modal_train.cli(["--kind", "train", "--epochs", "100", "--dry-run"])
    assert rc == 0
    plan = json.loads(
        buf.getvalue().splitlines()[0] if "{" not in buf.getvalue()[0] else buf.getvalue().split("Local CLI")[0]
    )
    assert plan["kind"] == "train"
    assert plan["smoke"] is False
    assert plan["epochs"] == 100


def test_cli_train_kind_honors_explicit_smoke(modal_train):
    """``--kind train --smoke`` should run in smoke mode (explicit override)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = modal_train.cli(["--kind", "train", "--smoke", "--epochs", "100", "--dry-run"])
    assert rc == 0
    plan = json.loads(buf.getvalue().split("Local CLI")[0])
    assert plan["kind"] == "train"
    assert plan["smoke"] is True


def test_cli_smoke_kind_forces_smoke_and_caps_epochs(modal_train):
    """``--kind smoke --epochs 100`` must clamp to smoke + epochs<=5."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = modal_train.cli(["--kind", "smoke", "--epochs", "100", "--dry-run"])
    assert rc == 0
    plan = json.loads(buf.getvalue().split("Local CLI")[0])
    assert plan["smoke"] is True
    assert plan["epochs"] == 5


def test_train_command_forwards_blend_lambdas(modal_train):  # pylint: disable=redefined-outer-name,protected-access
    """Remote training command must not drop the operator's blend-lambda plan."""
    command = modal_train._build_train_command(
        submission_dir=Path("/tmp/modal-output"),
        latent_dim=5,
        seed=7,
        epochs=100,
        lr=1e-3,
        smoke=False,
        benchmarks=["mmlupro.parquet", "ai2d_test.parquet"],
        blend_lambdas=[0.3, 0.6, 0.9],
    )

    assert command[-6:] == [
        "mmlupro.parquet",
        "ai2d_test.parquet",
        "--blend-lambdas",
        "0.3",
        "0.6",
        "0.9",
    ]


@pytest.mark.skipif(
    importlib.util.find_spec("modal") is None,
    reason="modal not installed; main entrypoint only defined when modal is importable",
)
def test_main_smoke_default_is_false(modal_train):
    """Without ``--smoke``, ``main`` must default to non-smoke so kind drives the plan."""
    main = modal_train.main
    main = getattr(getattr(main, "info", None), "raw_f", main)
    sig = inspect.signature(main)
    assert sig.parameters["smoke"].default is False, (
        f"main smoke default is {sig.parameters['smoke'].default}; "
        "must be False so --kind train does not silently produce smoke artifacts"
    )
