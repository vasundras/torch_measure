# Copyright (c) 2026 AIMS Foundations. MIT License.

"""End-to-end smoke for submission/train.py.

Runs ``python submission/train.py --smoke`` via subprocess in a tmp output
directory and verifies the three artifacts (``caimira_lite.pt``,
``caimira_lite.meta.json``, ``eb_tables.json``) round-trip through the
shippable :class:`submission.caimira_lite.CAIMIRALite` class.

Marked ``slow`` and ``network`` because the smoke run pulls the pinned
``aims-foundations/measurement-db`` parquet plus loads the encoder
``sentence-transformers/all-mpnet-base-v2``. Excluded from the default CI
loop (``-m "not slow and not network"``) and exercised only in the slow
nightly workflow when network egress is available.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = REPO_ROOT / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))


@pytest.mark.slow
@pytest.mark.network
def test_train_smoke_produces_loadable_artifacts(tmp_path):
    out_dir = tmp_path / "submission_out"
    out_dir.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(SUBMISSION_DIR / "train.py"),
            "--smoke",
            "--output-dir",
            str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"train.py --smoke failed with exit code {result.returncode}:\n"
        f"stdout=\n{result.stdout}\nstderr=\n{result.stderr}"
    )

    head_path = out_dir / "caimira_lite.pt"
    meta_path = out_dir / "caimira_lite.meta.json"
    eb_path = out_dir / "eb_tables.json"
    assert head_path.exists(), "caimira_lite.pt was not produced"
    assert meta_path.exists(), "caimira_lite.meta.json was not produced"
    assert eb_path.exists(), "eb_tables.json was not produced"

    meta = json.loads(meta_path.read_text())
    assert {"n_subjects", "n_items", "embed_dim", "latent_dim", "subject_to_idx"} <= meta.keys()

    from caimira_lite import CAIMIRALite

    head = CAIMIRALite(
        n_subjects=int(meta["n_subjects"]),
        n_items=int(meta["n_items"]),
        embedding_dim=int(meta.get("embed_dim", 768)),
        latent_dim=int(meta.get("latent_dim", 5)),
    )
    state = torch.load(head_path, map_location="cpu", weights_only=True)
    missing, unexpected = head.load_state_dict(state, strict=True)
    assert missing == [] and unexpected == []
    head.eval()
    logit = head.caimira_logit(0, torch.zeros(int(meta.get("embed_dim", 768))))
    assert isinstance(logit, float)
    assert -1e3 < logit < 1e3

    eb = json.loads(eb_path.read_text())
    required_eb_keys = {"sbc", "sb", "subj", "bench", "global", "name_aliases", "name_lc"}
    assert required_eb_keys <= eb.keys(), f"eb_tables.json missing keys: {required_eb_keys - eb.keys()}"
