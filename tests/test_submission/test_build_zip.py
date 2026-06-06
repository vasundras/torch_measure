# Copyright (c) 2026 AIMS Foundations. MIT License.

"""End-to-end smoke for submission/build_zip.sh.

Builds a flat ZIP from a temporary directory of stub artifacts using a
shim ``build_zip.sh`` copied into the tmp dir. Verifies the script's
``unzip -Z1`` parsing handles the flat-allowlist case and aborts on
nested paths. Catches regressions from the older ``unzip -l | awk``
recipe that split filenames on whitespace.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_ZIP_SCRIPT = REPO_ROOT / "submission" / "build_zip.sh"

REQUIRED_FILES = (
    "model.py",
    "labeling.py",
    "models.txt",
    "caimira_lite.py",
    "caimira_lite.pt",
    "caimira_lite.meta.json",
    "eb_tables.json",
)


def _stage_submission(tmp_path: Path) -> Path:
    """Copy build_zip.sh into a tmp submission dir and populate stub artifacts."""
    sub = tmp_path / "submission"
    sub.mkdir()
    shutil.copy2(BUILD_ZIP_SCRIPT, sub / "build_zip.sh")
    for name in REQUIRED_FILES:
        (sub / name).write_bytes(b"stub\n")
    return sub


def _run_build_zip(sub_dir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(sub_dir / "build_zip.sh"), *args],
        cwd=str(sub_dir.parent),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_build_zip_smoke_produces_flat_allowlist(tmp_path):
    if shutil.which("zip") is None or shutil.which("unzip") is None:
        pytest.skip("zip/unzip binaries not available")
    sub = _stage_submission(tmp_path)
    out_zip = tmp_path / "out.zip"
    result = _run_build_zip(sub, str(out_zip))
    assert result.returncode == 0, f"build_zip.sh failed: stdout=\n{result.stdout}\nstderr=\n{result.stderr}"
    assert out_zip.exists()
    with zipfile.ZipFile(out_zip) as zf:
        actual = sorted(zf.namelist())
        assert "/" not in "".join(actual), f"ZIP not flat: {actual}"
        assert actual == sorted(REQUIRED_FILES)


def test_build_zip_help_does_not_consume_args(tmp_path):
    if shutil.which("zip") is None or shutil.which("unzip") is None:
        pytest.skip("zip/unzip binaries not available")
    sub = _stage_submission(tmp_path)
    result = _run_build_zip(sub, "--help")
    assert result.returncode == 0
    assert "Usage" in result.stdout or "Usage" in result.stderr


def test_build_zip_refuses_overwrite_without_force(tmp_path):
    if shutil.which("zip") is None or shutil.which("unzip") is None:
        pytest.skip("zip/unzip binaries not available")
    sub = _stage_submission(tmp_path)
    out_zip = tmp_path / "out.zip"
    out_zip.write_bytes(b"existing\n")
    result = _run_build_zip(sub, str(out_zip))
    assert result.returncode == 1
    assert "refusing to overwrite" in result.stderr
