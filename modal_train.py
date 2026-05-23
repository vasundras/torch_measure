# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Modal artifact factory for the torch_measure CAIMIRA competition scaffold.

Modal is used only to produce artifacts under ``runs/modal/<run_id>/``. This
file must not read secrets, package production submissions, or submit to
Codabench. Device 1 remains responsible for local no-network gates, D-9, ZIP
packaging, and any upload decisions.

Typical use:

    modal run modal_train.py::main --kind smoke --latent-dim 5 --seed 42 --epochs 5
    modal run modal_train.py::main --kind train --latent-dim 5 --seed 7 --epochs 100

Local help works even when the Modal package is not installed:

    python modal_train.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover - exercised by operator envs without Modal.
    modal = None  # type: ignore[assignment]


APP_NAME = "torch-measure-caimira-modal-light"
ARTIFACT_VOLUME_NAME = "torch-measure-caimira-artifacts"
REPO_ROOT = Path(__file__).resolve().parent
LOCAL_ARTIFACT_ROOT = Path("runs/modal")
GPU_FALLBACK = ["H100", "A100-80GB", "L40S"]
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
FORBIDDEN_ENV_RE = re.compile(r"codabench", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_run_id(run_id: str) -> str:
    # SECURITY-REVIEW: run IDs become local and Modal Volume paths, so reject
    # separators/traversal and keep the alphabet narrow. Bare ``.`` and ``..``
    # are explicitly rejected because the regex below admits them (they pass
    # ``[A-Za-z0-9_.-]+`` since ``.`` is in the alphabet) but they resolve to
    # the parent / current directory rather than producing a new artifact tree.
    if run_id in {".", ".."}:
        raise ValueError(f"unsafe run_id: {run_id!r}")
    if not run_id or not RUN_ID_RE.fullmatch(run_id) or "/" in run_id or ".." in run_id:
        raise ValueError(f"unsafe run_id: {run_id!r}")
    return run_id


def assert_no_codabench_capability() -> None:
    """Refuse to run if Codabench credentials are visible in the environment."""
    leaked = [name for name, value in os.environ.items() if value and FORBIDDEN_ENV_RE.search(name)]
    if leaked:
        raise RuntimeError(
            "modal_train.py refuses to start with Codabench-related env vars "
            f"present: {sorted(leaked)}. This lane is artifact-only."
        )


def default_run_id(kind: str, latent_dim: int, seed: int, epochs: int) -> str:
    # Append 4-char hex suffix (~1/65k collision probability per same-second
    # invocation) so back-to-back smoke runs don't clobber each other's
    # ``runs/modal/<run_id>/`` tree. Underscore-separated to remain inside
    # the ``RUN_ID_RE = ^[A-Za-z0-9_.-]+$`` alphabet.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = os.urandom(2).hex()
    return f"modal_light_caimira_m{latent_dim}_seed{seed}_ep{epochs}_{kind}_{stamp}_{suffix}"


def _resolve_smoke_and_epochs(kind: str, smoke: bool, epochs: int) -> tuple[bool, int]:
    """Reconcile ``kind`` / ``smoke`` / ``epochs`` into a coherent plan.

    ``kind="smoke"`` forces ``smoke=True`` and caps ``epochs`` at 5; the
    smoke-kind contract is "fast plumbing check, ignore caller's training
    config". ``kind="train"`` preserves the caller's explicit ``smoke``
    value so an operator can still opt into ``--kind train --smoke`` if
    they want the train code path with a smoke dataset, but a plain
    ``--kind train`` does not silently downgrade to smoke mode.
    """
    if kind == "smoke":
        return True, min(epochs, 5)
    return smoke, epochs


def _build_train_command(
    submission_dir: Path,
    latent_dim: int,
    seed: int,
    epochs: int,
    lr: float,
    smoke: bool,
    benchmarks: list[str],
    blend_lambdas: list[float],
) -> list[str]:
    command = [
        sys.executable,
        "/root/submission/train.py",
        "--output-dir",
        str(submission_dir),
        "--latent-dim",
        str(latent_dim),
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
    ]
    if smoke:
        command.append("--smoke")
    if benchmarks:
        command.append("--benchmarks")
        command.extend(benchmarks)
    if blend_lambdas:
        command.append("--blend-lambdas")
        command.extend(str(lambda_) for lambda_ in blend_lambdas)
    return command


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact_manifest(run_dir: Path, run_id: str, config: dict[str, Any]) -> dict[str, Any]:
    files = []
    for fp in sorted(run_dir.rglob("*")):
        if not fp.is_file() or fp.name == "artifact_manifest.json":
            continue
        files.append(
            {
                "path": fp.relative_to(run_dir).as_posix(),
                "bytes": fp.stat().st_size,
                "sha256": sha256_file(fp),
            }
        )
    return {
        "run_id": run_id,
        "created_utc": utc_now(),
        "config": config,
        "artifact_files": files,
        "codabench_submission_allowed": False,
        "production_zip": False,
    }


def _modal_required() -> Any:
    if modal is None:
        raise RuntimeError(
            "Modal is not installed in this environment. Install/configure Modal "
            "before launching remote jobs; local help and dry-run remain available."
        )
    return modal


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch==2.4.1",
            "sentence-transformers==3.0.1",
            "transformers==4.44.2",
            "huggingface_hub==0.24.7",
            "pyarrow==17.0.0",
            "numpy==2.0.1",
        )
        .add_local_dir(REPO_ROOT / "src", remote_path="/root/src", copy=True)
        .add_local_dir(REPO_ROOT / "submission", remote_path="/root/submission", copy=True)
        .env({"PYTHONPATH": "/root/src:/root/submission", "PYTHONUNBUFFERED": "1"})
    )
    app = modal.App(APP_NAME)
    artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)
else:
    image = None
    app = None
    artifact_volume = None


def _remote_train_impl(
    run_id: str,
    kind: str,
    latent_dim: int,
    seed: int,
    epochs: int,
    lr: float,
    smoke: bool,
    benchmarks: list[str],
    blend_lambdas: list[float],
) -> dict[str, Any]:
    run_id = validate_run_id(run_id)
    started = utc_now()
    out_dir = Path("/artifacts") / run_id
    submission_dir = out_dir / "submission"
    out_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "kind": kind,
        "latent_dim": latent_dim,
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "smoke": smoke,
        "benchmarks": benchmarks,
        "blend_lambdas": blend_lambdas,
        "codabench_submission_allowed": False,
    }
    write_json(out_dir / "config.json", config)

    command = _build_train_command(
        submission_dir=submission_dir,
        latent_dim=latent_dim,
        seed=seed,
        epochs=epochs,
        lr=lr,
        smoke=smoke,
        benchmarks=benchmarks,
        blend_lambdas=blend_lambdas,
    )
    (out_dir / "command.txt").write_text(" ".join(shlex.quote(part) for part in command) + "\n", encoding="utf-8")

    start = time.time()
    proc = subprocess.run(
        command,
        cwd="/root",
        text=True,
        capture_output=True,
        check=False,
    )
    duration = time.time() - start
    (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")

    meta_path = submission_dir / "caimira_lite.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    metrics = {
        "status": "pass" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "duration_seconds": round(duration, 3),
        "final_train_loss": (meta.get("summary") or {}).get("final_train_loss"),
        "n_examples": (meta.get("summary") or {}).get("n_examples"),
        "label_mean": (meta.get("summary") or {}).get("label_mean"),
        "smoke": smoke,
        "item_holdout": {"nll": None, "auc": None},
        "distribution": None,
        "path_coverage": None,
    }
    write_json(out_dir / "metrics.json", metrics)

    notes = [
        f"# Modal light CAIMIRA artifact — {run_id}",
        "",
        f"- Kind: `{kind}`",
        f"- Latent dim: `{latent_dim}`",
        f"- Seed: `{seed}`",
        f"- Epochs: `{epochs}`",
        f"- Smoke: `{smoke}`",
        f"- Started UTC: `{started}`",
        f"- Return code: `{proc.returncode}`",
        "",
        "No Codabench package was produced. Device 1 must run local no-network",
        "gates and D-9 transfer audit before any submission decision.",
    ]
    (out_dir / "package_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    manifest = artifact_manifest(out_dir, run_id, config)
    write_json(out_dir / "artifact_manifest.json", manifest)
    if artifact_volume is not None:
        artifact_volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"{run_id} training failed with return code {proc.returncode}")
    return manifest


if modal is not None:

    @app.function(
        image=image,
        gpu=GPU_FALLBACK,
        timeout=3600,
        memory=32768,
        volumes={"/artifacts": artifact_volume},
    )
    def train_job(
        run_id: str,
        kind: str,
        latent_dim: int,
        seed: int,
        epochs: int,
        lr: float,
        smoke: bool,
        benchmarks: list[str],
        blend_lambdas: list[float],
    ) -> dict[str, Any]:
        return _remote_train_impl(
            run_id=run_id,
            kind=kind,
            latent_dim=latent_dim,
            seed=seed,
            epochs=epochs,
            lr=lr,
            smoke=smoke,
            benchmarks=benchmarks,
            blend_lambdas=blend_lambdas,
        )


def pull_volume_tree(run_id: str, local_root: Path = LOCAL_ARTIFACT_ROOT) -> int:
    run_id = validate_run_id(run_id)
    m = _modal_required()
    vol = m.Volume.from_name(ARTIFACT_VOLUME_NAME)
    vol.reload()
    local_dir = local_root / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    n_pulled = 0

    def walk(prefix: str) -> None:
        nonlocal n_pulled
        for entry in vol.iterdir(prefix):
            entry_type = (
                entry.type.name
                if hasattr(entry, "type") and hasattr(entry.type, "name")
                else str(getattr(entry, "type", "FILE"))
            ).upper()
            if "DIR" in entry_type:
                walk(entry.path)
                continue
            entry_parts = Path(str(entry.path).lstrip("/")).parts
            prefix_parts = Path(run_id).parts
            if entry_parts[: len(prefix_parts)] == prefix_parts:
                rel = Path(*entry_parts[len(prefix_parts) :])
            else:
                rel = Path(entry_parts[-1])
            target = local_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as fh:
                for chunk in vol.read_file(entry.path):
                    fh.write(chunk)
            n_pulled += 1

    walk(f"/{run_id}")
    return n_pulled


if modal is not None:

    @app.local_entrypoint()
    def main(
        kind: str = "smoke",
        run_id: str = "",
        latent_dim: int = 5,
        seed: int = 42,
        epochs: int = 5,
        lr: float = 1e-3,
        smoke: bool = False,
        benchmarks: str = "",
        blend_lambdas: str = "0.0,0.5,0.6,0.9,1.0",
        dry_run: bool = False,
        skip_pull: bool = False,
    ) -> None:
        assert_no_codabench_capability()
        if kind not in {"smoke", "train"}:
            raise ValueError("--kind must be smoke or train")
        smoke, epochs = _resolve_smoke_and_epochs(kind, smoke, epochs)
        run_id = validate_run_id(run_id or default_run_id(kind, latent_dim, seed, epochs))
        bench_list = [part for part in benchmarks.split(",") if part]
        lambda_list = [float(part) for part in blend_lambdas.split(",") if part]
        plan = {
            "run_id": run_id,
            "kind": kind,
            "latent_dim": latent_dim,
            "seed": seed,
            "epochs": epochs,
            "lr": lr,
            "smoke": smoke,
            "benchmarks": bench_list,
            "blend_lambdas": lambda_list,
            "artifact_volume": ARTIFACT_VOLUME_NAME,
            "local_artifact_dir": str(LOCAL_ARTIFACT_ROOT / run_id),
            "codabench_submission_allowed": False,
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        if dry_run:
            return
        manifest = train_job.remote(
            run_id=run_id,
            kind=kind,
            latent_dim=latent_dim,
            seed=seed,
            epochs=epochs,
            lr=lr,
            smoke=smoke,
            benchmarks=bench_list,
            blend_lambdas=lambda_list,
        )
        print(f"remote manifest files={len(manifest.get('artifact_files', []))}")
        if not skip_pull:
            n_pulled = pull_volume_tree(run_id)
            print(f"pulled {n_pulled} files to {LOCAL_ARTIFACT_ROOT / run_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", default="smoke", choices=["smoke", "train"])
    parser.add_argument("--run-id", default="")
    parser.add_argument("--latent-dim", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmarks", default="")
    parser.add_argument("--blend-lambdas", default="0.0,0.5,0.6,0.9,1.0")
    parser.add_argument(
        "--skip-pull",
        action="store_true",
        help=(
            "Don't pull artifacts from the Modal Volume after training. "
            "Local CLI only emits this in the dry-run plan; the actual "
            "skip-pull behaviour is honored by ``modal run modal_train.py::main "
            "--skip-pull`` (the Modal entrypoint)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assert_no_codabench_capability()
    kind = args.kind
    smoke, epochs = _resolve_smoke_and_epochs(kind, args.smoke, args.epochs)
    run_id = validate_run_id(args.run_id or default_run_id(kind, args.latent_dim, args.seed, epochs))
    plan = {
        "run_id": run_id,
        "kind": kind,
        "latent_dim": args.latent_dim,
        "seed": args.seed,
        "epochs": epochs,
        "lr": args.lr,
        "smoke": smoke,
        "benchmarks": [part for part in args.benchmarks.split(",") if part],
        "blend_lambdas": [float(part) for part in args.blend_lambdas.split(",") if part],
        "skip_pull": args.skip_pull,
        "modal_available": modal is not None,
        "codabench_submission_allowed": False,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.dry_run:
        print("Local CLI is dry-run only. Launch with: modal run modal_train.py::main ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
