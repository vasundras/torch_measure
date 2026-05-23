# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Offline trainer for the CAIMIRA+EB Codabench submission.

Produces the three artifacts that ``submission/model.py`` loads at module init:

* ``caimira_lite.pt`` — state_dict of :class:`submission.caimira_lite.CAIMIRALite`
* ``caimira_lite.meta.json`` — training provenance + ``subject_to_idx`` mapping
* ``eb_tables.json`` — the 6-level EB fallback hierarchy (``sbc / sb / subj /
  bench / global``) + ``name_aliases / name_lc``

Differences from :program:`python notebooks/train_caimira.py` (the
predictive-eval-competition trainer):

* This trainer runs FROM the ``torch_measure`` package (uses
  :class:`torch_measure.models.CAIMIRA` directly for training and
  :class:`torch_measure.models.ColdStartLookupPredictor`'s class for
  reference; the EB tables themselves are built here, not via a separate
  ``m3_build_colab.py`` script).
* Output paths default to ``submission/`` (sibling of this file), not
  ``predictive-eval-competition/submission/``.
* Adds explicit EB-table build (the parent-repo trainer doesn't, since
  the parent repo's submission relied on the standalone
  ``ColdStartLookupPredictor``'s own ``m3_build_colab.py``).

Smoke run (single benchmark, 5 epochs, ~30 seconds on CPU):

    python submission/train.py --smoke

Full run (all binary benchmarks, default 200 epochs):

    python submission/train.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, hf_hub_download
from sentence_transformers import SentenceTransformer

from torch_measure.models import CAIMIRA

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from caimira_lite import EMBED_DIM  # noqa: E402

REPO_ID = "aims-foundations/measurement-db"
REVISION = "589ccfdb8e82e6e0b5e35e9d23cd83a6df85018f"
ENCODER_REPO = "sentence-transformers/all-mpnet-base-v2"

REGISTRY_FILES = frozenset({"subjects.parquet", "items.parquet", "benchmarks.parquet"})

# Response-table columns we read (8-column schema per kit README).
# We deliberately ignore "trace" (large blob) and "correct_answer" (unused
# at training time) to keep memory bounded for smoke/full runs.
_RESPONSE_COLS = ("subject_id", "item_id", "benchmark_id", "test_condition", "response")

# EB-table build hyperparameters.
_EB_SBC_MIN_N = 3        # triple cell needs >= 3 obs to land in sbc
_EB_SUBJ_MIN_N = 10      # subject prior needs >= 10 obs
_EB_SB_ALPHA = 5.0       # Bayesian pseudo-counts toward IRT blend at level 2

# Probability clipping at table-build time. Avoids inf in logit().
_TABLE_CLIP_LO = 0.05
_TABLE_CLIP_HI = 0.95

DEFAULT_OUTPUT_DIR = _THIS_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true",
                   help="Single benchmark (mmlupro), 5 epochs, 1000-row cap.")
    p.add_argument("--benchmarks", nargs="*", default=None,
                   help="Subset of response parquets to train on.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--latent-dim", type=int, default=5,
                   help="CAIMIRA latent skill dimension; matches paper m=5.")
    p.add_argument("--blend-lambdas", type=float, nargs="*", default=[0.6],
                   help="Runtime CAIMIRA-vs-EB blend weights to carry into artifact metadata.")
    p.add_argument("--difficulty-reg", type=float, default=1e-4)
    p.add_argument("--skill-reg", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help=f"Where to write artifacts. Default {DEFAULT_OUTPUT_DIR}.")
    p.add_argument("--encode-batch", type=int, default=64)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_response_files(repo_id: str = REPO_ID, revision: str = REVISION) -> list[str]:
    files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    return sorted(
        n for n in files
        if n.endswith(".parquet") and n not in REGISTRY_FILES and not n.endswith("_traces.parquet")
    )


def _read_parquet_rows(filename: str, columns: tuple[str, ...] | None = None) -> list[dict]:
    """Download one parquet from the pinned HF revision and return row dicts.

    Uses ``huggingface_hub.hf_hub_download`` + ``pyarrow`` directly so we
    don't pull in the heavier ``datasets`` library (which is not in
    ``pyproject.toml``'s top-level deps).
    """
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        revision=REVISION,
    )
    table = pq.read_table(path, columns=list(columns) if columns else None)
    return table.to_pylist()


def load_responses(response_files: list[str]) -> list[dict]:
    """Concatenate the requested response parquets into one row list."""
    rows: list[dict] = []
    for fname in response_files:
        rows.extend(_read_parquet_rows(fname, columns=_RESPONSE_COLS))
    return rows


def load_registries() -> tuple[list[dict], list[dict], list[dict]]:
    subjects = _read_parquet_rows("subjects.parquet")
    items = _read_parquet_rows("items.parquet")
    benchmarks = _read_parquet_rows("benchmarks.parquet")
    return subjects, items, benchmarks


def collect_binary_examples(
    responses: list[dict],
    subjects_by_id: dict[str, dict],
    items_by_id: dict[str, dict],
    smoke: bool,
) -> list[dict[str, Any]]:
    """Filter to binary {0,1} rows + extract the visible-content fields."""
    if smoke:
        responses = responses[: min(1000, len(responses))]

    by_bench: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in responses:
        item = items_by_id.get(row["item_id"], {})
        subject = subjects_by_id.get(row["subject_id"], {})
        item_content = item.get("content")
        if not item_content:
            continue
        subject_display = subject.get("display_name") or row["subject_id"]
        subject_content = f"Name: {subject_display}"
        by_bench[row["benchmark_id"]].append({
            "benchmark": row["benchmark_id"],
            "condition": row["test_condition"] or "none",
            "subject_name": subject_display,
            "subject_content": subject_content,
            "item_content": item_content,
            "label": float(row["response"]),
        })

    examples: list[dict[str, Any]] = []
    for _bench, rows in by_bench.items():
        labels = [r["label"] for r in rows]
        is_binary = bool(labels) and all(v in (0.0, 1.0) for v in labels)
        if is_binary:
            examples.extend(rows)
    print(f"[train_caimira] Kept {len(examples):,} rows from binary benchmarks.")
    return examples


def render_item_text(example: dict[str, Any]) -> str:
    """Render the exact text embedded by training and runtime prediction."""
    benchmark = str(example.get("benchmark") or "").strip()
    condition = str(example.get("condition") or "none").strip() or "none"
    item_content = str(example.get("item_content") or "")
    return f"Benchmark: {benchmark}\nCondition: {condition}\nItem:\n{item_content}"


def build_indices(examples: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int], list[str]]:
    """Build subject and composite-item indices plus ordered rendered item texts."""
    subject_to_idx: dict[str, int] = {}
    item_to_idx: dict[str, int] = {}
    ordered_items: list[str] = []
    for ex in examples:
        if ex["subject_name"] not in subject_to_idx:
            subject_to_idx[ex["subject_name"]] = len(subject_to_idx)
        item_text = render_item_text(ex)
        if item_text not in item_to_idx:
            item_to_idx[item_text] = len(item_to_idx)
            ordered_items.append(item_text)
    return subject_to_idx, item_to_idx, ordered_items


def encode_items(texts: list[str], encoder: SentenceTransformer, batch: int) -> torch.Tensor:
    print(f"[train_caimira] Encoding {len(texts):,} unique items ...")
    matrix = encoder.encode(texts, batch_size=batch, show_progress_bar=True, convert_to_numpy=True)
    return torch.from_numpy(np.asarray(matrix, dtype=np.float32))


def build_long_form(
    examples: list[dict[str, Any]],
    subject_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(examples)
    s = np.empty(n, dtype=np.int64)
    i = np.empty(n, dtype=np.int64)
    y = np.empty(n, dtype=np.float32)
    for k, ex in enumerate(examples):
        s[k] = subject_to_idx[ex["subject_name"]]
        i[k] = item_to_idx[render_item_text(ex)]
        y[k] = float(ex["label"])
    return torch.from_numpy(s), torch.from_numpy(i), torch.from_numpy(y)


def _logit(p: float) -> float:
    p = max(1e-7, min(1 - 1e-7, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _table_clip(p: float) -> float:
    return max(_TABLE_CLIP_LO, min(_TABLE_CLIP_HI, float(p)))


def build_eb_tables(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the 6-level fallback lookup tables from binary training rows.

    Returns a dict matching :meth:`submission.caimira_lite.EBLookup.from_json`'s
    schema: ``{"sbc", "sb", "subj", "bench", "global", "name_aliases", "name_lc"}``.
    """
    print("[train_caimira] Building EB tables ...")
    sbc_acc: dict[str, list[float]] = defaultdict(list)
    sb_acc: dict[str, list[float]] = defaultdict(list)
    subj_acc: dict[str, list[float]] = defaultdict(list)
    bench_acc: dict[str, list[float]] = defaultdict(list)
    all_labels: list[float] = []

    for ex in examples:
        s = ex["subject_name"]
        b = ex["benchmark"]
        c = ex["condition"] or "none"
        y = ex["label"]
        sbc_acc[f"{s}||{b}||{c}"].append(y)
        sb_acc[f"{s}||{b}"].append(y)
        subj_acc[s].append(y)
        bench_acc[b].append(y)
        all_labels.append(y)

    global_mean = _table_clip(sum(all_labels) / max(len(all_labels), 1))
    bench = {b: _table_clip(sum(v) / len(v)) for b, v in bench_acc.items()}
    subj = {s: _table_clip(sum(v) / len(v)) for s, v in subj_acc.items() if len(v) >= _EB_SUBJ_MIN_N}
    sbc = {k: _table_clip(sum(v) / len(v)) for k, v in sbc_acc.items() if len(v) >= _EB_SBC_MIN_N}

    # Level 2: Bayesian shrinkage toward the IRT blend.
    sb: dict[str, float] = {}
    for key, ys in sb_acc.items():
        if not ys:
            continue
        n = len(ys)
        s_name, b_name = key.split("||", 1)
        raw_p = sum(ys) / n
        subj_p = subj.get(s_name)
        bench_p = bench.get(b_name)
        if subj_p is not None and bench_p is not None:
            prior_p = _sigmoid(_logit(subj_p) + _logit(bench_p) - _logit(global_mean))
        elif bench_p is not None:
            prior_p = bench_p
        else:
            prior_p = global_mean
        shrunk = (n * raw_p + _EB_SB_ALPHA * prior_p) / (n + _EB_SB_ALPHA)
        sb[key] = _table_clip(shrunk)

    # Name aliases + lowercase view for subject-name resolution.
    canonical_names = list(subj.keys())
    name_lc = {name.lower(): name for name in canonical_names}

    print(
        f"[train_caimira]   global_mean={global_mean:.4f}  "
        f"bench={len(bench)}  subj={len(subj)}  "
        f"sb={len(sb)}  sbc={len(sbc)}"
    )

    return {
        "sbc": sbc,
        "sb": sb,
        "subj": subj,
        "bench": bench,
        "global": global_mean,
        "name_aliases": {},
        "name_lc": name_lc,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[train_caimira] Device: {device}")

    response_files = (
        ["mmlupro.parquet"] if args.smoke
        else (args.benchmarks if args.benchmarks else list_response_files())
    )
    print(f"[train_caimira] Benchmarks: {response_files}")

    responses = load_responses(response_files)
    subjects, items, _benchmarks = load_registries()
    subjects_by_id = {row["subject_id"]: row for row in subjects}
    items_by_id = {row["item_id"]: row for row in items}

    examples = collect_binary_examples(responses, subjects_by_id, items_by_id, smoke=args.smoke)
    if not examples:
        raise SystemExit("[train_caimira] No binary examples — abort.")

    subject_to_idx, item_to_idx, ordered_items = build_indices(examples)
    print(
        f"[train_caimira] Indices: subjects={len(subject_to_idx):,}  items={len(item_to_idx):,}"
    )

    encoder = SentenceTransformer(ENCODER_REPO, device=str(device))
    embeddings = encode_items(ordered_items, encoder, args.encode_batch)
    print(f"[train_caimira] embeddings.shape={tuple(embeddings.shape)}")

    s_all, i_all, y_all = build_long_form(examples, subject_to_idx, item_to_idx)

    epochs = 5 if args.smoke else args.epochs

    # NOTE: we use torch_measure.models.CAIMIRA (the package version) for
    # TRAINING. The state_dict it produces loads cleanly into
    # caimira_lite.CAIMIRALite (the standalone predict-only sibling) at
    # submission runtime, because the two classes have identical
    # parameter names (skill, relevance_head.weight, relevance_head.bias,
    # difficulty_head.weight) and matching buffer (_difficulty_mean).
    model = CAIMIRA(
        n_subjects=len(subject_to_idx),
        n_items=len(item_to_idx),
        embedding_dim=embeddings.shape[1],
        latent_dim=args.latent_dim,
    )
    wide_response = _build_wide_response(
        s_all, i_all, y_all, len(subject_to_idx), len(item_to_idx)
    )
    history = model.fit(
        wide_response,
        embeddings=embeddings,
        max_epochs=epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        difficulty_reg=args.difficulty_reg,
        skill_reg=args.skill_reg,
        verbose=True,
    )

    # Build EB tables from the SAME binary training rows so the fallback
    # priors are consistent with what CAIMIRA optimized against.
    eb_tables = build_eb_tables(examples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pt_path = args.output_dir / "caimira_lite.pt"
    meta_path = args.output_dir / "caimira_lite.meta.json"
    eb_path = args.output_dir / "eb_tables.json"

    model_cpu = model.to("cpu").eval()
    torch.save(model_cpu.state_dict(), pt_path)
    print(f"[train_caimira] Saved CAIMIRA state_dict → {pt_path}")

    # Round-trip verification: load through the SHIPPABLE caimira_lite class
    # (not the upstream CAIMIRA) to catch silent state_dict-shape mismatches
    # between the trainer and the submission-runtime classes.
    from caimira_lite import CAIMIRALite  # noqa: E402,WPS433

    fresh = CAIMIRALite(
        n_subjects=len(subject_to_idx),
        n_items=len(item_to_idx),
        embedding_dim=embeddings.shape[1],
        latent_dim=args.latent_dim,
    )
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    missing, unexpected = fresh.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"[train_caimira] state_dict shape mismatch — missing={missing} unexpected={unexpected}"
        )
    print("[train_caimira]   round-trip into CAIMIRALite OK (no missing/unexpected keys)")

    eb_path.write_text(json.dumps(eb_tables, indent=2))
    print(f"[train_caimira] Saved EB tables → {eb_path}")

    metadata = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "encoder_repo": ENCODER_REPO,
        "embed_dim": EMBED_DIM,
        "latent_dim": args.latent_dim,
        "n_subjects": len(subject_to_idx),
        "n_items": len(item_to_idx),
        "subject_to_idx": subject_to_idx,
        "benchmarks": response_files,
        "item_template": "Benchmark: {benchmark}\nCondition: {condition}\nItem:\n{item_content}",
        "item_key": "benchmark||condition||item_content",
        "epochs": epochs,
        "lr": args.lr,
        "blend_lambdas": args.blend_lambdas,
        "weight_decay": args.weight_decay,
        "difficulty_reg": args.difficulty_reg,
        "skill_reg": args.skill_reg,
        "seed": args.seed,
        "smoke": args.smoke,
        "summary": {
            "final_train_loss": history["losses"][-1] if history.get("losses") else None,
            "n_examples": len(examples),
            "label_mean": float(np.mean([ex["label"] for ex in examples])),
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"[train_caimira] Saved metadata → {meta_path}")

    print("\n[train_caimira] DONE. To package the submission ZIP:")
    print("  bash submission/build_zip.sh")
    print("\nBefore uploading to Codabench: validate via the local smoke test")
    print("AND run the D-9 transfer-audit gate per llm_wiki/decisions/D9-*.md")
    print("(predictive-eval-competition repo). The 2026-05-22 transfer postmortem")
    print("documents that prior CAIMIRA m=5 submissions had best local val_nll")
    print("AND worst hidden score — D-9 is a hard prerequisite, not a formality.")


def _build_wide_response(
    s: torch.Tensor, i: torch.Tensor, y: torch.Tensor, n_subj: int, n_items: int
) -> torch.Tensor:
    """Long-form → wide-form (NaN for unobserved); only positional helper.

    CAIMIRA.fit accepts wide-form tensors per its
    :meth:`torch_measure.models._base.IRTModel.fit` contract.
    """
    wide = torch.full((n_subj, n_items), float("nan"))
    wide[s, i] = y
    return wide


if __name__ == "__main__":
    main()
