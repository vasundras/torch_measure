# Copyright (c) 2026 AIMS Foundations. MIT License.
# pylint: disable=redefined-builtin
"""Codabench submission entry point — CAIMIRA + EB hybrid back-stop.

Contract surface
----------------
The hosted runtime imports this module once per container, then calls
:func:`predict` once per hidden item. Adaptive labeling is handled by
:mod:`submission.labeling`.

* ``predict(input: dict, labeled: list[dict] | None = None) -> float``
* The ``input`` dict has exactly four string keys: ``benchmark``,
  ``condition``, ``subject_content``, ``item_content``.
* The return MUST be a native Python finite float in ``[0, 1]`` (clipped
  here to ``[1e-4, 1-1e-4]`` to keep the log-loss bounded).

D-3 / no-broad-except discipline
--------------------------------
This module deliberately has NO ``try/except → 0.5`` blocks. If module
init fails (import error, missing artifact, ``torch.load`` failure with
``weights_only=True``), the platform surfaces a ``[PAIEC-PREDICT-002]``
error before any predictions run. If ``predict()`` raises mid-round, the
platform fails the submission — that is the correct behavior. Silent
swallowing degrades the submission to the constant-0.5 leaderboard
signature (-0.69 NLL / 0.50 AUC), which is the canonical "model is
broken" symptom that took 3 weeks to diagnose in the April 2026 NCF
silent-load-failure post-mortem.

Hybrid prediction
-----------------
For in-vocabulary ``(subject_name, benchmark, condition)`` triples we
blend the CAIMIRA logit with the EB-fallback logit by a fixed weight
``lambda_=0.6`` (CAIMIRA-leaning). For OUT-of-vocabulary subjects (CAIMIRA
has no trained skill vector), we use the EB fallback alone — which is
the ``cold_start_lookup``-shaped 6-level hierarchy + per-benchmark
intercept-only Platt calibration from the platform's K=5 ``labeled``
reveals.

NOT-yet-shipped
---------------
This file IS ``submission/model.py`` and CAN be packaged via
:program:`bash submission/build_zip.sh`, but the trained artifact
(``caimira_lite.pt`` + ``caimira_lite.meta.json`` + ``eb_tables.json``)
is NOT shipped in the repo (gitignored ``*.pt``; 500 KB pre-commit cap).
You must run :program:`python submission/train.py` to produce them
before packaging. See ``submission/README.md`` for the full flow.

A D-9 pre-submission transfer-audit gate must pass before any Codabench
upload — the 2026-05-22 transfer postmortem documents that the prior
CAIMIRA m=5 submissions had the BEST local val_nll AND the WORST hidden
score. This hybrid back-stop is a structural improvement over the
2026-05-22 ``<UNK>``-at-index-0 fallback, but it has NOT been validated
against a hidden leaderboard.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

_LOCAL_SMOKE = os.environ.get("PREDICTIVE_EVAL_LOCAL_SMOKE_TEST", "") == "1"

_SUBMISSION_DIR = str(Path(__file__).resolve().parent)
if _SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, _SUBMISSION_DIR)

from caimira_lite import (  # noqa: E402
    EMBED_DIM,
    CAIMIRALite,
    EBLookup,
    clip_for_predict,
    parse_subject_name,
)

# CAIMIRA-vs-EB blend weight in logit space. Higher = more CAIMIRA-leaning.
# Fixed at 0.6 as the conservative default per the Wave-2 Lane-B
# recommendation. D-9 ablation candidates: {0.3, 0.5, 0.7, 0.9}.
_BLEND_LAMBDA: float = 0.6

# Artifact paths (resolved relative to this file's directory).
_HEAD_PATH = Path(_SUBMISSION_DIR) / "caimira_lite.pt"
_META_PATH = Path(_SUBMISSION_DIR) / "caimira_lite.meta.json"
_EB_PATH = Path(_SUBMISSION_DIR) / "eb_tables.json"

# Encoder repo — must also appear in submission/models.txt so the
# hosted runtime pre-fetches it into the HF cache.
ENCODER_REPO = "sentence-transformers/all-mpnet-base-v2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _logit(p: float) -> float:
    """Numerically safe logit; mirrors caimira_lite._logit."""
    import math

    p = max(1e-7, min(1 - 1e-7, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid; mirrors caimira_lite._sigmoid."""
    import math

    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _blend_logits(caimira_p: float, eb_p: float, lambda_: float = _BLEND_LAMBDA) -> float:
    """Weighted-logit blend; ``lambda_=1`` is pure CAIMIRA, ``lambda_=0`` is pure EB."""
    return _sigmoid(lambda_ * _logit(caimira_p) + (1.0 - lambda_) * _logit(eb_p))


# ---------------------------------------------------------------------------
# Module-level loading.
#
# Local smoke test (PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1, set by the kit's
# tools/check_submission_zip.py and tools/run_smoke_test.py) skips the
# heavy loads so the contract test runs without the artifact or the
# encoder pre-fetched. The hosted Codabench container does NOT set this
# variable; production always runs the full load path.
# ---------------------------------------------------------------------------

ENCODER: Any | None = None
CAIMIRA: CAIMIRALite | None = None
EB: EBLookup | None = None
META: dict[str, Any] | None = None
SUBJECT_TO_IDX: dict[str, int] = {}

if _LOCAL_SMOKE:
    print("[submission/model] PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1 — skipping artifact loads")
else:
    from sentence_transformers import SentenceTransformer

    META = json.loads(_META_PATH.read_text())
    n_subjects = int(META["n_subjects"])
    n_items = int(META["n_items"])
    embed_dim = int(META.get("embed_dim", EMBED_DIM))
    latent_dim = int(META.get("latent_dim", 5))

    CAIMIRA = CAIMIRALite(
        n_subjects=n_subjects,
        n_items=n_items,
        embedding_dim=embed_dim,
        latent_dim=latent_dim,
    ).to(DEVICE)
    state = torch.load(_HEAD_PATH, map_location=DEVICE, weights_only=True)
    CAIMIRA.load_state_dict(state)
    CAIMIRA.eval()
    for p in CAIMIRA.parameters():
        p.requires_grad_(False)

    EB = EBLookup.from_json(_EB_PATH)

    SUBJECT_TO_IDX = dict(META.get("subject_to_idx", {}))

    ENCODER = SentenceTransformer(ENCODER_REPO, device=str(DEVICE))

_item_cache: dict[str, torch.Tensor] = {}


def _encode_item(item_content: str) -> torch.Tensor:
    """Encode ``item_content`` once per round (module-level cache)."""
    assert ENCODER is not None, "Encoder not loaded; called in LOCAL_SMOKE mode?"
    cached = _item_cache.get(item_content)
    if cached is not None:
        return cached
    vec = ENCODER.encode(
        item_content,
        convert_to_numpy=False,
        convert_to_tensor=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    vec = vec.to(DEVICE).float().view(-1)
    _item_cache[item_content] = vec
    return vec


def predict(
    input: dict,
    labeled: list[dict] | None = None,
) -> float:
    """Return ``P(correct)`` for the hidden ``(subject, item)`` triple.

    Contract per ``starting_kit/README.md:212-228``: native Python float in
    ``[0, 1]``, NaN/inf/tensors/strings/booleans rejected by the platform's
    pre-validator with the verbatim error
    ``Invalid predict() output: predict() must return a finite float in [0, 1].``
    The final clip ``[1e-4, 1-1e-4]`` matches the team's NCF and keeps
    log-loss bounded.

    For ``LOCAL_SMOKE`` mode the function returns a fixed ``0.5`` sentinel
    so the kit's local smoke test can exercise the import + signature
    surface without requiring the trained artifact or the encoder in the
    local HF cache. The hosted container does NOT set this env var.

    Parameters
    ----------
    input : dict
        Four string keys: ``benchmark`` (identifier, not display name),
        ``condition``, ``subject_content`` (starts with ``Name:``),
        ``item_content``.
    labeled : list[dict] or None
        Optional list of K=5-per-category revealed examples with the same
        four content keys plus a ``label`` field in ``{0, 1}``. May be
        ``None`` or ``[]`` for cold-start rounds.

    Returns
    -------
    float
        Predicted probability in ``[1e-4, 1-1e-4]``.
    """
    if _LOCAL_SMOKE:
        return 0.5

    assert CAIMIRA is not None and EB is not None and META is not None  # noqa: S101

    benchmark = (input.get("benchmark") or "").strip()
    condition = (input.get("condition") or "none").strip() or "none"
    subject_content = input.get("subject_content") or ""
    item_content = input.get("item_content") or ""

    raw_name = parse_subject_name(subject_content)
    subj_name = EB.resolve_name(raw_name)

    eb_p = EB.lookup_p(subj_name, benchmark, condition)

    subject_idx = SUBJECT_TO_IDX.get(subj_name)
    if subject_idx is None:
        subject_idx = SUBJECT_TO_IDX.get(raw_name)

    if subject_idx is None or not item_content:
        # Out-of-vocab subject OR missing item text → EB-only path.
        # Distinguishable from the silent-0.5 D-3 antipattern because eb_p
        # varies with (subject, benchmark, condition) — a leaderboard run
        # producing constant 0.5 has a different root cause than "EB
        # fallback is firing for every row".
        p = eb_p
    else:
        item_embedding = _encode_item(item_content)
        caimira_logit = CAIMIRA.caimira_logit(subject_idx, item_embedding)
        caimira_p = _sigmoid(caimira_logit)
        p = _blend_logits(caimira_p, eb_p, _BLEND_LAMBDA)

    if labeled:
        EB.fit_platt(labeled)
        if benchmark in EB._platt:  # noqa: SLF001 — same module's class
            slope, intercept = EB._platt[benchmark]  # noqa: SLF001
            p = _sigmoid(slope * _logit(p) + intercept)

    return clip_for_predict(p)
