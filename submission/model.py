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
This module deliberately has NO ``try/except → 0.5`` blocks. Module-init
failures (import error, missing artifact, ``torch.load`` failure with
``weights_only=True``) fall into Codabench's GENERIC fallback tier —
"No additional details are safe to show" — so the failure is loud (the
submission errors) but not deeply diagnosed by the platform. Only
``predict()`` output-shape violations (NaN/inf, non-float, out-of-range)
surface as the specific ``[PAIEC-PREDICT-002]`` code. See
``../docs/solutions/design-patterns/codabench-two-tier-error-reporting-paiec-system-2026-05-19.md``
for the full two-tier taxonomy. Silent swallowing would degrade the
submission to the constant-0.5 leaderboard signature (-0.69 NLL / 0.50
AUC), the canonical "model is broken" symptom that took 3 weeks to
diagnose in the April 2026 NCF silent-load-failure post-mortem.

Hybrid prediction
-----------------
For in-vocabulary ``(subject_name, benchmark, condition)`` triples we
blend the CAIMIRA logit with the EB-fallback logit by a fixed weight
``lambda_=0.6`` (CAIMIRA-leaning). For OUT-of-vocabulary subjects (CAIMIRA
has no trained skill vector), we use the EB fallback alone — which is
the ``cold_start_lookup``-shaped 6-level hierarchy. Per-benchmark
intercept-only Platt calibration is fit on the actual hybrid base
predictor logits for the platform's K=5 ``labeled`` reveals, not on EB-only
logits.

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
import math
import os
import sys
from collections import OrderedDict
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
# Default 0.6 (conservative per Wave-2 Lane-B recommendation), but the trainer
# can override via ``META["blend_lambda"]`` so D-9 ablation candidates
# ``{0.3, 0.5, 0.7, 0.9}`` can ship without code edits. ``_initialize_runtime``
# rebinds this module global when META is loaded.
_BLEND_LAMBDA: float = 0.6

# Artifact paths (resolved relative to this file's directory).
_HEAD_PATH = Path(_SUBMISSION_DIR) / "caimira_lite.pt"
_META_PATH = Path(_SUBMISSION_DIR) / "caimira_lite.meta.json"
_EB_PATH = Path(_SUBMISSION_DIR) / "eb_tables.json"

# Encoder repo — must also appear in submission/models.txt so the
# hosted runtime pre-fetches it into the HF cache.
ENCODER_REPO = "sentence-transformers/all-mpnet-base-v2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Env-configurable LRU bound for the per-round item-embedding cache. Default
# 8192 = ~24 MB at fp32x768 (well under any plausible container budget);
# operators can shrink via ``PEVAL_MAX_ITEM_CACHE=512`` for memory-constrained
# debug runs. The test-time monkey-patch ``mod._MAX_ITEM_CACHE = N`` continues
# to work because the module attribute is an int rebound at the call site.
_MAX_ITEM_CACHE = int(os.environ.get("PEVAL_MAX_ITEM_CACHE", "8192"))


def _logit(p: float) -> float:
    """Numerically safe logit; mirrors caimira_lite._logit.

    Non-finite input raises ``ValueError`` rather than silently collapsing
    to ``±16.118`` via the ``max/min`` ladder's NaN semantics.
    """
    if not math.isfinite(p):
        raise ValueError(f"_logit requires finite input, got {p!r}")
    p = max(1e-7, min(1 - 1e-7, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid; mirrors caimira_lite._sigmoid."""
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

# Per-round Platt-calibration cache; hoisted here so _initialize_runtime() can
# safely call ``_PLATT_CACHE.clear()`` during module import on a fresh Codabench
# container (LEGB lookup needs the binding to exist before the call site runs;
# the pre-Lane-B layout declared this AFTER _initialize_runtime() was called,
# which raised ``NameError: name '_PLATT_CACHE' is not defined`` and is the
# Codabench-fatal B7 bug). Keyed by ``len(labeled)``; values are the
# per-benchmark intercept-only Platt shifts fit against
# ``_base_predict_probability`` (the same base predict() walk at apply time,
# so fit-base and apply-base are consistent).
_PLATT_CACHE: dict[int, dict[str, tuple[float, float]]] = {}


def _load_encoder(encoder_cls: Any, device: str) -> Any:
    """Load the prefetched encoder without allowing runtime network access."""
    return encoder_cls(
        ENCODER_REPO,
        device=device,
        local_files_only=True,
    )


def _initialize_runtime(
    head_path: Path = _HEAD_PATH,
    meta_path: Path = _META_PATH,
    eb_path: Path = _EB_PATH,
    encoder_factory: Any = None,
    device: torch.device | None = None,
) -> None:
    """Load CAIMIRA + EB + encoder into module globals.

    Called automatically at module import unless ``PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1``.
    Tests can re-invoke with custom paths + a fake encoder factory to exercise
    the production prediction path without touching the real artifact files
    or the SentenceTransformer cache.
    """
    global DEVICE, META, CAIMIRA, EB, SUBJECT_TO_IDX, ENCODER, _item_cache, _BLEND_LAMBDA  # noqa: PLW0603

    _PLATT_CACHE.clear()
    runtime_device = torch.device(device) if device is not None else torch.device(DEVICE)
    DEVICE = runtime_device

    META = json.loads(Path(meta_path).read_text())
    n_subjects = int(META["n_subjects"])
    n_items = int(META["n_items"])
    embed_dim = int(META.get("embed_dim", EMBED_DIM))
    latent_dim = int(META.get("latent_dim", 5))
    # Trainer can ship per-artifact blend lambda for D-9 ablation (override
    # the conservative 0.6 default at the module-global). The module-level
    # ``_BLEND_LAMBDA`` was defined before this call site, so falls back to
    # 0.6 if META omits the key (older artifacts predating the meta-keyed
    # blend behavior remain compatible).
    _BLEND_LAMBDA = float(META.get("blend_lambda", _BLEND_LAMBDA))

    head = CAIMIRALite(
        n_subjects=n_subjects,
        n_items=n_items,
        embedding_dim=embed_dim,
        latent_dim=latent_dim,
    ).to(DEVICE)
    state = torch.load(Path(head_path), map_location=DEVICE, weights_only=True)
    head.load_state_dict(state)
    head.eval()
    for param in head.parameters():
        param.requires_grad_(False)
    CAIMIRA = head

    EB = EBLookup.from_json(eb_path)
    SUBJECT_TO_IDX = dict(META.get("subject_to_idx", {}))

    if encoder_factory is None:
        from sentence_transformers import SentenceTransformer

        encoder_factory = SentenceTransformer
    ENCODER = _load_encoder(encoder_factory, device=str(DEVICE))
    _item_cache = OrderedDict()


_item_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

if _LOCAL_SMOKE:
    print("[submission/model] PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1 — skipping artifact loads")
else:
    _initialize_runtime()


def _render_item_text(ex: dict) -> str:
    """Render the exact Benchmark/Condition/Item template used in training."""
    benchmark = (ex.get("benchmark") or "").strip()
    condition = (ex.get("condition") or "none").strip() or "none"
    item_content = ex.get("item_content") or ""
    return f"Benchmark: {benchmark}\nCondition: {condition}\nItem:\n{item_content}"


def _encode_item(ex: dict) -> torch.Tensor:
    """Encode one rendered item once per round (module-level cache)."""
    assert ENCODER is not None, "Encoder not loaded; called in LOCAL_SMOKE mode?"
    item_text = _render_item_text(ex)
    cached = _item_cache.get(item_text)
    if cached is not None:
        _item_cache.move_to_end(item_text)
        return cached
    vec = ENCODER.encode(
        item_text,
        convert_to_numpy=False,
        convert_to_tensor=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    # Anchor the embedding's target device on CAIMIRA's actual parameter
    # device, not the module-global ``DEVICE``. ``DEVICE`` can go stale when
    # ``_initialize_runtime`` is called with an explicit ``device=`` after the
    # initial module-import value has already been baked in (e.g. a CUDA-state
    # artifact loaded onto CPU, or a test stub). Falls back to ``DEVICE`` only
    # when CAIMIRA is unloaded (LOCAL_SMOKE-mode tests that monkey-patch
    # ENCODER without setting CAIMIRA); production always has CAIMIRA loaded.
    target_device = next(CAIMIRA.parameters()).device if CAIMIRA is not None else DEVICE
    vec = vec.to(target_device).float().view(-1)
    _item_cache[item_text] = vec
    _item_cache.move_to_end(item_text)
    if len(_item_cache) > _MAX_ITEM_CACHE:
        _item_cache.popitem(last=False)
    return vec


def _base_predict_probability(ex: dict) -> float:
    """Return the uncalibrated CAIMIRA+EB base probability for one row."""
    assert CAIMIRA is not None and EB is not None  # noqa: S101

    benchmark = (ex.get("benchmark") or "").strip()
    condition = (ex.get("condition") or "none").strip() or "none"
    subject_content = ex.get("subject_content") or ""
    item_content = ex.get("item_content") or ""

    raw_name = parse_subject_name(subject_content)
    subj_name = EB.resolve_name(raw_name)
    eb_p = EB.lookup_p(subj_name, benchmark, condition)

    subject_idx = SUBJECT_TO_IDX.get(subj_name)
    if subject_idx is None:
        subject_idx = SUBJECT_TO_IDX.get(raw_name)

    if subject_idx is None or not item_content:
        return eb_p

    item_embedding = _encode_item(ex)
    caimira_logit = CAIMIRA.caimira_logit(subject_idx, item_embedding)
    caimira_p = _sigmoid(caimira_logit)
    return _blend_logits(caimira_p, eb_p, _BLEND_LAMBDA)


def _fit_base_logit_platt(
    labeled: list[dict],
    base_predictor: Any,
) -> dict[str, tuple[float, float]]:
    """Fit intercept-only Platt shifts on actual base predictor logits.

    **Codabench-only optimization.** Cached by ``len(labeled)`` to avoid
    recomputing the Platt fit on every ``predict()`` call in the round. The
    platform sends the same labeled list throughout a round; recomputing the
    fit per hidden item is wasted work and (more importantly) triggers
    ``base_predictor`` per labeled row per hidden item — a CAIMIRA forward
    + EB lookup blow-up at ~10,000 hidden items per Ed #160 staff reply
    2026-05-21.

    The sibling :meth:`submission.caimira_lite.EBLookup.fit_platt` carries
    the same ``len(labeled)``-key cache for the same reason.

    **Library classes MUST NOT use this pattern.** The upstream
    :meth:`torch_measure.models.cold_start_lookup.ColdStartLookupPredictor.calibrate`
    no longer caches by ``len(labeled)`` (the previous fork-only cache was
    removed in the Lane B / PR #2 v2 round). Library callers may pass
    distinct labeled lists of equal length within a single process; the
    ``len``-key cache pattern only works on the Codabench-specific
    guarantee that the platform sends the same ``labeled`` throughout a
    round. The cache here is cleared by :func:`_initialize_runtime` so
    test re-inits do not leak stale entries.
    """
    cache_key = len(labeled)
    cached = _PLATT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    by_benchmark: dict[str, list[tuple[float, float]]] = {}
    for row in labeled:
        benchmark = (row.get("benchmark") or "").strip()
        if not benchmark:
            continue
        try:
            label = float(row["label"])
        except (KeyError, TypeError, ValueError):
            continue
        if label not in (0.0, 1.0):
            continue
        base_p = float(base_predictor(row))
        by_benchmark.setdefault(benchmark, []).append((_logit(base_p), label))

    platt: dict[str, tuple[float, float]] = {}
    for benchmark, pairs in by_benchmark.items():
        if not pairs:
            continue
        mean_y = sum(label for _x, label in pairs) / len(pairs)
        if mean_y <= 0.0 or mean_y >= 1.0:
            continue
        mean_x = sum(x for x, _label in pairs) / len(pairs)
        intercept = max(-1.5, min(1.5, _logit(mean_y) - mean_x))
        platt[benchmark] = (1.0, intercept)
    _PLATT_CACHE[cache_key] = platt
    return platt


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

    benchmark = (input.get("benchmark") or "").strip()
    base_p = _base_predict_probability(input)

    if labeled:
        platt = _fit_base_logit_platt(labeled, _base_predict_probability)
        if benchmark in platt:
            slope, intercept = platt[benchmark]
            base_p = _sigmoid(slope * _logit(base_p) + intercept)

    return clip_for_predict(base_p)
