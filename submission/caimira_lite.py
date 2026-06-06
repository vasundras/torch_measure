# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Self-contained CAIMIRA + EB cold-start fallback for Codabench submission.

This module is a SHIPPABLE STANDALONE port of two pieces of the
``torch_measure`` library:

* :class:`torch_measure.models.CAIMIRA` — the paper-faithful content-aware
  multidimensional IRT model (predict path only; the trainer still lives in
  :mod:`torch_measure.models.caimira`).
* :class:`torch_measure.models.ColdStartLookupPredictor` — the 6-level
  empirical-Bayes fallback hierarchy + intercept-only Platt calibration
  (lookup + calibrate paths; the lookup-table BUILD path lives in
  :mod:`submission.train`).

Why standalone? The hosted Codabench container ships an organizer-supplied
``torch_measure`` package that may NOT track this fork's ``feat/caimira``
branch. The ZIP must carry its own self-contained implementation; we cannot
``import torch_measure.models.CAIMIRA`` at runtime. This file is therefore
the predict-time wrapper; it does NOT import anything from ``torch_measure``.

Architecture (CAIMIRA): per ``arXiv:2410.06524`` §3.3, footnote 5, Appendix B:

* ``relevance_head``: ``nn.Linear`` with bias, softmax-normalized over latent axis.
* ``difficulty_head``: ``nn.Linear`` without bias (zero-centering absorbs constant).
* ``skill``: directly-learned per-subject vectors.
* Response: ``P = sigmoid((s_i - d_j)^T r_j)``.
* Zero-centering: frozen training-bank mean snapshot in a buffer
  (cold-start inference reads this; training-time dynamic centering lives
  in the upstream :class:`torch_measure.models.CAIMIRA` trainer).

EB fallback (per ``cold_start_lookup.py`` 6-level hierarchy):

1. ``sbc[subject||benchmark||condition]`` triple
2. ``sb[subject||benchmark]`` (Bayesian-shrunk)
3. ``sigmoid(logit(subj) + logit(bench) - logit(global))`` (1-PL Rasch blend)
4. ``bench[benchmark]``
5. ``subj[subject]``
6. ``global_mean``

Hybrid (CAIMIRA + EB):

* In-vocabulary ``(subject, benchmark, condition)``: blend CAIMIRA logit with
  EB logit by a fixed weight (default ``lambda_=0.6``; CAIMIRA-leaning).
* Out-of-vocabulary subject (CAIMIRA can't produce a skill vector for text
  it never saw at train time): use EB only.
* Per-benchmark intercept-only Platt from the platform's ``labeled``
  reveals (slope fixed at ``1.0``, shift capped at ``±1.5`` logit units —
  prevents K=5 reveals from destroying ranking).

References
----------
- Gor, M., Daumé III, H., Zhou, F., Boyd-Graber, J. "Do great minds think
  alike? Investigating Human-AI Complementarity in Question Answering with
  CAIMIRA." EMNLP 2024. arXiv:2410.06524.
- Rasch, G. (1960). Probabilistic models for some intelligence and
  attainment tests. Danish Institute for Educational Research.
- Platt, J. (1999). Probabilistic outputs for support vector machines.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

EMBED_DIM = 768
CAIMIRA_ARCH_VERSION = 1

# Provider prefixes that subjects sometimes appear with in subject_content
# but not in display_name (e.g. "meta-llama/Llama-2-7b-chat" vs "Llama-2-7b-chat").
# Sourced verbatim from torch_measure.models.cold_start_lookup.
_DEFAULT_PROVIDER_PREFIXES: tuple[str, ...] = (
    "meta-llama/",
    "openai/",
    "mistralai/",
    "google/",
    "anthropic/",
    "microsoft/",
    "huggingface/",
    "tiiuae/",
    "deepmind/",
    "cohere/",
    "allenai/",
    "01-ai/",
    "qwen/",
    "baichuan-inc/",
)

# Predict-time probability bounds. Avoid 0 / 1 in logit() and keep NLL finite.
_CLIP_LO: float = 0.05
_CLIP_HI: float = 0.95

# Kit-contract bounds for the final return (predict() must clip to (0,1)
# strictly to keep log-loss finite — the team's NCF uses [1e-4, 1-1e-4]).
_PREDICT_LO: float = 1e-4
_PREDICT_HI: float = 1.0 - 1e-4

_DEFAULT_PLATT_SHIFT_CAP: float = 1.5


def _logit(p: float) -> float:
    """Numerically safe logit. Inputs are clipped away from 0 and 1.

    Non-finite input (NaN, ±inf) raises ``ValueError`` rather than silently
    collapsing to ``±16.118`` via the ``max(1e-7, min(1-1e-7, nan))`` ladder
    (``1e-7 < nan == False`` makes the inner ``min`` return the clip lower
    bound, which then maps to a finite logit and ``sigmoid`` ≈ ``0.9999`` —
    the constant-prediction signature the D-3 invariant defends against).
    """
    if not math.isfinite(p):
        raise ValueError(f"_logit requires finite input, got {p!r}")
    p = max(1e-7, min(1 - 1e-7, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid that avoids overflow for large |x|."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _clip(p: float, lo: float = _CLIP_LO, hi: float = _CLIP_HI) -> float:
    return max(lo, min(hi, float(p)))


def clip_for_predict(p: float) -> float:
    """Final clip applied at the kit boundary, narrower than ``_clip``."""
    return float(max(_PREDICT_LO, min(_PREDICT_HI, float(p))))


def parse_subject_name(subject_content: str) -> str:
    """Extract the model name from a ``subject_content`` block.

    The competition runtime renders subject metadata as::

        Name: <display_name>
        Organization: <provider>
        Parameters: <params>
        ...

    Returns the first line after the ``Name:`` label, stripped; falls back
    to the first non-empty line if no ``Name:`` is present. Empty input
    returns the empty string.
    """
    if not subject_content:
        return ""
    m = re.match(r"\s*Name:\s*(.+)", subject_content)
    if m:
        lines = m.group(1).strip().splitlines()
        if lines:
            return lines[0].strip()
        return ""
    return subject_content.split("\n", 1)[0].strip()


def resolve_subject_name(
    raw_name: str,
    subj_lookup: dict[str, float],
    sb_lookup: dict[str, float],
    sbc_lookup: dict[str, float],
    aliases: dict[str, str],
    name_lc: dict[str, str],
    provider_prefixes: Iterable[str] = _DEFAULT_PROVIDER_PREFIXES,
) -> str:
    """Resolve a raw display name to a canonical lookup-table key.

    Tries the raw name first, then a provider-prefix-stripped variant
    (``meta-llama/Llama-2-7b-chat`` → ``Llama-2-7b-chat``), through three
    resolution paths each: direct, alias map, case-insensitive map.
    """
    candidates = [raw_name]
    raw_lower = raw_name.lower()
    for prefix in provider_prefixes:
        if raw_lower.startswith(prefix):
            candidates.append(raw_name[len(prefix) :].strip())
            break

    for cand in candidates:
        if cand in subj_lookup or cand in sb_lookup or cand in sbc_lookup:
            return cand
        if cand in aliases:
            return aliases[cand]
        lc = cand.lower()
        if lc in name_lc:
            return name_lc[lc]
    return raw_name


class CAIMIRALite(nn.Module):
    r"""Self-contained CAIMIRA forward (no torch_measure dependency).

    Matches :class:`torch_measure.models.CAIMIRA`'s mathematical intent
    byte-for-byte. Only the trainer is dropped (predict-only); the buffer
    ``_difficulty_mean`` is loaded from the state_dict, NOT recomputed —
    so this class always uses frozen centering, the correct mode for
    cold-start inference.

    State_dict keys (must match the upstream trainer):

    * ``skill``: ``(n_subjects, latent_dim)`` parameter
    * ``relevance_head.weight``: ``(latent_dim, embedding_dim)``
    * ``relevance_head.bias``: ``(latent_dim,)``
    * ``difficulty_head.weight``: ``(latent_dim, embedding_dim)`` (no bias)
    * ``_difficulty_mean``: ``(latent_dim,)`` buffer

    Parameters
    ----------
    n_subjects, n_items : int
        Universe sizes used at training (``n_items`` is only used to size
        the buffer; predict path does not reference it because item
        embeddings flow through ``predict_one``).
    embedding_dim : int
        Item embedding dimensionality (768 for ``all-mpnet-base-v2``).
    latent_dim : int
        CAIMIRA latent skill dimension (default 5; matches paper experiments).
    """

    def __init__(
        self,
        n_subjects: int,
        n_items: int,
        embedding_dim: int = EMBED_DIM,
        latent_dim: int = 5,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        self.n_subjects = n_subjects
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

        self.skill = nn.Parameter(torch.zeros(n_subjects, latent_dim))
        self.relevance_head = nn.Linear(embedding_dim, latent_dim, bias=True)
        self.difficulty_head = nn.Linear(embedding_dim, latent_dim, bias=False)
        self.register_buffer("_difficulty_mean", torch.zeros(latent_dim))

    def caimira_logit(
        self,
        subject_idx: int,
        item_embedding: torch.Tensor,
    ) -> float:
        """Compute the CAIMIRA logit for one ``(subject_idx, item_embedding)``.

        Uses frozen-mean centering (the ``_difficulty_mean`` buffer).
        Returns a Python float for use in the hybrid blend.
        """
        with torch.no_grad():
            emb = item_embedding.view(-1).to(self.relevance_head.weight.device).float()
            relevance = torch.softmax(self.relevance_head(emb), dim=-1)
            d_raw = self.difficulty_head(emb)
            difficulty = d_raw - self._difficulty_mean
            skill = self.skill[subject_idx]
            logit = ((skill - difficulty) * relevance).sum().item()
        return float(logit)


class EBLookup:
    """Self-contained 6-level EB fallback hierarchy + intercept-only Platt.

    Loaded from a JSON artifact produced by :func:`submission.train.build_eb_tables`,
    matching the schema of
    :meth:`torch_measure.models.ColdStartLookupPredictor.to_lookup_json`.

    The JSON schema is::

        {
            "sbc":   {"<subject>||<benchmark>||<condition>": <P>},
            "sb":    {"<subject>||<benchmark>":              <P>},
            "subj":  {"<subject>":                           <P>},
            "bench": {"<benchmark>":                         <P>},
            "global": <P>,
            "name_aliases": {<prefix-stripped name>: <canonical>},
            "name_lc":      {<name.lower()>: <canonical>}
        }
    """

    def __init__(
        self,
        sbc: dict[str, float],
        sb: dict[str, float],
        subj: dict[str, float],
        bench: dict[str, float],
        global_mean: float,
        name_aliases: dict[str, str] | None = None,
        name_lc: dict[str, str] | None = None,
        platt_shift_cap: float = _DEFAULT_PLATT_SHIFT_CAP,
        output_clip: tuple[float, float] = (0.02, 0.98),
    ) -> None:
        self.sbc = dict(sbc)
        self.sb = dict(sb)
        self.subj = dict(subj)
        self.bench = dict(bench)
        self.global_mean = float(global_mean)
        self.name_aliases = dict(name_aliases or {})
        self.name_lc = dict(name_lc or {})
        self.platt_shift_cap = float(platt_shift_cap)
        self.output_clip = (float(output_clip[0]), float(output_clip[1]))

        self._sbc_ci = {k.lower(): v for k, v in self.sbc.items()}
        self._sb_ci = {k.lower(): v for k, v in self.sb.items()}
        self._subj_ci = {k.lower(): v for k, v in self.subj.items()}
        self._bench_ci = {k.lower(): v for k, v in self.bench.items()}

        self._platt: dict[str, tuple[float, float]] = {}
        self._platt_fit_key: int = -1

    @classmethod
    def from_json(cls, path: str | Path, **overrides: Any) -> EBLookup:
        data = json.loads(Path(path).read_text())
        return cls(
            sbc=data["sbc"],
            sb=data["sb"],
            subj=data["subj"],
            bench=data["bench"],
            global_mean=data["global"],
            name_aliases=data.get("name_aliases", {}),
            name_lc=data.get("name_lc", {}),
            **overrides,
        )

    def resolve_name(self, raw_name: str) -> str:
        return resolve_subject_name(
            raw_name,
            self.subj,
            self.sb,
            self.sbc,
            self.name_aliases,
            self.name_lc,
        )

    def lookup_p(self, subj_name: str, benchmark: str, condition: str) -> float:
        """Walk the 6-level fallback hierarchy. Returns the first match."""
        condition = condition or "none"

        key3 = f"{subj_name}||{benchmark}||{condition}"
        if key3 in self.sbc:
            return self.sbc[key3]
        if key3.lower() in self._sbc_ci:
            return self._sbc_ci[key3.lower()]
        key3_none = f"{subj_name}||{benchmark}||none"
        if key3_none in self.sbc:
            return self.sbc[key3_none]
        if key3_none.lower() in self._sbc_ci:
            return self._sbc_ci[key3_none.lower()]

        key2 = f"{subj_name}||{benchmark}"
        if key2 in self.sb:
            return self.sb[key2]
        if key2.lower() in self._sb_ci:
            return self._sb_ci[key2.lower()]

        # Explicit ``is None`` rather than ``or`` so a legitimate ``0.0`` prior
        # is not short-circuited to the case-insensitive view.
        subj_p = self.subj.get(subj_name)
        if subj_p is None:
            subj_p = self._subj_ci.get(subj_name.lower())
        bench_p = self.bench.get(benchmark)
        if bench_p is None:
            bench_p = self._bench_ci.get(benchmark.lower())
        if subj_p is not None and bench_p is not None:
            return _clip(_sigmoid(_logit(subj_p) + _logit(bench_p) - _logit(self.global_mean)))

        if bench_p is not None:
            return bench_p
        if subj_p is not None:
            return subj_p
        return self.global_mean

    def fit_platt(self, labeled: list[dict[str, Any]]) -> None:
        """Fit intercept-only Platt per benchmark; slope fixed at 1.0.

        Idempotent w.r.t. labeled list length (cached). With K=5 examples
        per benchmark a 2-parameter fit destroys subject ordering (slope
        collapses to ~0.1), so we shift only and cap at ``±platt_shift_cap``.

        **Codabench-only optimization.** The ``len(labeled)``-key cache is
        valid here because the hosted Codabench runtime sends the same
        ``labeled`` list throughout a round; library callers that pass
        distinct labeled lists of equal length within one process MUST NOT
        rely on this pattern. The upstream
        :meth:`torch_measure.models.cold_start_lookup.ColdStartLookupPredictor.calibrate`
        intentionally does NOT carry this cache for that reason.
        """
        new_key = len(labeled)
        if new_key == self._platt_fit_key:
            return
        self._platt_fit_key = new_key
        self._platt.clear()

        by_bench: dict[str, list[tuple[float, float]]] = {}
        for ex in labeled:
            if "label" not in ex:
                continue
            bench = (ex.get("benchmark") or "").strip()
            cond = (ex.get("condition") or "none").strip() or "none"
            subject_content = ex.get("subject_content", "") or ""
            raw_name = parse_subject_name(subject_content)
            subj_name = self.resolve_name(raw_name)
            try:
                label = float(ex["label"])
            except (TypeError, ValueError):
                continue
            p = self.lookup_p(subj_name, bench, cond)
            by_bench.setdefault(bench, []).append((_logit(p), label))

        for bench, pairs in by_bench.items():
            if len(pairs) < 2:
                continue
            xs = [x for x, _ in pairs]
            ys = [y for _, y in pairs]
            mean_y = sum(ys) / len(ys)
            if mean_y <= 0 or mean_y >= 1:
                # mean_y at the boundary would produce ±inf shift; the EB
                # tables are clipped to [_CLIP_LO, _CLIP_HI] for the same
                # reason, but the OBSERVED label mean can still hit 0 or 1
                # at K=5 — guard explicitly per the lane-B FAILURE NOTE.
                continue
            mean_x = sum(xs) / len(xs)
            target_logit = math.log(mean_y / (1.0 - mean_y))
            shift = target_logit - mean_x
            shift = max(-self.platt_shift_cap, min(self.platt_shift_cap, shift))
            self._platt[bench] = (1.0, shift)

    def predict(
        self,
        subject_content: str,
        benchmark: str,
        condition: str,
        labeled: list[dict[str, Any]] | None = None,
    ) -> float:
        """Full EB prediction including Platt calibration if ``labeled`` given."""
        raw_name = parse_subject_name(subject_content or "")
        subj_name = self.resolve_name(raw_name)
        p = self.lookup_p(subj_name, benchmark, condition)

        if labeled:
            self.fit_platt(labeled)
            if benchmark in self._platt:
                slope, intercept = self._platt[benchmark]
                p = _sigmoid(slope * _logit(p) + intercept)

        lo, hi = self.output_clip
        return max(lo, min(hi, float(p)))
