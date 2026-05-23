# Copyright (c) 2026 AIMS Foundations. MIT License.
# pylint: disable=global-statement,broad-except,broad-exception-caught
"""Stdlib-only adaptive-labeling acquisition via bounded SimHash diversity.

The hosted platform calls ``acquisition_function(input)`` once per hidden
candidate BEFORE ``predict()``. The naive "encode every candidate with the
sentence-transformer encoder" pattern is too expensive for the streaming
10K-candidate-per-round path, and earlier ports of that pattern had a
documented NaN-poisoning bug whose blast radius is global (any one
exception / NaN / inf in any one acquisition score discards ALL
acquisition scores for the whole round → random labeling fallback for
every category, every round).

This implementation is:

* **stdlib-only** (``hashlib.blake2b``, ``re``, ``math``) — no torch import,
  no encoder dependency, no NumPy.
* **bounded cost**: 64-bit SimHash signatures over the visible text fields,
  ranked against a 128-entry reservoir; O(_MAX_TOKENS + _MAX_SEEN · _BITS)
  per call, fixed regardless of the round's total candidate count.
* **bulletproof**: the single ``except Exception`` returns the distinguishable
  sentinel ``0.0`` (see :func:`acquisition_function` docstring).

Port note: this file mirrors the production-vetted SimHash acquisition
shipping in the parent CS321M competition repo at
``predictive-eval-competition/submission/labeling.py`` (commit lineage
"feat: SimHash diversity acquisition for adaptive labeling"). Kept
verbatim except for the docstring header so the submission ZIP carries
identical behavior across both upstreams.
"""

from __future__ import annotations

import hashlib
import math
import re

_BITS = 64
_MAX_TOKENS = 256
_MAX_SEEN = 128
_MAX_STRATA = 256
_TIE_EPSILON = 0.01
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_seen_signatures: list[int] = []
_stratum_counts: dict[tuple[str, str, str], int] = {}
_candidate_count = 0


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _visible_text(ex: dict) -> str:
    return "\n".join(
        (
            _text(ex.get("benchmark")),
            _text(ex.get("condition")),
            _text(ex.get("subject_content")),
            _text(ex.get("item_content")),
        )
    ).lower()


def _tokens(ex: dict) -> list[str]:
    return _TOKEN_RE.findall(_visible_text(ex))[:_MAX_TOKENS]


def _hash_u64(text: str, person: bytes = b"peval-v1") -> int:
    digest = hashlib.blake2b(
        text.encode("utf-8", "replace"),
        digest_size=8,
        person=person,
    ).digest()
    return int.from_bytes(digest, "little")


def _stable_unit_interval(text: str, person: bytes) -> float:
    return _hash_u64(text, person=person) / float(1 << 64)


def _simhash(ex: dict) -> int:
    accum = [0] * _BITS
    toks = _tokens(ex)
    if not toks:
        toks = ["<empty>"]

    for token in toks:
        h = _hash_u64(token, person=b"simhash-v1")
        for bit in range(_BITS):
            accum[bit] += 1 if (h >> bit) & 1 else -1

    signature = 0
    for bit, value in enumerate(accum):
        if value >= 0:
            signature |= 1 << bit
    return signature


def _coarse_subject_bucket(subject_content: object) -> str:
    text = _text(subject_content).lower()
    name = ""
    org = ""
    param_bucket = "params_unknown"

    for raw_line in text.splitlines()[:8]:
        line = raw_line.strip()
        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip()[:32]
        elif line.startswith("organization:"):
            org = line.split(":", 1)[1].strip()[:32]
        elif line.startswith("parameters:"):
            digits = "".join(ch for ch in line if ch.isdigit())
            if digits:
                param_bucket = f"params_digits_{len(digits)}"

    if org:
        return f"org:{org}"
    if name:
        return f"name_hash:{_hash_u64(name, person=b'name-v1') % 32}"
    return param_bucket


def _stratum_key(ex: dict) -> tuple[str, str, str]:
    benchmark = _text(ex.get("benchmark")).lower()[:64]
    condition = _text(ex.get("condition")).lower()[:64]
    subject_bucket = _coarse_subject_bucket(ex.get("subject_content"))
    return benchmark, condition, subject_bucket


def _metadata_bonus(ex: dict) -> float:
    key = _stratum_key(ex)
    count = _stratum_counts.get(key, 0)
    return 0.20 / float(1 + count)


def _update_stratum(ex: dict) -> None:
    key = _stratum_key(ex)
    if key in _stratum_counts or len(_stratum_counts) < _MAX_STRATA:
        _stratum_counts[key] = _stratum_counts.get(key, 0) + 1


def _candidate_key(ex: dict) -> str:
    return "\n".join(
        (
            _text(ex.get("benchmark")),
            _text(ex.get("condition")),
            _text(ex.get("subject_content")),
            _text(ex.get("item_content")),
        )
    )


def _diversity_score(signature: int) -> float:
    if not _seen_signatures:
        return 1.0
    nearest = min((signature ^ old).bit_count() for old in _seen_signatures)
    return nearest / float(_BITS)


def _update_reservoir(signature: int, candidate_key: str) -> None:
    if len(_seen_signatures) < _MAX_SEEN:
        _seen_signatures.append(signature)
        return

    slot = _hash_u64(
        f"{candidate_key}\n{_candidate_count}",
        person=b"reservoir-v1",
    ) % _candidate_count
    if slot < _MAX_SEEN:
        _seen_signatures[slot] = signature


def _clamp_score(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(max(0.0, min(2.0, value)))


# pylint: disable=redefined-builtin
def acquisition_function(input: dict) -> float:
    """Return a deterministic finite priority score for one candidate.

    Output domain: legitimate scores are in ``(0, 2]`` — every successful
    path adds a strictly-positive ``tie_break = _stable_unit_interval(key,
    person=b"tie-v1") * _TIE_EPSILON`` where ``_stable_unit_interval``
    is ``BLAKE2b(key) / 2^64`` and ``_TIE_EPSILON = 0.01``. An exact 0
    from the underlying 64-bit hash has probability ``~2^-64`` (one in
    18 quintillion), so the legitimate-path floor is essentially never
    zero in practice.

    INTENTIONAL distinguishable sentinel — the bare-``except`` path
    below returns exactly ``0.0`` as the design-distinguishable failure
    signal per the project's distinguishable-defensive-fallbacks rule
    (1): values OUTSIDE the legitimate output domain. Because the
    legitimate floor is non-zero, an exact-0.0 return ALWAYS signals
    exception fallback and is distinguishable from any real low-priority
    score. Do NOT widen the fallback to a positive value (would collide
    with legit candidates) or to NaN/inf (kit-rejected per the kit
    README's `Invalid predict() output` error). The companion
    ``predict()`` failure cascade rule is: a single NaN/inf/exception
    in ANY ``acquisition_function`` call for ANY candidate discards ALL
    acquisition scores for the whole round.
    """
    global _candidate_count

    try:
        ex = dict(input or {})
        signature = _simhash(ex)
        key = _candidate_key(ex)
        diversity = _diversity_score(signature)
        metadata = _metadata_bonus(ex)
        tie_break = _stable_unit_interval(key, person=b"tie-v1") * _TIE_EPSILON
        score = _clamp_score(diversity + metadata + tie_break)

        _candidate_count += 1
        _update_reservoir(signature, key)
        _update_stratum(ex)
        return score
    except Exception:
        # INTENTIONAL distinguishable sentinel — see function docstring.
        return 0.0
