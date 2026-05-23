# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Production-path coverage for submission/model.py.

All other contract tests run under ``PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1``,
which short-circuits ``predict()`` to return ``0.5`` and skips the CAIMIRA+EB
hybrid entirely. This test exercises the real production path: artifacts on
disk, ``_initialize_runtime`` populating module globals, ``predict()`` walking
in-vocab, OOV, empty-content, and Platt-calibrated paths.

The fixture writes tiny artifacts to a tmp directory and injects a fake
SentenceTransformer that returns deterministic embeddings, so the test
needs no network access and no model cache.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = REPO_ROOT / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))


class FakeEncoder:
    """Deterministic stand-in for ``SentenceTransformer``.

    Produces a fixed-shape tensor whose values depend on a stable hash of the
    input text, so encoding is reproducible without loading real weights.
    """

    def __init__(self, repo: str, device: str = "cpu", local_files_only: bool = True):
        self.repo = repo
        self.device = device
        self.local_files_only = local_files_only

    def encode(self, text, *, convert_to_numpy=False, convert_to_tensor=False, show_progress_bar=False, normalize_embeddings=False):
        import hashlib

        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(digest, "little") % (2**31 - 1)
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(768, generator=generator)


@pytest.fixture
def production_runtime(tmp_path, monkeypatch):
    """Build tiny CAIMIRA + EB artifacts in tmp_path and load them as production."""
    monkeypatch.delenv("PREDICTIVE_EVAL_LOCAL_SMOKE_TEST", raising=False)

    from caimira_lite import CAIMIRALite

    n_subjects, n_items, embed_dim, latent_dim = 3, 5, 768, 5
    head = CAIMIRALite(
        n_subjects=n_subjects,
        n_items=n_items,
        embedding_dim=embed_dim,
        latent_dim=latent_dim,
    )
    torch.manual_seed(0)
    with torch.no_grad():
        head.skill.normal_(mean=0.0, std=0.1)
        head.relevance_head.weight.normal_(mean=0.0, std=0.01)
        head.relevance_head.bias.zero_()
        head.difficulty_head.weight.normal_(mean=0.0, std=0.01)
        head._difficulty_mean.zero_()

    head_path = tmp_path / "caimira_lite.pt"
    meta_path = tmp_path / "caimira_lite.meta.json"
    eb_path = tmp_path / "eb_tables.json"

    torch.save(head.state_dict(), head_path)
    meta_path.write_text(
        json.dumps(
            {
                "n_subjects": n_subjects,
                "n_items": n_items,
                "embed_dim": embed_dim,
                "latent_dim": latent_dim,
                "subject_to_idx": {"gpt-4": 0, "claude-3": 1, "Llama-2-7b-chat": 2},
            }
        )
    )
    eb_path.write_text(
        json.dumps(
            {
                "sbc": {"gpt-4||mmlupro||cot": 0.85},
                "sb": {"gpt-4||mmlupro": 0.80},
                "subj": {"gpt-4": 0.78, "claude-3": 0.72, "Llama-2-7b-chat": 0.65},
                "bench": {"mmlupro": 0.50, "ai2d": 0.65},
                "global": 0.55,
                "name_aliases": {},
                "name_lc": {
                    "gpt-4": "gpt-4",
                    "claude-3": "claude-3",
                    "llama-2-7b-chat": "Llama-2-7b-chat",
                },
            }
        )
    )

    model = importlib.import_module("submission.model")
    model = importlib.reload(model)
    model._initialize_runtime(
        head_path=head_path,
        meta_path=meta_path,
        eb_path=eb_path,
        encoder_factory=FakeEncoder,
        device=torch.device("cpu"),
    )
    return model


class TestProductionPredictPath:
    def test_in_vocab_subject_in_range(self, production_runtime):
        model = production_runtime
        p = model.predict(
            {
                "benchmark": "mmlupro",
                "condition": "cot",
                "subject_content": "Name: gpt-4",
                "item_content": "What is 2+2?",
            },
            labeled=None,
        )
        assert isinstance(p, float)
        assert math.isfinite(p)
        assert 1e-4 <= p <= 1 - 1e-4

    def test_oov_subject_uses_eb_only(self, production_runtime):
        model = production_runtime
        p = model.predict(
            {
                "benchmark": "ai2d",
                "condition": "x",
                "subject_content": "Name: mystery-model",
                "item_content": "An unrelated question.",
            },
            labeled=None,
        )
        assert isinstance(p, float)
        assert math.isfinite(p)
        assert 1e-4 <= p <= 1 - 1e-4

    def test_empty_item_content_does_not_crash(self, production_runtime):
        model = production_runtime
        p = model.predict(
            {
                "benchmark": "ai2d",
                "condition": "x",
                "subject_content": "Name: gpt-4",
                "item_content": "",
            },
            labeled=None,
        )
        assert isinstance(p, float)
        assert math.isfinite(p)
        assert 1e-4 <= p <= 1 - 1e-4

    def test_labeled_applies_platt_shift(self, production_runtime):
        model = production_runtime
        labeled = [
            {
                "benchmark": "ai2d",
                "condition": "x",
                "subject_content": f"Name: model-{i}",
                "item_content": f"Q{i}",
                "label": 1.0 if i % 2 == 0 else 0.0,
            }
            for i in range(6)
        ]
        p = model.predict(
            {
                "benchmark": "ai2d",
                "condition": "x",
                "subject_content": "Name: gpt-4",
                "item_content": "A question.",
            },
            labeled=labeled,
        )
        assert isinstance(p, float)
        assert math.isfinite(p)
        assert 1e-4 <= p <= 1 - 1e-4

    def test_provider_prefixed_subject_resolves(self, production_runtime):
        model = production_runtime
        p = model.predict(
            {
                "benchmark": "mmlupro",
                "condition": "cot",
                "subject_content": "Name: meta-llama/Llama-2-7b-chat",
                "item_content": "Some question.",
            },
            labeled=None,
        )
        assert isinstance(p, float)
        assert math.isfinite(p)
        assert 1e-4 <= p <= 1 - 1e-4
