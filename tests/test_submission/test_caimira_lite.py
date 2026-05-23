# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Unit tests for submission.caimira_lite — standalone CAIMIRA + EB lookup.

Covers the SHIPPABLE submission classes (no torch_measure import). These
tests verify (a) state_dict round-trip from the upstream CAIMIRA, (b) the
6-level EB fallback walks each level correctly, (c) parse_subject_name +
resolve_subject_name handle the documented edge cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = REPO_ROOT / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from caimira_lite import (  # noqa: E402
    CAIMIRALite,
    EBLookup,
    clip_for_predict,
    parse_subject_name,
    resolve_subject_name,
)

from torch_measure.models import CAIMIRA  # noqa: E402


class TestCAIMIRALiteStateDictParity:
    """The standalone CAIMIRALite MUST accept state_dicts from upstream CAIMIRA."""

    def test_keys_match_upstream(self):
        n_s, n_i, d, k = 5, 7, 16, 3
        upstream = CAIMIRA(n_subjects=n_s, n_items=n_i, embedding_dim=d, latent_dim=k)
        lite = CAIMIRALite(n_subjects=n_s, n_items=n_i, embedding_dim=d, latent_dim=k)
        upstream_keys = set(upstream.state_dict().keys())
        lite_keys = set(lite.state_dict().keys())
        assert upstream_keys == lite_keys

    def test_state_dict_round_trips(self):
        n_s, n_i, d, k = 5, 7, 16, 3
        upstream = CAIMIRA(n_subjects=n_s, n_items=n_i, embedding_dim=d, latent_dim=k)
        with torch.no_grad():
            upstream.skill.fill_(0.3)
            upstream.relevance_head.weight.fill_(0.1)
            upstream.relevance_head.bias.fill_(-0.05)
            upstream.difficulty_head.weight.fill_(0.2)
            upstream._difficulty_mean.fill_(0.07)

        lite = CAIMIRALite(n_subjects=n_s, n_items=n_i, embedding_dim=d, latent_dim=k)
        missing, unexpected = lite.load_state_dict(upstream.state_dict(), strict=True)
        assert missing == [] and unexpected == []
        assert torch.allclose(lite.skill, upstream.skill)
        assert torch.allclose(lite.relevance_head.weight, upstream.relevance_head.weight)
        assert torch.allclose(lite.relevance_head.bias, upstream.relevance_head.bias)
        assert torch.allclose(lite.difficulty_head.weight, upstream.difficulty_head.weight)
        assert torch.allclose(lite._difficulty_mean, upstream._difficulty_mean)


class TestCAIMIRALiteForward:
    def test_init_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            CAIMIRALite(n_subjects=5, n_items=10, embedding_dim=0, latent_dim=3)
        with pytest.raises(ValueError):
            CAIMIRALite(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=0)

    def test_caimira_logit_returns_float(self):
        m = CAIMIRALite(n_subjects=4, n_items=6, embedding_dim=8, latent_dim=3)
        m.eval()
        emb = torch.randn(8)
        out = m.caimira_logit(subject_idx=2, item_embedding=emb)
        assert isinstance(out, float)
        assert -100 < out < 100  # sane finite range

    def test_caimira_logit_matches_manual_equation(self):
        """End-to-end check: caimira_logit == ((s - d) * r).sum() with frozen centering."""
        torch.manual_seed(0)
        n_s, n_i, d, k = 4, 6, 8, 3
        m = CAIMIRALite(n_subjects=n_s, n_items=n_i, embedding_dim=d, latent_dim=k)
        m.eval()
        with torch.no_grad():
            m._difficulty_mean.fill_(0.25)
        emb = torch.randn(d)
        idx = 2
        out = m.caimira_logit(subject_idx=idx, item_embedding=emb)
        with torch.no_grad():
            relevance = torch.softmax(m.relevance_head(emb), dim=-1)
            d_raw = m.difficulty_head(emb)
            difficulty = d_raw - m._difficulty_mean
            skill = m.skill[idx]
            manual = ((skill - difficulty) * relevance).sum().item()
        assert abs(out - manual) < 1e-6


class TestParseSubjectName:
    def test_extracts_name_after_label(self):
        assert parse_subject_name("Name: GPT-4") == "GPT-4"

    def test_extracts_first_line_when_multi_line(self):
        content = "Name: Claude-3-Opus\nOrganization: Anthropic\nParameters: 175B"
        assert parse_subject_name(content) == "Claude-3-Opus"

    def test_falls_back_to_first_line_when_no_name_label(self):
        assert parse_subject_name("Llama-2-70b") == "Llama-2-70b"

    def test_empty_input_returns_empty(self):
        assert parse_subject_name("") == ""
        assert parse_subject_name(None or "") == ""


class TestResolveSubjectName:
    def test_direct_match_returns_unchanged(self):
        subj = {"GPT-4": 0.8}
        name = resolve_subject_name("GPT-4", subj, {}, {}, {}, {})
        assert name == "GPT-4"

    def test_provider_prefix_stripped(self):
        subj = {"Llama-2-7b-chat": 0.7}
        name = resolve_subject_name(
            "meta-llama/Llama-2-7b-chat",
            subj, {}, {}, {}, {},
        )
        assert name == "Llama-2-7b-chat"

    def test_case_insensitive_fallback(self):
        name_lc = {"gpt-4": "GPT-4"}
        name = resolve_subject_name("gpt-4", {}, {}, {}, {}, name_lc)
        assert name == "GPT-4"

    def test_unresolved_returns_raw(self):
        name = resolve_subject_name("UnknownModel", {}, {}, {}, {}, {})
        assert name == "UnknownModel"


class TestEBLookup:
    """Verify the 6-level fallback hierarchy walks correctly."""

    @pytest.fixture
    def lookup(self) -> EBLookup:
        return EBLookup(
            sbc={"gpt-4||mmlupro||cot": 0.85},
            sb={"gpt-4||mmlupro": 0.80, "gpt-4||ai2d": 0.70},
            subj={"gpt-4": 0.78, "claude-3": 0.72},
            bench={"mmlupro": 0.50, "ai2d": 0.65, "untested_bench": 0.30},
            global_mean=0.55,
        )

    def test_level1_sbc_triple(self, lookup):
        assert lookup.lookup_p("gpt-4", "mmlupro", "cot") == 0.85

    def test_level2_sb_pair(self, lookup):
        # condition not in sbc triple → falls to sb pair
        assert lookup.lookup_p("gpt-4", "ai2d", "any_condition") == 0.70

    def test_level3_irt_blend(self, lookup):
        # subject + bench known, but no sb entry → IRT blend
        # logit(0.72) + logit(0.65) - logit(0.55)
        p = lookup.lookup_p("claude-3", "ai2d", "x")
        assert 0.05 <= p <= 0.95
        # Should be ABOVE the bench prior since subject is also above-average.
        assert p > 0.65

    def test_level4_bench_only(self, lookup):
        # Unknown subject, known bench → bench prior
        assert lookup.lookup_p("MysteryModel", "untested_bench", "x") == 0.30

    def test_level5_subj_only(self, lookup):
        # Known subject, unknown bench → subject prior
        assert lookup.lookup_p("gpt-4", "MysteryBench", "x") == 0.78

    def test_level6_global_fallback(self, lookup):
        # Both unknown → global mean
        assert lookup.lookup_p("MysteryModel", "MysteryBench", "x") == 0.55


class TestEBLookupPlatt:
    """Intercept-only Platt with ±1.5 shift cap, slope fixed at 1.0."""

    def test_calibrate_shifts_intercept(self):
        lookup = EBLookup(
            sbc={},
            sb={},
            subj={},
            bench={"b1": 0.30},
            global_mean=0.50,
        )
        labeled = [
            {"subject_content": "Name: m1", "benchmark": "b1", "condition": "x", "label": 1.0},
            {"subject_content": "Name: m2", "benchmark": "b1", "condition": "x", "label": 1.0},
            {"subject_content": "Name: m3", "benchmark": "b1", "condition": "x", "label": 1.0},
        ]
        # Mean label = 1.0 → degenerate target_logit = +inf → should skip per
        # the mean_y <= 0 or >= 1 guard, NOT crash.
        lookup.fit_platt(labeled)
        assert "b1" not in lookup._platt  # skipped because mean_y == 1.0

    def test_calibrate_caps_shift_at_1_5(self):
        lookup = EBLookup(
            sbc={},
            sb={},
            subj={},
            bench={"b1": 0.05},  # very low prior; would want huge upward shift
            global_mean=0.50,
        )
        labeled = [
            {"subject_content": f"Name: m{i}", "benchmark": "b1", "condition": "x",
             "label": 1.0 if i % 2 == 0 else 0.0}
            for i in range(8)
        ]
        # mean_y = 0.5 → target_logit = 0; mean_x = logit(0.05) ≈ -2.94.
        # Wanted shift = 0 - (-2.94) = +2.94, capped to +1.5.
        lookup.fit_platt(labeled)
        assert "b1" in lookup._platt
        slope, intercept = lookup._platt["b1"]
        assert slope == 1.0
        assert abs(intercept - 1.5) < 1e-6  # capped

    def test_predict_applies_platt_when_labeled(self):
        lookup = EBLookup(
            sbc={},
            sb={},
            subj={},
            bench={"b1": 0.30},
            global_mean=0.50,
        )
        p_no_labeled = lookup.predict("Name: m", "b1", "x", labeled=None)
        labeled = [
            {"subject_content": f"Name: m{i}", "benchmark": "b1", "condition": "x",
             "label": 1.0 if i % 2 == 0 else 0.0}
            for i in range(4)
        ]
        p_with_labeled = lookup.predict("Name: m", "b1", "x", labeled=labeled)
        # mean_y = 0.5, mean_x = logit(0.30) ≈ -0.85.
        # Shift should be positive (upward) → p_with > p_no.
        assert p_with_labeled > p_no_labeled


class TestClipForPredict:
    def test_clips_to_kit_range(self):
        assert clip_for_predict(-0.1) == 1e-4
        assert clip_for_predict(1.5) == 1.0 - 1e-4
        assert clip_for_predict(0.5) == 0.5

    def test_returns_native_float(self):
        out = clip_for_predict(torch.tensor(0.7).item())
        assert type(out) is float
