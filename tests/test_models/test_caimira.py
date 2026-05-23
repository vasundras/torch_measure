# Copyright (c) 2026 AIMS Foundations. MIT License.

import torch
from torch import nn

from torch_measure.models import CAIMIRA
from torch_measure.models._predictor import predict_dense


class TestCAIMIRA:
    def test_init(self):
        model = CAIMIRA(n_subjects=10, n_items=20, embedding_dim=64, latent_dim=3)
        assert model.n_subjects == 10
        assert model.n_items == 20
        assert model.embedding_dim == 64
        assert model.latent_dim == 3
        assert model.skill.shape == (10, 3)

    def test_default_latent_dim_matches_paper(self):
        """Default latent_dim mirrors the value used in CAIMIRA paper experiments (m=5)."""
        model = CAIMIRA(n_subjects=4, n_items=6, embedding_dim=8)
        assert model.latent_dim == 5
        assert model.skill.shape == (4, 5)

    def test_init_rejects_nonpositive_dims(self):
        try:
            CAIMIRA(n_subjects=5, n_items=10, embedding_dim=0, latent_dim=3)
            raise AssertionError("Expected ValueError for embedding_dim=0")
        except ValueError:
            pass
        try:
            CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=0)
            raise AssertionError("Expected ValueError for latent_dim=0")
        except ValueError:
            pass

    def test_difficulty_head_is_linear_and_biasless(self):
        """Architectural invariant: difficulty head is bias-free (centering absorbs the bias)."""
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=4)
        assert isinstance(model.difficulty_head, nn.Linear)
        assert model.difficulty_head.bias is None
        assert model.difficulty_head.weight.shape == (4, 8)

    def test_relevance_head_is_linear_with_bias(self):
        """Architectural invariant: relevance head is linear with bias (informative under softmax)."""
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=4)
        assert isinstance(model.relevance_head, nn.Linear)
        assert model.relevance_head.bias is not None
        assert model.relevance_head.weight.shape == (4, 8)
        assert model.relevance_head.bias.shape == (4,)

    def test_set_embeddings_wrong_n_items(self):
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=3)
        try:
            model.set_embeddings(torch.randn(7, 8))  # wrong n_items
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_set_embeddings_wrong_embedding_dim(self):
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=3)
        try:
            model.set_embeddings(torch.randn(10, 16))  # wrong embedding_dim
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_predict_requires_embeddings(self):
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=3)
        try:
            predict_dense(model)
            raise AssertionError("Should have raised RuntimeError")
        except RuntimeError:
            pass

    def test_predict_shape(self):
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=3)
        model.set_embeddings(torch.randn(10, 8))
        probs = predict_dense(model)
        assert probs.shape == (5, 10)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_relevance_rows_sum_to_one(self):
        """Softmax invariant: each item's relevance vector lies on the simplex."""
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=4)
        model.set_embeddings(torch.randn(10, 8))
        relevance, _ = model.compute_item_params()
        assert relevance.shape == (10, 4)
        assert torch.allclose(relevance.sum(dim=-1), torch.ones(10), atol=1e-6)

    def test_relevance_is_nonnegative(self):
        """Softmax invariant: relevance entries are non-negative."""
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=4)
        model.set_embeddings(torch.randn(10, 8))
        relevance, _ = model.compute_item_params()
        assert (relevance >= 0).all()

    def test_difficulty_zero_centered_over_training_bank_dynamic(self):
        """In dynamic centering mode, difficulty across the training bank has zero mean."""
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=4)
        model.set_embeddings(torch.randn(10, 8))
        _, difficulty = model.compute_item_params(center="dynamic")
        assert difficulty.shape == (10, 4)
        assert torch.allclose(difficulty.mean(dim=0), torch.zeros(4), atol=1e-6)

    def test_frozen_difficulty_mean_buffer_used_at_inference(self):
        """After fit() the snapshot is taken; frozen-mode centering reads the buffer."""
        torch.manual_seed(0)
        n_subjects, n_items, embed = 6, 12, 8
        model = CAIMIRA(n_subjects=n_subjects, n_items=n_items, embedding_dim=embed, latent_dim=3)
        embeddings = torch.randn(n_items, embed)
        responses = torch.bernoulli(torch.full((n_subjects, n_items), 0.5))
        model.fit(responses, embeddings, max_epochs=5, verbose=False)
        # Buffer should equal the training-bank d_raw mean.
        with torch.no_grad():
            d_raw_train = model.difficulty_head(embeddings)
            assert torch.allclose(model._difficulty_mean, d_raw_train.mean(dim=0), atol=1e-6)
        # Cold-start: new items should be centered by the SAME buffer, not their own mean.
        new_embeddings = torch.randn(5, embed)
        _, d_new = model.compute_item_params(new_embeddings, center="frozen")
        with torch.no_grad():
            d_raw_new = model.difficulty_head(new_embeddings)
            expected = d_raw_new - model._difficulty_mean
        assert torch.allclose(d_new, expected, atol=1e-6)

    def test_predict_matches_manual_caimira_equation(self):
        """End-to-end check: predict() matches sigmoid((s_i - d_j) · r_j) row by row."""
        torch.manual_seed(0)
        n_subjects, n_items, embed, latent = 4, 6, 8, 3
        model = CAIMIRA(n_subjects=n_subjects, n_items=n_items, embedding_dim=embed, latent_dim=latent)
        model.eval()  # use frozen centering (buffer is still zero → no centering effect)
        embeddings = torch.randn(n_items, embed)
        model.set_embeddings(embeddings)

        query = {
            "subject_idx": torch.tensor([0, 1, 2, 3, 0]),
            "item_idx": torch.tensor([1, 2, 3, 4, 5]),
        }
        with torch.no_grad():
            probs = model.predict(query)
            relevance, difficulty = model.compute_item_params(center="frozen")
            manual = torch.sigmoid(
                (
                    (model.skill[query["subject_idx"]] - difficulty[query["item_idx"]]) * relevance[query["item_idx"]]
                ).sum(dim=-1)
            )
        assert probs.shape == (5,)
        assert torch.allclose(probs, manual, atol=1e-6)

    def test_predict_embeddings_scores_cold_start_items_with_frozen_centering(self):
        """Cold-start item embeddings should score without mutating the training bank."""
        torch.manual_seed(0)
        n_subjects, n_items, embed, latent = 4, 6, 8, 3
        model = CAIMIRA(n_subjects=n_subjects, n_items=n_items, embedding_dim=embed, latent_dim=latent)
        train_embeddings = torch.randn(n_items, embed)
        model.set_embeddings(train_embeddings)
        with torch.no_grad():
            model._difficulty_mean.fill_(0.25)
        model.eval()

        subject_idx = torch.tensor([0, 1, 2, 3])
        cold_embeddings = torch.randn(4, embed)
        probs = model.predict_embeddings(subject_idx, cold_embeddings, center="frozen")

        relevance, difficulty = model.compute_item_params(cold_embeddings, center="frozen")
        manual = torch.sigmoid(((model.skill[subject_idx] - difficulty) * relevance).sum(dim=-1))
        assert probs.shape == (4,)
        assert torch.allclose(probs, manual, atol=1e-6)
        assert model._embeddings is not None
        assert torch.allclose(model._embeddings, train_embeddings)

    def test_difficulty_mean_buffer_survives_state_dict_roundtrip(self):
        """The frozen centering buffer is part of the deployable state_dict."""
        model = CAIMIRA(n_subjects=3, n_items=5, embedding_dim=7, latent_dim=2)
        with torch.no_grad():
            model._difficulty_mean.copy_(torch.tensor([0.3, -0.2]))
        clone = CAIMIRA(n_subjects=3, n_items=5, embedding_dim=7, latent_dim=2)
        clone.load_state_dict(model.state_dict())
        assert torch.allclose(clone._difficulty_mean, torch.tensor([0.3, -0.2]))

    def test_embedding_paths_follow_parameter_device_after_to(self):
        """Embedding inputs should follow the module's actual parameter device after .to()."""
        model = CAIMIRA(n_subjects=3, n_items=5, embedding_dim=7, latent_dim=2)
        model.to("meta")
        embeddings = torch.empty(5, 7)
        model.set_embeddings(embeddings)
        assert model._embeddings is not None
        assert model._embeddings.device.type == "meta"

    def test_fit_reduces_loss(self, small_response_matrix):
        torch.manual_seed(0)
        n_subjects, n_items = small_response_matrix.shape
        embeddings = torch.randn(n_items, 16)
        model = CAIMIRA(n_subjects=n_subjects, n_items=n_items, embedding_dim=16, latent_dim=3)
        history = model.fit(small_response_matrix, embeddings, max_epochs=50, verbose=False)
        assert history["losses"][-1] < history["losses"][0]

    def test_eval_mode_uses_frozen_centering(self):
        """In eval mode, compute_item_params uses the buffer (not dynamic recompute)."""
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=3)
        model.set_embeddings(torch.randn(10, 8))
        # Manually set the buffer to a known non-zero value
        with torch.no_grad():
            model._difficulty_mean.fill_(0.5)
        model.eval()
        _, d_frozen = model.compute_item_params(center="auto")
        with torch.no_grad():
            d_raw = model.difficulty_head(model._embeddings)
            expected = d_raw - 0.5
        assert torch.allclose(d_frozen, expected, atol=1e-6)

    def test_exported_from_models_init(self):
        """CAIMIRA must be importable from torch_measure.models top-level."""
        from torch_measure.models import CAIMIRA as ExportedCAIMIRA

        assert ExportedCAIMIRA is CAIMIRA
