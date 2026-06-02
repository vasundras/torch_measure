# Copyright (c) 2026 AIMS Foundations. MIT License.
"""Standalone MLP regressor with a scikit-learn-compatible interface.

This module provides :class:`MLPRegressor`, a simple PyTorch-backed neural
network that maps fixed-dimensional embeddings to scalar predictions. It is
used as one of the candidate regressors in the
:mod:`torch_measure.experimental.content_aware.difficulty_regressor` model
selection tournament, and may be the winning model saved inside
``difficulty_reg_stage2a_output.pkl``.

Pickle compatibility
--------------------
The class must be importable at deserialization time on any machine that
loads a ``difficulty_reg_stage2a_output.pkl`` artifact produced by the
CS321M competition training pipeline. The attribute names (``net``,
``device``, ``hidden_dims``, ``dropout``, ``lr``, ``weight_decay``,
``epochs``) and the ``__init__`` signature must remain stable. Do not
rename or reorder them.

Device handling
---------------
Artifacts are typically trained on CUDA (Modal GPU) and loaded on CPU
(Codabench, local Mac). :meth:`predict` automatically remaps ``cuda``
to ``cpu`` when CUDA is unavailable, so no manual intervention is needed
after loading a GPU-trained pkl on a CPU machine.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLPRegressor:
    """Simple multi-layer perceptron with a scikit-learn-compatible interface.

    Maps fixed-dimensional input embeddings (e.g. 768-dimensional sentence
    embeddings) to scalar predictions via a stack of fully-connected layers
    with ReLU activations and dropout regularization.

    This class is intentionally minimal — it contains only the architecture
    and inference logic. All data loading, encoding, and model selection
    logic lives in
    :mod:`torch_measure.experimental.content_aware.difficulty_regressor`.

    Parameters
    ----------
    hidden_dims : tuple[int, ...]
        Width of each hidden layer. For example ``(256, 256)`` produces two
        hidden layers of width 256.
    dropout : float
        Dropout probability applied after each hidden layer's ReLU.
    lr : float
        Adam learning rate used during :meth:`fit`.
    weight_decay : float
        L2 regularization coefficient for Adam.
    epochs : int
        Number of full passes over the training data during :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> X_train = np.random.randn(100, 768).astype(np.float32)
    >>> y_train = np.random.randn(100).astype(np.float32)
    >>> reg = MLPRegressor(hidden_dims=(64,), dropout=0.1, epochs=2)
    >>> reg.fit(X_train, y_train)
    >>> preds = reg.predict(X_train[:5])
    >>> preds.shape
    (5,)
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (256, 256),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
    ) -> None:
        self.hidden_dims  = hidden_dims
        self.dropout      = dropout
        self.lr           = lr
        self.weight_decay = weight_decay
        self.epochs       = epochs
        self.net: nn.Sequential | None = None
        # Device is set at fit() time. Stored as a string so the object
        # remains picklable without torch.device serialization quirks.
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_net(self, input_dim: int) -> nn.Sequential:
        """Construct the network for the given input dimensionality."""
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(self.dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def _resolve_device(self) -> str:
        """Return the device to use for inference.

        If the model was saved on CUDA but the current machine has no GPU,
        remaps to CPU and moves the network weights accordingly.
        """
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            if self.net is not None:
                self.net = self.net.cpu()
        return self.device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPRegressor":
        """Train the MLP on embedding–target pairs.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input embeddings (e.g. sentence embeddings, float32).
        y : np.ndarray of shape (n_samples,)
            Regression targets (normalized difficulty z-scores, float32).

        Returns
        -------
        self
        """
        device = self._resolve_device()
        self.net = self._build_net(X.shape[1]).to(device)

        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        y_t = torch.tensor(y, dtype=torch.float32).to(device)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=512, shuffle=True
        )

        self.net.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(self.net(X_batch).squeeze(-1), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                avg = epoch_loss / len(loader)
                print(f"[MLPRegressor] epoch {epoch + 1}/{self.epochs}  loss={avg:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict scalar outputs for a batch of embeddings.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input embeddings. Must have the same number of features as the
            data passed to :meth:`fit`.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted scalar values (normalized difficulty z-scores).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called and ``self.net`` is ``None``.
        """
        if self.net is None:
            raise RuntimeError(
                "MLPRegressor.predict() called before fit(). "
                "Either call fit() or load a pre-trained artifact."
            )
        device = self._resolve_device()
        self.net.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            return self.net(X_t).squeeze(-1).cpu().numpy()
