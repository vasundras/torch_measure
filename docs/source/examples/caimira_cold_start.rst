Cold-start probabilities with ``CAIMIRA.predict_embeddings``
============================================================

:meth:`~torch_measure.models.CAIMIRA.predict_embeddings` scores items that
were **not** part of the training bank, by accepting their question
embeddings directly. This is the inference path you want whenever a
fitted CAIMIRA needs to estimate
:math:`P(\text{subject } i \text{ answers item } j \text{ correctly})`
for a fresh batch of items: the training-bank embeddings set by
:meth:`~torch_measure.models.CAIMIRA.set_embeddings` are not read or
mutated, and the model's learned coordinate system is reused via the
frozen difficulty-mean buffer.

Worked example
--------------

The snippet below fits a small CAIMIRA on a synthetic 20-subject /
30-item bank, then scores four cold-start items pulled from outside the
bank. ``predict_embeddings`` defaults to ``center='frozen'``, so the new
items are centred under the same learned mean that the training-bank
items were.

.. code-block:: python

   import torch
   from torch_measure.models import CAIMIRA

   torch.manual_seed(0)
   n_subjects, n_items, embed_dim = 20, 30, 16

   # 1. Train CAIMIRA on the training bank.
   model = CAIMIRA(n_subjects, n_items, embed_dim, latent_dim=3)
   train_embeddings = torch.randn(n_items, embed_dim)
   responses = torch.bernoulli(torch.full((n_subjects, n_items), 0.5))
   model.fit(responses, train_embeddings, max_epochs=10, verbose=False)

   # 2. Score a fresh batch of cold-start items.
   subject_idx = torch.tensor([0, 1, 2, 3])
   cold_embeddings = torch.randn(4, embed_dim)
   probs = model.predict_embeddings(subject_idx, cold_embeddings)
   print(probs.shape)  # torch.Size([4])

Why ``center='frozen'`` by default?
-----------------------------------

CAIMIRA's response equation
:math:`\sigma((s_i - d_j)^\top r_j)` is defined relative to a particular
zero-centering of difficulty. During training the centering mean is
**dynamic** -- recomputed each batch over the training bank, with
gradient flowing through the mean -- so the latent coordinate system
co-evolves with the parameters. At inference time, however, the mean is
fixed at the snapshot taken at the end of
:meth:`~torch_measure.models.CAIMIRA.fit` and stored in the persistent
``_difficulty_mean`` buffer.

Re-centering cold-start items on **their own** mean (i.e.
``center='dynamic'`` at inference) would silently move the latent origin
away from the trained one, perturbing every subject's relative skill
score and degrading calibration. ``center='frozen'`` is the paper-
faithful choice: cold-start items are projected into the SAME learned
coordinate system as the training bank, and the buffer survives
``state_dict()`` save / load so the contract holds across serialise /
deserialise.

See also
--------

* :meth:`torch_measure.models.CAIMIRA.compute_item_params` -- the
  building block that produces ``(relevance, difficulty)`` for an
  arbitrary embedding batch and exposes the ``center`` parameter
  directly.
* :meth:`torch_measure.models.CAIMIRA.set_embeddings` -- registers the
  training-bank embeddings used by :meth:`predict` (index-based path).
