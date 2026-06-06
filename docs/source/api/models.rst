Models
======

.. automodule:: torch_measure.models
   :members:

IRT Models
----------

.. autoclass:: torch_measure.models.Rasch
   :members:
   :undoc-members:

.. autoclass:: torch_measure.models.TwoPL
   :members:
   :undoc-members:

.. autoclass:: torch_measure.models.ThreePL
   :members:
   :undoc-members:

.. autoclass:: torch_measure.models.AmortizedIRT
   :members:
   :undoc-members:

.. autoclass:: torch_measure.models.CAIMIRA
   :members:
   :undoc-members:

See also :doc:`/examples/caimira_cold_start` for a worked example of
scoring cold-start items with
:meth:`~torch_measure.models.CAIMIRA.predict_embeddings`.

Predictive Evaluation Models
----------------------------

Experimental / negative-result predictive-evaluation models live in
:mod:`torch_measure.experimental` -- see :doc:`experimental` for the
rendered API page. The class below
(:class:`~torch_measure.models.ColdStartLookupPredictor`) is fork-only
infrastructure for the Stanford CS321M Predictive AI Evaluation
Challenge (Codabench competition 15934); see
``docs/SPLIT_MERGE_TARGETS.md`` for the upstream-eligible vs fork-only
file split.

.. autoclass:: torch_measure.models.ColdStartLookupPredictor
   :members:
   :undoc-members:

.. autoclass:: torch_measure.models.TabPFNPredictor
   :members:
   :undoc-members:

.. autoclass:: torch_measure.models.MultiFacetRasch
   :members:
   :undoc-members:

Beta IRT Models
---------------

.. autoclass:: torch_measure.models.BetaRasch
   :members:
   :undoc-members:

.. autoclass:: torch_measure.models.BetaTwoPL
   :members:
   :undoc-members:

Factor Models
-------------

.. autoclass:: torch_measure.models.LogisticFM
   :members:
   :undoc-members:

.. autoclass:: torch_measure.models.Bifactor
   :members:
   :undoc-members:

Rotation Utilities
------------------

.. autofunction:: torch_measure.models.varimax_rotation

.. autofunction:: torch_measure.models.promax_rotation

.. autofunction:: torch_measure.models.bifactor_rotation
