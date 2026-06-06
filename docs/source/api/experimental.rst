Experimental
============

.. note::

   The :mod:`torch_measure.experimental` namespace collects models that are
   documented for completeness and reproducibility but are **not** part of
   the supported public API of ``torch_measure``. The class below
   self-discloses a convergent negative result; production code should
   prefer :class:`torch_measure.models.ColdStartLookupPredictor`.

.. automodule:: torch_measure.experimental
   :members:

LLM-Augmented IRT (negative result)
-----------------------------------

.. autoclass:: torch_measure.experimental.LLMJudgeIRT
   :members:
   :undoc-members:

.. autofunction:: torch_measure.experimental.build_difficulty_prompt
