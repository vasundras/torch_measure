---
title: "Modal CAIMIRA training runtime failures"
date: 2026-05-23
last_updated: 2026-05-23
category: runtime-errors
module: modal_train
problem_type: runtime_error
component: tooling
symptoms:
  - "Wave 1 Modal jobs failed before training with `ModuleNotFoundError: No module named 'tabpfn'`"
  - "Wave 2 Modal jobs reached CAIMIRA MLE and failed with PyTorch Bernoulli Boolean-support validation on float 0/1 labels"
  - "Foreground `modal run` commands did not survive Cursor shell handoff, requiring detached Modal function calls plus separate volume sync"
  - "Wave 3 completed eight synced CAIMIRA artifacts only after both runtime gaps were fixed"
root_cause: config_error
resolution_type: code_fix
severity: high
related_components:
  - development_workflow
  - testing_framework
tags:
  - modal
  - caimira
  - tabpfn
  - pytorch
  - binary-cross-entropy
  - dependency-drift
---

# Modal CAIMIRA training runtime failures

## Problem

The Modal CAIMIRA training lane was not equivalent to the local training
environment. The first launch wave failed before training because the
Modal image did not include dependencies imported by `torch_measure.models`;
after that was fixed, the next wave failed inside CAIMIRA MLE because the
loss used a PyTorch distribution API that rejected float binary labels.

## Symptoms

- Wave 1 jobs produced no usable model files. `stderr.log` showed
  `ModuleNotFoundError: No module named 'tabpfn'` during
  `from torch_measure.models import CAIMIRA`, because package import also
  imported `TabPFNPredictor`.
- Wave 2 jobs got through item encoding and reached `MLE fitting`, then
  failed in `src/torch_measure/fitting/_losses.py` with:

```text
ValueError: Expected value argument ... to be within the support (Boolean())
of the distribution Bernoulli(probs: ...)
```

- The final Wave 3 relaunch completed eight synced artifacts as `hold` in
  `runs/modal/goldrush_20260523/modal_watch_state.json`, with the first
  two waves preserved as rejected audit records.

## What Didn't Work

Relaunching the same `modal run` commands was not enough. Cursor's shell
handoff aborted the foreground processes after the plan printed, so the
lane switched to detached `train_job.spawn(...)` calls and a separate
volume sync loop. That solved control-plane durability but did not solve
the runtime environment mismatch.

Adding only `tabpfn` also would not have been enough. The failing import
path pulled the package's broader declared dependency surface, and once
the image imported successfully, the training step exposed the separate
Bernoulli-loss bug. The previous-session transcript captured this as a
three-wave sequence: missing Modal client/image setup, dependency import
failure, then the PyTorch loss validation failure. (session history)

## Solution

The Modal image now installs the dependencies needed by the package import
surface exercised during CAIMIRA training:

```python
modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.1",
    "sentence-transformers==3.0.1",
    "transformers==4.44.2",
    "huggingface_hub==0.24.7",
    "pyarrow==17.0.0",
    "numpy==2.0.1",
    "pandas>=2.3",
    "pyro-ppl>=1.8",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "tueplots>=0.0.14",
    "tabpfn>=2.2,<3",
)
```

`bernoulli_nll()` now uses the tensor loss primitive that matches the
training data representation:

```python
import torch.nn.functional as F

return F.binary_cross_entropy(
    predicted_probs,
    observed.to(dtype=predicted_probs.dtype),
)
```

The Modal entrypoint regression test also unwraps Modal's
`LocalEntrypoint` wrapper before inspecting defaults, so installing Modal
locally no longer makes the test inspect only `(*args, **kwargs)`:

```python
main = getattr(getattr(main, "info", None), "raw_f", main)
sig = inspect.signature(main)
```

The fix was verified with:

```text
.venv/bin/python -m pytest tests/test_modal_train.py tests/test_models/test_caimira.py -q
36 passed
```

## Why This Works

The Modal image now matches the import-time dependency surface, not just
the narrow code path the training job intended to use. That matters in
this package because importing `torch_measure.models` imports multiple
model modules, including optional-looking predictors that still require
their dependencies at import time.

`F.binary_cross_entropy()` computes the Bernoulli negative log likelihood
for probability predictions against float `0.0/1.0` labels without going
through `torch.distributions.Bernoulli` sample validation. The previous
implementation was mathematically plausible but used the wrong API for
the tensor shape and dtype emitted by the CAIMIRA training path.

Unwrapping `main.info.raw_f` keeps the test pinned to the operator-facing
function defaults even when Modal is installed locally and decorates the
entrypoint object.

## Prevention

- Treat remote training images as production runtimes: install every
  dependency needed by import-time package initialization, not just the
  function body that seems relevant.
- Prefer `torch.nn.functional` loss functions for optimization over
  distribution `log_prob()` calls when the labels are already numeric
  tensors.
- When a remote job fails early, relaunch only after asking whether the
  fix merely reveals the next runtime stage. In this case, import success
  exposed MLE loss validation.
- For Modal `@app.local_entrypoint` tests, inspect the raw callable
  (`main.info.raw_f`) when Modal is installed, and keep a no-Modal path for
  environments where the decorator is absent.
- Preserve failed remote waves as audit artifacts instead of overwriting
  them; they explain why later waves changed.

## Related Issues

- `docs/solutions/logic-errors/modal-main-silent-smoke-default-subverts-kind-train.md`
  documents the earlier Modal entrypoint default bug. It overlaps in the
  Modal artifact-factory surface but has a different root cause.
- `docs/solutions/logic-errors/modal-train-blend-lambdas-command-forwarding.md`
  documents a sibling boundary-forwarding bug in the same wrapper.
- Parent-repo Modal workflow notes on detached spawns and control-plane
  durability are related to the launch/sync mechanism, but not to the
  dependency or loss-function runtime failures documented here.

