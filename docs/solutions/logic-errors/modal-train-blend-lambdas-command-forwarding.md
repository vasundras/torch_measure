---
title: "Modal `blend_lambdas` metadata drift from the training command"
date: 2026-05-22
category: logic-errors
module: modal_train
problem_type: logic_error
component: tooling
symptoms:
  - "Operator-supplied `blend_lambdas` appeared in the Modal plan/config but were not passed to `submission/train.py`"
  - "`submission/train.py` did not accept `--blend-lambdas`, so a Modal-only forwarding fix would have failed the subprocess"
  - "`command.txt` gave no evidence that the operator-visible blend-lambda values were used during training"
root_cause: missing_workflow_step
resolution_type: code_fix
severity: medium
related_components:
  - submission_training
  - testing_framework
tags:
  - modal
  - subprocess-command
  - argument-forwarding
  - blend-lambdas
  - regression-test
---

# Modal `blend_lambdas` metadata drift from the training command

## Problem

`modal_train.py` accepted `blend_lambdas` through the Modal entrypoint, printed it in the operator plan, and persisted it to `config.json`, but the remote training subprocess never received the corresponding `--blend-lambdas` arguments. The result was a silent contract gap: operator-facing metadata said one thing, while `/root/submission/train.py` trained with its own defaults.

## Symptoms

- A custom `--blend-lambdas` value appeared in the Modal plan/config as `blend_lambdas`, giving the operator confidence that the setting was part of the run.
- The actual `command` list passed to `subprocess.run(...)` stopped at `--lr`, plus optional `--smoke` and `--benchmarks`; it never appended `--blend-lambdas`.
- The child trainer had no `--blend-lambdas` parser argument, so appending the flag only in Modal would have turned the silent no-op into an argparse failure.
- `command.txt` would not show the operator-supplied values, even though the surrounding run metadata did.

## What Didn't Work

Treating Modal plan/config output as evidence of training behavior was insufficient. Those surfaces only proved the entrypoint parsed the setting, not that the subprocess consumed it.

A one-line Modal-only fix was also incomplete. Appending `--blend-lambdas` to the command would have produced an argparse failure until `submission/train.py` learned the same CLI argument.

The prior-session scan found recent sessions for this repo, but no separate prior-session findings specific to this `blend_lambdas` command-forwarding bug.

## Solution

Centralize command construction in `modal_train.py` with `_build_train_command(...)`, then have `_remote_train_impl` call that helper before `subprocess.run(...)`.

The load-bearing behavior is:

```python
if blend_lambdas:
    command.append("--blend-lambdas")
    command.extend(str(lambda_) for lambda_ in blend_lambdas)
```

`_remote_train_impl` now passes the same parsed list into both config metadata and the command builder, so the plan/config path and execution path share one source of truth:

```python
command = _build_train_command(
    submission_dir=submission_dir,
    latent_dim=latent_dim,
    seed=seed,
    epochs=epochs,
    lr=lr,
    smoke=smoke,
    benchmarks=benchmarks,
    blend_lambdas=blend_lambdas,
)
```

`submission/train.py` also now accepts the flag:

```python
p.add_argument("--blend-lambdas", type=float, nargs="*", default=[0.6],
               help="Runtime CAIMIRA-vs-EB blend weights to carry into artifact metadata.")
```

The trainer persists the consumed value into `caimira_lite.meta.json`:

```python
"blend_lambdas": args.blend_lambdas,
```

Finally, `tests/test_modal_train.py` pins the forwarding behavior with `test_train_command_forwards_blend_lambdas`, asserting the constructed command includes:

```python
"--blend-lambdas", "0.3", "0.6", "0.9"
```

## Why This Works

The fix closes both halves of the drift. Modal no longer merely records `blend_lambdas`; it forwards the exact values to the child training process. The child process now has a matching argparse contract, so forwarding the flag is executable rather than aspirational.

Extracting `_build_train_command()` also makes the subprocess command unit-testable without launching Modal or running a training job. That is the right regression surface for this class of bug because the failure lived between "operator plan" and "actual command list."

## Prevention

- For artifact factories, test the command or API boundary that actually launches work, not just the high-level plan object.
- Any operator-visible option should have a regression test proving it crosses each boundary: entrypoint parse, plan/config metadata, subprocess command, child parser, and final artifact metadata.
- Keep command construction in a pure helper like `_build_train_command()` so tests can assert exact arguments without GPUs, network access, or remote execution.
- When adding a wrapper flag, update both sides of the boundary in the same change: the parent wrapper that forwards it and the child command that parses or persists it.

A useful invariant for future Modal wrappers is:

```text
operator_option in plan
operator_option in command
operator_option accepted by child parser
operator_option persisted by child metadata
```

## Related Issues

This bug is one of a cluster of "sibling-path drift" fixes on
`feat/caimira`, all sharing the theme "two paths that should converge
have silently diverged":

- `docs/solutions/logic-errors/modal-main-silent-smoke-default-subverts-kind-train.md`
  — Modal `main`'s `smoke: bool = True` default + `if kind == "smoke":`
  with no `else` made `--kind train` silently run in smoke mode. Sibling
  drift: Modal `main` vs local `cli`, both meant to translate
  `(kind, smoke, epochs)` into the same training plan.
- Commit `8add379` `fix(cold-start-lookup): predict_batch forwards labeled to predict()`
  — `predict_batch` called `self.calibrate(labeled)` but invoked
  `self.predict(r)` without forwarding `labeled`, so the Platt
  correction was fit and silently ignored. Sibling drift: `predict_batch`
  vs `predict`.
- Commit `931adf8` `fix(eb-priors): preserve 0.0 priors in level-3 IRT blend`
  — `or` short-circuit on a legitimately-`0.0` prior in BOTH
  `cold_start_lookup.ColdStartLookupPredictor._lookup_p` AND
  `caimira_lite.EBLookup.lookup_p`. Sibling drift: two copies of the
  same EB hierarchy that drifted on the same Python-truthy footgun.
- Commit `2d3bf8a` `fix(submission-model): cache _fit_base_logit_platt by len(labeled)`
  — module-level helper recomputed per-call while sibling implementations
  (`ColdStartLookupPredictor.calibrate`, `EBLookup.fit_platt`) both
  cached by `len(labeled)`. Sibling drift: three EB-style fitters that
  should share a caching contract.

All five issues are silent semantic mismatches between the operator's or
caller's declared intent and the actual downstream behavior. The
generalized prevention rule across the cluster: when two or more paths
share an intent-resolution or contract-forwarding step, extract a shared
helper (or single source of truth) and pin the contract with a regression
test on the helper itself rather than the wrapper.
