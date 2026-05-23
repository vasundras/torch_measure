---
title: "Modal `main` entrypoint silent smoke default subverts `--kind train`"
date: 2026-05-22
last_updated: 2026-05-22
category: logic-errors
module: modal_train
problem_type: logic_error
component: tooling
symptoms:
  - "`modal run modal_train.py::main --kind train --epochs 100` printed a plan with `kind: train` but `smoke: true`, and the subprocess command (logged to `command.txt`) contained `--smoke`"
  - Modal artifact metadata at `runs/modal/<run_id>/submission/caimira_lite.meta.json` carried smoke-shaped row counts (≤1000 per benchmark) on what should have been a full-train artifact
  - Sibling local `cli(...)` path produced the correct non-smoke plan for the same `--kind train` invocation — Modal vs CLI behavioral divergence on identical user intent
  - Downstream D-9 transfer-audit gate would have consumed Modal GPU minutes producing artifacts ineligible to enter the pre-submission stress-test gate as a `train` candidate
  - No error, no warning — silent semantic mismatch between declared `kind=train` intent and actual smoke-truncated output
root_cause: logic_error
resolution_type: code_fix
severity: high
related_components:
  - development_workflow
  - testing_framework
tags:
  - modal
  - cli-defaults
  - silent-failure
  - sibling-drift
  - shared-helper
  - default-arg-trap
---

# Modal `main` entrypoint silent smoke default subverts `--kind train`

## Problem

The Modal artifact-factory entrypoint `main` in `modal_train.py` declared
`smoke: bool = True` and the body's `if kind == "smoke":` block had no
`else`, so the documented full-train invocation
`modal run modal_train.py::main --kind train --epochs 100` silently
inherited `smoke=True` from the signature default, passed `--smoke` to
the `submission/train.py` subprocess, and produced a smoke-truncated
1000-row artifact instead of a full training run.

## Symptoms

- `modal run modal_train.py::main --kind train --epochs 100` printed a
  plan with `"kind": "train"` but `"smoke": true`; the subprocess command
  (logged to `command.txt`) contained `--smoke`.
- The artifact at `runs/modal/<run_id>/submission/caimira_lite.{pt,meta.json}`
  carried smoke-shaped metadata (`n_examples ≤ 1000` per benchmark instead
  of the full per-benchmark dataset).
- The sibling local `cli(...)` path (argparse, `--smoke` `action="store_true"`
  default `False`) behaved correctly, so the two entrypoints disagreed on
  the same `--kind train` invocation — a behavioral fork between the Modal
  CLI and the dry-run CLI.
- Downstream gates (D-9 transfer audit, no-network local smoke against the
  artifact) would have consumed Modal GPU minutes producing artifacts that
  could not legitimately enter the pre-submission stress-test gate as a
  "train" candidate.

## What Didn't Work

N/A — the bug was identified directly from code review during a focused
audit pass after the 12-finding multi-wave robust-pr-review (the newly
added `modal_train.py` was not in that review's original scope). No
prior fix attempts. (session history)

## Solution

Three coordinated changes in `modal_train.py` and one new regression
test file (commit `8f22924`).

**1. Extract a shared resolver at module scope.** The explicit
`return smoke, epochs` for the non-smoke branch is the load-bearing
fix — there is no fall-through path that lets a stale `smoke` default
leak through.

```python
def _resolve_smoke_and_epochs(kind: str, smoke: bool, epochs: int) -> tuple[bool, int]:
    """Reconcile ``kind`` / ``smoke`` / ``epochs`` into a coherent plan.
    ...
    """
    if kind == "smoke":
        return True, min(epochs, 5)
    return smoke, epochs
```

**2. Flip `main`'s default and route through the helper.**

BEFORE (from the commit diff):

```python
@app.local_entrypoint()
def main(
    kind: str = "smoke",
    ...
    smoke: bool = True,
    ...
) -> None:
    assert_no_codabench_capability()
    if kind not in {"smoke", "train"}:
        raise ValueError("--kind must be smoke or train")
    if kind == "smoke":
        smoke = True
        epochs = min(epochs, 5)
```

AFTER:

```python
@app.local_entrypoint()
def main(
    kind: str = "smoke",
    ...
    smoke: bool = False,
    ...
) -> None:
    assert_no_codabench_capability()
    if kind not in {"smoke", "train"}:
        raise ValueError("--kind must be smoke or train")
    smoke, epochs = _resolve_smoke_and_epochs(kind, smoke, epochs)
```

**3. Route `cli` through the same helper.** Replaces the previously
correct-but-duplicated `epochs = min(args.epochs, 5) if kind == "smoke" else args.epochs`
+ `"smoke": args.smoke or kind == "smoke"` inline pair with a single
helper call, eliminating the divergent-implementation risk.

**4. Pin the resolution with regression tests** (`tests/test_modal_train.py`,
11 test IDs):

- `test_resolve_smoke_and_epochs` — 7-row parametrized table covering
  every `(kind, smoke_in, epochs_in) → (smoke_out, epochs_out)`
  transition including the explicit-override `("train", True, 50, True, 50)`
  case.
- `test_cli_train_kind_does_not_force_smoke` — end-to-end through
  `cli(["--kind", "train", "--epochs", "100", "--dry-run"])`, asserts
  `plan["smoke"] is False` and `plan["epochs"] == 100`.
- `test_main_smoke_default_is_false` —
  `inspect.signature(main).parameters["smoke"].default is False`;
  skipped when Modal is not importable (because `main` is only defined
  inside the `if modal is not None` guard).

## Why This Works

**(a) Modal's `@app.local_entrypoint` makes the signature default the
CLI default.** Modal translates the decorated function's signature
directly into a CLI surface — each parameter becomes a flag, and the
Python default *is* the CLI default. So `smoke: bool = True` did not
behave like a normal Python kwarg that "the caller will override"; it
meant "every Modal invocation that does not explicitly say
`--no-smoke` will see `smoke=True`". The pre-fix author likely
reasoned about the function as a normal callable (where the inline
`if kind == "smoke": smoke = True` looked sufficient) and missed that
the same `smoke` parameter is also the user-facing default. Flipping
it to `smoke: bool = False` makes the safer state the implicit one and
forces opt-in for smoke mode on the train path.

**(b) `if`-without-`else` over an enum-like `kind` parameter is a
fragile pattern.** When the right-hand side mutates state
(`smoke = True`), the unwritten else branch isn't a no-op — it's
"preserve whatever was in scope", which couples behavior to *whatever
happens to be in scope at this line*. In this case, what was in scope
was the misleading signature default. A pure-function resolver with
`if ... return X / return Y` makes the else explicit, makes the
resolution side-effect-free, and removes the dependence on
local-variable shadowing of the parameter.

**(c) Sibling entrypoints sharing intent-resolution should share a
helper.** Pre-fix, `main` and `cli` both had to translate
`(kind, smoke, epochs)` into the same downstream training plan, but
the logic was inlined twice in two different styles (`main`:
imperative reassignment; `cli`: boolean-OR expression). `cli` happened
to be correct because argparse's `action="store_true"` default is
`False`, but the implementations had drifted enough that a future
change to one would not propagate to the other. Routing both through
`_resolve_smoke_and_epochs` makes the resolution policy a single
auditable function and the regression test on the helper covers both
call sites.

**(d) Module-scope helper is testable without Modal installed.**
The decision to lift `_resolve_smoke_and_epochs` out of `main` (rather
than just adding the `else` branch inside the `@app.local_entrypoint`-
decorated function) makes the resolution policy a top-level callable
that runs from a plain `cli` import. The
`test_resolve_smoke_and_epochs` parametrized table runs in CI on every
commit even though Modal is not installed in the local venv — the
`test_main_smoke_default_is_false` signature pin then layers a second
guard that only activates when Modal IS installed. An inline-`else`
fix would have left the resolution policy un-testable in
Modal-less environments. (session history)

## Prevention

1. **For Modal `@app.local_entrypoint` (and any framework where the
   function signature *is* the CLI surface): default boolean flags to
   the safer-on-omission value.** Here that meant `smoke: bool = False` —
   the operator who wants a smoke run can pass `--smoke` (or use
   `--kind smoke`), but a plain invocation produces the full artifact
   rather than a silently-truncated one. Apply the same rule to any
   flag where the unsafe direction is "do less work and produce a
   degraded artifact" (e.g. `dry_run`, `mock`, `fast`, `sample`).
2. **When extending a CLI entrypoint with an enum-like
   `kind`/`mode`/`type` arg, extract a `_resolve_*(kind, ...) -> tuple[...]`
   pure helper** that takes the enum + caller flags and returns the
   resolved tuple. Do not write `if kind == "X": <mutate>` in the
   entrypoint body — even with a correct `else`, that pattern hides
   the resolution policy inside the orchestration code and resists
   testing.
3. **For every `if x == "literal":` branch that mutates state, write
   the `else` explicitly or convert to a single-expression resolver.**
   Even if the else is "leave alone", the author of the next change
   should see it as a deliberate decision rather than a fall-through.
4. **When two sibling entrypoints (here: Modal `main` + local `cli`)
   share an intent-resolution step, route both through the same helper
   before writing tests.** Then a parametrized table-driven test on
   the helper covers both call sites with one assertion surface, and
   the two entrypoints cannot drift in opposite directions during
   later edits.
5. **Pin the signature default with an
   `inspect.signature(...).parameters[...].default` assertion** when
   the default is load-bearing and the function is exposed as a CLI by
   signature reflection (Modal, Typer, Click via `params`, FastAPI
   dependency callables). This catches the regression class "someone
   flipped the default back" that the table-driven helper test cannot
   catch on its own.

## Related Issues

This bug is one of several structurally similar "sibling-path
drift" bugs surfaced in the same `feat/caimira` review session — all
on this branch and sharing the theme "two paths that should converge
have silently diverged":

- `8add379` `fix(cold-start-lookup): predict_batch forwards labeled to predict()`
  — `predict_batch` called `self.calibrate(labeled)` but invoked
  `self.predict(r)` without forwarding `labeled`, so the Platt
  correction was fit and silently ignored. Sibling drift:
  `predict_batch` vs `predict`.
- `931adf8` `fix(eb-priors): preserve 0.0 priors in level-3 IRT blend`
  — `or` short-circuit on a legitimately-`0.0` prior in BOTH
  `cold_start_lookup.ColdStartLookupPredictor._lookup_p` AND
  `caimira_lite.EBLookup.lookup_p`. Sibling drift: two copies of the
  same EB hierarchy that drifted on the same Python-truthy footgun.
- `2d3bf8a` `fix(submission-model): cache _fit_base_logit_platt by len(labeled)`
  — module-level helper recomputed per-call while sibling
  implementations (`ColdStartLookupPredictor.calibrate`,
  `EBLookup.fit_platt`) both cached by `len(labeled)`. Sibling drift:
  three EB-style fitters that should share a caching contract.
- `8f22924` (this fix) — Modal `main` entrypoint silent smoke default.
  Sibling drift: Modal `main` vs local `cli`, both meant to translate
  `(kind, smoke, epochs)` into the same training plan.
- `f1ac3b7` / `modal-train-blend-lambdas-command-forwarding.md` —
  Modal accepted `blend_lambdas` in operator plan/config metadata but
  did not forward the values into the `submission/train.py` subprocess
  command. Sibling drift: operator-visible run metadata vs actual child
  process arguments.

The May-22 12-finding robust-pr-review caught the first three but
**not** the Modal default-arg variant (the new `modal_train.py` was
out of that review's scope). This doc closes that gap and adds the
`inspect.signature` pin as a structural defense against the regression
class. (session history)

Closely related parent-repo learnings at
`../../../../docs/solutions/` (the CS321M Stanford-parent knowledge
store):

- `design-patterns/distinguishable-defensive-fallbacks-2026-05-18.md`
  — generalization of "default behavior should be the safer value so
  silent fallback is distinguishable from explicit choice". This new
  doc is the **default-arg + missing-else** variant of that pattern;
  consider adding the Modal example as a third canonical instance
  alongside the silent-NCF-load-failure and silent-`0.5`-fallback
  cases.
- `logic-errors/labeling-py-nan-fallback-bomb-2026-05-17.md` — closest
  structural sibling: "sibling-path drift missed by code-reading; only
  force-injection unit tests caught it". Prevention practice #5 here
  (the `inspect.signature(...).parameters[...].default` pin) is the
  CLI-surface analogue of that doc's "force-injection unit tests for
  every `return` site" rule.

## References

- Commit: `8f22924` on `feat/caimira` of `vasundras/torch_measure`
- Modified files: `modal_train.py` (helper extraction + default flip +
  `cli` routing), `tests/test_modal_train.py` (new, 11 test IDs)
- Local test run: `.venv/bin/python -m pytest tests/test_modal_train.py -q`
  → 10 passed, 1 skipped (`test_main_smoke_default_is_false` skipped
  because Modal is not installed in the local venv; it runs on any
  host with Modal installed)
- Modal docs (referenced for the signature-as-CLI-default claim):
  Modal's `@app.local_entrypoint` decorator exposes function
  parameters as CLI flags with the Python defaults as the CLI defaults.
