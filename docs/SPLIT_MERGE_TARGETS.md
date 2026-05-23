# Split-merge targets — `vasundras_torch_measure` fork

**Date:** 2026-05-22
**Patch:** PR #2 v2 (Lane C, API hygiene)
**Upstream:** `aims-foundations/torch_measure` (branch: `feat/caimira`)
**Fork base commit:** `4d7ef2f6` (branch: `feat/caimira` HEAD at fork time)

This fork accumulated three logically-distinct bodies of work:

1. **CAIMIRA model improvements** (paper-faithful centering, etc.) — these
   are the only changes intended for the upstream `feat/caimira` branch.
2. **CS321M Predictive Evaluation Challenge submission infrastructure**
   (Codabench competition 15934) — fork-only; CS321M-specific shape and
   contract assumptions.
3. **Operational / agent-process artifacts** (PR-review outputs, design
   patterns docs, run logs, postmortems) — fork-only; not source.

The split below is what `git format-patch` / cherry-pick should respect
when preparing the upstream PR. Lane C does not produce the actual
upstream patch series — it just enforces the **structural** preconditions
(API hygiene, namespace placement, docs/test layout) so that a clean
cherry-pick is possible later.

## Upstream-eligible files

These are the only paths that should land on
`aims-foundations/torch_measure:feat/caimira`.

| Path | Type | Rationale |
| --- | --- | --- |
| `src/torch_measure/models/caimira.py` | source | CAIMIRA model improvements (general-purpose; not CS321M-shaped). |
| `tests/test_models/test_caimira.py` | test | Direct test of `caimira.py`; mirrors the upstream test layout. |
| `docs/source/api/models.rst` (CAIMIRA `autoclass` block only) | docs | The `autoclass:: torch_measure.models.CAIMIRA` block stays; the surrounding "Predictive Evaluation Models" section is fork-only. |
| `docs/source/examples/*caimira*` (if added by Lane A) | docs | CAIMIRA usage examples. Lane A may or may not add these; this row is conditional on Lane A's output. |

## Fork-only files

These are CS321M-specific or operationally-scoped and must **never** land
upstream.

**Default rule (catch-all):** *Any path NOT explicitly listed in
"Upstream-eligible files" above is implicitly fork-only and must not be
cherry-picked.* The table below enumerates the load-bearing paths; the
default rule covers everything else (operator docs, IDE/agent artifacts,
CI workflows, etc.).

| Path | Rationale |
| --- | --- |
| `submission/` (entire directory: `model.py`, `caimira_lite.py`, `labeling.py`, `train.py`, `README.md`, `build_zip.sh`, `models.txt`, `*.pt`, `*.meta.json`, `*.json`) | CS321M Codabench-specific submission scaffold; not a general library feature. |
| `modal_train.py` | Modal-specific training entrypoint for CS321M. |
| `reviews/` | PR-review artifacts; not source. |
| `docs/solutions/` | CS321M / Codabench-specific design-pattern store; not framework docs. |
| `runs/`, `handoffs/` | Operational artifacts (postmortem-style files, leaderboard ledgers, transfer-audit reports); not source. |
| `src/torch_measure/models/cold_start_lookup.py` | CS321M-shaped (uses Codabench input dict shape, 909-subject closed set, K=5 Platt-shift cap); fork-only per Lane C decision. |
| `tests/test_models/test_cold_start_lookup.py` | Direct test of the fork-only `cold_start_lookup.py`. |
| `tests/test_models/test_predict_batch_calibration.py` | Exercises `cold_start_lookup.ColdStartLookupPredictor.predict_batch`; transitively fork-only. |
| `docs/source/api/models.rst` — `ColdStartLookupPredictor` `autoclass` block + the "Predictive Evaluation Models" section header + the fork-only cross-reference paragraph Lane C added | Fork-only API surface; renders the CS321M predictor. |
| `src/torch_measure/experimental/` (`LLMJudgeIRT` + `build_difficulty_prompt`) | Self-disclaimed convergent negative result; fork-only. The module docstring documents the negative result; the `torch_measure.experimental` namespace exists to prevent the symbol from accidentally surfacing on `torch_measure.models`. |
| `tests/test_experimental/` | Tests for the experimental namespace; mirrors the fork-only namespace placement. |
| `docs/source/api/experimental.rst` | Rendered API page for the experimental namespace; fork-only. |
| `docs/SPLIT_MERGE_TARGETS.md` (this file) | The split-merge ledger itself; meta-doc, not source. |
| `AGENTS.md`, `CLAUDE.md`, `HANDOFF.md` (root-level) | Operator orientation docs for CS321M agents; fork-only by construction. |
| `.claude/`, `.cursor/`, `.cursorindexingignore`, `.specstory/`, `.session_context/` | IDE / agent artifacts; fork-only by construction. |
| `.github/workflows/test.yml`, `.github/workflows/slow-tests.yml` | CI workflow tweaks specific to the fork's test layout (`tests/test_submission/`, `tests/test_experimental/`); upstream has its own CI. Re-evaluate per change before any cherry-pick. |
| `README.md` (the project-level README) | Fork-specific framing; upstream has its own README. |
| `tests/test_modal_train.py` | Tests for the fork-only `modal_train.py`; transitively fork-only. |
| `tests/test_submission/` (entire directory) | Tests for the fork-only `submission/` scaffold; transitively fork-only. |
| `src/torch_measure/models/__init__.py` — PEP 562 `__getattr__` shim block ONLY (lines that re-export `LLMJudgeIRT` / `build_difficulty_prompt` with `DeprecationWarning`) | Fork-only backward-compat shim; would re-introduce symbols upstream doesn't have. The rest of `__init__.py` (CAIMIRA export, etc.) is upstream-eligible. |

## Documentation cross-references

After this patch:

- `docs/source/index.rst` includes both `api/models` and `api/experimental`
  in the API Reference toctree. The upstream `feat/caimira` branch does
  NOT have `api/experimental` and should not pick up that toctree entry.
- `docs/source/api/models.rst` "Predictive Evaluation Models" section is
  fork-only (it renders `ColdStartLookupPredictor`, which is fork-only).
  The CAIMIRA `autoclass` block in the same file IS upstream-eligible.
- `src/torch_measure/models/__init__.py` carries a backward-compat shim
  (PEP 562 `__getattr__`) that re-exports `LLMJudgeIRT` and
  `build_difficulty_prompt` from `torch_measure.experimental` with a
  `DeprecationWarning`. The shim is fork-only and must NOT propagate
  upstream — it would re-introduce a symbol upstream doesn't have.

## How to apply the upstream split

When the team is ready to send the upstream PR against
`aims-foundations/torch_measure:feat/caimira`:

1. Identify the commits in this fork that touch ONLY the upstream-eligible
   files listed above. The simplest path:

   ```bash
   git log --oneline -- \
     src/torch_measure/models/caimira.py \
     tests/test_models/test_caimira.py
   ```

2. Cherry-pick or `git format-patch` those commits onto a clean branch
   forked from upstream `feat/caimira` HEAD. Use a path-restricted
   checkout to avoid accidentally dragging in fork-only files:

   ```bash
   git checkout upstream/feat/caimira -b upstream-caimira-pr
   git cherry-pick <commit-sha> -- \
     src/torch_measure/models/caimira.py \
     tests/test_models/test_caimira.py
   ```

3. For the `docs/source/api/models.rst` edits, hand-apply only the CAIMIRA
   `autoclass` lines — do NOT pick up the "Predictive Evaluation Models"
   section header or the `ColdStartLookupPredictor` block.

4. **Do NOT** cherry-pick:

   - Any commit that touches `submission/`, `modal_train.py`, `reviews/`,
     `docs/solutions/`, `runs/`, `handoffs/`, or `src/torch_measure/experimental/`.
   - The `tests/test_models/test_cold_start_lookup.py` and
     `tests/test_experimental/` test files.
   - The backward-compat shim block in `src/torch_measure/models/__init__.py`
     (the `__getattr__` function added by the 2026-05-22 patch — it re-exports
     symbols that don't exist upstream).

5. Before pushing the upstream PR, run a paranoia check that the diff
   doesn't contain any fork-only path:

   ```bash
   git diff upstream/feat/caimira... --stat | \
     grep -E '^(submission|modal_train|reviews|docs/solutions|runs|handoffs|src/torch_measure/experimental|src/torch_measure/models/cold_start_lookup|tests/test_experimental|tests/test_models/test_cold_start_lookup)' \
     && { echo "ABORT: fork-only path leaked into upstream diff" >&2; exit 1; } \
     || echo "OK: upstream diff is fork-clean"
   ```

If any fork-only path appears in that grep, stop and revisit the
cherry-pick set before pushing.
