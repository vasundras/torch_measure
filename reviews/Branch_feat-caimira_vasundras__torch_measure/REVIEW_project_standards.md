# REVIEW_project_standards

Reviewer lane: **project-standards** (Wave 1).
Branch: `feat/caimira` on `vasundras/torch_measure` (fork of `aims-foundations/torch_measure`).
Schema: `CRIT-STD-N`.

```json
{
  "reviewer": "project-standards",
  "findings": [
    {
      "id": "CRIT-STD-1",
      "title": "Missing Sphinx doc entries for new public APIs (CAIMIRA, ColdStartLookupPredictor, LLMJudgeIRT, build_difficulty_prompt)",
      "severity": "P1",
      "category": "upstream-eligible / CONTRIBUTING.md hard requirement",
      "confidence": 100,
      "anchored_confidence_rubric": "anchor 100 — verbatim CONTRIBUTING.md rule + mechanical absence in the source files",
      "rule_citation": {
        "file": "CONTRIBUTING.md",
        "lines": "33-38",
        "quote": "All new public APIs need: A numpy-style docstring with a short usage example; A corresponding entry in the Sphinx docs under `docs/source/`; At least one test, with appropriate pytest markers (`slow`, `gpu`, `network`) for expensive cases."
      },
      "violation": {
        "diff_anchors": [
          "src/torch_measure/models/__init__.py:13 — `from torch_measure.models.caimira import CAIMIRA` (new export)",
          "src/torch_measure/models/__init__.py:14 — `from torch_measure.models.cold_start_lookup import ColdStartLookupPredictor` (new export)",
          "src/torch_measure/models/__init__.py:17 — `from torch_measure.models.llm_judge_irt import LLMJudgeIRT, build_difficulty_prompt` (new exports)",
          "src/torch_measure/models/__init__.py:37-39, 46 — added to `__all__`"
        ],
        "missing_files": [
          "docs/source/api/models.rst — no `.. autoclass:: torch_measure.models.CAIMIRA` block",
          "docs/source/api/models.rst — no `.. autoclass:: torch_measure.models.ColdStartLookupPredictor` block",
          "docs/source/api/models.rst — no `.. autoclass:: torch_measure.models.LLMJudgeIRT` block",
          "docs/source/api/models.rst — no `.. autofunction:: torch_measure.models.build_difficulty_prompt` block"
        ],
        "evidence_of_existing_pattern": "docs/source/api/models.rst:10-63 lists every pre-existing public class (Rasch, TwoPL, ThreePL, AmortizedIRT, TabPFNPredictor, MultiFacetRasch, BetaRasch, BetaTwoPL, LogisticFM, Bifactor) as `.. autoclass::` blocks under domain headings, and each top-level helper (varimax_rotation, promax_rotation, bifactor_rotation) under `Rotation Utilities`. The 4 new public symbols MUST be added to match the established pattern. The `.. automodule:: torch_measure.models :members:` directive at line 4 does NOT auto-cover the new classes for a curated layout — the existing file deliberately enumerates each class under a section header (IRT Models / Beta IRT Models / Factor Models / Rotation Utilities)."
      },
      "scope_class": "upstream-eligible",
      "remediation": {
        "file": "docs/source/api/models.rst",
        "proposed_diff": "Add a new section before or after `Factor Models` (e.g. `Content-Aware IRT` for CAIMIRA, `Cold-Start / Empirical-Bayes Predictors` for ColdStartLookupPredictor + LLMJudgeIRT + build_difficulty_prompt). Stub:\n\n```rst\nContent-Aware IRT\n-----------------\n\n.. autoclass:: torch_measure.models.CAIMIRA\n   :members:\n   :undoc-members:\n\nCold-Start Predictors\n---------------------\n\n.. autoclass:: torch_measure.models.ColdStartLookupPredictor\n   :members:\n   :undoc-members:\n\n.. autoclass:: torch_measure.models.LLMJudgeIRT\n   :members:\n   :undoc-members:\n\n.. autofunction:: torch_measure.models.build_difficulty_prompt\n```"
      }
    },
    {
      "id": "CRIT-STD-2",
      "title": "LLMJudgeIRT and build_difficulty_prompt lack a docstring usage example (no `>>>` snippet)",
      "severity": "P1",
      "category": "upstream-eligible / CONTRIBUTING.md hard requirement",
      "confidence": 100,
      "anchored_confidence_rubric": "anchor 100 — verbatim rule + mechanical absence verified by grep",
      "rule_citation": {
        "file": "CONTRIBUTING.md",
        "lines": "35",
        "quote": "A numpy-style docstring with a short usage example"
      },
      "violation": {
        "diff_anchors": [
          "src/torch_measure/models/llm_judge_irt.py:54-74 — `build_difficulty_prompt`: docstring has no `Examples` section and no `>>>` doctest snippet",
          "src/torch_measure/models/llm_judge_irt.py:77-107 — `LLMJudgeIRT` class docstring has Parameters / Notes but no `Examples` block and no `>>>` snippet"
        ],
        "evidence": "`grep -c '>>>' src/torch_measure/models/llm_judge_irt.py` = 0; `grep 'Examples' src/torch_measure/models/llm_judge_irt.py` returns no hits. By contrast, `caimira.py:92-106` and `cold_start_lookup.py:190-209` both carry an `Examples` section with a runnable `>>>` snippet, demonstrating the project's docstring convention. LLMJudgeIRT is the sibling new public class to those two and is the only one without an Examples block."
      },
      "scope_class": "upstream-eligible",
      "remediation": {
        "file": "src/torch_measure/models/llm_judge_irt.py",
        "proposed_diff": "Add an `Examples` block to the `LLMJudgeIRT` class docstring (before `Notes`) using a minimal lookup + a trivial deterministic judge_fn so the snippet is reproducible without an LLM call. Suggested skeleton:\n\n```python\n    Examples\n    --------\n    >>> from torch_measure.models import (\n    ...     ColdStartLookupPredictor, LLMJudgeIRT,\n    ... )\n    >>> lookup = ColdStartLookupPredictor(\n    ...     sbc={}, sb={},\n    ...     subj={'gpt-4': 0.75},\n    ...     bench={'mmlupro': 0.50},\n    ...     global_mean=0.6,\n    ...     name_aliases={}, name_lc={'gpt-4': 'gpt-4'},\n    ... )\n    >>> def stub_judge(item, bench):\n    ...     return 0.0  # neutral difficulty\n    >>> model = LLMJudgeIRT(lookup, stub_judge, alpha=0.5, beta=0.0)\n    >>> record = {\n    ...     'subject_content': 'Name: gpt-4',\n    ...     'benchmark': 'mmlupro', 'condition': 'none',\n    ...     'item_content': 'q',\n    ... }\n    >>> p = model.predict(record)\n    >>> 0 < p < 1\n    True\n```\n\nFor `build_difficulty_prompt`, add a short snippet to its docstring:\n\n```python\n    Examples\n    --------\n    >>> prompt = build_difficulty_prompt('What is 2+2?', 'mmlupro')\n    >>> prompt.rstrip().endswith('Answer:')\n    True\n```"
      }
    },
    {
      "id": "CRIT-STD-3",
      "title": "Submission/ tree is invisible to CI lint (`ruff check src/ tests/` scope)",
      "severity": "P2",
      "category": "fork-only / CI configuration drift",
      "confidence": 100,
      "anchored_confidence_rubric": "anchor 100 — workflow file is exact-quoted; submission/ contains 4 new .py files; the README.md explicitly acknowledges the gap",
      "rule_citation": {
        "file": ".github/workflows/lint.yml",
        "lines": "30, 33",
        "quote": "run: ruff check src/ tests/\\n          ...\\n      - name: Ruff format check\\n        run: ruff format --check src/ tests/"
      },
      "violation": {
        "diff_anchors": [
          "submission/model.py:1 (258 lines) — new file, not in lint scope",
          "submission/labeling.py:1 (222 lines) — new file, not in lint scope",
          "submission/caimira_lite.py:1 (408 lines) — new file, not in lint scope",
          "submission/train.py:1 (458 lines) — new file, not in lint scope",
          "submission/README.md:168-173 — Open follow-ups: 'CI lint coverage. The repo\\'s `.github/workflows/lint.yml:30,33` only runs `ruff check src/ tests/` — this `submission/` directory is invisible to CI lint. A follow-up should either (a) extend the lint workflow or (b) document the contributor-side `ruff check submission/` step.'"
        ]
      },
      "scope_class": "fork-only",
      "note": "This is fork-only because `submission/` is not part of the upstream library (it is CS321M-specific scaffolding); upstream PRs would never include it. The fork's own CI loop is what fails to catch regressions here.",
      "remediation": {
        "preferred_option_a": "Extend lint.yml to include `submission/`:\n\n```yaml\n      - name: Ruff check\n        run: ruff check src/ tests/ submission/\n\n      - name: Ruff format check\n        run: ruff format --check src/ tests/ submission/\n```\n\n(Note: this requires `submission/` to be ruff-clean — verify locally with `ruff check submission/` first.)",
        "alternative_option_b": "Add a top-level Makefile target or contributor-facing note in submission/README.md naming the exact pre-push command (`ruff check submission/ && ruff format --check submission/`)."
      }
    },
    {
      "id": "CRIT-STD-4",
      "title": "Test workflow runs submission tests but won't load the gitignored .pt artifact — verify LOCAL_SMOKE guard is comprehensive",
      "severity": "P3",
      "category": "fork-only / CI test compatibility",
      "confidence": 75,
      "anchored_confidence_rubric": "anchor 75 — `.github/workflows/test.yml:37` literally runs `pytest tests/ -v ...` which will pick up `tests/test_submission/`; the LOCAL_SMOKE branch is invoked in `tests/test_submission/test_model_contract.py:25` but `tests/test_submission/test_caimira_lite.py` does NOT use LOCAL_SMOKE — manual review confirmed the latter never imports `model.py` (it imports `caimira_lite` directly, which has no artifact dependency) so the absence is benign, but the contract is non-obvious",
      "rule_citation": {
        "file": ".github/workflows/test.yml",
        "lines": "37",
        "quote": "run: pytest tests/ -v --cov=torch_measure --cov-report=xml -m \"not slow and not gpu\""
      },
      "violation": {
        "diff_anchors": [
          "tests/test_submission/test_model_contract.py:25 — sets `PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1` before importing model.py (guarded — OK)",
          "tests/test_submission/test_caimira_lite.py:24 — imports `caimira_lite` directly (no artifact dependency — OK)",
          "submission/model.py:136-163 — module-level load WILL fail without LOCAL_SMOKE=1 because `caimira_lite.pt`, `caimira_lite.meta.json`, `eb_tables.json` are gitignored and absent in CI"
        ],
        "evidence": "Inspected both test files: test_caimira_lite.py imports `from caimira_lite import (CAIMIRALite, EBLookup, clip_for_predict, parse_subject_name, resolve_subject_name)` and constructs new instances directly — no `caimira_lite.pt` load. test_model_contract.py sets LOCAL_SMOKE=1 before every `_load_model_module()` call. Both should pass cleanly in CI.",
        "residual_risk": "If a future test under tests/test_submission/ imports submission.model directly without setting LOCAL_SMOKE, the CI run will crash at module import (META path doesn't exist → FileNotFoundError). The CONTRIBUTING.md rule 'No committed data files' is what blocks shipping the artifact; the LOCAL_SMOKE escape hatch is the project's chosen workaround but it is contributor-discipline, not test-framework-enforced."
      },
      "scope_class": "fork-only",
      "remediation": "Recommend a `tests/test_submission/conftest.py` autouse fixture that sets `PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1` for every test under that directory, so a future contributor cannot forget. Stub:\n\n```python\nimport os\nimport pytest\n\n@pytest.fixture(autouse=True)\ndef _force_local_smoke(monkeypatch):\n    monkeypatch.setenv('PREDICTIVE_EVAL_LOCAL_SMOKE_TEST', '1')\n    yield\n```\n\nThis is fork-only because the LOCAL_SMOKE env var is competition-specific."
    },
    {
      "id": "CRIT-STD-5",
      "title": "README.md doesn't mention CAIMIRA, ColdStartLookupPredictor, or LLMJudgeIRT in the feature blurb",
      "severity": "P3",
      "category": "upstream-eligible / documentation completeness",
      "confidence": 75,
      "anchored_confidence_rubric": "anchor 75 — README.md is a 29-line marketing surface that names every other major capability area; the new content-aware IRT model is substantive enough to warrant a one-line mention. Not strictly a CONTRIBUTING.md rule but an upstream-PR-quality concern.",
      "rule_citation": {
        "file": "CONTRIBUTING.md",
        "lines": "116",
        "quote": "Docs under `docs/source/` updated if you changed user-facing behavior"
      },
      "violation": {
        "diff_anchors": [
          "README.md:10 — current blurb: 'Built on PyTorch, with GPU-accelerated IRT, factor models, amortized inference, adaptive testing, and tabular baselines.'",
          "README.md (unchanged in diff) — CAIMIRA is a new content-aware multidimensional IRT model published at EMNLP 2024; it is a meaningful new capability category, not a minor enhancement to an existing model."
        ]
      },
      "scope_class": "upstream-eligible",
      "note": "CONTRIBUTING.md:116 names docs/source/ explicitly; README.md is not strictly required. However, the existing README enumerates every capability area and CAIMIRA is substantive enough to add. Treat this as a soft recommendation rather than a hard violation.",
      "remediation": {
        "file": "README.md",
        "proposed_diff": "Adjust line 10 to include CAIMIRA, e.g.:\n\n`Built on PyTorch, with GPU-accelerated IRT (Rasch / 2PL / 3PL / multifaceted), content-aware multidimensional IRT (CAIMIRA), factor models, amortized inference, adaptive testing, and tabular baselines.`"
      }
    },
    {
      "id": "CRIT-STD-6",
      "title": "Commit messages on feat/caimira are multi-line — CONTRIBUTING.md prefers single-line; mitigated by squash-merge default",
      "severity": "P3",
      "category": "upstream-eligible / commit-message convention",
      "confidence": 75,
      "anchored_confidence_rubric": "anchor 75 — CONTRIBUTING.md line 117 is explicit; the 4 commits are observably multi-line; CONTRIBUTING.md line 123 explicitly says squash-merge is default which neutralizes most of the impact",
      "rule_citation": {
        "file": "CONTRIBUTING.md",
        "lines": "117, 123",
        "quote": "Commit messages are short, single-line, and describe the change\\n[...]\\nWe squash-merge PRs by default."
      },
      "violation": {
        "diff_anchors": [
          "4d7ef2f — 'Document D-9 smoke-gate run + coverage gap on sub-gate (a)' (subject line OK) followed by ~30-line body",
          "89b1d61 — 'Gitignore submission/ trainer artifacts + Codabench ZIPs' (subject line OK) + body",
          "1e889f2 — 'Add Codabench submission scaffolding (CAIMIRA + EB hybrid back-stop)' + body",
          "1e4ec1c — 'Add CAIMIRA content-aware multidimensional IRT model' + body"
        ],
        "note": "Each commit's subject line IS single-line and describes the change. The bodies are long but conventional Git practice (subject ≤72 chars + blank line + body) does not violate CONTRIBUTING.md as long as the subject is single-line and describes the change. With squash-merge as the default, the final upstream commit will have ONE subject line authored by the maintainer at merge time, so this is a low-impact note."
      },
      "scope_class": "upstream-eligible",
      "remediation": "No action required if the PR will be squash-merged. If the contributor wants to play it safe for a multi-commit PR, they can:\n  (a) keep the 4 commits + rely on squash-merge (CONTRIBUTING.md:123),\n  (b) or interactive-rebase before pushing to consolidate the work into 2-3 focused commits with terse subjects.\n\nFlag for the contributor's awareness, not a blocking issue."
    },
    {
      "id": "CRIT-STD-7",
      "title": "submission/README.md cross-references a path outside this fork (../predictive-eval-competition/scripts/run_d9_gate.py)",
      "severity": "P2",
      "category": "fork-only / reproducibility",
      "confidence": 100,
      "anchored_confidence_rubric": "anchor 100 — the README quote is exact, and the referenced path does not exist within the fork's repo root",
      "rule_citation": {
        "file": "CONTRIBUTING.md",
        "lines": "Implicit — the repo is intended to be self-contained per the standard open-source convention that a clone of vasundras/torch_measure is sufficient to reproduce the documented flow. No explicit rule line, so this is a soft P2.",
        "quote": "(no direct quote; implicit from the README's purpose as a contributor on-ramp)"
      },
      "violation": {
        "diff_anchors": [
          "submission/README.md:86 — `python ../predictive-eval-competition/scripts/run_d9_gate.py submission_caimira.zip`",
          "submission/README.md:155-156 — references `../docs/solutions/architecture-patterns/pre-submission-transfer-audit-stress-test-gate-2026-05-22.md`"
        ],
        "evidence": "Both relative paths assume a sibling repository layout (the reviewer's `CS321M/predictive-eval-competition/` parent). Someone cloning only `vasundras/torch_measure` will not find these files. The submission/README.md is otherwise self-contained — these two cross-references are the load-bearing exceptions."
      },
      "scope_class": "fork-only",
      "remediation": {
        "options": [
          "(a) Vendor `run_d9_gate.py` and the architecture-patterns doc into `submission/scripts/` and `submission/docs/` so the flow is self-contained.",
          "(b) Add an explicit prerequisite note at the top of submission/README.md naming the sibling repo dependency: '> **Reproduction note:** The D-9 gate script lives in a sibling repo at `../predictive-eval-competition/`. To reproduce the full flow, clone that repo at the same parent directory level. The submission ZIP can still be built and validated without it; D-9 is an extra-layer transfer-safety check used by the CS321M team and not strictly required to upload to Codabench.'",
          "(c) Replace the cross-references with stubs that describe the gate's four sub-gates (a/b/c/d) inline so the README is self-contained for someone who doesn't have the sibling repo."
        ]
      }
    },
    {
      "id": "CRIT-STD-8",
      "title": "submission/model.py uses `# noqa: S101` and `# noqa: SLF001` — rule codes are not in the project's ruff select",
      "severity": "P4",
      "category": "fork-only / lint configuration hygiene",
      "confidence": 100,
      "anchored_confidence_rubric": "anchor 100 — pyproject.toml exact-quotes the select list; the noqa codes are observably outside it",
      "rule_citation": {
        "file": "pyproject.toml",
        "lines": "99-101",
        "quote": "[tool.ruff.lint]\\nselect = [\"E\", \"F\", \"W\", \"I\", \"UP\", \"B\", \"SIM\"]\\nignore = [\"E501\"]"
      },
      "violation": {
        "diff_anchors": [
          "submission/model.py:223 — `# noqa: S101` (S = flake8-bandit, NOT in select)",
          "submission/model.py:254 — `# noqa: SLF001 — same module's class` (SLF = flake8-self, NOT in select)",
          "submission/model.py:255 — `# noqa: SLF001`",
          "submission/train.py:390 — `# noqa: E402,WPS433` (WPS = wemake-python-styleguide, NOT in select; ruff doesn't even implement WPS rules)"
        ],
        "evidence": "The repo's ruff config selects only E/F/W/I/UP/B/SIM. S, SLF, and WPS rules are not active, so the noqa comments are no-ops. Ruff in `--fix` mode could (with the right flag) report these as `RUF100` (unused noqa) violations — but RUF is also not in select, so this is currently dormant. Harmless but misleading: a future reader assumes the rule is enforced when it is not."
      },
      "scope_class": "fork-only",
      "note": "submission/ is not linted by CI today (see CRIT-STD-3); if/when it is added to the lint scope, these noqa comments should either (a) be removed as unused, or (b) the corresponding rules should be added to `select` if the contributor wants the protection.",
      "remediation": "If the contributor wants to keep these protections, propose adding `S` (bandit), `SLF` (private member access), and dropping `WPS` (not implementable in ruff) to the ruff select. Otherwise remove the noqa comments. Lowest-friction path: drop the noqas, since the patterns they cover (`assert` for type narrowing, accessing `EB._platt` from `model.py` where `EB` is a same-module class) are intentional and not actual code smells under the current ruleset."
    },
    {
      "id": "CRIT-STD-9",
      "title": "submission/ Python code is not covered by coverage tool (`source = [\"src/torch_measure\"]`)",
      "severity": "P3",
      "category": "fork-only / coverage configuration",
      "confidence": 100,
      "anchored_confidence_rubric": "anchor 100 — coverage source list is exact-quoted; submission/ is structurally outside src/torch_measure",
      "rule_citation": {
        "file": "pyproject.toml",
        "lines": "106-108",
        "quote": "[tool.coverage.run]\\nsource = [\"src/torch_measure\"]\\nomit = [\"*/tests/*\", \"*/_version.py\"]"
      },
      "violation": {
        "diff_anchors": [
          "submission/model.py:1 (258 lines)",
          "submission/labeling.py:1 (222 lines)",
          "submission/caimira_lite.py:1 (408 lines)",
          "tests/test_submission/test_model_contract.py + test_caimira_lite.py exist (404 LoC of tests) but their target code is invisible to `--cov=torch_measure`."
        ],
        "evidence": "The .github/workflows/test.yml:37 invocation passes `--cov=torch_measure` (just the package name), and the [tool.coverage.run] source includes only src/torch_measure. Coverage of submission/ code is therefore 0% from CI's perspective — the tests pass and exercise it, but the Codecov upload doesn't see it."
      },
      "scope_class": "fork-only",
      "note": "This is CORRECT behavior for upstream coverage reporting (submission/ isn't part of the published package). It is only a finding because the contributor may believe submission/ is covered when it isn't.",
      "remediation": "No change required. Document the limitation in submission/README.md if useful: 'Coverage of submission/ code is intentionally not reported to Codecov because it is fork-specific and outside the package source root. Run `pytest tests/test_submission/ -v` locally to verify changes manually.'"
    },
    {
      "id": "CRIT-STD-10",
      "title": "License headers present and consistent across all new files",
      "severity": "INFO",
      "category": "compliance verified",
      "confidence": 100,
      "anchored_confidence_rubric": "anchor 100 — verified by reading line 1 of every new file",
      "rule_citation": {
        "file": "(no explicit CONTRIBUTING.md rule; project convention)",
        "lines": "—",
        "quote": "Existing files in src/torch_measure/models/ all carry `# Copyright (c) 2026 AIMS Foundations. MIT License.` on line 1."
      },
      "verification": {
        "checked_files": [
          "src/torch_measure/models/caimira.py:1 — present",
          "src/torch_measure/models/cold_start_lookup.py:1 — present",
          "src/torch_measure/models/llm_judge_irt.py:1 — present",
          "submission/model.py:1 — present",
          "submission/caimira_lite.py:1 — present",
          "submission/labeling.py:1 — present",
          "submission/train.py:1 — present",
          "submission/build_zip.sh:1-2 — `#!/usr/bin/env bash` then license comment (idiomatic)",
          "tests/test_models/test_caimira.py:1 — present",
          "tests/test_models/test_cold_start_lookup.py:1 — present",
          "tests/test_models/test_llm_judge_irt.py:1 — present",
          "tests/test_submission/test_caimira_lite.py:1 — present",
          "tests/test_submission/test_model_contract.py:1 — present"
        ],
        "verdict": "PASS — all 13 new code/test files carry the license header."
      },
      "scope_class": "upstream-eligible",
      "remediation": "None — informational confirmation only."
    },
    {
      "id": "CRIT-STD-11",
      "title": "Tutorial notebook embedded outputs are within pre-commit large-file budget",
      "severity": "INFO",
      "category": "compliance verified",
      "confidence": 100,
      "anchored_confidence_rubric": "anchor 100 — filesize 28344 bytes (~28 KB) is well under the 500 KB hook limit",
      "rule_citation": {
        "file": ".pre-commit-config.yaml",
        "lines": "9-10",
        "quote": "      - id: check-added-large-files\\n        args: [\"--maxkb=500\"]"
      },
      "verification": {
        "filesize": "28344 bytes (~28 KB)",
        "output_cells": "10 cells with embedded outputs (per `grep -c '\"output_type\"'`), all text/plain or trivially small",
        "verdict": "PASS — well below the 500 KB threshold; pre-commit would not have flagged this. No need for `--no-verify` to commit it."
      },
      "scope_class": "fork-only",
      "remediation": "None — informational confirmation only."
    },
    {
      "id": "CRIT-STD-12",
      "title": "New tests carry no pytest markers — verify slow/network/gpu eligibility",
      "severity": "P2",
      "category": "upstream-eligible / CONTRIBUTING.md marker requirement",
      "confidence": 75,
      "anchored_confidence_rubric": "anchor 75 — CONTRIBUTING.md rule is verbatim; the new tests have ZERO markers (`grep -nE '(@pytest.mark|pytestmark)' ...` returned 0 hits); however whether each individual test is 'expensive' enough to warrant `slow` requires judgment. The judgment is documented inline below.",
      "rule_citation": {
        "file": "CONTRIBUTING.md",
        "lines": "37-38",
        "quote": "At least one test, with appropriate pytest markers (`slow`, `gpu`, `network`) for expensive cases."
      },
      "violation": {
        "diff_anchors": [
          "tests/test_models/test_caimira.py — 17 tests, no markers. `test_fit_reduces_loss` (line 151) runs `model.fit(..., max_epochs=50)` on a 20×30 matrix — likely <2s on CPU; under the unstated 5s threshold so `slow` is debatable.",
          "tests/test_models/test_cold_start_lookup.py — 19 tests, no markers. All pure-Python, hash-table-only; well under 1s total.",
          "tests/test_models/test_llm_judge_irt.py — 13 tests, no markers. `TestFitAlphaBeta` uses scipy.optimize.minimize with maxiter=800; `test_strong_signal_recovers_alpha_sign` constructs n=400 samples. Each ~0.1-0.5s on CPU. Marginal slow candidate.",
          "tests/test_submission/test_caimira_lite.py — 16 tests, no markers. State_dict round-trip + EB lookup; fast.",
          "tests/test_submission/test_model_contract.py — 8 tests, no markers. LOCAL_SMOKE-only; very fast (no encoder load, no .pt load)."
        ],
        "evidence": "`grep -nE '(@pytest.mark|pytestmark)' tests/test_models/test_caimira.py tests/test_models/test_cold_start_lookup.py tests/test_models/test_llm_judge_irt.py tests/test_submission/test_caimira_lite.py tests/test_submission/test_model_contract.py` returned no hits. The existing tests/test_models/test_tabpfn_predictor.py:10 carries `@pytest.mark.slow` so the convention is in use elsewhere. None of the new tests appear to require `network` (no HF download — encoders are mocked or skipped via LOCAL_SMOKE) or `gpu` (CPU-only).",
        "judgment": "The strictest CONTRIBUTING.md reading would require a `slow` marker on at least:\n  - `test_caimira.py::TestCAIMIRA::test_fit_reduces_loss` (50 epochs)\n  - `test_caimira.py::TestCAIMIRA::test_frozen_difficulty_mean_buffer_used_at_inference` (calls model.fit with 5 epochs)\n  - `test_llm_judge_irt.py::TestFitAlphaBeta::test_strong_signal_recovers_alpha_sign` (n=400, scipy.minimize)\n\nIf any of these run > a few seconds on the slowest CI Python version (3.10 on ubuntu-latest), the marker is justified to keep the CI loop fast. I cannot measure runtime from static review; the contributor or a critic with a CI run in hand should verify."
      },
      "scope_class": "upstream-eligible",
      "remediation": "Run `pytest tests/test_models/test_caimira.py tests/test_models/test_llm_judge_irt.py -v --durations=0` locally; mark any test that exceeds ~2s wall-clock as `@pytest.mark.slow` so it gets filtered out by `pytest -m 'not slow'` in the standard CI loop (CONTRIBUTING.md:104-108). The `slow-tests.yml` weekly cron (lines 8-9 of that workflow) will still exercise them. None of the new tests appear to need `network` or `gpu`."
    },
    {
      "id": "CRIT-STD-13",
      "title": ".gitignore entries are reachable and not redundant",
      "severity": "INFO",
      "category": "compliance verified",
      "confidence": 100,
      "anchored_confidence_rubric": "anchor 100 — diff anchors are exact; reachability verified against build_zip.sh's output path",
      "rule_citation": {
        "file": "CONTRIBUTING.md",
        "lines": "118-119",
        "quote": "No committed data files — data belongs in [measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db)"
      },
      "verification": {
        "anchors": [
          ".gitignore:52 — `submission/caimira_lite.meta.json` (matches `submission/train.py:435`-ish output path)",
          ".gitignore:53 — `submission/eb_tables.json` (matches train.py output)",
          ".gitignore:55 — `/submission_*.zip` (root-anchored; matches `build_zip.sh`'s default `submission_caimira.zip` at repo root per the script's `cd \"$(cd \"${SUBMISSION_DIR}/..\" && pwd)\"` pattern)",
          ".gitignore:40 — `*.pt` (pre-existing global rule; already covers `submission/caimira_lite.pt` without an explicit entry)"
        ],
        "verdict": "PASS — all three new entries are reachable and non-redundant with the pre-existing `*.pt` rule. CONTRIBUTING.md 'no committed data files' is respected: caimira_lite.pt, caimira_lite.meta.json, eb_tables.json, and the submission ZIP are all gitignored. The training data (parquets) is pulled from HF at pinned revision, not committed."
      },
      "scope_class": "fork-only",
      "remediation": "None — informational confirmation."
    },
    {
      "id": "CRIT-STD-14",
      "title": "Pre-commit ruff hooks would run on src/ and tests/ changes but not submission/",
      "severity": "INFO",
      "category": "fork-only / pre-commit scope",
      "confidence": 75,
      "anchored_confidence_rubric": "anchor 75 — the pre-commit-config.yaml is exact; ruff-pre-commit's default scope is 'all staged Python files' so it WILL run on submission/*.py if the contributor has pre-commit installed. The CI lint workflow is the one that excludes submission/ (CRIT-STD-3). This finding is informational because the pre-commit scope behaves differently from the CI workflow scope.",
      "rule_citation": {
        "file": ".pre-commit-config.yaml",
        "lines": "12-17",
        "quote": "  - repo: https://github.com/astral-sh/ruff-pre-commit\\n    rev: v0.8.0\\n    hooks:\\n      - id: ruff-check\\n        args: [--fix, --show-fixes]\\n      - id: ruff-format"
      },
      "verification": {
        "behavior": "By default, the ruff-pre-commit hooks have `types_or: [python, pyi]` and run on EVERY staged .py file regardless of directory. So a contributor who ran `pre-commit install` once and then staged `submission/model.py` would have ruff run on it locally — even though CI does not.",
        "implication": "The local pre-commit gate is broader than the CI gate. Asymmetry: a contributor who skips pre-commit (or uses `--no-verify`) can push submission/ code that fails ruff locally but passes CI. The CRIT-STD-3 remediation (extend lint.yml to include submission/) closes this asymmetry."
      },
      "scope_class": "fork-only",
      "remediation": "Already addressed by CRIT-STD-3 (extend lint.yml). Including this finding only because it explains why the contributor was able to satisfy ruff locally for submission/ code (per submission/README.md:171 'The submission code currently passes ruff manually') without CI catching anything: the pre-commit hooks have a broader scope than the CI workflow."
    }
  ],
  "residual_risks": [
    {
      "id": "RR-STD-1",
      "title": "If/when the contributor opens an upstream PR to aims-foundations/torch_measure, the upstream maintainers will reject (or block-on-changes) absent the Sphinx doc entries (CRIT-STD-1) and the LLMJudgeIRT/build_difficulty_prompt example snippets (CRIT-STD-2). These are CONTRIBUTING.md hard requirements.",
      "scope_class": "upstream-eligible",
      "trigger_condition": "PR opened against aims-foundations/torch_measure:main with these files unchanged",
      "blast_radius": "Block on PR review until CRIT-STD-1 + CRIT-STD-2 are fixed; no impact to the fork itself."
    },
    {
      "id": "RR-STD-2",
      "title": "Submission/ lint drift gap (CRIT-STD-3): a future contributor edits submission/model.py / labeling.py / etc. and introduces an SIM / B / I / UP violation that ruff would catch locally (via pre-commit, CRIT-STD-14) but CI cannot. If the contributor bypasses pre-commit (`git commit --no-verify`), the violation lands silently.",
      "scope_class": "fork-only",
      "trigger_condition": "Contributor commits with --no-verify or without pre-commit install",
      "blast_radius": "Low — style drift only; submission code still runs."
    },
    {
      "id": "RR-STD-3",
      "title": "Pytest marker absence (CRIT-STD-12) means the standard CI loop runs the new tests at every push; if any of them is >5s on the slowest matrix Python (3.10), CI gets slower over time as similar tests accumulate. The slow-tests.yml weekly cron will still cover them, so correctness is fine; speed is the residual concern.",
      "scope_class": "upstream-eligible",
      "trigger_condition": "More CAIMIRA-class tests are added without markers",
      "blast_radius": "CI feedback latency grows by ~test-runtime per push."
    },
    {
      "id": "RR-STD-4",
      "title": "submission/README.md's cross-references to ../predictive-eval-competition/ (CRIT-STD-7) mean someone cloning ONLY the fork cannot run the D-9 gate. The fork's own build_zip.sh + check_submission.sh shimmed flow still works without it, but the README mentions the gate as the prerequisite for any Codabench upload, which is misleading if the gate script isn't accessible.",
      "scope_class": "fork-only",
      "trigger_condition": "A new collaborator clones vasundras/torch_measure standalone and follows submission/README.md verbatim",
      "blast_radius": "Confusion at step 5; flow still works through step 4."
    }
  ],
  "testing_gaps": [
    {
      "id": "TG-STD-1",
      "title": "No test exercises the actual submission/model.py module-init (non-LOCAL_SMOKE) path",
      "category": "intentional-by-CONTRIBUTING.md-compliance",
      "anchor": "submission/model.py:136-163 vs tests/test_submission/test_model_contract.py:25",
      "explanation": "Every test_model_contract.py test sets LOCAL_SMOKE=1, which bypasses lines 136-163 (the SentenceTransformer load, the torch.load of caimira_lite.pt, the EBLookup.from_json, the SUBJECT_TO_IDX populate). The 'no committed data files' rule (CONTRIBUTING.md:118) blocks shipping the artifact in the repo, so a true integration test isn't possible inside the CI loop. The submission/README.md's D-9 gate run history table is the closest thing to an end-to-end test, and it's documented as fork-only.",
      "scope_class": "fork-only",
      "remediation": "If the contributor wants integration coverage, options are:\n  (a) Add a `tests/test_submission/test_integration.py` gated by `@pytest.mark.slow` AND `@pytest.mark.network` that downloads a tiny fixture artifact from HF on demand (won't run in standard CI; will run weekly via slow-tests.yml).\n  (b) Document explicitly that the D-9 gate (parent repo) IS the integration test and the CI loop is signature-coverage only.\n\nNot blocking — the project's design accepts this gap intentionally."
    },
    {
      "id": "TG-STD-2",
      "title": "No test verifies submission/build_zip.sh actually produces a kit-pre-validator-passing ZIP",
      "category": "bash-script-untested",
      "anchor": "submission/build_zip.sh:1-139 vs tests/test_submission/",
      "explanation": "build_zip.sh has 139 lines of bash with strict-mode (`set -euo pipefail`) and an explicit allowlist + post-build sanity check. No test in tests/test_submission/ invokes it. The script's correctness is verified only by manual contributor testing (per submission/README.md:64-76). A regression in zip layout (e.g., a future edit that adds a subdirectory) would not be caught by CI.",
      "scope_class": "fork-only",
      "remediation": "Optional — add `tests/test_submission/test_build_zip.py` that runs `bash submission/build_zip.sh /tmp/out.zip --force` and inspects the resulting ZIP's manifest (must be flat, must contain the 7 required files, must contain exactly 1 .pt). Gated by `@pytest.mark.slow` since it needs the trained artifact. Could also be a smoke test that creates fake artifact stubs (touch caimira_lite.pt; echo '{}' > caimira_lite.meta.json; ...) before invoking the script, to verify the script's flow without needing a real trained model."
    }
  ],
  "scope_summary": {
    "upstream-eligible": ["CRIT-STD-1", "CRIT-STD-2", "CRIT-STD-5", "CRIT-STD-6", "CRIT-STD-10", "CRIT-STD-12", "RR-STD-1", "RR-STD-3"],
    "fork-only": ["CRIT-STD-3", "CRIT-STD-4", "CRIT-STD-7", "CRIT-STD-8", "CRIT-STD-9", "CRIT-STD-11", "CRIT-STD-13", "CRIT-STD-14", "RR-STD-2", "RR-STD-4", "TG-STD-1", "TG-STD-2"]
  },
  "verdict_summary": {
    "p1_findings": ["CRIT-STD-1", "CRIT-STD-2"],
    "p2_findings": ["CRIT-STD-3", "CRIT-STD-7", "CRIT-STD-12"],
    "p3_findings": ["CRIT-STD-4", "CRIT-STD-5", "CRIT-STD-6", "CRIT-STD-9"],
    "p4_findings": ["CRIT-STD-8"],
    "info_findings": ["CRIT-STD-10", "CRIT-STD-11", "CRIT-STD-13", "CRIT-STD-14"],
    "blocking_for_upstream_pr": ["CRIT-STD-1", "CRIT-STD-2"],
    "blocking_for_fork_only_codabench_submission": []
  }
}
```

## Notes for Wave 2 critics

- **Load-bearing P1 findings are CRIT-STD-1 and CRIT-STD-2.** Both cite CONTRIBUTING.md:33-38 verbatim; both have mechanically-verifiable diff anchors (the new public exports in `__init__.py:13-17, 37-39, 46`; the absence of `.. autoclass::` directives in `docs/source/api/models.rst`; the absence of `>>>` snippets in `llm_judge_irt.py`). These are confidence 100 and would block an upstream PR. They are NOT blocking for the fork-only Codabench submission flow.

- **CRIT-STD-3 is the load-bearing fork-only finding.** submission/README.md:168-173 already self-flags the gap; the remediation is a one-line workflow edit. The submission contributor was disciplined enough to surface their own coverage gap, which speaks well for the rest of the diff.

- **CRIT-STD-12 has confidence 75 because runtime-dependent.** The strict CONTRIBUTING.md reading requires markers; without `--durations=0` data I cannot confirm which tests exceed the project's implicit `slow` threshold. A critic with CI access can downgrade this to P3 if no test exceeds 2s.

- **CRIT-STD-8 is intentionally P4.** It's noise reduction in an environment where the relevant rules aren't even active. Including it because a reviewer who notices the unfamiliar codes (`S`, `SLF`, `WPS`) will wonder if the rules are enforced — and the answer is "no, but the contributor wrote them defensively."

- **All other findings (CRIT-STD-4 through CRIT-STD-11) are P3-INFO and primarily document compliance or soft gaps.** None block the diff under normal-quality review.
