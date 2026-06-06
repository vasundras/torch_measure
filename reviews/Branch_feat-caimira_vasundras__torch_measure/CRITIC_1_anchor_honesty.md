```json
{
  "critic": "C1_anchor_honesty",
  "round": 1,
  "artifact_id": "Branch feat/caimira (vasundras/torch_measure)",
  "scope_mode": "pr",
  "findings_checked": 80,
  "per_finding": [
    {
      "reviewer": "correctness",
      "finding_id": "CRIT-CORR-1",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "_RESPONSE_COLS = (\"subject_id\", \"item_id\", \"benchmark_id\", \"test_condition\", \"response\")",
      "anchor_resolution": "submission/train.py:69 + :174-178 + :190-201 + :444-454"
    },
    {
      "reviewer": "correctness",
      "finding_id": "CRIT-CORR-2",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "def clip_for_predict(p: float) -> float: ...max(_PREDICT_LO, min(_PREDICT_HI, float(p)))",
      "anchor_resolution": "submission/caimira_lite.py:115-117 + submission/model.py:258"
    },
    {
      "reviewer": "correctness",
      "finding_id": "CRIT-CORR-3",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "embedding_dim=embeddings.shape[1] (L357) vs \"embed_dim\": EMBED_DIM (L413)",
      "anchor_resolution": "submission/train.py:357 + :413"
    },
    {
      "reviewer": "correctness",
      "finding_id": "CRIT-CORR-4",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "try: judge_logit = float(self.judge_fn(...)) except Exception: judge_logit = 0.0",
      "anchor_resolution": "src/torch_measure/models/llm_judge_irt.py:142-151"
    },
    {
      "reviewer": "correctness",
      "finding_id": "CRIT-CORR-5",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "subj_p = self.subj.get(subj_name) or self._subj_ci.get(subj_name.lower())",
      "anchor_resolution": "submission/caimira_lite.py:335-336 + src/torch_measure/models/cold_start_lookup.py:320-321"
    },
    {
      "reviewer": "correctness",
      "finding_id": "CRIT-CORR-6",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "def predict(self, query): s = query[...]; relevance, difficulty = self.compute_item_params()",
      "anchor_resolution": "src/torch_measure/models/caimira.py:223-246"
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-1",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "submission/train.py 458 LoC; tests/test_submission/{test_caimira_lite,test_model_contract}",
      "anchor_resolution": "submission/train.py:1-458 (absence: tests/test_submission/test_train.py)"
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-2",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "if _LOCAL_SMOKE: return 0.5; in-vocab/OOV branch lines 220-258",
      "anchor_resolution": "submission/model.py:220-258"
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-3",
      "verdict": "PARAPHRASED-OK",
      "evidence_quote_excerpt": "stateful globals _seen_signatures/_stratum_counts/_candidate_count at 44-47",
      "anchor_resolution": "submission/labeling.py:45-47 (44 is blank; 45-47 contain content); :137-171; :207-222"
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-4",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "fill_(0.3) / fill_(0.1) / fill_(-0.05) / fill_(0.2) / fill_(0.07); strict=True at L57",
      "anchor_resolution": "tests/test_submission/test_caimira_lite.py:35-63"
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-5",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "isinstance(out, float); math.isfinite(out); 0.0 <= out <= 1.0",
      "anchor_resolution": "tests/test_submission/test_model_contract.py:52-103"
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-6",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "test_fit_reduces_loss at line 151; zero markers verified via grep",
      "anchor_resolution": "tests/test_models/test_caimira.py:1-178 (5 test files scanned: 0 markers found)"
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-7",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "assert history[\"losses\"][-1] < history[\"losses\"][0]",
      "anchor_resolution": "tests/test_models/test_caimira.py:151-157"
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-8",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "TestEBLookup tests 6 levels with lookup fixture",
      "anchor_resolution": "tests/test_submission/test_caimira_lite.py:141-241"
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-9",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "139 LoC bash strict-mode; set -euo pipefail at line 26",
      "anchor_resolution": "submission/build_zip.sh:1-139"
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-10",
      "verdict": "PARAPHRASED-OK",
      "evidence_quote_excerpt": "labeled list 5 entries all mmlupro; assert cybench unchanged",
      "anchor_resolution": "tests/test_models/test_cold_start_lookup.py:210-225 (finding says '4 distinct labels' but file has 5 entries; mean still 0.2)"
    },
    {
      "reviewer": "testing",
      "finding_id": "CRIT-TEST-11",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "tutorial notebook 622 LoC; no CI re-execution",
      "anchor_resolution": "tutorials/predictive_evaluation_challenge.ipynb (622 lines verified)"
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-1",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "_DEFAULT_PROVIDER_PREFIXES / _logit / _sigmoid / _clip / parse/resolve duplicated",
      "anchor_resolution": "submission/caimira_lite.py:79-83,97-100,103-108,111-112,120-139,142-172 + src/torch_measure/models/cold_start_lookup.py:48-52,63-66,69-74,77-78,81-99,102-154"
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-2",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "_logit/_sigmoid defined at model.py:97-112 with per-function import math",
      "anchor_resolution": "submission/model.py:97-117 (L99,107 import math; L72-78 caimira_lite imports)"
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-3",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "class ColdStartLookupPredictor: (no base); predict(record,labeled)->float",
      "anchor_resolution": "src/torch_measure/models/cold_start_lookup.py:157,338-380 + _predictor.py:14-25 + _base.py:17-28 + tabpfn_predictor.py:23"
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-4",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "Negative-result disclosure; NOT-FOR-PRODUCTION; in __all__ at __init__:38-39",
      "anchor_resolution": "src/torch_measure/models/llm_judge_irt.py:1-37 + src/torch_measure/models/__init__.py:17,38-39"
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-5",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "_EB_SBC_MIN_N=3, _EB_SUBJ_MIN_N=10, _EB_SB_ALPHA=5.0, _TABLE_CLIP_LO/HI=0.05/0.95",
      "anchor_resolution": "submission/train.py:71-78"
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-6",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "_BLEND_LAMBDA: float = 0.6 at L83; def _blend_logits(..., lambda_=_BLEND_LAMBDA) at L115-117; call site L250",
      "anchor_resolution": "submission/model.py:80-83,115-117,250"
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-7",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "_item_cache: dict[str, torch.Tensor] = {} at L165; _encode_item L168-183",
      "anchor_resolution": "submission/model.py:165-183"
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-8",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "center: str = \"auto\" at L164; ValueError at L211",
      "anchor_resolution": "src/torch_measure/models/caimira.py:161-213"
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-9",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "submission/README.md status: scaffolding callout",
      "anchor_resolution": "submission/README.md:1-19 (status block actually at L10-16; close paraphrase)"
    },
    {
      "reviewer": "maintainability",
      "finding_id": "CRIT-MAINT-10",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "minimal_lookup fixture at L27-37; predictor fixture at L33-83",
      "anchor_resolution": "tests/test_models/test_llm_judge_irt.py:27-37 + tests/test_models/test_cold_start_lookup.py:33-83"
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-1",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "from CAIMIRA/ColdStartLookupPredictor/LLMJudgeIRT/build_difficulty_prompt imported & in __all__",
      "anchor_resolution": "src/torch_measure/models/__init__.py:13,14,17,37-39,46 + CONTRIBUTING.md:33-38 (Sphinx absence verified)"
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-2",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "grep '>>>' returned 0; grep 'Examples' returned 0",
      "anchor_resolution": "src/torch_measure/models/llm_judge_irt.py:54-74 (build_difficulty_prompt) + :77-107 (LLMJudgeIRT class); absence verified"
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-3",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "ruff check src/ tests/ at L30; ruff format --check src/ tests/ at L33",
      "anchor_resolution": ".github/workflows/lint.yml:30,33 + submission/README.md:168-173"
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-4",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "pytest tests/ -v --cov=torch_measure ... -m \"not slow and not gpu\"",
      "anchor_resolution": ".github/workflows/test.yml:37 + tests/test_submission/test_model_contract.py:25"
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-5",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "Built on PyTorch, with GPU-accelerated IRT, factor models, amortized inference",
      "anchor_resolution": "README.md:10"
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-6",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "Commit messages are short, single-line ... We squash-merge PRs by default.",
      "anchor_resolution": "CONTRIBUTING.md:117,123 (4 commits on feat/caimira have multi-line bodies)"
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-7",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "python ../predictive-eval-competition/scripts/run_d9_gate.py submission_caimira.zip",
      "anchor_resolution": "submission/README.md:86,155-156"
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-8",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "# noqa: S101 at :223; # noqa: SLF001 at :254-255; # noqa: WPS433 at train.py:390",
      "anchor_resolution": "submission/model.py:223,254-255 + submission/train.py:390 + pyproject.toml:99-101"
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-9",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "[tool.coverage.run] source = [\"src/torch_measure\"]",
      "anchor_resolution": "pyproject.toml:106-108"
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-10",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "License header on line 1 of all 13 new files",
      "anchor_resolution": "all 13 new files (verified line 1 for each carries # Copyright (c) 2026 AIMS Foundations. MIT License.)"
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-11",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "- id: check-added-large-files\\n        args: [\"--maxkb=500\"]",
      "anchor_resolution": ".pre-commit-config.yaml:9-10"
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-12",
      "verdict": "PARAPHRASED-OK",
      "evidence_quote_excerpt": "0 marker hits across 5 new test files; counts (17/19/13/16/8) cited",
      "anchor_resolution": "tests/test_models/test_caimira.py:1-178 + 4 siblings (claimed counts 17/19/13/16/8 vs actual 17/25/12/24/10; absence-claim still HONEST)"
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-13",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "submission/caimira_lite.meta.json (L52) / eb_tables.json (L53) / /submission_*.zip (L55) / *.pt (L40)",
      "anchor_resolution": ".gitignore:40,52,53,55"
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-14",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "ruff-pre-commit hooks scope all staged .py files",
      "anchor_resolution": ".pre-commit-config.yaml:12-17 + submission/README.md:171"
    },
    {
      "reviewer": "learnings_research",
      "finding_id": "CRIT-LEARN-1",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "torch.save(model_cpu.state_dict(), pt_path); torch.load(...weights_only=True); CAIMIRA_ARCH_VERSION=1",
      "anchor_resolution": "submission/train.py:384,387-404 + submission/model.py:153 + submission/caimira_lite.py:74"
    },
    {
      "reviewer": "learnings_research",
      "finding_id": "CRIT-LEARN-2",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "model.py:17-27 D-3 docstring; labeling.py:220-222 except Exception: return 0.0",
      "anchor_resolution": "submission/model.py:17-27 + submission/labeling.py:182-204,220-222 + submission/README.md:129-140"
    },
    {
      "reviewer": "learnings_research",
      "finding_id": "CRIT-LEARN-3",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "No 'from model import' in labeling.py; only return float at L66,79,134,156,176,177,222",
      "anchor_resolution": "submission/labeling.py (single except at L220; _clamp_score at L174-177)"
    },
    {
      "reviewer": "learnings_research",
      "finding_id": "CRIT-LEARN-4",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "labeling.py port note at L24-29 + parent file at predictive-eval-competition/submission/labeling.py",
      "anchor_resolution": "submission/labeling.py:1-30 + parent labeling.py confirmed to exist"
    },
    {
      "reviewer": "learnings_research",
      "finding_id": "CRIT-LEARN-5",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "submission/README.md \"D-9 gate run history\" table; smoke 2026-05-22 OVERALL FAIL",
      "anchor_resolution": "submission/README.md:142-159 (table + addendum)"
    },
    {
      "reviewer": "learnings_research",
      "finding_id": "CRIT-LEARN-6",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "[PAIEC-PREDICT-002] error before any predictions run",
      "anchor_resolution": "submission/model.py:17-27 + submission/README.md:135-136"
    },
    {
      "reviewer": "learnings_research",
      "finding_id": "CRIT-LEARN-7",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "build_zip.sh explicit REQUIRED_FILES allowlist at L55-63; set -euo pipefail at L26",
      "anchor_resolution": "submission/build_zip.sh:54-127 (specifically L55-63, L65-81, L83-86, L88-99, L110-116, L118-127, L26)"
    },
    {
      "reviewer": "learnings_research",
      "finding_id": "CRIT-LEARN-8",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "build_zip.sh has G + partial H; no A/D/E/F gates",
      "anchor_resolution": "submission/build_zip.sh (all 139 lines scanned; no AST/subprocess predict() invocation/labeling smoke present)"
    },
    {
      "reviewer": "learnings_research",
      "finding_id": "CRIT-LEARN-9",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "K=5 ``labeled`` reveals at model.py:36; Optional list of K=5-per-category at :211",
      "anchor_resolution": "submission/model.py:36,211"
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-1",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "p_lookup = self.lookup._lookup_p(subj_name, benchmark, condition)",
      "anchor_resolution": "src/torch_measure/models/llm_judge_irt.py:139 + cold_start_lookup.py:288 + submission/caimira_lite.py:316"
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-2",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "center: str = \"auto\" at L164",
      "anchor_resolution": "src/torch_measure/models/caimira.py:161-213"
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-3",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "_logit at L97 with 'import math' inside; _sigmoid at L105 same",
      "anchor_resolution": "submission/model.py:97-112 + submission/caimira_lite.py:64"
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-4",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "def predict(input: dict, labeled: list[dict] | None = None)",
      "anchor_resolution": "submission/model.py:186-219 + tests/test_submission/test_model_contract.py:53"
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-5",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "_seen_signatures: list[int] = []; _stratum_counts annotated; _candidate_count = 0 unannotated",
      "anchor_resolution": "submission/labeling.py:45-47 + L205 (global statement)"
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-6",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "def fit(self, data, embeddings: torch.Tensor, ...)",
      "anchor_resolution": "src/torch_measure/models/caimira.py:248-260 + _base.py:32 + amortized.py:135"
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-7",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "constructor takes 5 dict[str, float] params; key shape via docstring",
      "anchor_resolution": "src/torch_measure/models/cold_start_lookup.py:212-243 + submission/train.py:260 + submission/caimira_lite.py:316"
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-8",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "from typing import Any, Callable",
      "anchor_resolution": "src/torch_measure/models/llm_judge_irt.py:43 + cold_start_lookup.py:42 (uses collections.abc.Iterable)"
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-9",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "_logit/_sigmoid/_clip duplicated across 4 files",
      "anchor_resolution": "submission/caimira_lite.py:97-117 + submission/train.py:226-239 + submission/model.py:97-112 + cold_start_lookup.py:63-78"
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-10",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "from caimira_lite import CAIMIRALite  # noqa: E402,WPS433 inside main()",
      "anchor_resolution": "submission/train.py:58,390"
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-11",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "def predict(self, query: dict[str, torch.Tensor]) -> torch.Tensor at 4 sites",
      "anchor_resolution": "src/torch_measure/models/caimira.py:223 + rasch.py:37 + twopl.py:42 + amortized.py:121"
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-12",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "self.calibrate(labeled) at predict body",
      "anchor_resolution": "src/torch_measure/models/cold_start_lookup.py:374,388 + submission/caimira_lite.py:401"
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-13",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "try/except + raise AssertionError pattern at 4 places",
      "anchor_resolution": "tests/test_models/test_caimira.py:25-35,52-58,60-66,68-74"
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-14",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "test_caimira.py L1-8 lacks from __future__ import annotations",
      "anchor_resolution": "tests/test_models/test_caimira.py:1-8 (absence verified) + tests/test_models/test_cold_start_lookup.py:14 (presence verified)"
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-15",
      "verdict": "PARAPHRASED-OK",
      "evidence_quote_excerpt": "assert parse_subject_name(None or \"\") == \"\" at L114",
      "anchor_resolution": "tests/test_submission/test_caimira_lite.py:114 (verified); sibling test_cold_start_lookup.py:167 (cited as L168; off by 1)"
    },
    {
      "reviewer": "kieran_python",
      "finding_id": "CRIT-PY-16",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "argparse.ArgumentParser(description=__doc__) at L84; docstring with ReST :: blocks L1-33",
      "anchor_resolution": "submission/train.py:1-33,84 (title says model.py but anchor file/lines correctly point to train.py)"
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-1",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "_logit clip; clip_for_predict at 115-117; _blend_logits at 115-117",
      "anchor_resolution": "submission/caimira_lite.py:97-101,115-117 + submission/model.py:115-117"
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-2",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "__all__ contains \"TabPFNPredictor\" at L47 but no matching import",
      "anchor_resolution": "src/torch_measure/models/__init__.py:5-26,47 (absence-of-import verified)"
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-3",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "_metadata_bonus at 131-134; _update_stratum at 137-140; _MAX_STRATA=256 at 41",
      "anchor_resolution": "submission/labeling.py:131-141,41"
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-4",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "awk 'NR > 3 && /\\// ...' at L111; allowlist diff at L119",
      "anchor_resolution": "submission/build_zip.sh:111,119"
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-5",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "p = self.lookup_p(...); by_bench.setdefault(...)append((_logit(p), float(ex['label'])))",
      "anchor_resolution": "submission/model.py:252-256 + submission/caimira_lite.py:346-387,368-369,377-382"
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-6",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "subj_p = self.subj.get(...) or self._subj_ci.get(...)",
      "anchor_resolution": "submission/caimira_lite.py:335-336 + src/torch_measure/models/cold_start_lookup.py:320-321"
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-7",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "self._subj_ci = {k.lower(): v for k, v in self.subj.items()} (and 3 siblings)",
      "anchor_resolution": "submission/caimira_lite.py:288-291"
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-8",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "_logit at L97-101; _sigmoid at L103-108; clip_for_predict at L115-117",
      "anchor_resolution": "submission/caimira_lite.py:97-101,103-108,115-117"
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-9",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "labeling try/except at 206-222; model.py predict body 185-258 no try/except",
      "anchor_resolution": "submission/labeling.py:206-222 + submission/model.py:185-258"
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-10",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "new_key = len(labeled); if new_key == self._platt_fit_key: return",
      "anchor_resolution": "submission/caimira_lite.py:353-356 + src/torch_measure/models/cold_start_lookup.py:412-415"
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-11",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "by_bench.setdefault(bench, []).append((_logit(p), float(ex[\"label\"])))",
      "anchor_resolution": "submission/caimira_lite.py:369 + src/torch_measure/models/cold_start_lookup.py:431"
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-12",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "TestModelAcquisitionContract class with 4 tests; resets at L164-166",
      "anchor_resolution": "tests/test_submission/test_model_contract.py:107-176"
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-13",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "caimira_logit at L226-243; model.py:235-237 subject_idx lookup",
      "anchor_resolution": "submission/caimira_lite.py:226-243 + submission/model.py:235-250"
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-14",
      "verdict": "HONEST",
      "evidence_quote_excerpt": "fit_nll = float(np.mean(y_arr * np.log(p) + (1 - y_arr) * np.log(1 - p)))",
      "anchor_resolution": "src/torch_measure/models/llm_judge_irt.py:215-221"
    }
  ],
  "summary": {
    "HONEST": 76,
    "PARAPHRASED-OK": 4,
    "NOT-FOUND": 0,
    "PRIOR-VERSION-LEAK": 0
  }
}
```

## Summary of concerning anchor defects

Mechanical anchor honesty is uniformly high across all 7 Wave 1 reviewers. 80 findings inspected (6 CORR + 11 TEST + 10 MAINT + 14 STD + 9 LEARN + 16 PY + 14 ADV); multi-file anchors verified independently per charter rule. No `NOT-FOUND` anchors; no fabricated quotes; no mis-cited line ranges that fail to resolve.

The four `PARAPHRASED-OK` cases are minor:

1. `CRIT-TEST-3` cites `labeling.py:44-47` for the three stateful globals; line 44 is blank, the content sits at `:45-47`. The numeric range is one-off but the content claim is faithful.
2. `CRIT-TEST-10` describes the labeled fixture as "5 entries all for benchmark `mmlupro`" in the rationale prose and as "4 distinct labels with mean 0.2" in the title; the file has 5 entries with 2 distinct labels (0,1) and mean 0.2. Title is paraphrased, evidence body is exact.
3. `CRIT-STD-12` cites per-file test counts (`17/19/13/16/8`) where the actual counts are `17/25/12/24/10`. The numbers are slightly off but the absence-of-markers core claim is mechanically verified (grep `@pytest.mark` returns zero across all 5 files).
4. `CRIT-PY-15` cites `test_cold_start_lookup.py:168` for the `parse_subject_name(None)` baseline; actual line is `:167` (one-off).

`CRIT-PY-16`'s title says "submission/model.py top-level docstring" but the `where.file` and `where.lines` anchors correctly point at `submission/train.py:1-33,84`; the body discusses argparse consuming `__doc__` which is a train.py concern. Title is misleading but the anchor metadata is HONEST. Flagging here so C2/synthesis can decide whether the title-vs-anchor mismatch is a P3 reviewer-side bug.

No anchor-defects warrant FP-promotion; every cited file:line resolves to text matching either verbatim or with light paraphrase. Confidence interpretation of findings (correct? severity-appropriate?) is C2/C3's lane.
