# CRITIC_4 — External claims (round 1)

```json
{
  "critic": "C4_external_claims",
  "round": 1,
  "artifact_id": "Branch feat/caimira (vasundras/torch_measure)",
  "corpus_claims": [
    {
      "reviewer": "correctness",
      "finding_id": "CRIT-CORR-2",
      "claim_excerpt": "_logit(NaN) via max(1e-7, min(1-1e-7, NaN)) yields ~16.118 (NaN coerces to confident finite logit)",
      "claim_type": "API-FACT",
      "verifier_command_or_url": "python3 -c \"import math; p=float('nan'); pc=max(1e-7,min(1-1e-7,p)); print(math.log(pc/(1-pc)))\"",
      "command_output_excerpt": "16.11809555148467 (also: min(1-1e-7, nan)=0.9999999; max(1e-7, nan)=1e-07; combined max(1e-7,min(1-1e-7,nan))=0.9999999)",
      "verdict": "VERIFIED"
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-1",
      "claim_excerpt": "NaN in CAIMIRA forward silently coerces to confident 0.9999 via _logit(NaN)≈16.118 → _sigmoid(0.6*16.118)≈0.99994",
      "claim_type": "API-FACT",
      "verifier_command_or_url": "python3 _logit(nan) reproduction + Python max/min NaN-comparison semantics check",
      "command_output_excerpt": "logit(nan)=16.118; max(1e-7, nan)=1e-07 (first arg); min(1-1e-7, nan)=0.9999999 (first arg); composition gives 1-1e-7 → log((1-1e-7)/1e-7)=16.118",
      "verdict": "VERIFIED"
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-8",
      "claim_excerpt": "_logit(NaN) and _sigmoid(NaN) produce silent finite outputs via Python max/min NaN semantics",
      "claim_type": "API-FACT",
      "verifier_command_or_url": "python3 -c with _logit and clip_for_predict on NaN input",
      "command_output_excerpt": "_logit(nan)=16.11809555148467; clip_for_predict(nan) would map nan through max/min → 0.9999 (final clip bound)",
      "verdict": "VERIFIED"
    },
    {
      "reviewer": "project_standards",
      "finding_id": "CRIT-STD-1",
      "claim_excerpt": "docs/source/api/models.rst lacks .. autoclass:: blocks for CAIMIRA, ColdStartLookupPredictor, LLMJudgeIRT, build_difficulty_prompt",
      "claim_type": "FILE-EXISTS",
      "verifier_command_or_url": "grep -E '(CAIMIRA|ColdStartLookupPredictor|LLMJudgeIRT|build_difficulty_prompt)' docs/source/api/models.rst",
      "command_output_excerpt": "NO MATCHES — confirms zero entries for the 4 new public APIs; the .rst file exists but does not enumerate them",
      "verdict": "VERIFIED"
    },
    {
      "reviewer": "learnings_research",
      "finding_id": "CRIT-LEARN-6",
      "claim_excerpt": "[PAIEC-PREDICT-002] is for predict() output violations (NaN/inf/non-float-in-[0,1]); module-init failures fall to GENERIC banner",
      "claim_type": "PROJECT-CONVENTION",
      "verifier_command_or_url": "grep -A5 'PAIEC-PREDICT-002' /Users/ankit.aggarwal/Dropbox/Stanford/CS321M/docs/solutions/design-patterns/codabench-two-tier-error-reporting-paiec-system-2026-05-19.md",
      "command_output_excerpt": "Line 50-53: 'ERROR: [PAIEC-PREDICT-002] Invalid predict() output: predict() must return a finite float in [0, 1]. ... predict() returned NaN; expected a finite float in [0, 1]'. Line 135+ Example 2: full-package submissions 740083 and 740453 failed with GENERIC banner, NOT PAIEC-coded message — confirming submission/model.py:17-27 and README.md:135-136 claim that module-init failures surface as [PAIEC-PREDICT-002] is FACTUALLY INCORRECT per canonical pattern doc.",
      "verdict": "VERIFIED"
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-2",
      "claim_excerpt": "__all__ declares 'TabPFNPredictor' at line 47 without a corresponding 'from torch_measure.models.tabpfn_predictor import TabPFNPredictor' statement",
      "claim_type": "FILE-EXISTS",
      "verifier_command_or_url": "grep -nE '(import.*TabPFNPredictor|TabPFNPredictor)' src/torch_measure/models/__init__.py + ls src/torch_measure/models/tabpfn_predictor.py",
      "command_output_excerpt": "Only match: line 47 = \"TabPFNPredictor\", (the __all__ entry). NO from-import line. tabpfn_predictor.py file DOES exist in models/ directory. AttributeError on `from torch_measure.models import *` empirically reproduced via minimal module case (types.ModuleType + exec).",
      "verdict": "VERIFIED"
    },
    {
      "reviewer": "correctness",
      "finding_id": "CRIT-CORR-5",
      "claim_excerpt": "`subj_p = self.subj.get(subj_name) or self._subj_ci.get(...)` masks legitimate 0.0 priors via or-short-circuit on falsy",
      "claim_type": "API-FACT",
      "verifier_command_or_url": "python3 -c \"d={'X':0.0}; ci={'x':0.7}; print(d.get('X') or ci.get('X'.lower()))\"",
      "command_output_excerpt": "Result: 0.7 (expected 0.0). Direct hit value 0.0 is treated as falsy → or-chain falls through to case-insensitive lookup. Also verified case-collision: {'Weak_Model': 0.0, 'weak_model': 0.4} comprehended to {'weak_model': 0.4} (last-iteration wins).",
      "verdict": "VERIFIED"
    },
    {
      "reviewer": "adversarial",
      "finding_id": "CRIT-ADV-3",
      "claim_excerpt": "_metadata_bonus at _MAX_STRATA=256 cap: overflow strata get max bonus 0.20; in-dict strata at count>=1 get bonus<=0.10",
      "claim_type": "API-FACT",
      "verifier_command_or_url": "python3 reproduction with _stratum_counts cap at 256",
      "command_output_excerpt": "len=256 after attempting 257 unique inserts (last rejected); bonus(key_0)=0.1 (in dict); bonus(key_256)=0.2 (not in dict, gets max bonus) — confirms asymmetry favoring late-arriving novel strata once cap is reached.",
      "verdict": "VERIFIED"
    },
    {
      "reviewer": "external_research_primer",
      "finding_id": "PYTORCH-2.6-FLIP",
      "claim_excerpt": "PyTorch 2.6 flipped torch.load weights_only default from False to True (backward-incompatible)",
      "claim_type": "API-FACT",
      "verifier_command_or_url": "WebFetch https://pytorch.org/blog/pytorch2-6/",
      "command_output_excerpt": "PyTorch 2.6 (released Jan 29 2025) explicitly: 'we have changed the default value for weights_only parameter of torch.load. This is a backward compatibility-breaking change'. Confirms submission/model.py:153 correctly uses weights_only=True.",
      "verdict": "VERIFIED"
    },
    {
      "reviewer": "external_research_primer",
      "finding_id": "ARXIV-2410.06524-CAIMIRA-PAPER",
      "claim_excerpt": "Paper Eq. 7 specifies frozen-bank centering; Section 4.3 reports λ_s = 1e-5 (vs diff default 1e-4)",
      "claim_type": "ACADEMIC-CLAIM",
      "verifier_command_or_url": "WebFetch https://arxiv.org/abs/2410.06524 AND https://aclanthology.org/2024.emnlp-main.1144/ AND https://arxiv.org/pdf/2410.06524v2",
      "command_output_excerpt": "All three URLs: WebFetch permission denied. Cannot independently verify the paper's exact Eq. 7 text or the Section 4.3 λ_s value in this session. The diff's code does carry latent_dim=5 default (verified) and bias asymmetry W_R+b_R vs W_d-no-bias (verified in caimira.py:131-132), which are CONSISTENT with the External Research summary's faithfulness claims (d, e); but the Eq. 7 frozen-bank claim and the λ_s = 1e-5 value remain unverifiable in this session.",
      "verdict": "UNVERIFIABLE-IN-SESSION"
    },
    {
      "reviewer": "external_research_primer",
      "finding_id": "SKLEARN-1.8-DEPRECATIONS",
      "claim_excerpt": "scikit-learn 1.8 deprecates LogisticRegression.penalty and .n_jobs (removed in 1.10)",
      "claim_type": "LIB-DEPRECATED",
      "verifier_command_or_url": "WebSearch 'scikit-learn 1.8 LogisticRegression penalty n_jobs deprecated 2025'",
      "command_output_excerpt": "WebSearch permission denied. Cannot independently verify the 1.8 deprecation in this session. Note: the diff does NOT use sklearn.LogisticRegression.penalty or .n_jobs anywhere (manual grep on submission/ + src/ confirms no LogisticRegression instantiation in the diff), so this primer claim is not load-bearing for any finding.",
      "verdict": "UNVERIFIABLE-IN-SESSION"
    }
  ],
  "promoted_from_fn_corroborations": [
    {
      "fn_candidate_id": "FN-V1-1",
      "claim_excerpt": "_TOKEN_RE = r'[a-z0-9]+' strips all non-ASCII content from SimHash signatures; CJK/Cyrillic/Arabic collapse to '<empty>'",
      "claim_type": "API-FACT",
      "verifier_command_or_url": "python3 -c \"import re; r=re.compile(r'[a-z0-9]+'); print(r.findall('你好 世界 hello 123 مرحبا'))\"",
      "command_output_excerpt": "Unicode test '你好 世界 hello 123 مرحبا' → ['hello', '123']. CJK math item '如果x+y=10且x-y=2,求x' → ['x','y','10','x','y','2','x'] (ASCII tokens only; Chinese characters stripped). Mixed 'Привет hello 世界 français' → ['hello', 'fran', 'ais'] (Cyrillic stripped, ç stripped fragmenting French). Confirms: pure-CJK or pure-Cyrillic or pure-Arabic content yields []; combined with the '['<empty>']' sentinel branch at labeling.py:85-86, all such items get the same SimHash signature → diversity score collapses to 0.",
      "verdict": "CORROBORATED"
    },
    {
      "fn_candidate_id": "FN-V1-11",
      "claim_excerpt": "CAIMIRALite.skill = nn.Parameter(torch.zeros(...)) vs upstream CAIMIRA = nn.Parameter(torch.randn(...) * 0.01)",
      "claim_type": "FILE-EXISTS",
      "verifier_command_or_url": "grep -nE 'self.skill = nn.Parameter' on both files",
      "command_output_excerpt": "submission/caimira_lite.py:221: self.skill = nn.Parameter(torch.zeros(n_subjects, latent_dim)) — zeros init. src/torch_measure/models/caimira.py:126: self.skill = nn.Parameter(torch.randn(n_subjects, latent_dim, device=self._device) * 0.01) — small-randn init. Confirms divergent untrained behavior; masked by state-dict overwrite in production but exposed in any test that instantiates without loading state.",
      "verdict": "CORROBORATED"
    },
    {
      "fn_candidate_id": "FN-V1-12",
      "claim_excerpt": "test_predict_matches_manual_caimira_equation is tautological — both sides compute (s_i - d_j)·r_j via the SAME compute_item_params() code path",
      "claim_type": "FILE-EXISTS",
      "verifier_command_or_url": "Read tests/test_models/test_caimira.py:127-149 + Read src/torch_measure/models/caimira.py:241-246 (predict() implementation)",
      "command_output_excerpt": "Test (line 141): probs = model.predict(query) → predict() at caimira.py:241-246 calls compute_item_params() then (skill[s] - difficulty[i]) * relevance[i]).sum(-1) → sigmoid. Test 'manual' (lines 142-147): relevance, difficulty = model.compute_item_params(center='frozen') then torch.sigmoid((skill[query['subject_idx']] - difficulty[query['item_idx']]) * relevance[query['item_idx']]).sum(-1)). Both sides invoke the SAME compute_item_params method on the SAME model instance in eval() mode (where center='auto' resolves to 'frozen' per line 204). The test verifies self-consistency of predict() against itself, not against an independent paper-equation reference. Test is structurally tautological as claimed.",
      "verdict": "CORROBORATED"
    }
  ],
  "summary": {
    "corpus_claims_VERIFIED": 9,
    "corpus_claims_REFUTED": 0,
    "corpus_claims_UNVERIFIABLE": 2,
    "promoted_CORROBORATED": 3,
    "promoted_NOT_CORROBORATED": 0,
    "promoted_UNVERIFIABLE": 0
  }
}
```

## Verification methodology notes

- **Independent commands run**: 11 distinct verifications (10 successful, 2 academic/external-search blocked by tool permissions).
- **Empirical reproductions verified**: `_logit(nan)=16.118`; SimHash non-ASCII strip; or-chain falsy fallthrough; case-collision overwrite; `_MAX_STRATA=256` overflow asymmetry; `__all__`-without-import AttributeError mechanism.
- **File-exists / grep verifications**: Sphinx docs missing entries (0 matches); TabPFNPredictor in `__all__` without import (line 47 only); CAIMIRALite zeros vs upstream randn (line 221 vs 126); test fixture tautology (line 141 vs 142-147 both call `compute_item_params`).
- **External docs verified**: PyTorch 2.6 weights_only flip via PyTorch blog (success).
- **External docs blocked**: arXiv 2410.06524 paper text (Eq. 7, λ_s value) — WebFetch denied on `arxiv.org/abs/`, `aclanthology.org`, and `arxiv.org/pdf/`; the External Research summary's paper-faithfulness claims (a) and (b) for centering protocol and λ_s value remain unverifiable in this session. Synthesis should downgrade confidence on any finding whose verdict hinges on paper-text comparison. scikit-learn 1.8 deprecation also unverifiable in session but is not load-bearing for any finding (diff does not use the deprecated APIs).
- **Round-2 follow-up suggestion**: if the synthesis needs paper-faithfulness verification, dispatch a separate verifier with WebFetch permissions for arxiv.org or fetch via `gh api` against the paper's GitHub mirror (if one exists).
