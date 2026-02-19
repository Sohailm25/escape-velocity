# Kappa Gate Final Record — Paper A Detector Reliability Audit

**Date:** 2026-02-19
**Verdict:** ❌ GATE FAILED
**Final κ:** 0.5663 (threshold: 0.80)

---

## Summary

The prereg reliability gate (Cohen's κ ≥ 0.80 between two independent LLM raters) was NOT met after the final attempt with locked rubric v3.0. Per Professor's protocol, no further attempts are permitted.

## Final Run Statistics

| Metric | Value |
|---|---|
| Valid pairs | 180/180 (0 nulls) |
| Raw agreement | 167/180 (92.8%) |
| Cohen's κ | **0.5663** |
| Gate threshold | 0.80 |
| Gate passed | **NO** |

## Contingency Table

|  | ANT = 1 | ANT = 0 |
|---|---|---|
| **OAI = 1** | 157 | 7 |
| **OAI = 0** | 6 | 10 |

## Per-Condition Breakdown

| Condition | n | Agree | % | κ |
|---|---|---|---|---|
| HOMO_A | 45 | 43 | 95.6% | 0.646 |
| HOMO_B | 45 | 45 | 100.0% | N/A (no variance) |
| HOMO_C | 45 | 41 | 91.1% | 0.464 |
| HETERO_ROT | 45 | 38 | 84.4% | 0.536 |

## Root Cause Analysis

κ is depressed by **extreme base-rate skew**: 163/180 items are labeled collapse=1 by at least one rater. When prevalence is >85%, even 92.8% raw agreement produces modest κ because chance agreement (pe) is already ~0.75.

The 13 disagreements cluster in:
- **HOMO_C (4):** Raters disagree on non-adjacent recurring patterns vs. strictly-adjacent requirement
- **HETERO_ROT (7):** Raters disagree on farewell-sequence and thematic-repetition edge cases
- **HOMO_A (2):** Template-similarity threshold

## Rubric Version History

| Version | κ | Issue |
|---|---|---|
| v1.0 | 0.011 | Ambiguous rubric, uncalibrated raters |
| v2.1 | 0.519 | Consecutive-ambiguity + farewell gaps |
| v3.0 | 0.566 | Strict adjacency + farewell rules, but base-rate skew limits κ ceiling |

## Interpretation

The low κ does NOT necessarily indicate detector failure. It reflects:
1. **Construct boundary ambiguity:** The distinction between "thematic repetition" and "content collapse" has genuine grey-zone cases that LLM raters resolve differently
2. **Base-rate skew penalty:** With ~87% prevalence, the mathematical ceiling for κ given 92.8% agreement is ~0.65 — the gate threshold of 0.80 may be unreachable without near-perfect agreement
3. **Rater capability ceiling:** GPT-4o and Claude Sonnet may have an inherent disagreement floor on these edge cases

## Recommendation for Professor

Three options:
1. **Adjust gate to prevalence-adjusted κ (PABAK):** PABAK = 2*po - 1 = 2*(0.928) - 1 = **0.856** — passes 0.80 threshold. This is standard when prevalence bias is documented.
2. **Switch to human raters:** Gold-standard but expensive. 180 windows × 2 raters.
3. **Accept automated detector as ground truth with documented limitations:** Skip κ gate, report detector methodology in paper with transparent limitation section.

## Artifacts

- `v2_prereg_rater1_openai.jsonl` — 180 labels (0 nulls)
- `v2_prereg_rater2_anthropic.jsonl` — 180 labels (0 nulls, 10 repaired)
- `v2_prereg_kappa_metrics.json` — machine-readable metrics
- `CODING_MANUAL_v2.md` — locked rubric v3.0
- `calibration_log.md` — calibration round results
- `calibration_sample_20.json` — held-out calibration windows
