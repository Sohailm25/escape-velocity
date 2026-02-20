# Deviation-and-Correction Table — Phase 3 Bridge Analysis

**Date:** 2026-02-19
**Reason:** Professor review identified two protocol deviations from PREREG_v2.md

## Deviation 1: Test #4 variable mapping

| | Old (incorrect) | Corrected (prereg-aligned) |
|---|---|---|
| **Variable** | `lambda1_std` (tangent-seed std) | `layerwise_variance_mean` (layerwise profile variance) |
| **PREREG definition** | "lambda1_var = variance of layerwise λ₁ profile" | ← same |
| **ρ** | −0.059 | **+0.511** |
| **p_holm** | 0.221 | **2.41e-48** |
| **CI** | [−0.132, +0.012] | **[+0.450, +0.568]** |
| **Pass** | ❌ | **✅** |

**Impact:** Substantial. The correct variable (layerwise profile variance) shows a strong positive association with collapse rate, consistent with the hypothesis that models with more variable depth dynamics collapse more.

## Deviation 2: first_collapse_turn missingness handling

| | Old (incorrect) | Corrected (prereg-aligned) |
|---|---|---|
| **Primary handling** | Exclude non-collapsing runs (n=540) | Censor at turn=40 (n=720) |
| **PREREG definition** | "Primary: censored value 40 for non-collapsing" | ← same |

**Impact on Test #2 (lambda1_mean × first_collapse_turn):**

| | Old | Corrected |
|---|---|---|
| n | 540 | 720 |
| ρ | −0.382 | −0.251 |
| p_holm | 1.34e-19 | 2.29e-11 |
| Pass | ❌ (|ρ| < 0.40) | ❌ (|ρ| < 0.40) |

**Impact on Test #6 (lambda1_slope × first_collapse_turn):**

| | Old | Corrected |
|---|---|---|
| n | 540 | 720 |
| ρ | +0.539 | +0.507 |
| p_holm | 2.38e-41 | 1.22e-47 |
| Pass | ✅ | ✅ |

**Impact:** Test #2 weakens (still fails). Test #6 attenuates slightly but still passes comfortably.

## Summary of pass/fail changes

| Test | Old pass | Corrected pass | Change |
|---|---|---|---|
| 1 (mean × rate) | ❌ | ❌ | — |
| 2 (mean × first_turn) | ❌ | ❌ | — |
| 3 (mean × incidence) | ❌ | ❌ | — |
| 4 (var × rate) | ❌ | **✅** | PROMOTED |
| 5 (slope × rate) | ✅ | ✅ | — |
| 6 (slope × first_turn) | ✅ | ✅ | — |

**Net effect:** 2/6 → 3/6 passing. Prereg success criterion remains MET.
