# Phase 3 Freeze — Paper B (FTLE / λ₁) — CORRECTED

**Date:** 2026-02-19
**Status:** COMPLETE (protocol-corrected rerun)
**Owner:** Ghost

## Frozen Artifacts

| Artifact | SHA256 |
|---|---|
| `results/phase3/bridge_results.json` | `d563c0ddce9f2cc8822290a99f29ad5d16a0bcfc6292e03408ce174e7648ebcf` |
| `results/phase3/DEVIATION_TABLE.md` | see commit |

## Input Artifacts (verified)

| Artifact | SHA256 |
|---|---|
| `results/phase2/ftle_bundle.csv` | `753a62b3c9c814013dd0d6d1606d423653e726dd19c0a95477fecd2cade6f1a8` |
| `results/internal/paper-a/phase3_baseline/analysis_bundle.csv` | `8219ff8cce8ba45cb6c775d499ac3b927aae7bff03f0dc7edd4d554068623810` |

## Protocol Corrections (from initial run)

1. **Test #4 variable:** Changed from `lambda1_std` (tangent-seed std) to `layerwise_variance_mean` (layerwise profile variance) per PREREG_v2 definition.
2. **first_collapse_turn handling:** Primary set now uses censored value=40 for non-collapsing trajectories (n=720), per PREREG_v2. Collapse-only subset available as sensitivity check.

See `DEVIATION_TABLE.md` for old vs corrected results comparison.

## Results Summary

**Confirmatory status based on corrected prereg-aligned run.**

### Primary (n=720)

| # | λ₁ summary | Paper A metric | n | ρ | p_holm | 95% CI | Pass |
|---|---|---|---|---|---|---|---|
| 1 | lambda1_mean | collapse_rate | 720 | +0.246 | 4.26e-11 | [+0.17, +0.32] | ❌ |
| 2 | lambda1_mean | first_collapse_turn | 720 | −0.251 | 2.29e-11 | [−0.32, −0.18] | ❌ |
| 3 | lambda1_mean | collapse_incidence | 720 | +0.036 | 0.337 | [−0.03, +0.10] | ❌ |
| 4 | lambda1_var | collapse_rate | 720 | **+0.511** | **2.41e-48** | [+0.45, +0.57] | ✅ |
| 5 | lambda1_slope | collapse_rate | 720 | **−0.536** | **5.24e-54** | [−0.59, −0.48] | ✅ |
| 6 | lambda1_slope | first_collapse_turn | 720 | **+0.507** | **1.22e-47** | [+0.45, +0.56] | ✅ |

Passing: 3/6. Prereg success criterion: **MET**.

### Sensitivity (n=167 rater-agreed)

| # | λ₁ summary | Paper A metric | n | ρ | p_holm | Pass |
|---|---|---|---|---|---|---|
| 1 | lambda1_mean | collapse_rate | 167 | +0.255 | 1.75e-03 | ❌ |
| 2 | lambda1_mean | first_collapse_turn | 167 | −0.265 | 1.64e-03 | ❌ |
| 3 | lambda1_mean | collapse_incidence | 167 | −0.005 | 0.947 | ❌ |
| 4 | lambda1_var | collapse_rate | 167 | **+0.545** | **1.39e-13** | ✅ |
| 5 | lambda1_slope | collapse_rate | 167 | **−0.557** | **3.15e-14** | ✅ |
| 6 | lambda1_slope | first_collapse_turn | 167 | **+0.516** | **3.86e-12** | ✅ |

Passing: 3/6. Consistent with primary.

## Interpretation Notes

- Rerun r and ICC from Phase 1 assess deterministic reproducibility; tangent-seed CV is the primary robustness metric.
- Paper A collapse labels have unconfirmed inter-rater reliability (κ = 0.566, threshold 0.80 not met). This caveat applies to all bridge correlations.
- Bootstrap: n=10,000 resamples, bias-corrected percentile CIs, RNG seed=42.
