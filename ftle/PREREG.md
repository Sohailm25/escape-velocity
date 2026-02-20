# PREREG — Paper B (FTLE / λ₁) — Version 2.0

**Changelog:**
- v1.0 (2026-02-15): Initial prereg — estimator, stability, acceptance, success criteria
- v2.0 (2026-02-19): Added bridge-analysis protocol, reliability framework, null/attrition policy

---

## Hypothesis (pre-commit)
A numerically stable estimate of top-1 finite-time Lyapunov exponent (λ₁), computed for transformer depth dynamics, differs across model families and provides predictive association (not causal proof) with Paper A entrenchment/escape behavior.

## Linking hypothesis and time-axis scope (locked)
Paper A outcomes are across-turn conversational dynamics; Paper B λ₁ is within-forward-pass depth dynamics.
Therefore, linkage claims are restricted to:
- predictive association between depth-dynamics summaries and across-turn outcomes,
- not direct mechanistic identity between the two time axes.

Any causal or equivalence claims require additional bridge analyses and are out of scope for this prereg.

## Estimator definition (locked)
Mathematical object:
- Let `F_l` be the residual-stream update map at layer `l` for a fixed token-context state.
- Local Jacobian at layer `l`: `J_l = ∂F_l/∂h_l`.
- Finite-depth tangent product over `L` layers: `P_L = J_L ... J_2 J_1`.
- Top-1 FTLE estimate: `λ₁ = (1/L) * log(||P_L v_0|| / ||v_0||)` using JVP-based tangent propagation with periodic renormalization (QR-equivalent for top mode).

Implementation:
- Depth-as-time mapping with JVP-based tangent propagation.
- QR/renormalization tracking for λ₁ across layers.
- Report both scalar λ₁ and layerwise λ₁ profile.

## Primary metrics
### A) Numerical stability
1. **Tangent-seed robustness:** coefficient of variation of λ₁ across tangent initializations.
2. **Renormalization sensitivity:** absolute λ₁ shift across cadence settings.
3. **Run repeatability:** agreement across reruns with same seed/config.
4. **Small-model sanity check:** agreement against finite-difference approximation.

### B) External validity vs Paper A
- Correlation between λ₁ (or λ₁ profile summary) and:
  - collapse entrenchment depth,
  - escape probability,
  - time-to-recollapse.

## Acceptance thresholds (locked)
A model-family estimate is accepted only if all hold:
1. Tangent-seed CV <= **0.15**
2. Renorm-cadence sensitivity |Δλ₁| <= **0.05** between cadence settings {1,2,4}
3. Rerun repeatability >= **0.85** Pearson r
4. Small-model finite-difference agreement >= **0.80** correlation

## Analysis plan
- Compute λ₁ on same seed/condition strata used by Paper A where possible.
- Use rank + linear correlations with bootstrap CIs.
- Treat causal claims as out of scope; focus on predictive/mechanistic association.

## Success criteria (locked)
1. Stability acceptance achieved in >= **2/3** model families.
2. At least one preregistered λ₁ summary shows |ρ| >= **0.40** with a primary Paper A metric, 95% CI excluding 0.

## Abort / downscope criteria (locked)
- If stability thresholds fail in all model families, downscope to reduced-layer/smaller model diagnostic and report estimator failure mode; do not claim mechanistic validity at production scale.

## Dependency gate
No primary Paper B claim is made unless Paper A **regime-establishment criteria** are satisfied — specifically:
1. Complete baseline execution (720/720 unique successful tuples across all 4 conditions)
2. Protocol integrity (all runs 40 turns, no early stopping)
3. Collapse incidence ≥ 60% in at least one condition

These criteria were met (Paper A Path B outcome). The Paper A detector reliability gate (κ ≥ 0.80) was NOT met; this does not block Paper B claims but requires that all bridge-analysis results carry the explicit caveat that Paper A collapse labels have unconfirmed inter-rater reliability (see Bridge-Analysis Protocol, Paper A Label Caveat section).

---

## Bridge-Analysis Protocol (locked) — v2.0 addition

### Paper A Variables Consumed (frozen)

Source: `results/internal/paper-a/phase3_baseline/analysis_bundle.csv`
Freeze SHA256 (analysis_bundle.csv): `8219ff8cce8ba45cb6c775d499ac3b927aae7bff03f0dc7edd4d554068623810`
Freeze SHA256 (coverage_matrix.json): `60fc4f9a25bd2518b5f554a220ac3ef8c98df88f35f6c5eb8fcb7bf9a59e69b1`

| Variable | Type | Description |
|---|---|---|
| `collapse_rate` | float [0,1] | Fraction of turns classified as collapsed |
| `first_collapse_turn` | int/null | Turn index of first collapse detection |
| `collapse_incidence` | binary | Whether any collapse was detected (derived: collapse_rate > 0) |

**No additional Paper A variables may be added post-hoc.** If analysis suggests other variables would be informative, they must be declared as exploratory and not used for primary claims.

### Paper B λ₁ Summaries (pre-registered)

| Summary | Computation | Description |
|---|---|---|
| `lambda1_mean` | Mean λ₁ across 10 tangent seeds | Primary FTLE estimate per trajectory |
| `lambda1_var` | Variance of layerwise λ profile | Depth-heterogeneity of expansion rate |
| `lambda1_slope` | OLS slope of λ profile vs layer index | Trend in expansion rate across depth |

### Pre-Registered Pairings (6 total)

| # | λ₁ Summary | Paper A Variable | Test | Hypothesis Direction |
|---|---|---|---|---|
| 1 | `lambda1_mean` | `collapse_rate` | Spearman ρ | Exploratory (no directional prior) |
| 2 | `lambda1_mean` | `first_collapse_turn` | Spearman ρ | Exploratory |
| 3 | `lambda1_mean` | `collapse_incidence` | Spearman ρ | Exploratory |
| 4 | `lambda1_var` | `collapse_rate` | Spearman ρ | Exploratory |
| 5 | `lambda1_slope` | `collapse_rate` | Spearman ρ | Exploratory |
| 6 | `lambda1_slope` | `first_collapse_turn` | Spearman ρ | Exploratory |

**No post-hoc variable selection:** All 6 pairings are tested regardless of initial results. Multiple comparison adjustment: Bonferroni-Holm on the 6 p-values.

### Statistical Methods

- **Primary:** Spearman rank correlation (robust to non-normality)
- **Sensitivity:** OLS linear regression with condition as covariate
- **Confidence intervals:** Bootstrap (n=10,000 resamples), percentile method
- **Multiple comparisons:** Bonferroni-Holm across 6 primary tests
- **Effect size threshold:** |ρ| ≥ 0.40 with Bonferroni-Holm-adjusted 95% CI excluding 0

### Handling of Missingness

- If λ₁ computation fails (NaN/Inf) for a trajectory, exclude from bridge analysis
- **Max attrition:** 10% per condition (18/180). If exceeded → ABORT, diagnose, Professor review
- If one entire condition fails stability (Phase 2), exclude that condition from bridge analysis and report as estimator failure for that model family
- Missing `first_collapse_turn` (non-collapsed runs): use the full 40-turn value as a censored observation; sensitivity check: exclude non-collapsed runs entirely

### Paper A Label Caveat

Paper A collapse labels were generated by a detector whose reliability gate was not met (κ = 0.566). Bridge analysis results must include this caveat in all reporting. Sensitivity check: repeat bridge analysis using only trajectories where both LLM raters agreed on collapse status (167/180 audit pairs).

---

## Reliability Framework (locked) — v2.0 addition

### Lesson from Paper A

Paper A's reliability gate used Cohen's κ on a binary classification with ~87% prevalence. Extreme base-rate skew mathematically depresses κ even at high raw agreement (92.8% agreement → κ = 0.566). Paper B avoids this failure mode by using a continuous reliability metric (ICC) on a continuous outcome (λ₁).

### Primary Reliability Criterion

**Metric:** ICC(2,1) — two-way random, single-measures, absolute agreement
**Target:** ICC(2,1) ≥ 0.75 (good reliability per Koo & Li, 2016)
**Computed on:** λ₁ rerun agreement — same (model, seed, tangent_seed, cadence) run 3 times
**Sample:** All Phase 1 pilot runs (5 seeds × 10 tangent seeds × 3 cadences = 150 configurations, each run 3 times)

### Sensitivity Metrics (diagnostic only, not gate-switching)

| Metric | Purpose | Threshold (advisory) |
|---|---|---|
| CV across tangent seeds | Tangent initialization robustness | ≤ 0.15 (from acceptance thresholds) |
| Bland-Altman limits of agreement | Rerun systematic bias detection | Bias ≤ 0.01, 95% LoA width ≤ 0.10 |
| Renorm cadence |Δλ₁| | Numerical stability | ≤ 0.05 (from acceptance thresholds) |

**These sensitivity metrics cannot replace the primary ICC gate post-hoc.** They are reported for diagnostic interpretation only.

### Null and Attrition Handling

- Runs producing NaN or Inf: excluded from reliability computation, counted as attrition
- **Max attrition for reliability assessment:** 5% of pilot configurations. If exceeded → investigate numerical instability before proceeding.
- Runs with compute time > 5× median: flagged as outliers, included in primary analysis, excluded in sensitivity check

### Calibration vs. Locked Eval Split

- **Calibration set (not counted toward gates):** First 2 seeds from Phase 1 (2/5 = 40% of pilot)
  - Used to verify JVP implementation, tune QR cadence, debug infrastructure
  - Stability metrics computed but NOT used for GO/NO-GO
- **Locked eval set (counted toward gates):** Remaining 3 seeds from Phase 1
  - ICC, CV, Bland-Altman computed on this set only
  - Results determine Phase 1 GO/NO-GO

### Escalation Policy

1. **Primary ICC ≥ 0.75 on locked eval:** GO for Phase 2
2. **Primary ICC < 0.75:**
   - Diagnose root cause (tangent seed instability? cadence sensitivity? model-specific?)
   - One amendment permitted with Professor approval (e.g., increase tangent seeds from 10 to 20, fix identified numerical bug)
   - Re-run locked eval with amendment
3. **Second failure (ICC still < 0.75 after amendment):**
   - Downscope to diagnostic report
   - Report estimator instability as primary finding
   - Do not proceed to Phase 2 full-scale
   - No further amendments permitted

### No Fishing Rule

- The primary gate metric (ICC) was chosen before any data was collected
- Sensitivity metrics (CV, Bland-Altman) are pre-specified as diagnostic
- If ICC fails but CV passes, that does NOT constitute a gate pass
- Post-hoc introduction of alternative reliability metrics is forbidden
