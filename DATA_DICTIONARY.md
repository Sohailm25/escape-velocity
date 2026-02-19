# Data Dictionary

## analysis_bundle.csv (720 rows)

The primary per-run dataset. Each row represents one unique (condition, seed, repeat) tuple.

| Column | Type | Description |
|---|---|---|
| condition | string | Experimental condition: HOMO_A, HOMO_B, HOMO_C, or HETERO_ROT |
| seed_id | string | Prompt seed identifier (e.g., "paraphrase_0", "factual_qa_3") |
| repeat_index | int | Repeat number (0–4) |
| collapse_rate | float | Fraction of turns classified as collapsed (0.0–1.0) |
| total_turns | int | Number of assistant turns in trajectory (always 40) |
| collapsed_turns | int | Number of turns meeting collapse criterion |
| first_collapse_turn | int/null | Turn index where collapse first detected (null if never) |
| early_stopped | bool | Whether run was early-stopped (always false for confirmatory) |
| status | string | Run outcome: "success" or "failed" |
| run_id | string | Unique run identifier |

## condition_summary.csv

Condition-level aggregate statistics.

| Column | Type | Description |
|---|---|---|
| condition | string | Condition name |
| n | int | Number of successful runs (180 each) |
| mean_collapse_rate | float | Mean of collapse_rate across runs |
| median_collapse_rate | float | Median of collapse_rate |
| sd_collapse_rate | float | Standard deviation of collapse_rate |

## kappa_metrics.json

Machine-readable detector reliability audit results.

| Field | Type | Description |
|---|---|---|
| n_valid | int | Number of valid rater pairs (180) |
| n_total | int | Total planned pairs (180) |
| null_count | int | Null labels after repair (0) |
| agreement_raw | int | Number of concordant pairs |
| agreement_pct | float | Percent agreement |
| cohens_kappa | float | Cohen's κ statistic |
| contingency | object | 2×2 contingency table counts |
| rubric_version | string | Rubric version used ("3.0") |
| gate_threshold | float | Prereg threshold (0.80) |
| gate_passed | bool | Whether gate was met (false) |
