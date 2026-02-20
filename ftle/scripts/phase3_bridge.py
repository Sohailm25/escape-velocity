# ABOUTME: Phase 3 bridge analysis — 6 pre-registered Spearman correlations between λ₁ summaries and Paper A collapse metrics.
# ABOUTME: Loads FTLE bundle and Paper A analysis bundle, computes correlations with Bonferroni-Holm correction and bootstrap CIs.

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FTLE_BUNDLE = PROJECT_ROOT / "results" / "phase2" / "ftle_bundle.csv"
PAPER_A_BUNDLE = (
    PROJECT_ROOT
    / "results"
    / "internal"
    / "paper-a"
    / "phase3_baseline"
    / "analysis_bundle.csv"
)
RATER1_PATH = (
    PROJECT_ROOT
    / "paper-a-escape-velocity"
    / "final_artifacts"
    / "v2_prereg_rater1_openai.jsonl"
)
RATER2_PATH = (
    PROJECT_ROOT
    / "paper-a-escape-velocity"
    / "final_artifacts"
    / "v2_prereg_rater2_anthropic.jsonl"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase3"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOOTSTRAP_N = 10_000
PREREG_RHO_THRESHOLD = 0.40
PREREG_ALPHA = 0.05
EXPECTED_ROWS = 720
RNG_SEED = 42

# 6 pre-registered test pairings (PREREG_v2.md §Bridge-Analysis Protocol)
# Column names mapped from PREREG variable names to CSV column names:
#   lambda1_mean  → lambda1_mean
#   lambda1_var   → layerwise_variance_mean  (variance of layerwise λ profile, per PREREG_v2)
#   lambda1_slope → profile_slope_mean
#   collapse_incidence → has_collapse
TESTS = [
    {"test_id": 1, "lambda1_summary": "lambda1_mean", "paper_a_metric": "collapse_rate"},
    {"test_id": 2, "lambda1_summary": "lambda1_mean", "paper_a_metric": "first_collapse_turn"},
    {"test_id": 3, "lambda1_summary": "lambda1_mean", "paper_a_metric": "has_collapse"},
    {"test_id": 4, "lambda1_summary": "layerwise_variance_mean", "paper_a_metric": "collapse_rate"},
    {"test_id": 5, "lambda1_summary": "profile_slope_mean", "paper_a_metric": "collapse_rate"},
    {"test_id": 6, "lambda1_summary": "profile_slope_mean", "paper_a_metric": "first_collapse_turn"},
]

# PREREG names for output reporting
PREREG_NAMES = {
    "layerwise_variance_mean": "lambda1_var",
    "profile_slope_mean": "lambda1_slope",
    "has_collapse": "collapse_incidence",

}

# Censored value for first_collapse_turn when no collapse occurred (PREREG_v2 primary handling)
CENSORED_FIRST_COLLAPSE = 40


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def merge_key(row: dict) -> tuple[str, str, int]:
    return (row["condition"], row["seed_id"], int(row["repeat_index"]))


def bonferroni_holm(p_values: list[float]) -> list[float]:
    """Apply Bonferroni-Holm step-down correction to a list of p-values."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        corrected = p * (n - rank)
        # Enforce monotonicity: adjusted p can't decrease
        cummax = max(cummax, corrected)
        adjusted[orig_idx] = min(cummax, 1.0)

    return adjusted


def bootstrap_ci_bc(
    x: np.ndarray,
    y: np.ndarray,
    observed_rho: float,
    n_bootstrap: int,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Bias-corrected percentile bootstrap CI for Spearman rho."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    n = len(x)
    boot_rhos = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rho_b, _ = stats.spearmanr(x[idx], y[idx])
        boot_rhos[i] = rho_b

    # Bias correction factor z0
    prop_below = np.mean(boot_rhos < observed_rho)
    # Clamp to avoid infinite z0
    prop_below = np.clip(prop_below, 1e-10, 1.0 - 1e-10)
    z0 = stats.norm.ppf(prop_below)

    # Adjusted percentiles
    z_alpha_low = stats.norm.ppf(alpha / 2)
    z_alpha_high = stats.norm.ppf(1 - alpha / 2)

    p_low = stats.norm.cdf(2 * z0 + z_alpha_low) * 100
    p_high = stats.norm.cdf(2 * z0 + z_alpha_high) * 100

    ci_lower = float(np.percentile(boot_rhos, p_low))
    ci_upper = float(np.percentile(boot_rhos, p_high))

    return ci_lower, ci_upper


def run_tests(
    merged: list[dict],
    tests: list[dict],
    label: str,
    rng: np.random.Generator,
) -> dict:
    """Run all 6 pre-registered Spearman correlations on a dataset."""
    results = []
    raw_p_values = []

    for test in tests:
        x_col = test["lambda1_summary"]
        y_col = test["paper_a_metric"]

        # Primary handling per PREREG_v2: use ALL trajectories.
        # For first_collapse_turn: non-collapsing runs censored at turn=40.
        subset = merged
        n = len(subset)
        x = np.array([float(r[x_col]) for r in subset])

        # has_collapse is "true"/"false" string — convert to 0/1
        if y_col == "has_collapse":
            y = np.array([1.0 if r[y_col] == "true" else 0.0 for r in subset])
        elif y_col == "first_collapse_turn":
            y = np.array([
                float(r[y_col]) if r["has_collapse"] == "true"
                else float(CENSORED_FIRST_COLLAPSE)
                for r in subset
            ])
        else:
            y = np.array([float(r[y_col]) for r in subset])

        rho, p_raw = stats.spearmanr(x, y)

        ci_lower, ci_upper = bootstrap_ci_bc(
            x, y, rho, BOOTSTRAP_N, PREREG_ALPHA, rng
        )

        # PREREG names for output
        x_prereg = PREREG_NAMES.get(x_col, x_col)
        y_prereg = PREREG_NAMES.get(y_col, y_col)

        result = {
            "test_id": test["test_id"],
            "lambda1_summary": x_prereg,
            "lambda1_summary_csv_col": x_col,
            "paper_a_metric": y_prereg,
            "paper_a_metric_csv_col": y_col,
            "n": n,
            "rho": round(float(rho), 6),
            "p_raw": float(p_raw),
            "p_holm": None,  # filled after Holm correction
            "ci_95_lower": round(ci_lower, 6),
            "ci_95_upper": round(ci_upper, 6),
            "pass_prereg": None,  # filled after Holm correction
        }
        results.append(result)
        raw_p_values.append(float(p_raw))

    # Apply Bonferroni-Holm correction
    adjusted = bonferroni_holm(raw_p_values)
    for i, result in enumerate(results):
        result["p_holm"] = float(adjusted[i])
        ci_excludes_zero = not (result["ci_95_lower"] <= 0 <= result["ci_95_upper"])
        result["pass_prereg"] = (
            abs(result["rho"]) >= PREREG_RHO_THRESHOLD
            and adjusted[i] < PREREG_ALPHA
            and ci_excludes_zero
        )

    n_passing = sum(1 for r in results if r["pass_prereg"])
    return {
        "tests": results,
        "any_pass": n_passing > 0,
        "n_passing": n_passing,
        "holm_adjustment_details": {
            "method": "Bonferroni-Holm step-down",
            "n_tests": len(tests),
            "family_alpha": PREREG_ALPHA,
            "raw_p_values": [round(p, 8) for p in raw_p_values],
            "adjusted_p_values": [round(p, 8) for p in adjusted],
        },
    }


def load_rater_agreed_keys() -> set[tuple[str, str, int]] | None:
    """Load 167 rater-agreed trajectory keys from kappa audit labels."""
    if not RATER1_PATH.exists() or not RATER2_PATH.exists():
        return None

    rater1 = load_jsonl(RATER1_PATH)
    rater2 = load_jsonl(RATER2_PATH)

    r1_map = {r["sample_id"]: r for r in rater1}
    r2_map = {r["sample_id"]: r for r in rater2}

    agreed_keys = set()
    for sid in r1_map:
        if sid not in r2_map:
            continue
        l1 = r1_map[sid].get("label")
        l2 = r2_map[sid].get("label")
        if l1 is None or l2 is None:
            continue
        if l1 == l2:
            parts = r1_map[sid]["tuple_key"].split("|")
            agreed_keys.add((parts[0], parts[1], int(parts[2])))

    return agreed_keys if agreed_keys else None


def print_summary(primary: dict, sensitivity: dict | None) -> None:
    """Print a clean summary table to stdout."""
    print("=" * 80)
    print("PHASE 3 BRIDGE ANALYSIS — PRE-REGISTERED SPEARMAN CORRELATIONS")
    print("=" * 80)
    print()

    def _table(results: dict, label: str) -> None:
        print(f"--- {label} ---")
        print()
        header = f"{'#':>2}  {'λ₁ summary':<18} {'Paper A metric':<22} {'n':>5}  {'ρ':>8}  {'p_raw':>10}  {'p_holm':>10}  {'95% CI':>20}  {'Pass':>5}"
        print(header)
        print("-" * len(header))
        for t in results["tests"]:
            ci_str = f"[{t['ci_95_lower']:+.4f}, {t['ci_95_upper']:+.4f}]"
            pass_str = "YES" if t["pass_prereg"] else "no"
            print(
                f"{t['test_id']:>2}  {t['lambda1_summary']:<18} {t['paper_a_metric']:<22} {t['n']:>5}  "
                f"{t['rho']:>+8.4f}  {t['p_raw']:>10.2e}  {t['p_holm']:>10.2e}  {ci_str:>20}  {pass_str:>5}"
            )
        print()
        print(f"  Passing tests: {results['n_passing']}/6")
        print(f"  Any pass (prereg success): {results['any_pass']}")
        print()

    _table(primary, "PRIMARY (all trajectories)")

    if sensitivity is not None and isinstance(sensitivity, dict) and "tests" in sensitivity:
        _table(sensitivity, "SENSITIVITY (rater-agreed subset)")
    elif isinstance(sensitivity, dict):
        print(f"--- SENSITIVITY ---")
        print(f"  Not available: {sensitivity.get('not_available', 'unknown reason')}")
        print()

    print("=" * 80)
    print("NOTE: Descriptive patterns only. Paper A collapse labels have unconfirmed")
    print("inter-rater reliability (κ = 0.566, threshold 0.80 not met).")
    print("=" * 80)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    # -----------------------------------------------------------------------
    # 1. Load and hash input files
    # -----------------------------------------------------------------------
    print("Loading data...")
    ftle_sha = sha256_file(FTLE_BUNDLE)
    papera_sha = sha256_file(PAPER_A_BUNDLE)
    print(f"  FTLE bundle SHA256:    {ftle_sha}")
    print(f"  Paper A bundle SHA256: {papera_sha}")

    ftle_rows = load_csv(FTLE_BUNDLE)
    papera_rows = load_csv(PAPER_A_BUNDLE)

    assert len(ftle_rows) == EXPECTED_ROWS, (
        f"FTLE bundle has {len(ftle_rows)} rows, expected {EXPECTED_ROWS}"
    )
    assert len(papera_rows) == EXPECTED_ROWS, (
        f"Paper A bundle has {len(papera_rows)} rows, expected {EXPECTED_ROWS}"
    )

    # -----------------------------------------------------------------------
    # 2. Merge on (condition, seed_id, repeat_index)
    # -----------------------------------------------------------------------
    print("Merging datasets...")
    papera_by_key = {}
    for row in papera_rows:
        key = merge_key(row)
        assert key not in papera_by_key, f"Duplicate Paper A key: {key}"
        papera_by_key[key] = row

    merged = []
    missing_keys = []
    for ftle_row in ftle_rows:
        key = merge_key(ftle_row)
        papera_row = papera_by_key.get(key)
        if papera_row is None:
            missing_keys.append(key)
            continue
        # Merge: FTLE columns + Paper A collapse columns
        m = dict(ftle_row)
        m["collapse_rate"] = papera_row["collapse_rate"]
        m["first_collapse_turn"] = papera_row["first_collapse_turn"]
        m["has_collapse"] = papera_row["has_collapse"]
        merged.append(m)

    n_merged = len(merged)
    n_excluded = len(missing_keys)

    print(f"  Merged: {n_merged} rows")
    if n_excluded > 0:
        print(f"  WARNING: {n_excluded} FTLE rows had no Paper A match: {missing_keys[:5]}")

    assert n_merged == EXPECTED_ROWS, (
        f"Merge produced {n_merged} rows, expected {EXPECTED_ROWS}. "
        f"Missing keys: {missing_keys[:10]}"
    )

    # -----------------------------------------------------------------------
    # 3. Primary analysis — all 720 trajectories
    # -----------------------------------------------------------------------
    print(f"\nRunning primary analysis (n={n_merged})...")
    primary = run_tests(merged, TESTS, "primary", rng)

    # -----------------------------------------------------------------------
    # 4. Sensitivity analysis — rater-agreed subset
    # -----------------------------------------------------------------------
    print("Loading rater agreement data for sensitivity analysis...")
    agreed_keys = load_rater_agreed_keys()

    sensitivity: dict
    if agreed_keys is not None:
        sensitivity_rows = [r for r in merged if merge_key(r) in agreed_keys]
        n_sensitivity = len(sensitivity_rows)
        n_expected_agreed = 167
        print(f"  Rater-agreed trajectories found: {n_sensitivity} (expected ~{n_expected_agreed})")

        if n_sensitivity >= 50:  # minimum viable sample
            sensitivity = run_tests(sensitivity_rows, TESTS, "sensitivity", rng)
            sensitivity["n_trajectories"] = n_sensitivity
            sensitivity["n_audited"] = 180
            sensitivity["n_agreed"] = len(agreed_keys)
        else:
            sensitivity = {
                "not_available": f"Only {n_sensitivity} rater-agreed trajectories found (need >= 50)"
            }
    else:
        sensitivity = {
            "not_available": "Rater label files not found at expected paths"
        }

    # -----------------------------------------------------------------------
    # 5. Assemble and save results
    # -----------------------------------------------------------------------
    prereg_success = primary["any_pass"]

    output = {
        "paper_a_freeze_sha256": papera_sha,
        "paper_b_ftle_bundle_sha256": ftle_sha,
        "n_trajectories": EXPECTED_ROWS,
        "n_merged": n_merged,
        "n_excluded": n_excluded,
        "primary_results": primary,
        "sensitivity_results": sensitivity,
        "prereg_success": prereg_success,
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_rng_seed": RNG_SEED,
        "prereg_rho_threshold": PREREG_RHO_THRESHOLD,
        "prereg_alpha": PREREG_ALPHA,
        "label_caveat": (
            "Paper A collapse labels generated by detector with unconfirmed "
            "inter-rater reliability (Cohen's kappa = 0.566, threshold 0.80 not met). "
            "See PREREG_v2.md Paper A Label Caveat section."
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    output_path = OUTPUT_DIR / "bridge_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # -----------------------------------------------------------------------
    # 6. Print summary
    # -----------------------------------------------------------------------
    print()
    print_summary(
        primary,
        sensitivity if "tests" in sensitivity else sensitivity,
    )

    # Exit code: 0 if prereg success, 1 if not (informational, not an error)
    if prereg_success:
        print("\nPRE-REGISTRATION SUCCESS CRITERION MET")
    else:
        print("\nPre-registration success criterion NOT met")

    return


if __name__ == "__main__":
    main()
