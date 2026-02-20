# ABOUTME: Phase 1 pilot canary for Paper B — dispatches 450 FTLE runs on Modal A100-80GB.
# ABOUTME: Computes stability metrics (CV, renorm sensitivity, rerun r, ICC) and emits GO/NO-GO.

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDS_PATH = PROJECT_ROOT / "seeds" / "SEEDS_V2.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase1"

# Pilot protocol
PILOT_BUCKETS = [
    "factual_qa",
    "instruction_following",
    "creative_rewrite",
    "argumentative",
    "planning",
]
PROMPT_INDEX = 0  # first prompt from each bucket
TANGENT_SEEDS = list(range(10))  # 0-9
RENORM_CADENCES = [1, 2, 4]
N_RERUNS = 3  # 3 independent reruns

# Thresholds from PREREG_v2
THRESHOLD_CV = 0.15
THRESHOLD_RENORM_DELTA = 0.05
THRESHOLD_RERUN_R = 0.85
THRESHOLD_ICC = 0.75

# Calibration/locked eval split (from PREREG_v2 Reliability Framework)
CALIBRATION_SEEDS = 2  # first 2 seeds are calibration
LOCKED_EVAL_SEEDS = 3  # remaining 3 are locked eval

# Max attrition for reliability assessment
MAX_ATTRITION_FRACTION = 0.05


# ---------------------------------------------------------------------------
# ICC(2,1) computation — manual formula
# ---------------------------------------------------------------------------

def compute_icc_2_1(data_matrix: np.ndarray) -> float:
    """Compute ICC(2,1) — two-way random, single measures, absolute agreement.

    Args:
        data_matrix: (n_subjects, k_raters) array. Each row is a subject
            (configuration), each column is a rater (rerun).

    Returns:
        ICC(2,1) value.

    Formula:
        ICC(2,1) = (BMS - EMS) / (BMS + (k-1)*EMS + k*(JMS-EMS)/n)
        where BMS = between-subjects mean square
              JMS = between-judges (raters) mean square
              EMS = residual (error) mean square
              k = number of raters
              n = number of subjects
    """
    n, k = data_matrix.shape
    grand_mean = data_matrix.mean()

    # Row means (subject means)
    row_means = data_matrix.mean(axis=1)
    # Column means (rater/judge means)
    col_means = data_matrix.mean(axis=0)

    # Sum of squares
    ss_between = k * np.sum((row_means - grand_mean) ** 2)  # between subjects
    ss_judges = n * np.sum((col_means - grand_mean) ** 2)  # between judges
    ss_total = np.sum((data_matrix - grand_mean) ** 2)
    ss_error = ss_total - ss_between - ss_judges  # residual

    # Mean squares
    df_between = n - 1
    df_judges = k - 1
    df_error = (n - 1) * (k - 1)

    bms = ss_between / df_between if df_between > 0 else 0.0
    jms = ss_judges / df_judges if df_judges > 0 else 0.0
    ems = ss_error / df_error if df_error > 0 else 0.0

    # ICC(2,1) formula
    denominator = bms + (k - 1) * ems + k * (jms - ems) / n
    if denominator == 0:
        return 0.0
    return (bms - ems) / denominator


# ---------------------------------------------------------------------------
# Stability metric computation
# ---------------------------------------------------------------------------

def compute_stability_metrics(results: list[dict]) -> dict:
    """Compute all Phase 1 stability metrics from raw results.

    Args:
        results: List of result dicts from FTLERunner.run().

    Returns:
        Dict with all stability metrics, split by calibration vs locked eval.
    """
    # Filter out failed results
    valid = [r for r in results if not _is_failed(r)]
    failed = [r for r in results if _is_failed(r)]
    attrition_rate = len(failed) / len(results) if results else 0.0

    # Group results by seed_id
    by_seed = defaultdict(list)
    for r in valid:
        by_seed[r["seed_id"]].append(r)

    # Split into calibration and locked eval
    seed_ids = [f"{b}_{PROMPT_INDEX}" for b in PILOT_BUCKETS]
    calibration_ids = set(seed_ids[:CALIBRATION_SEEDS])
    locked_eval_ids = set(seed_ids[CALIBRATION_SEEDS:])

    calibration_results = [r for r in valid if r["seed_id"] in calibration_ids]
    locked_eval_results = [r for r in valid if r["seed_id"] in locked_eval_ids]

    # Compute metrics on each split
    cal_metrics = _compute_split_metrics(calibration_results, "calibration")
    eval_metrics = _compute_split_metrics(locked_eval_results, "locked_eval")

    # Determine cadence lock: lowest median CV across all valid data
    all_cadence_cvs = defaultdict(list)
    for r_set in [calibration_results, locked_eval_results]:
        cadence_cvs = _compute_cadence_cvs(r_set)
        for cadence, cvs in cadence_cvs.items():
            all_cadence_cvs[cadence].extend(cvs)

    cadence_median_cv = {}
    for cadence, cvs in all_cadence_cvs.items():
        if cvs:
            cadence_median_cv[cadence] = float(np.median(cvs))

    locked_cadence = min(cadence_median_cv, key=cadence_median_cv.get) if cadence_median_cv else 2

    # GO/NO-GO on locked eval only
    go_cv = eval_metrics["median_cv"] <= THRESHOLD_CV
    go_renorm = eval_metrics["max_renorm_delta"] <= THRESHOLD_RENORM_DELTA
    go_rerun_r = eval_metrics["median_rerun_r"] >= THRESHOLD_RERUN_R
    go_icc = eval_metrics["icc_2_1"] >= THRESHOLD_ICC
    overall_go = go_cv and go_renorm and go_rerun_r and go_icc

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "model_revision": "0e9e39f2",
            "n_seeds": len(PILOT_BUCKETS),
            "n_tangent_seeds": len(TANGENT_SEEDS),
            "n_cadences": len(RENORM_CADENCES),
            "n_reruns": N_RERUNS,
            "total_dispatched": len(results),
            "total_valid": len(valid),
            "total_failed": len(failed),
            "attrition_rate": round(attrition_rate, 4),
            "calibration_seed_ids": sorted(calibration_ids),
            "locked_eval_seed_ids": sorted(locked_eval_ids),
        },
        "calibration_metrics": cal_metrics,
        "locked_eval_metrics": eval_metrics,
        "cadence_selection": {
            "cadence_median_cv": cadence_median_cv,
            "locked_cadence": locked_cadence,
            "selection_criterion": "lowest_median_cv",
        },
        "gates": {
            "tangent_seed_cv": {
                "threshold": THRESHOLD_CV,
                "actual": eval_metrics["median_cv"],
                "pass": go_cv,
            },
            "renorm_sensitivity": {
                "threshold": THRESHOLD_RENORM_DELTA,
                "actual": eval_metrics["max_renorm_delta"],
                "pass": go_renorm,
            },
            "rerun_pearson_r": {
                "threshold": THRESHOLD_RERUN_R,
                "actual": eval_metrics["median_rerun_r"],
                "pass": go_rerun_r,
            },
            "icc_2_1": {
                "threshold": THRESHOLD_ICC,
                "actual": eval_metrics["icc_2_1"],
                "pass": go_icc,
            },
            "attrition": {
                "threshold": MAX_ATTRITION_FRACTION,
                "actual": attrition_rate,
                "pass": attrition_rate <= MAX_ATTRITION_FRACTION,
            },
        },
        "verdict": "GO" if overall_go else "NO_GO",
    }


def _is_failed(r: dict) -> bool:
    """Check if a result represents a failed run."""
    if "error" in r:
        return True
    if r.get("nan_detected") or r.get("inf_detected"):
        return True
    if math.isnan(r.get("lambda1", float("nan"))):
        return True
    return False


def _compute_split_metrics(results: list[dict], split_name: str) -> dict:
    """Compute stability metrics for a split (calibration or locked eval)."""
    if not results:
        return {
            "split": split_name,
            "n_results": 0,
            "median_cv": float("nan"),
            "all_cvs": [],
            "max_renorm_delta": float("nan"),
            "all_renorm_deltas": [],
            "median_rerun_r": float("nan"),
            "all_rerun_rs": [],
            "icc_2_1": float("nan"),
        }

    # 1. Tangent-seed CV: group by (seed_id, cadence, rerun_idx)
    cv_values = []
    by_config = defaultdict(list)
    for r in results:
        key = (r["seed_id"], r["renorm_cadence"], r["rerun_idx"])
        by_config[key].append(r["lambda1"])

    for key, lambdas in by_config.items():
        if len(lambdas) >= 2:
            arr = np.array(lambdas)
            mean_val = np.mean(arr)
            if mean_val != 0:
                cv = float(np.std(arr, ddof=1) / abs(mean_val))
                cv_values.append(cv)

    # 2. Renorm sensitivity: group by (seed_id, tangent_seed, rerun_idx)
    renorm_deltas = []
    by_renorm = defaultdict(dict)
    for r in results:
        key = (r["seed_id"], r["tangent_seed"], r["rerun_idx"])
        by_renorm[key][r["renorm_cadence"]] = r["lambda1"]

    for key, cadence_map in by_renorm.items():
        cadences = sorted(cadence_map.keys())
        for i in range(len(cadences)):
            for j in range(i + 1, len(cadences)):
                delta = abs(cadence_map[cadences[i]] - cadence_map[cadences[j]])
                renorm_deltas.append(delta)

    # 3. Rerun Pearson r: group by (seed_id, tangent_seed, cadence)
    rerun_rs = []
    by_rerun = defaultdict(dict)
    for r in results:
        key = (r["seed_id"], r["tangent_seed"], r["renorm_cadence"])
        by_rerun[key][r["rerun_idx"]] = r["lambda1"]

    # Compute pairwise Pearson r across reruns for matched configurations
    # Group all configs that share (seed_id, cadence) and have all 3 reruns
    rerun_vectors = defaultdict(lambda: defaultdict(list))
    for r in results:
        group_key = (r["seed_id"], r["renorm_cadence"])
        rerun_vectors[group_key][r["rerun_idx"]].append(
            (r["tangent_seed"], r["lambda1"])
        )

    for group_key, rerun_map in rerun_vectors.items():
        rerun_indices = sorted(rerun_map.keys())
        if len(rerun_indices) < 2:
            continue
        for i in range(len(rerun_indices)):
            for j in range(i + 1, len(rerun_indices)):
                # Build matched vectors
                seeds_i = {ts: lam for ts, lam in rerun_map[rerun_indices[i]]}
                seeds_j = {ts: lam for ts, lam in rerun_map[rerun_indices[j]]}
                common_seeds = sorted(set(seeds_i.keys()) & set(seeds_j.keys()))
                if len(common_seeds) >= 3:
                    vec_i = [seeds_i[s] for s in common_seeds]
                    vec_j = [seeds_j[s] for s in common_seeds]
                    r_val, _ = stats.pearsonr(vec_i, vec_j)
                    if not math.isnan(r_val):
                        rerun_rs.append(r_val)

    # 4. ICC(2,1): build (n_configs, k_reruns) matrix
    icc_configs = defaultdict(dict)
    for r in results:
        config_key = (r["seed_id"], r["tangent_seed"], r["renorm_cadence"])
        icc_configs[config_key][r["rerun_idx"]] = r["lambda1"]

    # Only include configs with all reruns present
    complete_configs = []
    for config_key, rerun_map in icc_configs.items():
        if len(rerun_map) == N_RERUNS:
            row = [rerun_map[i] for i in range(N_RERUNS)]
            complete_configs.append(row)

    if len(complete_configs) >= 3:
        icc_matrix = np.array(complete_configs)
        icc_value = compute_icc_2_1(icc_matrix)
    else:
        icc_value = float("nan")

    return {
        "split": split_name,
        "n_results": len(results),
        "median_cv": float(np.median(cv_values)) if cv_values else float("nan"),
        "mean_cv": float(np.mean(cv_values)) if cv_values else float("nan"),
        "all_cvs": [round(v, 6) for v in cv_values],
        "max_renorm_delta": float(max(renorm_deltas)) if renorm_deltas else float("nan"),
        "median_renorm_delta": float(np.median(renorm_deltas)) if renorm_deltas else float("nan"),
        "all_renorm_deltas": [round(v, 6) for v in renorm_deltas],
        "median_rerun_r": float(np.median(rerun_rs)) if rerun_rs else float("nan"),
        "mean_rerun_r": float(np.mean(rerun_rs)) if rerun_rs else float("nan"),
        "all_rerun_rs": [round(v, 6) for v in rerun_rs],
        "icc_2_1": round(float(icc_value), 6),
        "icc_n_complete_configs": len(complete_configs),
    }


def _compute_cadence_cvs(results: list[dict]) -> dict[int, list[float]]:
    """Compute per-cadence CVs for cadence locking decision."""
    by_config = defaultdict(list)
    for r in results:
        key = (r["seed_id"], r["renorm_cadence"], r["rerun_idx"])
        by_config[key].append(r["lambda1"])

    cadence_cvs = defaultdict(list)
    for (seed_id, cadence, rerun_idx), lambdas in by_config.items():
        if len(lambdas) >= 2:
            arr = np.array(lambdas)
            mean_val = np.mean(arr)
            if mean_val != 0:
                cv = float(np.std(arr, ddof=1) / abs(mean_val))
                cadence_cvs[cadence].append(cv)

    return dict(cadence_cvs)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def build_dispatch_list() -> list[dict]:
    """Build the list of 450 FTLE calls to dispatch."""
    with open(SEEDS_PATH) as f:
        seeds_data = json.load(f)

    calls = []
    for bucket in PILOT_BUCKETS:
        prompt = seeds_data["buckets"][bucket][PROMPT_INDEX]
        seed_id = f"{bucket}_{PROMPT_INDEX}"

        for tangent_seed in TANGENT_SEEDS:
            for cadence in RENORM_CADENCES:
                for rerun_idx in range(N_RERUNS):
                    calls.append({
                        "prompt": prompt,
                        "tangent_seed": tangent_seed,
                        "renorm_cadence": cadence,
                        "rerun_idx": rerun_idx,
                        "seed_id": seed_id,
                    })

    return calls


def dispatch_modal(calls: list[dict]) -> list[dict]:
    """Dispatch calls to Modal FTLERunner and collect results."""
    # Import modal runner
    sys.path.insert(0, str(PROJECT_ROOT))
    import modal

    runner_cls = modal.Cls.from_name("paper-b-ftle", "FTLERunner")
    runner = runner_cls()

    results = []
    total = len(calls)
    t0 = time.time()

    print(f"Dispatching {total} FTLE calls to Modal...")
    print(f"  Seeds: {len(PILOT_BUCKETS)}")
    print(f"  Tangent seeds: {len(TANGENT_SEEDS)}")
    print(f"  Cadences: {RENORM_CADENCES}")
    print(f"  Reruns: {N_RERUNS}")
    print()

    # Use starmap for parallel dispatch
    call_args = [
        (c["prompt"], c["tangent_seed"], c["renorm_cadence"], c["rerun_idx"], c["seed_id"])
        for c in calls
    ]

    for i, result in enumerate(runner.run.starmap(call_args)):
        results.append(result)
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (total - i - 1) / rate if rate > 0 else 0

        # Progress every 10 results
        if (i + 1) % 10 == 0 or (i + 1) == total:
            n_valid = sum(1 for r in results if not _is_failed(r))
            n_failed = sum(1 for r in results if _is_failed(r))
            print(
                f"  [{i+1}/{total}] valid={n_valid} failed={n_failed} "
                f"rate={rate:.1f}/s ETA={eta:.0f}s"
            )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build dispatch list
    calls = build_dispatch_list()
    assert len(calls) == 450, f"Expected 450 calls, got {len(calls)}"
    print(f"Phase 1 Pilot: {len(calls)} FTLE calls prepared")
    print(f"Budget cap: $100")
    print()

    # Dispatch to Modal
    t_start = time.time()
    results = dispatch_modal(calls)
    t_elapsed = time.time() - t_start

    # Save raw results
    raw_path = RESULTS_DIR / "raw_results.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to {raw_path}")
    print(f"Total wall time: {t_elapsed:.0f}s ({t_elapsed/60:.1f}min)")

    # Compute stability metrics
    print("\n" + "=" * 60)
    print("STABILITY METRICS")
    print("=" * 60)

    metrics = compute_stability_metrics(results)

    # Save pilot results
    pilot_path = RESULTS_DIR / "pilot_results.json"
    with open(pilot_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Pilot results saved to {pilot_path}")

    # Print summary
    _print_summary(metrics)

    # Print SHA256
    with open(pilot_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    print(f"\npilot_results.json SHA256: {sha}")

    return metrics


def _print_summary(metrics: dict):
    """Print human-readable summary of pilot results."""
    proto = metrics["protocol"]
    print(f"\nProtocol: {proto['total_dispatched']} dispatched, "
          f"{proto['total_valid']} valid, {proto['total_failed']} failed")
    print(f"Attrition: {proto['attrition_rate']:.2%}")

    print(f"\n--- Calibration (diagnostic, not gating) ---")
    cal = metrics["calibration_metrics"]
    print(f"  Median CV: {cal['median_cv']:.4f}")
    print(f"  Max |Δλ₁|: {cal['max_renorm_delta']:.4f}")
    print(f"  Median rerun r: {cal['median_rerun_r']:.4f}")
    print(f"  ICC(2,1): {cal['icc_2_1']:.4f}")

    print(f"\n--- Locked Eval (GATING) ---")
    ev = metrics["locked_eval_metrics"]
    print(f"  Median CV: {ev['median_cv']:.4f}")
    print(f"  Max |Δλ₁|: {ev['max_renorm_delta']:.4f}")
    print(f"  Median rerun r: {ev['median_rerun_r']:.4f}")
    print(f"  ICC(2,1): {ev['icc_2_1']:.4f}")

    print(f"\n--- Cadence Selection ---")
    cs = metrics["cadence_selection"]
    for cadence, cv in sorted(cs["cadence_median_cv"].items(), key=lambda x: int(x[0])):
        locked = " ← LOCKED" if int(cadence) == cs["locked_cadence"] else ""
        print(f"  cadence={cadence}: median CV={cv:.4f}{locked}")

    print(f"\n--- GATES ---")
    for gate_name, gate in metrics["gates"].items():
        status = "PASS" if gate["pass"] else "FAIL"
        print(f"  {gate_name}: {status} (actual={gate['actual']:.4f}, "
              f"threshold={gate['threshold']})")

    verdict = metrics["verdict"]
    print(f"\n{'='*60}")
    print(f"  VERDICT: {verdict}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
