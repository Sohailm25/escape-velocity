# ABOUTME: Phase 2 full-scale coordinator — dispatches 7,200 FTLE runs across 3 models on Modal.
# ABOUTME: Reads 720 Paper A trajectories, computes λ₁ per trajectory (10 tangent seeds), emits ftle_bundle.csv.

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDS_PATH = PROJECT_ROOT / "seeds" / "SEEDS_V2.json"
PAPER_A_BUNDLE = PROJECT_ROOT / "results" / "internal" / "paper-a" / "phase3_baseline" / "analysis_bundle.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2"

# Phase 2 protocol (locked from Phase 1)
TANGENT_SEEDS = list(range(10))  # 0-9
RENORM_CADENCE = 4  # locked from Phase 1 pilot
N_TANGENT_SEEDS = 10
EXPECTED_TRAJECTORIES = 720
EXPECTED_TOTAL_CALLS = 7200

# Attrition threshold (per IMPLEMENTATION_SPEC)
MAX_ATTRITION_PER_CONDITION = 0.10

# Model mapping: condition → (modal_app_name, model_id, model_revision)
MODEL_MAP = {
    "HOMO_A": (
        "paper-b-ftle-llama",
        "meta-llama/Llama-3.1-8B-Instruct",
        "0e9e39f249a16976918f6564b8830bc894c89659",
    ),
    "HOMO_B": (
        "paper-b-ftle-qwen",
        "Qwen/Qwen2.5-7B-Instruct",
        "a09a35458c702b33eeacc393d103063234e8bc28",
    ),
    "HOMO_C": (
        "paper-b-ftle-mistral",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "c170c708c41dac9275d15a8fff4eca08d52bab71",
    ),
    "HETERO_ROT": (
        "paper-b-ftle-llama",
        "meta-llama/Llama-3.1-8B-Instruct",
        "0e9e39f249a16976918f6564b8830bc894c89659",
    ),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_seeds() -> dict[str, str]:
    """Load SEEDS_V2.json and return a flat mapping: seed_id → prompt text."""
    with open(SEEDS_PATH) as f:
        seeds_data = json.load(f)

    seed_map = {}
    for bucket, prompts in seeds_data["buckets"].items():
        for idx, prompt in enumerate(prompts):
            seed_id = f"{bucket}_{idx}"
            seed_map[seed_id] = prompt
    return seed_map


def load_trajectories() -> list[dict]:
    """Load the 720 Paper A trajectories from analysis_bundle.csv."""
    trajectories = []
    with open(PAPER_A_BUNDLE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trajectories.append({
                "condition": row["condition"],
                "seed_id": row["seed_id"],
                "repeat_index": int(row["repeat_index"]),
            })
    return trajectories


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def build_dispatch_groups(
    trajectories: list[dict],
    seed_map: dict[str, str],
) -> dict[str, list[tuple]]:
    """Build per-model-app dispatch groups.

    Returns: {modal_app_name: [(prompt, tangent_seed, cadence, condition, seed_id, repeat_index), ...]}
    """
    groups = defaultdict(list)
    for traj in trajectories:
        condition = traj["condition"]
        seed_id = traj["seed_id"]
        repeat_index = traj["repeat_index"]
        prompt = seed_map[seed_id]
        app_name = MODEL_MAP[condition][0]

        for tangent_seed in TANGENT_SEEDS:
            groups[app_name].append((
                prompt,
                tangent_seed,
                RENORM_CADENCE,
                condition,
                seed_id,
                repeat_index,
            ))
    return dict(groups)


def dispatch_model_group(app_name: str, call_args: list[tuple]) -> list[dict]:
    """Dispatch all calls for one model app via starmap and collect results."""
    import modal

    runner_cls = modal.Cls.from_name(app_name, "FTLERunner")
    runner = runner_cls()

    results = []
    total = len(call_args)
    t0 = time.time()

    print(f"  [{app_name}] Dispatching {total} calls...")

    for i, result in enumerate(runner.run.starmap(call_args)):
        results.append(result)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            n_valid = sum(1 for r in results if not _is_failed(r))
            n_failed = len(results) - n_valid
            print(
                f"  [{app_name}] [{i+1}/{total}] valid={n_valid} failed={n_failed} "
                f"rate={rate:.1f}/s ETA={eta:.0f}s"
            )

    elapsed = time.time() - t0
    print(f"  [{app_name}] Done: {total} calls in {elapsed:.0f}s")
    return results


def dispatch_all(groups: dict[str, list[tuple]]) -> list[dict]:
    """Dispatch all model groups in parallel using threads."""
    all_results = []

    # Run model groups in parallel (each uses its own Modal app with starmap)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for app_name, call_args in groups.items():
            future = executor.submit(dispatch_model_group, app_name, call_args)
            futures[future] = app_name

        for future in as_completed(futures):
            app_name = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                print(f"  [{app_name}] Collected {len(results)} results")
            except Exception as e:
                print(f"  [{app_name}] FATAL ERROR: {e}")
                raise

    return all_results


# ---------------------------------------------------------------------------
# Result processing
# ---------------------------------------------------------------------------

def _is_failed(r: dict) -> bool:
    """Check if a result represents a failed run."""
    if "error" in r:
        return True
    if r.get("nan_detected") or r.get("inf_detected"):
        return True
    if math.isnan(r.get("lambda1", float("nan"))):
        return True
    return False


def compute_trajectory_summaries(results: list[dict]) -> list[dict]:
    """Aggregate per-tangent-seed results into per-trajectory summaries.

    Groups by (condition, seed_id, repeat_index), computes mean/std/cv over
    the 10 tangent seeds.
    """
    # Group by trajectory
    by_trajectory = defaultdict(list)
    for r in results:
        key = (r["condition"], r["seed_id"], r["repeat_index"])
        by_trajectory[key].append(r)

    summaries = []
    for (condition, seed_id, repeat_index), traj_results in sorted(by_trajectory.items()):
        lambdas = []
        layerwise_variances = []
        profile_slopes = []
        compute_times = []
        nan_count = 0

        for r in traj_results:
            if _is_failed(r):
                nan_count += 1
                continue

            lambdas.append(r["lambda1"])
            compute_times.append(r.get("compute_time_s", 0))

            profile = r.get("layerwise_profile", [])
            if len(profile) >= 2:
                arr = np.array(profile)
                layerwise_variances.append(float(np.var(arr)))
                # Slope: linear regression of profile vs layer index
                x = np.arange(len(arr))
                slope = float(np.polyfit(x, arr, 1)[0])
                profile_slopes.append(slope)

        n_valid = len(lambdas)
        if n_valid > 0:
            arr = np.array(lambdas)
            lambda1_mean = float(np.mean(arr))
            lambda1_std = float(np.std(arr, ddof=1)) if n_valid > 1 else 0.0
            lambda1_cv = lambda1_std / abs(lambda1_mean) if lambda1_mean != 0 else float("nan")
        else:
            lambda1_mean = float("nan")
            lambda1_std = float("nan")
            lambda1_cv = float("nan")

        model_id = MODEL_MAP[condition][1]

        summaries.append({
            "condition": condition,
            "seed_id": seed_id,
            "repeat_index": repeat_index,
            "lambda1_mean": round(lambda1_mean, 8),
            "lambda1_std": round(lambda1_std, 8),
            "lambda1_cv": round(lambda1_cv, 6) if not math.isnan(lambda1_cv) else float("nan"),
            "layerwise_variance_mean": round(float(np.mean(layerwise_variances)), 8) if layerwise_variances else float("nan"),
            "profile_slope_mean": round(float(np.mean(profile_slopes)), 8) if profile_slopes else float("nan"),
            "n_tangent_seeds": N_TANGENT_SEEDS,
            "n_valid": n_valid,
            "nan_count": nan_count,
            "renorm_cadence": RENORM_CADENCE,
            "model_id": model_id,
            "compute_time_s": round(float(np.sum(compute_times)), 2),
        })

    return summaries


def check_attrition(results: list[dict]) -> bool:
    """Check per-condition attrition. Returns True if OK, prints warnings."""
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r.get("condition", "UNKNOWN")].append(r)

    all_ok = True
    for condition in sorted(by_condition):
        cond_results = by_condition[condition]
        n_total = len(cond_results)
        n_failed = sum(1 for r in cond_results if _is_failed(r))
        attrition = n_failed / n_total if n_total > 0 else 0.0

        status = "OK" if attrition <= MAX_ATTRITION_PER_CONDITION else "STOP"
        print(f"  {condition}: {n_failed}/{n_total} failed ({attrition:.1%}) — {status}")

        if attrition > MAX_ATTRITION_PER_CONDITION:
            print(f"  *** ATTRITION EXCEEDED for {condition} (>{MAX_ATTRITION_PER_CONDITION:.0%}). STOPPING. ***")
            all_ok = False

    return all_ok


def save_ftle_bundle(summaries: list[dict], output_path: Path):
    """Save ftle_bundle.csv with the per-trajectory summaries."""
    fieldnames = [
        "condition", "seed_id", "repeat_index",
        "lambda1_mean", "lambda1_std", "lambda1_cv",
        "layerwise_variance_mean", "profile_slope_mean",
        "n_tangent_seeds", "n_valid", "nan_count",
        "renorm_cadence", "model_id", "compute_time_s",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 2 Full-Scale FTLE Computation")
    print("=" * 70)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Budget cap: $500")
    print(f"Renorm cadence: {RENORM_CADENCE} (locked from Phase 1)")
    print(f"Tangent seeds: {N_TANGENT_SEEDS} (0-9)")
    print()

    # Load data
    print("Loading seeds...")
    seed_map = load_seeds()
    print(f"  {len(seed_map)} seeds loaded")

    print("Loading Paper A trajectories...")
    trajectories = load_trajectories()
    print(f"  {len(trajectories)} trajectories loaded")
    assert len(trajectories) == EXPECTED_TRAJECTORIES, (
        f"Expected {EXPECTED_TRAJECTORIES} trajectories, got {len(trajectories)}"
    )

    # Build dispatch groups
    print("\nBuilding dispatch groups...")
    groups = build_dispatch_groups(trajectories, seed_map)
    total_calls = sum(len(v) for v in groups.values())
    print(f"  Total calls: {total_calls}")
    assert total_calls == EXPECTED_TOTAL_CALLS, (
        f"Expected {EXPECTED_TOTAL_CALLS} calls, got {total_calls}"
    )
    for app_name, calls in sorted(groups.items()):
        print(f"  {app_name}: {len(calls)} calls")

    # Dispatch
    print("\n" + "-" * 70)
    print("DISPATCHING")
    print("-" * 70)

    t_start = time.time()
    all_results = dispatch_all(groups)
    t_elapsed = time.time() - t_start

    print(f"\nTotal wall time: {t_elapsed:.0f}s ({t_elapsed/60:.1f}min)")
    print(f"Total results: {len(all_results)}")

    # Save raw results
    raw_path = RESULTS_DIR / "raw_results.json"
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Raw results saved to {raw_path}")

    # Check attrition
    print("\n" + "-" * 70)
    print("ATTRITION CHECK")
    print("-" * 70)
    attrition_ok = check_attrition(all_results)
    if not attrition_ok:
        print("\n*** ATTRITION EXCEEDED. Stopping per IMPLEMENTATION_SPEC. ***")
        print("Raw results have been saved. Investigate failures before re-running.")
        sys.exit(1)

    # Compute trajectory summaries
    print("\n" + "-" * 70)
    print("COMPUTING TRAJECTORY SUMMARIES")
    print("-" * 70)
    summaries = compute_trajectory_summaries(all_results)
    print(f"  {len(summaries)} trajectory summaries computed")
    assert len(summaries) == EXPECTED_TRAJECTORIES, (
        f"Expected {EXPECTED_TRAJECTORIES} summaries, got {len(summaries)}"
    )

    # Save ftle_bundle.csv
    bundle_path = RESULTS_DIR / "ftle_bundle.csv"
    save_ftle_bundle(summaries, bundle_path)
    print(f"  Saved to {bundle_path}")

    # Summary statistics
    print("\n" + "-" * 70)
    print("SUMMARY STATISTICS")
    print("-" * 70)

    valid_lambdas = [s["lambda1_mean"] for s in summaries if not math.isnan(s["lambda1_mean"])]
    total_nan = sum(s["nan_count"] for s in summaries)
    total_valid = sum(s["n_valid"] for s in summaries)

    print(f"  Trajectories: {len(summaries)}")
    print(f"  Valid λ₁ values: {len(valid_lambdas)}/{len(summaries)}")
    print(f"  Total tangent-seed results: valid={total_valid}, NaN={total_nan}")
    if valid_lambdas:
        arr = np.array(valid_lambdas)
        print(f"  λ₁ range: [{arr.min():.6f}, {arr.max():.6f}]")
        print(f"  λ₁ mean: {arr.mean():.6f} ± {arr.std():.6f}")

    # Per-condition breakdown
    print("\n  Per-condition:")
    by_cond = defaultdict(list)
    for s in summaries:
        by_cond[s["condition"]].append(s)

    for cond in ["HOMO_A", "HOMO_B", "HOMO_C", "HETERO_ROT"]:
        cond_summaries = by_cond[cond]
        cond_lambdas = [s["lambda1_mean"] for s in cond_summaries if not math.isnan(s["lambda1_mean"])]
        cond_nans = sum(s["nan_count"] for s in cond_summaries)
        if cond_lambdas:
            arr = np.array(cond_lambdas)
            print(f"    {cond}: n={len(cond_lambdas)}, λ₁={arr.mean():.6f}±{arr.std():.6f}, NaN_tangent={cond_nans}")
        else:
            print(f"    {cond}: n=0, all NaN")

    # SHA256 hashes
    print("\n" + "-" * 70)
    print("FILE HASHES (SHA256)")
    print("-" * 70)

    for fpath in [raw_path, bundle_path]:
        with open(fpath, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        print(f"  {fpath.name}: {sha}")

    print(f"\nWall time: {t_elapsed:.0f}s ({t_elapsed/60:.1f}min)")
    print("Phase 2 COMPLETE.")


if __name__ == "__main__":
    main()
