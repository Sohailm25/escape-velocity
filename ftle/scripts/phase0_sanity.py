# ABOUTME: Phase 0 sanity check — validates JVP FTLE estimator against finite-difference on GPT-2.
# ABOUTME: Compares λ₁ estimates across 5 prompts × 10 tangent seeds × 3 renorm cadences.

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy/torch types in JSON serialization."""

    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.compute_ftle import compute_ftle, compute_ftle_finite_difference

# --- Configuration ---
TEST_PROMPTS = [
    "The theory of general relativity predicts that massive objects",
    "In a recent study published in Nature, researchers found that",
    "The quick brown fox jumps over the lazy dog near the riverbank",
    "When asked about the future of artificial intelligence, the professor explained",
    "The economic implications of climate change include rising sea levels and",
]
TANGENT_SEEDS = list(range(10))  # 0..9
RENORM_CADENCES = [1, 2, 4]
FD_EPSILON = 1e-7  # Optimal for float64


def main():
    t_start = time.time()
    output_dir = Path(__file__).resolve().parent.parent / "results" / "phase0"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    print(f"Device: {device}")
    print(f"Loading GPT-2 in float64 for ground-truth FD comparison...")

    model = GPT2LMHeadModel.from_pretrained("gpt2").double()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()

    print(f"Model: GPT-2 (12 layers, d_model=768)")
    print(f"Prompts: {len(TEST_PROMPTS)}")
    print(f"Tangent seeds: {len(TANGENT_SEEDS)}")
    print(f"Renorm cadences: {RENORM_CADENCES}")
    print(f"FD epsilon: {FD_EPSILON}")
    print(f"Total configurations: {len(TEST_PROMPTS) * len(TANGENT_SEEDS) * len(RENORM_CADENCES)}")
    print()

    per_prompt_results = []
    jvp_lambdas = []
    fd_lambdas = []
    n_nan_jvp = 0
    n_nan_fd = 0

    for p_idx, prompt in enumerate(TEST_PROMPTS):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        print(f"Prompt {p_idx}: \"{prompt[:50]}...\" ({input_ids.shape[1]} tokens)")

        for t_seed in TANGENT_SEEDS:
            for cadence in RENORM_CADENCES:
                jvp_result = compute_ftle(
                    model, input_ids, tangent_seed=t_seed,
                    renorm_cadence=cadence, device=device,
                )
                fd_result = compute_ftle_finite_difference(
                    model, input_ids, tangent_seed=t_seed,
                    renorm_cadence=cadence, device=device,
                    epsilon=FD_EPSILON,
                )

                jvp_l1 = jvp_result["lambda1"]
                fd_l1 = fd_result["lambda1"]

                if math.isnan(jvp_l1):
                    n_nan_jvp += 1
                if math.isnan(fd_l1):
                    n_nan_fd += 1

                per_prompt_results.append({
                    "prompt_idx": p_idx,
                    "prompt": prompt[:80],
                    "n_tokens": input_ids.shape[1],
                    "tangent_seed": t_seed,
                    "cadence": cadence,
                    "jvp_lambda1": jvp_l1,
                    "fd_lambda1": fd_l1,
                    "abs_diff": abs(jvp_l1 - fd_l1) if not (math.isnan(jvp_l1) or math.isnan(fd_l1)) else None,
                    "jvp_nan": bool(jvp_result["nan_detected"]),
                    "fd_nan": bool(fd_result["nan_detected"]),
                    "jvp_inf": bool(jvp_result["inf_detected"]),
                    "fd_inf": bool(fd_result["inf_detected"]),
                })

                if not math.isnan(jvp_l1) and not math.isnan(fd_l1):
                    jvp_lambdas.append(jvp_l1)
                    fd_lambdas.append(fd_l1)

        # Progress summary per prompt
        prompt_jvp = [r["jvp_lambda1"] for r in per_prompt_results if r["prompt_idx"] == p_idx and not math.isnan(r["jvp_lambda1"])]
        prompt_fd = [r["fd_lambda1"] for r in per_prompt_results if r["prompt_idx"] == p_idx and not math.isnan(r["fd_lambda1"])]
        if prompt_jvp and prompt_fd:
            print(f"  JVP λ₁ range: [{min(prompt_jvp):.4f}, {max(prompt_jvp):.4f}]  "
                  f"FD λ₁ range: [{min(prompt_fd):.4f}, {max(prompt_fd):.4f}]")

    # Compute aggregate statistics
    print()
    n_valid = len(jvp_lambdas)
    n_total = len(per_prompt_results)
    print(f"Valid pairs: {n_valid}/{n_total}")
    print(f"NaN (JVP): {n_nan_jvp}  NaN (FD): {n_nan_fd}")

    if n_valid >= 3:
        r_val, p_val = pearsonr(jvp_lambdas, fd_lambdas)
        abs_diffs = [abs(j - f) for j, f in zip(jvp_lambdas, fd_lambdas)]
        mean_abs_diff = sum(abs_diffs) / len(abs_diffs)
        max_abs_diff = max(abs_diffs)
    else:
        r_val = float("nan")
        p_val = float("nan")
        mean_abs_diff = float("nan")
        max_abs_diff = float("nan")

    go_criterion = not math.isnan(r_val) and r_val >= 0.80

    print(f"Pearson r: {r_val:.6f} (p={p_val:.2e})")
    print(f"Mean |diff|: {mean_abs_diff:.8f}")
    print(f"Max  |diff|: {max_abs_diff:.8f}")
    print(f"GO criterion (r >= 0.80): {'GO' if go_criterion else 'NO-GO'}")

    elapsed = time.time() - t_start
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    report = {
        "phase": "phase0_sanity",
        "model": "gpt2",
        "dtype": "float64",
        "fd_epsilon": FD_EPSILON,
        "n_prompts": len(TEST_PROMPTS),
        "n_tangent_seeds": len(TANGENT_SEEDS),
        "renorm_cadences": RENORM_CADENCES,
        "n_total_configs": n_total,
        "n_valid_pairs": n_valid,
        "n_nan_jvp": n_nan_jvp,
        "n_nan_fd": n_nan_fd,
        "pearson_r": r_val,
        "pearson_p": p_val,
        "mean_absolute_difference": mean_abs_diff,
        "max_absolute_difference": max_abs_diff,
        "go_criterion": go_criterion,
        "go_criterion_threshold": 0.80,
        "per_prompt_results": per_prompt_results,
        "implementation_notes": [
            "flash attention lacks forward-mode AD; forced MATH SDPA backend via sdp_kernel",
            "float64 used for both JVP and FD to avoid float32 catastrophic cancellation in FD",
            f"FD epsilon {FD_EPSILON} optimal for float64 (sqrt(machine_eps) ~ 1e-8)",
            "production FTLE runs will use float32 JVP (no FD needed)",
        ],
        "compute_time_s": round(elapsed, 1),
        "timestamp_utc": timestamp,
    }

    report_path = output_dir / "sanity_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, cls=_NumpyEncoder)
    print(f"\nReport saved: {report_path}")
    print(f"Compute time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
