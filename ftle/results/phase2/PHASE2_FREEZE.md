# Phase 2 Freeze — Paper B (FTLE / λ₁)

**Date:** 2026-02-19
**Status:** COMPLETE
**Owner:** Ghost

## Frozen Artifacts

| Artifact | SHA256 |
|---|---|
| `results/phase2/raw_results.json` | `237c280bb44ce01066de3c8b8b93bb2c50cedf1b139215800cf12fb9e4ab9a97` |
| `results/phase2/ftle_bundle.csv` | `753a62b3c9c814013dd0d6d1606d423653e726dd19c0a95477fecd2cade6f1a8` |

## Protocol Summary

- **Trajectories:** 720 (4 conditions × 36 seeds × 5 repeats)
- **Tangent seeds per trajectory:** 10 (0-9)
- **Total FTLE calls:** 7,200
- **Renorm cadence:** 4 (locked from Phase 1)
- **GPU:** A100-80GB (Modal), max 2 containers per model (6 total)
- **Total valid:** 7,200 (0 failed, 0.0% attrition)
- **Wall time:** 1,402s (23.4 min)
- **Estimated cost:** ~$20 (well under $500 cap)

## Models

| Condition | Model | Revision | Calls |
|---|---|---|---|
| HOMO_A | meta-llama/Llama-3.1-8B-Instruct | `0e9e39f249a16976918f6564b8830bc894c89659` | 1,800 |
| HOMO_B | Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` | 1,800 |
| HOMO_C | mistralai/Mistral-7B-Instruct-v0.3 | `c170c708c41dac9275d15a8fff4eca08d52bab71` | 1,800 |
| HETERO_ROT | meta-llama/Llama-3.1-8B-Instruct | `0e9e39f249a16976918f6564b8830bc894c89659` | 1,800 |

## Attrition (per IMPLEMENTATION_SPEC threshold: ≤10%)

| Condition | Failed | Total | Attrition |
|---|---|---|---|
| HOMO_A | 0 | 1,800 | 0.0% |
| HOMO_B | 0 | 1,800 | 0.0% |
| HOMO_C | 0 | 1,800 | 0.0% |
| HETERO_ROT | 0 | 1,800 | 0.0% |

## Summary Statistics

| Condition | n | λ₁ mean | λ₁ std | Model |
|---|---|---|---|---|
| HOMO_A | 180 | 0.089217 | 0.008199 | Llama-3.1-8B |
| HOMO_B | 180 | 0.130927 | 0.007392 | Qwen2.5-7B |
| HOMO_C | 180 | 0.116811 | 0.022688 | Mistral-7B |
| HETERO_ROT | 180 | 0.089217 | 0.008199 | Llama-3.1-8B |
| **Overall** | **720** | **0.106543** | **0.022384** | |

λ₁ range: [0.073806, 0.170373]

## Notes

- HOMO_A and HETERO_ROT yield identical λ₁ distributions because both use Llama-3.1-8B-Instruct. FTLE is computed on the seed prompt (first user message), which is the same for both conditions. The conditions differ in multi-turn dynamics (Paper A), not in the first-turn forward pass.
- Within each (condition, seed_id), all 5 repeat_indices produce identical λ₁ values. This is expected and correct: FTLE depends only on (model, prompt, tangent_seed, cadence), which are invariant across repeats. The repeat_index matters for Paper A collapse metrics (which vary by repeat), not for FTLE.
- `ftle_bundle.csv` has 720 rows (one per trajectory) as required for Phase 3 bridge analysis.
- All computations used float32 (locked decision from INTERFACE.md).
- Zero NaN/Inf detections across all 7,200 tangent propagations.

## Throughput

| App | Calls | Wall Time | Rate |
|---|---|---|---|
| paper-b-ftle-llama | 3,600 | 1,402s | 2.6/s |
| paper-b-ftle-qwen | 1,800 | 517s | 3.5/s |
| paper-b-ftle-mistral | 1,800 | 517s | 3.5/s |

Average compute time per call: 0.45s
