# Artifact Index — FTLE (FTLE / λ₁)

**Date:** 2026-02-20
**Status:** CLOSEOUT COMPLETE

Cross-links all freeze manifests and data artifacts for Phases 0–3.

## Phase 0 — Sanity Check (GPT-2 JVP validation)

| Artifact | Path | SHA256 |
|---|---|---|
| Phase 0 Freeze | `results/phase0/PHASE0_FREEZE.md` | `977b1dfb6a08994b5674870f587fe9b268b497a8fbd3a6c89b0a427f70bd33db` |
| Sanity report (f64) | `results/phase0/sanity_report.json` | `15ddecb8a56a2e502fd8ad160c31f88b49c0b0fe39e8fe8c9c34c0b21702907b` |
| Dtype transfer (f32) | `results/phase0/sanity_report_float32.json` | `78eb9293aa1714608748ce5de454a8f1e6ee58e1f935e8033d7253a9c1ec3d95` |
| 7B smoke test | `results/phase0/sanity_report_7b_smoke.json` | `9aaec74d5490cc0d0a69d2d9162657203785de7afa6838c701e89a4d2f149fb0` |

## Phase 1 — Pilot Canary (Llama-3.1-8B, 450 calls)

| Artifact | Path | SHA256 |
|---|---|---|
| Phase 1 Freeze | `results/phase1/PHASE1_FREEZE.md` | `75d436fa8482865d668d25188327e83c2a661cfaf7cf98a89a115b8a8861f0da` |
| Pilot results | `results/phase1/pilot_results.json` | `6835cac8` (prefix, see PHASE1_FREEZE.md) |
| Raw results | `results/phase1/raw_results.json` | `253e9c71` (prefix, see PHASE1_FREEZE.md) |

## Phase 2 — Full-Scale (720 trajectories × 10 tangent seeds = 7,200 calls)

| Artifact | Path | SHA256 |
|---|---|---|
| Phase 2 Freeze | `results/phase2/PHASE2_FREEZE.md` | `f223bddcf008425a34789c5b6aeaeb6a25beb0811402aea4214c9478a2b78f88` |
| FTLE bundle (720 rows) | `results/phase2/ftle_bundle.csv` | `753a62b3c9c814013dd0d6d1606d423653e726dd19c0a95477fecd2cade6f1a8` |
| Raw results | `results/phase2/raw_results.json` | `237c280bb44ce01066de3c8b8b93bb2c50cedf1b139215800cf12fb9e4ab9a97` |

## Phase 3 — Bridge Analysis (6 pre-registered Spearman correlations)

| Artifact | Path | SHA256 |
|---|---|---|
| Phase 3 Freeze | `results/phase3/PHASE3_FREEZE.md` | `81c8ba1feaa60d5df9ec87a8c2006b062e5b545262fd6e3506034b01f7181922` |
| Bridge results | `results/phase3/bridge_results.json` | `d563c0ddce9f2cc8822290a99f29ad5d16a0bcfc6292e03408ce174e7648ebcf` |
| Deviation table | `results/phase3/DEVIATION_TABLE.md` | see commit |

## Escape Velocity Input Artifacts (verified at Phase 3)

| Artifact | Path | SHA256 |
|---|---|---|
| Analysis bundle | `results/internal/paper-a/phase3_baseline/analysis_bundle.csv` | `8219ff8cce8ba45cb6c775d499ac3b927aae7bff03f0dc7edd4d554068623810` |
| Analysis freeze | `results/internal/paper-a/phase3_baseline/ANALYSIS_FREEZE.md` | `60fc4f9a25bd2518b5f554a220ac3ef8c98df88f35f6c5eb8fcb7bf9a59e69b1` |
