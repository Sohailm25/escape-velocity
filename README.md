# Conversational Collapse in Multi-Turn LLM Self-Play

## Overview

This repository contains the data, code, and reproducibility artifacts for two companion studies of conversational collapse in multi-turn self-play between 7B-parameter language models.

**Escape Velocity — Condition-Dependent Collapse Dynamics** establishes that collapse dynamics are condition-dependent across 720 trajectories (4 conditions × 36 seeds × 5 repeats).

**FTLE — Depth-Dynamics Signatures** (`ftle/`) estimates the top-1 finite-time Lyapunov exponent (λ₁) for transformer depth dynamics and tests whether depth-dynamics summaries are associated with conversational collapse.

**Key finding:** Collapse dynamics are condition-dependent. Qwen2.5-7B homogeneous self-play exhibits the highest collapse rate (mean 0.773), while Mistral-7B shows the lowest (0.141). Heterogeneous model rotation produces intermediate collapse (0.250).

## Scope and Limitations

This study followed a preregistered confirmatory design (Path B outcome):

✅ **What is established:**
- Complete baseline execution: 720/720 unique trajectories (4 conditions × 36 seeds × 5 repeats)
- Protocol integrity: all runs completed 40 turns with no early stopping
- Condition-dependent collapse patterns are reproducible across seeds and repeats
- Baseline collapse incidence ≥60% in at least one condition (HOMO_B: 0.773)

⚠️ **What is limited:**
- The preregistered detector reliability criterion (Cohen's κ ≥ 0.80) was **not met** (κ = 0.566; raw agreement 92.8%)
- All collapse-pattern findings are **descriptive and condition-comparative**, not detector-validated
- No intervention efficacy claims (baseline study only)

See [METHODS.md](METHODS.md) for full detector specification and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for artifact verification.

## Experimental Design

| Parameter | Value |
|---|---|
| Conditions | HOMO_A (Llama-3.1-8B), HOMO_B (Qwen2.5-7B), HOMO_C (Mistral-7B), HETERO_ROT (rotation) |
| Seeds | 36 (SEEDS_V2, 6 prompt buckets × 6 seeds) |
| Repeats | 5 per seed per condition |
| Total trajectories | 720 |
| Turns per trajectory | 40 (locked, no early stopping) |
| Generation | temp=0.7, top_p=0.95, max_tokens=256 |

## Links

**Escape Velocity:**
- Distill page: https://sohailmo.ai/research/escape-velocity/
- PDF: https://sohailmo.ai/papers/escape-velocity-2026.pdf

**FTLE:**
- Distill page: https://sohailmo.ai/research/ftle/
- PDF: https://sohailmo.ai/papers/ftle-2026.pdf

**Code/data:** https://github.com/Sohailm25/escape-velocity

## Quick Start

```bash
# View condition-level results
cat results/condition_summary.csv

# Regenerate themed figures from frozen data
pip install matplotlib pandas numpy
python scripts/generate_figures_themed.py

# Verify data integrity
python scripts/verify_hashes.py
```

## Repository Structure

```
├── README.md                    # This file
├── ftle/                        # FTLE: depth-dynamics analysis
│   ├── README.md                # FTLE overview
│   ├── PREREG.md                # Pre-registration (v2)
│   ├── RESULTS.md               # Final results summary
│   ├── LIMITATIONS_AND_CAVEATS.md
│   ├── ARTIFACT_INDEX.md
│   ├── arxiv/                   # LaTeX manuscript + PDF
│   ├── figures/                 # Themed figures + generation script
│   ├── results/                 # Frozen Phase 2 + Phase 3 data
│   └── scripts/                 # FTLE computation + analysis pipeline
├── METHODS.md                   # Detector specification + reliability outcome
├── REPRODUCIBILITY.md           # Artifact hashes + verification
├── DATA_DICTIONARY.md           # Field definitions for all data files
├── STATUS.md                    # Summary of what passed/failed
├── LICENSE                      # MIT License
├── CITATION.cff                 # Citation metadata
├── results/
│   ├── analysis_bundle.csv      # Per-run collapse metrics (720 rows)
│   ├── condition_summary.csv    # Condition-level statistics
│   └── kappa_metrics.json       # Detector reliability audit results
├── scripts/
│   ├── generate_figures_themed.py   # Figure generation from frozen data
│   ├── verify_hashes.py            # Artifact integrity verification
│   └── theme.py                    # Matplotlib theme (sohail_research style)
├── figures/
│   ├── figure_1_mean_collapse_by_condition.png
│   ├── figure_2_collapse_distribution.png
│   └── figure_3_first_collapse_turn.png
├── paper/
│   ├── sohail_research.mplstyle    # Matplotlib style file
│   └── main.tex                    # LaTeX manuscript source
└── reliability_audit/
    ├── coding_manual_v3.md         # Locked rater rubric
    └── kappa_gate_final_record.md  # Gate outcome documentation
```

## Citation

```bibtex
@article{mohammad2026collapse,
  title={Condition-Dependent Collapse Dynamics in Multi-Turn LLM Self-Play},
  author={Mohammad, Sohail},
  year={2026}
}

@article{mohammad2026ftle,
  title={Depth-Dynamics Signatures of Conversational Collapse: Finite-Time Lyapunov Analysis of Transformer Forward Passes},
  author={Mohammad, Sohail},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
