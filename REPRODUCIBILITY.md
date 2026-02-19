# Reproducibility

## Artifact Verification

All key artifacts can be verified against SHA256 hashes computed at analysis freeze time.

### How to verify
```bash
python scripts/verify_hashes.py
```

### Key Hashes

| Artifact | SHA256 |
|---|---|
| analysis_bundle.csv | Computed at verification time — see verify_hashes.py |
| condition_summary.csv | Computed at verification time |
| kappa_metrics.json | Computed at verification time |

### Frozen Dataset
The analysis was frozen on 2026-02-19. All artifacts in this repository are derived from the frozen dataset. No re-runs were performed after the freeze.

## Environment

| Component | Version/ID |
|---|---|
| Compute | Modal (A100-80GB) |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Seed policy | SEEDS_V2 (k=36) |
| Seed hash | 092d7b16aa4e4d7891a92d16a76a60243dede4bd7e6d76d7ac62bb242d4118c7 |

## Reproducing Figures

```bash
pip install matplotlib pandas numpy
python scripts/generate_figures_themed.py
```

Figures are generated from `results/analysis_bundle.csv` using the locked theme in `scripts/theme.py`.
