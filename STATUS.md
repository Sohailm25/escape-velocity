# Status

## Paper A — Condition-Dependent Collapse Dynamics in Multi-Turn LLM Self-Play

**Submission status:** Path B (transparent downgrade)

### What passed
- ✅ Baseline execution: 720/720 unique successful tuples
- ✅ Protocol integrity: all 40 turns, no early stopping, across all runs
- ✅ Collapse incidence criterion: HOMO_B mean 0.773 ≥ 0.60 threshold
- ✅ Complete analysis freeze with SHA256 verification

### What failed
- ❌ Detector reliability: κ = 0.566 < 0.80 prereg threshold

### Claim scope
All collapse-pattern findings are descriptive and condition-comparative. The preregistered detector reliability criterion was not met (κ = 0.566); conclusions are therefore limited to baseline execution and descriptive condition-level patterns.
