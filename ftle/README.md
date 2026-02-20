# Paper B (Phase 2): FTLE / Top-1 Lyapunov Exponent (λ₁) for Transformers

**Umbrella:** compute-full-lyapunov-spectrum-production-transformer

## Scope
Develop a numerically stable, reproducible procedure to estimate **FTLE / top-1 Lyapunov exponent (λ₁)** for a transformer map (depth-as-time), then test predictive association with Phase 1 (Paper A) escape-velocity measurements.

Important scope guard:
- Paper B does **not** claim time-axis equivalence between depth dynamics and across-turn conversational dynamics.
- Any causal/mechanistic identity claim is out of scope for this phase.

## Not in scope
- Escape benchmark as the primary claim (that is Paper A)
- Full Lyapunov spectrum at production scale as the initial milestone (downscope to λ₁ first)

## Protocol
Paper B inherits all shared protocol definitions from:
- `../INTERFACE.md`

## Gate
Paper B proceeds only after Paper A establishes a stable collapse regime and measurable escape curves.
