# Methods

## Collapse Detection Algorithm

Collapse is detected via an embedding-based algorithm operating on sequential assistant turns.

### Embedding
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Each assistant turn is embedded into a 384-dimensional vector

### Per-Turn Classification
For each turn t ≥ 3:

1. **Compute similarities:**
   - s1_t = cos(e_t, e_{t-1}) (lag-1 similarity)
   - s2_t = cos(e_t, e_{t-2}) (lag-2 similarity)

2. **Mark periodic:**
   - Period-1: s1_t ≥ 0.92
   - Period-2: s2_t ≥ 0.90

3. **Sustained periodicity:** Either period-1 or period-2 flag must hold for ≥3 consecutive turns

4. **Collapse criterion:** Sustained periodicity AND low drift:
   - d1 = 1 - s1 ≤ 0.08
   - d2 = 1 - s2 ≤ 0.10

### Locked Thresholds
| Parameter | Value | Source |
|---|---|---|
| S1_THRESHOLD | 0.92 | Preregistration |
| S2_THRESHOLD | 0.90 | Preregistration |
| D1_THRESHOLD | 0.08 | Preregistration |
| D2_THRESHOLD | 0.10 | Preregistration |
| SUSTAINED_WINDOW | 3 turns | Preregistration |

### Run-Level Metric
`collapse_rate` = (number of collapsed turns) / (total turns) for each 40-turn trajectory.

## Models

| Condition | Model | HuggingFace Revision |
|---|---|---|
| HOMO_A | Llama-3.1-8B-Instruct | `0e9e39f249a16976918f6564b8830bc894c89659` |
| HOMO_B | Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` |
| HOMO_C | Mistral-7B-Instruct-v0.3 | `c170c708c41dac9275d15a8fff4eca08d52bab71` |
| HETERO_ROT | Round-robin rotation | All three models above |

## Detector Reliability Outcome

The preregistered reliability criterion (Cohen's κ ≥ 0.80) was **not met**.

| Metric | Value |
|---|---|
| Cohen's κ | 0.566 |
| Raw agreement | 92.8% (167/180) |
| Valid pairs | 180/180 |
| Rater 1 | GPT-4o (OpenAI) |
| Rater 2 | Claude Sonnet 4 (Anthropic) |
| Rubric version | v3.0 (locked) |
| PABAK (supplementary) | 0.856 |

The low κ despite high raw agreement reflects extreme base-rate skew (~87% collapse prevalence). See `reliability_audit/kappa_gate_final_record.md` for full analysis.

**Consequence:** All collapse-pattern claims are descriptive and condition-comparative, not detector-validated.
