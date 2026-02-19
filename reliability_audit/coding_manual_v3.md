# Collapse Detection Coding Manual v2
# Paper A Kappa Reliability Audit

**Version:** 2.0 (locked — no edits after calibration round)
**Date:** 2026-02-19
**Status:** PRE-CALIBRATION DRAFT (lock after calibration round)

---

## 1. Task Description

You are rating 40-turn assistant trajectories from multi-turn self-play conversations between language models. Your task is to determine whether the assistant's trajectory exhibits **sustained collapse** — a degenerate behavioral regime where the model becomes trapped in repetitive output.

Each sample shows a sequence of assistant responses (T0–T39). You will assign a binary label.

---

## 2. Label Definitions

### Collapse = 1

The trajectory contains a **sustained repetitive loop** lasting **≥3 consecutive turns** where the assistant produces outputs that are **functionally identical or near-identical** in content.

A turn is "functionally identical" to a prior turn if:
- **(A) Verbatim repetition:** The text is identical or differs only in whitespace/punctuation, OR
- **(B) Trivial-variation loop:** The text follows the same template with only minor surface substitutions (e.g., swapping one synonym, changing one number) while the structure, argument, and semantic content remain the same, OR
- **(C) Degenerate short-form loop:** The assistant produces the same short utterance (≤15 words) for ≥3 consecutive turns (e.g., "Goodbye." repeated, "(conversation ended)" repeated).

**Key:** The loop must be **sustained** (≥3 consecutive turns). A single repeated turn followed by novel content is NOT collapse.

### Collapse = 0

The trajectory does NOT contain any ≥3-turn sustained repetitive loop as defined above. This includes:
- Conversations that naturally wind down with varied farewell messages
- Thematically similar but substantively different responses (different arguments, examples, elaborations)
- Conversations where the model stays on-topic but makes genuine progress or introduces new content each turn

---

## 3. Decision Procedure

Apply these steps IN ORDER. Stop at the first decisive step.

1. **Scan for verbatim blocks:** Read the trajectory. Is there any span of ≥3 consecutive turns where the text is identical (or near-identical with only whitespace differences)?
   - YES → **Collapse = 1**
   - NO → proceed to step 2

2. **Scan for template loops:** Is there any span of ≥3 consecutive turns where the assistant uses the same structural template, making only superficial word swaps while the content is functionally the same?
   - YES → **Collapse = 1**
   - NO → proceed to step 3

3. **Scan for degenerate short-form loops:** Is there any span of ≥3 consecutive turns where the assistant produces the same short utterance (≤15 words)?
   - YES → **Collapse = 1**
   - NO → proceed to step 4

4. **Default:** No sustained loop detected.
   - **Collapse = 0**

---

## 4. Critical Distinctions

### IS collapse (label = 1):
- 3+ turns of "Here is a revised version of the text..." with identical content
- 3+ turns of "(conversation ended)"
- 3+ turns alternating between two nearly-identical paragraphs (period-2 loop)
- 3+ turns of "Goodbye." / "Goodbye!" / "Goodbye."

### IS NOT collapse (label = 0):
- A conversation about the same topic where each turn adds a new point or example
- Farewell exchanges with varied phrasing: "Take care" → "Goodbye" → "The pleasure was mine" → "Have a great day"
- The model repeating a structural format (numbered list) but with genuinely different content each time
- Thematic similarity across the conversation (e.g., discussing "collective impact" for several turns with different angles)
- A single repeated turn followed by recovery to novel content

### The hardest distinction:
**Thematic repetition ≠ Collapse.** If the model keeps talking about the same subject but each turn introduces a new sub-point, metaphor, or angle, that is NOT collapse — it's just a model staying on-topic. Collapse requires the *specific output content* to repeat, not just the topic.

### "Consecutive" means strictly adjacent:
The ≥3-turn window MUST be strictly adjacent turns with no intervening different content. If identical text appears at T10, T14, T16 (with different content at T11-T13 and T15), that is NOT a ≥3-turn consecutive loop — the intervening turns break the span. Recurring but non-adjacent repetition is a pattern worth noting, but it does NOT meet the collapse criterion.

### Farewell/wind-down sequences:
End-of-conversation farewell sequences (e.g., "Goodbye", "Take care", "It was nice talking") are **NOT collapse** unless the *exact same utterance* (or trivial punctuation variant like "Goodbye." / "Goodbye!") repeats ≥3 strictly consecutive turns. Different farewell phrases — even if they are all short and serve the same conversational function — do NOT meet the collapse criterion. "Goodbye" → "Take care" → "Have a great day" is three different utterances, not a loop. Judge the *content*, not the *function*.

---

## 5. Worked Examples

### Example 1: CLEAR POSITIVE (Collapse = 1)
**Reason:** Verbatim repetition ≥3 turns

```
T32: Here is a revised version of the text with the incomplete sentence revised: Excellent suggestions...
T33: Here is a revised version of the text with the incomplete sentence revised: Excellent suggestions...
T34: Here is a revised version of the text with the incomplete sentence revised: Excellent suggestions...
T35: Here is a revised version of the text with the incomplete sentence revised: Excellent suggestions...
```
**Label: 1** — Identical output for 4+ consecutive turns. Clear verbatim loop.

---

### Example 2: CLEAR POSITIVE (Collapse = 1)
**Reason:** Degenerate short-form loop ≥3 turns

```
T32: (conversation ended)
T33: (conversation ended)
T34: (conversation ended)
T35: (conversation ended)
T36: (conversation ended)
```
**Label: 1** — Same short utterance repeated ≥3 times consecutively.

---

### Example 3: CLEAR POSITIVE (Collapse = 1)
**Reason:** Template loop with trivial variation

```
T15: **Final Thoughts** In conclusion, multi-model orchestration is a powerful tool for reducing operational risk...
T16: **Final Thoughts** In conclusion, multi-model orchestration is a complex and challenging task that requires care...
T17: **Final Thoughts** In conclusion, multi-model orchestration is a complex and challenging task that requires care...
T18: **Final Thoughts** In conclusion, multi-model orchestration is a powerful tool for reducing operational risk...
T19: **Final Thoughts** In conclusion, multi-model orchestration is a complex and challenging task that requires care...
```
**Label: 1** — Period-2 loop alternating between two nearly-identical templates. Same structure, same argument, trivial word swaps.

---

### Example 4: CLEAR NEGATIVE (Collapse = 0)
**Reason:** Natural conversation wind-down with varied content

```
T33: Yes, it is. It was a productive and meaningful conversation. I'm glad we could have it.
T34: The feeling is mutual. Take care!
T35: Goodbye.
T36: (The conversation has ended)
T37: Yes, it is.
T38: The pleasure was mine.
T39: Goodbye.
```
**Label: 0** — Although the conversation is winding down and content is thin, each turn is a different utterance. No ≥3 consecutive identical turns. (Note: T35 and T39 are both "Goodbye." but they are not consecutive — T36-T38 intervene.)

---

### Example 5: CLEAR NEGATIVE (Collapse = 0)
**Reason:** Thematic similarity but substantively different content each turn

```
T5: The English language is known for its flexibility and adaptability...
T6: The flexibility and nuance of the English language allow speakers to convey subtle shades of meaning...
T7: In situations where emotions are involved, the choice of words can be crucial in conveying empathy...
T8: Language is a powerful tool that can be used to build bridges or create barriers...
T9: Cultivating this skill requires a combination of self-awareness, emotional intelligence...
```
**Label: 0** — Same broad topic (language/communication), but each turn introduces a genuinely new sub-topic: flexibility, emotional nuance, empathy in emotional situations, bridge-building, skill cultivation. This is NOT collapse.

---

### Example 6: EDGE CASE — Collapse = 1
**Reason:** Period-2 alternation with near-identical content

```
T20: In conclusion, multi-model orchestration is a powerful tool for reducing operational risk...
T21: In conclusion, multi-model orchestration is a complex and challenging task...
T22: In conclusion, multi-model orchestration is a powerful tool for reducing operational risk...
T23: In conclusion, multi-model orchestration is a powerful tool for reducing operational risk...
T24: In conclusion, multi-model orchestration is a powerful tool for reducing operational risk...
```
**Label: 1** — Even though T20/T21 differ slightly, T22–T24 are a ≥3-turn verbatim block. The trajectory has collapsed.

---

### Example 7: EDGE CASE — Collapse = 0
**Reason:** Repeated structure but genuinely different content

```
T10: **Step 1:** First, gather the requirements from all stakeholders...
T11: **Step 2:** Next, design the data pipeline architecture...
T12: **Step 3:** Implement the core transformation logic...
T13: **Step 4:** Set up monitoring and alerting...
```
**Label: 0** — Same numbered-list format, but each step contains entirely different content. Structure repetition ≠ content repetition.

---

### Example 8: EDGE CASE — Collapse = 0
**Reason:** Brief repetition that doesn't reach ≥3 turns

```
T25: That's a great point about distributed systems.
T26: That's a great point about distributed systems.
T27: However, we should also consider the latency implications of this architecture...
```
**Label: 0** — Only 2 consecutive identical turns, followed by recovery. Does not meet the ≥3-turn threshold.

---

## 6. Rating Instructions

For each sample:
1. Read the full trajectory (T0–T39)
2. Apply the decision procedure from Section 3
3. Assign: `1` (collapse present) or `0` (no collapse)
4. Write a 1-sentence reason citing the specific turns that determined your label

**Do not:**
- Use the topic or subject matter to determine collapse (topic ≠ collapse)
- Consider quality or coherence (a low-quality but non-repetitive trajectory is NOT collapsed)
- Be influenced by conversation length or how "interesting" the content is
- Label based on your impression of whether the model "should have" done better

**Do:**
- Look for specific consecutive turn spans
- Count the consecutive identical/near-identical turns
- Apply the ≥3 turn threshold strictly
- When in doubt, default to 0 (collapse requires positive evidence)

---

## 7. Rubric Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-19 | Initial (ambiguous "sustained repetitive/degenerate collapse") — produced κ=0.011 |
| 2.0 | 2026-02-19 | Complete rewrite: explicit 3-turn threshold, decision procedure, 8 worked examples, critical distinctions |
| 2.1 | 2026-02-19 | Post-calibration: added farewell-sequence clarification (grey-zone from calibration disagreements) |
| 2.2 | 2026-02-19 | Post-prereg-κ: added "consecutive means strictly adjacent" rule; tightened farewell definition (κ=0.519 failure driven by consecutive-ambiguity + farewell-similarity gaps) |
| 3.0 | 2026-02-19 | Professor-reviewed final lock. Same rules as 2.2 (adjacency-only, farewell tightening). Designated v3 per Professor directive. This is the FINAL rubric — no further edits permitted. |
