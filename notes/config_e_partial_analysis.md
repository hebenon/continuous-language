# Config E Partial Analysis

**Status**: Partial — Config A-D results needed for primary comparison (P1)

## Observed Data

| Step | train_bpc | val_bpc | Notes |
|------|-----------|---------|-------|
| 200  | 3.5144    | 3.3143  | |
| 500  | 2.3922    | 2.6058  | |
| 700  | 2.1315    | 2.3198  | |
| 1000 | 1.8934    | 2.2354  | checkpoint saved |
| 1200 | 1.8892    | **2.1656** | ← best val |
| 1500 | 1.6336    | 2.2072  | |
| 1700 | 1.6724    | 2.2704  | |
| 2000 | 1.4861    | 2.3519  | checkpoint saved |
| 2200 | 1.2968    | —       | run hung, terminated |

## Key Statistics

- Best val BPC: **2.1656** at step 1200 (LR = 9.41e-04, still in warmup-decay)
- Train-val gap at best: 0.2764 BPC
- Train-val gap at step 2000: 0.8658 BPC (3× the gap at best)
- Val degradation from best to step 2000: +0.1863 BPC
- Training continued falling: 1.89 → 1.30 BPC from step 1200 to 2200 (−0.59)

## Observations

**Early peak, aggressive overfit**: Best val at step 1200 — only 24% through training. From step 1200, training loss kept dropping steeply while val rose. The model memorized TinyShakespeare rather than generalizing.

**Overfit rate**: Train-val gap tripled (0.2764 → 0.8658) in 800 steps. This is faster divergence than expected if coupling was organizing representations beneficially.

**Interpretation options**:
1. *HierarchicalWave has more parameters* (coupling mechanism = extra capacity), causing faster overfit on TinyShakespeare's ~1M char training set — true regardless of whether coupling helps
2. *Coupling is actively harmful*: sigma grew large (P5 scenario) making coupling effectively random state mixing, adding noise to representations
3. *Early peak is normal for this architecture*: TinyShakespeare may simply saturate fast for all wave configs — need D for comparison

**Cannot conclude without Config D:**
- Whether val=2.1656 is better, worse, or same as Config D best
- Whether Config D peaked earlier or later, and its overfit rate
- Whether the overfit pattern is coupling-specific or general to TinyShakespeare at this scale

**Cannot assess P3/P4/P5 (sigma predictions):**
Run terminated before normal completion; no sigma readout available. Would need to rerun Config E to step 1200 on Kaggle (~10 min) and extract sigma from checkpoint.

## What to do when Config A-D results arrive

1. Build comparison table: all five configs, best val BPC, step at best val, train-val gap at best
2. Assess P1: E vs D gap
3. Check Config D's overfit rate — is E faster?
4. If E ≈ D: neutral result confirmed, consistent with revised prior (70%)
5. If E < D (E is better): positive result — investigate sigma
6. If E > D (E is worse): coupling hurts — note overfitting as mechanism

## Sigma readout (deferred)

If needed: rerun Config E for exactly 1200 steps on Kaggle (free T4, ~10 min), extract checkpoint, read sigma value. Only worth doing if primary comparison (E vs D) shows an interesting result.
