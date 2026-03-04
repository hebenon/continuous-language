# Lambda Run Guide — GatedWave Char-LM Experiment

*Written pre-run. Use this to interpret results quickly when they arrive.*

---

## What We're Testing

Three questions, in priority order:

1. **Does GatedWave outperform minGRU on character-level language modeling?**
   This is the fundamental architecture comparison. Both bugs are now fixed (scale 0
   integrator, input gain asymmetry). All prior results used broken architectures.
   This is the first clean test.

2. **Does language-tuned theta initialization improve over the default?**
   H9 was confirmed in the narrow form (test_h9_resonance.py, 2026-03-04): theta
   matching dramatically helps for periodic signal detection. Broad form hypothesis:
   tuning theta to English text periodicities (word ~5 chars, sentence ~100 chars)
   should give the oscillatory scales a better prior than the default linspace.

3. **Does GatedWave scale?** (from Pi smoke tests to A100 full runs)
   Pi results suggest gradient flow is healthy. Do the dynamics stay stable at larger
   d and L? Do training curves look normal (smooth, no instability)?

---

## Experimental Configs

### Small (stability check, ~30 min on A100)

```bash
# Quick check — verify no instability, reasonable loss curve
python train_char_lm.py --model gated_wave --d_model 512 --n_layers 6 \
    --n_scales 4 --steps 5000 --eval_every 500

# Expected: loss should decline smoothly from ~4.2 (random) toward ~2.0
# within 5k steps. If loss explodes or stays flat → architectural problem.
```

### Main comparison (~4-8 hours on A100 per run)

```bash
# Condition A: GatedWave default theta
python train_char_lm.py --model gated_wave --d_model 1024 --n_layers 12 \
    --n_scales 4 --steps 50000 --wandb --wandb_project gated-wave-char-lm

# Condition B: GatedWave language-tuned theta
python train_char_lm.py --model gated_wave --d_model 1024 --n_layers 12 \
    --n_scales 4 --theta_min 0.063 --theta_max 1.257 --steps 50000 \
    --wandb --wandb_project gated-wave-char-lm

# Condition C: minGRU baseline (parameter-matched)
python train_char_lm.py --model mingru --d_model 1800 --n_layers 12 \
    --steps 50000 --wandb --wandb_project gated-wave-char-lm
```

### Small-scale versions (parallel stability check of all three configs)

```bash
python train_char_lm.py --model gated_wave --d_model 512 --n_layers 6 \
    --n_scales 4 --steps 10000 --wandb
python train_char_lm.py --model gated_wave --d_model 512 --n_layers 6 \
    --n_scales 4 --theta_min 0.063 --theta_max 1.257 --steps 10000 --wandb
python train_char_lm.py --model mingru --d_model 900 --n_layers 6 \
    --steps 10000 --wandb
```

---

## Parameter Counts

| Config | d | L | S | ~Params |
|--------|---|---|---|---------|
| GatedWave small | 512 | 6 | 4 | 10.2M |
| GatedWave large | 1024 | 12 | 4 | 79.8M |
| MinGRU small | 900 | 6 | — | 10.7M |
| MinGRU large | 1800 | 12 | — | 81.3M |

---

## Expected BPC Reference Points

| Comparison point | Expected BPC |
|-----------------|-------------|
| Uniform random (1/65 each token) | 6.02 |
| Baseline without sequence model (unigram) | ~4.5 |
| Reasonable small model (~10M, 50k steps) | ~1.5–1.6 |
| Reasonable large model (~80M, 50k steps) | ~1.35–1.45 |
| State-of-art char-LM (large, well-tuned) | ~1.2 |

TinyShakespeare is a small corpus (1.1M chars). At 80M params, there's real overfitting
risk. Watch train vs val BPC gap — if train << val, we're in overfit territory and the
comparison isn't valid.

---

## How to Interpret Results

### Pattern 1: Wave (A) ≈ Wave (B) > MinGRU (C)
- GatedWave has genuine advantage; language-tuned theta doesn't help extra
- Interpretation: Architecture matters; frequency initialization doesn't (the model
  learns its own effective frequency selectivity). Core hypothesis supported.

### Pattern 2: Wave (B) > Wave (A) > MinGRU (C)
- Best case: both architecture AND theta initialization matter
- Strong H9 broad form confirmation. Language-tuned theta gives a better prior.
- Interpretation: Theta diversity helps; matching to text frequencies helps further.

### Pattern 3: MinGRU (C) ≥ Wave (A) ≈ Wave (B)
- Architecture doesn't help; wave dynamics add nothing over minGRU
- Interpretation: Need to rethink. Possible causes:
  - At char-LM scale, the periodic structure isn't the bottleneck
  - Gating mechanism is too rigid (the gate overrides the oscillatory dynamics)
  - The optimization is harder for GatedWave at this scale (complex states, more params)
- Action: examine training curves in detail. Is GatedWave converging at all?

### Pattern 4: Wave (A or B) diverges / loss doesn't decrease
- Architectural instability — complex states exploding, gradient issues
- Check: are oscillatory scale states blowing up? Is LayerNorm holding them?
- Action: lower lr, check grad norms in wandb

### Pattern 5: Wave (A) > MinGRU but Wave (B) < Wave (A)
- Language-tuned theta hurts — the theta range 0.063→1.257 is wrong for char-LM
- Interpretation: English text periodicities at character scale don't match these
  values, or the model needs the slow oscillators (theta<0.063) for something.
- Action: try theta_min=0.01, theta_max=1.257 (include slow scale, extend fast end)

---

## What to Watch in W&B

1. **Train loss and val loss curves** — looking for smooth decline, not divergence
2. **Train vs val BPC gap** — overfitting indicator (should stay < 0.3 for small, < 0.5 for large)
3. **Gradient norm** — should stay bounded (clip is 1.0); if regularly hitting clip → lr too high
4. **Val BPC at convergence** (step 40k-50k) — primary comparison metric
5. **Early BPC at step 5k** — convergence speed (does wave learn faster early?)

## Post-Run Architecture Analysis (GatedWave Only)

If GatedWave trains successfully, examine gate behavior per scale. This requires a small
diagnostic script (not in train_char_lm.py) that loads a checkpoint and runs inference
on a sample, logging gate activation values per scale.

**What to look for: per-scale gate sparsity**

Each GatedWave scale has fully independent gate parameters — scale 0 can fire frequently
while scales 1-3 fire rarely. A well-functioning architecture should show differential
gate behavior:
- Scale 0 (integrator, r=1.0): relatively frequent gate firing — resets at segment boundaries
- Scales 1-3 (oscillatory, r<1): sparse gate firing — accumulate across boundaries, preserve
  oscillatory phase between gates

**Why this matters**: If all scales have similar gate firing rates, the multi-scale
differentiation isn't being learned. The oscillatory scales need to stay "alive" (rarely
reset) to accumulate periodic signals coherently — the H9 mechanism requires low gate
frequency on oscillatory scales.

**If gate sparsity is wrong** (oscillatory scales as frequently reset as integrator):
- The architecture isn't using its multi-scale structure
- Consider adding a sparsity regularization loss on oscillatory gates
- Or: decouple gate conditioning — integrator gate conditioned on x, oscillatory gates
  conditioned on frequency-filtered version of prev_h

**Quick diagnostic** (write post-run):
```python
# After loading params from checkpoint, run on ~1000 chars of val data
# Log: mean(gate) per scale per layer → expect scale 0 > scales 1-3
```

---

## Architecture Status (Pre-Run)

Both bugs fixed as of 2026-03-03:
1. **Scale 0 integrator dead** — `b * (1-r)` zeroed input for r=1.0. Fixed: uniform 1/sqrt(d).
2. **Input gain asymmetry** — (1-r) normalization created 250:1 gain ratio; oscillatory scales
   starved. Fixed: uniform 1/sqrt(d) for all scales. State boundedness comes from r<1 in `a`.

All prior Pi experiments used the broken architecture. This Lambda run is the first valid test.

H9 narrow form confirmed (2026-03-04): theta-matched oscillator reaches 87.5% accuracy on
signal detection task vs 58-64% for all other configs. Mechanism validated. Broad form (char-LM)
is what this run will tell us.

---

## Sequence of Operations

1. Provision A100 instance on Lambda Labs (see `memory/workflows/lambda_gh200.md` for procedure)
2. Clone repo, install deps: `pip install jax[cuda12] flax optax wandb`
3. Download data: `mkdir -p data/shakespeare && curl -O ... data/shakespeare/input.txt`
4. Run quick stability check first (5k steps, ~30 min)
5. If stable, launch all three main configs (can run in parallel if budget allows)
6. Monitor via W&B; check in at step ~5k and ~20k
7. Terminate early if any config diverges
8. After run, compare final val BPC across all three. Record in L2.

---

## Decision Tree Post-Results

- **Wave wins** → proceed with GatedWave as primary arch; design hierarchical coupling
  experiments (P1 item 3 from backlog)
- **Wave loses** → investigate why (gating mechanism? optimization difficulty?).
  Consider: removing gate (pure resonance, no reset), or simplifying to 2 scales
- **Language-tuned wins** → add theta presets as standard config; investigate
  whether higher theta_max (faster oscillators) or period-matched initialization helps further
- **Overfitting** → reduce model size or add dropout; comparison is still valid if
  both models overfit similarly

---

*Last updated: 2026-03-04. Architecture: both bugs fixed. Three configs ready.*
