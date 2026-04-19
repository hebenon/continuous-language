# Continuous-Time Language Modelling

Research into whether **multi-scale temporal dynamics** can improve sequence modelling — specifically, whether oscillatory or continuous-time recurrent architectures with diverse timescales outperform simpler alternatives on tasks requiring memory across varying ranges.

The core question: do learnable time constants (τ) in continuous-time recurrent networks actually learn to specialise across timescales, and can architectural coupling between those scales yield practical gains?

---

## Architecture Lineage

### 1. Continuous-Time LM (`continuous_lm.py`)
ODE-based sequence model using `diffrax` + `equinox`. Implements `ContinuousDynamics` with learnable log-τ per dimension. Initial experiments on bracket matching and variable-delay copy tasks — τ diversity alone proved insufficient without a routing or coupling mechanism.

### 2. minGRU Multi-Scale (`train_minGRU.py`)
Lightweight multi-scale gated recurrence. The first architecture to show genuine benefit from scale diversity: **98% accuracy on Path-32** (1024-token long-range dependency task) vs ~75% for single-scale baseline. Path-X (16384 tokens) remained intractable — linear recurrence hits an expressivity wall at that scale.

### 3. Reaction-Diffusion (`reaction_diffusion.py`, `train_reaction_diffusion.py`)
Spatial coupling via reaction-diffusion dynamics across scale dimensions. Negative result: diffusion is symmetric; causal sequence modelling requires asymmetric coupling that respects token order. Peak 80% → collapsed to ~50%.

### 4. Gated Wave Dynamics (`gated_wave.py`, `train_gated_wave.py`)
Oscillatory hidden states (complex-valued, parameterised by radius r and frequency θ) with gated input drive and non-linear conditioning. Positive result on repeating-sum task (loss 0.03–0.07). The missing ingredient from earlier architectures: non-linear coupling between input and oscillatory state.

### 5. Hierarchical Wave Model (`hierarchical_wave.py`) — current
Adds **soft hierarchical coupling** on top of gated wave dynamics. Dimensions with similar oscillatory frequencies (θ) mix their hidden states via a learnable Gaussian kernel:

```
coupling(i, j) = exp(-|θ_i - θ_j| / σ)
```

Row-normalised so each scale receives a weighted average of nearby-frequency states. Implemented as `SoftHierarchicalCoupling` + `HierarchicalWaveLayer`, drop-in compatible with the gated wave interface. Training on TinyShakespeare (character-level) ongoing.

---

## Experimental Results

| Experiment | Task | Result |
|---|---|---|
| Learnable τ vs frozen τ | Bracket matching | NULL — no difference (93% vs 95%) |
| τ diversity | Variable-delay copy | NEGATIVE — frozen τ wins decisively |
| Multi-scale minGRU | Path-32 (1024 tokens) | **SUCCESS — 98% vs 75% baseline** |
| Multi-scale minGRU | Path-X (16384 tokens) | NEGATIVE — all configs ~50% (chance) |
| Reaction-diffusion coupling | Path-32 | NEGATIVE — symmetric diffusion breaks causality |
| Gated wave dynamics | Repeating sum | SUCCESS — loss 0.03–0.07 |
| Hierarchical wave | TinyShakespeare | In progress |

**Key insight**: τ diversity alone does not produce multi-scale behaviour — the dynamics network compensates, making τ redundant. *Coupling between dimensions is the missing ingredient.* This is what the gated wave and hierarchical architectures address.

---

## Hypothesis Tests (`test_h*.py`)

Numbered hypothesis tests, each isolating a specific architectural question:

- `test_h9_resonance.py` — resonance behaviour in wave dynamics
- `test_h10_hold.py` — information holding across time steps
- `test_h11_logtheta.py` — log-space θ parameterisation
- `test_h12_dual_frequency.py` — dual-frequency oscillatory interaction
- `test_h13_amplitude_modulation.py` — amplitude modulation detection
- `test_h13v2_fixed.py` — confound-fixed replication of H13
- `test_h15_prediction.py` — prediction framework for Config E
- `test_h16_seq_length.py` — sequence length scaling

---

## File Reference

| File | Purpose |
|---|---|
| `continuous_lm.py` | ODE-based continuous-time LM (Marimo notebook) |
| `gated_wave.py` | Gated oscillatory dynamics layer |
| `hierarchical_wave.py` | Soft hierarchical coupling + HierarchicalWaveModel |
| `reaction_diffusion.py` | Reaction-diffusion coupling (negative result) |
| `train_char_wave.py` | Character-level training on wave model |
| `train_hierarchical_char.py` | Character-level training on hierarchical model |
| `train_minGRU.py` | minGRU multi-scale training |
| `train_pathx_jax.py` | Path-X evaluation |
| `generate_hierarchical.py` | Text generation from hierarchical model |
| `pathway_validation.py` | Pathfinder task validation utilities |
| `gate_sparsity_diagnostic.py` | Gate activation diagnostics |
| `hierarchical_loss.json` | Saved loss curves |

---

## Setup

Requires JAX with GPU/TPU support. Training runs are designed for cloud GPUs (Lambda Labs, Colab).

```bash
pip install jax[cuda12] flax optax equinox diffrax marimo
```

For CPU-only exploration:

```bash
pip install jax flax optax equinox diffrax
```

---

## Status

Active research. Architecture is converging on the hierarchical wave approach; current work is character-level language modelling to establish whether soft coupling generalises beyond synthetic tasks.
