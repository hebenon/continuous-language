# Wave Dynamics Discretization for Recurrent LMs

**Status**: Architectural research
**Date**: 2026-02-05
**Goal**: Design stable second-order ODE discretization for minGRU-like architecture

## Problem Statement

Current CfC (Closed-form Continuous):
```
dh/dt = a_t ⊙ h_t + b_t
```

Each dimension has ONE parameter: τ (time constant).

**Hypothesis**: Second-order dynamics enable periodic pattern recognition.
```
d²h/dt² + α·dh/dt + ω²h = input(t)
```

Each dimension has TWO parameters: τ (decay) and ω (oscillation frequency).

## Discretization via Euler Method

Given ODE system:
```
dh/dt = v                        (velocity)
dv/dt = -α·v - ω²·h + input(t)  (acceleration = damping + restoring force + drive)
```

Euler forward step (step size Δt, typically Δt=1 for token-per-step):
```
h_{t+1} = h_t + Δt · v_t
v_{t+1} = v_t + Δt · (-α·v_t - ω²·h_t + input(t))
        = (1 - α·Δt)·v_t - ω²·Δt·h_t + Δt·input(t)
```

With Δt=1:
```
h_{t+1} = h_t + v_t
v_{t+1} = (1 - α)·v_t - ω²·h_t + input(t)
```

**Problem**: This is unstable if |eigenvalues| > 1. 

The system matrix is:
```
[h_{t+1}]   [    1        1   ] [h_t]       [0]
[v_{t+1}] = [-ω²   (1-α) ] [v_t]  +  [input(t)]
```

Eigenvalues: λ = (1-α)/2 ± √((1-α)²/4 - ω²)

For stability: |λ| < 1, which requires:
- α > 0 (damping)
- ω² must satisfy certain bounds relative to α

## Bilinear Transform (Better Stability)

Map continuous pole to discrete pole via:
```
s = 2(z-1)/(z+1) / Δt
```

Continuous pole at s = -α/2 ± iω maps to discrete pole.

For underdamped oscillator (α < 2ω):
```
Magnitude: |z| = √((α² + 4ω²) / (α² + 4ω²)) = 1  ← MARGINAL STABILITY!
```

Critically, bilinear transform **preserves pole magnitude**. An underdamped oscillator stays on the unit circle (neutral stability, neither grows nor decays).

## Practical Choice: Modified Euler with Decay Gating

Rather than pure Euler or bilinear, use **gated state update**:

```python
# Per dimension, per timestep:
# State: [h, v] (position and velocity)

# Compute acceleration
accel = -alpha * v - omega_sq * h + input_t

# Velocity update with decay gate
v_gate = exp(-tau)  # decay from time constant τ
v_new = v_gate * v + accel

# Position update (standard)
h_new = h + v_new

# Optional: clip to prevent explosion
h_new = clip(h_new, -1, 1)
```

**Why this works**:
1. The `exp(-tau)` term (from CfC framework) decays velocity naturally
2. The oscillatory dynamics emerge from (-ω²·h) term
3. Two timescales: τ controls overall decay, ω controls oscillation frequency
4. Gating prevents runaway growth like oHC had

## Test Case: Repeating-Sum Task

**Input sequence**: [1, 2, 1, 2, 1, 2, 1, 2, ...]
**Output**: Cumulative sum, but **reset every 3 steps**
```
Step 0: input=1, output=1, reset_counter=0
Step 1: input=2, output=3, reset_counter=1
Step 2: input=1, output=4, reset_counter=2
Step 3: [RESET] output=0, input=1, output=1, reset_counter=0
Step 4: input=2, output=3, reset_counter=1
...
```

**Why this tests wave dynamics**:
- Fast oscillation needed to track the pattern repeat (every 3 steps)
- Slow decay needed to accumulate sum within each cycle
- Pure gating (constant τ) struggles because no frequency detection
- Wave dynamics can dedicate fast dimensions to cycle tracking, slow dimensions to accumulation

**Metric**: Accuracy on steps 100-150 (extrapolation beyond 50-step training).

## Hyperparameter Constraints

For stability and learning:

| Parameter | Range | Why |
|-----------|-------|-----|
| τ | [0.1, 10] | Decay rate; 0.1 = fast, 10 = slow |
| ω | [0.1, 2π] | Oscillation frequency; avoid > π/Δt (Nyquist) |
| α (damping) | [0.5, 2.0] | Controls oscillation decay rate |

**Initialization**:
- τ: uniform [0.1, 10] (same as minGRU)
- ω: uniform [0.5, 2] (avoid extremes initially)
- α: learnable but initialized to 1.0 (critical damping)

## Expected Failure Modes (from oHC experience)

1. **Gradient explosion**: If ω too large, h can spiral. Mitigate: clip h, or use tanh nonlinearity.
2. **Learning failure**: If τ and ω are independent, gradients may not flow. Mitigate: regularize ω updates (small learning rate for frequency parameters).
3. **Instability at high depth**: If stacking 72L with oscillatory layers, resonance could amplify. Mitigate: test on shallow network first (16L).

## Implementation Roadmap

### Phase 1: Synthetic Test (No executor needed)
```python
# wave_dynamics.py
class WaveDimension:
    def __init__(self, tau, omega, alpha):
        self.h = 0.0
        self.v = 0.0
        self.tau = tau
        self.omega = omega
        self.alpha = alpha
    
    def step(self, input_val):
        # Gated velocity update
        v_gate = exp(-self.tau)
        accel = -self.alpha * self.v - (self.omega ** 2) * self.h + input_val
        self.v = v_gate * self.v + accel
        self.h = self.h + self.v
        return clip(self.h, -1, 1)

# Test on repeating-sum task
# Measure: accuracy over 50-100 steps vs minGRU baseline
```

### Phase 2: JAX Integration (With executor)
- Implement WaveDynamicsLayer for Fathom
- Add to test suite
- Train on repeat task, measure convergence

### Phase 3: Scale Test (On Lambda if budget available)
- Add to Fathom architecture
- Test on WikiText (does periodic structure help language modeling?)
- Compare to baseline Gated-EFLA

## Open Questions

1. **Can oscillation emerge from random init?** Or does ω need specific seeding?
2. **Does coupling between (τ, ω) dimensions help?** (Like how spatial coupling failed in reaction-diffusion)
3. **Is bilinear transform better than gated Euler?** Stability vs. simplicity tradeoff.
4. **Does this generalize to natural language?** Or only synthetic repeating patterns?

## References

- CfC paper (Hasani et al., 2022): dh/dt = a_t ⊙ h + b_t formulation
- ODE discretization: Dormand-Prince, bilinear transform
- Damped oscillator physics: standard mechanics

---

**Next**: Implement Phase 1 synthetic test. Can write pure Python simulation without ML framework.
