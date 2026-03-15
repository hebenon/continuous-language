"""
H13: Amplitude Modulation Detection

Tests whether state-side coupling (HierarchicalWave) adds benefit beyond
input-side coupling (GatedWave) for genuinely hierarchical cross-scale structure.

H12 confirmed: GatedWave simultaneously detects two independent signals (+38pp vs minGRU).
H2 (revised, conf=55%): Input-side coupling is sufficient; state-side not required.
H13 asks: Does a genuinely HIERARCHICAL signal — where one scale modulates another —
expose a limitation of input-side-only coupling?

Task: 3-class classification
  Class 0: pure noise
  Class 1: noise + A*sin(2π*t/T_fast)                              [fast signal, flat amplitude]
  Class 2: noise + A*(1 + m*sin(2π*t/T_slow))*sin(2π*t/T_fast)   [amplitude-modulated]

Discriminating Class 2 from Class 1 requires detecting that the amplitude of the fast
oscillation varies at the slow period — a genuinely cross-scale hierarchical relationship.

Models:
  GatedWave:        input-side coupling (gate on cross-layer prev_h), no within-layer mixing
  HierarchicalWave: GatedWave + SoftHierarchicalCoupling (within-layer state mixing by timescale)
  minGRU:           baseline, no multi-scale structure

H2 prediction: GatedWave ≈ HierarchicalWave (input-side sufficient)
H2 alternative: HierarchicalWave > GatedWave on Class 2 (state-side coupling needed)

T_fast=7, T_slow=31 (coprime), L=128.
Pi config: d=32, n_layers=2, n_scales=4, 2000 steps (~5-8 min).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from gated_wave import GatedWaveModel
from hierarchical_wave import HierarchicalWaveLayer
from train_minGRU import MinGRU as MinGRUModel


# ── Task ────────────────────────────────────────────────────────────────────

def generate_am_batch(
    rng: np.random.Generator,
    batch_size: int,
    seq_len: int,
    T_fast: int = 7,
    T_slow: int = 31,
    amplitude: float = 0.5,
    modulation_depth: float = 0.6,
    noise_std: float = 1.0,
) -> tuple:
    """
    Returns (inputs, labels):
      inputs: (batch, seq_len, 1) float32
      labels: (batch,)            int32 in {0, 1, 2}

    Class 0: noise
    Class 1: A*sin(2π*t/T_fast) + noise          [constant amplitude]
    Class 2: A*(1+m*sin(2π*t/T_slow))*sin(2π*t/T_fast) + noise  [AM signal]
    """
    t = np.arange(seq_len, dtype=np.float32)
    fast_signal = amplitude * np.sin(2 * np.pi * t / T_fast)
    slow_envelope = modulation_depth * np.sin(2 * np.pi * t / T_slow)
    am_signal = amplitude * (1.0 + slow_envelope) * np.sin(2 * np.pi * t / T_fast)

    labels = rng.integers(0, 3, size=batch_size)
    noise = rng.normal(0, noise_std, size=(batch_size, seq_len)).astype(np.float32)

    inputs = noise.copy()
    inputs[labels == 1] += fast_signal
    inputs[labels == 2] += am_signal

    return inputs[:, :, None], labels.astype(np.int32)


# ── HierarchicalWave model (GatedWave + state-side coupling) ─────────────────

class HierarchicalWaveSeqModel(nn.Module):
    """GatedWave + SoftHierarchicalCoupling within each layer."""
    d_model: int = 32
    n_layers: int = 2
    n_scales: int = 4
    r_min: float = 0.9
    r_max: float = 0.999

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.d_model, name="input_proj")(x)
        prev_h = jnp.concatenate([x, jnp.zeros_like(x)], axis=-1)

        for i in range(self.n_layers):
            residual = x
            x = nn.LayerNorm(name=f"ln_{i}")(x)

            # HierarchicalWaveLayer: GatedWaveDynamicsLayer + SoftHierarchicalCoupling
            # Output: (batch, seq_len, d_model * 2)
            h_complex = HierarchicalWaveLayer(
                d_model=self.d_model,
                n_scales=self.n_scales,
                r_min=self.r_min,
                r_max=self.r_max,
                name=f"layer_{i}",
            )(x, conditioning=prev_h)

            prev_h = h_complex
            # Project back to d_model for residual
            x = nn.Dense(self.d_model, name=f"out_proj_{i}")(h_complex)
            x = x + residual
        return x


# ── Classifiers ──────────────────────────────────────────────────────────────

def make_gated_wave_classifier(d_model, n_layers, n_scales, n_classes=3):
    base = GatedWaveModel(
        d_model=d_model,
        n_layers=n_layers,
        n_scales=n_scales,
        theta_min=0.01,
        theta_max=1.0,
        log_theta=False,
    )

    def init(rng, x):
        base_params = base.init(rng, x)
        head_rng = jax.random.fold_in(rng, 99)
        head_w = jax.random.normal(head_rng, (d_model, n_classes)) * 0.01
        head_b = jnp.zeros(n_classes)
        return {"base": base_params, "head_w": head_w, "head_b": head_b}

    def apply(params, x):
        h = base.apply(params["base"], x)   # (batch, seq_len, d_model)
        h_last = h[:, -1, :]
        return h_last @ params["head_w"] + params["head_b"]

    return init, apply


def make_hierarchical_classifier(d_model, n_layers, n_scales, n_classes=3):
    base = HierarchicalWaveSeqModel(
        d_model=d_model,
        n_layers=n_layers,
        n_scales=n_scales,
    )

    def init(rng, x):
        base_params = base.init(rng, x)
        head_rng = jax.random.fold_in(rng, 99)
        head_w = jax.random.normal(head_rng, (d_model, n_classes)) * 0.01
        head_b = jnp.zeros(n_classes)
        return {"base": base_params, "head_w": head_w, "head_b": head_b}

    def apply(params, x):
        h = base.apply(params["base"], x)   # (batch, seq_len, d_model)
        h_last = h[:, -1, :]
        return h_last @ params["head_w"] + params["head_b"]

    return init, apply


def make_mingru_classifier(d_model, n_layers, n_classes=3):
    base = MinGRUModel(d_model=d_model, n_layers=n_layers)

    def init(rng, x):
        base_params = base.init(rng, x)
        head_rng = jax.random.fold_in(rng, 99)
        head_w = jax.random.normal(head_rng, (d_model, n_classes)) * 0.01
        head_b = jnp.zeros(n_classes)
        return {"base": base_params, "head_w": head_w, "head_b": head_b}

    def apply(params, x):
        h = base.apply(params["base"], x)
        h_last = h[:, -1, :]
        return h_last @ params["head_w"] + params["head_b"]

    return init, apply


# ── Training loop ────────────────────────────────────────────────────────────

def train_and_eval(
    name: str,
    init_fn,
    apply_fn,
    rng_seed: int = 42,
    seq_len: int = 128,
    batch_size: int = 64,
    n_steps: int = 2000,
    report_interval: int = 400,
    lr: float = 3e-4,
    T_fast: int = 7,
    T_slow: int = 31,
) -> dict:
    rng = jax.random.PRNGKey(rng_seed)
    np_rng = np.random.default_rng(rng_seed)

    x_dummy = jnp.zeros((batch_size, seq_len, 1))
    params = init_fn(rng, x_dummy)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, x, y):
        logits = apply_fn(params, x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    @jax.jit
    def step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def predict(params, x):
        return jnp.argmax(apply_fn(params, x), axis=-1)

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")

    final_acc = 0.0
    per_class_final = [0.0, 0.0, 0.0]

    for s in range(n_steps):
        x_np, y_np = generate_am_batch(np_rng, batch_size, seq_len, T_fast, T_slow)
        x = jnp.array(x_np)
        y = jnp.array(y_np)
        params, opt_state, loss = step(params, opt_state, x, y)

        if (s + 1) % report_interval == 0:
            x_eval, y_eval = generate_am_batch(np_rng, 600, seq_len, T_fast, T_slow)
            preds = np.array(predict(params, jnp.array(x_eval)))
            acc = np.mean(preds == y_eval)

            per_class = []
            labels_str = ["noise", "fast", "AM"]
            for c in range(3):
                mask = y_eval == c
                per_class.append(np.mean(preds[mask] == c) if mask.sum() > 0 else float("nan"))

            per_str = "  ".join(f"{labels_str[c]}:{per_class[c]:.3f}" for c in range(3))
            print(f"  step {s+1:4d}  loss={loss:.4f}  acc={acc:.3f}  [{per_str}]")
            final_acc = acc
            per_class_final = per_class

    return {
        "name": name,
        "acc": final_acc,
        "per_class": per_class_final,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    D = 32
    N_LAYERS = 2
    N_SCALES = 4
    T_FAST = 7
    T_SLOW = 31
    SEQ_LEN = 128   # ~4 slow cycles (31*4=124), ~18 fast cycles
    STEPS = 2000
    BATCH = 64

    print(f"\nH13: Amplitude Modulation Detection")
    print(f"T_fast={T_FAST} (theta={2*np.pi/T_FAST:.3f}), T_slow={T_SLOW} (theta={2*np.pi/T_SLOW:.3f})")
    print(f"L={SEQ_LEN}, d={D}, {N_LAYERS}L, {N_SCALES} scales, {STEPS} steps")
    print(f"\nTask:")
    print(f"  Class 0 (noise): baseline — 33% expected at chance")
    print(f"  Class 1 (fast):  detect A*sin(2π*t/{T_FAST}) in noise")
    print(f"  Class 2 (AM):    detect amplitude-modulated signal")
    print(f"  Key: Class 1 vs Class 2 requires detecting amplitude variation at period {T_SLOW}")

    configs = [
        ("GatedWave (input-side only)",    *make_gated_wave_classifier(D, N_LAYERS, N_SCALES)),
        ("HierarchicalWave (+ state-side)", *make_hierarchical_classifier(D, N_LAYERS, N_SCALES)),
        ("minGRU (baseline)",               *make_mingru_classifier(D, N_LAYERS)),
    ]

    results = []
    for name, init_fn, apply_fn in configs:
        r = train_and_eval(
            name, init_fn, apply_fn,
            seq_len=SEQ_LEN, batch_size=BATCH,
            n_steps=STEPS, report_interval=500,
            T_fast=T_FAST, T_slow=T_SLOW,
        )
        results.append(r)

    print(f"\n{'═'*55}")
    print(f"  H13 RESULTS SUMMARY")
    print(f"{'═'*55}")
    print(f"  {'Model':<35} {'Overall':>8}  noise   fast   AM")
    print(f"  {'─'*53}")
    for r in results:
        pc = r["per_class"]
        print(f"  {r['name']:<35} {r['acc']:>8.3f}  {pc[0]:.3f}  {pc[1]:.3f}  {pc[2]:.3f}")

    print(f"\n  Chance baseline: 0.333 overall")
    print(f"  Key question: is AM class accuracy significantly different between")
    print(f"  GatedWave and HierarchicalWave?")
    print(f"  H2 (input-side sufficient, conf=55%): expect GatedWave ≈ HierarchicalWave")

    # H2 assessment
    gw = results[0]
    hw = results[1]
    mg = results[2]
    am_gap = hw["per_class"][2] - gw["per_class"][2]
    print(f"\n  AM class gap (HierarchicalWave - GatedWave): {am_gap:+.3f}")
    if abs(am_gap) < 0.05:
        print(f"  → H2 SUPPORTED: state-side coupling adds no significant AM benefit")
    elif am_gap > 0.05:
        print(f"  → H2 CHALLENGED: state-side coupling helps for hierarchical AM detection")
    else:
        print(f"  → H2 SUPPORTED (GatedWave marginally better — coupling may be noise)")


if __name__ == "__main__":
    main()
