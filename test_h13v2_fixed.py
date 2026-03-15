"""
H13v2: Amplitude Modulation Detection — Confound-Fixed

Fixes three confounds from H13 (2026-03-15):
  1. Power asymmetry: AM signal had 18% higher RMS → minGRU detected power not structure.
     Fix: normalize AM signal to same RMS as flat-amplitude signal.
  2. Theta mismatch: linear theta [0.01→1.0] gives scales at periods [628, 18, 9, 6].
     T_slow=31 is unrepresented. Fix: log-theta [0.063→1.257] gives [100, 33, 11, 5] —
     covers both T_fast=7 (between 5 and 11) and T_slow=31 (at ~33).
  3. Too few steps: run 3000 instead of 2000.

H2 claim: input-side coupling (GatedWave gate on prev_h) sufficient for hierarchical AM detection.
Prediction if H2 correct: GatedWave ≈ HierarchicalWave on AM class.
Key diagnostic: minGRU AM accuracy should drop significantly once power confound is fixed.

T_fast=7, T_slow=31 (coprime), L=128, d=32, 2L, n_scales=4, 3000 steps.
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


# ── Task (power-normalized) ───────────────────────────────────────────────────

def generate_am_batch_normalized(
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
    Power-normalized version. Class 1 and Class 2 have identical RMS amplitude.

    Class 0: noise
    Class 1: A*sin(2π*t/T_fast) + noise                                [flat amp]
    Class 2: A*(1+m*sin(2π*t/T_slow))*sin(2π*t/T_fast)/sqrt(1+m²/2) + noise  [AM, normalized]

    E[(1+m*sin)²] = 1 + m²/2, so dividing by sqrt(1+m²/2) normalizes to same power as Class 1.
    """
    t = np.arange(seq_len, dtype=np.float32)
    fast_signal = amplitude * np.sin(2 * np.pi * t / T_fast)

    slow_envelope = modulation_depth * np.sin(2 * np.pi * t / T_slow)
    am_raw = amplitude * (1.0 + slow_envelope) * np.sin(2 * np.pi * t / T_fast)
    # Normalize AM to same expected power as flat signal
    am_rms_factor = np.sqrt(1.0 + modulation_depth ** 2 / 2.0)
    am_signal = am_raw / am_rms_factor

    labels = rng.integers(0, 3, size=batch_size)
    noise = rng.normal(0, noise_std, size=(batch_size, seq_len)).astype(np.float32)

    inputs = noise.copy()
    inputs[labels == 1] += fast_signal
    inputs[labels == 2] += am_signal

    return inputs[:, :, None], labels.astype(np.int32)


# ── HierarchicalWave model ────────────────────────────────────────────────────

class HierarchicalWaveSeqModel(nn.Module):
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
            h_complex = HierarchicalWaveLayer(
                d_model=self.d_model,
                n_scales=self.n_scales,
                r_min=self.r_min,
                r_max=self.r_max,
                name=f"layer_{i}",
            )(x, conditioning=prev_h)
            prev_h = h_complex
            x = nn.Dense(self.d_model, name=f"out_proj_{i}")(h_complex)
            x = x + residual
        return x


# ── Classifiers ───────────────────────────────────────────────────────────────

def make_gated_wave_classifier(d_model, n_layers, n_scales, n_classes=3,
                                theta_min=0.063, theta_max=1.257, log_theta=True):
    base = GatedWaveModel(
        d_model=d_model,
        n_layers=n_layers,
        n_scales=n_scales,
        theta_min=theta_min,
        theta_max=theta_max,
        log_theta=log_theta,
    )

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


def make_hierarchical_classifier(d_model, n_layers, n_scales, n_classes=3):
    base = HierarchicalWaveSeqModel(
        d_model=d_model, n_layers=n_layers, n_scales=n_scales,
    )

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


# ── Training loop ─────────────────────────────────────────────────────────────

def train_and_eval(
    name: str,
    init_fn,
    apply_fn,
    rng_seed: int = 42,
    seq_len: int = 128,
    batch_size: int = 64,
    n_steps: int = 3000,
    report_interval: int = 500,
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
        x_np, y_np = generate_am_batch_normalized(np_rng, batch_size, seq_len, T_fast, T_slow)
        x = jnp.array(x_np)
        y = jnp.array(y_np)
        params, opt_state, loss = step(params, opt_state, x, y)

        if (s + 1) % report_interval == 0:
            x_eval, y_eval = generate_am_batch_normalized(np_rng, 600, seq_len, T_fast, T_slow)
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

    return {"name": name, "acc": final_acc, "per_class": per_class_final}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    D = 32
    N_LAYERS = 2
    N_SCALES = 4
    T_FAST = 7
    T_SLOW = 31
    SEQ_LEN = 128
    STEPS = 3000
    BATCH = 64

    # Log-theta, language-tuned range: periods [100, 33, 11, 5] — covers T_slow≈31 and T_fast=7
    THETA_MIN = 0.063   # period ~100
    THETA_MAX = 1.257   # period ~5

    print(f"\nH13v2: Amplitude Modulation Detection (Confound-Fixed)")
    print(f"Fixes from H13: power normalization + log-theta covering T_slow={T_SLOW}")
    print(f"T_fast={T_FAST} (theta={2*np.pi/T_FAST:.3f}), T_slow={T_SLOW} (theta={2*np.pi/T_SLOW:.3f})")
    print(f"Theta range: [{THETA_MIN}, {THETA_MAX}] log-spaced → periods ≈ [100, 33, 11, 5]")
    print(f"L={SEQ_LEN}, d={D}, {N_LAYERS}L, {N_SCALES} scales, {STEPS} steps")
    print(f"\nKey confound fixed: AM amplitude normalized by 1/sqrt(1+m²/2)={1/np.sqrt(1+0.36):.3f}")
    print(f"  minGRU's H13 AM=55.3% was likely power detection, not structure detection.")
    print(f"  Prediction: minGRU AM accuracy drops to near chance; GW/HW gap reveals true signal.")

    configs = [
        ("GatedWave (input-side, log-theta)",   *make_gated_wave_classifier(
            D, N_LAYERS, N_SCALES, theta_min=THETA_MIN, theta_max=THETA_MAX, log_theta=True)),
        ("HierarchicalWave (+ state-side)",     *make_hierarchical_classifier(D, N_LAYERS, N_SCALES)),
        ("minGRU (baseline)",                    *make_mingru_classifier(D, N_LAYERS)),
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
    print(f"  H13v2 RESULTS SUMMARY")
    print(f"{'═'*55}")
    print(f"  {'Model':<38} {'Overall':>7}  noise   fast   AM")
    print(f"  {'─'*56}")
    for r in results:
        pc = r["per_class"]
        print(f"  {r['name']:<38} {r['acc']:>7.3f}  {pc[0]:.3f}  {pc[1]:.3f}  {pc[2]:.3f}")

    print(f"\n  Chance: 0.333.  AM normalization factor: 1/sqrt(1+m²/2) ≈ 0.857")
    print(f"\n  H2 assessment (input-side sufficient):")
    gw = results[0]; hw = results[1]; mg = results[2]
    am_gap = hw["per_class"][2] - gw["per_class"][2]
    mg_drop = 0.553 - mg["per_class"][2]  # vs H13 baseline
    print(f"    AM gap (HierarchicalWave - GatedWave): {am_gap:+.3f}")
    print(f"    minGRU AM drop vs H13 (power-cheat eliminated): {mg_drop:+.3f}")
    if abs(am_gap) < 0.05:
        print(f"  → H2 SUPPORTED: state-side coupling adds no significant AM benefit (after fixing confounds)")
    elif am_gap > 0.05:
        print(f"  → H2 CHALLENGED: state-side coupling genuinely helps hierarchical AM detection")
    else:
        print(f"  → H2 WEAKLY SUPPORTED: GW marginally better — coupling may add noise")


if __name__ == "__main__":
    main()
