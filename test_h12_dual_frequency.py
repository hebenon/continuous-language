"""
H12: Dual-Frequency Simultaneous Detection

Tests whether multi-scale GatedWave can detect TWO superimposed sinusoids
simultaneously — the genuine multi-scale hypothesis.

H9 confirmed: single-frequency detection, theta-matched model +30pp.
H11 confirmed: log-spaced theta improves multi-frequency coverage.
H12 asks: can the model simultaneously track both signals in the same input?

Task: 4-class classification
  Class 0: pure noise
  Class 1: noise + A*sin(2π*t/T1)   (fast signal, T1=7)
  Class 2: noise + A*sin(2π*t/T2)   (slow signal, T2=29)
  Class 3: noise + both signals

T1 and T2 chosen to be coprime (no harmonic relationship), forcing the model
to maintain two independent tracking channels.

Models compared:
  - GatedWave, log-spaced theta (H11 winner), n_scales=4
  - GatedWave, linear theta (H11 baseline), n_scales=4
  - GatedWave, theta_matched (scales tuned to T1 and T2), n_scales=4
  - minGRU (no oscillatory dynamics, serves as ceiling-free baseline)

Expected: log-spaced and theta_matched should outperform linear.
Key question: does per-class accuracy reveal which signals the model struggles with?
(Class 3 "both" should be hardest — tests whether the model has independent tracking.)

Run on Pi (d=32, L=128, 2000 steps ~5 min). Scale on Lambda for d=256 L=4L.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from gated_wave import GatedWaveModel
from train_minGRU import MinGRU as MinGRUModel  # reuse existing baseline


# ── Task ────────────────────────────────────────────────────────────────────

def generate_dual_frequency_batch(
    rng: np.random.Generator,
    batch_size: int,
    seq_len: int,
    T1: int = 7,
    T2: int = 29,
    amplitude: float = 0.4,
    noise_std: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (inputs, labels) where:
      inputs: (batch, seq_len, 1)  float32
      labels: (batch,)             int32  in {0,1,2,3}
    """
    t = np.arange(seq_len, dtype=np.float32)
    sin1 = amplitude * np.sin(2 * np.pi * t / T1)
    sin2 = amplitude * np.sin(2 * np.pi * t / T2)

    labels = rng.integers(0, 4, size=batch_size)
    noise = rng.normal(0, noise_std, size=(batch_size, seq_len)).astype(np.float32)

    inputs = noise.copy()
    inputs[labels == 1] += sin1
    inputs[labels == 2] += sin2
    inputs[labels == 3] += sin1 + sin2

    return inputs[:, :, None], labels.astype(np.int32)


# ── Thin classifiers wrapping existing recurrent models ─────────────────────

def make_gated_wave_classifier(d_model, n_layers, n_scales, theta_min, theta_max,
                                log_theta, n_classes=4):
    """Returns (init_fn, apply_fn) for a GatedWave + linear head classifier."""
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
        # Head: d_model -> n_classes
        head_rng = jax.random.fold_in(rng, 99)
        head_w = jax.random.normal(head_rng, (d_model, n_classes)) * 0.01
        head_b = jnp.zeros(n_classes)
        return {"base": base_params, "head_w": head_w, "head_b": head_b}

    def apply(params, x):
        # x: (batch, seq, 1)
        h = base.apply(params["base"], x)       # (batch, seq, d_model)
        h_last = h[:, -1, :]                    # (batch, d_model)
        logits = h_last @ params["head_w"] + params["head_b"]
        return logits                            # (batch, n_classes)

    return init, apply


def make_mingru_classifier(d_model, n_layers, n_classes=4):
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
        logits = h_last @ params["head_w"] + params["head_b"]
        return logits

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
    eval_steps: int = 200,
    lr: float = 3e-4,
    T1: int = 7,
    T2: int = 29,
    report_interval: int = 400,
) -> dict:
    rng = jax.random.PRNGKey(rng_seed)
    np_rng = np.random.default_rng(rng_seed)

    # Dummy init batch
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

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")

    for s in range(n_steps):
        x_np, y_np = generate_dual_frequency_batch(np_rng, batch_size, seq_len, T1, T2)
        x = jnp.array(x_np)
        y = jnp.array(y_np)
        params, opt_state, loss = step(params, opt_state, x, y)

        if (s + 1) % report_interval == 0:
            # Eval on fresh batch
            x_eval, y_eval = generate_dual_frequency_batch(np_rng, 512, seq_len, T1, T2)
            preds = predict(params, jnp.array(x_eval))
            acc = np.mean(np.array(preds) == y_eval)
            # Per-class accuracy
            per_class = []
            for c in range(4):
                mask = y_eval == c
                if mask.sum() > 0:
                    per_class.append(np.mean(np.array(preds)[mask] == c))
                else:
                    per_class.append(float('nan'))
            labels = ["none", f"T1={T1}", f"T2={T2}", "both"]
            per_str = "  ".join(f"{labels[c]}:{per_class[c]:.2f}" for c in range(4))
            print(f"  step {s+1:4d}  loss={loss:.4f}  acc={acc:.3f}  [{per_str}]")

    # Final eval
    x_eval, y_eval = generate_dual_frequency_batch(np_rng, 1024, seq_len, T1, T2)
    preds = np.array(predict(params, jnp.array(x_eval)))
    final_acc = np.mean(preds == y_eval)
    per_class_final = [np.mean(preds[y_eval == c] == c) for c in range(4)]

    return {
        "name": name,
        "final_acc": final_acc,
        "per_class": per_class_final,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    D = 32
    L = 2
    T1, T2 = 7, 29
    SEQ = 128
    STEPS = 2000

    # Theta ranges corresponding to periods [T2..T1] = [29..7] in radians/step
    # omega = 2pi/T, so theta_min = 2pi/T2, theta_max = 2pi/T1
    theta_min_lang  = 2 * np.pi / T2   # ~0.217, period 29
    theta_max_lang  = 2 * np.pi / T1   # ~0.898, period 7

    configs = [
        # name, theta_min, theta_max, log_theta, is_mingru
        ("GatedWave log-theta  [matched T1/T2]", theta_min_lang, theta_max_lang, True,  False),
        ("GatedWave linear-theta [matched T1/T2]", theta_min_lang, theta_max_lang, False, False),
        ("GatedWave log-theta [broad 0.01-1.0]",  0.01,           1.0,            True,  False),
        ("minGRU baseline",                        None,           None,           None,  True),
    ]

    results = []
    for name, tmin, tmax, log_th, is_mingru in configs:
        if is_mingru:
            init_fn, apply_fn = make_mingru_classifier(D, L)
        else:
            init_fn, apply_fn = make_gated_wave_classifier(
                D, L, n_scales=4, theta_min=tmin, theta_max=tmax, log_theta=log_th
            )
        r = train_and_eval(
            name, init_fn, apply_fn,
            seq_len=SEQ, n_steps=STEPS, T1=T1, T2=T2, report_interval=500,
        )
        results.append(r)

    print(f"\n{'═'*60}")
    print("  H12 Final Results — Dual-Frequency Detection")
    print(f"  T1={T1} (fast), T2={T2} (slow), seq={SEQ}, d={D}, L={L}")
    print(f"{'═'*60}")
    labels = ["none", f"T1={T1}", f"T2={T2}", "both"]
    for r in results:
        pc = "  ".join(f"{labels[c]}:{r['per_class'][c]:.3f}" for c in range(4))
        print(f"  {r['name']:<42}  overall={r['final_acc']:.3f}  [{pc}]")
    print(f"{'═'*60}")
    print("  Chance: 0.250  |  Key: 'both' class tests simultaneous tracking")


if __name__ == "__main__":
    main()
