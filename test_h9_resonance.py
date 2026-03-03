"""
H9 Hypothesis Test: Theta-Period Matching for Oscillatory Resonance.

Tests whether theta = 2pi/T accelerates signal detection compared to mismatched theta.

Task design (avoids prior confounds):
- Binary classification: signal present (A*sin(2pi*t/T) + noise) vs noise only
- Sequence length L = N*T (exact multiples of period)
- Per-sample SNR << 1 (can't classify from any single sample)
- CRITICAL: integrator CAN'T help — sin(2pi*t/T) sums to 0 over complete periods
  The oscillatory state at theta=2pi/T accumulates coherently; mismatched theta does not.

Configs tested (all have integrator as scale 0):
  matched:    theta_max = 2pi/T  (oscillatory scale resonates at signal frequency)
  fast:       theta_max = 1.0    (period ~6.3, much faster than T)
  slow:       theta_max = 0.05   (period ~126, much slower than T)
  pure_int:   n_scales=1, r=1.0  (pure integrator — should ~random on this task)

Prediction: matched converges substantially faster and/or achieves higher accuracy.

Usage:
    python test_h9_resonance.py
    python test_h9_resonance.py --T 30 --L 300 --steps 3000
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import argparse
from typing import List, Tuple

from gated_wave import GatedWaveDynamicsLayer


# ── Model ─────────────────────────────────────────────────────────────────────

class SignalDetector(nn.Module):
    """Minimal wrapper: project input → oscillatory dynamics → classify final state."""
    d_model: int
    n_scales: int
    r_min: float = 0.9
    r_max: float = 0.999
    theta_min: float = 0.0
    theta_max: float = 1.0

    @nn.compact
    def __call__(self, x):
        # x: [batch, seq, 1] float
        x_proj = nn.Dense(self.d_model, name="input_proj")(x)  # [batch, seq, d]
        h = GatedWaveDynamicsLayer(
            d_model=self.d_model,
            n_scales=self.n_scales,
            r_min=self.r_min,
            r_max=self.r_max,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            name="dynamics",
        )(x_proj)  # [batch, seq, 2*d] (real + imag concatenated)
        # Classify from final timestep's accumulated state
        return nn.Dense(2, name="classifier")(h[:, -1, :])  # [batch, 2]


# ── Data ──────────────────────────────────────────────────────────────────────

def make_batch(
    batch_size: int,
    seq_len: int,
    period: int,
    amplitude: float,
    noise_std: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate [batch_size] sequences of length [seq_len].
    Class 0: pure noise
    Class 1: A*sin(2*pi*t/T) + noise

    Returns x: [batch, seq, 1], y: [batch] int
    """
    labels = rng.integers(0, 2, batch_size)
    t = np.arange(seq_len, dtype=np.float32)
    signal = amplitude * np.sin(2 * np.pi * t / period)  # [seq]

    xs = []
    for label in labels:
        noise = rng.standard_normal(seq_len).astype(np.float32) * noise_std
        x = (signal + noise) if label == 1 else noise
        xs.append(x)

    x_arr = np.stack(xs)[:, :, np.newaxis]  # [batch, seq, 1]
    return x_arr, labels.astype(np.int32)


# ── Training ──────────────────────────────────────────────────────────────────

def count_params(params) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def run_experiment(
    name: str,
    d_model: int,
    n_scales: int,
    r_min: float,
    r_max: float,
    theta_min: float,
    theta_max: float,
    seq_len: int,
    period: int,
    amplitude: float,
    noise_std: float,
    steps: int,
    batch_size: int,
    lr: float,
    eval_every: int,
    seed: int,
) -> List[Tuple[int, float]]:
    """Train one config, return list of (step, accuracy) eval points."""
    model = SignalDetector(
        d_model=d_model,
        n_scales=n_scales,
        r_min=r_min,
        r_max=r_max,
        theta_min=theta_min,
        theta_max=theta_max,
    )

    key = jax.random.PRNGKey(seed)
    dummy_x = jnp.zeros((batch_size, seq_len, 1))
    params = model.init(key, dummy_x)
    n_params = count_params(params)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, x, y):
        def loss_fn(p):
            logits = model.apply(p, x)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def eval_batch(params, x):
        logits = model.apply(params, x)
        return jnp.argmax(logits, axis=-1)

    rng_train = np.random.default_rng(seed)
    rng_eval = np.random.default_rng(seed + 1000)

    print(f"\n  {name} (params={n_params:,}, theta_max={theta_max:.4f})")
    results = []

    for step in range(1, steps + 1):
        x, y = make_batch(batch_size, seq_len, period, amplitude, noise_std, rng_train)
        x_jax, y_jax = jnp.array(x), jnp.array(y)
        params, opt_state, loss = train_step(params, opt_state, x_jax, y_jax)

        if step % eval_every == 0:
            # Evaluate on 200 samples
            n_eval = 200
            x_ev, y_ev = make_batch(n_eval, seq_len, period, amplitude, noise_std, rng_eval)
            preds = eval_batch(params, jnp.array(x_ev))
            acc = float(jnp.mean(preds == jnp.array(y_ev)))
            print(f"    step {step:5d} | loss={float(loss):.4f} | acc={acc:.3f}")
            results.append((step, acc))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=20, help="Signal period in timesteps")
    parser.add_argument("--L", type=int, default=200, help="Sequence length (should be multiple of T)")
    parser.add_argument("--amplitude", type=float, default=0.3, help="Signal amplitude (class 1)")
    parser.add_argument("--noise_std", type=float, default=1.0, help="Noise std (should >> amplitude)")
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--n_scales", type=int, default=2, help=">=2 for oscillatory test")
    parser.add_argument("--r_max", type=float, default=0.999, help="Oscillatory scale r value")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    T = args.T
    theta_matched = float(2 * np.pi / T)

    print(f"H9 Resonance Test")
    print(f"  Period T={T}, seq_len={args.L} ({args.L // T} full periods)")
    print(f"  Signal amplitude={args.amplitude}, noise_std={args.noise_std}")
    print(f"  Per-sample SNR: {args.amplitude / args.noise_std:.2f}")
    print(f"  Coherent integration SNR (approx): {args.amplitude / args.noise_std * np.sqrt(args.L / 2):.2f}")
    print(f"  theta_matched = 2pi/{T} = {theta_matched:.4f}")
    print(f"\nNote: integrator CAN'T help — sin(2pi*t/T) sums to ~0 over {args.L // T} complete periods.")
    print("Prediction: matched config converges faster than mismatched or integrator-only.\n")

    configs = [
        # name, n_scales, r_min, r_max, theta_min, theta_max
        ("matched",   args.n_scales, 0.9, args.r_max, theta_matched * 0.9, theta_matched),
        ("fast",      args.n_scales, 0.9, args.r_max, 0.8,                 1.0),
        ("slow",      args.n_scales, 0.9, args.r_max, 0.01,                0.05),
        ("pure_int",  1,             1.0, 1.0,        0.0,                 0.0),  # pure integrator
    ]

    all_results = {}
    for name, n_scales, r_min, r_max, theta_min, theta_max in configs:
        results = run_experiment(
            name=name,
            d_model=args.d_model,
            n_scales=n_scales,
            r_min=r_min,
            r_max=r_max,
            theta_min=theta_min,
            theta_max=theta_max,
            seq_len=args.L,
            period=T,
            amplitude=args.amplitude,
            noise_std=args.noise_std,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_every=args.eval_every,
            seed=args.seed,
        )
        all_results[name] = results

    print("\n" + "=" * 60)
    print("SUMMARY: Final accuracy by config")
    print("=" * 60)
    print(f"  {'Config':<12} {'Final acc':>10}  {'Peak acc':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}")
    for name, results in all_results.items():
        if results:
            final_acc = results[-1][1]
            peak_acc = max(r[1] for r in results)
            print(f"  {name:<12} {final_acc:>10.3f}  {peak_acc:>10.3f}")

    print("\nH9 supported if: matched > fast ≈ slow > pure_int")
    print("H9 refuted if: all configs reach similar accuracy at same pace")

    # Also report convergence speed (steps to reach 0.7 accuracy)
    print("\n  Steps to reach 0.70 accuracy:")
    for name, results in all_results.items():
        steps_to_70 = None
        for step, acc in results:
            if acc >= 0.70:
                steps_to_70 = step
                break
        print(f"  {name:<12} {steps_to_70 if steps_to_70 else '>'+str(args.steps)}")


if __name__ == "__main__":
    main()
