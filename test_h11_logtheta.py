"""
H11 Hypothesis Test: Log-spaced theta outperforms linear for multi-scale frequency tasks.

Task design:
  3-class classification — each sequence contains ONE periodic signal:
    Class 0: A*sin(2π*t/5)  + noise  (fast, T=5)
    Class 1: A*sin(2π*t/20) + noise  (medium, T=20)
    Class 2: A*sin(2π*t/80) + noise  (slow, T=80)

  L = 160 = 2*T_max = 2 full periods of the slowest signal.
  Per-sample SNR ~0.3 (requires coherent accumulation over many periods for classes 0,1).

Why this tests H11:
  With n_scales=4, theta range 2π/80 to 2π/5 (0.079 to 1.257):

  Linear spacing: [0, 0.079, 0.57, 1.26]
    → resonates well at T=80 (0.079≈2π/80) and T=5 (1.26≈2π/5)
    → T=20 resonance (0.31) falls in a gap

  Log spacing: [0, 0.079, 0.31, 1.26]  (geometric: √(0.079*1.26) ≈ 0.32)
    → resonates at all three target periods
    → more even octave coverage

Prediction (H11): Log-spaced theta converges faster on class 1 (T=20) detection specifically.
  Overall accuracy advantage expected but may be modest at 2000 steps.

Usage:
    cd ~/projects/continuous-language && source .venv/bin/activate
    PYTHONPATH=/home/meridian/src python test_h11_logtheta.py
    PYTHONPATH=/home/meridian/src python test_h11_logtheta.py --steps 3000 --n_scales 6
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import argparse
from typing import Tuple

from gated_wave import GatedWaveDynamicsLayer

import warnings
warnings.filterwarnings('ignore')

# Periods to distinguish — geometric spacing across 4 octaves
PERIODS = [5, 20, 80]
T_MAX = max(PERIODS)
T_MIN = min(PERIODS)


class MultiPeriodDetector(nn.Module):
    d_model: int
    n_scales: int
    r_min: float = 0.9
    r_max: float = 0.999
    theta_min: float = 0.0
    theta_max: float = 1.0
    log_theta: bool = False

    @nn.compact
    def __call__(self, x):
        x_proj = nn.Dense(self.d_model, name="input_proj")(x)
        h = GatedWaveDynamicsLayer(
            d_model=self.d_model,
            n_scales=self.n_scales,
            r_min=self.r_min,
            r_max=self.r_max,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            log_theta=self.log_theta,
            name="dynamics",
        )(x_proj)
        return nn.Dense(len(PERIODS), name="classifier")(h[:, -1, :])


def make_batch(
    batch_size: int,
    seq_len: int,
    amplitude: float,
    noise_std: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    t = np.arange(seq_len, dtype=np.float32)
    x = rng.normal(0, noise_std, (batch_size, seq_len)).astype(np.float32)
    labels = rng.integers(0, len(PERIODS), size=batch_size)
    for i, period in enumerate(PERIODS):
        mask = labels == i
        x[mask] += amplitude * np.sin(2 * np.pi * t / period)
    return x[:, :, None], labels


def run_config(name: str, log_theta: bool, args) -> dict:
    theta_min = 2 * np.pi / T_MAX  # ≈ 0.0785 (matches T=80)
    theta_max = 2 * np.pi / T_MIN  # ≈ 1.257  (matches T=5)

    if log_theta:
        osc = np.exp(np.linspace(np.log(theta_min), np.log(theta_max), args.n_scales - 1))
    else:
        osc = np.linspace(theta_min, theta_max, args.n_scales - 1)

    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"  theta_min={theta_min:.4f}  theta_max={theta_max:.4f}")
    print(f"  Oscillatory theta values: {[f'{v:.3f}' for v in osc]}")
    period_labels = [f"T={2*np.pi/v:.0f}" for v in osc]
    print(f"  → resonant periods:       {period_labels}")
    print(f"  (target periods: T=5, T=20, T=80)")

    model = MultiPeriodDetector(
        d_model=args.d_model,
        n_scales=args.n_scales,
        r_min=0.9,
        r_max=0.999,
        theta_min=theta_min,
        theta_max=theta_max,
        log_theta=log_theta,
    )

    rng = np.random.default_rng(42)
    seq_len = args.L
    key = jax.random.PRNGKey(0)
    x_init = jnp.ones((1, seq_len, 1))
    params = model.init(key, x_init)

    n_params = sum(v.size for v in jax.tree_util.tree_leaves(params))
    print(f"  Parameters: {n_params:,}")

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, x, y):
        def loss_fn(p):
            logits = model.apply(p, x)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_acc(params, x, y):
        logits = model.apply(params, x)
        preds = jnp.argmax(logits, axis=-1)
        return jnp.mean(preds == y)

    print(f"\n  step  loss    acc   (acc by class: T=5  T=20  T=80)")
    print(f"  {'─'*55}")

    history = {'step': [], 'loss': [], 'acc': [], 'per_class': []}

    for step in range(1, args.steps + 1):
        x_batch, y_batch = make_batch(args.batch_size, seq_len, 0.3, 1.0, rng)
        x_jax = jnp.array(x_batch)
        y_jax = jnp.array(y_batch)
        params, opt_state, loss = train_step(params, opt_state, x_jax, y_jax)

        if step % args.log_every == 0 or step == args.steps:
            # Eval on a larger batch
            x_eval, y_eval = make_batch(512, seq_len, 0.3, 1.0, rng)
            x_e = jnp.array(x_eval)
            y_e = jnp.array(y_eval)
            acc = float(eval_acc(params, x_e, y_e))

            # Per-class accuracy
            logits = model.apply(params, x_e)
            preds = jnp.argmax(logits, axis=-1)
            per_class = []
            for c in range(len(PERIODS)):
                mask = y_e == c
                if mask.sum() > 0:
                    per_class.append(float(jnp.mean(preds[mask] == c)))
                else:
                    per_class.append(float('nan'))

            print(f"  {step:5d}  {float(loss):.4f}  {acc:.3f}  "
                  f"({per_class[0]:.2f}  {per_class[1]:.2f}  {per_class[2]:.2f})")

            history['step'].append(step)
            history['loss'].append(float(loss))
            history['acc'].append(acc)
            history['per_class'].append(per_class)

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--n_scales', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--L', type=int, default=160,
                        help='Sequence length (default: 2*T_max=160)')
    parser.add_argument('--log_every', type=int, default=200)
    args = parser.parse_args()

    print(f"H11 Test: Log-spaced vs Linear theta for multi-period detection")
    print(f"Task: 3-class (T={PERIODS}), L={args.L}, {args.steps} steps")
    print(f"Periods: geometric ratio {PERIODS[1]/PERIODS[0]}x between consecutive classes")
    print(f"n_scales={args.n_scales} ({args.n_scales-1} oscillatory + 1 integrator)")

    h_linear = run_config("Linear theta", log_theta=False, args=args)
    h_log = run_config("Log theta", log_theta=True, args=args)

    print(f"\n{'='*60}")
    print(f"SUMMARY — final step ({args.steps})")
    print(f"{'Config':<20} {'Acc':>6}  T=5   T=20  T=80")
    print(f"{'─'*50}")

    def fmt(h):
        acc = h['acc'][-1]
        pc = h['per_class'][-1]
        return f"{acc:.3f}  {pc[0]:.2f}  {pc[1]:.2f}  {pc[2]:.2f}"

    print(f"{'Linear theta':<20} {fmt(h_linear)}")
    print(f"{'Log theta':<20} {fmt(h_log)}")

    # H11 verdict
    acc_linear = h_linear['acc'][-1]
    acc_log = h_log['acc'][-1]
    t20_linear = h_linear['per_class'][-1][1]
    t20_log = h_log['per_class'][-1][1]

    print(f"\nH11 assessment:")
    print(f"  Overall: log={acc_log:.3f} vs linear={acc_linear:.3f} (Δ={acc_log-acc_linear:+.3f})")
    print(f"  T=20 class: log={t20_log:.3f} vs linear={t20_linear:.3f} (Δ={t20_log-t20_linear:+.3f})")

    if t20_log > t20_linear + 0.05:
        print(f"  → H11 SUPPORTED on T=20 class (log-spacing fills the resonance gap)")
    elif t20_log > t20_linear:
        print(f"  → H11 WEAK on T=20 class (positive trend, not decisive)")
    else:
        print(f"  → H11 NOT SUPPORTED (log-spacing provides no T=20 advantage)")

    if acc_log > acc_linear + 0.03:
        print(f"  → Overall H11 SUPPORTED (log beats linear by >{3:.0f}pp)")
    else:
        print(f"  → Overall H11 NOT SUPPORTED at this scale")


if __name__ == "__main__":
    main()
