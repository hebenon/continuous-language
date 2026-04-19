"""
H16: Sequence Length Scaling — Does Write-Coupling Advantage Grow with Context?

Extends H15 across multiple sequence lengths to test:
1. GW→HW gap (write-coupling advantage): stays near-zero (linear readout sufficiency
   is length-independent) or grows (composition harder at longer contexts)?
2. GW→minGRU gap: grows with length (GW advantage is about long-range periodic tracking)?

Task: same AM prediction as H15
  y_t = A(t) * sin(2π*t/7 + φ) + ε,  A(t) = 0.5 + 0.4*sin(2π*t/49 + φ')
  T_fast=7, T_slow=49, noise_std=0.3, phases randomized per sequence.

Sequence lengths tested:
  L=98  (2 slow cycles)  — baseline, replicates H15
  L=196 (4 slow cycles)
  L=392 (8 slow cycles)

Hypothesis (H16):
  GW→HW gap ≈ noise at all L (linear readout sufficient regardless of context length)
  GW→minGRU gap grows with L (GW better at long-range periodic memory)

If GW→HW gap grows: write-coupling is more useful at longer contexts —
  the temporal composition becomes harder as more slow cycles accumulate.
  This would suggest Config E might show more benefit on longer text sequences.

Filed: 2026-03-30
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


# ── Data generation (identical to H15) ────────────────────────────────────────

def generate_am_prediction_batch(
    rng, batch_size, seq_len,
    T_fast=7, T_slow=49,
    base_amplitude=0.5, modulation_depth=0.4,
    noise_std=0.3,
):
    t = np.arange(seq_len, dtype=np.float32)
    fast_phases = rng.uniform(0, 2 * np.pi, size=(batch_size, 1))
    slow_phases = rng.uniform(0, 2 * np.pi, size=(batch_size, 1))

    slow_mod = base_amplitude + modulation_depth * np.sin(
        2 * np.pi * t[None, :] / T_slow + slow_phases
    )
    y = slow_mod * np.sin(
        2 * np.pi * t[None, :] / T_fast + fast_phases
    ).astype(np.float32)
    y += rng.normal(0, noise_std, size=(batch_size, seq_len)).astype(np.float32)

    return y[:, :-1, None], y[:, 1:, None]


# ── Model definitions (identical to H15) ─────────────────────────────────────

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
                d_model=self.d_model, n_scales=self.n_scales,
                r_min=self.r_min, r_max=self.r_max, name=f"layer_{i}",
            )(x, conditioning=prev_h)
            prev_h = h_complex
            x = nn.Dense(self.d_model, name=f"out_proj_{i}")(h_complex)
            x = x + residual
        return x


def make_gated_wave_regressor(d_model, n_layers, n_scales,
                               theta_min=0.063, theta_max=1.257, log_theta=True):
    base = GatedWaveModel(d_model=d_model, n_layers=n_layers, n_scales=n_scales,
                          theta_min=theta_min, theta_max=theta_max, log_theta=log_theta)

    def init(rng, x):
        base_params = base.init(rng, x)
        head_rng = jax.random.fold_in(rng, 99)
        head_w = jax.random.normal(head_rng, (d_model, 1)) * 0.01
        return {"base": base_params, "head_w": head_w, "head_b": jnp.zeros(1)}

    def apply(params, x):
        h = base.apply(params["base"], x)
        return h @ params["head_w"] + params["head_b"]

    return init, apply


def make_hierarchical_regressor(d_model, n_layers, n_scales):
    base = HierarchicalWaveSeqModel(d_model=d_model, n_layers=n_layers, n_scales=n_scales)

    def init(rng, x):
        base_params = base.init(rng, x)
        head_rng = jax.random.fold_in(rng, 99)
        head_w = jax.random.normal(head_rng, (d_model, 1)) * 0.01
        return {"base": base_params, "head_w": head_w, "head_b": jnp.zeros(1)}

    def apply(params, x):
        h = base.apply(params["base"], x)
        return h @ params["head_w"] + params["head_b"]

    return init, apply


def make_mingru_regressor(d_model, n_layers):
    base = MinGRUModel(d_model=d_model, n_layers=n_layers)

    def init(rng, x):
        base_params = base.init(rng, x)
        head_rng = jax.random.fold_in(rng, 99)
        head_w = jax.random.normal(head_rng, (d_model, 1)) * 0.01
        return {"base": base_params, "head_w": head_w, "head_b": jnp.zeros(1)}

    def apply(params, x):
        h = base.apply(params["base"], x)
        return h @ params["head_w"] + params["head_b"]

    return init, apply


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_eval(name, init_fn, apply_fn, seq_len,
                   rng_seed=42, batch_size=64, n_steps=2000,
                   T_fast=7, T_slow=49, warmup_steps=49):
    rng = jax.random.PRNGKey(rng_seed)
    np_rng = np.random.default_rng(rng_seed)

    x_dummy = jnp.zeros((batch_size, seq_len - 1, 1))
    params = init_fn(rng, x_dummy)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def step_fn(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(
            lambda p: jnp.mean((apply_fn(p, x) - y) ** 2)
        )(params)
        updates, new_opt = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), new_opt, loss

    for s in range(n_steps):
        x_np, y_np = generate_am_prediction_batch(np_rng, batch_size, seq_len, T_fast, T_slow)
        params, opt_state, _ = step_fn(params, opt_state, jnp.array(x_np), jnp.array(y_np))

    # Final evaluation
    x_e, y_e = generate_am_prediction_batch(np_rng, 256, seq_len, T_fast, T_slow)
    preds = np.array(apply_fn(params, jnp.array(x_e)))
    y_e = np.array(y_e)

    total   = float(np.mean((preds - y_e) ** 2))
    post    = float(np.mean((preds[:, warmup_steps:] - y_e[:, warmup_steps:]) ** 2))
    return {"name": name, "mse": total, "post_mse": post}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    D        = 32
    N_LAYERS = 2
    N_SCALES = 4
    T_FAST   = 7
    T_SLOW   = 49
    WARMUP   = T_SLOW        # fixed: skip first slow cycle
    STEPS    = 2000
    BATCH    = 64
    THETA_MIN = 0.063
    THETA_MAX = 1.257

    SEQ_LENGTHS = [98, 196, 392]   # 2, 4, 8 slow cycles

    noise_floor = 0.3 ** 2

    print(f"\nH16: Sequence Length Scaling")
    print(f"Task: AM prediction (T_fast={T_FAST}, T_slow={T_SLOW}, noise_std=0.3)")
    print(f"Testing L = {SEQ_LENGTHS} ({[L//T_SLOW for L in SEQ_LENGTHS]} slow cycles each)")
    print(f"Noise floor: {noise_floor:.4f}")
    print(f"H16 prediction: GW→HW gap ≈ noise at all L; GW→minGRU gap grows with L")

    all_results = {}

    for L in SEQ_LENGTHS:
        print(f"\n{'═'*60}")
        print(f"  L = {L}  ({L // T_SLOW} slow cycles)")
        print(f"{'═'*60}")

        configs = [
            ("GatedWave",        *make_gated_wave_regressor(D, N_LAYERS, N_SCALES, THETA_MIN, THETA_MAX, True)),
            ("HierarchicalWave", *make_hierarchical_regressor(D, N_LAYERS, N_SCALES)),
            ("minGRU",           *make_mingru_regressor(D, N_LAYERS)),
        ]

        results = []
        for name, init_fn, apply_fn in configs:
            print(f"  Training {name}...", flush=True)
            r = train_and_eval(name, init_fn, apply_fn, seq_len=L,
                               batch_size=BATCH, n_steps=STEPS,
                               T_fast=T_FAST, T_slow=T_SLOW, warmup_steps=WARMUP)
            print(f"    post-MSE: {r['post_mse']:.4f}")
            results.append(r)

        all_results[L] = results

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  H16 SUMMARY  (post-warmup MSE, noise floor = {noise_floor:.4f})")
    print(f"{'═'*70}")
    print(f"  {'L':>5}  {'GatedWave':>10}  {'HW':>10}  {'minGRU':>10}  "
          f"{'GW→HW%':>8}  {'GW→minGRU%':>11}")
    print(f"  {'─'*70}")

    for L, results in all_results.items():
        gw = results[0]["post_mse"]
        hw = results[1]["post_mse"]
        mg = results[2]["post_mse"]
        gw_hw_pct = 100 * (gw - hw) / max(gw, 1e-9)
        mg_gw_pct = 100 * (mg - gw) / max(mg, 1e-9)
        print(f"  {L:>5}  {gw:>10.4f}  {hw:>10.4f}  {mg:>10.4f}  "
              f"{gw_hw_pct:>+8.1f}%  {mg_gw_pct:>+10.1f}%")

    print(f"\n  Interpretation:")
    print(f"  GW→HW% near zero at all L → linear readout sufficiency holds across context lengths")
    print(f"  GW→HW% growing with L     → write-coupling more useful at longer contexts")
    print(f"  GW→minGRU% growing with L → wave architecture advantage scales with context length")


if __name__ == "__main__":
    main()
