"""
H15: Hierarchical Sequence Prediction — Write-Coupling vs Read-Coupling

Tests whether write-coupling (HierarchicalWave) outperforms read-coupling (GatedWave)
in sequence prediction of hierarchically amplitude-modulated signals.

Key difference from H13v2: REGRESSION (predict next value) not CLASSIFICATION.
Sequence prediction is closer to char-LM than binary AM detection.

Task: y_t = A(t) * sin(2π*t/T_fast + φ) + ε
      A(t) = 0.5 + 0.4 * sin(2π*t/T_slow + φ')
      T_fast=7, T_slow=49 (ratio=7), L=98 (2 full slow cycles)
      ε ~ N(0, 0.3²), phases randomized per sequence

To predict y_{t+1} accurately:
  Need fast oscillator phase AND slow oscillator amplitude simultaneously.
  GatedWave: can track both independently (H12 confirmed), but must compose
    via gate conditioning on prev_h — cannot directly reshape fast-scale state.
  HierarchicalWave: write-coupling lets slow state directly modulate fast state.

H14 prediction: HierarchicalWave MSE < GatedWave MSE < minGRU MSE (post-warmup)
The GW→HW gap calibrates expected Config E BPC improvement:
  large gap (>15% MSE reduction) → expect substantial Config E improvement
  small gap (<5%)                 → gate conditioning sufficient; little improvement

Filed: 2026-03-27
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


# ── Data generation ───────────────────────────────────────────────────────────

def generate_am_prediction_batch(
    rng: np.random.Generator,
    batch_size: int,
    seq_len: int,
    T_fast: int = 7,
    T_slow: int = 49,
    base_amplitude: float = 0.5,
    modulation_depth: float = 0.4,
    noise_std: float = 0.3,
    phase_randomize: bool = True,
) -> tuple:
    """
    Generate AM prediction sequences with random initial phases.

    y_t = A(t) * sin(2π*t/T_fast + φ_fast) + ε
    A(t) = base_amplitude + modulation_depth * sin(2π*t/T_slow + φ_slow)

    Phase randomization forces the model to track oscillator state from context
    rather than memorizing a fixed phase alignment.

    Returns: (inputs, targets) both shape (batch, seq_len-1, 1)
    inputs  = y[0:L-1], targets = y[1:L]
    """
    t = np.arange(seq_len, dtype=np.float32)

    if phase_randomize:
        fast_phases = rng.uniform(0, 2 * np.pi, size=(batch_size, 1))
        slow_phases = rng.uniform(0, 2 * np.pi, size=(batch_size, 1))
    else:
        fast_phases = np.zeros((batch_size, 1))
        slow_phases = np.zeros((batch_size, 1))

    # A(t): slow envelope
    slow_mod = base_amplitude + modulation_depth * np.sin(
        2 * np.pi * t[None, :] / T_slow + slow_phases
    )  # (batch, seq_len)

    # y_t: AM signal
    y = slow_mod * np.sin(
        2 * np.pi * t[None, :] / T_fast + fast_phases
    ).astype(np.float32)  # (batch, seq_len)

    noise = rng.normal(0, noise_std, size=(batch_size, seq_len)).astype(np.float32)
    y = y + noise

    inputs  = y[:, :-1, None]   # (batch, seq_len-1, 1)
    targets = y[:, 1:,  None]   # (batch, seq_len-1, 1)
    return inputs, targets


# ── HierarchicalWave sequence model ──────────────────────────────────────────

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
        return x  # (batch, seq_len, d_model)


# ── Regression model factories ────────────────────────────────────────────────

def make_gated_wave_regressor(d_model, n_layers, n_scales,
                               theta_min=0.063, theta_max=1.257, log_theta=True):
    base = GatedWaveModel(
        d_model=d_model, n_layers=n_layers, n_scales=n_scales,
        theta_min=theta_min, theta_max=theta_max, log_theta=log_theta,
    )

    def init(rng, x):
        base_params = base.init(rng, x)
        head_rng = jax.random.fold_in(rng, 99)
        head_w = jax.random.normal(head_rng, (d_model, 1)) * 0.01
        head_b = jnp.zeros(1)
        return {"base": base_params, "head_w": head_w, "head_b": head_b}

    def apply(params, x):
        h = base.apply(params["base"], x)          # (batch, seq, d_model)
        return h @ params["head_w"] + params["head_b"]  # (batch, seq, 1)

    return init, apply


def make_hierarchical_regressor(d_model, n_layers, n_scales):
    base = HierarchicalWaveSeqModel(
        d_model=d_model, n_layers=n_layers, n_scales=n_scales,
    )

    def init(rng, x):
        base_params = base.init(rng, x)
        head_rng = jax.random.fold_in(rng, 99)
        head_w = jax.random.normal(head_rng, (d_model, 1)) * 0.01
        head_b = jnp.zeros(1)
        return {"base": base_params, "head_w": head_w, "head_b": head_b}

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
        head_b = jnp.zeros(1)
        return {"base": base_params, "head_w": head_w, "head_b": head_b}

    def apply(params, x):
        h = base.apply(params["base"], x)
        return h @ params["head_w"] + params["head_b"]

    return init, apply


# ── Training loop ─────────────────────────────────────────────────────────────

def train_and_eval(
    name: str,
    init_fn,
    apply_fn,
    rng_seed: int = 42,
    seq_len: int = 98,
    batch_size: int = 64,
    n_steps: int = 2000,
    report_interval: int = 500,
    lr: float = 1e-3,
    T_fast: int = 7,
    T_slow: int = 49,
    warmup_steps: int = 49,
) -> dict:
    rng = jax.random.PRNGKey(rng_seed)
    np_rng = np.random.default_rng(rng_seed)

    x_dummy = jnp.zeros((batch_size, seq_len - 1, 1))
    params = init_fn(rng, x_dummy)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, x, y):
        preds = apply_fn(params, x)
        return jnp.mean((preds - y) ** 2)

    @jax.jit
    def step_fn(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, new_opt = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, new_opt, loss

    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")

    final_mse = warmup_mse = post_mse = 0.0

    for s in range(n_steps):
        x_np, y_np = generate_am_prediction_batch(
            np_rng, batch_size, seq_len, T_fast, T_slow
        )
        params, opt_state, loss = step_fn(
            params, opt_state, jnp.array(x_np), jnp.array(y_np)
        )

        if (s + 1) % report_interval == 0:
            x_e, y_e = generate_am_prediction_batch(np_rng, 256, seq_len, T_fast, T_slow)
            preds = np.array(apply_fn(params, jnp.array(x_e)))
            y_e = np.array(y_e)

            total = float(np.mean((preds - y_e) ** 2))
            wu    = float(np.mean((preds[:, :warmup_steps] - y_e[:, :warmup_steps]) ** 2))
            pw    = float(np.mean((preds[:, warmup_steps:] - y_e[:, warmup_steps:]) ** 2))

            print(f"  step {s+1:4d}  loss={loss:.4f}  "
                  f"total={total:.4f}  warmup={wu:.4f}  post={pw:.4f}")
            final_mse, warmup_mse, post_mse = total, wu, pw

    return {"name": name, "mse": final_mse, "warmup_mse": warmup_mse, "post_mse": post_mse}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    D        = 32
    N_LAYERS = 2
    N_SCALES = 4
    T_FAST   = 7
    T_SLOW   = 49
    SEQ_LEN  = 98      # 2 full slow cycles
    STEPS    = 2000
    BATCH    = 64
    WARMUP   = T_SLOW  # skip first slow cycle in post-warmup MSE

    THETA_MIN = 0.063   # period ~100 (covers T_slow=49 at theta≈0.128)
    THETA_MAX = 1.257   # period ~5   (covers T_fast=7 at theta≈0.898)

    print(f"\nH15: Hierarchical Sequence Prediction (H14 calibration test)")
    print(f"Task: y_t = A(t) * sin(2π*t/{T_FAST} + φ) + noise(0.3)")
    print(f"      A(t) = 0.5 + 0.4 * sin(2π*t/{T_SLOW} + φ')")
    print(f"Phases randomized per sequence. L={SEQ_LEN}, d={D}, {N_LAYERS}L, {N_SCALES} scales.")
    print(f"T_fast theta≈{2*np.pi/T_FAST:.3f}, T_slow theta≈{2*np.pi/T_SLOW:.3f}")
    print(f"Theta [{THETA_MIN},{THETA_MAX}] log-spaced → periods ≈ [100, 33, 11, 5]")
    print(f"Noise floor (irreducible MSE): {0.3**2:.4f}")
    print(f"\nH14 prediction:")
    print(f"  minGRU:           high MSE (can't compose AM structure)")
    print(f"  GatedWave:        lower MSE (tracks both; gate must compose via prev_h)")
    print(f"  HierarchicalWave: lowest MSE (write-coupling directly modulates fast state)")
    print(f"  GW→HW gap >15% MSE reduction → expect substantial Config E BPC improvement")
    print(f"  GW→HW gap <5%               → gate conditioning sufficient; little improvement")

    configs = [
        ("GatedWave (read-coupling, log-theta)",
         *make_gated_wave_regressor(D, N_LAYERS, N_SCALES, THETA_MIN, THETA_MAX, True)),
        ("HierarchicalWave (write-coupling)",
         *make_hierarchical_regressor(D, N_LAYERS, N_SCALES)),
        ("minGRU (baseline)",
         *make_mingru_regressor(D, N_LAYERS)),
    ]

    results = []
    for name, init_fn, apply_fn in configs:
        r = train_and_eval(
            name, init_fn, apply_fn,
            seq_len=SEQ_LEN, batch_size=BATCH,
            n_steps=STEPS, report_interval=500,
            T_fast=T_FAST, T_slow=T_SLOW,
            warmup_steps=WARMUP,
        )
        results.append(r)

    print(f"\n{'═'*65}")
    print(f"  H15 RESULTS SUMMARY  (noise floor = {0.3**2:.4f})")
    print(f"{'═'*65}")
    print(f"  {'Model':<42} {'Total':>6}  {'Warmup':>6}  {'Post':>6}")
    print(f"  {'─'*65}")
    for r in results:
        print(f"  {r['name']:<42} {r['mse']:>6.4f}  "
              f"{r['warmup_mse']:>6.4f}  {r['post_mse']:>6.4f}")

    gw, hw, mg = results[0], results[1], results[2]
    gw_hw_pct = 100 * (gw["post_mse"] - hw["post_mse"]) / max(gw["post_mse"], 1e-9)
    mg_gw_pct = 100 * (mg["post_mse"] - gw["post_mse"]) / max(mg["post_mse"], 1e-9)

    print(f"\n  GatedWave → HierarchicalWave reduction: {gw_hw_pct:+.1f}%")
    print(f"  minGRU   → GatedWave reduction:         {mg_gw_pct:+.1f}%")

    print(f"\n  H14 calibration for Config E:")
    if gw_hw_pct > 15:
        print(f"  → LARGE gap: expect Config E substantial BPC improvement (>0.05 BPC)")
    elif gw_hw_pct > 5:
        print(f"  → MODERATE gap: expect Config E modest improvement (0.01–0.05 BPC)")
    elif abs(gw_hw_pct) <= 5:
        print(f"  → SMALL gap: gate conditioning sufficient here; Config E ~neutral")
    else:
        print(f"  → REVERSED: GW beats HW — investigate potential design confound")


if __name__ == "__main__":
    main()
