import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from gated_wave import GatedWaveDynamicsLayer


class SoftHierarchicalCoupling(nn.Module):
    """Couples scale dimensions based on frequency proximity.

    Uses theta (frequency) values for pairwise distances — much better
    differentiation than r_values (which span only 0.9→1.0).
    """
    d_model: int
    n_scales: int

    @nn.compact
    def __call__(self, h_scales: jnp.ndarray, scale_values: jnp.ndarray):
        # h_scales: (batch, seq_len, d_model * 2) — complex state as real/imag
        # scale_values: (n_scales,) — e.g. theta values for each scale
        batch, seq_len, d_total = h_scales.shape
        d_per_scale = d_total // (self.n_scales * 2)

        # (batch, seq_len, n_scales, 2, d_per_scale)
        h_reshaped = h_scales.reshape(batch, seq_len, self.n_scales, 2, d_per_scale)

        # Pairwise distances in frequency space
        scales_diff = jnp.abs(scale_values[:, None] - scale_values[None, :])

        # Learnable bandwidth — initialised to spread matching typical theta range
        sigma = self.param("sigma", nn.initializers.constant(0.3), (1,))
        coupling_matrix = jnp.exp(-scales_diff / (jnp.abs(sigma) + 1e-6))

        # Row-normalise so each scale's coupled state is a weighted average
        coupling_matrix = coupling_matrix / jnp.sum(coupling_matrix, axis=-1, keepdims=True)

        # Mix across scales: (n_scales, n_scales) × (batch, seq, n_scales, 2, d)
        h_coupled = jnp.einsum('ij,bsj...->bsi...', coupling_matrix, h_reshaped)

        return h_coupled.reshape(batch, seq_len, d_total)


class HierarchicalWaveLayer(nn.Module):
    """GatedWaveDynamicsLayer + SoftHierarchicalCoupling.

    Adds state-side mixing between scales based on theta-frequency proximity.
    Compatible with GatedWaveModel's interface.
    """
    d_model: int = 128
    n_scales: int = 4
    r_min: float = 0.9
    r_max: float = 0.999
    theta_min: float = 0.063
    theta_max: float = 1.257
    log_theta: bool = True

    @nn.compact
    def __call__(self, x, conditioning=None):
        # Run base oscillatory dynamics
        h_uncoupled = GatedWaveDynamicsLayer(
            d_model=self.d_model,
            n_scales=self.n_scales,
            r_min=self.r_min,
            r_max=self.r_max,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            log_theta=self.log_theta,
            name="dynamics",
        )(x, conditioning=conditioning)

        # Build theta values that match GatedWaveDynamicsLayer's internal schedule
        # Scale 0: integrator (r=1.0); use theta_min/10 as low-freq proxy for coupling
        if self.log_theta:
            osc_thetas = np.exp(
                np.linspace(np.log(self.theta_min), np.log(self.theta_max), self.n_scales - 1)
            )
        else:
            osc_thetas = np.linspace(self.theta_min, self.theta_max, self.n_scales - 1)
        all_thetas = jnp.array(
            [self.theta_min / 10.0] + list(osc_thetas), dtype=jnp.float32
        )

        # Apply hierarchical coupling (state-side mixing)
        h_coupled = SoftHierarchicalCoupling(
            d_model=self.d_model,
            n_scales=self.n_scales,
            name="coupling",
        )(h_uncoupled, all_thetas)

        return h_coupled


class HierarchicalWaveModel(nn.Module):
    """Multi-layer HierarchicalWave sequence model.

    Drop-in replacement for GatedWaveModel with added state-side coupling.
    Same interface: (batch, seq_len, d_in) → (batch, seq_len, d_model).
    """
    d_model: int = 128
    n_layers: int = 4
    n_scales: int = 4
    r_min: float = 0.9
    r_max: float = 0.999
    theta_min: float = 0.063
    theta_max: float = 1.257
    log_theta: bool = True

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
                theta_min=self.theta_min,
                theta_max=self.theta_max,
                log_theta=self.log_theta,
                name=f"layer_{i}",
            )(x, conditioning=prev_h)

            prev_h = h_complex
            x = nn.Dense(self.d_model, name=f"out_proj_{i}")(h_complex)
            x = x + residual

        return x
