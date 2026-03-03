import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Optional, Tuple

def parallel_scan(a, b):
    """Associative scan for linear recurrence: h_t = a_t * h_{t-1} + b_t.
    
    Args:
        a: (batch, seq_len, d_model) - decay/rotation coefficients
        b: (batch, seq_len, d_model) - driving force / input
    Returns:
        h: (batch, seq_len, d_model) - hidden states
    """
    def binary_op(left, right):
        a_left, b_left = left
        a_right, b_right = right
        return a_left * a_right, a_right * b_left + b_right

    _, h = jax.lax.associative_scan(binary_op, (a, b), axis=1)
    return h

class GatedWaveDynamicsLayer(nn.Module):
    """Gated Wave Dynamics layer.
    
    Each dimension is a discrete damped oscillator with an input-dependent reset gate.
    s_t = (1 - g_t) * a * s_{t-1} + b_t
    
    where g_t = sigmoid(Linear(x_t)) is the reset gate.
    """
    d_model: int = 128
    n_scales: int = 2
    r_min: float = 0.9
    r_max: float = 0.999
    theta_min: float = 0.01
    theta_max: float = 1.0
    
    @nn.compact
    def __call__(self, x, conditioning=None):
        batch, seq_len, d_in = x.shape
        
        # If no conditioning provided, use x itself
        if conditioning is None:
            conditioning = x
            
        d_per_scale = self.d_model // self.n_scales
        remainder = self.d_model % self.n_scales
        
        if self.n_scales == 1:
            r_values = [(self.r_min + self.r_max) / 2]
            theta_values = [(self.theta_min + self.theta_max) / 2]
        else:
            # Scale 0 is always a pure integrator (long-term memory / accumulator)
            r_values = [1.0] + list(np.linspace(self.r_max, self.r_min, self.n_scales - 1))
            theta_values = [0.0] + list(np.linspace(self.theta_min, self.theta_max, self.n_scales - 1))
        
        h_scales = []
        for i in range(self.n_scales):
            d_this_scale = d_per_scale + (remainder if i == self.n_scales - 1 else 0)
            
            r = float(r_values[i])
            theta = float(theta_values[i])
            
            # 1. Fixed base coefficient
            a_base = r * jnp.exp(1j * theta)
            
            # 2. Input-dependent reset gate (MLP for more expressivity)
            g1 = nn.Dense(d_this_scale, name=f"gate_1_{i}")(conditioning)
            g2 = nn.relu(g1)
            gate_logits = nn.Dense(d_this_scale, name=f"gate_2_{i}", 
                                   bias_init=nn.initializers.constant(-4.0))(g2)
            gate = nn.sigmoid(gate_logits)
            
            # 3. Gated coefficient
            # a_t = (1 - gate_t) * a_base
            a = (1.0 - gate).astype(jnp.complex64) * a_base
            
            # 4. Input-dependent driving force
            drive_re = nn.Dense(d_this_scale, name=f"drive_re_{i}")(x)
            drive_im = nn.Dense(d_this_scale, name=f"drive_im_{i}")(x)
            b = (drive_re + 1j * drive_im).astype(jnp.complex64)
            # Normalize input gain. For oscillatory scales (r<1): (1-r) keeps state bounded.
            # For the integrator scale (r=1): (1-r)=0 would zero the input — use 1/sqrt(d)
            # instead so the accumulator actually receives a signal.
            if r >= 1.0:
                b = b / jnp.sqrt(float(d_this_scale))
            else:
                b = b * (1 - r)
            
            # 5. Parallel scan with gated input
            b_gated = (1.0 - gate).astype(jnp.complex64) * b
            state = parallel_scan(a, b_gated)
            
            # Extract real and imag parts for richer conditioning
            h_scales.append(state.real)
            h_scales.append(state.imag)
        
        return jnp.concatenate(h_scales, axis=-1)

class GatedWaveModel(nn.Module):
    d_model: int = 128
    n_layers: int = 4
    n_scales: int = 2
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.d_model, name="input_proj")(x)
        
        # Initial conditioning is just the input
        # We repeat it to match the expected d_model * 2 (real + imag)
        prev_h = jnp.concatenate([x, jnp.zeros_like(x)], axis=-1)
        
        for i in range(self.n_layers):
            residual = x
            x = nn.LayerNorm(name=f"ln_{i}")(x)
            
            # Layer output is now d_model * 2 due to real+imag
            h_complex = GatedWaveDynamicsLayer(d_model=self.d_model, n_scales=self.n_scales, name=f"layer_{i}")(x, conditioning=prev_h)
            
            # For the next layer, use the full complex state
            prev_h = h_complex
            
            # For the residual path, we need to project back to d_model
            x = nn.Dense(self.d_model, name=f"out_proj_{i}")(h_complex)
            x = x + residual
        return x

def test_gated_wave():
    model = GatedWaveModel(d_model=64, n_layers=2, n_scales=4)
    x = jnp.ones((1, 100, 16))
    params = model.init(jax.random.PRNGKey(0), x)
    out = model.apply(params, x)
    print(f"Output shape: {out.shape}")
    assert out.shape == (1, 100, 64)
    print("Test passed!")

if __name__ == "__main__":
    test_gated_wave()
