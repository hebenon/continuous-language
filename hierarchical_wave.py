import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Optional, Tuple
from gated_wave import parallel_scan

class SoftHierarchicalCoupling(nn.Module):
    """Couples dimensions based on timescale proximity."""
    d_model: int
    n_scales: int
    
    @nn.compact
    def __call__(self, h_scales: jnp.ndarray, r_values: jnp.ndarray):
        # h_scales: (batch, seq_len, d_model * 2) - complex state as real/imag
        # r_values: (n_scales,) - decay coefficients
        
        batch, seq_len, d_total = h_scales.shape
        d_per_scale = d_total // (self.n_scales * 2)
        
        # Reshape to (batch, seq_len, n_scales, 2, d_per_scale)
        # 2 represents real and imaginary parts
        h_reshaped = h_scales.reshape(batch, seq_len, self.n_scales, 2, d_per_scale)
        
        # Compute pairwise distance between scales based on r
        # scales_diff: (n_scales, n_scales)
        scales_diff = jnp.abs(r_values[:, None] - r_values[None, :])
        
        # Kernel: exp(-dist / sigma)
        # sigma is learnable
        sigma = self.param("sigma", nn.initializers.constant(0.1), (1,))
        coupling_matrix = jnp.exp(-scales_diff / (sigma + 1e-6))
        
        # Normalize coupling matrix (each row sums to 1)
        coupling_matrix = coupling_matrix / jnp.sum(coupling_matrix, axis=-1, keepdims=True)
        
        # Apply coupling across scales
        # h_reshaped is (batch, seq_len, n_scales, 2, d_per_scale)
        # coupling_matrix is (n_scales, n_scales)
        # Result: (batch, seq_len, n_scales, 2, d_per_scale)
        h_coupled = jnp.einsum('ij,bsj...->bsi...', coupling_matrix, h_reshaped)
        
        return h_coupled.reshape(batch, seq_len, d_total)

class HierarchicalWaveLayer(nn.Module):
    d_model: int = 128
    n_scales: int = 2
    r_min: float = 0.9
    r_max: float = 0.999
    
    @nn.compact
    def __call__(self, x, conditioning=None):
        batch, seq_len, d_in = x.shape
        
        # Standard Gated Wave Logic (simplified for brevity)
        # In a real impl, I'd subclass or refactor GatedWaveDynamicsLayer
        from gated_wave import GatedWaveDynamicsLayer
        
        base_layer = GatedWaveDynamicsLayer(
            d_model=self.d_model, 
            n_scales=self.n_scales,
            r_min=self.r_min,
            r_max=self.r_max
        )
        
        # Get uncoupled states
        h_uncoupled = base_layer(x, conditioning=conditioning)
        
        # Apply hierarchical coupling
        r_values = jnp.array([1.0] + list(np.linspace(self.r_max, self.r_min, self.n_scales - 1)))
        h_coupled = SoftHierarchicalCoupling(
            d_model=self.d_model, 
            n_scales=self.n_scales
        )(h_uncoupled, r_values)
        
        return h_coupled

