import jax
import jax.numpy as jnp
from hierarchical_wave import HierarchicalWaveLayer

def test_coupling():
    d_model = 64
    n_scales = 4
    batch = 2
    seq_len = 16
    
    layer = HierarchicalWaveLayer(d_model=d_model, n_scales=n_scales)
    x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, 16))
    
    params = layer.init(jax.random.PRNGKey(1), x)
    out = layer.apply(params, x)
    
    print(f"Output shape: {out.shape}")
    # Output is d_model * 2 (real + imag)
    assert out.shape == (batch, seq_len, d_model * 2)
    print("Coupling test passed!")

if __name__ == "__main__":
    test_coupling()
