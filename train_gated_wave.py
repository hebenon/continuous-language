import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from gated_wave import GatedWaveModel

def generate_repeating_sum(length=200, reset_period=5, max_len=300):
    inputs = []
    targets = []
    current_sum = 0
    for i in range(length):
        val = (i % 2) + 1
        if i % reset_period == 0:
            current_sum = 0
        current_sum += val
        inputs.append([float(val)])
        targets.append([float(current_sum)])
    
    # Padding
    actual_len = len(inputs)
    for _ in range(max_len - actual_len):
        inputs.append([0.0])
        targets.append([0.0])
        
    return jnp.array([inputs]), jnp.array([targets]), actual_len

def train():
    max_len = 300
    reset_period = 7
    
    model = GatedWaveModel(d_model=128, n_layers=4, n_scales=4)
    key = jax.random.PRNGKey(42)
    
    # Initial dummy input for init
    dummy_in = jnp.zeros((1, max_len, 1))
    
    # We need a readout head
    class Readout(nn.Module):
        model: nn.Module
        @nn.compact
        def __call__(self, x):
            x = self.model(x)
            return nn.Dense(1)(x)
            
    full_model = Readout(model=model)
    params = full_model.init(key, dummy_in)
    
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def train_step(params, opt_state, x, y, actual_len):
        def loss_fn(params):
            pred = full_model.apply(params, x)
            # Only compute loss on non-padded part
            mask = jnp.arange(max_len) < actual_len
            mask = mask[None, :, None]
            return jnp.sum(((pred - y)**2) * mask) / jnp.sum(mask)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for step in range(5000):
        # Variable length training
        curr_len = np.random.randint(100, 250)
        inputs, targets, actual_len = generate_repeating_sum(curr_len, reset_period, max_len)
        
        params, opt_state, loss = train_step(params, opt_state, inputs, targets, actual_len)
        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss:.6f}", flush=True)
            
    # Test extrapolation
    test_len = 300
    test_inputs, test_targets, _ = generate_repeating_sum(test_len, reset_period, max_len)
    pred = full_model.apply(params, test_inputs)
    
    print("\nExtrapolation Test (last 10 steps):")
    for t in range(test_len-10, test_len):
        print(f"Step {t}: Target {test_targets[0, t, 0]:.1f}, Pred {pred[0, t, 0]:.1f}")

if __name__ == "__main__":
    train()
