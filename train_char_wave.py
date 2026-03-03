import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from gated_wave import GatedWaveModel

# Character level setup
def load_tiny_shakespeare(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    data = [char_to_ix[ch] for ch in text]
    return np.array(data), vocab_size, char_to_ix, ix_to_char

class GatedWaveCharLM(nn.Module):
    vocab_size: int
    d_model: int
    n_layers: int
    n_scales: int

    @nn.compact
    def __call__(self, x):
        # x is (batch, seq_len, vocab_size) - one-hot encoded
        # Project one-hot to d_model
        x = nn.Dense(self.d_model)(x)
        
        # Gated Wave Core
        model = GatedWaveModel(
            d_model=self.d_model, 
            n_layers=self.n_layers, 
            n_scales=self.n_scales
        )
        x = model(x)
        
        # Readout to logits
        logits = nn.Dense(self.vocab_size)(x)
        return logits

def get_batch(data, vocab_size, seq_len=64, batch_size=4):
    ix = np.random.randint(0, len(data) - seq_len, batch_size)
    x = np.stack([data[i:i+seq_len] for i in ix])
    y = np.stack([data[i+1:i+seq_len+1] for i in ix])
    
    # One-hot encoding for inputs
    x_one_hot = jax.nn.one_hot(x, vocab_size)
    return jnp.array(x_one_hot), jnp.array(y)

def train():
    data_path = "/home/meridian/projects/continuous-language/data/shakespeare/tiny.txt"
    data, vocab_size, char_to_ix, ix_to_char = load_tiny_shakespeare(data_path)
    print(f"Vocab size: {vocab_size}")
    
    seq_len = 32
    batch_size = 2 # Small batch for Pi
    d_model = 32   # Smaller model for Pi
    n_layers = 2
    n_scales = 2
    
    model = GatedWaveCharLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_scales=n_scales
    )
    
    key = jax.random.PRNGKey(42)
    dummy_x = jnp.zeros((batch_size, seq_len, vocab_size))
    params = model.init(key, dummy_x)
    
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        def loss_fn(params):
            logits = model.apply(params, x)
            # Softmax cross entropy
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
            return jnp.mean(loss)
            
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    print("Starting training on Pi...")
    for step in range(3001):
        x, y = get_batch(data, vocab_size, seq_len, batch_size)
        params, opt_state, loss = train_step(params, opt_state, x, y)
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")
            
                # Simple generation check
            if step % 500 == 0:
                gen_len = 100
                seed_text = "First"
                context = [char_to_ix[ch] for ch in seed_text]
                
                # Temperature sampling
                temperature = 0.8
                
                for _ in range(gen_len):
                    curr_x = jax.nn.one_hot(jnp.array([context[-seq_len:]]), vocab_size)
                    logits = model.apply(params, curr_x)
                    next_char_logits = logits[0, -1, :]
                    
                    key, subkey = jax.random.split(key)
                    next_char_ix = jax.random.categorical(subkey, next_char_logits / temperature)
                    context.append(int(next_char_ix))
                
                generated = "".join([ix_to_char[i] for i in context])
                print(f"--- Sample (T={temperature}): {generated}")
                
    # Save parameters after training
    import pickle
    with open("/home/meridian/projects/continuous-language/char_wave_params.pkl", "wb") as f:
        pickle.dump(params, f)
    print("Training complete. Parameters saved.")

if __name__ == "__main__":
    train()
