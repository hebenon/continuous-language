import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import pickle
import sys
import os

# Add current dir to path
sys.path.append(os.getcwd())

from hierarchical_wave import HierarchicalWaveLayer

def load_tiny_shakespeare(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    return vocab_size, char_to_ix, ix_to_char

class HierarchicalWaveModel(nn.Module):
    d_model: int = 128
    n_layers: int = 4
    n_scales: int = 2
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.d_model, name="input_proj")(x)
        
        # Initial conditioning
        prev_h = jnp.concatenate([x, jnp.zeros_like(x)], axis=-1)
        
        for i in range(self.n_layers):
            residual = x
            x = nn.LayerNorm(name=f"ln_{i}")(x)
            
            # Use HierarchicalWaveLayer
            h_complex = HierarchicalWaveLayer(
                d_model=self.d_model, 
                n_scales=self.n_scales, 
                name=f"layer_{i}"
            )(x, conditioning=prev_h)
            
            prev_h = h_complex
            x = nn.Dense(self.d_model, name=f"out_proj_{i}")(h_complex)
            x = x + residual
        return x

class HierarchicalCharLM(nn.Module):
    vocab_size: int
    d_model: int
    n_layers: int
    n_scales: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.d_model)(x)
        model = HierarchicalWaveModel(
            d_model=self.d_model, 
            n_layers=self.n_layers, 
            n_scales=self.n_scales
        )
        x = model(x)
        logits = nn.Dense(self.vocab_size)(x)
        return logits

def generate():
    data_path = "/home/meridian/projects/continuous-language/data/shakespeare/tiny.txt"
    vocab_size, char_to_ix, ix_to_char = load_tiny_shakespeare(data_path)
    
    # MATCH TRAINING PARAMS
    d_model = 32
    n_layers = 2
    n_scales = 4
    seq_len = 32
    
    model = HierarchicalCharLM(
        vocab_size=vocab_size, 
        d_model=d_model, 
        n_layers=n_layers,
        n_scales=n_scales
    )
    
    params_path = "/home/meridian/projects/continuous-language/hierarchical_wave_params.pkl"
    with open(params_path, "rb") as f:
        params = pickle.load(f)
        
    seed_text = "First Citizen:"
    context = [char_to_ix[ch] for ch in seed_text]
    temperature = 0.8
    key = jax.random.PRNGKey(123)
    
    print(f"Generating 500 characters with Hierarchical Wave (T={temperature})...")
    
    generated_text = seed_text
    for _ in range(100):
        # Slice context to match training seq_len
        curr_context = context[-seq_len:]
        curr_x = jax.nn.one_hot(jnp.array([curr_context]), vocab_size)
        logits = model.apply(params, curr_x)
        next_char_logits = logits[0, -1, :]
        
        key, subkey = jax.random.split(key)
        next_char_ix = jax.random.categorical(subkey, next_char_logits / temperature)
        context.append(int(next_char_ix))
        generated_text += ix_to_char[int(next_char_ix)]
        
    print("\n--- GENERATED SAMPLE ---")
    print(generated_text)
    print("------------------------")

if __name__ == "__main__":
    generate()
