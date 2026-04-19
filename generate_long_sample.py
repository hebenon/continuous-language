import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import pickle
from gated_wave import GatedWaveModel

# Setup matches train_char_wave.py
def load_tiny_shakespeare(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    return vocab_size, char_to_ix, ix_to_char

class GatedWaveCharLM(nn.Module):
    vocab_size: int
    d_model: int
    n_layers: int
    n_scales: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.d_model)(x)
        model = GatedWaveModel(d_model=self.d_model, n_layers=self.n_layers, n_scales=self.n_scales)
        x = model(x)
        logits = nn.Dense(self.vocab_size)(x)
        return logits

def generate():
    data_path = "/home/meridian/projects/continuous-language/data/shakespeare/tiny.txt"
    vocab_size, char_to_ix, ix_to_char = load_tiny_shakespeare(data_path)
    
    d_model = 32
    n_layers = 2
    n_scales = 2
    seq_len = 32
    
    model = GatedWaveCharLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_scales=n_scales)
    
    with open("/home/meridian/projects/continuous-language/char_wave_params.pkl", "rb") as f:
        params = pickle.load(f)
        
    seed_text = "First Citizen:"
    context = [char_to_ix[ch] for ch in seed_text]
    temperature = 0.7
    key = jax.random.PRNGKey(123)
    
    print(f"Generating 500 characters with T={temperature}...")
    
    generated_text = seed_text
    for _ in range(500):
        curr_x = jax.nn.one_hot(jnp.array([context[-seq_len:]]), vocab_size)
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
