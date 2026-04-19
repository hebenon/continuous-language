import pickle
import jax

def print_params(params, prefix=""):
    if isinstance(params, dict):
        for k, v in params.items():
            print(f"{prefix}{k}")
            print_params(v, prefix + "  ")

with open('hierarchical_wave_params.pkl', 'rb') as f:
    p = pickle.load(f)

print("Parameter Structure:")
print_params(p['params'])
