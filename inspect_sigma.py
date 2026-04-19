import pickle
import jax

def find_sigma(params, path=""):
    if isinstance(params, dict):
        for k, v in params.items():
            res = find_sigma(v, path + "/" + k)
            if res is not None:
                return res
    elif path.endswith("sigma"):
        return params

with open('hierarchical_wave_params.pkl', 'rb') as f:
    p = pickle.load(f)

sigma = find_sigma(p['params'])
print(f"Sigma: {sigma}")
