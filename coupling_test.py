import numpy as np

class CoupledDimension:
    def __init__(self, tau, omega, alpha, idx):
        self.h = 0.0
        self.v = 0.0
        self.tau = tau
        self.omega = omega
        self.alpha = alpha
        self.idx = idx
    
    def update_state(self, input_val):
        # Standard wave dynamics update
        v_gate = np.exp(-self.tau)
        accel = -self.alpha * self.v - (self.omega ** 2) * self.h + input_val
        self.v = v_gate * self.v + accel
        self.h = self.h + self.v
        self.h = np.clip(self.h, -10, 10)
        return self.h

def run_coupled_experiment():
    length = 150
    reset_period = 3
    
    # Generate task
    inputs = []
    targets = []
    current_sum = 0
    for i in range(length):
        val = (i % 2) + 1
        if i % reset_period == 0:
            current_sum = 0
        current_sum += val
        inputs.append(val)
        targets.append(current_sum)
    inputs = np.array(inputs)
    targets = np.array(targets)

    # Initialize dimensions with diverse timescales
    taus = [0.5, 1.0, 2.0, 5.0]
    omegas = [0.5, 1.0, 1.5, 2.0]
    dimensions = []
    for t in taus:
        for o in omegas:
            dimensions.append(CoupledDimension(tau=t, omega=o, alpha=1.0, idx=len(dimensions)))
    
    n_dims = len(dimensions)
    results = np.zeros((n_dims, length))
    
    # Coupling matrix: weight based on tau distance
    # Closer timescales influence each other more
    coupling_matrix = np.zeros((n_dims, n_dims))
    for i in range(n_dims):
        for j in range(n_dims):
            dist = abs(dimensions[i].tau - dimensions[j].tau)
            coupling_matrix[i, j] = np.exp(-dist) # RBF-like coupling
    
    # Normalize coupling
    coupling_matrix /= np.sum(coupling_matrix, axis=1, keepdims=True)
    
    coupling_strength = 0.1 # How much dimensions mix
    
    for t in range(length):
        # 1. Independent update
        current_h = np.zeros(n_dims)
        for i, d in enumerate(dimensions):
            current_h[i] = d.update_state(inputs[t])
        
        # 2. Coupling step (mixing states)
        mixed_h = (1 - coupling_strength) * current_h + coupling_strength * (coupling_matrix @ current_h)
        
        # 3. Apply mixed state back to dimensions
        for i, d in enumerate(dimensions):
            d.h = mixed_h[i]
            results[i, t] = mixed_h[i]

    # Best fit readout
    def get_best_fit(results, targets, train_limit=50):
        best_mse = float('inf')
        best_pred = None
        for i in range(results.shape[0]):
            scale = np.mean(targets[:train_limit] / (results[i, :train_limit] + 1e-6))
            pred = results[i, :] * scale
            mse = np.mean((pred[train_limit:] - targets[train_limit:])**2)
            if mse < best_mse:
                best_mse = mse
                best_pred = pred
        return best_pred, best_mse

    coupled_pred, coupled_mse = get_best_fit(results, targets)
    
    print(f"Coupled Wave Dynamics MSE (extrapolation): {coupled_mse:.4f}")
    
    # Last 20 steps
    print("\nTarget vs Coupled Prediction (last 20 steps):")
    print("Step | Target | Coupled")
    for t in range(length-20, length):
        print(f"{t:4d} | {targets[t]:6.1f} | {coupled_pred[t]:6.1f}")

if __name__ == "__main__":
    run_coupled_experiment()
