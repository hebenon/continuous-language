import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class GatedWaveDimension:
    def __init__(self, tau, omega, alpha, idx):
        self.h = 0.0
        self.v = 0.0
        self.tau = tau
        self.omega = omega
        self.alpha = alpha
        self.idx = idx
        
        # Random gating weights for "reset" detection
        # In a real model, these would be learned
        self.w_gate = np.random.randn() * 0.1
        self.b_gate = -2.0 # Bias towards not resetting
    
    def update_state(self, input_val):
        # 1. Candidate wave update
        v_gate = np.exp(-self.tau)
        accel = -self.alpha * self.v - (self.omega ** 2) * self.h + input_val
        new_v = v_gate * self.v + accel
        new_h = self.h + new_v
        
        # 2. Gated reset
        # If state gets too high, it might trigger a reset
        # Or if the input has a certain pattern
        gate = sigmoid(self.w_gate * self.h + self.b_gate)
        
        self.v = (1 - gate) * new_v
        self.h = (1 - gate) * new_h
        
        self.h = np.clip(self.h, -10, 10)
        return self.h, gate

def run_gated_experiment():
    np.random.seed(42)
    length = 150
    reset_period = 3
    
    # Generate task: Repeating Sum
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

    # Initialize dimensions
    taus = [0.1, 0.5, 1.0, 2.0]
    omegas = [0.5, 1.0, 1.5, 2.0]
    dimensions = []
    for t in taus:
        for o in omegas:
            dimensions.append(GatedWaveDimension(tau=t, omega=o, alpha=1.0, idx=len(dimensions)))
    
    n_dims = len(dimensions)
    results = np.zeros((n_dims, length))
    gates = np.zeros((n_dims, length))
    
    # We'll also try a "Learned" reset by manually setting one dimension's gate
    # to fire at the reset period, to see if the architecture CAN represent it.
    # This simulates a perfectly learned reset detector.
    dimensions[0].b_gate = 10.0 # Force reset
    # We'll modulate its "reset" by the actual reset period in the loop for testing feasibility
    
    for t in range(length):
        for i, d in enumerate(dimensions):
            # For dimension 0, we'll "cheat" to see if gating works
            if i == 0:
                if t % reset_period == 0:
                    d.b_gate = 10.0 # Trigger reset
                else:
                    d.b_gate = -10.0 # No reset
            
            h, g = d.update_state(inputs[t])
            results[i, t] = h
            gates[i, t] = g

    # Best fit readout
    def get_best_fit(results, targets, train_limit=50):
        best_mse = float('inf')
        best_pred = None
        best_idx = -1
        for i in range(results.shape[0]):
            # Linear regression for scale and shift
            X = results[i, :train_limit].reshape(-1, 1)
            y = targets[:train_limit]
            # Simple scaling for now
            scale = np.mean(targets[:train_limit] / (results[i, :train_limit] + 1e-6))
            pred = results[i, :] * scale
            mse = np.mean((pred[train_limit:] - targets[train_limit:])**2)
            if mse < best_mse:
                best_mse = mse
                best_pred = pred
                best_idx = i
        return best_pred, best_mse, best_idx

    gated_pred, gated_mse, best_idx = get_best_fit(results, targets)
    
    print(f"Gated Wave Dynamics MSE (extrapolation): {gated_mse:.4f}")
    print(f"Best dimension index: {best_idx} (Dim 0 was the 'cheated' reset)")
    
    # Last 20 steps
    print("\nTarget vs Gated Prediction (last 20 steps):")
    print("Step | Target | Gated | Gate Val")
    for t in range(length-20, length):
        print(f"{t:4d} | {targets[t]:6.1f} | {gated_pred[t]:6.1f} | {gates[best_idx, t]:.2f}")

if __name__ == "__main__":
    run_gated_experiment()
