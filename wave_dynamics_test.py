import numpy as np

class WaveDimension:
    def __init__(self, tau, omega, alpha):
        self.h = 0.0
        self.v = 0.0
        self.tau = tau
        self.omega = omega
        self.alpha = alpha
    
    def step(self, input_val):
        # Gated velocity update
        v_gate = np.exp(-self.tau)
        # Acceleration = damping + restoring force + drive
        accel = -self.alpha * self.v - (self.omega ** 2) * self.h + input_val
        self.v = v_gate * self.v + accel
        self.h = self.h + self.v
        # Clip to prevent explosion (as per roadmap)
        self.h = np.clip(self.h, -10, 10) 
        return self.h

class MinGRUDimension:
    def __init__(self, tau):
        self.h = 0.0
        self.tau = tau
    
    def step(self, input_val):
        # dh/dt = -h/tau + input
        # h_{t+1} = h_t + (-h_t/tau + input)
        #         = (1 - 1/tau) * h_t + input
        gate = np.exp(-1.0/self.tau)
        self.h = gate * self.h + input_val
        return self.h

def generate_task(length=200, reset_period=3):
    inputs = []
    outputs = []
    current_sum = 0
    for i in range(length):
        val = (i % 2) + 1 # [1, 2, 1, 2, ...]
        if i % reset_period == 0:
            current_sum = 0
        current_sum += val
        inputs.append(val)
        outputs.append(current_sum)
    return np.array(inputs), np.array(outputs)

def run_experiment():
    length = 150
    reset_period = 3
    inputs, targets = generate_task(length, reset_period)
    
    # Initialize models
    # We'll use a small "ensemble" of dimensions to see if any can track the pattern
    # For Wave: search for (tau, omega) that works
    # For MinGRU: search for tau that works
    
    wave_models = []
    for tau in [0.5, 1.0, 2.0]:
        for omega in [0.5, 1.0, 1.5, 2.0]:
            wave_models.append(WaveDimension(tau=tau, omega=omega, alpha=1.0))
            
    mingru_models = []
    for tau in [0.5, 1.0, 2.0, 5.0, 10.0]:
        mingru_models.append(MinGRUDimension(tau=tau))
        
    wave_results = np.zeros((len(wave_models), length))
    mingru_results = np.zeros((len(mingru_models), length))
    
    for t in range(length):
        for i, m in enumerate(wave_models):
            wave_results[i, t] = m.step(inputs[t])
        for i, m in enumerate(mingru_models):
            mingru_results[i, t] = m.step(inputs[t])
            
    # Find best fit for each model type (simple linear regression on the hidden state)
    # This simulates a "readout" layer
    
    def get_best_fit(results, targets, train_limit=50):
        best_mse = float('inf')
        best_pred = None
        
        # Try each dimension as a predictor
        for i in range(results.shape[0]):
            # Simple scaling factor
            scale = np.mean(targets[:train_limit] / (results[i, :train_limit] + 1e-6))
            pred = results[i, :] * scale
            mse = np.mean((pred[train_limit:] - targets[train_limit:])**2)
            if mse < best_mse:
                best_mse = mse
                best_pred = pred
        return best_pred, best_mse

    wave_pred, wave_mse = get_best_fit(wave_results, targets)
    mingru_pred, mingru_mse = get_best_fit(mingru_results, targets)
    
    print(f"Wave Dynamics MSE (extrapolation): {wave_mse:.4f}")
    print(f"MinGRU Baseline MSE (extrapolation): {mingru_mse:.4f}")
    
    # Simple ASCII plot for verification
    print("\nTarget vs Predictions (last 20 steps):")
    print("Step | Target | Wave | MinGRU")
    for t in range(length-20, length):
        print(f"{t:4d} | {targets[t]:6.1f} | {wave_pred[t]:6.1f} | {mingru_pred[t]:6.1f}")

if __name__ == "__main__":
    run_experiment()
