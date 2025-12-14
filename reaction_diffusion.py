import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    # ============================================================================
    # CELL 1: Setup
    # ============================================================================
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx
    import diffrax
    import optax
    from functools import partial
    from typing import Tuple, List, Optional, NamedTuple
    import matplotlib.pyplot as plt

    print(f"JAX devices: {jax.devices()}")
    return diffrax, eqx, jax, jnp, jr, optax, plt


@app.cell
def _(diffrax, eqx, jax, jnp, jr):
    # ============================================================================
    # Reaction-Diffusion Dynamics (Spatially Coupled)
    # ============================================================================

    class ReactionDiffusionDynamics(eqx.Module):
        """
        ODE dynamics with:
        - Reaction: local dynamics with per-dimension τ
        - Diffusion: spatial coupling via discrete Laplacian

        dh/dt = (target - h) / τ + D * ∇²h
        """
        hidden_dim: int
        dynamics_net: eqx.nn.MLP
        log_tau: jnp.ndarray
        log_diffusion: jnp.ndarray  # Learnable diffusion coefficient(s)
        freeze_tau: bool = False

        def __init__(
            self, 
            hidden_dim: int, 
            width: int, 
            depth: int, 
            freeze_tau: bool = False,
            tau_min: float = 0.1,
            tau_max: float = 20.0,
            diffusion_init: float = 0.1,
            *, 
            key
        ):
            self.hidden_dim = hidden_dim
            self.freeze_tau = freeze_tau

            keys = jr.split(key, 3)
            self.dynamics_net = eqx.nn.MLP(
                in_size=hidden_dim,
                out_size=hidden_dim,
                width_size=width,
                depth=depth,
                activation=jax.nn.tanh,
                key=keys[0]
            )

            # Log-uniform τ initialization
            log_min, log_max = jnp.log(tau_min), jnp.log(tau_max)
            self.log_tau = jr.uniform(keys[1], (hidden_dim,), minval=log_min, maxval=log_max)

            # Learnable diffusion coefficient (per-dimension or scalar)
            # Start small to not overwhelm reaction term
            self.log_diffusion = jnp.full((hidden_dim,), jnp.log(diffusion_init))

        def get_tau(self) -> jnp.ndarray:
            tau = jnp.exp(self.log_tau)
            if self.freeze_tau:
                tau = jnp.full_like(tau, tau.mean())
            return tau

        def get_diffusion(self) -> jnp.ndarray:
            return jnp.exp(self.log_diffusion)

        def __call__(self, t, state, args=None):
            # === Reaction term (local dynamics) ===
            target = self.dynamics_net(state)
            tau = self.get_tau()
            reaction = (target - state) / tau

            # === Diffusion term (spatial coupling) ===
            # Discrete Laplacian with periodic boundary
            # ∇²h[i] ≈ h[i-1] - 2*h[i] + h[i+1]
            D = self.get_diffusion()
            h_left = jnp.roll(state, 1)   # h[i-1]
            h_right = jnp.roll(state, -1) # h[i+1]
            laplacian = h_left - 2*state + h_right
            diffusion = D * laplacian

            # === Combined dynamics ===
            dh_dt = reaction + diffusion

            # Soft clip for stability
            return jnp.tanh(dh_dt) * 5.0


    class ReactionDiffusionLM(eqx.Module):
        """Language model with reaction-diffusion dynamics."""
        hidden_dim: int
        vocab_size: int
        dynamics: ReactionDiffusionDynamics
        embedding: eqx.nn.Embedding
        input_gate: eqx.nn.Linear
        output_proj: eqx.nn.Linear
        freeze_tau: bool

        def __init__(
            self, 
            vocab_size: int,
            hidden_dim: int = 128,
            dynamics_width: int = 256,
            dynamics_depth: int = 2,
            freeze_tau: bool = False,
            tau_min: float = 0.1,
            tau_max: float = 20.0,
            diffusion_init: float = 0.1,
            *, 
            key
        ):
            keys = jr.split(key, 4)

            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim
            self.freeze_tau = freeze_tau

            self.dynamics = ReactionDiffusionDynamics(
                hidden_dim, 
                dynamics_width, 
                dynamics_depth,
                freeze_tau=freeze_tau,
                tau_min=tau_min,
                tau_max=tau_max,
                diffusion_init=diffusion_init,
                key=keys[0]
            )

            self.embedding = eqx.nn.Embedding(vocab_size, hidden_dim, key=keys[1])
            self.input_gate = eqx.nn.Linear(hidden_dim * 2, hidden_dim, key=keys[2])
            self.output_proj = eqx.nn.Linear(hidden_dim, vocab_size, key=keys[3])

        def inject_token(self, state: jnp.ndarray, token: int) -> jnp.ndarray:
            token_emb = self.embedding(token)
            gate_input = jnp.concatenate([state, token_emb])
            gate = jax.nn.sigmoid(self.input_gate(gate_input))
            return state + 0.5 * gate * token_emb

        def get_emission_logits(self, state: jnp.ndarray) -> jnp.ndarray:
            return self.output_proj(state)

        def evolve(
            self, 
            state: jnp.ndarray, 
            t_start: float, 
            t_end: float,
            dt0: float = 0.05,
            max_steps: int = 256
        ) -> jnp.ndarray:
            term = diffrax.ODETerm(self.dynamics)
            solver = diffrax.Heun()
            stepsize_controller = diffrax.PIDController(rtol=1e-2, atol=1e-2)

            sol = diffrax.diffeqsolve(
                term, solver,
                t0=t_start, t1=t_end, dt0=dt0, y0=state,
                saveat=diffrax.SaveAt(t1=True),
                stepsize_controller=stepsize_controller,
                max_steps=max_steps
            )
            return sol.ys[-1]

        def get_dynamics_stats(self) -> dict:
            tau = self.dynamics.get_tau()
            D = self.dynamics.get_diffusion()
            return {
                'tau_mean': float(tau.mean()),
                'tau_std': float(tau.std()),
                'D_mean': float(D.mean()),
                'D_std': float(D.std()),
            }
    return (ReactionDiffusionLM,)


@app.cell
def _(jnp, jr):
    # ============================================================================
    # Multi-Scale Copy Task - Requires different τ for different delays
    # ============================================================================

    def generate_copy_example(key, vocab_size=16, min_delay=3, max_delay=50):
        """
        Generate a variable-delay copy task example.
    
        Format: [token] [distractors...] [token]
        Target: [ignore...] [predict token at end]
    
        The model must remember 'token' across variable delays.
        """
        k1, k2, k3 = jr.split(key, 3)
    
        # Random token to remember (exclude 0 which is distractor)
        token = jr.randint(k1, (), 1, vocab_size)
    
        # Random delay length
        delay = jr.randint(k2, (), min_delay, max_delay + 1)
    
        # Distractor is always 0
        distractors = jnp.zeros(delay, dtype=jnp.int32)
    
        # Input: [token, distractors..., token]
        # We repeat token at end as "query" signal
        input_seq = jnp.concatenate([
            jnp.array([token]),
            distractors,
            jnp.array([token])  # Query: "what was that token?"
        ])
    
        # Target: only care about final prediction
        # Use -1 for "don't care" positions, token for final
        target_seq = jnp.concatenate([
            jnp.full(delay + 1, -1, dtype=jnp.int32),  # Ignore these
            jnp.array([token])  # Must predict this
        ])
    
        return input_seq, target_seq, int(delay)

    def generate_copy_batch(key, batch_size=32, **kwargs):
        """Generate batch of variable-delay copy examples."""
        keys = jr.split(key, batch_size)
        examples = [generate_copy_example(k, **kwargs) for k in keys]
    
        # Pad to max length in batch
        max_len = max(len(x) for x, _, _ in examples)
    
        inputs = jnp.stack([
            jnp.pad(x, (0, max_len - len(x)), constant_values=0)
            for x, _, _ in examples
        ])
        targets = jnp.stack([
            jnp.pad(y, (0, max_len - len(y)), constant_values=-1)
            for _, y, _ in examples
        ])
        delays = jnp.array([d for _, _, d in examples])
    
        return inputs, targets, delays

    # Test it
    test_key = jr.PRNGKey(0)
    for i in range(5):
        key = jr.fold_in(test_key, i)
        inp, tgt, delay = generate_copy_example(key, vocab_size=16, min_delay=3, max_delay=30)
        print(f"Delay {delay:2d}: input={inp.tolist()}, target_token={int(tgt[-1])}")
    return (generate_copy_example,)


@app.cell
def _(ReactionDiffusionLM, eqx, generate_copy_example, jax, jnp, jr, optax):
    # ============================================================================
    # Train Reaction-Diffusion Model on Copy Task
    # ============================================================================

    def compute_copy_loss_rd(model, input_seq, target_seq, time_per_token=0.5):
        def step(state, inputs):
            token, target = inputs
            state = model.inject_token(state, token)
            state = model.evolve(state, 0.0, time_per_token)
            logits = model.get_emission_logits(state)

            loss = jnp.where(
                target >= 0,
                optax.softmax_cross_entropy_with_integer_labels(logits, target),
                0.0
            )
            correct = jnp.where(
                target >= 0,
                (jnp.argmax(logits) == target).astype(jnp.float32),
                0.0
            )
            return state, (loss, correct, target >= 0)

        init_state = jnp.zeros(model.hidden_dim)
        _, (losses, corrects, masks) = jax.lax.scan(
            step, init_state, (input_seq, target_seq)
        )

        total_mask = masks.sum()
        avg_loss = jnp.where(total_mask > 0, losses.sum() / total_mask, 0.0)
        accuracy = jnp.where(total_mask > 0, corrects.sum() / total_mask, 0.0)

        return avg_loss, {'loss': avg_loss, 'accuracy': accuracy}

    @eqx.filter_jit
    def train_step_rd(model, opt_state, optimizer, input_seq, target_seq):
        def loss_fn(m):
            return compute_copy_loss_rd(m, input_seq, target_seq)

        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, metrics

    # === Training ===
    VOCAB_SIZE = 16
    HIDDEN_DIM = 64
    NUM_STEPS = 2000
    MIN_DELAY = 3
    MAX_DELAY = 40

    def train_rd_model(freeze_tau: bool, diffusion_init: float = 0.1, seed: int = 42):
        key = jr.PRNGKey(seed)
        key, model_key = jr.split(key)

        model = ReactionDiffusionLM(
            vocab_size=VOCAB_SIZE,
            hidden_dim=HIDDEN_DIM,
            dynamics_width=128,
            dynamics_depth=2,
            freeze_tau=freeze_tau,
            tau_min=0.1,
            tau_max=20.0,
            diffusion_init=diffusion_init,
            key=model_key
        )

        optimizer = optax.adam(3e-4)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        history = {'loss': [], 'accuracy': [], 'D_mean': [], 'tau_std': []}

        for step in range(NUM_STEPS):
            key, batch_key = jr.split(key)
            inp, tgt, delay = generate_copy_example(
                batch_key, VOCAB_SIZE, MIN_DELAY, MAX_DELAY
            )

            model, opt_state, loss, metrics = train_step_rd(
                model, opt_state, optimizer, inp, tgt
            )

            history['loss'].append(float(loss))
            history['accuracy'].append(float(metrics['accuracy']))

            stats = model.get_dynamics_stats()
            history['D_mean'].append(stats['D_mean'])
            history['tau_std'].append(stats['tau_std'])

            if (step + 1) % 200 == 0:
                recent_acc = sum(history['accuracy'][-100:]) / 100
                print(f"Step {step+1:4d} | Loss: {loss:.4f} | Acc: {recent_acc:.1%} | "
                      f"D: {stats['D_mean']:.3f} | τ_std: {stats['tau_std']:.2f}")

        return model, history

    # Train with reaction-diffusion
    print("=" * 70)
    print("REACTION-DIFFUSION: Learned τ + Learned D")
    print("=" * 70)
    model_rd_learned, hist_rd_learned = train_rd_model(freeze_tau=False, diffusion_init=0.1)

    print("\n" + "=" * 70)
    print("REACTION-DIFFUSION: Frozen τ + Learned D") 
    print("=" * 70)
    model_rd_frozen, hist_rd_frozen = train_rd_model(freeze_tau=True, diffusion_init=0.1)

    # Also compare to original CfC (no diffusion)
    print("\n" + "=" * 70)
    print("BASELINE: Original CfC (no diffusion, frozen τ)")
    print("=" * 70)
    model_rd_nodiff, hist_rd_nodiff = train_rd_model(freeze_tau=True, diffusion_init=0.001)  # ~no diffusion
    return hist_rd_frozen, hist_rd_learned, hist_rd_nodiff


@app.cell
def _(hist_rd_frozen, hist_rd_learned, hist_rd_nodiff, jnp, plt):
    # ============================================================================
    # Compare Results
    # ============================================================================

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    window = 50
    def smooth(x):
        return jnp.convolve(jnp.array(x), jnp.ones(window)/window, mode='valid')

    # Accuracy
    axes[0,0].plot(smooth(hist_rd_learned['accuracy']), label='RD + Learned τ')
    axes[0,0].plot(smooth(hist_rd_frozen['accuracy']), label='RD + Frozen τ')
    axes[0,0].plot(smooth(hist_rd_nodiff['accuracy']), label='No diffusion (baseline)', linestyle='--')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].set_title('Copy Task Accuracy')
    axes[0,0].legend()

    # Loss
    axes[0,1].plot(smooth(hist_rd_learned['loss']), label='RD + Learned τ')
    axes[0,1].plot(smooth(hist_rd_frozen['loss']), label='RD + Frozen τ')
    axes[0,1].plot(smooth(hist_rd_nodiff['loss']), label='No diffusion', linestyle='--')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].set_title('Copy Task Loss')
    axes[0,1].legend()

    # Diffusion coefficient evolution
    axes[1,0].plot(hist_rd_learned['D_mean'], label='Learned τ')
    axes[1,0].plot(hist_rd_frozen['D_mean'], label='Frozen τ')
    axes[1,0].set_xlabel('Step')
    axes[1,0].set_ylabel('D (diffusion)')
    axes[1,0].set_title('Learned Diffusion Coefficient')
    axes[1,0].legend()

    # τ diversity (only for learned τ)
    axes[1,1].plot(hist_rd_learned['tau_std'], label='τ std')
    axes[1,1].set_xlabel('Step')
    axes[1,1].set_ylabel('τ std')
    axes[1,1].set_title('τ Diversity Over Training (Learned τ)')

    plt.tight_layout()
    plt.show()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    final_acc_learned = sum(hist_rd_learned['accuracy'][-100:]) / 100
    final_acc_frozen = sum(hist_rd_frozen['accuracy'][-100:]) / 100
    final_acc_nodiff = sum(hist_rd_nodiff['accuracy'][-100:]) / 100

    print(f"Final Accuracy:")
    print(f"  RD + Learned τ:  {final_acc_learned:.1%}")
    print(f"  RD + Frozen τ:   {final_acc_frozen:.1%}")
    print(f"  No diffusion:    {final_acc_nodiff:.1%}")

    print(f"\nFinal Diffusion Coefficient:")
    print(f"  Learned τ: D = {hist_rd_learned['D_mean'][-1]:.4f}")
    print(f"  Frozen τ:  D = {hist_rd_frozen['D_mean'][-1]:.4f}")

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
