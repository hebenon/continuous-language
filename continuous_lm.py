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
    # CELL 2: Core Continuous Language Model (v2 - with τ ablation support)
    # ============================================================================

    class ContinuousDynamics(eqx.Module):
        """ODE dynamics with learnable multi-scale time constants."""
        hidden_dim: int
        dynamics_net: eqx.nn.MLP
        log_tau: jnp.ndarray  # Store in log space for log-uniform init
        freeze_tau: bool = False

        def __init__(
            self, 
            hidden_dim: int, 
            width: int, 
            depth: int, 
            freeze_tau: bool = False,
            tau_min: float = 0.1,
            tau_max: float = 20.0,
            *, 
            key
        ):
            self.hidden_dim = hidden_dim
            self.freeze_tau = freeze_tau

            key, net_key, tau_key = jr.split(key, 3)

            self.dynamics_net = eqx.nn.MLP(
                in_size=hidden_dim,
                out_size=hidden_dim,
                width_size=width,
                depth=depth,
                activation=jax.nn.tanh,
                key=net_key
            )

            # Log-uniform initialization: τ ∈ [tau_min, tau_max]
            # log(τ) ~ Uniform(log(tau_min), log(tau_max))
            log_tau_min = jnp.log(tau_min)
            log_tau_max = jnp.log(tau_max)
            self.log_tau = jr.uniform(
                tau_key, (hidden_dim,), 
                minval=log_tau_min, 
                maxval=log_tau_max
            )

        def get_tau(self) -> jnp.ndarray:
            """Get time constants, optionally frozen to mean."""
            tau = jnp.exp(self.log_tau)
            if self.freeze_tau:
                tau = jnp.full_like(tau, tau.mean())
            return tau

        def __call__(self, t, state, args=None):
            target = self.dynamics_net(state)
            tau = self.get_tau()
            dxdt = (target - state) / tau
            # Soft clip to [-5, 5] for stability
            dxdt = jnp.tanh(dxdt) * 5.0
            return dxdt


    class ContinuousLM(eqx.Module):
        """Continuous-time language model with ODE-based state evolution."""
        hidden_dim: int
        vocab_size: int
        dynamics: ContinuousDynamics
        embedding: eqx.nn.Embedding
        input_gate: eqx.nn.Linear
        output_proj: eqx.nn.Linear
        time_per_token: float
        freeze_tau: bool

        def __init__(
            self, 
            vocab_size: int,
            hidden_dim: int = 128,
            dynamics_width: int = 256,
            dynamics_depth: int = 2,
            time_per_token: float = 0.5,
            freeze_tau: bool = False,
            tau_min: float = 0.1,
            tau_max: float = 20.0,
            *, 
            key
        ):
            keys = jr.split(key, 4)

            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim
            self.time_per_token = time_per_token
            self.freeze_tau = freeze_tau

            self.dynamics = ContinuousDynamics(
                hidden_dim, 
                dynamics_width, 
                dynamics_depth,
                freeze_tau=freeze_tau,
                tau_min=tau_min,
                tau_max=tau_max,
                key=keys[0]
            )

            self.embedding = eqx.nn.Embedding(vocab_size, hidden_dim, key=keys[1])
            self.input_gate = eqx.nn.Linear(hidden_dim * 2, hidden_dim, key=keys[2])
            self.output_proj = eqx.nn.Linear(hidden_dim, vocab_size, key=keys[3])

        def inject_token(self, state: jnp.ndarray, token: int) -> jnp.ndarray:
            """Inject a token into the state via gated addition."""
            token_emb = self.embedding(token)
            gate_input = jnp.concatenate([state, token_emb])
            gate = jax.nn.sigmoid(self.input_gate(gate_input))
            return state + 0.5 * gate * token_emb

        def get_emission_logits(self, state: jnp.ndarray) -> jnp.ndarray:
            """Project state to vocabulary logits."""
            return self.output_proj(state)

        def evolve(
            self, 
            state: jnp.ndarray, 
            t_start: float = 0.0, 
            t_end: float = None,
            dt0: float = 0.05,
            max_steps: int = 256
        ) -> jnp.ndarray:
            """Evolve state through ODE dynamics."""
            if t_end is None:
                t_end = t_start + self.time_per_token

            term = diffrax.ODETerm(self.dynamics)
            solver = diffrax.Heun()
            stepsize_controller = diffrax.PIDController(rtol=1e-2, atol=1e-2)

            sol = diffrax.diffeqsolve(
                term,
                solver,
                t0=t_start,
                t1=t_end,
                dt0=dt0,
                y0=state,
                saveat=diffrax.SaveAt(t1=True),
                stepsize_controller=stepsize_controller,
                max_steps=max_steps
            )
            return sol.ys[-1]

        def get_tau_stats(self) -> dict:
            """Get statistics about learned time constants."""
            tau = self.dynamics.get_tau()
            return {
                'min': float(tau.min()),
                'max': float(tau.max()),
                'mean': float(tau.mean()),
                'std': float(tau.std()),
                'values': tau
            }
    return (ContinuousLM,)


@app.cell
def _(ContinuousLM, eqx, jax, jnp, optax):
    # ============================================================================
    # CELL 3: Training with lax.scan (10-50x faster than Python loop)
    # ============================================================================

    def compute_loss_scan(
        model: ContinuousLM,
        input_tokens: jnp.ndarray,
        target_tokens: jnp.ndarray,
        tau_diversity_weight: float = 0.0
    ) -> tuple[float, dict]:
        """
        Compute cross-entropy loss using jax.lax.scan for efficiency.

        Args:
            model: The ContinuousLM model
            input_tokens: Input token sequence [seq_len]
            target_tokens: Target token sequence [seq_len]
            tau_diversity_weight: Weight for τ diversity regularization (0 = disabled)
        """
        def step(state, inputs):
            token, target = inputs
            # Inject token and evolve
            state = model.inject_token(state, token)
            state = model.evolve(state)
            # Compute loss for this position
            logits = model.get_emission_logits(state)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, target)
            return state, loss

        init_state = jnp.zeros(model.hidden_dim)
        _, losses = jax.lax.scan(step, init_state, (input_tokens, target_tokens))
        ce_loss = losses.mean()

        # Optional: τ diversity regularization (penalize collapse to uniform)
        metrics = {'ce_loss': ce_loss}
        total_loss = ce_loss

        if tau_diversity_weight > 0:
            tau = model.dynamics.get_tau()
            # Encourage spread: penalize low variance in log-τ
            log_tau = jnp.log(tau)
            tau_variance = jnp.var(log_tau)
            # We want HIGH variance, so penalize low variance
            tau_penalty = -tau_variance  # negative because we minimize
            total_loss = ce_loss + tau_diversity_weight * tau_penalty
            metrics['tau_variance'] = tau_variance
            metrics['tau_penalty'] = tau_penalty

        metrics['total_loss'] = total_loss
        return total_loss, metrics

    @eqx.filter_jit
    def train_step(
        model: ContinuousLM,
        opt_state,
        optimizer,
        input_tokens: jnp.ndarray,
        target_tokens: jnp.ndarray,
        tau_diversity_weight: float = 0.0
    ):
        """Single JIT-compiled training step."""
        def loss_fn(model):
            return compute_loss_scan(
                model, input_tokens, target_tokens, tau_diversity_weight
            )

        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)

        # Compute gradient norm for monitoring
        grad_norm = jnp.sqrt(sum(
            jnp.sum(x**2) for x in jax.tree_util.tree_leaves(grads) 
            if isinstance(x, jnp.ndarray)
        ))
        metrics['grad_norm'] = grad_norm

        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss, metrics
    return (train_step,)


@app.cell
def _(jnp, jr):
    # ============================================================================
    # CELL 4: Bracket Matching Dataset
    # ============================================================================
    import random

    def generate_bracket_sequence(max_depth=8, max_length=150, seed=None):
        """Generate bracket sequences with distractors for multi-scale memory test."""
        if seed is not None:
            random.seed(seed)

        pairs = [('(', ')'), ('[', ']'), ('{', '}'), ('<', '>')]
        distractors = list('abcdefghijklmnopqrstuvwxyz0123456789.,:;!? ')

        stack = []
        sequence = []

        while len(sequence) < max_length:
            if random.random() < 0.3 and sequence:
                num_distractors = random.randint(1, 5)
                sequence.extend(random.choices(distractors, k=num_distractors))
                continue

            can_open = len(stack) < max_depth
            can_close = len(stack) > 0

            if can_open and can_close:
                action = random.choice(['open', 'open', 'close'] if len(stack) < 3 
                                       else ['open', 'close', 'close'])
            elif can_open:
                action = 'open'
            elif can_close:
                action = 'close'
            else:
                break

            if action == 'open':
                pair = random.choice(pairs)
                stack.append(pair)
                sequence.append(pair[0])
            else:
                pair = stack.pop()
                sequence.append(pair[1])

        while stack:
            if random.random() < 0.3:
                sequence.extend(random.choices(distractors, k=random.randint(1, 3)))
            pair = stack.pop()
            sequence.append(pair[1])

        return ''.join(sequence)

    def generate_corpus(num_sequences=800, max_depth=8, max_length=150):
        return '\n'.join(
            generate_bracket_sequence(max_depth=max_depth, max_length=max_length, seed=i)
            for i in range(num_sequences)
        )

    # Generate corpus
    CORPUS = generate_corpus(num_sequences=800, max_depth=8, max_length=150)

    chars = sorted(list(set(CORPUS)))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    vocab_size = len(chars)
    data = jnp.array([char2idx[c] for c in CORPUS])

    print(f"Vocab size: {vocab_size}")
    print(f"Corpus length: {len(CORPUS):,} characters")
    print(f"\nExample sequences:")
    for seq in CORPUS.split('\n')[:3]:
        print(f"  {seq[:80]}{'...' if len(seq) > 80 else ''}")

    def get_batch(data, seq_len, key):
        max_start = len(data) - seq_len - 1
        start = jr.randint(key, (), 0, max_start)
        x = data[start:start + seq_len]
        y = data[start + 1:start + seq_len + 1]
        return x, y
    return (
        char2idx,
        data,
        generate_bracket_sequence,
        get_batch,
        idx2char,
        vocab_size,
    )


@app.cell
def _(ContinuousLM, data, eqx, get_batch, jr, optax, train_step, vocab_size):
    # ============================================================================
    # CELL 5: Training with Ablation Support
    # ============================================================================

    def run_training(
        freeze_tau: bool = False,
        tau_diversity_weight: float = 0.01,
        hidden_dim: int = 128,
        dynamics_width: int = 256,
        seq_len: int = 80,
        num_steps: int = 1500,
        learning_rate: float = 3e-4,
        time_per_token: float = 0.5,
        tau_min: float = 0.1,
        tau_max: float = 20.0,
        seed: int = 42,
        log_every: int = 50
    ):
        """
        Run training with specified configuration.

        Args:
            freeze_tau: If True, all τ values are frozen to their mean (ablation)
            tau_diversity_weight: Regularization weight for τ spread (0 = disabled)
        """
        key = jr.PRNGKey(seed)
        key, model_key = jr.split(key)

        model = ContinuousLM(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            dynamics_width=dynamics_width,
            dynamics_depth=2,
            time_per_token=time_per_token,
            freeze_tau=freeze_tau,
            tau_min=tau_min,
            tau_max=tau_max,
            key=model_key
        )

        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        config_str = f"freeze_τ={freeze_tau}, τ_reg={tau_diversity_weight}"
        print(f"Training [{config_str}]...")

        # Log initial τ distribution
        tau_stats = model.get_tau_stats()
        print(f"Initial τ: range=[{tau_stats['min']:.2f}, {tau_stats['max']:.2f}], "
              f"std={tau_stats['std']:.3f}")

        history = {
            'loss': [], 'ce_loss': [], 'grad_norm': [], 
            'tau_std': [], 'tau_min': [], 'tau_max': []
        }

        for step in range(num_steps):
            key, batch_key = jr.split(key)
            x, y = get_batch(data, seq_len, batch_key)

            model, opt_state, loss, metrics = train_step(
                model, opt_state, optimizer, x, y, tau_diversity_weight
            )

            # Record history
            history['loss'].append(float(loss))
            history['ce_loss'].append(float(metrics.get('ce_loss', loss)))
            history['grad_norm'].append(float(metrics['grad_norm']))

            tau_stats = model.get_tau_stats()
            history['tau_std'].append(tau_stats['std'])
            history['tau_min'].append(tau_stats['min'])
            history['tau_max'].append(tau_stats['max'])

            if (step + 1) % log_every == 0:
                print(f"Step {step+1:4d}/{num_steps} | Loss: {loss:.4f} | "
                      f"τ_std: {tau_stats['std']:.3f} | grad: {metrics['grad_norm']:.2f}")

        print(f"Final τ: range=[{tau_stats['min']:.2f}, {tau_stats['max']:.2f}], "
              f"std={tau_stats['std']:.3f}")

        return model, history

    # Run with learned τ
    print("=" * 60)
    print("EXPERIMENT 1: Learned τ (multi-scale)")
    print("=" * 60)
    model_learned, history_learned = run_training(
        freeze_tau=False, 
        tau_diversity_weight=0.01,
        num_steps=1000  # Shorter for quick comparison
    )
    return history_learned, model_learned, run_training


@app.cell
def _(history_learned, plt, run_training):
    # ============================================================================
    # CELL 6: Ablation - Frozen τ
    # ============================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Frozen τ (uniform time constants)")
    print("=" * 60)
    model_frozen, history_frozen = run_training(
        freeze_tau=True,
        tau_diversity_weight=0.0,  # No point regularizing if frozen
        num_steps=1000
    )

    # Compare results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss comparison
    axes[0, 0].plot(history_learned['loss'], label='Learned τ', alpha=0.8)
    axes[0, 0].plot(history_frozen['loss'], label='Frozen τ', alpha=0.8)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # τ std over time (learned only)
    axes[0, 1].plot(history_learned['tau_std'], label='Learned τ std')
    axes[0, 1].axhline(y=history_frozen['tau_std'][0], color='r', linestyle='--', 
                       label='Frozen τ std')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('τ std')
    axes[0, 1].set_title('τ Diversity Over Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # τ range over time
    axes[1, 0].fill_between(range(len(history_learned['tau_min'])),
                            history_learned['tau_min'], 
                            history_learned['tau_max'],
                            alpha=0.3, label='Learned τ range')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('τ value')
    axes[1, 0].set_title('τ Range Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Gradient norms
    axes[1, 1].plot(history_learned['grad_norm'], label='Learned τ', alpha=0.7)
    axes[1, 1].plot(history_frozen['grad_norm'], label='Frozen τ', alpha=0.7)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Gradient Norm')
    axes[1, 1].set_title('Training Stability')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.show()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Final loss - Learned τ: {history_learned['loss'][-1]:.4f}")
    print(f"Final loss - Frozen τ:  {history_frozen['loss'][-1]:.4f}")
    print(f"Difference: {history_frozen['loss'][-1] - history_learned['loss'][-1]:.4f}")
    return (model_frozen,)


@app.cell
def _(char2idx, generate_bracket_sequence, idx2char, jnp):
    # ============================================================================
    # CELL 7: Bracket Matching Evaluation
    # ============================================================================

    def evaluate_bracket_matching(model, num_tests=50, verbose=False):
        """Test if model predicts correct closing brackets."""
        correct = 0
        total = 0

        for i in range(num_tests):
            full_seq = generate_bracket_sequence(max_depth=4, max_length=40, seed=2000+i)

            # Find a position where we need to close a bracket
            stack = []
            cut_point = None
            expected = None

            for j, c in enumerate(full_seq):
                if c in '([{<':
                    stack.append(c)
                elif c in ')]}>':
                    if stack and len(full_seq) - j > 3:
                        cut_point = j
                        expected = c
                        break
                    if stack:
                        stack.pop()

            if cut_point is None or expected is None:
                continue

            # Feed prefix to model
            prefix = full_seq[:cut_point]
            tokens = jnp.array([char2idx.get(c, 0) for c in prefix])

            state = jnp.zeros(model.hidden_dim)
            for token in tokens:
                state = model.inject_token(state, token)
                state = model.evolve(state)

            logits = model.get_emission_logits(state)
            predicted_idx = int(jnp.argmax(logits))
            predicted = idx2char.get(predicted_idx, '?')

            if predicted == expected:
                correct += 1
            total += 1

            if verbose and i < 5:
                status = '✓' if predicted == expected else '✗'
                print(f"  '{prefix[-30:]}' → '{predicted}' (expected '{expected}') {status}")

        accuracy = correct / total if total > 0 else 0
        return accuracy, correct, total
    return (evaluate_bracket_matching,)


@app.cell
def _(evaluate_bracket_matching, model_frozen, model_learned):
    # ============================================================================
    # CELL 8: Compare Bracket Matching Accuracy
    # ============================================================================
    print("Evaluating bracket matching accuracy...")
    print()

    print("Learned τ model:")
    acc_learned, c1, t1 = evaluate_bracket_matching(model_learned, num_tests=100, verbose=True)
    print(f"Accuracy: {acc_learned:.1%} ({c1}/{t1})")

    print()
    print("Frozen τ model:")
    acc_frozen, c2, t2 = evaluate_bracket_matching(model_frozen, num_tests=100, verbose=True)
    print(f"Accuracy: {acc_frozen:.1%} ({c2}/{t2})")

    print()
    print("=" * 40)
    if acc_learned > acc_frozen:
        print(f"✓ Learned τ wins by {acc_learned - acc_frozen:.1%}")
    elif acc_frozen > acc_learned:
        print(f"✗ Frozen τ wins by {acc_frozen - acc_learned:.1%}")
    else:
        print("≈ No difference")
    return


@app.cell
def _(jnp, model_learned, plt):
    # ============================================================================
    # CELL 9: Visualize Final τ Distribution
    # ============================================================================
    tau_values = model_learned.dynamics.get_tau()

    tau_final_fig, tau_final_axes = plt.subplots(1, 3, figsize=(14, 4))

    # Bar plot
    tau_final_axes[0].bar(range(len(tau_values)), sorted(tau_values, reverse=True))
    tau_final_axes[0].set_xlabel('Dimension (sorted)')
    tau_final_axes[0].set_ylabel('τ (time constant)')
    tau_final_axes[0].set_title('Learned Time Constants (sorted)')
    tau_final_axes[0].set_yscale('log')

    # Histogram
    tau_final_axes[1].hist(tau_values, bins=25, edgecolor='black', alpha=0.7)
    tau_final_axes[1].set_xlabel('τ value')
    tau_final_axes[1].set_ylabel('Count')
    tau_final_axes[1].set_title(f'τ Distribution (std={float(jnp.std(tau_values)):.3f})')
    tau_final_axes[1].axvline(x=float(jnp.mean(tau_values)), color='r', linestyle='--', 
                    label=f'mean={float(jnp.mean(tau_values)):.2f}')
    tau_final_axes[1].legend()

    # Log histogram (to see if multi-modal in log space)
    tau_final_axes[2].hist(jnp.log(tau_values), bins=25, edgecolor='black', alpha=0.7)
    tau_final_axes[2].set_xlabel('log(τ)')
    tau_final_axes[2].set_ylabel('Count')
    tau_final_axes[2].set_title('log(τ) Distribution')

    plt.tight_layout()
    plt.show()

    print(f"τ range: [{float(tau_values.min()):.3f}, {float(tau_values.max()):.3f}]")
    print(f"τ mean: {float(tau_values.mean()):.3f}, std: {float(tau_values.std()):.3f}")
    print(f"τ ratio (max/min): {float(tau_values.max() / tau_values.min()):.1f}x")
    return (tau_values,)


@app.cell
def _(char2idx, jnp, model_learned, plt, tau_values):
    # ============================================================================
    # CELL 10: Analyze τ vs Bracket Depth Correlation
    # ============================================================================

    def analyze_tau_depth_correlation(model, test_seq="({[<>]})" * 5):
        """Check if slow dimensions (high τ) track bracket depth."""
        tokens = jnp.array([char2idx.get(c, 0) for c in test_seq])

        # Compute depth at each position
        depths = []
        d = 0
        for c in test_seq:
            if c in '([{<': 
                d += 1
            depths.append(d)
            if c in ')]}>': 
                d -= 1

        # Run model and collect states
        state = jnp.zeros(model.hidden_dim)
        states = []
        for token in tokens:
            state = model.inject_token(state, token)
            state = model.evolve(state)
            states.append(state)

        states = jnp.stack(states)
        depths = jnp.array(depths, dtype=jnp.float32)

        # Correlation of each dimension with depth
        correlations = []
        for dim in range(model.hidden_dim):
            dim_vals = states[:, dim]
            # Normalize for correlation
            dim_norm = (dim_vals - dim_vals.mean()) / (dim_vals.std() + 1e-8)
            depth_norm = (depths - depths.mean()) / (depths.std() + 1e-8)
            corr = jnp.mean(dim_norm * depth_norm)
            correlations.append(float(corr))

        return jnp.array(correlations)

    #tau_values = model_learned.dynamics.get_tau()
    correlations = analyze_tau_depth_correlation(model_learned)

    plt.figure(figsize=(10, 5))
    plt.scatter(tau_values, jnp.abs(correlations), alpha=0.6, s=50)
    plt.xlabel('τ (time constant)', fontsize=12)
    plt.ylabel('|correlation with bracket depth|', fontsize=12)
    plt.title('Do Slow Dimensions Track Bracket Depth?', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)

    # Add trend line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        jnp.log(tau_values), jnp.abs(correlations)
    )
    x_line = jnp.linspace(tau_values.min(), tau_values.max(), 100)
    y_line = slope * jnp.log(x_line) + intercept
    plt.plot(x_line, y_line, 'r--', alpha=0.7, 
             label=f'trend (r={r_value:.2f}, p={p_value:.3f})')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Correlation between log(τ) and |depth correlation|: r={r_value:.3f}, p={p_value:.3f}")
    if p_value < 0.05:
        if r_value > 0:
            print("✓ Significant positive correlation: slower dimensions DO track depth better")
        else:
            print("✗ Significant negative correlation: faster dimensions track depth better")
    else:
        print("≈ No significant correlation between τ and depth tracking")
    return


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
def _(eqx, jax, jnp, optax):
    # ============================================================================
    # Training for Copy Task (only penalize final prediction)
    # ============================================================================

    def compute_copy_loss(model, input_seq, target_seq, time_per_token=0.5):
        """
        Loss only on positions where target != -1 (i.e., final prediction).
        """
        def step(state, inputs):
            token, target = inputs
            state = model.inject_token(state, token)
            state = model.evolve(state, 0.0, time_per_token)
            logits = model.get_emission_logits(state)
        
            # Only compute loss where target != -1
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
    
        # Average only over masked positions
        total_mask = masks.sum()
        avg_loss = jnp.where(total_mask > 0, losses.sum() / total_mask, 0.0)
        accuracy = jnp.where(total_mask > 0, corrects.sum() / total_mask, 0.0)
    
        return avg_loss, {'loss': avg_loss, 'accuracy': accuracy}

    @eqx.filter_jit
    def train_copy_step(model, opt_state, optimizer, input_seq, target_seq):
        def loss_fn(m):
            return compute_copy_loss(m, input_seq, target_seq)
    
        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
    
        return model, opt_state, loss, metrics
    return compute_copy_loss, train_copy_step


@app.cell
def _(
    ContinuousLM,
    eqx,
    generate_copy_example,
    jnp,
    jr,
    optax,
    plt,
    train_copy_step,
):
    # ============================================================================
    # Run Copy Task Experiment
    # ============================================================================

    VOCAB_SIZE = 16
    HIDDEN_DIM = 64
    NUM_STEPS = 2000
    MIN_DELAY = 3
    MAX_DELAY = 40

    def train_copy_model(freeze_tau: bool, seed: int = 42):
        key = jr.PRNGKey(seed)
        key, model_key = jr.split(key)
    
        model = ContinuousLM(
            vocab_size=VOCAB_SIZE,
            hidden_dim=HIDDEN_DIM,
            dynamics_width=128,
            dynamics_depth=2,
            freeze_tau=freeze_tau,
            tau_min=0.1,
            tau_max=20.0,
            key=model_key
        )
    
        optimizer = optax.adam(3e-4)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
        history = {'loss': [], 'accuracy': []}
    
        for step in range(NUM_STEPS):
            key, batch_key = jr.split(key)
            inp, tgt, delay = generate_copy_example(
                batch_key, VOCAB_SIZE, MIN_DELAY, MAX_DELAY
            )
        
            model, opt_state, loss, metrics = train_copy_step(
                model, opt_state, optimizer, inp, tgt
            )
        
            history['loss'].append(float(loss))
            history['accuracy'].append(float(metrics['accuracy']))
        
            if (step + 1) % 200 == 0:
                recent_acc = sum(history['accuracy'][-100:]) / 100
                print(f"Step {step+1:4d} | Loss: {loss:.4f} | Acc: {recent_acc:.1%}")
    
        return model, history

    # Train both
    print("=" * 60)
    print("COPY TASK: Learned τ")
    print("=" * 60)
    copy_model_learned, copy_hist_learned = train_copy_model(freeze_tau=False)

    print("\n" + "=" * 60)
    print("COPY TASK: Frozen τ")
    print("=" * 60)
    copy_model_frozen, copy_hist_frozen = train_copy_model(freeze_tau=True)

    # Plot
    copy_fig, copy_axes = plt.subplots(1, 2, figsize=(12, 4))

    window = 50
    def smooth(x):
        return jnp.convolve(jnp.array(x), jnp.ones(window)/window, mode='valid')

    copy_axes[0].plot(smooth(copy_hist_learned['accuracy']), label='Learned τ')
    copy_axes[0].plot(smooth(copy_hist_frozen['accuracy']), label='Frozen τ')
    copy_axes[0].set_xlabel('Step')
    copy_axes[0].set_ylabel('Accuracy')
    copy_axes[0].set_title('Copy Task Accuracy')
    copy_axes[0].legend()

    copy_axes[1].plot(smooth(copy_hist_learned['loss']), label='Learned τ')
    copy_axes[1].plot(smooth(copy_hist_frozen['loss']), label='Frozen τ')
    copy_axes[1].set_xlabel('Step')
    copy_axes[1].set_ylabel('Loss')
    copy_axes[1].set_title('Copy Task Loss')
    copy_axes[1].legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    compute_copy_loss,
    generate_copy_example,
    jr,
    model_frozen,
    model_learned,
    plt,
):
    def evaluate_copy_task():
        # ============================================================================
        # Evaluate by delay length (this is the key test!)
        # ============================================================================
    
        def evaluate_by_delay(model, delays_to_test, num_samples=50):
            """Test accuracy at each specific delay length."""
            results = {}
        
            for delay in delays_to_test:
                correct = 0
                for i in range(num_samples):
                    key = jr.PRNGKey(10000 + delay * 1000 + i)
                    inp, tgt, _ = generate_copy_example(
                        key, vocab_size=16, min_delay=delay, max_delay=delay
                    )
                    _, metrics = compute_copy_loss(model, inp, tgt)
                    correct += float(metrics['accuracy'])
            
                results[delay] = correct / num_samples
        
            return results
    
        delays = [3, 5, 10, 15, 20, 25, 30, 35, 40]
    
        print("Evaluating accuracy by delay length...")
        acc_learned = evaluate_by_delay(model_learned, delays)
        acc_frozen = evaluate_by_delay(model_frozen, delays)
    
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(delays, [acc_learned[d] for d in delays], 'o-', label='Learned τ', linewidth=2)
        plt.plot(delays, [acc_frozen[d] for d in delays], 's--', label='Frozen τ', linewidth=2)
        plt.xlabel('Delay Length')
        plt.ylabel('Accuracy')
        plt.title('Copy Task: Accuracy vs Delay Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.show()
    
        # Summary
        print("\nAccuracy by delay:")
        print(f"{'Delay':>6} | {'Learned τ':>10} | {'Frozen τ':>10} | {'Winner':>10}")
        print("-" * 45)   # ============================================================================
        # Evaluate by delay length (this is the key test!)
        # ============================================================================
    
        def evaluate_by_delay(model, delays_to_test, num_samples=50):
            """Test accuracy at each specific delay length."""
            results = {}
        
            for delay in delays_to_test:
                correct = 0
                for i in range(num_samples):
                    key = jr.PRNGKey(10000 + delay * 1000 + i)
                    inp, tgt, _ = generate_copy_example(
                        key, vocab_size=16, min_delay=delay, max_delay=delay
                    )
                    _, metrics = compute_copy_loss(model, inp, tgt)
                    correct += float(metrics['accuracy'])
            
                results[delay] = correct / num_samples
        
            return results
    
        delays = [3, 5, 10, 15, 20, 25, 30, 35, 40]
    
        print("Evaluating accuracy by delay length...")
        acc_learned = evaluate_by_delay(model_learned, delays)
        acc_frozen = evaluate_by_delay(model_frozen, delays)
    
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(delays, [acc_learned[d] for d in delays], 'o-', label='Learned τ', linewidth=2)
        plt.plot(delays, [acc_frozen[d] for d in delays], 's--', label='Frozen τ', linewidth=2)
        plt.xlabel('Delay Length')
        plt.ylabel('Accuracy')
        plt.title('Copy Task: Accuracy vs Delay Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.show()
    
        # Summary
        print("\nAccuracy by delay:")
        print(f"{'Delay':>6} | {'Learned τ':>10} | {'Frozen τ':>10} | {'Winner':>10}")
        print("-" * 45)
        for d in delays:
            winner = "Learned" if acc_learned[d] > acc_frozen[d] else "Frozen" if acc_frozen[d] > acc_learned[d] else "Tie"
            print(f"{d:>6} | {acc_learned[d]:>10.1%} | {acc_frozen[d]:>10.1%} | {winner:>10}")
    
        # Key metric: does learned τ win at long delays?
        short_delays = [3, 5, 10]
        long_delays = [30, 35, 40]
    
        learned_short = sum(acc_learned[d] for d in short_delays) / len(short_delays)
        learned_long = sum(acc_learned[d] for d in long_delays) / len(long_delays)
        frozen_short = sum(acc_frozen[d] for d in short_delays) / len(short_delays)
        frozen_long = sum(acc_frozen[d] for d in long_delays) / len(long_delays)
    
        print(f"\nShort delays (3-10): Learned={learned_short:.1%}, Frozen={frozen_short:.1%}")
        print(f"Long delays (30-40): Learned={learned_long:.1%}, Frozen={frozen_long:.1%}")
    
        if learned_long > frozen_long + 0.05:
            print("\n✓ Learned τ shows advantage at long delays - multi-scale helps!")
        elif frozen_long > learned_long + 0.05:
            print("\n✗ Frozen τ wins at long delays - multi-scale not helping")
        else:
            print("\n≈ No significant difference - task may still be too easy")
        for d in delays:
            winner = "Learned" if acc_learned[d] > acc_frozen[d] else "Frozen" if acc_frozen[d] > acc_learned[d] else "Tie"
            print(f"{d:>6} | {acc_learned[d]:>10.1%} | {acc_frozen[d]:>10.1%} | {winner:>10}")
    
        # Key metric: does learned τ win at long delays?
        short_delays = [3, 5, 10]
        long_delays = [30, 35, 40]
    
        learned_short = sum(acc_learned[d] for d in short_delays) / len(short_delays)
        learned_long = sum(acc_learned[d] for d in long_delays) / len(long_delays)
        frozen_short = sum(acc_frozen[d] for d in short_delays) / len(short_delays)
        frozen_long = sum(acc_frozen[d] for d in long_delays) / len(long_delays)
    
        print(f"\nShort delays (3-10): Learned={learned_short:.1%}, Frozen={frozen_short:.1%}")
        print(f"Long delays (30-40): Learned={learned_long:.1%}, Frozen={frozen_long:.1%}")
    
        if learned_long > frozen_long + 0.05:
            print("\n✓ Learned τ shows advantage at long delays - multi-scale helps!")
        elif frozen_long > learned_long + 0.05:
            print("\n✗ Frozen τ wins at long delays - multi-scale not helping")
        else:
            print("\n≈ No significant difference - task may still be too easy")

    evaluate_copy_task()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
