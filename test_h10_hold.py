"""
H10 Hypothesis Test: Separate Forget/Input Gates Enable HOLD Behavior in Integrator.

Tests whether separate forget (f) and input (i) gates for the scale-0 integrator
outperform the tied gate on an accumulate-then-hold task.

Background (from hypotheses.md):
  The tied gate h_t = (1-g)*(h_{t-1} + b_t) uses the same scalar g to control
  both state retention AND input acceptance. The four gate-state combinations are:
    f≈1, i≈1 → INTEGRATE  [tied: g≈0 — available]
    f≈0, i≈0 → ZERO       [tied: g≈1 — available]
    f≈1, i≈0 → HOLD       [tied: NOT available — requires g≈0 AND g≈1 simultaneously]
    f≈0, i≈1 → NEW        [tied: NOT available]

  HOLD is uniquely critical for the integrator (r=1) because:
    - Oscillatory scales have natural decay (r<1) — temporal windowing is partly intrinsic
    - The integrator (r=1) has NO natural decay — the gate is its sole windowing mechanism
    - When the integrator has accumulated a value, it must HOLD while ignoring subsequent input

Task: Accumulate-Then-Hold
  Phase A: K tokens in {0, 1} — accumulate (count the 1s)
  SEP:     Token with value 2 — triggers HOLD
  Phase H: N tokens in {0, 1} — hold state, ignore these (same distribution as A-phase)
  QUERY:   Token with value 3 — output the accumulated count
  Target:  sum(x_0..x_{K-1})  [ignoring H-phase entirely]

Why tied gate should struggle:
  - At SEP, gate must simultaneously preserve state (g≈0) AND reject new input (g≈1)
  - These are contradictory — any single gate value causes partial corruption
  - Alternative: suppress via drive layer (b_t≈0 during H-phase) — harder to learn,
    requires the drive to discriminate H-phase tokens from A-phase tokens by value alone,
    but A and H tokens are drawn from the same {0,1} distribution

Why separate gates succeed:
  - At SEP: f→low (retain state), i→high (block new input) — independently controllable
  - H-phase: f≈0, i≈0 → HOLD: accumulated count preserved without distractor contamination

Note: This test uses scale-0 only (pure integrator, r=1, theta=0) to isolate the gate effect.
      With oscillatory scales (r<1), intrinsic decay provides partial windowing that mitigates
      the tied gate limitation — so the H10 effect may be smaller in the full architecture.

Parameter fairness:
  Tied:     gate_1 (d→d), gate_2 (d→d) + drive (d→d) = 3 dense layers
  Separate: forget  (d→d), input_g (d→d) + drive (d→d) = 3 dense layers
  Both have identical parameter counts (within ~0.5% due to bias terms).

Configs:
  tied:     Pure integrator, tied (1-g) gate — current gated_wave.py architecture
  separate: Pure integrator, separate forget + input gates — H10 variant

Prediction: separate > tied. Gap should increase with longer H-phase (larger N).

Usage:
    cd ~/projects/continuous-language
    source .venv/bin/activate
    python test_h10_hold.py
    python test_h10_hold.py --K 15 --N 20 --steps 3000
    python test_h10_hold.py --K 10 --N 30  # stress-test with long hold phase
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import argparse
from typing import List, Tuple

from gated_wave import parallel_scan


# ── Gate Architectures ────────────────────────────────────────────────────────

class TiedGateIntegrator(nn.Module):
    """Pure integrator (r=1, theta=0) with tied (1-g) gate — current architecture.

    h_t = (1 - g_t) * h_{t-1} + (1 - g_t) * b_t

    The same gate g_t controls both state retention and input acceptance.
    HOLD (retain state, reject input) requires g_t≈0 AND g_t≈1 — structurally impossible.

    Gate uses a 2-layer MLP (matching the gated_wave.py implementation).
    """
    d_model: int

    @nn.compact
    def __call__(self, x):
        # 2-layer gate MLP (matches gated_wave.py)
        g1 = nn.Dense(self.d_model, name="gate_1")(x)
        g2 = nn.relu(g1)
        g = nn.sigmoid(
            nn.Dense(self.d_model, name="gate_2",
                     bias_init=nn.initializers.constant(-4.0))(g2)
        )
        # Drive
        b = nn.Dense(self.d_model, name="drive")(x) / jnp.sqrt(float(self.d_model))

        # Pure integrator: a_base = r * exp(i*theta) = 1.0
        a = (1.0 - g).astype(jnp.complex64)
        b_gated = (1.0 - g).astype(jnp.complex64) * b.astype(jnp.complex64)

        state = parallel_scan(a, b_gated)
        return state.real  # [batch, seq, d_model]; imag is zero for theta=0


class SeparateGateIntegrator(nn.Module):
    """Pure integrator (r=1, theta=0) with separate forget and input gates — H10 variant.

    h_t = (1 - f_t) * h_{t-1} + i_t * b_t

    f_t (forget gate): controls state retention — f≈0 → retain, f≈1 → wipe
    i_t (input gate):  controls new input acceptance — i≈1 → accept, i≈0 → block

    These gates are independent, enabling all four gate-state combinations including
    HOLD (f≈0, i≈0): preserve accumulated state while rejecting new input.

    Initialization:
    - forget biased -4.0 → f≈0.018 (start retaining state)
    - input  biased +4.0 → i≈0.982 (start accepting input)
    Both configs begin from "integrate everything" — same starting behavior as tied gate.
    """
    d_model: int

    @nn.compact
    def __call__(self, x):
        # Forget gate: f≈0 → retain state, f≈1 → wipe state
        f = nn.sigmoid(
            nn.Dense(self.d_model, name="forget",
                     bias_init=nn.initializers.constant(-4.0))(x)
        )
        # Input gate: i≈1 → accept input, i≈0 → block input
        inp = nn.sigmoid(
            nn.Dense(self.d_model, name="input_gate",
                     bias_init=nn.initializers.constant(4.0))(x)
        )
        # Drive
        b = nn.Dense(self.d_model, name="drive")(x) / jnp.sqrt(float(self.d_model))

        # Pure integrator: a_base = 1.0
        a = (1.0 - f).astype(jnp.complex64)
        b_gated = inp.astype(jnp.complex64) * b.astype(jnp.complex64)

        state = parallel_scan(a, b_gated)
        return state.real  # [batch, seq, d_model]


# ── Task Model ────────────────────────────────────────────────────────────────

class AccHoldModel(nn.Module):
    """Input projection → integrator dynamics → classify at QUERY position."""
    d_model: int
    n_classes: int   # K+1 possible counts (0..K)
    gate_type: str   # "tied" or "separate"

    @nn.compact
    def __call__(self, x):
        # x: [batch, seq, 1] — token value (pre-normalized)
        x_proj = nn.Dense(self.d_model, name="input_proj")(x)

        if self.gate_type == "tied":
            h = TiedGateIntegrator(d_model=self.d_model, name="integrator")(x_proj)
        else:
            h = SeparateGateIntegrator(d_model=self.d_model, name="integrator")(x_proj)

        # Classify from final timestep (QUERY token position)
        return nn.Dense(self.n_classes, name="classifier")(h[:, -1, :])  # [batch, n_classes]


# ── Data ──────────────────────────────────────────────────────────────────────

SEP_VALUE   = 2.0   # separator token raw value
QUERY_VALUE = 3.0   # query token raw value
NORM_FACTOR = 3.0   # divides all raw values; A/H tokens in [0,0.33], SEP=0.67, QUERY=1.0


def make_batch(
    batch_size: int,
    K: int,
    N: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate [batch_size] accumulate-then-hold sequences of fixed length K+N+2.

    Sequence layout (raw values):
      [a_0..a_{K-1}]  A-phase: K tokens in {0, 1}
      [SEP=2.0]       Separator: 1 token
      [h_0..h_{N-1}]  H-phase: N tokens in {0, 1}  (same distribution as A-phase)
      [QUERY=3.0]     Query: 1 token

    Target: sum(a_0..a_{K-1})  — count of 1s in A-phase only.
    The model must ignore H-phase tokens entirely.

    Returns:
        x: [batch, K+N+2, 1]  float32, values in [0, 1] after normalization
        y: [batch]             int32, count of 1s in A-phase (range: 0..K)
    """
    a_phases = rng.integers(0, 2, (batch_size, K)).astype(np.float32)
    h_phases = rng.integers(0, 2, (batch_size, N)).astype(np.float32)

    counts = a_phases.sum(axis=1).astype(np.int32)

    sep   = np.full((batch_size, 1), SEP_VALUE,   dtype=np.float32)
    query = np.full((batch_size, 1), QUERY_VALUE, dtype=np.float32)

    # [batch, K + 1 + N + 1]
    seqs = np.concatenate([a_phases, sep, h_phases, query], axis=1)
    x = seqs[:, :, np.newaxis] / NORM_FACTOR  # [batch, seq, 1]

    return x, counts


# ── Training ──────────────────────────────────────────────────────────────────

def count_params(params) -> int:
    return sum(v.size for v in jax.tree_util.tree_leaves(params))


def run_experiment(
    name: str,
    gate_type: str,
    d_model: int,
    K: int,
    N: int,
    steps: int,
    batch_size: int,
    lr: float,
    eval_every: int,
    seed: int,
    batch_fn=None,
) -> List[Tuple[int, float]]:
    """Train one gate config, return list of (step, accuracy) checkpoints."""
    n_classes = K + 1

    model = AccHoldModel(d_model=d_model, n_classes=n_classes, gate_type=gate_type)

    seq_len = K + 1 + N + 1
    key = jax.random.PRNGKey(seed)
    dummy_x = jnp.zeros((batch_size, seq_len, 1))
    params = model.init(key, dummy_x)
    n_params = count_params(params)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, x, y):
        def loss_fn(p):
            logits = model.apply(p, x)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), new_opt_state, loss

    @jax.jit
    def eval_step(params, x):
        logits = model.apply(params, x)
        return jnp.argmax(logits, axis=-1)

    if batch_fn is None:
        batch_fn = make_batch

    rng_train = np.random.default_rng(seed)
    rng_eval  = np.random.default_rng(seed + 1000)

    # Baseline: random chance
    baseline_acc = 1.0 / n_classes
    print(f"\n  {name} [{gate_type}] (params={n_params:,}, baseline={baseline_acc:.3f})")
    results = []

    for step in range(1, steps + 1):
        x, y = batch_fn(batch_size, K, N, rng_train)
        x_jax, y_jax = jnp.array(x), jnp.array(y)
        params, opt_state, loss = train_step(params, opt_state, x_jax, y_jax)

        if step % eval_every == 0:
            n_eval = 400
            x_ev, y_ev = batch_fn(n_eval, K, N, rng_eval)
            preds = eval_step(params, jnp.array(x_ev))
            acc = float(jnp.mean(preds == jnp.array(y_ev)))
            print(f"    step {step:5d} | loss={float(loss):.4f} | acc={acc:.3f}")
            results.append((step, acc))

    return results


def make_batch_distinct(
    batch_size: int,
    K: int,
    N: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Revised task variant: H-phase tokens have DISTINCT value from A-phase.

    Sequence layout:
      A-phase: K tokens in {0, 1}                (raw values: 0.0 or 1.0)
      SEP:     1 token, value=2.0
      H-phase: N tokens, ALL constant=0.5        (distinct from A-phase {0,1})
      QUERY:   1 token, value=3.0

    After normalization by NORM_FACTOR=3.0:
      A-phase tokens: {0.000, 0.333}
      SEP:            0.667
      H-phase tokens: 0.167                      (distinct — gate can detect by value)
      QUERY:          1.000

    WHY THIS MATTERS FOR H10:
      The gate can now detect "I'm in H-phase" by seeing x=0.167 (after normalization).
      Tied gate: can set g (tied value) to either retain (integrate 0.167-drive) or zero state.
        Drive suppression: gate g≈0 + learn drive(0.167)≈0 — works but requires learned suppression
        State zeroing: gate g≈1 — wipes the accumulated count, bad
      Separate gate: f≈0 (retain state), i≈0 (block input) = TRUE HOLD.
        No learning required for drive suppression — gate directly prevents contamination.

    Prediction: separate gate advantage should be clearer in this variant.
    """
    a_phases = rng.integers(0, 2, (batch_size, K)).astype(np.float32)
    counts = a_phases.sum(axis=1).astype(np.int32)

    sep   = np.full((batch_size, 1), SEP_VALUE,   dtype=np.float32)
    h_val = 0.5  # distinct from A/H {0, 1}
    h_phases = np.full((batch_size, N), h_val, dtype=np.float32)
    query = np.full((batch_size, 1), QUERY_VALUE, dtype=np.float32)

    seqs = np.concatenate([a_phases, sep, h_phases, query], axis=1)
    x = seqs[:, :, np.newaxis] / NORM_FACTOR
    return x, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=10,
                        help="A-phase length: number of tokens to accumulate")
    parser.add_argument("--N", type=int, default=10,
                        help="H-phase length: number of distractor tokens to ignore")
    parser.add_argument("--distinct_hold", action="store_true",
                        help="Use distinct H-phase token value (0.5 vs A-phase {0,1}). "
                             "Enables value-based gate detection of H-phase. "
                             "Better test for HOLD vs drive suppression.")
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    K, N = args.K, args.N
    seq_len = K + 1 + N + 1
    n_classes = K + 1
    baseline = 1.0 / n_classes

    batch_fn = make_batch_distinct if args.distinct_hold else make_batch
    task_desc = ("DISTINCT H-phase (0.5, detectable by value)" if args.distinct_hold
                 else "SAME H-phase distribution as A-phase (harder: can't distinguish by value)")

    print(f"H10 Accumulate-Then-Hold Test")
    print(f"  K={K} (A-phase), N={N} (H-phase), seq_len={seq_len}")
    print(f"  Task variant: {task_desc}")
    print(f"  Target: count of 1s in first {K} tokens (range 0..{K}, {n_classes} classes)")
    print(f"  Baseline (random): {baseline:.3f}")
    if not args.distinct_hold:
        print(f"\n  NOTE: H-phase uses same {{0,1}} distribution as A-phase.")
        print(f"  Gate conditioned on x only (no state) — cannot distinguish A from H.")
        print(f"  Both configs expected to use drive suppression, not true HOLD.")
        print(f"  Use --distinct_hold for a cleaner H10 test.")
    else:
        print(f"\n  H-phase tokens = 0.5 (distinct value). Gate CAN detect H-phase by value.")
        print(f"  Tied gate must use drive suppression; separate gate can use explicit HOLD.")
    print(f"\nH10 prediction: separate > tied (HOLD behavior enables clean state preservation)")
    print(f"  Gap should increase with larger N (longer hold phase = more opportunity for contamination)")

    configs = [
        ("tied_gate",     "tied"),
        ("separate_gate", "separate"),
    ]

    all_results = {}
    for name, gate_type in configs:
        results = run_experiment(
            name=name,
            gate_type=gate_type,
            d_model=args.d_model,
            K=K,
            N=N,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_every=args.eval_every,
            seed=args.seed,
            batch_fn=batch_fn,
        )
        all_results[name] = results

    print("\n" + "=" * 60)
    print("SUMMARY: Final accuracy by config")
    print("=" * 60)
    print(f"  {'Config':<16} {'Final acc':>10}  {'Peak acc':>10}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*10}")
    for name, results in all_results.items():
        if results:
            final_acc = results[-1][1]
            peak_acc  = max(r[1] for r in results)
            print(f"  {name:<16} {final_acc:>10.3f}  {peak_acc:>10.3f}")

    print(f"\n  Baseline (random):   {baseline:>10.3f}")
    print(f"\n  K={K}, N={N}, d={args.d_model}, {args.steps} steps")

    # Convergence speed: steps to reach 50% accuracy
    print(f"\n  Steps to reach 50% accuracy (well above random={baseline:.2f}):")
    for name, results in all_results.items():
        steps_to_50 = None
        for step, acc in results:
            if acc >= 0.50:
                steps_to_50 = step
                break
        marker = f"step {steps_to_50}" if steps_to_50 else f">step {args.steps}"
        print(f"  {name:<16} {marker}")

    print("\nH10 supported if: separate > tied by meaningful margin")
    print("H10 refuted if: tied catches up at longer training, or gap is negligible")
    print("\nNote: if tied reaches comparable accuracy, the network may be using drive")
    print("suppression (b_t→0 for H-phase tokens) rather than gate-based HOLD.")
    print("Consider ablating the drive layer to distinguish these mechanisms.")


if __name__ == "__main__":
    main()
