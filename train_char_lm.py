"""
Character-level language model training for GatedWave vs. minGRU baseline.

Trains on TinyShakespeare (input.txt) with 90/5/5 train/val/test split.
Reports bits-per-character (BPC). Designed for A100 Lambda runs.

Usage:
    # GatedWave (fixed architecture, both bugs corrected)
    python train_char_lm.py --model gated_wave --d_model 512 --n_layers 6 --n_scales 4
    python train_char_lm.py --model gated_wave --d_model 1024 --n_layers 12 --n_scales 4 --wandb

    # minGRU baseline (matched parameter count)
    python train_char_lm.py --model mingru --d_model 724 --n_layers 6   # ~same params as wave d=512

    # Quick stability check (30 min on A100)
    python train_char_lm.py --model gated_wave --d_model 512 --n_layers 6 --n_scales 4 --steps 5000

    # Language-tuned theta (H9: match theta to English text periodicities)
    # theta_min=0.063 (period~100 chars, sentence scale)
    # theta_max=1.257 (period~5 chars, word scale)
    python train_char_lm.py --model gated_wave --d_model 1024 --n_layers 12 --n_scales 4 \\
        --theta_min 0.063 --theta_max 1.257 --wandb
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import argparse
import time
import json
import pickle
from pathlib import Path
from typing import Optional

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from gated_wave import GatedWaveModel

# ── minGRU baseline ──────────────────────────────────────────────────────────

def parallel_scan_real(a, b):
    """Associative scan for real-valued recurrence: h_t = a_t * h_{t-1} + b_t."""
    def binary_op(left, right):
        a_l, b_l = left
        a_r, b_r = right
        return a_l * a_r, a_r * b_l + b_r
    _, h = jax.lax.associative_scan(binary_op, (a, b), axis=1)
    return h

class MinGRULayer(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, x):
        z = nn.sigmoid(nn.Dense(self.d_model, name="z_proj")(x))
        h_tilde = nn.Dense(self.d_model, name="h_proj")(x)
        return parallel_scan_real(1 - z, z * h_tilde)

class MinGRU(nn.Module):
    d_model: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.d_model, name="input_proj")(x)
        for i in range(self.n_layers):
            residual = x
            x = nn.LayerNorm(name=f"ln_{i}")(x)
            x = MinGRULayer(d_model=self.d_model, name=f"gru_{i}")(x)
            x = x + residual
        return x

# ── Character LM wrappers ─────────────────────────────────────────────────────

class CharLM(nn.Module):
    """Wraps a sequence model with embedding + output projection."""
    vocab_size: int
    d_model: int
    n_layers: int
    n_scales: int
    model_type: str  # "gated_wave" | "mingru"
    theta_min: float = 0.01
    theta_max: float = 1.0

    @nn.compact
    def __call__(self, x_indices):
        # x_indices: (batch, seq_len) int32
        # Embed via one-hot + linear (matches existing train_char_wave.py approach)
        x = jax.nn.one_hot(x_indices, self.vocab_size)  # (batch, seq, vocab)
        x = nn.Dense(self.d_model, name="embed")(x)

        if self.model_type == "gated_wave":
            core = GatedWaveModel(
                d_model=self.d_model,
                n_layers=self.n_layers,
                n_scales=self.n_scales,
                theta_min=self.theta_min,
                theta_max=self.theta_max,
            )
        else:
            core = MinGRU(d_model=self.d_model, n_layers=self.n_layers)

        h = core(x)
        return nn.Dense(self.vocab_size, name="lm_head")(h)  # (batch, seq, vocab)

# ── Data ──────────────────────────────────────────────────────────────────────

def load_shakespeare(path: str):
    with open(path) as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int32)
    return data, vocab_size, stoi, {i: c for c, i in stoi.items()}

def make_splits(data: np.ndarray):
    n = len(data)
    return {
        "train": data[:int(0.9 * n)],
        "val":   data[int(0.9 * n):int(0.95 * n)],
        "test":  data[int(0.95 * n):],
    }

def get_batch(data: np.ndarray, seq_len: int, batch_size: int, rng: np.random.Generator):
    ix = rng.integers(0, len(data) - seq_len, batch_size)
    x = np.stack([data[i:i+seq_len] for i in ix])
    y = np.stack([data[i+1:i+seq_len+1] for i in ix])
    return jnp.array(x), jnp.array(y)

# ── Training ──────────────────────────────────────────────────────────────────

def count_params(params) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

def bpc(loss: float) -> float:
    """Convert nats to bits-per-character."""
    return loss / np.log(2)

def eval_split(model, params, data, vocab_size, seq_len, batch_size, n_batches=50):
    rng = np.random.default_rng(0)
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, seq_len, batch_size, rng)
        logits = model.apply(params, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        losses.append(float(loss))
    return float(np.mean(losses))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gated_wave", choices=["gated_wave", "mingru"])
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_scales", type=int, default=4, help="GatedWave only")
    parser.add_argument("--theta_min", type=float, default=0.01,
        help="GatedWave: min oscillatory theta (default 0.01 = period~628). "
             "Language preset: 0.063 (period~100, sentence scale)")
    parser.add_argument("--theta_max", type=float, default=1.0,
        help="GatedWave: max oscillatory theta (default 1.0 = period~6.3). "
             "Language preset: 1.257 (period~5, word scale)")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--data", default="data/shakespeare/input.txt")
    parser.add_argument("--save_dir", default="checkpoints/char_lm")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="gated-wave-char-lm")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Data
    data, vocab_size, stoi, itos = load_shakespeare(args.data)
    splits = make_splits(data)
    print(f"Vocab size: {vocab_size}")
    print(f"Train: {len(splits['train']):,} | Val: {len(splits['val']):,} | Test: {len(splits['test']):,} chars")

    # Model
    model = CharLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_scales=args.n_scales,
        model_type=args.model,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
    )

    key = jax.random.PRNGKey(args.seed)
    dummy_x = jnp.zeros((args.batch_size, args.seq_len), dtype=jnp.int32)
    params = model.init(key, dummy_x)
    n_params = count_params(params)
    print(f"Parameters: {n_params:,}")
    print(f"Model: {args.model} | d={args.d_model} | L={args.n_layers}" +
          (f" | scales={args.n_scales}" if args.model == "gated_wave" else ""))

    # Optimizer with warmup + cosine decay
    schedule = optax.join_schedules([
        optax.linear_schedule(0.0, args.lr, args.warmup),
        optax.cosine_decay_schedule(args.lr, args.steps - args.warmup),
    ], boundaries=[args.warmup])
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=0.01),
    )
    opt_state = optimizer.init(params)

    # W&B
    run_name = f"{args.model}_d{args.d_model}_L{args.n_layers}" + \
               (f"_s{args.n_scales}" if args.model == "gated_wave" else "")
    if args.wandb and HAS_WANDB:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args) | {"n_params": n_params, "vocab_size": vocab_size},
        )
        print(f"W&B: {wandb.run.url}")

    @jax.jit
    def train_step(params, opt_state, x, y):
        def loss_fn(params):
            logits = model.apply(params, x)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    rng = np.random.default_rng(args.seed)
    best_val_bpc = float("inf")
    t0 = time.time()

    print(f"\nTraining for {args.steps:,} steps...")
    for step in range(1, args.steps + 1):
        x, y = get_batch(splits["train"], args.seq_len, args.batch_size, rng)
        params, opt_state, loss = train_step(params, opt_state, x, y)

        if step % 100 == 0:
            elapsed = time.time() - t0
            train_bpc = bpc(float(loss))
            lr_now = float(schedule(step))
            print(f"step {step:6d} | train_bpc={train_bpc:.4f} | lr={lr_now:.2e} | {elapsed:.0f}s")
            if args.wandb and HAS_WANDB:
                wandb.log({"train/bpc": train_bpc, "lr": lr_now}, step=step)

        if step % args.eval_every == 0:
            val_loss = eval_split(model, params, splits["val"], vocab_size,
                                  args.seq_len, args.batch_size)
            val_bpc_val = bpc(val_loss)
            print(f"  → val_bpc={val_bpc_val:.4f}")
            if args.wandb and HAS_WANDB:
                wandb.log({"val/bpc": val_bpc_val}, step=step)
            if val_bpc_val < best_val_bpc:
                best_val_bpc = val_bpc_val
                with open(save_dir / "best_params.pkl", "wb") as f:
                    pickle.dump(params, f)
                print(f"  ↑ New best val BPC: {best_val_bpc:.4f} (saved)")

        if step % args.save_every == 0:
            ckpt_path = save_dir / f"params_step{step}.pkl"
            with open(ckpt_path, "wb") as f:
                pickle.dump(params, f)
            print(f"  Checkpoint: {ckpt_path}")

    # Final test eval
    test_loss = eval_split(model, params, splits["test"], vocab_size,
                           args.seq_len, args.batch_size, n_batches=100)
    test_bpc_val = bpc(test_loss)
    print(f"\nFinal test BPC: {test_bpc_val:.4f}")
    print(f"Best val BPC:   {best_val_bpc:.4f}")

    results = {
        "model": args.model, "d_model": args.d_model, "n_layers": args.n_layers,
        "n_scales": args.n_scales if args.model == "gated_wave" else None,
        "n_params": n_params, "seq_len": args.seq_len, "steps": args.steps,
        "best_val_bpc": best_val_bpc, "test_bpc": test_bpc_val,
    }
    with open(save_dir / f"{run_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {save_dir}/{run_name}_results.json")

    if args.wandb and HAS_WANDB:
        wandb.log({"test/bpc": test_bpc_val, "best_val_bpc": best_val_bpc})
        wandb.finish()

if __name__ == "__main__":
    main()
