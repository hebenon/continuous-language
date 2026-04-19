"""
Reaction-Diffusion minGRU for Pathfinder experiments.

Combines minGRU's parallelizable gates with reaction-diffusion spatial coupling.
Hypothesis: diffusion enables cross-scale information flow that helps long-range tasks.

Usage:
    python train_reaction_diffusion.py --task path32 --wandb
    python train_reaction_diffusion.py --task pathx --wandb
"""

import jax
import jax.numpy as jnp
from jax import random, lax
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
import numpy as np
from functools import partial
from pathlib import Path

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("wandb not installed.")


# === Task Configurations ===

TASK_CONFIGS = {
    "path32": {"seq_len": 1024, "image_size": 32},
    "path64": {"seq_len": 4096, "image_size": 64},
    "pathx": {"seq_len": 16384, "image_size": 128},
}


# === Parallel Scan ===

def binary_operator_diag(q_i, q_j):
    a_i, b_i = q_i
    a_j, b_j = q_j
    return a_i * a_j, a_j * b_i + b_j


def parallel_scan(a, b):
    a = jnp.moveaxis(a, 1, 0)
    b = jnp.moveaxis(b, 1, 0)
    _, h = lax.associative_scan(binary_operator_diag, (a, b))
    h = jnp.moveaxis(h, 0, 1)
    return h


# === Reaction-Diffusion minGRU ===

class ReactionDiffusionMinGRULayer(nn.Module):
    """minGRU layer with reaction-diffusion coupling between dimensions."""
    d_model: int
    diffusion_steps: int = 1
    diffusion_init: float = 0.1
    
    @nn.compact
    def __call__(self, x):
        # === minGRU (parallel scan) ===
        z_pre = nn.Dense(self.d_model, name="z_proj")(x)
        h_tilde = nn.Dense(self.d_model, name="h_proj")(x)
        
        z = nn.sigmoid(z_pre)
        a = 1 - z
        b = z * h_tilde
        
        h = parallel_scan(a, b)
        
        # === Diffusion coupling ===
        log_D = self.param(
            "log_diffusion",
            nn.initializers.constant(jnp.log(self.diffusion_init)),
            (self.d_model,)
        )
        D = jnp.exp(log_D)
        
        for _ in range(self.diffusion_steps):
            h_left = jnp.roll(h, 1, axis=-1)
            h_right = jnp.roll(h, -1, axis=-1)
            laplacian = h_left - 2*h + h_right
            h = h + D * laplacian
        
        return h


class ReactionDiffusionMinGRU(nn.Module):
    """Stacked Reaction-Diffusion minGRU."""
    d_model: int
    n_layers: int = 4
    diffusion_steps: int = 1
    diffusion_init: float = 0.1
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.d_model, name="input_proj")(x)
        
        for i in range(self.n_layers):
            residual = x
            x = nn.LayerNorm(name=f"ln_{i}")(x)
            x = ReactionDiffusionMinGRULayer(
                d_model=self.d_model,
                diffusion_steps=self.diffusion_steps,
                diffusion_init=self.diffusion_init,
                name=f"rd_gru_{i}"
            )(x)
            x = x + residual
        
        return x


class ReactionDiffusionClassifier(nn.Module):
    """Classifier for pathfinder."""
    d_model: int = 128
    n_layers: int = 4
    num_classes: int = 2
    diffusion_steps: int = 1
    diffusion_init: float = 0.1
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training=True):
        x = nn.Dense(self.d_model, name="embed")(x)
        
        x = ReactionDiffusionMinGRU(
            d_model=self.d_model,
            n_layers=self.n_layers,
            diffusion_steps=self.diffusion_steps,
            diffusion_init=self.diffusion_init,
            name="encoder"
        )(x)
        
        x = jnp.mean(x, axis=1)
        x = nn.LayerNorm(name="final_ln")(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.num_classes, name="classifier")(x)
        
        return x


# === Training Config ===

@dataclass
class TrainConfig:
    task: str = "path32"
    d_model: int = 128
    n_layers: int = 4
    diffusion_steps: int = 1
    diffusion_init: float = 0.1
    dropout_rate: float = 0.1
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_steps: int = 20000
    eval_every: int = 500
    data_dir: str = "./data"
    use_wandb: bool = False
    wandb_project: str = "reaction-diffusion-pathfinder"
    seed: int = 42
    
    @property
    def seq_len(self) -> int:
        return TASK_CONFIGS[self.task]["seq_len"]
    
    @property
    def image_size(self) -> int:
        return TASK_CONFIGS[self.task]["image_size"]
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["seq_len"] = self.seq_len
        d["image_size"] = self.image_size
        return d


# === Data Loading ===

def load_pathfinder_data(config: TrainConfig, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load from local npz files."""
    image_size = config.image_size
    
    split_map = {"validation": "val"}
    file_split = split_map.get(split, split)
    
    npz_path = Path(config.data_dir) / f"pathfinder{image_size}_{file_split}.npz"
    
    if not npz_path.exists():
        raise FileNotFoundError(f"Data not found: {npz_path}")
    
    print(f"Loading: {npz_path}")
    data = np.load(npz_path)
    
    images = data["images"].reshape(len(data["images"]), -1).astype(np.float32) / 255.0
    labels = data["labels"].astype(np.int32)
    print(f"Loaded {len(images)} samples")
    
    return images, labels


# === Training Functions ===

def create_train_state(rng, config: TrainConfig):
    model = ReactionDiffusionClassifier(
        d_model=config.d_model,
        n_layers=config.n_layers,
        num_classes=2,
        diffusion_steps=config.diffusion_steps,
        diffusion_init=config.diffusion_init,
        dropout_rate=config.dropout_rate,
    )
    
    dummy_input = jnp.ones((1, config.seq_len, 1))
    params = model.init(rng, dummy_input, training=False)
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {param_count:,}")
    
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=max(config.max_steps, config.warmup_steps + 1),
        end_value=config.learning_rate * 0.1,
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=config.weight_decay),
    )
    
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))


def accuracy(logits, labels):
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


@partial(jax.jit, static_argnums=(3,))
def train_step(state, batch, rng, dropout_rate):
    images, labels = batch
    images = images[..., None]
    
    def loss_fn(params):
        logits = state.apply_fn(params, images, training=True, rngs={"dropout": rng})
        loss = cross_entropy_loss(logits, labels)
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    acc = accuracy(logits, labels)
    
    return state, loss, acc


@jax.jit
def eval_step(state, batch):
    images, labels = batch
    images = images[..., None]
    logits = state.apply_fn(state.params, images, training=False)
    return cross_entropy_loss(logits, labels), accuracy(logits, labels)


def data_generator(images, labels, batch_size, shuffle=True):
    n = len(images)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_indices = indices[start:end]
        yield images[batch_indices], labels[batch_indices]


# === Main Training ===

def train(config: TrainConfig) -> Dict[str, Any]:
    print("=" * 60)
    print(f"Reaction-Diffusion minGRU on {config.task}")
    print(f"diffusion_steps={config.diffusion_steps}, D_init={config.diffusion_init}")
    print("=" * 60)
    
    if config.use_wandb and HAS_WANDB:
        wandb.init(
            project=config.wandb_project,
            config=config.to_dict(),
            name=f"rd_{config.task}_d{config.d_model}_diff{config.diffusion_steps}",
        )
    
    train_images, train_labels = load_pathfinder_data(config, "train")
    val_images, val_labels = load_pathfinder_data(config, "val")
    
    rng = random.PRNGKey(config.seed)
    rng, init_rng = random.split(rng)
    state = create_train_state(init_rng, config)
    
    best_val_acc = 0.0
    step = 0
    
    while step < config.max_steps:
        for batch in data_generator(train_images, train_labels, config.batch_size):
            if step >= config.max_steps:
                break
            
            rng, step_rng = random.split(rng)
            state, loss, acc = train_step(state, batch, step_rng, config.dropout_rate)
            
            if step % 100 == 0:
                print(f"Step {step:5d} | Loss: {loss:.4f} | Acc: {acc:.3f}")
                if config.use_wandb and HAS_WANDB:
                    wandb.log({"train/loss": float(loss), "train/acc": float(acc)}, step=step)
            
            if step > 0 and step % config.eval_every == 0:
                val_losses, val_accs = [], []
                for val_batch in data_generator(val_images, val_labels, config.batch_size, shuffle=False):
                    vl, va = eval_step(state, val_batch)
                    val_losses.append(float(vl))
                    val_accs.append(float(va))
                
                val_loss = np.mean(val_losses)
                val_acc = np.mean(val_accs)
                
                print(f"  [EVAL] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print(f"  [NEW BEST] {best_val_acc:.3f}")
                
                if config.use_wandb and HAS_WANDB:
                    wandb.log({"val/loss": val_loss, "val/acc": val_acc, "val/best_acc": best_val_acc}, step=step)
            
            step += 1
    
    print(f"\nFinal best val accuracy: {best_val_acc:.3f}")
    
    if config.use_wandb and HAS_WANDB:
        wandb.log({"final/best_acc": best_val_acc})
        wandb.finish()
    
    return {"best_val_acc": best_val_acc}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="path32", choices=["path32", "path64", "pathx"])
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--diffusion_steps", type=int, default=1)
    parser.add_argument("--diffusion_init", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    config = TrainConfig(
        task=args.task,
        d_model=args.d_model,
        n_layers=args.n_layers,
        diffusion_steps=args.diffusion_steps,
        diffusion_init=args.diffusion_init,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        use_wandb=args.wandb,
        seed=args.seed,
    )
    
    train(config)
