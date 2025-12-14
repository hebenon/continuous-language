"""
Hierarchical CfC in JAX/Flax for Path/Path-X with W&B tracking.

Usage:
    python train_pathx_jax.py --task path32 --coupling full --wandb
    python train_pathx_jax.py --task path32 --coupling all --wandb
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
import json
import time

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("wandb not installed. Run `pip install wandb` for experiment tracking.")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("datasets not installed. Run `pip install datasets` for HuggingFace data.")


# === Task Configurations ===

TASK_CONFIGS = {
    "path32": {
        "seq_len": 1024,
        "image_size": 32,
        "hf_dataset": "lra-benchmark/pathfinder32",
        "description": "Path 32x32 - faster iteration, still multi-scale",
    },
    "path64": {
        "seq_len": 4096,
        "image_size": 64,
        "hf_dataset": "lra-benchmark/pathfinder64",
        "description": "Path 64x64 - middle ground",
    },
    "pathx": {
        "seq_len": 16384,
        "image_size": 128,
        "hf_dataset": "lra-benchmark/pathfinder128",
        "description": "Path-X 128x128 - full benchmark",
    },
}


# === Hierarchical CfC Architecture ===

class CfCCell(nn.Module):
    """Single step of hierarchical CfC - to be wrapped with nn.scan."""
    
    d_fast: int
    d_slow: int
    τ_fast: jnp.ndarray
    τ_slow: jnp.ndarray
    hidden_mult: int = 2
    coupling: str = "full"
    
    @nn.compact
    def __call__(self, carry, x_t):
        h_fast, h_slow = carry
        fast_hidden = self.d_fast * self.hidden_mult
        slow_hidden = self.d_slow * self.hidden_mult
        
        # Input projection
        x_proj = nn.Dense(self.d_fast, name="input_proj")(x_t)
        
        # Cross-scale communication
        if self.coupling in ["full", "no_fast_to_slow"]:
            gate = nn.sigmoid(nn.Dense(self.d_slow, name="gate")(h_slow))
            gated_slow = h_slow * gate
        else:
            gated_slow = jnp.zeros_like(h_slow)
        
        if self.coupling in ["full", "no_slow_to_fast"]:
            fast_summary = nn.Dense(self.d_fast, name="pool")(h_fast)
        else:
            fast_summary = jnp.zeros_like(h_fast)
        
        # Fast update
        fast_input = jnp.concatenate([h_fast, gated_slow, x_proj], axis=-1)
        f_fast = nn.Dense(self.d_fast, name="f_fast_out")(
            nn.silu(nn.Dense(fast_hidden, name="f_fast_in")(fast_input))
        )
        g_fast = nn.Dense(self.d_fast, name="g_fast_out")(
            nn.silu(nn.Dense(fast_hidden, name="g_fast_in")(fast_input))
        )
        sigma_fast = nn.sigmoid(-f_fast / self.τ_fast)
        h_fast_new = sigma_fast * g_fast + (1 - sigma_fast) * h_fast
        
        # Slow update
        slow_input = jnp.concatenate([h_slow, fast_summary, x_proj], axis=-1)
        f_slow = nn.Dense(self.d_slow, name="f_slow_out")(
            nn.silu(nn.Dense(slow_hidden, name="f_slow_in")(slow_input))
        )
        g_slow = nn.Dense(self.d_slow, name="g_slow_out")(
            nn.silu(nn.Dense(slow_hidden, name="g_slow_in")(slow_input))
        )
        sigma_slow = nn.sigmoid(-f_slow / self.τ_slow)
        h_slow_new = sigma_slow * g_slow + (1 - sigma_slow) * h_slow
        
        out = jnp.concatenate([h_fast_new, h_slow_new], axis=-1)
        return (h_fast_new, h_slow_new), out


class HierarchicalCfC(nn.Module):
    """Two-compartment CfC with bidirectional cross-scale communication."""
    
    d_fast: int = 64
    d_slow: int = 64
    τ_fast_range: Tuple[float, float] = (1.0, 10.0)
    τ_slow_range: Tuple[float, float] = (10.0, 100.0)
    hidden_mult: int = 2
    coupling: str = "full"

    @nn.compact
    def __call__(self, x, return_all_states=False):
        batch_size, seq_len, _ = x.shape
        
        # Timescales
        τ_fast = jnp.exp(jnp.linspace(
            jnp.log(self.τ_fast_range[0]),
            jnp.log(self.τ_fast_range[1]),
            self.d_fast
        ))
        τ_slow = jnp.exp(jnp.linspace(
            jnp.log(self.τ_slow_range[0]),
            jnp.log(self.τ_slow_range[1]),
            self.d_slow
        ))
        
        # Wrap cell with nn.scan for parameter sharing across time
        ScanCell = nn.scan(
            CfCCell,
            variable_broadcast='params',
            split_rngs={'params': False},
            in_axes=1,   # scan over seq dim of input
            out_axes=1,  # stack outputs along seq dim
        )
        
        cell = ScanCell(
            d_fast=self.d_fast,
            d_slow=self.d_slow,
            τ_fast=τ_fast,
            τ_slow=τ_slow,
            hidden_mult=self.hidden_mult,
            coupling=self.coupling,
            name="cell",
        )
        
        # Initial state
        h_fast_init = jnp.zeros((batch_size, self.d_fast))
        h_slow_init = jnp.zeros((batch_size, self.d_slow))
        
        # Run scan
        (h_fast_final, h_slow_final), all_states = cell(
            (h_fast_init, h_slow_init), x
        )
        
        if return_all_states:
            return all_states  # (batch, seq, d_fast + d_slow)
        return jnp.concatenate([h_fast_final, h_slow_final], axis=-1)


class HierarchicalCfCClassifier(nn.Module):
    """Wrapper for sequence classification."""
    
    num_classes: int = 2
    d_fast: int = 64
    d_slow: int = 64
    τ_fast_range: Tuple[float, float] = (0.1, 1.0)
    τ_slow_range: Tuple[float, float] = (10.0, 100.0)
    dropout_rate: float = 0.1
    coupling: str = "full"
    
    @nn.compact
    def __call__(self, x, training=True):
        state = HierarchicalCfC(
            d_fast=self.d_fast, 
            d_slow=self.d_slow,
            τ_fast_range=self.τ_fast_range, 
            τ_slow_range=self.τ_slow_range,
            coupling=self.coupling,
        )(x)
        
        d_state = self.d_fast + self.d_slow
        x = nn.LayerNorm()(state)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(d_state)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        return nn.Dense(self.num_classes)(x)

# === Configuration ===

@dataclass
class TrainConfig:
    # Task
    task: str = "path32"
    
    # Model
    coupling: str = "full"
    d_fast: int = 64
    d_slow: int = 64
    τ_fast_range: Tuple[float, float] = (0.1, 1.0)
    τ_slow_range: Tuple[float, float] = (10.0, 100.0)
    dropout_rate: float = 0.1
    
    # Data
    data_dir: str = "./data"
    cache_dir: str = "./data/hf_cache"
    
    # Training
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 200
    warmup_epochs: int = 10
    grad_clip: float = 1.0
    
    # Infrastructure
    seed: int = 42
    log_every: int = 100
    eval_every: int = 1
    save_dir: str = "./checkpoints"
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "hierarchical-cfc"
    wandb_entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: Optional[Tuple[str, ...]] = None
    
    @property
    def seq_len(self) -> int:
        return TASK_CONFIGS[self.task]["seq_len"]
    
    @property
    def image_size(self) -> int:
        return TASK_CONFIGS[self.task]["image_size"]
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['τ_fast_range'] = list(d['τ_fast_range'])
        d['τ_slow_range'] = list(d['τ_slow_range'])
        d['seq_len'] = self.seq_len
        d['image_size'] = self.image_size
        if d['tags']:
            d['tags'] = list(d['tags'])
        return d


# === W&B Logger ===

class WandbLogger:
    def __init__(self, config: TrainConfig):
        self.enabled = config.use_wandb and HAS_WANDB
        
        if self.enabled:
            run_name = config.run_name or f"{config.task}_{config.coupling}_d{config.d_fast}"
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=run_name,
                config=config.to_dict(),
                tags=list(config.tags) if config.tags else [config.task, config.coupling],
            )
            print(f"W&B: {wandb.run.url}")
    
    def log(self, metrics: Dict, step: Optional[int] = None):
        if self.enabled:
            wandb.log(metrics, step=step)
    
    def log_summary(self, metrics: Dict):
        if self.enabled:
            for k, v in metrics.items():
                wandb.run.summary[k] = v
    
    def finish(self):
        if self.enabled:
            wandb.finish()


# === Data Loading ===

def load_from_huggingface(task: str, split: str, cache_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load pathfinder data from HuggingFace datasets."""
    
    if not HAS_DATASETS:
        raise ImportError("datasets library not installed. Run: pip install datasets")
    
    task_config = TASK_CONFIGS[task]
    hf_dataset = task_config["hf_dataset"]
    seq_len = task_config["seq_len"]
    
    print(f"Loading from HuggingFace: {hf_dataset} [{split}]")
    
    # Map split names (HF might use different names)
    split_map = {"val": "validation", "train": "train", "test": "test"}
    hf_split = split_map.get(split, split)
    
    ds = load_dataset(hf_dataset, split=hf_split, cache_dir=cache_dir, trust_remote_code=True)
    
    # Handle different possible column names
    image_col = None
    label_col = None
    
    for col in ['image', 'input', 'inputs', 'x']:
        if col in ds.column_names:
            image_col = col
            break
    
    for col in ['label', 'labels', 'target', 'y']:
        if col in ds.column_names:
            label_col = col
            break
    
    if image_col is None or label_col is None:
        print(f"Available columns: {ds.column_names}")
        raise ValueError(f"Could not find image/label columns in dataset")
    
    # Convert to numpy
    images = np.array(ds[image_col], dtype=np.float32)
    labels = np.array(ds[label_col], dtype=np.int32)
    
    # Normalize if needed (check if values are 0-255 or already 0-1)
    if images.max() > 1.0:
        images = images / 255.0
    
    # Flatten if 2D images
    if len(images.shape) == 3:
        images = images.reshape(len(images), -1)
    
    # Ensure correct sequence length
    if images.shape[1] != seq_len:
        print(f"Warning: Expected seq_len {seq_len}, got {images.shape[1]}")
    
    # Add channel dimension: (batch, seq) -> (batch, seq, 1)
    images = images[:, :, np.newaxis]
    
    print(f"Loaded {split}: {len(images)} samples, shape={images.shape}")
    return images, labels


def load_from_local(data_dir: str, task: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load pathfinder data from local files."""
    
    task_config = TASK_CONFIGS[task]
    image_size = task_config["image_size"]
    seq_len = task_config["seq_len"]
    data_path = Path(data_dir)
    
    # Try various path patterns
    patterns = [
        data_path / f"pathfinder{image_size}_{split}.npz",
        data_path / task / f"{split}.npz",
        data_path / f"lra_release/pathfinder{image_size}" / f"{split}.npz",
        data_path / f"pathfinder{image_size}" / f"{split}.npz",
    ]
    
    for npz_file in patterns:
        if npz_file.exists():
            data = np.load(npz_file)
            
            # Handle different key names
            image_key = next((k for k in ['images', 'image', 'x', 'inputs'] if k in data), None)
            label_key = next((k for k in ['labels', 'label', 'y', 'targets'] if k in data), None)
            
            if image_key and label_key:
                images = data[image_key].astype(np.float32)
                labels = data[label_key].astype(np.int32)
                
                if images.max() > 1.0:
                    images = images / 255.0
                if len(images.shape) == 3:
                    images = images.reshape(-1, seq_len)
                images = images[:, :, np.newaxis]
                
                print(f"Loaded {split}: {len(images)} samples from {npz_file}")
                return images, labels
    
    raise FileNotFoundError(f"No local data found for {task}/{split}. Tried: {patterns}")


def create_synthetic_data(num_samples: int, seq_len: int, seed: int = 42):
    """Synthetic data for pipeline testing."""
    np.random.seed(seed)
    images = np.random.rand(num_samples, seq_len, 1).astype(np.float32)
    labels = np.random.randint(0, 2, num_samples).astype(np.int32)
    print(f"Created synthetic data: {num_samples} samples (TESTING ONLY)")
    return images, labels


def load_pathfinder_data(
    task: str,
    split: str,
    data_dir: str = "./data",
    cache_dir: str = "./data/hf_cache",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pathfinder data with fallback chain:
    1. HuggingFace datasets (preferred)
    2. Local files
    3. Synthetic (for testing only)
    """
    
    # Try HuggingFace first
    if HAS_DATASETS:
        try:
            return load_from_huggingface(task, split, cache_dir)
        except Exception as e:
            print(f"HuggingFace loading failed: {e}")
    
    # Try local files
    try:
        return load_from_local(data_dir, task, split)
    except FileNotFoundError as e:
        print(f"Local loading failed: {e}")
    
    # Synthetic fallback
    print("" + "="*60)
    print("WARNING: Using synthetic data - results are NOT meaningful!")
    print("Install datasets library: pip install datasets")
    print("="*60 + "")
    
    seq_len = TASK_CONFIGS[task]["seq_len"]
    n_samples = {"train": 50000, "val": 10000, "test": 10000}.get(split, 1000)
    seed = {"train": 42, "val": 43, "test": 44}.get(split, 42)
    
    return create_synthetic_data(n_samples, seq_len, seed)


# === Training Infrastructure ===

def create_train_state(rng, config: TrainConfig, steps_per_epoch: int):
    model = HierarchicalCfCClassifier(
        d_fast=config.d_fast, d_slow=config.d_slow,
        τ_fast_range=config.τ_fast_range, τ_slow_range=config.τ_slow_range,
        dropout_rate=config.dropout_rate, coupling=config.coupling,
    )
    
    params = model.init(rng, jnp.ones((1, config.seq_len, 1)), training=False)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {n_params:,}")
    
    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.epochs * steps_per_epoch
    
    schedule = optax.join_schedules([
        optax.linear_schedule(0.0, config.lr, warmup_steps),
        optax.cosine_decay_schedule(config.lr, total_steps - warmup_steps),
    ], boundaries=[warmup_steps])
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay),
    )
    
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer), n_params


def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))


def accuracy(logits, labels):
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


@partial(jax.jit, static_argnums=(3,))
def train_step(state, batch, rng, dropout_rate):
    x, y = batch
    
    def loss_fn(params):
        logits = state.apply_fn(params, x, training=True, rngs={"dropout": rng})
        return cross_entropy_loss(logits, y), logits
    
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    
    return state.apply_gradients(grads=grads), {
        "loss": loss, "acc": accuracy(logits, y), "grad_norm": grad_norm
    }


@jax.jit
def eval_step(state, batch):
    x, y = batch
    logits = state.apply_fn(state.params, x, training=False)
    return {"loss": cross_entropy_loss(logits, y), "acc": accuracy(logits, y)}


def data_generator(images, labels, batch_size, rng, shuffle=True):
    n = len(images)
    indices = np.arange(n)
    if shuffle:
        indices = np.array(jax.random.permutation(rng, indices))
    
    for i in range(0, n, batch_size):
        batch_idx = indices[i:i + batch_size]
        yield jnp.array(images[batch_idx]), jnp.array(labels[batch_idx])


# === Training Loop ===

def train(config: TrainConfig) -> Dict[str, Any]:
    """Main training function."""
    
    rng = random.PRNGKey(config.seed)
    logger = WandbLogger(config)
    
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Task: {config.task} ({TASK_CONFIGS[config.task]['description']})")
    print(f"Sequence length: {config.seq_len}")
    print(f"Coupling: {config.coupling}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    train_images, train_labels = load_pathfinder_data(
        config.task, "train", config.data_dir, config.cache_dir
    )
    val_images, val_labels = load_pathfinder_data(
        config.task, "val", config.data_dir, config.cache_dir
    )
    test_images, test_labels = load_pathfinder_data(
        config.task, "test", config.data_dir, config.cache_dir
    )
    
    # Check if using synthetic data
    using_synthetic = len(train_images) <= 1000
    
    steps_per_epoch = len(train_images) // config.batch_size
    
    # Initialize
    rng, init_rng = random.split(rng)
    state, n_params = create_train_state(init_rng, config, steps_per_epoch)
    
    logger.log({
        "data/train_size": len(train_images),
        "data/val_size": len(val_images),
        "data/seq_len": config.seq_len,
        "model/n_params": n_params,
        "data/using_synthetic": using_synthetic,
    }, step=0)
    
    # Training
    best_val_acc = 0.0
    best_params = None
    global_step = 0
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        
        # Train
        rng, data_rng = random.split(rng)
        train_metrics = []
        
        for batch_idx, batch in enumerate(data_generator(
            train_images, train_labels, config.batch_size, data_rng
        )):
            rng, step_rng = random.split(rng)
            state, metrics = train_step(state, batch, step_rng, config.dropout_rate)
            train_metrics.append({k: float(v) for k, v in metrics.items()})
            
            if global_step % config.log_every == 0:
                logger.log({f"train/{k}": v for k, v in metrics.items()}, step=global_step)
                print(f"[{epoch}:{batch_idx}/{steps_per_epoch}] loss={metrics['loss']:.4f} acc={metrics['acc']:.4f}")
            
            global_step += 1
        
        epoch_time = time.time() - epoch_start
        avg_train = {k: np.mean([m[k] for m in train_metrics]) for k in train_metrics[0]}
        
        logger.log({
            "train/epoch_loss": avg_train["loss"],
            "train/epoch_acc": avg_train["acc"],
            "train/epoch_time": epoch_time,
            "epoch": epoch,
        }, step=global_step)
        
        # Validate
        if (epoch + 1) % config.eval_every == 0:
            rng, data_rng = random.split(rng)
            val_metrics = []
            
            for batch in data_generator(val_images, val_labels, config.batch_size, data_rng, shuffle=False):
                metrics = eval_step(state, batch)
                val_metrics.append({k: float(v) for k, v in metrics.items()})
            
            avg_val = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0]}
            
            logger.log({
                "val/loss": avg_val["loss"],
                "val/acc": avg_val["acc"],
                "epoch": epoch,
            }, step=global_step)
            
            print(f"Epoch {epoch} | Train: {avg_train['acc']:.4f} | Val: {avg_val['acc']:.4f} | {epoch_time:.1f}s")
            
            if avg_val["acc"] > best_val_acc:
                best_val_acc = avg_val["acc"]
                best_params = state.params
                print(f"  ↑ New best: {best_val_acc:.4f}")
    
    # Test
    print(f"\n{'='*60}")
    print("Final Test Evaluation")
    print(f"{'='*60}")
    
    if best_params is not None:
        state = state.replace(params=best_params)
    
    rng, data_rng = random.split(rng)
    test_metrics = []
    for batch in data_generator(test_images, test_labels, config.batch_size, data_rng, shuffle=False):
        metrics = eval_step(state, batch)
        test_metrics.append({k: float(v) for k, v in metrics.items()})
    
    avg_test = {k: np.mean([m[k] for m in test_metrics]) for k in test_metrics[0]}
    print(f"Test Acc: {avg_test['acc']:.4f}")
    
    logger.log_summary({
        "best_val_acc": best_val_acc,
        "test_acc": avg_test["acc"],
        "n_params": n_params,
    })
    
    results = {
        "task": config.task,
        "coupling": config.coupling,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(avg_test["acc"]),
        "n_params": n_params,
        "using_synthetic": using_synthetic,
        "config": config.to_dict(),
    }
    
    results_path = Path(config.save_dir) / f"{config.task}_{config.coupling}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.finish()
    return results


def run_ablation_suite(base_config: TrainConfig):
    """Run all ablation variants."""
    
    variants = ["full", "no_coupling", "no_slow_to_fast", "no_fast_to_slow"]
    results = {}
    
    for coupling in variants:
        print(f"\n{'#'*60}")
        print(f"# {coupling}")
        print(f"{'#'*60}")
        
        config = TrainConfig(
            task=base_config.task,
            coupling=coupling,
            d_fast=base_config.d_fast,
            d_slow=base_config.d_slow,
            τ_fast_range=base_config.τ_fast_range,
            τ_slow_range=base_config.τ_slow_range,
            epochs=base_config.epochs,
            batch_size=base_config.batch_size,
            lr=base_config.lr,
            seed=base_config.seed,
            data_dir=base_config.data_dir,
            cache_dir=base_config.cache_dir,
            save_dir=base_config.save_dir,
            use_wandb=base_config.use_wandb,
            wandb_project=base_config.wandb_project,
            tags=(base_config.task, coupling, "ablation"),
        )
        
        results[coupling] = train(config)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"ABLATION SUMMARY - {base_config.task}")
    print(f"{'='*60}")
    print(f"{'Variant':<20} {'Val':<10} {'Test':<10}")
    print("-" * 40)
    for v, r in results.items():
        print(f"{v:<20} {r['best_val_acc']:<10.4f} {r['test_acc']:<10.4f}")
    
    # Save summary
    summary_path = Path(base_config.save_dir) / f"{base_config.task}_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {summary_path}")
    
    return results


# === CLI ===

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hierarchical CfC on Path/Path-X")
    
    # Task & Model
    parser.add_argument("--task", type=str, default="path32", choices=["path32", "path64", "pathx"])
    parser.add_argument("--coupling", type=str, default="full",
                        choices=["full", "no_coupling", "no_slow_to_fast", "no_fast_to_slow", "all"])
    parser.add_argument("--d_fast", type=int, default=64)
    parser.add_argument("--d_slow", type=int, default=64)
    
    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    
    # Data & Output
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="hierarchical-cfc")
    
    args = parser.parse_args()
    
    config = TrainConfig(
        task=args.task,
        coupling=args.coupling,
        d_fast=args.d_fast,
        d_slow=args.d_slow,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )
    
    if args.coupling == "all":
        run_ablation_suite(config)
    else:
        train(config)
