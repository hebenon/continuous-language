"""
Hierarchical minGRU in JAX/Flax for Path-X experiments.

Based on "Were RNNs All We Needed?" (Feng et al., Oct 2024)
Extended with timescale-separated compartments.

Usage:
    python train_minGRU.py --task path32 --model vanilla --wandb
    python train_minGRU.py --task path32 --model hierarchical --wandb
"""

import jax
import jax.numpy as jnp
from jax import random, lax
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Tuple, Optional, Dict, Any, Literal
from dataclasses import dataclass, asdict, field
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


# === Parallel Scan Primitive ===

def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence.
    
    For recurrence: h_t = a_t * h_{t-1} + b_t
    State q_t = (a_t, b_t)
    Combining: q_i @ q_j = (a_i * a_j, a_j * b_i + b_j)
    """
    a_i, b_i = q_i
    a_j, b_j = q_j
    return a_i * a_j, a_j * b_i + b_j


def parallel_scan(a, b):
    """Compute linear recurrence h_t = a_t * h_{t-1} + b_t via parallel scan.
    
    Args:
        a: (batch, seq, dim) - multiplicative coefficients
        b: (batch, seq, dim) - additive coefficients
    
    Returns:
        h: (batch, seq, dim) - hidden states
    """
    # associative_scan operates on leading axis, so we need (seq, batch, dim)
    a = jnp.moveaxis(a, 1, 0)  # (seq, batch, dim)
    b = jnp.moveaxis(b, 1, 0)
    
    _, h = lax.associative_scan(binary_operator_diag, (a, b))
    
    h = jnp.moveaxis(h, 0, 1)  # back to (batch, seq, dim)
    return h

# === Vanilla minGRU ===

class MinGRULayer(nn.Module):
    """Single minGRU layer."""
    d_model: int
    
    @nn.compact
    def __call__(self, x):
        z_pre = nn.Dense(self.d_model, name="z_proj")(x)
        h_tilde = nn.Dense(self.d_model, name="h_proj")(x)
        
        z = nn.sigmoid(z_pre)
        a = 1 - z
        b = z * h_tilde
        
        return parallel_scan(a, b)

class MinGRU(nn.Module):
    """Minimal GRU - fully parallelizable via linear recurrence.
    
    From "Were RNNs All We Needed?" (Feng et al., 2024)
    
    Update equations:
        z_t = σ(Linear(x_t))           # gate only depends on input
        h̃_t = Linear(x_t)              # candidate (no tanh)
        h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
    
    Rearranged for parallel scan:
        a_t = 1 - z_t
        b_t = z_t * h̃_t
        h_t = a_t * h_{t-1} + b_t
    """
    
    d_model: int
    n_layers: int = 4
    
    @nn.compact
    def __call__(self, x):
        # Project input to model dimension
        x = nn.Dense(self.d_model, name="input_proj")(x)
        
        for i in range(self.n_layers):
            # Pre-norm residual block
            residual = x
            x = nn.LayerNorm(name=f"ln_{i}")(x)
            x = MinGRULayer(d_model=self.d_model, name=f"gru_{i}")(x)
            x = x + residual  # Residual connection
        
        return x

# === Hierarchical minGRU ===

class HierarchicalMinGRULayer(nn.Module):
    """Single hierarchical minGRU layer with fast/slow compartments."""
    
    d_fast: int = 64
    d_slow: int = 64
    z_bias_fast: float = 2.0
    z_bias_slow: float = -2.0
    coupling: str = "output"
    
    @nn.compact
    def __call__(self, x):
        batch, seq_len, d_in = x.shape
        
        # === Fast compartment ===
        z_fast_pre = nn.Dense(
            self.d_fast,
            name="z_fast",
            bias_init=nn.initializers.constant(self.z_bias_fast)
        )(x)
        h_tilde_fast = nn.Dense(self.d_fast, name="h_fast")(x)
        
        z_fast = nn.sigmoid(z_fast_pre)
        a_fast = 1 - z_fast
        b_fast = z_fast * h_tilde_fast
        
        h_fast = parallel_scan(a_fast, b_fast)
        
        # === Slow compartment ===
        if self.coupling == "input":
            h_fast_shifted = jnp.concatenate([
                jnp.zeros((batch, 1, self.d_fast)),
                h_fast[:, :-1, :]
            ], axis=1)
            x_slow = jnp.concatenate([x, h_fast_shifted], axis=-1)
        else:
            x_slow = x
        
        z_slow_pre = nn.Dense(
            self.d_slow,
            name="z_slow",
            bias_init=nn.initializers.constant(self.z_bias_slow)
        )(x_slow)
        h_tilde_slow = nn.Dense(self.d_slow, name="h_slow")(x_slow)
        
        z_slow = nn.sigmoid(z_slow_pre)
        a_slow = 1 - z_slow
        b_slow = z_slow * h_tilde_slow
        
        h_slow = parallel_scan(a_slow, b_slow)
        
        # === Combine ===
        h_combined = jnp.concatenate([h_fast, h_slow], axis=-1)
        
        if self.coupling == "output":
            gate = nn.sigmoid(nn.Dense(self.d_fast + self.d_slow, name="output_gate")(h_combined))
            h_combined = gate * h_combined
        
        return h_combined


class HierarchicalMinGRU(nn.Module):
    """Stacked hierarchical minGRU with layer norm and residuals."""
    
    d_fast: int = 64
    d_slow: int = 64
    z_bias_fast: float = 2.0
    z_bias_slow: float = -2.0
    coupling: str = "output"
    n_layers: int = 4
    
    @nn.compact
    def __call__(self, x):
        d_model = self.d_fast + self.d_slow
        
        # Project input to model dimension
        x = nn.Dense(d_model, name="input_proj")(x)
        
        for i in range(self.n_layers):
            residual = x
            x = nn.LayerNorm(name=f"ln_{i}")(x)
            x = HierarchicalMinGRULayer(
                d_fast=self.d_fast,
                d_slow=self.d_slow,
                z_bias_fast=self.z_bias_fast,
                z_bias_slow=self.z_bias_slow,
                coupling=self.coupling,
                name=f"layer_{i}"
            )(x)
            x = x + residual
        
        return x


# === Classifier Wrapper ===

class MinGRUClassifier(nn.Module):
    """Sequence classifier using minGRU variants."""
    
    num_classes: int = 2
    d_model: int = 128
    model_type: Literal["vanilla", "hierarchical"] = "vanilla"
    # Hierarchical params
    d_fast: int = 64
    d_slow: int = 64
    z_bias_fast: float = 2.0
    z_bias_slow: float = -2.0
    coupling: Literal["none", "output", "input"] = "output"
    dropout_rate: float = 0.1
    n_layers: int = 4
    
    @nn.compact
    def __call__(self, x, training=True):
        if self.model_type == "vanilla":
            h = MinGRU(d_model=self.d_model, n_layers=self.n_layers)(x)
            d_out = self.d_model
        else:
            h = HierarchicalMinGRU(
                d_fast=self.d_fast,
                d_slow=self.d_slow,
                z_bias_fast=self.z_bias_fast,
                z_bias_slow=self.z_bias_slow,
                coupling=self.coupling,
                n_layers=self.n_layers,
            )(x)
            d_out = self.d_fast + self.d_slow
        
        # Take final hidden state
        h_final = h[:, -1, :]
        
        # Classification head
        h_final = nn.LayerNorm()(h_final)
        h_final = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(h_final)
        h_final = nn.Dense(d_out)(h_final)
        h_final = nn.gelu(h_final)
        h_final = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(h_final)
        
        return nn.Dense(self.num_classes)(h_final)
    
# === Configuration ===

@dataclass
class TrainConfig:
    # Task
    task: str = "path32"
    
    # Model
    model_type: str = "vanilla"  # "vanilla" or "hierarchical"
    d_model: int = 128  # for vanilla
    d_fast: int = 64    # for hierarchical
    d_slow: int = 64
    z_bias_fast: float = 2.0
    z_bias_slow: float = -2.0
    coupling: str = "output"  # "none", "output", "input"
    dropout_rate: float = 0.1
    n_layers: int = 4
    
    # Training
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 100
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    
    # Data
    data_dir: str = "./data"
    cache_dir: str = "./data/hf_cache"
    
    # Infrastructure
    seed: int = 42
    log_every: int = 100
    eval_every: int = 1
    save_dir: str = "./checkpoints"
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "hierarchical-minGRU"
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
            run_name = config.run_name or f"{config.task}_{config.model_type}_d{config.d_model if config.model_type == 'vanilla' else config.d_fast}"
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=run_name,
                config=config.to_dict(),
                tags=list(config.tags) if config.tags else [config.task, config.model_type],
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
    print("\n" + "="*60)
    print("WARNING: Using synthetic data - results are NOT meaningful!")
    print("Install datasets library: pip install datasets")
    print("="*60 + "\n")
    
    seq_len = TASK_CONFIGS[task]["seq_len"]
    n_samples = {"train": 50000, "val": 10000, "test": 10000}.get(split, 1000)
    seed = {"train": 42, "val": 43, "test": 44}.get(split, 42)
    
    return create_synthetic_data(n_samples, seq_len, seed)


def data_generator(images, labels, batch_size, rng, shuffle=True):
    n = len(images)
    indices = np.arange(n)
    if shuffle:
        indices = np.array(jax.random.permutation(rng, indices))
    
    for i in range(0, n, batch_size):
        batch_idx = indices[i:i + batch_size]
        yield jnp.array(images[batch_idx]), jnp.array(labels[batch_idx])

# === Training Infrastructure ===

def create_train_state(rng, config: TrainConfig, steps_per_epoch: int):
    model = MinGRUClassifier(
        model_type=config.model_type,
        d_model=config.d_model,
        d_fast=config.d_fast,
        d_slow=config.d_slow,
        z_bias_fast=config.z_bias_fast,
        z_bias_slow=config.z_bias_slow,
        coupling=config.coupling,
        dropout_rate=config.dropout_rate,
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

# === Training Loop ===

def train(config: TrainConfig) -> Dict[str, Any]:
    """Main training function."""
    
    rng = random.PRNGKey(config.seed)
    logger = WandbLogger(config)
    
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Task: {config.task} ({TASK_CONFIGS[config.task]['description']})")
    print(f"Model: {config.model_type}")
    if config.model_type == "hierarchical":
        print(f"  d_fast={config.d_fast}, d_slow={config.d_slow}")
        print(f"  z_bias_fast={config.z_bias_fast}, z_bias_slow={config.z_bias_slow}")
        print(f"  coupling={config.coupling}")
    else:
        print(f"  d_model={config.d_model}")
    print(f"Sequence length: {config.seq_len}")
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
        "model_type": config.model_type,
        "coupling": config.coupling,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(avg_test["acc"]),
        "n_params": n_params,
        "using_synthetic": using_synthetic,
        "config": config.to_dict(),
    }
    
    results_path = Path(config.save_dir) / f"{config.task}_{config.model_type}_{config.coupling}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.finish()
    return results

# === Validation ===

# Add after the parallel_scan function, run before training
def test_parallel_scan():
    """Verify parallel scan matches sequential computation."""
    key = jax.random.PRNGKey(0)
    batch, seq, dim = 2, 16, 4
    
    a = jax.random.uniform(key, (batch, seq, dim), minval=0.5, maxval=0.99)
    b = jax.random.normal(key, (batch, seq, dim))
    
    # Parallel
    h_parallel = parallel_scan(a, b)
    
    # Sequential (ground truth)
    h_seq = []
    h_t = jnp.zeros((batch, dim))
    for t in range(seq):
        h_t = a[:, t] * h_t + b[:, t]
        h_seq.append(h_t)
    h_sequential = jnp.stack(h_seq, axis=1)
    
    max_diff = jnp.max(jnp.abs(h_parallel - h_sequential))
    print(f"Parallel scan test: max diff = {max_diff:.2e}")
    assert max_diff < 1e-5, f"MISMATCH: {max_diff}"
    print("✓ Parallel scan OK")


# === CLI ===

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hierarchical minGRU on Path/Path-X")
    
    # Task & Model
    parser.add_argument("--task", type=str, default="path32", choices=["path32", "path64", "pathx"])
    parser.add_argument("--model", type=str, default="vanilla", choices=["vanilla", "hierarchical"])
    parser.add_argument("--d_model", type=int, default=128, help="Hidden dim for vanilla")
    parser.add_argument("--d_fast", type=int, default=64)
    parser.add_argument("--d_slow", type=int, default=64)
    parser.add_argument("--z_bias_fast", type=float, default=2.0)
    parser.add_argument("--z_bias_slow", type=float, default=-2.0)
    parser.add_argument("--coupling", type=str, default="output", choices=["none", "output", "input"])
    parser.add_argument("--n_layers", type=int, default=4)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    
    # Data & Output
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="hierarchical-minGRU")
    
    args = parser.parse_args()
    
    test_parallel_scan()

    config = TrainConfig(
        task=args.task,
        model_type=args.model,
        d_model=args.d_model,
        d_fast=args.d_fast,
        d_slow=args.d_slow,
        z_bias_fast=args.z_bias_fast,
        z_bias_slow=args.z_bias_slow,
        coupling=args.coupling,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )
    
    train(config)