"""
Hierarchical minGRU in JAX/Flax for Path-X experiments.

Based on "Were RNNs All We Needed?" (Feng et al., Oct 2024)
Extended with timescale-separated compartments and wave dynamics.

Usage:
    python train_minGRU.py --task path32 --model vanilla --wandb
    python train_minGRU.py --task path32 --model multiscale --wandb
    python train_minGRU.py --task path32 --model wave --wandb
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


# === Multi-Scale minGRU (Generalized) ===

class MultiScaleMinGRULayer(nn.Module):
    """Generalized multi-timescale minGRU layer with N timescale buckets."""
    
    d_model: int = 128
    n_scales: int = 2
    z_bias_min: float = -2.0  # slowest (long memory)
    z_bias_max: float = 2.0   # fastest (short memory)
    coupling: str = "output"
    
    @nn.compact
    def __call__(self, x):
        batch, seq_len, d_in = x.shape
        
        # Distribute dimensions across timescale buckets
        d_per_scale = self.d_model // self.n_scales
        remainder = self.d_model % self.n_scales
        
        # Generate z_bias values: evenly spaced from slow to fast
        if self.n_scales == 1:
            z_biases = [0.0]  # neutral
        else:
            z_biases = np.linspace(self.z_bias_min, self.z_bias_max, self.n_scales)
        
        h_scales = []
        for i in range(self.n_scales):
            # Last bucket gets remainder dimensions
            d_this_scale = d_per_scale + (remainder if i == self.n_scales - 1 else 0)
            
            z_pre = nn.Dense(
                d_this_scale,
                name=f"z_scale_{i}",
                bias_init=nn.initializers.constant(float(z_biases[i]))
            )(x)
            h_tilde = nn.Dense(d_this_scale, name=f"h_scale_{i}")(x)
            
            z = nn.sigmoid(z_pre)
            a = 1 - z
            b = z * h_tilde
            
            h_scales.append(parallel_scan(a, b))
        
        # Concatenate all timescales
        h_combined = jnp.concatenate(h_scales, axis=-1)
        
        if self.coupling == "output":
            gate = nn.sigmoid(nn.Dense(self.d_model, name="output_gate")(h_combined))
            h_combined = gate * h_combined
        
        return h_combined


class MultiScaleMinGRU(nn.Module):
    """Stacked multi-scale minGRU with layer norm and residuals."""
    
    d_model: int = 128
    n_scales: int = 2
    z_bias_min: float = -2.0
    z_bias_max: float = 2.0
    coupling: str = "output"
    n_layers: int = 4
    
    @nn.compact
    def __call__(self, x):
        # Project input to model dimension
        x = nn.Dense(self.d_model, name="input_proj")(x)
        
        for i in range(self.n_layers):
            residual = x
            x = nn.LayerNorm(name=f"ln_{i}")(x)
            x = MultiScaleMinGRULayer(
                d_model=self.d_model,
                n_scales=self.n_scales,
                z_bias_min=self.z_bias_min,
                z_bias_max=self.z_bias_max,
                coupling=self.coupling,
                name=f"layer_{i}"
            )(x)
            x = x + residual
        
        return x


# === Wave Dynamics (Coupled Harmonic Oscillators) ===

class WaveDynamicsLayer(nn.Module):
    """Damped harmonic oscillator layer using complex-valued parallel scan.
    
    Each dimension is a discrete damped oscillator:
        s_t = a * s_{t-1} + b_t
    where a = r * exp(i*theta) is a complex decay-rotation coefficient.
    
    - r ∈ (0, 1): decay rate (controls memory length)
    - θ ∈ (0, π): oscillation frequency
    - b_t: input-dependent complex driving force
    
    Dimensions are split across n_scales oscillator groups, each with
    different (r, θ) pairs — from slow/long-memory to fast/short-memory.
    
    The existing parallel_scan (associative scan) works unchanged with
    complex dtypes since the binary operator uses element-wise multiply.
    """
    
    d_model: int = 128
    n_scales: int = 2
    r_min: float = 0.9      # fastest decay (shortest memory)
    r_max: float = 0.999    # slowest decay (longest memory)
    theta_min: float = 0.01  # slowest oscillation
    theta_max: float = 1.0   # fastest oscillation (< π for Nyquist)
    
    @nn.compact
    def __call__(self, x):
        batch, seq_len, d_in = x.shape
        
        d_per_scale = self.d_model // self.n_scales
        remainder = self.d_model % self.n_scales
        
        # Fixed (r, θ) per scale: linspace from long-memory/slow to short-memory/fast
        if self.n_scales == 1:
            r_values = [(self.r_min + self.r_max) / 2]
            theta_values = [(self.theta_min + self.theta_max) / 2]
        else:
            r_values = np.linspace(self.r_max, self.r_min, self.n_scales)  # long → short
            theta_values = np.linspace(self.theta_min, self.theta_max, self.n_scales)
        
        h_scales = []
        for i in range(self.n_scales):
            d_this_scale = d_per_scale + (remainder if i == self.n_scales - 1 else 0)
            
            r = float(r_values[i])
            theta = float(theta_values[i])
            
            # Complex decay-rotation coefficient (fixed per scale, broadcast across seq)
            a_complex = r * jnp.exp(1j * theta)
            a = jnp.full((batch, seq_len, d_this_scale), a_complex, dtype=jnp.complex64)
            
            # Input-dependent driving force: project to real + imag components
            drive_re = nn.Dense(d_this_scale, name=f"drive_re_{i}")(x)
            drive_im = nn.Dense(d_this_scale, name=f"drive_im_{i}")(x)
            b = (drive_re + 1j * drive_im).astype(jnp.complex64)
            # Scale drive by (1 - r) to normalize: when r≈1, drive should be small
            b = b * (1 - r)
            
            # Parallel scan (reuses existing infrastructure — works with complex)
            state = parallel_scan(a, b)  # (batch, seq, d_this_scale), complex64
            
            # Extract real part (position); imaginary part is velocity
            h_real = state.real  # (batch, seq, d_this_scale)
            h_scales.append(h_real)
        
        h_combined = jnp.concatenate(h_scales, axis=-1)
        
        # Output gate (learnable mixing)
        gate = nn.sigmoid(nn.Dense(self.d_model, name="output_gate")(h_combined))
        return gate * h_combined


class WaveDynamicsMinGRU(nn.Module):
    """Stacked wave dynamics layers with layer norm and residuals."""
    
    d_model: int = 128
    n_scales: int = 2
    r_min: float = 0.9
    r_max: float = 0.999
    theta_min: float = 0.01
    theta_max: float = 1.0
    n_layers: int = 4
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.d_model, name="input_proj")(x)
        
        for i in range(self.n_layers):
            residual = x
            x = nn.LayerNorm(name=f"ln_{i}")(x)
            x = WaveDynamicsLayer(
                d_model=self.d_model,
                n_scales=self.n_scales,
                r_min=self.r_min,
                r_max=self.r_max,
                theta_min=self.theta_min,
                theta_max=self.theta_max,
                name=f"layer_{i}",
            )(x)
            x = x + residual
        
        return x


# === Classifier Wrapper ===

class MinGRUClassifier(nn.Module):
    """Sequence classifier using minGRU variants."""
    
    num_classes: int = 2
    d_model: int = 128
    model_type: str = "vanilla"  # "vanilla", "multiscale", or "wave"
    n_layers: int = 4
    # Multi-scale params
    n_scales: int = 2
    z_bias_min: float = -2.0
    z_bias_max: float = 2.0
    coupling: str = "output"
    # Wave dynamics params
    r_min: float = 0.9
    r_max: float = 0.999
    theta_min: float = 0.01
    theta_max: float = 1.0
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training=True):
        if self.model_type == "vanilla":
            h = MinGRU(d_model=self.d_model, n_layers=self.n_layers)(x)
        elif self.model_type == "wave":
            h = WaveDynamicsMinGRU(
                d_model=self.d_model,
                n_scales=self.n_scales,
                r_min=self.r_min,
                r_max=self.r_max,
                theta_min=self.theta_min,
                theta_max=self.theta_max,
                n_layers=self.n_layers,
            )(x)
        else:  # "multiscale" (or "hierarchical" for backwards compat)
            h = MultiScaleMinGRU(
                d_model=self.d_model,
                n_scales=self.n_scales,
                z_bias_min=self.z_bias_min,
                z_bias_max=self.z_bias_max,
                coupling=self.coupling,
                n_layers=self.n_layers,
            )(x)
        
        # Take final hidden state
        h_final = h[:, -1, :]
        
        # Classification head
        h_final = nn.LayerNorm()(h_final)
        h_final = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(h_final)
        h_final = nn.Dense(self.d_model)(h_final)
        h_final = nn.gelu(h_final)
        h_final = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(h_final)
        
        return nn.Dense(self.num_classes)(h_final)
    
# === Configuration ===

@dataclass
class TrainConfig:
    # Task
    task: str = "path32"
    
    # Model
    model_type: str = "vanilla"  # "vanilla", "multiscale", or "wave"
    d_model: int = 128
    n_layers: int = 4
    # Multi-scale params
    n_scales: int = 2
    z_bias_min: float = -2.0
    z_bias_max: float = 2.0
    coupling: str = "output"
    # Wave dynamics params
    r_min: float = 0.9
    r_max: float = 0.999
    theta_min: float = 0.01
    theta_max: float = 1.0
    dropout_rate: float = 0.1
    
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
            if config.model_type == "vanilla":
                run_name = config.run_name or f"{config.task}_vanilla_d{config.d_model}"
            elif config.model_type == "wave":
                run_name = config.run_name or f"{config.task}_wave_d{config.d_model}_s{config.n_scales}_r{config.r_min}-{config.r_max}"
            else:
                run_name = config.run_name or f"{config.task}_ms_d{config.d_model}_s{config.n_scales}_z{abs(config.z_bias_min):.0f}"
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


def load_pathfinder_memmap(data_dir: Path, split: str, size: int):
    """Load pathfinder data from memmap files."""
    path = data_dir / f"pathfinder{size}"
    
    # Load metadata to get shapes
    meta = np.load(path / f"{split}_meta.npy", allow_pickle=True).item()
    
    images = np.memmap(
        path / f"{split}_images.npy",
        dtype=np.float32, mode='r', shape=meta['image_shape']
    )
    labels = np.memmap(
        path / f"{split}_labels.npy", 
        dtype=np.int32, mode='r', shape=meta['label_shape']
    )
    
    return images, labels


def load_from_local(data_dir: str, task: str, split: str):
    """Load data from local files (memmap or npz format)."""
    data_path = Path(data_dir)
    task_config = TASK_CONFIGS[task]
    image_size = task_config.get("image_size", 32)
    
    # Try memmap format first (for large datasets)
    memmap_dir = data_path / f"pathfinder{image_size}"
    meta_file = memmap_dir / f"{split}_meta.npy"
    
    if meta_file.exists():
        print(f"Loading {split} from memmap files...")
        meta = np.load(meta_file, allow_pickle=True).item()
        
        images = np.memmap(
            memmap_dir / f"{split}_images.npy",
            dtype=np.float32, mode='r', shape=meta['image_shape']
        )
        labels = np.memmap(
            memmap_dir / f"{split}_labels.npy",
            dtype=np.int32, mode='r', shape=meta['label_shape']
        )
        # Add channel dimension: (batch, seq) -> (batch, seq, 1)
        images = images[:, :, np.newaxis]
        return images, labels
    
    # Fall back to npz format
    patterns = [
        data_path / f"pathfinder{image_size}" / f"{split}.npz",
        data_path / f"lra_release/pathfinder{image_size}" / f"{split}.npz",
        data_path / task / f"{split}.npz",
    ]
    
    for pattern in patterns:
        if pattern.exists():
            print(f"Loading {split} from {pattern}")
            data = np.load(pattern)
            images = data['images']
            # Add channel dimension if needed
            if len(images.shape) == 2:
                images = images[:, :, np.newaxis]
            return images, data['labels']
    
    raise FileNotFoundError(f"No data found for {task}/{split}. Tried:\n  " + 
                            "\n  ".join(str(p) for p in [meta_file] + patterns))


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
        n_layers=config.n_layers,
        n_scales=config.n_scales,
        z_bias_min=config.z_bias_min,
        z_bias_max=config.z_bias_max,
        coupling=config.coupling,
        r_min=config.r_min,
        r_max=config.r_max,
        theta_min=config.theta_min,
        theta_max=config.theta_max,
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
    print(f"  d_model={config.d_model}")
    print(f"  n_layers={config.n_layers}")
    if config.model_type in ("multiscale", "hierarchical"):
        print(f"  n_scales={config.n_scales}")
        print(f"  z_bias_range=[{config.z_bias_min}, {config.z_bias_max}]")
        print(f"  coupling={config.coupling}")
        # Show actual timescale distribution
        if config.n_scales > 1:
            import numpy as np
            biases = np.linspace(config.z_bias_min, config.z_bias_max, config.n_scales)
            gates = 1 / (1 + np.exp(-biases))  # sigmoid
            print(f"  timescales: {['%.0f%%' % (g*100) for g in gates]} new info/step")
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
    parser.add_argument("--model", type=str, default="vanilla", choices=["vanilla", "multiscale", "hierarchical", "wave"])
    parser.add_argument("--n_scales", type=int, default=2, help="Number of timescale buckets")
    parser.add_argument("--z_bias_min", type=float, default=-2.0, help="Slowest timescale bias")
    parser.add_argument("--z_bias_max", type=float, default=2.0, help="Fastest timescale bias")
    parser.add_argument("--d_model", type=int, default=128, help="Hidden dim for vanilla")
    #parser.add_argument("--d_fast", type=int, default=64)
    #parser.add_argument("--d_slow", type=int, default=64)
    #parser.add_argument("--z_bias_fast", type=float, default=2.0)
    #parser.add_argument("--z_bias_slow", type=float, default=-2.0)
    parser.add_argument("--coupling", type=str, default="output", choices=["none", "output", "input"])
    # Wave dynamics params
    parser.add_argument("--r_min", type=float, default=0.9, help="Fastest decay (shortest memory)")
    parser.add_argument("--r_max", type=float, default=0.999, help="Slowest decay (longest memory)")
    parser.add_argument("--theta_min", type=float, default=0.01, help="Slowest oscillation frequency")
    parser.add_argument("--theta_max", type=float, default=1.0, help="Fastest oscillation frequency")
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

    model_type = args.model if args.model != "hierarchical" else "multiscale"

    config = TrainConfig(
        task=args.task,
        model_type=model_type,
        n_scales=args.n_scales,
        z_bias_min=args.z_bias_min,
        z_bias_max=args.z_bias_max,
        d_model=args.d_model,
        coupling=args.coupling,
        r_min=args.r_min,
        r_max=args.r_max,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
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