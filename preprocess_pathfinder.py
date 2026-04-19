#!/usr/bin/env python3
"""
Memory-efficient preprocessing for LRA Pathfinder.
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

def count_samples(meta_dir: Path):
    """Count total samples without loading images."""
    total = 0
    for meta_file in meta_dir.glob("*.npy"):
        with open(meta_file, 'r') as f:
            total += sum(1 for line in f if len(line.strip().split()) >= 4)
    return total

def stream_pathfinder_data(base_dir: Path, size: int):
    """Generator that yields (image, label) one at a time."""
    
    data_dir = base_dir / f"pathfinder{size}" / "curv_baseline"
    imgs_dir = data_dir / "imgs"
    meta_dir = data_dir / "metadata"
    
    meta_files = sorted(meta_dir.glob("*.npy"), key=lambda x: int(x.stem))
    
    for meta_file in meta_files:
        with open(meta_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                
                img_path = imgs_dir / parts[0].replace("imgs/", "") / parts[1]
                label = int(parts[3])
                
                if not img_path.exists():
                    continue
                
                img = Image.open(img_path).convert('L')
                img_array = np.array(img, dtype=np.float32).flatten() / 255.0
                
                yield img_array, label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/lra_release/lra_release")
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--size", type=int, default=128, choices=[32, 64, 128, 256])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    input_dir = Path(args.input)
    output_dir = Path(args.output) / f"pathfinder{args.size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    meta_dir = input_dir / f"pathfinder{args.size}" / "curv_baseline" / "metadata"
    
    # Count samples first
    print("Counting samples...")
    n_total = count_samples(meta_dir)
    print(f"Total samples: {n_total}")
    
    # Create shuffled indices for splits
    indices = np.random.permutation(n_total)
    train_end = int(n_total * args.train_ratio)
    val_end = int(n_total * (args.train_ratio + args.val_ratio))
    
    split_assignment = np.empty(n_total, dtype='U5')
    split_assignment[indices[:train_end]] = 'train'
    split_assignment[indices[train_end:val_end]] = 'val'
    split_assignment[indices[val_end:]] = 'test'
    
    # Pre-allocate memory-mapped files (these become the final output)
    seq_len = args.size * args.size
    splits = {
        'train': {'n': train_end, 'idx': 0},
        'val': {'n': val_end - train_end, 'idx': 0},
        'test': {'n': n_total - val_end, 'idx': 0},
    }
    
    for name, info in splits.items():
        info['images'] = np.memmap(
            output_dir / f"{name}_images.npy",
            dtype=np.float32, mode='w+', shape=(info['n'], seq_len)
        )
        info['labels'] = np.memmap(
            output_dir / f"{name}_labels.npy",
            dtype=np.int32, mode='w+', shape=(info['n'],)
        )
    
    # Stream and assign to splits
    print("Processing images...")
    for i, (img, label) in enumerate(tqdm(
        stream_pathfinder_data(input_dir, args.size),
        total=n_total, desc="Loading"
    )):
        split = split_assignment[i]
        idx = splits[split]['idx']
        splits[split]['images'][idx] = img
        splits[split]['labels'][idx] = label
        splits[split]['idx'] += 1
    
    # Flush to disk
    print("\nFlushing to disk...")
    for name, info in splits.items():
        info['images'].flush()
        info['labels'].flush()
        print(f"  {name}: {info['n']} samples")
        
        # Save shape info for loading
        np.save(output_dir / f"{name}_meta.npy", {
            'n': info['n'],
            'seq_len': seq_len,
            'image_shape': (info['n'], seq_len),
            'label_shape': (info['n'],)
        })
    
    print("\nDone!")
    print(f"Files saved to {output_dir}/")
    print("  {split}_images.npy - memmap float32")
    print("  {split}_labels.npy - memmap int32")

if __name__ == "__main__":
    main()