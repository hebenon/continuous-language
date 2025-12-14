"""
Pathfinder Dataset Generator

Task: Given an image with two marked endpoints and multiple curvy dashed paths,
classify whether the endpoints are connected by a single path.
"""

import numpy as np
from pathlib import Path
import argparse
from typing import Tuple
from scipy.ndimage import gaussian_filter


def generate_bezier_path(
    start: Tuple[float, float],
    end: Tuple[float, float],
    num_points: int = 50,
    curvature: float = 0.3,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate a smooth curvy path using cubic Bezier curves."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Control points with random perturbation
    t = np.linspace(0, 1, num_points)
    
    # Two control points for cubic Bezier
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    
    # Perpendicular direction for control point offset
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.sqrt(dx**2 + dy**2) + 1e-6
    perp_x, perp_y = -dy / length, dx / length
    
    # Random offset for curvature
    offset1 = rng.uniform(-curvature, curvature) * length
    offset2 = rng.uniform(-curvature, curvature) * length
    
    ctrl1 = (start[0] + dx * 0.33 + perp_x * offset1, 
             start[1] + dy * 0.33 + perp_y * offset1)
    ctrl2 = (start[0] + dx * 0.66 + perp_x * offset2,
             start[1] + dy * 0.66 + perp_y * offset2)
    
    # Cubic Bezier formula
    points = np.zeros((num_points, 2))
    for i, ti in enumerate(t):
        b0 = (1 - ti) ** 3
        b1 = 3 * (1 - ti) ** 2 * ti
        b2 = 3 * (1 - ti) * ti ** 2
        b3 = ti ** 3
        points[i, 0] = b0 * start[0] + b1 * ctrl1[0] + b2 * ctrl2[0] + b3 * end[0]
        points[i, 1] = b0 * start[1] + b1 * ctrl1[1] + b2 * ctrl2[1] + b3 * end[1]
    
    return points


def draw_dashed_line(
    img: np.ndarray,
    points: np.ndarray,
    dash_len: int = 3,
    gap_len: int = 2,
    thickness: float = 1.0,
) -> np.ndarray:
    """Draw a dashed line on the image."""
    h, w = img.shape
    
    for i in range(len(points) - 1):
        # Dashing pattern
        if (i // dash_len) % 2 == 1:
            continue
            
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        
        # Simple line drawing (Bresenham-ish)
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for s in range(int(steps) + 1):
            t = s / max(steps, 1)
            x = int(x0 + t * (x1 - x0))
            y = int(y0 + t * (y1 - y0))
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = min(1.0, img[y, x] + thickness)
    
    return img


def draw_dot(img: np.ndarray, pos: Tuple[int, int], radius: int = 2) -> np.ndarray:
    """Draw a filled circle (endpoint marker)."""
    h, w = img.shape
    y, x = int(pos[1]), int(pos[0])
    
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx**2 + dy**2 <= radius**2:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    img[ny, nx] = 1.0
    return img


def generate_pathfinder_sample(
    size: int = 32,
    n_distractors: int = 2,
    connected: bool = True,
    rng: np.random.Generator = None,
    min_path_length: float = 0.3,
) -> np.ndarray:
    """Generate a single pathfinder sample."""
    if rng is None:
        rng = np.random.default_rng()
    
    img = np.zeros((size, size), dtype=np.float32)
    margin = size // 8
    
    # Generate two endpoint positions (not too close)
    while True:
        p1 = (rng.integers(margin, size - margin), rng.integers(margin, size - margin))
        p2 = (rng.integers(margin, size - margin), rng.integers(margin, size - margin))
        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        if dist > size * min_path_length:
            break
    
    # Draw connecting path if connected
    if connected:
        path = generate_bezier_path(p1, p2, num_points=size * 2, curvature=0.4, rng=rng)
        img = draw_dashed_line(img, path, dash_len=3, gap_len=2)
    
    # Draw distractor paths
    for _ in range(n_distractors):
        # Random start/end for distractors (avoiding the main endpoints)
        while True:
            d1 = (rng.integers(margin, size - margin), rng.integers(margin, size - margin))
            d2 = (rng.integers(margin, size - margin), rng.integers(margin, size - margin))
            dist_d = np.sqrt((d1[0] - d2[0])**2 + (d1[1] - d2[1])**2)
            
            # Make sure distractor doesn't start/end too close to main endpoints
            dist_to_p1 = min(np.sqrt((d1[0] - p1[0])**2 + (d1[1] - p1[1])**2),
                           np.sqrt((d2[0] - p1[0])**2 + (d2[1] - p1[1])**2))
            dist_to_p2 = min(np.sqrt((d1[0] - p2[0])**2 + (d1[1] - p2[1])**2),
                           np.sqrt((d2[0] - p2[0])**2 + (d2[1] - p2[1])**2))
            
            if dist_d > size * 0.2 and dist_to_p1 > margin and dist_to_p2 > margin:
                break
        
        path = generate_bezier_path(d1, d2, num_points=size * 2, curvature=0.5, rng=rng)
        img = draw_dashed_line(img, path, dash_len=3, gap_len=2)
    
    # Draw endpoint markers
    img = draw_dot(img, p1, radius=max(1, size // 16))
    img = draw_dot(img, p2, radius=max(1, size // 16))
    
    # Add slight blur for more natural appearance
    img = gaussian_filter(img, sigma=0.5)
    img = np.clip(img, 0, 1)
    
    return img


def generate_dataset(
    n_samples: int,
    size: int = 32,
    n_distractors: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate full dataset."""
    rng = np.random.default_rng(seed)
    
    images = []
    labels = []
    
    for i in range(n_samples):
        connected = rng.random() < 0.5
        img = generate_pathfinder_sample(
            size=size,
            n_distractors=n_distractors,
            connected=connected,
            rng=rng,
        )
        images.append(img)
        labels.append(1 if connected else 0)
        
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{n_samples} samples")
    
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description="Generate Pathfinder dataset")
    parser.add_argument("--size", type=int, default=32, help="Image size (32, 64, or 128)")
    parser.add_argument("--n_train", type=int, default=50000)
    parser.add_argument("--n_val", type=int, default=10000)
    parser.add_argument("--n_test", type=int, default=10000)
    parser.add_argument("--n_distractors", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating Pathfinder-{args.size} dataset...")
    
    # Generate splits
    for split, n_samples, seed_offset in [
        ("train", args.n_train, 0),
        ("val", args.n_val, 1),
        ("test", args.n_test, 2),
    ]:
        print(f"\n{split}: {n_samples} samples")
        images, labels = generate_dataset(
            n_samples=n_samples,
            size=args.size,
            n_distractors=args.n_distractors,
            seed=args.seed + seed_offset,
        )
        
        # Save as NPZ
        output_file = output_dir / f"pathfinder{args.size}_{split}.npz"
        np.savez_compressed(output_file, images=images, labels=labels)
        print(f"Saved to {output_file}")
        print(f"  Shape: {images.shape}, Labels: {labels.sum()}/{len(labels)} connected")


if __name__ == "__main__":
    main()
