"""
Gate Sparsity Diagnostic for GatedWave.

Loads a checkpoint and reports gate bias values per scale per layer.
Gate bias is a proxy for prior firing rate (how often the gate resets
the oscillatory state). Initialized to -4.0 (sigmoid ≈ 0.018 = very sparse).

Expected pattern for a healthy multi-scale architecture:
  Scale 0 (integrator, r=1.0): larger learned bias → resets more often
  Scales 1+ (oscillators, r<1): smaller bias → accumulates without reset

If all scales converge to similar biases, multi-scale differentiation
isn't being learned.

Usage:
    cd ~/projects/continuous-language && source .venv/bin/activate
    PYTHONPATH=/home/meridian/src python gate_sparsity_diagnostic.py \\
        --checkpoint checkpoints/char_lm/best_params.pkl \\
        --n_layers 12 --n_scales 4

    # To see full params tree (useful for verifying key paths):
    PYTHONPATH=/home/meridian/src python gate_sparsity_diagnostic.py \\
        --checkpoint checkpoints/char_lm/best_params.pkl --tree
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import argparse
import os


def load_checkpoint(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def print_tree(d, prefix='', max_depth=4, depth=0):
    """Print the params tree structure for debugging key paths."""
    if depth > max_depth:
        return
    if hasattr(d, 'items'):
        for k, v in sorted(d.items()):
            if hasattr(v, 'items'):
                print(f"{prefix}{k}/")
                print_tree(v, prefix + '  ', max_depth, depth + 1)
            else:
                shape = getattr(v, 'shape', '?')
                print(f"{prefix}{k}: {shape}")


def find_gate_biases(params: dict, n_layers: int, n_scales: int):
    """
    Navigate the params tree to find gate_2_{scale} biases per layer.

    Returns: dict {layer_idx: {scale_idx: (bias_mean, r_approx)}}

    The params tree for GatedWave inside CharLM is typically:
        params['GatedWaveModel_0']['layer_{i}']['gate_2_{s}']['bias']
    but the exact path depends on Flax's auto-naming. Use --tree to verify.
    """
    # Try to find the GatedWaveModel namespace
    p = params
    if 'params' in p:
        p = p['params']

    # Look for GatedWaveModel_0 or similar
    wave_key = None
    for k in p.keys():
        if 'GatedWaveModel' in str(k) or 'gated_wave' in str(k).lower():
            wave_key = k
            break

    if wave_key is None:
        # Maybe CharLM directly contains layer_0 etc. (if core isn't wrapped)
        wave_params = p
    else:
        wave_params = p[wave_key]

    results = {}
    for layer_idx in range(n_layers):
        layer_key = f'layer_{layer_idx}'
        if layer_key not in wave_params:
            continue
        layer_params = wave_params[layer_key]

        scale_data = {}
        # Approximate r values (we don't have r_min/r_max here, just use index)
        for scale_idx in range(n_scales):
            gate_key = f'gate_2_{scale_idx}'
            if gate_key not in layer_params:
                continue
            bias = layer_params[gate_key].get('bias', None)
            if bias is not None:
                mean_bias = float(jnp.mean(jnp.array(bias)))
                expected_activation = float(jax.nn.sigmoid(jnp.array(mean_bias)))
                scale_data[scale_idx] = {
                    'bias': mean_bias,
                    'activation': expected_activation,
                    'is_integrator': scale_idx == 0,
                }
        if scale_data:
            results[layer_idx] = scale_data

    return results


def print_report(results: dict, n_scales: int, r_min: float, r_max: float):
    if not results:
        print("No gate data found. Use --tree to inspect the checkpoint structure.")
        return

    print("\n" + "=" * 70)
    print("Gate Sparsity Report (bias-based prior estimate)")
    print("=" * 70)
    print(f"{'Layer':<7} {'Scale':<8} {'Type':<12} {'Gate bias':<14} {'Prior act.'}")
    print("-" * 70)

    all_integrator = []
    all_oscillator = []

    for layer_idx in sorted(results.keys()):
        for scale_idx in sorted(results[layer_idx].keys()):
            d = results[layer_idx][scale_idx]
            scale_type = "integrator" if d['is_integrator'] else f"oscillator"
            print(f"  {layer_idx:<5} {scale_idx:<8} {scale_type:<12} "
                  f"{d['bias']:+.4f}         {d['activation']:.4f}")
            if d['is_integrator']:
                all_integrator.append(d['activation'])
            else:
                all_oscillator.append(d['activation'])
        print()

    print("=" * 70)

    if all_integrator and all_oscillator:
        avg_int = np.mean(all_integrator)
        avg_osc = np.mean(all_oscillator)
        print(f"\nSummary across all layers:")
        print(f"  Integrator (scale 0): mean prior activation = {avg_int:.4f}")
        print(f"  Oscillators (1+):     mean prior activation = {avg_osc:.4f}")

        if avg_osc > 0:
            ratio = avg_int / avg_osc
            print(f"  Ratio: {ratio:.2f}x")
        else:
            ratio = float('inf')

        print()
        if ratio > 2.0:
            verdict = "STRONG DIFFERENTIATION — integrator resets far more than oscillators"
            detail = "The multi-scale architecture is working as intended."
        elif ratio > 1.3:
            verdict = "MODERATE DIFFERENTIATION — integrator resets more than oscillators"
            detail = "Some multi-scale behavior, but could be stronger."
        elif ratio > 0.8:
            verdict = "WEAK/NO DIFFERENTIATION — similar gate firing across scales"
            detail = ("The oscillatory scales may not be maintaining phase coherence.\n"
                      "  Consider: gate sparsity regularization, or decoupled gate conditioning\n"
                      "  (oscillatory gate conditioned on prev_h frequency band, not raw x)")
        else:
            verdict = "INVERTED — oscillators resetting more than integrator"
            detail = "Unexpected pattern. Check that scale 0 is truly the integrator (r=1.0)."

        print(f"Verdict: {verdict}")
        print(f"  {detail}")

    print()
    print("Note: This is a prior estimate from gate bias values only.")
    print("Actual firing rate on data also depends on input-dependent gate logits.")
    print("A full activation-level analysis requires an instrumented forward pass.")


def main():
    parser = argparse.ArgumentParser(description="GatedWave gate sparsity diagnostic")
    parser.add_argument('--checkpoint', required=True, help='Path to .pkl checkpoint')
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--n_scales', type=int, default=4)
    parser.add_argument('--r_min', type=float, default=0.9)
    parser.add_argument('--r_max', type=float, default=0.999)
    parser.add_argument('--tree', action='store_true',
                        help='Print full params tree and exit (for debugging key paths)')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    print(f"Loading: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint)

    if args.tree:
        print("\nParams tree:")
        print_tree(ckpt)
        return

    results = find_gate_biases(ckpt, args.n_layers, args.n_scales)
    print_report(results, args.n_scales, args.r_min, args.r_max)


if __name__ == "__main__":
    main()
