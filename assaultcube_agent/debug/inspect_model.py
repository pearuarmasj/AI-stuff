"""
Neural Network Inspector - See inside your trained model.

Usage:
    python -m assaultcube_agent.debug.inspect_model <path_to_model.zip>
    python -m assaultcube_agent.debug.inspect_model logs/assaultcube_v2/PPO_xxx/final_model.zip

Shows:
    - Network architecture
    - All weights and biases with statistics
    - Layer-by-layer breakdown
    - Total parameter count
"""

import sys
import numpy as np
from pathlib import Path

try:
    from stable_baselines3 import PPO
except ImportError:
    print("ERROR: stable-baselines3 not installed")
    print("Run: pip install stable-baselines3")
    sys.exit(1)


def inspect_model(model_path: str):
    """Inspect a trained model's neural network."""

    print("\n" + "=" * 70)
    print("  NEURAL NETWORK INSPECTOR")
    print("=" * 70)
    print(f"  Model: {model_path}")
    print("=" * 70)

    # Load model
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"\nERROR: Failed to load model: {e}")
        return

    policy = model.policy

    # Architecture overview
    print("\n" + "-" * 70)
    print("  ARCHITECTURE OVERVIEW")
    print("-" * 70)
    print(f"\n{policy}\n")

    # Detailed layer breakdown
    print("-" * 70)
    print("  LAYER-BY-LAYER BREAKDOWN")
    print("-" * 70)

    total_params = 0
    layer_info = []

    for name, param in policy.named_parameters():
        data = param.data.cpu().numpy()
        num_params = param.numel()
        total_params += num_params

        info = {
            'name': name,
            'shape': param.shape,
            'params': num_params,
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'abs_mean': float(np.mean(np.abs(data))),
        }
        layer_info.append(info)

        # Print layer info
        shape_str = "x".join(str(s) for s in param.shape)
        print(f"\n  [{name}]")
        print(f"    Shape:    {shape_str} ({num_params:,} params)")
        print(f"    Mean:     {info['mean']:+.6f}")
        print(f"    Std:      {info['std']:.6f}")
        print(f"    Min:      {info['min']:+.6f}")
        print(f"    Max:      {info['max']:+.6f}")
        print(f"    |Mean|:   {info['abs_mean']:.6f}")

    # Summary
    print("\n" + "-" * 70)
    print("  SUMMARY")
    print("-" * 70)
    print(f"\n  Total layers: {len(layer_info)}")
    print(f"  Total parameters: {total_params:,}")

    # Separate weights and biases
    weight_params = sum(l['params'] for l in layer_info if 'weight' in l['name'])
    bias_params = sum(l['params'] for l in layer_info if 'bias' in l['name'])
    print(f"  Weight parameters: {weight_params:,}")
    print(f"  Bias parameters: {bias_params:,}")

    # Network shape visualization
    print("\n" + "-" * 70)
    print("  NETWORK VISUALIZATION")
    print("-" * 70)

    # Find the main MLP layers
    mlp_weights = [l for l in layer_info if 'mlp_extractor' in l['name'] and 'weight' in l['name']]

    if mlp_weights:
        print("\n  Input → ", end="")
        for i, w in enumerate(mlp_weights):
            in_size = w['shape'][1] if len(w['shape']) > 1 else w['shape'][0]
            out_size = w['shape'][0]
            if i == 0:
                print(f"[{in_size}] → ", end="")
            print(f"[{out_size}] → ", end="")
        print("Output")

    print("\n" + "=" * 70)
    print("  INSPECTION COMPLETE")
    print("=" * 70 + "\n")


def compare_models(model_path1: str, model_path2: str):
    """Compare two models to see weight evolution."""

    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)

    model1 = PPO.load(model_path1)
    model2 = PPO.load(model_path2)

    print(f"\n  Model 1: {model_path1}")
    print(f"  Model 2: {model_path2}")
    print("-" * 70)

    for (name1, param1), (name2, param2) in zip(
        model1.policy.named_parameters(),
        model2.policy.named_parameters()
    ):
        if name1 != name2:
            print(f"  WARNING: Mismatched names {name1} vs {name2}")
            continue

        data1 = param1.data.cpu().numpy()
        data2 = param2.data.cpu().numpy()

        diff = data2 - data1

        print(f"\n  [{name1}]")
        print(f"    Mean change: {np.mean(diff):+.6f}")
        print(f"    Std change:  {np.std(diff):.6f}")
        print(f"    Max change:  {np.max(np.abs(diff)):.6f}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage:")
        print("  python -m assaultcube_agent.debug.inspect_model <model.zip>")
        print("  python -m assaultcube_agent.debug.inspect_model <model1.zip> <model2.zip>  (compare)")
        sys.exit(1)

    if len(sys.argv) == 2:
        inspect_model(sys.argv[1])
    elif len(sys.argv) == 3:
        compare_models(sys.argv[1], sys.argv[2])
    else:
        print("ERROR: Too many arguments")
        sys.exit(1)
