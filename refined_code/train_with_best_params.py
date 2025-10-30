"""
Example script to load and use best parameters from Talos optimization.

This script demonstrates how to:
1. Load best parameters from Talos results
2. Apply them to training configuration
3. Run training with optimized parameters
"""

import json
import os
import sys
from config import parse_args
from train import train


def load_best_params(talos_results_dir, dataset_name):
    """
    Load best parameters from Talos optimization results.
    
    Args:
        talos_results_dir: Directory containing Talos results
        dataset_name: Name of the dataset
    
    Returns:
        Dictionary of best parameters
    """
    best_params_file = os.path.join(talos_results_dir, dataset_name, 'best_params.json')
    
    if not os.path.exists(best_params_file):
        raise FileNotFoundError(
            f"Best parameters file not found: {best_params_file}\n"
            f"Please run train_talos.py first to generate optimization results."
        )
    
    with open(best_params_file, 'r') as f:
        best_params = json.load(f)
    
    print("=" * 70)
    print("Loaded Best Parameters from Talos Optimization")
    print("=" * 70)
    for key, value in best_params.items():
        if key != 'args':  # Skip internal args object
            print(f"{key:25s}: {value}")
    print("=" * 70)
    
    return best_params


def apply_best_params_to_args(args, best_params):
    """
    Apply best parameters from Talos to args object.
    
    Args:
        args: Argument object from parse_args()
        best_params: Dictionary of best parameters from Talos
    
    Returns:
        Updated args object
    """
    # Mapping of Talos parameter names to args attributes
    param_mapping = {
        'learning_rate': 'learning_rate',
        'batch_size': 'batch_size',
        'weight_decay': 'weight_decay',
        'hidden_size': 'hidden_size',
        'num_lstm_layers': 'num_lstm_layers',
        'model_module': 'model_module',
        'model_class': 'model_class',
        'use_conv1to3': 'use_conv1to3',
        'use_attention': 'use_attention',
        'cnn_input_channels': 'cnn_input_channels',
    }
    
    # Apply each parameter
    for talos_param, args_attr in param_mapping.items():
        if talos_param in best_params:
            setattr(args, args_attr, best_params[talos_param])
            print(f"Applied: {args_attr} = {best_params[talos_param]}")
    
    return args


def train_with_best_params(talos_results_dir='talos_results', 
                           dataset_name=None,
                           num_epochs=200):
    """
    Train model using best parameters from Talos optimization.
    
    Args:
        talos_results_dir: Directory containing Talos results
        dataset_name: Name of the dataset (if None, will auto-detect)
        num_epochs: Number of epochs for final training
    """
    # Parse base arguments
    args = parse_args()
    
    # Auto-detect dataset name if not provided
    if dataset_name is None:
        dataset_name = os.path.basename(args.data_path)
        print(f"Auto-detected dataset: {dataset_name}")
    
    # Load best parameters
    try:
        best_params = load_best_params(talos_results_dir, dataset_name)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run the Talos optimization first:")
        print("  python train_talos.py")
        sys.exit(1)
    
    # Apply best parameters
    print("\nApplying best parameters to training configuration...")
    args = apply_best_params_to_args(args, best_params)
    
    # Override epochs for final training
    args.num_epochs = num_epochs
    print(f"\nTraining for {num_epochs} epochs with optimized parameters...")
    
    # Run training
    train(args)
    
    print("\n" + "=" * 70)
    print("Training Complete with Optimized Parameters!")
    print("=" * 70)
    print(f"Validation MCC from Talos: {best_params.get('val_mcc', 'N/A')}")
    print("\nCheck the weights/ directory for model checkpoints.")
    print("Check the Excel file for detailed metrics.")


def compare_with_baseline(talos_results_dir='talos_results', dataset_name=None):
    """
    Compare Talos-optimized parameters with baseline.
    
    Args:
        talos_results_dir: Directory containing Talos results
        dataset_name: Name of the dataset
    """
    # Parse baseline arguments
    baseline_args = parse_args()
    
    # Auto-detect dataset name
    if dataset_name is None:
        dataset_name = os.path.basename(baseline_args.data_path)
    
    # Load best parameters
    try:
        best_params = load_best_params(talos_results_dir, dataset_name)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return
    
    # Compare parameters
    print("\n" + "=" * 70)
    print("Parameter Comparison: Baseline vs. Optimized")
    print("=" * 70)
    print(f"{'Parameter':<25s} {'Baseline':<20s} {'Optimized':<20s} {'Change':<15s}")
    print("-" * 70)
    
    comparisons = {
        'learning_rate': baseline_args.learning_rate,
        'batch_size': baseline_args.batch_size,
        'hidden_size': baseline_args.hidden_size,
        'num_lstm_layers': baseline_args.num_lstm_layers,
        'weight_decay': baseline_args.weight_decay,
        'model_module': baseline_args.model_module,
        'use_attention': baseline_args.use_attention,
    }
    
    for param, baseline_val in comparisons.items():
        if param in best_params:
            optimized_val = best_params[param]
            
            # Calculate change
            if isinstance(baseline_val, (int, float)) and isinstance(optimized_val, (int, float)):
                if baseline_val != 0:
                    change_pct = ((optimized_val - baseline_val) / baseline_val) * 100
                    change = f"{change_pct:+.1f}%"
                else:
                    change = "N/A"
            else:
                change = "Changed" if baseline_val != optimized_val else "Same"
            
            print(f"{param:<25s} {str(baseline_val):<20s} {str(optimized_val):<20s} {change:<15s}")
    
    print("-" * 70)
    print(f"{'Validation MCC':<25s} {'N/A':<20s} {best_params.get('val_mcc', 'N/A'):<20s}")
    print("=" * 70)


def show_usage():
    """Display usage instructions."""
    print("""
Usage Examples:
    
1. Train with best parameters from Talos:
    python train_with_best_params.py
    
2. Train with custom number of epochs:
    python train_with_best_params.py --epochs 300
    
3. Compare baseline with optimized parameters:
    python train_with_best_params.py --compare
    
4. Specify custom dataset:
    python train_with_best_params.py --dataset my_dataset_name

Requirements:
    - Must run train_talos.py first to generate best_params.json
    - Results should be in talos_results/<dataset_name>/best_params.json
    """)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train with Talos-optimized parameters')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs for final training (default: 200)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (default: auto-detect from data_path)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare baseline with optimized parameters')
    parser.add_argument('--help-usage', action='store_true',
                       help='Show detailed usage instructions')
    
    cmd_args = parser.parse_args()
    
    if cmd_args.help_usage:
        show_usage()
    elif cmd_args.compare:
        compare_with_baseline(dataset_name=cmd_args.dataset)
    else:
        train_with_best_params(
            dataset_name=cmd_args.dataset,
            num_epochs=cmd_args.epochs
        )
