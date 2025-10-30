"""
Hyperparameter Optimization Script using Talos for fNIR Base Model.

This script uses Talos to perform automated hyperparameter search across
multiple model architectures, learning rates, optimizers, and other parameters.
"""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, accuracy_score, matthews_corrcoef
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
import torch.nn as nn
from config import parse_args
from dobi_dataset import CustomDataset
from unit.metrics import compute_sen_spec
import random
import json
import os
import importlib
import talos


# Default random seed
seed = 42

# Random seed control
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Reproducibility priority
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_model_configs(config_path="model_configs.json"):
    """Load JSON configuration file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)


def get_model_params(args, model_class, config_dict):
    """Merge common and extra parameters"""
    config_key = f"{args.model_module}.{model_class}"

    if config_key not in config_dict:
        extra_params = {}
    else:
        extra_params = config_dict[config_key]

    # Common parameters (from args)
    common_params = {
        "input_size": args.input_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_lstm_layers,
        "output_size": args.output_size,
        "use_conv1to3": args.use_conv1to3,
        "use_attention": args.use_attention,
        "cnn_input_channels": args.cnn_input_channels
    }

    # Merge parameters
    model_params = {**common_params, **extra_params}
    return model_params


def create_model(module_name, class_name, args, config_path="model_configs.json"):
    """Create model instance"""
    config_dict = load_model_configs(config_path)
    module = importlib.import_module(f"model.{module_name}")
    model_class = getattr(module, class_name)
    model_params = get_model_params(args, class_name, config_dict)
    return model_class(**model_params)


def talos_model(x_train, y_train, x_val, y_val, params):
    """
    Talos training function.
    
    Args:
        x_train: Training data (DataLoader wrapped in tuple)
        y_train: Training labels (not used, included in DataLoader)
        x_val: Validation data (DataLoader wrapped in tuple)
        y_val: Validation labels (not used, included in DataLoader)
        params: Dictionary of hyperparameters from Talos
    
    Returns:
        history: Training history
        model: Trained model
    """
    # Extract args from params
    args = params['args']
    
    # Update args with Talos parameters
    args.learning_rate = params['learning_rate']
    args.batch_size = params['batch_size']
    args.hidden_size = params['hidden_size']
    args.num_lstm_layers = params['num_lstm_layers']
    args.weight_decay = params['weight_decay']
    args.model_module = params['model_module']
    args.model_class = params['model_class']
    args.use_conv1to3 = params['use_conv1to3']
    args.use_attention = params['use_attention']
    args.cnn_input_channels = params['cnn_input_channels']
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    # Extract DataLoaders from wrapped tuples
    train_loader = x_train[0]
    val_loader = x_val[0]
    
    # Create model
    config_dict = load_model_configs("model_configs.json")
    model = create_model(args.model_module, args.model_class, args).to(device)
    
    # Optimizer selection
    if params['optimizer'] == 'RAdam':
        optimizer = optim.RAdam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif params['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif params['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
    
    # Scheduler selection
    if params['scheduler'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=params['scheduler_T_max'], eta_min=params['scheduler_eta_min'])
    elif params['scheduler'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    elif params['scheduler'] == 'StepLR':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    else:  # None
        scheduler = None
    
    # Loss function with configurable pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(params['pos_weight']).to(device))
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_auc': [],
        'val_mcc': [],
        'val_sensitivity': [],
        'val_specificity': []
    }
    
    # Training loop (reduced epochs for faster search)
    num_epochs = params['num_epochs']
    best_val_mcc = -1
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        all_train_predicted = np.array([])
        all_train_predicted_score = np.array([])
        all_train_labels = np.array([])
        epoch_train_loss = 0.0
        
        for idx, (img, label, name) in enumerate(train_loader):
            train_img = img.to(device)
            targets = label.unsqueeze(1).float().to(device)
            optimizer.zero_grad()
            outputs = model(train_img)
            train_loss = criterion(outputs, targets)
            epoch_train_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()
            
            all_train_predicted_score = np.append(all_train_predicted_score,
                                                  torch.sigmoid(outputs).squeeze().cpu().detach().numpy())
            all_train_predicted = np.append(all_train_predicted,
                                            (torch.sigmoid(outputs) >= 0.5).squeeze().cpu().numpy().astype(int))
            all_train_labels = np.append(all_train_labels, targets.cpu().numpy().squeeze())
            
            del img, targets, outputs
            torch.cuda.empty_cache()
        
        # Step scheduler (if not ReduceLROnPlateau)
        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        all_val_predicted = np.array([])
        all_val_predicted_score = np.array([])
        all_val_labels = np.array([])
        
        with torch.no_grad():
            for idx, (img, label, name) in enumerate(val_loader):
                val_img = img.to(device)
                val_targets = label.unsqueeze(1).float().to(device)
                val_outputs = model(val_img)
                val_loss = criterion(val_outputs, val_targets)
                epoch_val_loss += val_loss.item()
                
                all_val_predicted = np.append(all_val_predicted,
                                              (torch.sigmoid(val_outputs) >= 0.5).squeeze().cpu().numpy().astype(int))
                all_val_predicted_score = np.append(all_val_predicted_score,
                                                    torch.sigmoid(val_outputs).squeeze().cpu().numpy())
                all_val_labels = np.append(all_val_labels, val_targets.cpu().numpy().squeeze())
                
                del val_img, val_targets, val_outputs
                torch.cuda.empty_cache()
        
        # Step ReduceLROnPlateau scheduler
        if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(epoch_val_loss / len(val_loader))
        
        # Calculate metrics
        train_loss_avg = epoch_train_loss / len(train_loader)
        val_loss_avg = epoch_val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_val_labels, all_val_predicted)
        val_sen, val_spe = compute_sen_spec(all_val_labels, all_val_predicted)
        
        # Calculate AUC (handle edge cases)
        try:
            fpr, tpr, _ = roc_curve(all_val_labels, all_val_predicted_score)
            val_auc = auc(fpr, tpr)
        except:
            val_auc = 0.5
        
        val_mcc = matthews_corrcoef(all_val_labels, all_val_predicted)
        
        # Store history
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss_avg)
        history['val_accuracy'].append(val_accuracy)
        history['val_auc'].append(val_auc)
        history['val_mcc'].append(val_mcc)
        history['val_sensitivity'].append(val_sen)
        history['val_specificity'].append(val_spe)
        
        # Track best MCC
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
        
        # Early stopping based on train MCC
        train_accuracy = accuracy_score(all_train_labels, all_train_predicted)
        train_sen, train_spe = compute_sen_spec(all_train_labels, all_train_predicted)
        try:
            fpr, tpr, _ = roc_curve(all_train_labels, all_train_predicted_score)
            train_auc = auc(fpr, tpr)
        except:
            train_auc = 0.5
        train_mcc = matthews_corrcoef(all_train_labels, all_train_predicted)
        
        if train_mcc > 0.9:
            print(f"Early stopping at epoch {epoch + 1} (Train MCC: {train_mcc:.4f})")
            break
    
    # Return final validation MCC as the metric to optimize
    return history, model


def run_talos_optimization(args, params):
    """
    Run Talos hyperparameter optimization.
    
    Args:
        args: Parsed arguments
        params: Dictionary of hyperparameter search space
    """
    print("=======================【Talos Hyperparameter Optimization】===============================")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    # Prepare datasets (using fixed batch_size for DataLoaders, will be recreated in talos_model)
    train_dataset = CustomDataset(
        data_path=os.path.join(args.data_path, args.train_data),
        labels_path=os.path.join(args.data_path, args.train_labels),
        names_path=os.path.join(args.data_path, args.train_names),
        channels=args.cnn_input_channels
    )
    
    val_dataset = CustomDataset(
        data_path=os.path.join(args.data_path, args.val_data),
        labels_path=os.path.join(args.data_path, args.val_labels),
        names_path=os.path.join(args.data_path, args.val_names),
        channels=args.cnn_input_channels
    )
    
    # Create DataLoaders (batch_size will be overridden by Talos params)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Wrap DataLoaders in tuples for Talos compatibility
    x_train = (train_loader,)
    y_train = None  # Labels are in the DataLoader
    x_val = (val_loader,)
    y_val = None
    
    # Add args to params for access in talos_model
    params['args'] = [args]
    
    # Create output directory for Talos results
    dataset_name = os.path.basename(args.data_path)
    talos_dir = os.path.join('talos_results', dataset_name)
    if not os.path.exists(talos_dir):
        os.makedirs(talos_dir)
    
    # Run Talos scan
    print(f"Starting Talos scan with {len(params['learning_rate'])} learning rates, "
          f"{len(params['batch_size'])} batch sizes, "
          f"{len(params['model_module'])} model modules...")
    
    scan_object = talos.Scan(
        x=x_train,
        y=y_train,
        params=params,
        model=talos_model,
        experiment_name=f'talos_scan_{dataset_name}',
        x_val=x_val,
        y_val=y_val,
        fraction_limit=params.get('fraction_limit', 1.0),  # Fraction of combinations to test
        reduction_method=params.get('reduction_method', None),  # e.g., 'correlation' for intelligent search
        reduction_interval=params.get('reduction_interval', 50),
        reduction_window=params.get('reduction_window', 20),
        minimize_loss=False,  # We want to maximize MCC
        disable_progress_bar=False,
        print_params=True,
        clear_session=True,
        save_weights=False
    )
    
    # Analyze results
    print("\n=======================【Talos Results】===============================")
    
    # Get best parameters
    best_params = scan_object.best_params(metric='val_mcc', exclude=[], n_models=1)
    print("\nBest Parameters:")
    print(best_params)
    
    # Save results to Excel
    results_df = scan_object.data
    results_file = os.path.join(talos_dir, 'talos_results.xlsx')
    results_df.to_excel(results_file, index=False)
    print(f"\nFull results saved to: {results_file}")
    
    # Save best parameters to JSON
    best_params_file = os.path.join(talos_dir, 'best_params.json')
    with open(best_params_file, 'w') as f:
        json.dump(best_params.to_dict('records')[0], f, indent=4)
    print(f"Best parameters saved to: {best_params_file}")
    
    # Deploy best model (optional)
    deploy_model = scan_object.best_model(metric='val_mcc', asc=False)
    
    return scan_object, best_params


def get_talos_params():
    """
    Define the hyperparameter search space for Talos.
    
    Returns:
        Dictionary with lists of hyperparameters to search
    """
    params = {
        # Model architecture
        'model_module': ['lambda_net', 'model', 'lstm_attention', 'LNN'],
        'model_class': ['CNNLSTMNet'],
        
        # CNN input configuration
        'cnn_input_channels': [1, 3],
        'use_conv1to3': [True, False],
        'use_attention': [True, False],
        
        # Training hyperparameters
        'learning_rate': [0.0001, 0.0005, 0.001, 0.00005],
        'batch_size': [8, 16, 32],
        'weight_decay': [1e-6, 1e-5, 1e-4, 0],
        
        # Model hyperparameters
        'hidden_size': [32, 64, 128],
        'num_lstm_layers': [3, 5, 7, 9],
        
        # Optimizer
        'optimizer': ['RAdam', 'Adam', 'AdamW'],
        
        # Scheduler
        'scheduler': ['CosineAnnealingLR', 'ReduceLROnPlateau', 'StepLR', None],
        'scheduler_T_max': [100, 150, 200],
        'scheduler_eta_min': [1e-5, 1e-6, 1e-7],
        
        # Loss function
        'pos_weight': [1.9167, 2.2167, 2.5, 3.0],
        
        # Training duration
        'num_epochs': [50],  # Reduced for faster search
        
        # Talos-specific parameters
        'fraction_limit': [0.1],  # Test 10% of all combinations
        'reduction_method': ['correlation'],  # Intelligent search
        'reduction_interval': [50],
        'reduction_window': [20],
    }
    
    return params


def get_minimal_talos_params():
    """
    Define a minimal hyperparameter search space for quick testing.
    
    Returns:
        Dictionary with reduced lists of hyperparameters
    """
    params = {
        # Model architecture
        'model_module': ['lambda_net', 'model'],
        'model_class': ['CNNLSTMNet'],
        
        # CNN input configuration
        'cnn_input_channels': [3],
        'use_conv1to3': [False],
        'use_attention': [True],
        
        # Training hyperparameters
        'learning_rate': [0.0001, 0.0005],
        'batch_size': [16],
        'weight_decay': [1e-6, 1e-5],
        
        # Model hyperparameters
        'hidden_size': [64, 128],
        'num_lstm_layers': [5, 7],
        
        # Optimizer
        'optimizer': ['RAdam', 'Adam'],
        
        # Scheduler
        'scheduler': ['CosineAnnealingLR'],
        'scheduler_T_max': [100],
        'scheduler_eta_min': [1e-5],
        
        # Loss function
        'pos_weight': [2.2167],
        
        # Training duration
        'num_epochs': [30],  # Even shorter for testing
        
        # Talos-specific parameters
        'fraction_limit': [0.2],  # Test 20% of combinations
        'reduction_method': [None],
        'reduction_interval': [50],
        'reduction_window': [20],
    }
    
    return params


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    
    # Choose search space
    # Use get_minimal_talos_params() for quick testing
    # Use get_talos_params() for comprehensive search
    print("Select hyperparameter search mode:")
    print("1. Minimal search (faster, for testing)")
    print("2. Comprehensive search (slower, thorough)")
    
    choice = input("Enter choice (1 or 2, default=1): ").strip()
    
    if choice == '2':
        params = get_talos_params()
        print("\nUsing comprehensive search space")
    else:
        params = get_minimal_talos_params()
        print("\nUsing minimal search space")
    
    # Run Talos optimization
    scan_object, best_params = run_talos_optimization(args, params)
    
    print("\n=======================【Optimization Complete】===============================")
    print("Use the best parameters from talos_results/best_params.json to train your final model.")
