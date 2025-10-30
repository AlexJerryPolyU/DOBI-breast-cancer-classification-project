# Hyperparameter Optimization with Talos

## Overview

The `train_talos.py` script provides automated hyperparameter optimization using the Talos library. It systematically searches through different combinations of hyperparameters to find the best configuration for your fNIR classification model.

## What is Talos?

Talos is an advanced hyperparameter optimization framework for Keras and PyTorch that:
- Automatically tests multiple hyperparameter combinations
- Uses intelligent search strategies to reduce computation time
- Tracks all experiments and their results
- Identifies the best performing configuration

## Installation

Before using the Talos optimization script, you need to install Talos:

```bash
pip install talos
```

Or add it to your `requirements.txt`:
```
talos
```

## Quick Start

### 1. Basic Usage

Run the script with default settings:

```bash
python train_talos.py
```

You'll be prompted to choose between:
1. **Minimal search** (faster, for testing)
2. **Comprehensive search** (slower, thorough)

### 2. With Custom Arguments

```bash
python train_talos.py --data_path data/npy/your_dataset --num_epochs 50
```

## Hyperparameters Optimized

The script optimizes the following hyperparameters:

### Model Architecture
- **model_module**: Different model implementations
  - `lambda_net`: Lambda network architecture
  - `model`: Base CNN-LSTM model
  - `lstm_attention`: LSTM with attention mechanisms
  - `LNN`: Liquid Neural Network
  
- **model_class**: Model class name (e.g., `CNNLSTMNet`)

- **cnn_input_channels**: Input channels (1 or 3)
  - `1`: Single-channel input
  - `3`: Three-channel input (RGB-like)

- **use_conv1to3**: Convert 1 channel to 3 (True/False)

- **use_attention**: Use attention mechanism (True/False)

### Training Hyperparameters

- **learning_rate**: Initial learning rate
  - Range: [0.00005, 0.0001, 0.0005, 0.001]
  
- **batch_size**: Batch size for training
  - Options: [8, 16, 32]
  
- **weight_decay**: L2 regularization strength
  - Range: [0, 1e-6, 1e-5, 1e-4]

### Model Hyperparameters

- **hidden_size**: LSTM hidden layer size
  - Options: [32, 64, 128]
  
- **num_lstm_layers**: Number of LSTM layers
  - Options: [3, 5, 7, 9]

### Optimizer Options

- **optimizer**: Optimization algorithm
  - `RAdam`: Rectified Adam (recommended)
  - `Adam`: Standard Adam optimizer
  - `AdamW`: Adam with weight decay
  - `SGD`: Stochastic Gradient Descent with momentum

### Learning Rate Scheduler

- **scheduler**: LR scheduling strategy
  - `CosineAnnealingLR`: Smooth cosine decay
  - `ReduceLROnPlateau`: Reduce when validation loss plateaus
  - `StepLR`: Step-wise learning rate decay
  - `None`: No scheduling

- **scheduler_T_max**: Period for CosineAnnealingLR [100, 150, 200]
- **scheduler_eta_min**: Minimum LR [1e-7, 1e-6, 1e-5]

### Loss Function

- **pos_weight**: Positive class weight for imbalanced data
  - Options: [1.9167, 2.2167, 2.5, 3.0]

## Search Modes

### Minimal Search (Quick Testing)

Best for initial exploration and testing:

```python
params = {
    'model_module': ['lambda_net', 'model'],
    'learning_rate': [0.0001, 0.0005],
    'batch_size': [16],
    'hidden_size': [64, 128],
    'num_lstm_layers': [5, 7],
    'optimizer': ['RAdam', 'Adam'],
    'num_epochs': [30],
    'fraction_limit': [0.2]  # Test 20% of combinations
}
```

**Estimated time:** 2-4 hours (depends on dataset size)

### Comprehensive Search

Full hyperparameter exploration:

```python
params = {
    'model_module': ['lambda_net', 'model', 'lstm_attention', 'LNN'],
    'learning_rate': [0.0001, 0.0005, 0.001, 0.00005],
    'batch_size': [8, 16, 32],
    'hidden_size': [32, 64, 128],
    'num_lstm_layers': [3, 5, 7, 9],
    'optimizer': ['RAdam', 'Adam', 'AdamW'],
    'scheduler': ['CosineAnnealingLR', 'ReduceLROnPlateau', 'StepLR', None],
    'num_epochs': [50],
    'fraction_limit': [0.1]  # Test 10% of combinations
}
```

**Estimated time:** 24-48 hours (depends on dataset size)

## Output Files

### 1. Results Excel File

Location: `talos_results/<dataset_name>/talos_results.xlsx`

Contains:
- All tested hyperparameter combinations
- Validation metrics (MCC, AUC, accuracy, sensitivity, specificity)
- Training loss and validation loss
- Execution time per configuration

### 2. Best Parameters JSON

Location: `talos_results/<dataset_name>/best_params.json`

Example:
```json
{
    "learning_rate": 0.0001,
    "batch_size": 16,
    "hidden_size": 64,
    "num_lstm_layers": 7,
    "optimizer": "RAdam",
    "scheduler": "CosineAnnealingLR",
    "model_module": "lambda_net",
    "use_attention": true,
    "val_mcc": 0.8542
}
```

## Understanding Results

### Key Metrics

The optimization focuses on **validation MCC** (Matthews Correlation Coefficient) as the primary metric:

- **MCC Range:** -1 to 1
- **Interpretation:**
  - 1.0: Perfect prediction
  - 0.0: Random prediction
  - -1.0: Complete disagreement

### Analyzing Results

1. **Open the Excel file:**
   ```
   talos_results/<dataset_name>/talos_results.xlsx
   ```

2. **Sort by val_mcc (descending)** to see best configurations

3. **Look for patterns:**
   - Which model modules perform best?
   - What learning rates work well?
   - Is attention mechanism helpful?

4. **Check for overfitting:**
   - Compare `train_loss` vs `val_loss`
   - Large gap indicates overfitting

## Using Best Parameters

After Talos finds the best parameters, use them for final training:

### Option 1: Manual Update

Edit `config.py` with the best parameters:

```python
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--hidden_size', type=int, default=64)
# ... etc
```

Then run:
```bash
python train.py
```

### Option 2: Command Line Override

```bash
python train.py \
    --learning_rate 0.0001 \
    --batch_size 16 \
    --hidden_size 64 \
    --num_lstm_layers 7 \
    --model_module lambda_net \
    --optimizer RAdam
```

### Option 3: Programmatic Loading

Create a script to load and apply best parameters:

```python
import json
from config import parse_args

# Load best parameters
with open('talos_results/<dataset>/best_params.json', 'r') as f:
    best_params = json.load(f)

# Update args
args = parse_args()
for key, value in best_params.items():
    if hasattr(args, key):
        setattr(args, key, value)

# Train with best parameters
from train import train
train(args)
```

## Advanced Configuration

### Custom Search Space

Modify `get_talos_params()` in `train_talos.py`:

```python
def get_talos_params():
    params = {
        # Focus on specific learning rates
        'learning_rate': [0.0001, 0.0002, 0.0003],
        
        # Test larger batch sizes
        'batch_size': [32, 64, 128],
        
        # Only test your preferred model
        'model_module': ['lambda_net'],
        
        # ... other parameters
    }
    return params
```

### Reduction Strategies

Talos can intelligently reduce the search space:

```python
params = {
    # ... hyperparameters ...
    
    'fraction_limit': [0.05],  # Test only 5% of combinations
    'reduction_method': ['correlation'],  # Use correlation-based reduction
    'reduction_interval': [25],  # Evaluate every 25 rounds
    'reduction_window': [10],  # Use last 10 rounds for decision
}
```

**Reduction Methods:**
- `'correlation'`: Removes parameters with low correlation to target metric
- `'spear'`: Uses Spearman correlation
- `'kendall'`: Uses Kendall's tau

### Early Stopping

Built-in early stopping when train MCC > 0.9:

```python
if train_mcc > 0.9:
    print(f"Early stopping at epoch {epoch + 1}")
    break
```

Adjust threshold in `talos_model()` function if needed.

## Performance Tips

### 1. Use GPU

Ensure CUDA is available:
```python
torch.cuda.is_available()  # Should return True
```

### 2. Start with Minimal Search

Test your setup first:
```bash
# Use minimal search mode
python train_talos.py
# Choose option 1
```

### 3. Reduce Epochs

For initial exploration:
```python
'num_epochs': [20, 30],  # Instead of [50, 100]
```

### 4. Limit Fraction

Test fewer combinations:
```python
'fraction_limit': [0.05],  # 5% of all combinations
```

### 5. Parallel Processing

Talos doesn't support parallel GPU execution natively, but you can:
- Run multiple Talos experiments on different datasets simultaneously
- Use different GPUs for different experiments

## Troubleshooting

### Out of Memory Error

**Solution 1:** Reduce batch size
```python
'batch_size': [8],  # Instead of [16, 32]
```

**Solution 2:** Reduce model size
```python
'hidden_size': [32, 64],  # Instead of [64, 128, 256]
'num_lstm_layers': [3, 5],  # Instead of [5, 7, 9]
```

**Solution 3:** Clear cache more frequently
Already implemented in the code:
```python
torch.cuda.empty_cache()
```

### Slow Training

**Solution 1:** Reduce search space
- Use fewer hyperparameter values
- Increase `fraction_limit`

**Solution 2:** Reduce epochs
```python
'num_epochs': [20],
```

**Solution 3:** Use smaller dataset
Test on a subset first

### Poor Results

**Solution 1:** Check data quality
- Verify data preprocessing
- Check for data imbalance

**Solution 2:** Expand search space
- Try more learning rates
- Test different optimizers

**Solution 3:** Increase training epochs
```python
'num_epochs': [100, 150],
```

## Example Workflow

### Complete Optimization Pipeline

```bash
# 1. Initial minimal search
python train_talos.py
# Select option 1 (minimal search)

# 2. Review results
# Open: talos_results/<dataset>/talos_results.xlsx
# Identify promising hyperparameter ranges

# 3. Refined search
# Edit get_talos_params() to focus on promising ranges
# Run again with comprehensive search

# 4. Final training
# Use best parameters from best_params.json
python train.py --learning_rate 0.0001 --batch_size 16 --hidden_size 64 ...
```

## Comparison with Manual Tuning

| Aspect | Manual Tuning | Talos Optimization |
|--------|---------------|-------------------|
| **Time Investment** | Days to weeks | Hours to days |
| **Coverage** | Limited combinations | Systematic exploration |
| **Reproducibility** | Difficult | Automatic logging |
| **Expertise Required** | High | Medium |
| **Best for** | Fine-tuning | Initial exploration |

## References

- **Talos Documentation:** https://github.com/autonomio/talos
- **Paper:** Autonomio Talos [Computer software] (2019)
- **PyTorch Documentation:** https://pytorch.org/docs/

## Support

For issues or questions:
1. Check the `talos_results/` directory for logs
2. Review the Excel file for anomalies
3. Verify GPU memory with `nvidia-smi`
4. Check Talos documentation for advanced features

## Next Steps

After optimization:
1. ✅ Identify best hyperparameters
2. ✅ Train final model with best parameters using `train.py`
3. ✅ Evaluate on test set
4. ✅ Analyze model performance
5. ✅ Deploy or iterate further

---

**Note:** Talos optimization is computationally intensive. Plan accordingly and monitor system resources.
