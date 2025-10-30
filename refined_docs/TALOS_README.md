# Talos Hyperparameter Optimization

## Overview

This directory contains an advanced hyperparameter optimization script using Talos for automated model tuning.

## 📋 What's New

- **train_talos.py**: Main script for Talos optimization
- **10_TALOS_OPTIMIZATION.md**: Comprehensive documentation
- **TALOS_QUICKSTART.md**: Quick start guide
- **requirements_talos.txt**: Updated requirements with Talos

## 🚀 Quick Start

### 1. Install Talos

```bash
pip install talos
```

### 2. Run Optimization

```bash
python train_talos.py
```

Choose option 1 for minimal search (recommended for first run).

### 3. View Results

Results are saved in `talos_results/<dataset_name>/`:
- `talos_results.xlsx` - All tested configurations
- `best_params.json` - Best hyperparameters found

## 🎯 What Gets Optimized

### Model Architecture
- Model type (lambda_net, LSTM, Transformer, LNN)
- Hidden layer sizes
- Number of LSTM layers
- Attention mechanisms
- Input channel configurations

### Training Parameters
- Learning rate
- Batch size
- Weight decay
- Optimizer (RAdam, Adam, AdamW, SGD)
- Learning rate scheduler
- Loss function weights

## 📊 Hyperparameters Searched

| Parameter | Minimal Search | Comprehensive Search |
|-----------|---------------|---------------------|
| Learning Rate | 2 values | 4 values |
| Batch Size | 1 value | 3 values |
| Hidden Size | 2 values | 3 values |
| LSTM Layers | 2 values | 4 values |
| Optimizers | 2 types | 3 types |
| Model Types | 2 models | 4 models |
| **Combinations** | ~40 | ~500+ |
| **Time** | 2-4 hours | 24-48 hours |

## 📁 Output Structure

```
talos_results/
└── <dataset_name>/
    ├── talos_results.xlsx      # All experiments
    ├── best_params.json         # Best configuration
    └── talos_scan_<dataset>_<timestamp>/
        └── (Talos internal files)
```

## 🔍 Understanding Results

### Excel File Columns

- **Hyperparameters**: All tested parameter combinations
- **val_mcc**: Validation Matthews Correlation Coefficient (primary metric)
- **val_auc**: Validation Area Under ROC Curve
- **val_accuracy**: Validation accuracy
- **val_sensitivity**: True positive rate
- **val_specificity**: True negative rate
- **train_loss**: Training loss
- **val_loss**: Validation loss

### Best Parameters JSON

```json
{
    "learning_rate": 0.0001,
    "batch_size": 16,
    "hidden_size": 64,
    "num_lstm_layers": 7,
    "optimizer": "RAdam",
    "scheduler": "CosineAnnealingLR",
    "model_module": "lambda_net",
    "model_class": "CNNLSTMNet",
    "use_attention": true,
    "cnn_input_channels": 3,
    "val_mcc": 0.8542
}
```

## 🎓 Usage Examples

### Example 1: Quick Test

```bash
# Minimal search on default dataset
python train_talos.py
# Select: 1 (minimal search)
```

### Example 2: Custom Dataset

```bash
# With custom data path
python train_talos.py --data_path data/npy/custom_dataset
```

### Example 3: Using Best Parameters

After optimization, apply best parameters:

```bash
# Extract from best_params.json and use
python train.py \
    --learning_rate 0.0001 \
    --batch_size 16 \
    --hidden_size 64 \
    --num_lstm_layers 7 \
    --model_module lambda_net \
    --num_epochs 200
```

## ⚙️ Configuration

### Modify Search Space

Edit `get_minimal_talos_params()` or `get_talos_params()` in `train_talos.py`:

```python
def get_minimal_talos_params():
    params = {
        'learning_rate': [0.0001, 0.0005, 0.001],  # Add more values
        'batch_size': [8, 16, 32],  # Test different sizes
        'model_module': ['lambda_net'],  # Focus on one model
        # ... other parameters
    }
    return params
```

### Search Strategies

```python
# Test all combinations (expensive)
'fraction_limit': [1.0],

# Test 10% randomly
'fraction_limit': [0.1],

# Intelligent reduction
'reduction_method': ['correlation'],
'reduction_interval': [25],
```

## 💡 Tips for Success

### 1. Start Small
- Use minimal search first
- Test on one dataset
- Verify GPU availability

### 2. Monitor Resources
```bash
# Check GPU usage
nvidia-smi

# Or continuously
watch -n 1 nvidia-smi
```

### 3. Incremental Optimization
1. Run minimal search (2-4 hours)
2. Analyze results
3. Refine search space based on findings
4. Run comprehensive search on refined space

### 4. Memory Management

If you encounter OOM errors:
```python
# Reduce batch size
'batch_size': [8],  # Instead of [16, 32]

# Reduce model size
'hidden_size': [32, 64],  # Instead of [64, 128, 256]
```

## 🐛 Troubleshooting

### Problem: Out of Memory

**Solution:**
```python
# Edit train_talos.py
'batch_size': [8],
'hidden_size': [32, 64],
'num_lstm_layers': [3, 5],
```

### Problem: Too Slow

**Solution:**
```python
# Reduce search space
'num_epochs': [20],
'fraction_limit': [0.05],  # 5% of combinations
```

### Problem: Poor Results

**Possible causes:**
1. Data quality issues
2. Insufficient epochs
3. Search space too narrow

**Solution:**
- Verify data preprocessing
- Increase `num_epochs`
- Expand parameter ranges

### Problem: Script Crashes

**Debug steps:**
1. Check CUDA availability: `torch.cuda.is_available()`
2. Verify data paths exist
3. Check disk space for results
4. Review error messages in console

## 📈 Performance Expectations

### Minimal Search (Default)
- **Configurations**: ~40
- **Time**: 2-4 hours
- **GPU Memory**: 6-8 GB
- **Improvement**: 5-15% over baseline

### Comprehensive Search
- **Configurations**: 500+
- **Time**: 24-48 hours
- **GPU Memory**: 6-8 GB
- **Improvement**: 10-25% over baseline

## 🔬 Advanced Features

### 1. Multi-Dataset Optimization

```python
datasets = ['dataset1', 'dataset2', 'dataset3']
for dataset in datasets:
    args.data_path = f'data/npy/{dataset}'
    run_talos_optimization(args, params)
```

### 2. Custom Metrics

Modify `talos_model()` to optimize different metrics:

```python
# Optimize for AUC instead of MCC
return history, model  # Talos will use val_auc if specified
```

### 3. Conditional Parameters

```python
# Only use attention with lambda_net
if params['model_module'] == 'lambda_net':
    args.use_attention = True
```

## 📚 Documentation

- **Full Documentation**: See `10_TALOS_OPTIMIZATION.md`
- **Quick Reference**: See `TALOS_QUICKSTART.md`
- **Training Pipeline**: See `06_TRAINING_PIPELINE.md`
- **Model Architecture**: See `05_MODEL_ARCHITECTURE.md`

## 🔗 Related Files

- `train.py` - Main training script
- `config.py` - Configuration management
- `model_configs.json` - Model-specific parameters
- `dobi_dataset.py` - Dataset loader

## 🎯 Expected Outcomes

After optimization, you should:
1. ✅ Identify best model architecture
2. ✅ Find optimal learning rate
3. ✅ Determine best batch size
4. ✅ Select appropriate optimizer
5. ✅ Achieve 5-25% performance improvement

## 📊 Example Results

Typical results from minimal search:

| Configuration | Val MCC | Val AUC | Val Acc |
|--------------|---------|---------|---------|
| lambda_net + RAdam | 0.8542 | 0.9234 | 0.8923 |
| model + Adam | 0.8234 | 0.9012 | 0.8756 |
| lstm_attention + RAdam | 0.8456 | 0.9156 | 0.8845 |

## 🚀 Next Steps

1. **Run Optimization**: `python train_talos.py`
2. **Analyze Results**: Open `talos_results.xlsx`
3. **Apply Best Params**: Use in `train.py`
4. **Train Final Model**: Run full training with best parameters
5. **Evaluate**: Test on held-out test set
6. **Deploy**: Use best model for production

## 📝 Notes

- Talos automatically handles parameter combinations
- Results are saved continuously (safe to interrupt)
- GPU is highly recommended
- Each configuration trains independently
- No manual tuning required

## 🤝 Support

For issues or questions:
1. Check `10_TALOS_OPTIMIZATION.md` for detailed help
2. Review `TALOS_QUICKSTART.md` for common issues
3. Inspect `talos_results.xlsx` for debugging
4. Check Talos documentation: https://github.com/autonomio/talos

---

**Ready to optimize?** Run `python train_talos.py` and let Talos find your best hyperparameters!
