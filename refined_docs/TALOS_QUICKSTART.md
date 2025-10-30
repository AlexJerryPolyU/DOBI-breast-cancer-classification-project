# Quick Start Guide: Talos Hyperparameter Optimization

## 1. Installation

```bash
# Install Talos
pip install talos

# Or update requirements.txt and install all
pip install -r requirements.txt
```

## 2. Run Optimization (Minimal Mode - Recommended First)

```bash
cd refined_code
python train_talos.py
```

When prompted, choose **1** for minimal search.

## 3. Monitor Progress

The script will display:
- Current hyperparameter combination being tested
- Training progress for each configuration
- Validation metrics after each test

Example output:
```
=======================【Talos Hyperparameter Optimization】===============================
Starting Talos scan with 2 learning rates, 1 batch sizes, 2 model modules...

>>> Round 1 | 2.5% | model_module: lambda_net | learning_rate: 0.0001 | batch_size: 16
Training: 100%|████████████| 30/30 [05:23<00:00]
Val MCC: 0.7234 | Val AUC: 0.8456 | Val Acc: 0.8123

>>> Round 2 | 5.0% | model_module: lambda_net | learning_rate: 0.0005 | batch_size: 16
...
```

## 4. Check Results

### Results Excel File
Location: `talos_results/<your_dataset_name>/talos_results.xlsx`

Open this file to see:
- All tested combinations
- Validation metrics (MCC, AUC, accuracy, sensitivity, specificity)
- Sorted by performance

### Best Parameters JSON
Location: `talos_results/<your_dataset_name>/best_params.json`

```json
{
    "learning_rate": 0.0001,
    "batch_size": 16,
    "hidden_size": 64,
    "num_lstm_layers": 7,
    "optimizer": "RAdam",
    "model_module": "lambda_net",
    "use_attention": true,
    "val_mcc": 0.8542
}
```

## 5. Use Best Parameters

### Option A: Command Line

```bash
python train.py \
    --learning_rate 0.0001 \
    --batch_size 16 \
    --hidden_size 64 \
    --num_lstm_layers 7 \
    --model_module lambda_net \
    --use_attention True \
    --num_epochs 200
```

### Option B: Edit config.py

Update default values in `config.py` with your best parameters:

```python
parser.add_argument('--learning_rate', type=float, default=0.0001)  # From Talos
parser.add_argument('--batch_size', type=int, default=16)  # From Talos
parser.add_argument('--hidden_size', type=int, default=64)  # From Talos
# ... etc
```

Then simply run:
```bash
python train.py --num_epochs 200
```

## 6. Full Training

After finding best parameters, train a complete model:

```bash
# Train for more epochs with best parameters
python train.py --num_epochs 200

# Or if you updated config.py
python train.py
```

## Estimated Time

| Mode | Search Space | Estimated Time | GPU Memory |
|------|--------------|----------------|------------|
| Minimal | ~40 combinations | 2-4 hours | 6-8 GB |
| Comprehensive | ~500+ combinations | 24-48 hours | 6-8 GB |

## Tips for Success

### 1. Start Small
✅ Use minimal search mode first
✅ Test on one dataset
✅ Verify everything works

### 2. Monitor GPU
```bash
# In another terminal
watch -n 1 nvidia-smi
```

### 3. Adjust Based on Results
If minimal search shows:
- High learning rates work better → Add more high LR values
- Larger models perform better → Expand hidden_size range
- Specific optimizer excels → Focus on that optimizer

### 4. Be Patient
- Each configuration trains for multiple epochs
- Progress bars show estimated time
- Results are saved continuously (safe to interrupt)

## Common Issues

### Issue: Out of Memory
**Solution:**
```python
# In train_talos.py, modify get_minimal_talos_params():
'batch_size': [8],  # Reduce from 16
'hidden_size': [32, 64],  # Reduce from [64, 128]
```

### Issue: Taking Too Long
**Solution:**
```python
# In train_talos.py, modify get_minimal_talos_params():
'num_epochs': [20],  # Reduce from 30
'fraction_limit': [0.1],  # Test only 10% of combinations
```

### Issue: Results Not Improving
**Solution:**
1. Check if data is loaded correctly
2. Verify preprocessing steps
3. Try expanding search space
4. Consider data quality issues

## What Talos Does

```
For each hyperparameter combination:
    ├── Create model with those parameters
    ├── Train for specified epochs
    ├── Evaluate on validation set
    ├── Record metrics (MCC, AUC, accuracy, etc.)
    └── Save results

After all combinations:
    ├── Rank by validation MCC
    ├── Identify best configuration
    └── Save results to Excel and JSON
```

## Next Steps After Optimization

1. ✅ Review `talos_results.xlsx` - understand what works
2. ✅ Check `best_params.json` - get optimal configuration
3. ✅ Train final model with best parameters
4. ✅ Evaluate on test set
5. ✅ (Optional) Run comprehensive search for fine-tuning
6. ✅ Deploy model or iterate further

## Example: Complete Workflow

```bash
# Step 1: Run minimal optimization
python train_talos.py
# Choose: 1 (minimal search)
# Wait: 2-4 hours

# Step 2: Check results
cd talos_results/<your_dataset>
# Open talos_results.xlsx

# Step 3: Apply best parameters
cd ../..
python train.py \
    --learning_rate <best_lr> \
    --batch_size <best_batch> \
    --hidden_size <best_hidden> \
    --num_epochs 200

# Step 4: Evaluate
# Model checkpoints saved in weights/
# Metrics saved in Excel file
```

## Need Help?

1. **Check documentation:** `10_TALOS_OPTIMIZATION.md`
2. **Review results:** Excel file shows all details
3. **Adjust parameters:** Edit `get_minimal_talos_params()` in `train_talos.py`
4. **Start fresh:** Delete `talos_results/` directory to restart

## Success Indicators

✅ Script runs without errors
✅ GPU utilization is high (>70%)
✅ Validation MCC improves over baseline
✅ Results file is generated
✅ Best parameters are identified

---

**Ready to optimize!** Run `python train_talos.py` and select option 1 to begin.
