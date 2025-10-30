# Quick Installation and Usage Guide - Talos Optimization

## ⚡ 5-Minute Quick Start

### Step 1: Install Talos (30 seconds)
```bash
pip install talos
```

### Step 2: Run Optimization (2-4 hours)
```bash
cd refined_code
python train_talos.py
```
When prompted, type `1` and press Enter (for minimal search).

### Step 3: View Results (2 minutes)
Results are saved in: `talos_results/<your_dataset_name>/`
- Open `talos_results.xlsx` to see all experiments
- Open `best_params.json` to see optimal configuration

### Step 4: Use Best Parameters (varies)
```bash
python train_with_best_params.py --epochs 200
```

## 📊 What Happens During Optimization

```
Starting optimization...
├── Testing configuration 1/40: lambda_net + RAdam + lr=0.0001
│   ├── Training 30 epochs...
│   ├── Val MCC: 0.7234 ✓
│   └── Saved results
├── Testing configuration 2/40: lambda_net + Adam + lr=0.0005
│   ├── Training 30 epochs...
│   ├── Val MCC: 0.6892
│   └── Saved results
├── ...
└── Testing configuration 40/40: model + AdamW + lr=0.001
    ├── Training 30 epochs...
    ├── Val MCC: 0.7456 ✓✓ (Best so far!)
    └── Saved results

Optimization complete!
Best configuration saved to: best_params.json
```

## 🎯 Example: Complete Workflow

```bash
# Day 1: Run optimization
cd refined_code
python train_talos.py
# Select: 1 (minimal search)
# Wait: 2-4 hours
# Result: Best parameters found and saved

# Day 2: Analyze and train
python train_with_best_params.py --compare  # Compare baseline vs optimized
python train_with_best_params.py --epochs 200  # Train final model
# Result: Optimized model trained
```

## 📁 Files You'll Get

After running Talos, you'll have:

```
talos_results/
└── <your_dataset_name>/
    ├── talos_results.xlsx          # All 40+ experiments with metrics
    │                                 # Columns: learning_rate, batch_size,
    │                                 #          val_mcc, val_auc, etc.
    │
    └── best_params.json             # Best configuration found
                                      # Example: {"learning_rate": 0.0001,
                                      #           "batch_size": 16, ...}
```

## 🔍 Understanding Your Results

### Excel File Columns

| Column | Meaning |
|--------|---------|
| `learning_rate` | Tested learning rate |
| `batch_size` | Tested batch size |
| `hidden_size` | LSTM hidden layer size |
| `model_module` | Model type used |
| `val_mcc` | **Most important: validation MCC** |
| `val_auc` | Validation AUC |
| `val_accuracy` | Validation accuracy |

**Pro tip:** Sort by `val_mcc` (descending) to see best configurations first!

### Best Parameters JSON

```json
{
    "learning_rate": 0.0001,      // Use this in train.py
    "batch_size": 16,              // Use this in train.py
    "hidden_size": 64,             // Use this in train.py
    "num_lstm_layers": 7,          // Use this in train.py
    "model_module": "lambda_net",  // Use this in train.py
    "optimizer": "RAdam",          // Use this optimizer
    "val_mcc": 0.8542              // Expected validation MCC
}
```

## 💡 Common Questions

### Q: How long does it take?
**A:** Minimal search: 2-4 hours. Comprehensive search: 24-48 hours.

### Q: Can I stop and resume?
**A:** Results are saved continuously, but you can't resume a stopped search. Just start a new one.

### Q: What if I run out of GPU memory?
**A:** Edit `train_talos.py` and change:
```python
'batch_size': [8],  # Reduce from [16, 32]
'hidden_size': [32, 64],  # Reduce from [64, 128]
```

### Q: How do I know if it's working?
**A:** You'll see progress bars and periodic updates like:
```
>>> Round 5 | 12.5% | model_module: lambda_net | lr: 0.0001
Training: 100%|██████████| 30/30 [05:23<00:00]
Val MCC: 0.7234 | Val AUC: 0.8456
```

### Q: What should I optimize for?
**A:** The script optimizes for **validation MCC** (Matthews Correlation Coefficient), which is ideal for imbalanced medical datasets.

### Q: Can I use my own hyperparameters?
**A:** Yes! Edit the `get_minimal_talos_params()` or `get_talos_params()` function in `train_talos.py`.

## 🚨 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'talos'"
**Solution:**
```bash
pip install talos
```

### Problem: "CUDA out of memory"
**Solution:** Reduce batch size and model size in `train_talos.py`:
```python
'batch_size': [8],
'hidden_size': [32, 64],
```

### Problem: "FileNotFoundError: model_configs.json"
**Solution:** Make sure you're running from the `refined_code` directory:
```bash
cd refined_code
python train_talos.py
```

### Problem: Results look bad (low MCC values)
**Possible causes:**
1. Data quality issues - check preprocessing
2. Not enough epochs - increase `num_epochs`
3. Search space too narrow - expand hyperparameter ranges

## 📚 More Information

- **Quick reference:** Read `TALOS_QUICKSTART.md`
- **Complete guide:** Read `10_TALOS_OPTIMIZATION.md`
- **Feature overview:** Read `TALOS_README.md`
- **All docs:** Check `INDEX.md`

## ✅ Success Checklist

- [ ] Installed Talos (`pip install talos`)
- [ ] Ran optimization (`python train_talos.py`)
- [ ] Got results files (Excel + JSON)
- [ ] Reviewed best parameters
- [ ] Applied to final training
- [ ] Achieved better performance than baseline

## 🎉 You're All Set!

You now have:
1. ✅ Automated hyperparameter optimization
2. ✅ Systematic parameter exploration
3. ✅ Optimal configuration for your dataset
4. ✅ Comprehensive results tracking
5. ✅ Time savings (hours vs. days of manual tuning)

**Next step:** Run `python train_talos.py` and let the optimization begin! 🚀

---

**Need help?** Check the documentation files or review code comments in `train_talos.py`.
