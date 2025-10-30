# Training Pipeline

## Training Script (train.py)

### Training Loop Structure

```python
for epoch in range(num_epochs):
    # 1. Training Phase
    model.train()
    for batch in train_loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    scheduler.step()
    
    # 2. Validation Phase
    model.eval()
    with torch.no_grad():
        val_metrics = evaluate(val_loader)
    
    # 3. Test Phase (every epoch)
    test_metrics = evaluate(test_loader)
    
    # 4. Save Checkpoint
    torch.save(model.state_dict(), f'weights/epoch_{epoch:03d}.pth')
    
    # 5. Update Metrics Excel
    metrics_df = metrics_df.append(epoch_results, ignore_index=True)
    metrics_df.to_excel(metrics_file, index=False)
    
    # 6. Early Stopping Check
    if train_mcc > 0.9:
        print(f"Early stopping at epoch {epoch} (MCC > 0.9)")
        break
```

## Training Components

### Loss Function

```python
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.2167]))
```

**Parameters:**
- `pos_weight=2.2167`: Class imbalance adjustment
- Calculated from dataset ratio (~1.9:1) plus margin (0.3)
- Applies higher weight to positive (malignant) cases

**Why BCEWithLogitsLoss:**
- Combines sigmoid and BCE for numerical stability
- No explicit sigmoid in model output required
- Better gradient flow than separate operations

### Optimizer

```python
optimizer = torch.optim.RAdam(
    model.parameters(),
    lr=0.0001,
    weight_decay=1e-6
)
```

**RAdam (Rectified Adam):**
- Warmup-free adaptive learning rate
- More stable than standard Adam
- Better generalization in early epochs

**Hyperparameters:**
- `lr=0.0001`: Initial learning rate
- `weight_decay=1e-6`: L2 regularization

### Learning Rate Scheduler

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=200,
    eta_min=1e-5
)
```

**Cosine Annealing:**
- Smooth decay from initial LR to minimum
- Periodic restarts possible (not used)
- Helps escape local minima

**Parameters:**
- `T_max=200`: Annealing period (epochs)
- `eta_min=1e-5`: Minimum learning rate

## Evaluation Metrics

### Metrics Tracked Per Epoch

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Loss** | BCEWithLogitsLoss | Training objective |
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Sensitivity** | TP / (TP + FN) | True positive rate (recall) |
| **Specificity** | TN / (TN + FP) | True negative rate |
| **AUC** | Area under ROC curve | Discrimination ability |
| **MCC** | Matthews Correlation | Balanced measure |

### MCC (Matthews Correlation Coefficient)

```python
MCC = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**Interpretation:**
- Range: [-1, 1]
- +1: Perfect prediction
- 0: Random prediction
- -1: Total disagreement

**Why MCC:**
- Balanced for imbalanced datasets
- Single metric capturing all confusion matrix elements
- Used for early stopping criterion

### Overfitting Indicators

```python
train_val_mcc_diff = abs(train_mcc - val_mcc)
train_val_loss_diff = abs(train_loss - val_loss)
```

**Healthy Training Signs:**
- MCC difference < 0.1
- Loss difference decreasing
- Validation metrics improving

## Checkpoint Management

### Directory Structure

```
weights/
└── {model}_{class}_c1to3{bool}_att{bool}/
    └── cH{channels}_{extra_params}/
        └── {dataset_name}/
            └── bs_{bs}_lr_{lr}_epoch_{epochs}_wd_{wd}_hs_{hs}_nl_{nl}/
                ├── epoch_001.pth
                ├── epoch_002.pth
                ├── ...
                └── metrics.xlsx
```

**Example:**
```
weights/
└── model_CNNLSTMNet_c1to3False_attFalse/
    └── cH3_None/
        └── dDOT_9layers_i4_d0_n0_e0/
            └── bs_16_lr_0.0001_epoch_400_wd_1e-06_hs_64_nl_7/
                ├── epoch_001.pth
                ├── ...
                └── metrics.xlsx
```

### Checkpoint Contents

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    # Note: Optimizer and scheduler states not saved
}
```

## Metrics Excel Output

### Sheet Structure

Each row represents one epoch with the following columns:

**Training Set:**
- `Train Loss`
- `Train Accuracy`
- `Train Sensitivity`
- `Train Specificity`
- `Train AUC`
- `Train MCC`

**Validation Set:**
- `Val Loss`
- `Val Accuracy`
- `Val Sensitivity`
- `Val Specificity`
- `Val AUC`
- `Val MCC`

**Test Set:**
- `Test Loss`
- `Test Accuracy`
- `Test Sensitivity`
- `Test Specificity`
- `Test AUC`
- `Test MCC`

**Overfitting Metrics:**
- `Train-Val_MCC`: Absolute difference
- `Train-Val_Loss`: Absolute difference

### Example Output

| Epoch | Train Loss | Train ACC | Train SEN | Train SPE | Train AUC | Train MCC | Val Loss | Val ACC | Val SEN | Val SPE | Val AUC | Val MCC |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|----------|---------|---------|---------|---------|---------|
| 1 | 0.6234 | 0.6500 | 0.7200 | 0.5800 | 0.6950 | 0.3015 | 0.6189 | 0.6600 | 0.7100 | 0.6100 | 0.7050 | 0.3205 |
| 2 | 0.5892 | 0.7100 | 0.7800 | 0.6400 | 0.7550 | 0.4215 | 0.5945 | 0.7000 | 0.7500 | 0.6500 | 0.7450 | 0.4005 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## Early Stopping

### Criterion

```python
if train_mcc > 0.9:
    print(f"Early stopping triggered at epoch {epoch}")
    break
```

**Rationale:**
- MCC > 0.9 indicates excellent performance
- Prevents overfitting to training set
- Based on training MCC (not validation)

**Limitations:**
- Only checks training performance
- May stop before best validation performance
- Fixed threshold (not adaptive)

### Recommended Alternative (Not Implemented)

```python
# Better approach: Use validation MCC with patience
if val_mcc > best_val_mcc:
    best_val_mcc = val_mcc
    patience_counter = 0
    save_best_model()
else:
    patience_counter += 1
    if patience_counter >= patience:
        break
```

## Hyperparameter Search (Auto_train.py)

### Grid Search Configuration

```python
# Wavelet hyperparameters
wavelet_types = ['mexican_hat', 'haar', 'morlet', 'gabor']
kernel_sizes = [1, 2, 3]
num_wavelets = [1, 2, 3]
multi_scale = [1, 2, 3]
use_fourier = [False]
num_recursive_layers = [1, 2]

# Total combinations: 4 × 3 × 3 × 3 × 1 × 2 = 216 runs
```

### Execution Flow

```python
for wavelet_type in wavelet_types:
    for kernel_size in kernel_sizes:
        for num_wavelet in num_wavelets:
            for scale in multi_scale:
                for fourier in use_fourier:
                    for layers in num_recursive_layers:
                        # Update model config
                        config = {
                            'wavelet_type': wavelet_type,
                            'kernel_size': kernel_size,
                            # ... other params
                        }
                        
                        # Train model
                        model = create_model(config)
                        train(model)
                        
                        # Save results
                        save_metrics(config, results)
```

### Output Organization

Each configuration generates:
```
weights/time_mil_CNNLSTMNet_wt_{type}_ks_{size}_nw_{num}_ms_{scale}/
└── cH3_fourier{bool}_rec{layers}/
    └── {dataset_name}/
        └── bs_{bs}_lr_{lr}_epoch_{epochs}_wd_{wd}_hs_{hs}_nl_{nl}/
            └── metrics.xlsx
```

## Training Best Practices

### 1. Data Preparation

**Before Training:**
- Verify preprocessed .npy files exist
- Check data shapes match expected format
- Validate train/val/test split ratios

```bash
# Check data
python -c "import numpy as np; print(np.load('data/npy/train_data.npy').shape)"
```

### 2. Initial Configuration

**Baseline Settings:**
```bash
python train.py \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --num_epochs 100 \
    --Recon_num_layers 9
```

### 3. Monitoring Training

**Watch for:**
- Validation loss plateauing (increase regularization)
- Train-val MCC gap increasing (overfitting)
- Training loss not decreasing (increase LR or check data)

### 4. GPU Memory Management

```python
# Clear cache between runs
torch.cuda.empty_cache()

# Reduce batch size if OOM
--batch_size 8  # or 4

# Use gradient accumulation (not implemented)
```

### 5. Hyperparameter Tuning Order

1. **Learning Rate**: [1e-5, 1e-4, 1e-3]
2. **Batch Size**: [8, 16, 32]
3. **LSTM Layers**: [5, 7, 9]
4. **Hidden Size**: [32, 64, 128]
5. **Dropout**: [0.3, 0.5, 0.7]

## Common Training Issues

### Issue 1: NaN Loss

**Causes:**
- Learning rate too high
- Gradient explosion in LSTM
- Corrupted data

**Solutions:**
```bash
--learning_rate 1e-5  # Reduce LR
# Add gradient clipping (not implemented)
```

### Issue 2: No Learning

**Causes:**
- Learning rate too low
- Dead ReLU neurons
- Frozen backbone weights

**Solutions:**
```bash
--learning_rate 1e-3  # Increase LR
# Check DenseNet pretrained weights loaded
```

### Issue 3: Overfitting

**Symptoms:**
- Train MCC > 0.85, Val MCC < 0.65
- Large train-val loss gap

**Solutions:**
```bash
--weight_decay 1e-5  # Increase regularization
# Add data augmentation (not implemented)
# Reduce model capacity: --num_lstm_layers 5
```

### Issue 4: Underfitting

**Symptoms:**
- Both train and val MCC < 0.6
- Loss plateaus early

**Solutions:**
```bash
--num_lstm_layers 9   # Increase capacity
--hidden_size 128     # Larger LSTM
--num_epochs 500      # Train longer
```

## Training Time Estimates

**Hardware: NVIDIA V100 (16 GB)**

| Configuration | Epochs | Time/Epoch | Total Time |
|--------------|--------|-----------|-----------|
| Base (bs=16, 100 epochs) | 100 | ~2.5 min | ~4.2 hours |
| MIL (bs=16, 100 epochs) | 100 | ~3.0 min | ~5.0 hours |
| Wavelet (bs=16, 100 epochs) | 100 | ~3.5 min | ~5.8 hours |
| Auto_train (216 configs) | 21,600 | ~2.5 min | ~37 days |

**Recommendations:**
- Use Auto_train only for targeted searches
- Run parallel jobs for independent configs
- Monitor first 20 epochs, stop poor performers early

## Design Patterns Used

**Template Method Pattern:**
- Consistent training structure
- Pluggable evaluation and checkpoint saving

**Strategy Pattern:**
- Interchangeable loss functions
- Configurable optimizers and schedulers

**Observer Pattern:**
- Metrics tracking throughout training
- Real-time Excel updates

## Next Steps After Training

1. **Review Metrics Excel**: Identify best epoch
2. **Run Threshold Optimization**: Use `cutoff.py`
3. **Test on Held-Out Data**: Use `pro.py`
4. **Analyze Misclassifications**: Check wrong predictions
5. **Iterate**: Adjust hyperparameters based on results
