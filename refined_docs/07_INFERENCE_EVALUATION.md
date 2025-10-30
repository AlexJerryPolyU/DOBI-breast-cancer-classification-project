# Inference and Evaluation

## Inference Script (pro.py)

### Main Functions

#### 1. test_model() - Full Dataset Evaluation

Evaluates a trained model on train/val/test splits with complete metrics.

**Usage:**
```python
# Configure paths
weights_path = "epoch/epoch_078.pth"
dataset_paths = {
    'train_data': 'data/npy/dataset/train_data.npy',
    'train_labels': 'data/npy/dataset/train_labels.npy',
    'train_names': 'data/npy/dataset/train_names.npy',
    'val_data': 'data/npy/dataset/val_data.npy',
    'val_labels': 'data/npy/dataset/val_labels.npy',
    'val_names': 'data/npy/dataset/val_names.npy',
    'test_data': 'data/npy/dataset/test_data.npy',
    'test_labels': 'data/npy/dataset/test_labels.npy',
    'test_names': 'data/npy/dataset/test_names.npy'
}

# Set model configuration
args.model_module = "model"
args.model_class = "CNNLSTMNet"
args.use_conv1to3 = False
args.use_attention = False

# Run evaluation
python pro.py
```

**Workflow:**
```python
def test_model():
    # 1. Load model
    model = create_model(args)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    # 2. Evaluate on each split
    for split in ['train', 'val', 'test']:
        dataset = CustomDataset(data, labels, names)
        dataloader = DataLoader(dataset, batch_size=16)
        
        predictions = []
        probabilities = []
        true_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).int()
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # 3. Calculate metrics
        metrics = calculate_metrics(predictions, true_labels)
        
        # 4. Save results
        save_results_to_excel(split, metrics, predictions, probabilities)
```

**Output Excel Structure:**

**Sheet 1: Summary Metrics**
| Split | Accuracy | Sensitivity | Specificity | AUC | MCC |
|-------|----------|------------|-------------|-----|-----|
| Train | 0.8850 | 0.9200 | 0.8500 | 0.9350 | 0.7720 |
| Val | 0.8650 | 0.8900 | 0.8400 | 0.9150 | 0.7315 |
| Test | 0.8550 | 0.8800 | 0.8300 | 0.9050 | 0.7125 |

**Sheet 2-4: Detailed Predictions (per split)**
| Patient Name | True Label | Predicted Label | Probability | Correct |
|-------------|-----------|----------------|------------|---------|
| patient001 | 0 | 0 | 0.1234 | ✓ |
| patient002 | 1 | 1 | 0.8765 | ✓ |
| patient003 | 0 | 1 | 0.6543 | ✗ |

**Sheet 5: Misclassifications**
| Patient Name | True Label | Predicted Label | Probability | Split |
|-------------|-----------|----------------|------------|-------|
| patient003 | 0 | 1 | 0.6543 | Train |
| patient087 | 1 | 0 | 0.4321 | Test |

#### 2. preprocess_and_save_npy_fNIR() - On-the-Fly Processing

Preprocesses raw .mat files and runs inference without saving intermediate .npy files.

**Usage:**
```python
# Configure preprocessing
args.excel_path = 'data/excel/phase1exp.xlsx'
args.data_folder = 'data/source data/fdDOT_DynamicFEM_ellips_height'
args.variable_name = 'img4D'
args.interpolation_method = 4
args.denoising_method = 0
args.normalization_method = 0

# Run preprocessing + inference
python pro.py
```

**Workflow:**
```python
def preprocess_and_save_npy_fNIR():
    # 1. Load Excel splits
    train_df = pd.read_excel(excel_path, sheet_name='train')
    val_df = pd.read_excel(excel_path, sheet_name='val')
    test_df = pd.read_excel(excel_path, sheet_name='test')
    
    # 2. Process each split
    for split, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        processed_data = []
        
        for idx, row in df.iterrows():
            # Load .mat file
            mat_file = os.path.join(data_folder, f"{row['filename']}.mat")
            mat_data = scipy.io.loadmat(mat_file)[variable_name]
            
            # Apply preprocessing
            processed = preprocess_pipeline(
                mat_data,
                interpolation_method,
                denoising_method,
                normalization_method
            )
            
            processed_data.append(processed)
        
        # 3. Run inference
        dataset = CustomDataset(processed_data, labels, names)
        results = evaluate_dataset(dataset)
        
        # 4. Save results
        save_to_excel(split, results)
```

## Threshold Optimization (cutoff.py)

### Purpose

Finds optimal probability thresholds for different sensitivity-specificity trade-offs.

### Algorithm

```python
def find_optimal_cutoff(val_probs, val_labels, test_probs, test_labels, target_sensitivities):
    """
    For each target sensitivity level:
    1. Find cutoffs on validation set meeting: sen >= target AND spe >= 0.65
    2. Among candidates, select cutoff maximizing test sensitivity
    3. If no valid cutoff, use cutoff that maximizes validation sensitivity
    """
    
    results = []
    
    for target_sen in [0.95, 0.94, 0.93, ..., 0.83]:
        # Step 1: Find candidate cutoffs on validation set
        candidates = []
        for cutoff in np.linspace(0.0, 1.0, 1000):
            val_preds = (val_probs > cutoff).astype(int)
            sen, spe = calculate_sen_spe(val_preds, val_labels)
            
            if sen >= target_sen and spe >= 0.65:
                candidates.append(cutoff)
        
        # Step 2: Select best cutoff
        if candidates:
            # Choose cutoff maximizing test sensitivity
            best_cutoff = max(
                candidates,
                key=lambda c: calculate_sensitivity(test_probs > c, test_labels)
            )
        else:
            # Fallback: cutoff maximizing validation sensitivity
            best_cutoff = find_max_val_sensitivity_cutoff(val_probs, val_labels)
        
        # Step 3: Calculate metrics for selected cutoff
        val_metrics = calculate_metrics(val_probs > best_cutoff, val_labels)
        test_metrics = calculate_metrics(test_probs > best_cutoff, test_labels)
        
        results.append({
            'target_sensitivity': target_sen,
            'cutoff': best_cutoff,
            'val_sen': val_metrics['sensitivity'],
            'val_spe': val_metrics['specificity'],
            'test_sen': test_metrics['sensitivity'],
            'test_spe': test_metrics['specificity']
        })
    
    return pd.DataFrame(results)
```

### Usage

```python
# Configure paths
weights_folder = "weights/model_CNNLSTMNet/.../bs_16_lr_0.0001_epoch_400_wd_1e-06_hs_64_nl_7"
dataset_paths = {
    'val_data': 'data/npy/dataset/val_data.npy',
    'val_labels': 'data/npy/dataset/val_labels.npy',
    'test_data': 'data/npy/dataset/test_data.npy',
    'test_labels': 'data/npy/dataset/test_labels.npy'
}
output_excel = "threshold_optimization.xlsx"

# Run threshold search
python cutoff.py
```

### Output Excel Structure

| Epoch | Target Sen | Cutoff | Val Sen | Val Spe | Val Acc | Val AUC | Test Sen | Test Spe | Test Acc | Test AUC |
|-------|-----------|--------|---------|---------|---------|---------|----------|----------|----------|----------|
| 78 | 0.95 | 0.2345 | 0.9500 | 0.6800 | 0.8150 | 0.9050 | 0.9300 | 0.7100 | 0.8200 | 0.9120 |
| 78 | 0.94 | 0.2678 | 0.9400 | 0.7200 | 0.8300 | 0.9050 | 0.9200 | 0.7500 | 0.8350 | 0.9120 |
| 78 | 0.93 | 0.2912 | 0.9300 | 0.7500 | 0.8425 | 0.9050 | 0.9100 | 0.7800 | 0.8450 | 0.9120 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Interpretation

**High Sensitivity Configuration (e.g., 0.95):**
- Lower threshold (e.g., 0.23)
- Fewer false negatives (missed cancers)
- More false positives (healthy classified as malignant)
- Use when: Missing cancer is very costly

**Balanced Configuration (e.g., 0.88):**
- Medium threshold (e.g., 0.50)
- Balanced sensitivity and specificity
- Use when: Both errors equally costly

**High Specificity Configuration (e.g., 0.83):**
- Higher threshold (e.g., 0.70)
- Fewer false positives
- More false negatives
- Use when: False alarms are very costly

## Metrics Calculation

### Core Metrics Functions

```python
def calculate_metrics(predictions, labels, probabilities=None):
    """
    Calculate all evaluation metrics
    """
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    
    # Basic metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Matthews Correlation Coefficient
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / denominator if denominator > 0 else 0.0
    
    # AUC (requires probabilities)
    if probabilities is not None:
        auc = roc_auc_score(labels, probabilities)
    else:
        auc = None
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'mcc': mcc,
        'auc': auc,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }
```

### Sensitivity vs Specificity Trade-off

```
High Threshold (e.g., 0.9)          Low Threshold (e.g., 0.1)
├─ Few Positives Predicted          ├─ Many Positives Predicted
├─ High Specificity                 ├─ High Sensitivity
├─ Low Sensitivity                  ├─ Low Specificity
└─ Conservative (avoid FP)          └─ Aggressive (avoid FN)
```

### Clinical Context

**In Breast Cancer Screening:**
- **False Negative (FN)**: Cancer missed → Potentially fatal
- **False Positive (FP)**: Unnecessary biopsy → Patient stress, cost

**Typical Requirements:**
- Sensitivity ≥ 0.85 (catch at least 85% of cancers)
- Specificity ≥ 0.65 (avoid too many false alarms)

## Batch Prediction Workflow

### Step-by-Step Example

```bash
# Step 1: Train model
python train.py \
    --data_path data/npy/phase1_dataset \
    --num_epochs 100 \
    --batch_size 16

# Step 2: Identify best epoch
# Review: weights/[model_folder]/metrics.xlsx
# Select epoch with best validation MCC

# Step 3: Full evaluation
# Edit pro.py:
#   weights_path = "weights/[model_folder]/epoch_078.pth"
python pro.py

# Step 4: Threshold optimization
# Edit cutoff.py:
#   weights_folder = "weights/[model_folder]"
python cutoff.py

# Step 5: Analyze results
# Review:
#   - pro_results.xlsx (detailed predictions)
#   - threshold_optimization.xlsx (optimal cutoffs)
```

## Common Evaluation Issues

### Issue 1: Poor Test Performance

**Symptoms:**
- High train/val metrics, low test metrics
- Large gap between validation and test AUC

**Possible Causes:**
- Overfitting to train/val distributions
- Test set from different distribution
- Data leakage during preprocessing

**Solutions:**
- Retrain with stronger regularization
- Check data preprocessing consistency
- Verify test set independence

### Issue 2: Inconsistent Predictions

**Symptoms:**
- Different predictions on same data
- Metrics vary between runs

**Possible Causes:**
- Model in training mode during inference
- Dropout active during evaluation
- Non-deterministic operations

**Solutions:**
```python
model.eval()  # Set to evaluation mode
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
```

### Issue 3: Memory Errors During Inference

**Symptoms:**
- CUDA out of memory errors
- System crashes on large batches

**Solutions:**
```python
# Reduce batch size
dataloader = DataLoader(dataset, batch_size=8)

# Use CPU if necessary
device = torch.device('cpu')

# Clear cache between batches
torch.cuda.empty_cache()
```

## Performance Analysis

### Confusion Matrix Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
```

### ROC Curve

```python
def plot_roc_curve(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc_score = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
```

## Design Patterns

**Strategy Pattern**: Pluggable threshold selection strategies
**Template Method**: Consistent evaluation pipeline
**Factory Pattern**: Dynamic dataset creation

## Next Steps After Evaluation

1. **Analyze Misclassifications**: Identify common failure patterns
2. **Clinical Validation**: Work with domain experts
3. **Error Analysis**: Review false positives and false negatives
4. **Model Refinement**: Adjust based on evaluation insights
5. **Deployment Preparation**: Export model for production use
