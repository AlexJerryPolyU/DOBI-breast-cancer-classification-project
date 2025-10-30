# Usage Examples and Workflows

## Quick Start Guide

### Prerequisites

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Verify Data Structure**
```
data/
├── excel/
│   └── phase1exp.xlsx
├── source data/
│   └── [reconstruction_folder]/
│       ├── patient001.mat
│       └── ...
└── npy/
    └── (will be created)
```

3. **Check Excel Format**
```
Sheet: train/val/test
Columns: filename | label
```

## Workflow 1: Complete Training Pipeline

### Step 1: Data Preprocessing

```bash
# Navigate to dataset directory
cd dataset

# Edit config_data.py or use command line arguments
python data_npy.py \
    --excel_path ../data/excel/phase1exp.xlsx \
    --data_folder ../data/source\ data/fdDOT_DynamicFEM_ellips_height \
    --cache_folder ../data/npy/phase1_i4_d0_n0_e0 \
    --variable_name img4D \
    --interpolation_method 4 \
    --denoising_method 0 \
    --normalization_method 0 \
    --enhancement_method 0 \
    --single_layer False
```

**Output:**
```
data/npy/phase1_i4_d0_n0_e0/
├── train_data.npy
├── train_labels.npy
├── train_names.npy
├── val_data.npy
├── val_labels.npy
├── val_names.npy
├── test_data.npy
├── test_labels.npy
└── test_names.npy
```

### Step 2: Train Model

```bash
# Return to root directory
cd ..

# Train baseline model
python train.py \
    --data_path data/npy/phase1_i4_d0_n0_e0 \
    --metrics_file phase1_baseline_results.xlsx \
    --model_module model \
    --model_class CNNLSTMNet \
    --Recon_num_layers 9 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --num_epochs 100 \
    --hidden_size 64 \
    --num_lstm_layers 7
```

**Monitor Training:**
- Watch console output for loss/metrics
- Check `weights/[model_folder]/metrics.xlsx` periodically

### Step 3: Identify Best Epoch

```python
# In Python or Excel
import pandas as pd

metrics = pd.read_excel('weights/[model_folder]/metrics.xlsx')

# Find epoch with best validation MCC
best_epoch = metrics.loc[metrics['Val MCC'].idxmax()]
print(f"Best epoch: {best_epoch.name}")
print(f"Val MCC: {best_epoch['Val MCC']:.4f}")
print(f"Test MCC: {best_epoch['Test MCC']:.4f}")
```

### Step 4: Full Evaluation

```python
# Edit pro.py
weights_path = "weights/[model_folder]/epoch_078.pth"
dataset_paths = {
    'train_data': 'data/npy/phase1_i4_d0_n0_e0/train_data.npy',
    'train_labels': 'data/npy/phase1_i4_d0_n0_e0/train_labels.npy',
    'train_names': 'data/npy/phase1_i4_d0_n0_e0/train_names.npy',
    # ... val and test paths
}

# Set model configuration
args.model_module = "model"
args.model_class = "CNNLSTMNet"
args.use_conv1to3 = False
args.use_attention = False

# Run evaluation
python pro.py
```

### Step 5: Threshold Optimization

```python
# Edit cutoff.py
weights_folder = "weights/[model_folder]"
dataset_paths = {
    'val_data': 'data/npy/phase1_i4_d0_n0_e0/val_data.npy',
    'val_labels': 'data/npy/phase1_i4_d0_n0_e0/val_labels.npy',
    'test_data': 'data/npy/phase1_i4_d0_n0_e0/test_data.npy',
    'test_labels': 'data/npy/phase1_i4_d0_n0_e0/test_labels.npy'
}
output_excel = "threshold_results.xlsx"

# Run optimization
python cutoff.py
```

## Workflow 2: Experiment with Different Preprocessing

### Test Multiple Preprocessing Combinations

```bash
# Combination 1: Cubic interpolation only
python dataset/data_npy.py \
    --cache_folder data/npy/phase1_i3_d0_n0_e0 \
    --interpolation_method 3 \
    --denoising_method 0 \
    --normalization_method 0 \
    --enhancement_method 0

# Train model
python train.py \
    --data_path data/npy/phase1_i3_d0_n0_e0 \
    --metrics_file phase1_i3_results.xlsx

# Combination 2: Cubic + Gaussian denoising
python dataset/data_npy.py \
    --cache_folder data/npy/phase1_i3_d2_n0_e0 \
    --interpolation_method 3 \
    --denoising_method 2 \
    --normalization_method 0 \
    --enhancement_method 0

# Train model
python train.py \
    --data_path data/npy/phase1_i3_d2_n0_e0 \
    --metrics_file phase1_i3_d2_results.xlsx

# Combination 3: Cubic + Gaussian + Z-score normalization
python dataset/data_npy.py \
    --cache_folder data/npy/phase1_i3_d2_n2_e0 \
    --interpolation_method 3 \
    --denoising_method 2 \
    --normalization_method 2 \
    --enhancement_method 0

# Train model
python train.py \
    --data_path data/npy/phase1_i3_d2_n2_e0 \
    --metrics_file phase1_i3_d2_n2_results.xlsx
```

### Compare Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load all results
results = {
    'i3_d0_n0_e0': pd.read_excel('weights/.../phase1_i3_results.xlsx'),
    'i3_d2_n0_e0': pd.read_excel('weights/.../phase1_i3_d2_results.xlsx'),
    'i3_d2_n2_e0': pd.read_excel('weights/.../phase1_i3_d2_n2_results.xlsx')
}

# Plot validation MCC comparison
plt.figure(figsize=(12, 6))
for name, df in results.items():
    plt.plot(df['Val MCC'], label=name)
plt.xlabel('Epoch')
plt.ylabel('Validation MCC')
plt.legend()
plt.title('Preprocessing Method Comparison')
plt.savefig('preprocessing_comparison.png')
```

## Workflow 3: Model Architecture Comparison

### Test Different Architectures

```bash
# 1. Baseline CNN-LSTM
python train.py \
    --model_module model \
    --model_class CNNLSTMNet \
    --data_path data/npy/phase1_i4_d0_n0_e0 \
    --metrics_file baseline.xlsx

# 2. CNN-LSTM with Attention
python train.py \
    --model_module model \
    --model_class CNNLSTMNet \
    --use_attention \
    --data_path data/npy/phase1_i4_d0_n0_e0 \
    --metrics_file with_attention.xlsx

# 3. MIL Instance-Level
# First, edit model_configs.json:
# "lstm_mil.CNNLSTMNet": {"mil_type": 2}
python train.py \
    --model_module lstm_mil \
    --model_class CNNLSTMNet \
    --data_path data/npy/phase1_i4_d0_n0_e0 \
    --metrics_file mil_instance.xlsx

# 4. Lambda Networks
python train.py \
    --model_module lambda_net \
    --model_class CNNLSTMNet \
    --data_path data/npy/phase1_i4_d0_n0_e0 \
    --metrics_file lambda_net.xlsx
```

### Compare Model Performance

```python
import pandas as pd

models = {
    'Baseline': 'baseline.xlsx',
    'Attention': 'with_attention.xlsx',
    'MIL Instance': 'mil_instance.xlsx',
    'Lambda Net': 'lambda_net.xlsx'
}

# Extract best performance
summary = []
for name, file in models.items():
    df = pd.read_excel(f'weights/.../{file}')
    best_idx = df['Val MCC'].idxmax()
    summary.append({
        'Model': name,
        'Best Epoch': best_idx,
        'Val MCC': df.loc[best_idx, 'Val MCC'],
        'Test MCC': df.loc[best_idx, 'Test MCC'],
        'Val AUC': df.loc[best_idx, 'Val AUC'],
        'Test AUC': df.loc[best_idx, 'Test AUC']
    })

summary_df = pd.DataFrame(summary)
print(summary_df)
summary_df.to_excel('model_comparison.xlsx', index=False)
```

## Workflow 4: Hyperparameter Tuning

### Manual Grid Search

```bash
# Learning Rate
for lr in 0.00001 0.0001 0.001; do
    python train.py \
        --learning_rate $lr \
        --data_path data/npy/phase1_i4_d0_n0_e0 \
        --metrics_file lr_${lr}.xlsx
done

# Batch Size
for bs in 8 16 32; do
    python train.py \
        --batch_size $bs \
        --data_path data/npy/phase1_i4_d0_n0_e0 \
        --metrics_file bs_${bs}.xlsx
done

# LSTM Layers
for nl in 5 7 9; do
    python train.py \
        --num_lstm_layers $nl \
        --data_path data/npy/phase1_i4_d0_n0_e0 \
        --metrics_file nl_${nl}.xlsx
done

# Hidden Size
for hs in 32 64 128; do
    python train.py \
        --hidden_size $hs \
        --data_path data/npy/phase1_i4_d0_n0_e0 \
        --metrics_file hs_${hs}.xlsx
done
```

### Automated Wavelet Search

```bash
# Runs 216 experiments automatically
python Auto_train.py

# Results saved in separate folders per configuration
# Review all results:
python -c "
import os
import pandas as pd

results = []
for root, dirs, files in os.walk('weights'):
    for file in files:
        if file == 'metrics.xlsx':
            path = os.path.join(root, file)
            df = pd.read_excel(path)
            best_mcc = df['Val MCC'].max()
            results.append({'path': path, 'best_val_mcc': best_mcc})

results_df = pd.DataFrame(results).sort_values('best_val_mcc', ascending=False)
print(results_df.head(10))
"
```

## Workflow 5: Testing Pretrained Model

### Use Existing Checkpoint

```python
# Create test script: test_pretrained.py
import torch
from dobi_dataset import CustomDataset
from torch.utils.data import DataLoader
import numpy as np

# Load model
from model.model import CNNLSTMNet
model = CNNLSTMNet(
    input_size=128,
    hidden_size=64,
    num_layers=7,
    output_size=1,
    cnn_input_channels=3
)

# Load weights
checkpoint = torch.load('epoch/epoch_078.pth')
model.load_state_dict(checkpoint)
model.eval()
model.to('cuda')

# Load test data
test_dataset = CustomDataset(
    'data/npy/phase1_i4_d0_n0_e0/test_data.npy',
    'data/npy/phase1_i4_d0_n0_e0/test_labels.npy',
    'data/npy/phase1_i4_d0_n0_e0/test_names.npy'
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Predict
predictions = []
probabilities = []
true_labels = []

with torch.no_grad():
    for images, labels, names in test_loader:
        images = images.to('cuda')
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        
        predictions.extend(preds.flatten())
        probabilities.extend(probs.flatten())
        true_labels.extend(labels.numpy())

# Calculate metrics
from sklearn.metrics import accuracy_score, roc_auc_score
acc = accuracy_score(true_labels, predictions)
auc = roc_auc_score(true_labels, probabilities)

print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
```

```bash
python test_pretrained.py
```

## Workflow 6: Clinical Deployment Simulation

### End-to-End Prediction Pipeline

```python
# deploy.py
import torch
import scipy.io
import numpy as np
from model.model import CNNLSTMNet
from dataset.image_processing import ImageProcessor

class ClinicalPredictor:
    def __init__(self, weights_path, threshold=0.5):
        self.model = CNNLSTMNet(
            input_size=128,
            hidden_size=64,
            num_layers=7,
            output_size=1,
            cnn_input_channels=3
        )
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
        self.model.to('cuda')
        self.threshold = threshold
        self.processor = ImageProcessor()
    
    def preprocess(self, mat_file_path, variable_name='img4D'):
        # Load raw data
        mat_data = scipy.io.loadmat(mat_file_path)[variable_name]
        
        # Apply preprocessing
        processed = self.processor.interpolate(mat_data, method=4)
        processed = self.processor.normalize(processed)
        
        # Convert to tensor
        tensor = torch.from_numpy(processed).float()
        
        # Add batch dimension and convert channels
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1, 1)
        
        return tensor.unsqueeze(0).to('cuda')
    
    def predict(self, mat_file_path):
        # Preprocess
        input_tensor = self.preprocess(mat_file_path)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probability = torch.sigmoid(output).item()
        
        # Classify
        prediction = 1 if probability >= self.threshold else 0
        confidence = probability if prediction == 1 else (1 - probability)
        
        return {
            'prediction': 'Malignant' if prediction == 1 else 'Benign',
            'probability': probability,
            'confidence': confidence,
            'recommendation': self.get_recommendation(probability)
        }
    
    def get_recommendation(self, probability):
        if probability >= 0.85:
            return "High risk - Immediate biopsy recommended"
        elif probability >= 0.65:
            return "Moderate risk - Further imaging recommended"
        elif probability >= 0.35:
            return "Low risk - Routine follow-up"
        else:
            return "Very low risk - Annual screening"

# Usage
predictor = ClinicalPredictor(
    weights_path='epoch/epoch_078.pth',
    threshold=0.5
)

# Test on new patient
result = predictor.predict('data/new_patient.mat')
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.3f}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Recommendation: {result['recommendation']}")
```

## Common Task Recipes

### Recipe 1: Quick Single Run

```bash
# One-liner: Preprocess + Train
python dataset/data_npy.py --cache_folder data/npy/quick_test && \
python train.py --data_path data/npy/quick_test --num_epochs 50
```

### Recipe 2: Resume Training (Not Natively Supported)

```python
# Workaround: Load checkpoint and continue
# Modify train.py to add:

# After model creation:
start_epoch = 0
if args.resume_from:
    checkpoint = torch.load(args.resume_from)
    model.load_state_dict(checkpoint)
    start_epoch = int(args.resume_from.split('_')[-1].split('.')[0])

# In training loop:
for epoch in range(start_epoch, num_epochs):
    # ... training code
```

### Recipe 3: Batch Inference on New Data

```python
# batch_inference.py
import glob
import pandas as pd

predictor = ClinicalPredictor('epoch/epoch_078.pth')

# Process all files in folder
mat_files = glob.glob('data/new_patients/*.mat')
results = []

for mat_file in mat_files:
    try:
        result = predictor.predict(mat_file)
        result['filename'] = os.path.basename(mat_file)
        results.append(result)
    except Exception as e:
        print(f"Error processing {mat_file}: {e}")

# Save results
df = pd.DataFrame(results)
df.to_excel('batch_predictions.xlsx', index=False)
print(f"Processed {len(results)} patients")
```

### Recipe 4: Export Model for Deployment

```python
# export_model.py
import torch
from model.model import CNNLSTMNet

# Load model
model = CNNLSTMNet(
    input_size=128,
    hidden_size=64,
    num_layers=7,
    output_size=1,
    cnn_input_channels=3
)
model.load_state_dict(torch.load('epoch/epoch_078.pth'))
model.eval()

# Export to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save('model_deployment.pt')

print("Model exported to model_deployment.pt")
```

## Troubleshooting Common Issues

### Issue: Import Errors

```bash
# Solution: Install missing packages
pip install -r requirements.txt

# Or individually
pip install torch torchvision scikit-learn pandas numpy scipy
```

### Issue: CUDA Out of Memory

```bash
# Solution: Reduce batch size
python train.py --batch_size 8  # or 4

# Or use CPU
python train.py --device cpu
```

### Issue: Data Shape Mismatch

```python
# Check data shapes
import numpy as np
data = np.load('data/npy/train_data.npy')
print(f"Data shape: {data.shape}")
# Expected: (N, 25, C, H, W) where C=1 or 3, H=102, W=128
```

### Issue: Low Performance

```bash
# Try different preprocessing
python dataset/data_npy.py --interpolation_method 4 --denoising_method 2

# Or different architecture
python train.py --use_attention --num_lstm_layers 9
```

## Next Steps

After completing these workflows:
1. Analyze results systematically
2. Document best configurations
3. Prepare models for clinical validation
4. Consider ensemble methods for robustness
