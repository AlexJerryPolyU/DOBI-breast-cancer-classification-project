# Model Architecture

## Available Architectures

| Model Module | Class | Description |
|-------------|-------|-------------|
| `model.py` | `CNNLSTMNet` | Base: DenseNet121 + LSTM |
| `lstm_mil.py` | `CNNLSTMNet` | + Multiple Instance Learning (3 variants) |
| `lambda_net.py` | `CNNLSTMNet` | + Lambda layers (LambdaLayer) |
| `cnn_mil.py` | `CNNLSTMNetMILins/bag` | MIL at CNN level |
| `time_mil.py` | `CNNLSTMNet` | + Wavelet transforms |
| `lstm_attention.py` | `CNNLSTMNet` | + Custom attention types |

## Base Architecture (model.py)

### Model Parameters

```python
CNNLSTMNet(
    input_size=128,           # LSTM input dimension
    hidden_size=64,           # LSTM hidden units
    num_layers=7,             # LSTM depth
    output_size=1,            # Binary classification
    cnn_input_channels=3,     # 1 or 3
    use_attention=False,      # SE-block toggle
    use_conv1to3=False        # 1→3 channel conversion
)
```

### Architecture Flow

```
┌─────────────────────┐
│  Input: (B, 25, C, H, W)
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Optional: Conv1→3   │  (if cnn_input_channels=1 & use_conv1to3)
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  DenseNet121         │  Per-frame feature extraction
│  (Pretrained)        │  Output: (B, 25, 1024)
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Optional: SE-block  │  (if use_attention=True)
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  FC: 1024→128        │  Dimension reduction
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  LSTM (7 layers)     │  Temporal modeling
│  Dropout: 0.5        │  Output: (B, 25, 64)
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Take Last Timestep  │  (B, 64)
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  FC: 64→1            │  Classification head
│  Dropout: 0.7        │
└──────────┬──────────┘
           ↓
    Output Logits
```

### Component Details

**DenseNet121 Backbone:**
- Pretrained on ImageNet
- Extracts spatial features from each frame independently
- Output feature dimension: 1024

**LSTM Temporal Processing:**
- 7-layer bidirectional LSTM
- Hidden size: 64
- Dropout rate: 0.5 between layers
- Captures temporal dependencies across frames

**Classification Head:**
- Fully connected layer: 64 → 1
- Dropout: 0.7
- Output: Raw logits (no sigmoid)

**Optional SE-Block Attention:**
- Squeeze-and-Excitation mechanism
- Applied after DenseNet features
- Recalibrates channel-wise features

## MIL Variants (lstm_mil.py)

Multiple Instance Learning treats the temporal sequence as a "bag" of instances.

### Instance-Level MIL (`mil_type=2`)

```
Features (B, 25, 1024)
    ↓
Attention Weights (25,)
    ↓
Weighted Sum → (B, 1024)
    ↓
Classifier
```

**Characteristics:**
- Attention over all 25 timesteps
- Learns which frames are most important
- Single representation per sequence

### Bag-Level MIL (`mil_type=1`)

```
Sliding Windows of Features
    ↓
Each Window → Representation
    ↓
Attention over Window Representations
    ↓
Classifier
```

**Characteristics:**
- Creates multiple "bags" from subsequences
- Attention over bag representations
- Captures local temporal patterns

### Bag-Instance Hybrid (`mil_type=3`)

```
Instance MIL → Representation_1
Bag MIL → Representation_2
    ↓
Average(Representation_1, Representation_2)
    ↓
Classifier
```

**Characteristics:**
- Combines both MIL approaches
- Balances global and local attention
- Most flexible but computationally expensive

## Wavelet Models (time_mil.py)

### Wavelet Transform Parameters

```python
WaveletTransform(
    wavelet_type='mexican_hat' | 'haar' | 'morlet' | 'gabor',
    kernel_size=1-3,
    num_wavelets=1-3,
    multi_scale=1-3,
    use_fourier=True/False
)
```

### Available Wavelet Types

| Wavelet | Characteristics | Best For |
|---------|-----------------|----------|
| **Mexican Hat** | Smooth, symmetric | General feature extraction |
| **Haar** | Sharp, discontinuous | Edge detection |
| **Morlet** | Sinusoidal with Gaussian envelope | Oscillatory patterns |
| **Gabor** | Oriented, frequency-selective | Texture analysis |

### Processing Flow

```
CNN Features (B, 25, 1024)
    ↓
Wavelet Transform (per scale)
    ↓
Multi-scale Features (B, 25, 1024 × scales)
    ↓
Optional: Fourier Transform
    ↓
LSTM Processing
    ↓
Classifier
```

## Lambda Networks (lambda_net.py)

Replaces traditional attention with Lambda layers for improved efficiency.

### Lambda Layer Characteristics

- **Query-Key-Value Free**: No explicit Q-K-V transformations
- **Linear Complexity**: O(n) instead of O(n²)
- **Position-Aware**: Encodes positional information directly

### Architecture Integration

```
DenseNet Features
    ↓
Lambda Layer (global context)
    ↓
Lambda Layer (local context)
    ↓
LSTM Processing
    ↓
Classifier
```

## Model Configuration (model_configs.json)

### Example Configurations

```json
{
    "model.CNNLSTMNet": {
        "hidden_size": 64,
        "num_lstm_layers": 7,
        "dropout_rate": 0.5
    },
    "lstm_mil.CNNLSTMNetMILins": {
        "hidden_size": 64,
        "num_lstm_layers": 5,
        "mil_pooling": "attention",
        "mil_type": 2
    },
    "time_mil.CNNLSTMNet": {
        "wavelet_type": "mexican_hat",
        "kernel_size": 2,
        "num_wavelets": 2,
        "multi_scale": 2,
        "use_fourier": false
    }
}
```

## Model Selection Guide

### Choose Base CNN-LSTM When:
- Starting baseline experiments
- Limited computational resources
- Interpretability is important

### Choose MIL Variants When:
- Temporal attention is needed
- Not all frames are equally important
- Working with variable-length sequences

### Choose Lambda Networks When:
- Computational efficiency is critical
- Large batch sizes needed
- Memory constraints exist

### Choose Wavelet Models When:
- Frequency information is relevant
- Multi-scale analysis needed
- Oscillatory patterns present

## Design Patterns

**Factory Pattern**: Dynamic model instantiation via `create_model()`
```python
def create_model(model_module, model_class, **params):
    module = importlib.import_module(f'model.{model_module}')
    ModelClass = getattr(module, model_class)
    return ModelClass(**params)
```

**Builder Pattern**: Flexible parameter composition
- Model configs loaded from JSON
- CLI arguments override defaults
- Runtime parameter merging

## Key Hyperparameters Summary

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `hidden_size` | 64 | 32-256 | LSTM capacity |
| `num_lstm_layers` | 7 | 1-10 | Temporal depth |
| `dropout_rate` | 0.5 | 0.0-0.7 | Regularization |
| `cnn_input_channels` | 3 | 1 or 3 | Input format |
| `use_attention` | False | Boolean | Feature recalibration |
| `use_conv1to3` | False | Boolean | Channel conversion |

## Adding New Model Architectures

### Step 1: Create Model File

```python
# model/my_model.py
import torch.nn as nn

class MyCustomNet(nn.Module):
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__()
        # Your architecture here
        
    def forward(self, x):
        # Forward pass logic
        return output
```

### Step 2: Add Configuration

```json
// model_configs.json
{
    "my_model.MyCustomNet": {
        "custom_param1": value1,
        "custom_param2": value2
    }
}
```

### Step 3: Run Training

```bash
python train.py \
    --model_module my_model \
    --model_class MyCustomNet \
    --data_path data/npy/your_dataset
```

## Performance Considerations

### Memory Usage (Approximate)

| Model | Batch Size 16 | Batch Size 32 |
|-------|---------------|---------------|
| Base CNN-LSTM | ~8 GB | ~14 GB |
| MIL Instance | ~9 GB | ~16 GB |
| MIL Bag | ~11 GB | ~20 GB |
| Wavelet (3 scales) | ~12 GB | ~22 GB |

### Training Speed (Relative)

| Model | Speed | Epochs/Hour (V100) |
|-------|-------|-------------------|
| Base CNN-LSTM | 1.0x | ~25 |
| MIL Instance | 0.9x | ~22 |
| MIL Bag | 0.7x | ~17 |
| Lambda Network | 1.1x | ~27 |
| Wavelet | 0.8x | ~20 |

### Recommendation

For Phase 1 data:
- Start with **Base CNN-LSTM** (baseline)
- Add **SE-Block attention** if overfitting
- Try **MIL Instance** for interpretability
- Use **Wavelet** only if frequency patterns evident
