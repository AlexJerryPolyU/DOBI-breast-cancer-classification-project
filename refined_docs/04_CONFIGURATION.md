# Configuration Management

## Configuration Files

The system uses multiple configuration sources:

1. **config.py** - Command-line arguments and training parameters
2. **config.yml** - YAML-based configuration (optional)
3. **model_configs.json** - Model-specific hyperparameters

## Training Configuration (`config.py`)

### Network Architecture Parameters

```python
# Reconstruction layers
parser.add_argument('--Recon_num_layers', type=int, default=9, 
                    choices=[1, 9],
                    help='Number of reconstruction layers in input image')

# LSTM Configuration
parser.add_argument('--hidden_size', type=int, default=64, 
                    help='Hidden size for LSTM layers')
parser.add_argument('--num_lstm_layers', type=int, default=7, 
                    help='Number of LSTM layers')
parser.add_argument('--input_size', type=int, default=128, 
                    help='Input size for LSTM')

# Output Configuration
parser.add_argument('--output_size', type=int, default=1, 
                    help='Output size (1 for binary classification)')
```

### Training Hyperparameters

```python
# Basic Training Parameters
parser.add_argument('--batch_size', type=int, default=16, 
                    help='Training batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, 
                    help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-6, 
                    help='L2 regularization weight decay')
parser.add_argument('--num_epochs', type=int, default=400, 
                    help='Maximum number of training epochs')
```

### Data Paths Configuration

```python
# Root data directory
parser.add_argument('--data_path', type=str, 
                    default='data/npy/dDOT_9layers_DOBI_Recon_npy_interpolation_4_denoising_0_normalization_0_enhancement_0',
                    help='Root path to the preprocessed dataset')

# Training data
parser.add_argument('--train_data', type=str, default='train_data.npy',
                    help='Filename of training data')
parser.add_argument('--train_labels', type=str, default='train_labels.npy',
                    help='Filename of training labels')
parser.add_argument('--train_names', type=str, default='train_names.npy',
                    help='Filename of training patient names')

# Validation data
parser.add_argument('--val_data', type=str, default='val_data.npy',
                    help='Filename of validation data')
parser.add_argument('--val_labels', type=str, default='val_labels.npy',
                    help='Filename of validation labels')
parser.add_argument('--val_names', type=str, default='val_names.npy',
                    help='Filename of validation patient names')

# Test data
parser.add_argument('--test_data', type=str, default='test_data.npy',
                    help='Filename of test data')
parser.add_argument('--test_labels', type=str, default='test_labels.npy',
                    help='Filename of test labels')
parser.add_argument('--test_names', type=str, default='test_names.npy',
                    help='Filename of test patient names')
```

### Output Configuration

```python
# Metrics file
parser.add_argument('--metrics_file', type=str, 
                    default='dDOT_9layers_DOBI_Recon_i_4_d_0_n_0_e_0.xlsx',
                    help='Filename for saving training metrics and results')

# Weights directory (generated automatically)
# Format: weights/{model_config}/{dataset_name}/
```

### Model Selection

```python
# Model architecture selection
parser.add_argument('--model_module', type=str, default='model',
                    choices=['model', 'lstm_mil', 'lambda_net', 'cnn_mil'],
                    help='Model module to import')
parser.add_argument('--model_class', type=str, default='CNNLSTMNet',
                    help='Model class name')

# Model-specific options
parser.add_argument('--use_conv1to3', action='store_true',
                    help='Use 1-to-3 channel conversion')
parser.add_argument('--use_attention', action='store_true',
                    help='Enable attention mechanisms')
parser.add_argument('--cnn_input_channels', type=int, default=3,
                    help='Number of input channels for CNN')
```

## Model-Specific Configuration (`model_configs.json`)

This file contains hyperparameters specific to each model architecture:

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
        "mil_pooling": "attention"
    }
}
```

## Configuration Best Practices

### 1. Data Path Configuration

When preprocessing data with specific methods, use a descriptive path:

```
data/npy/{reconstruction}_{interpolation}_i_{value}_d_{value}_n_{value}_e_{value}
```

Example:
```
data/npy/dDOT_9layers_DOBI_Recon_npy_interpolation_4_denoising_0_normalization_0_enhancement_0
```

### 2. Metrics File Naming

Match the metrics filename to the data configuration:

```python
--metrics_file='dDOT_9layers_DOBI_Recon_i_4_d_0_n_0_e_0.xlsx'
```

### 3. Reconstruction Layer Consistency

Ensure `--Recon_num_layers` matches your preprocessed data:
- Set to `1` if using single-layer reconstruction
- Set to `9` if using nine-layer reconstruction

### 4. Batch Size Selection

Choose batch size based on available GPU memory:
- **16 GB GPU:** batch_size = 16
- **24 GB GPU:** batch_size = 32
- **12 GB GPU:** batch_size = 8

## Example Configurations

### Configuration 1: Standard Training (9-layer)

```bash
python train.py \
    --Recon_num_layers 9 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --num_epochs 400 \
    --data_path 'data/npy/dDOT_9layers_DOBI_Recon_npy_interpolation_4_denoising_0_normalization_0_enhancement_0' \
    --metrics_file 'phase1_9layer_results.xlsx'
```

### Configuration 2: Single-layer with Attention

```bash
python train.py \
    --Recon_num_layers 1 \
    --use_attention \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --num_epochs 300 \
    --data_path 'data/npy/dDOT_1layer_DOBI_Recon_npy_interpolation_3_denoising_1_normalization_2_enhancement_1' \
    --metrics_file 'phase1_1layer_attention_results.xlsx'
```

### Configuration 3: MIL Architecture

```bash
python train.py \
    --model_module lstm_mil \
    --model_class CNNLSTMNetMILins \
    --Recon_num_layers 9 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --data_path 'data/npy/dDOT_9layers_DOBI_Recon_npy_interpolation_4_denoising_0_normalization_0_enhancement_0' \
    --metrics_file 'phase1_mil_results.xlsx'
```

## Verifying Configuration

Before training, verify your configuration:

```python
# In train.py or as a separate check
print(f"Data path: {args.data_path}")
print(f"Reconstruction layers: {args.Recon_num_layers}")
print(f"Batch size: {args.batch_size}")
print(f"Learning rate: {args.learning_rate}")
print(f"Model: {args.model_module}.{args.model_class}")
print(f"Output metrics: {args.metrics_file}")
```
