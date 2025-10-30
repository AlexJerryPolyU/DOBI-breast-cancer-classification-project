# Data Pipeline

## Input Data Format Requirements

### Raw Data Format (.mat files)

The input data must be in MATLAB .mat format with the following specifications:

**For 3-LED mode (smaller breast sizes A-C):**
```
Shape: (102, 128, 25, 3)
- Dimension 1: 102 (image height)
- Dimension 2: 128 (image width)
- Dimension 3: 25 (number of LED scanning cycles)
- Dimension 4: 3 (number of LEDs)
```

**For 5-LED mode (larger breast sizes D+):**
```
Shape: (102, 128, 15, 5)
- Dimension 1: 102 (image height)
- Dimension 2: 128 (image width)
- Dimension 3: 15 (number of LED scanning cycles)
- Dimension 4: 5 (number of LEDs)
```

### Data Files Organization

```
data/
├── excel/
│   └── phase1exp.xlsx          # Phase 1 dataset splits
│       ├── train (sheet)       # Training set filenames and labels
│       ├── val (sheet)         # Validation set filenames and labels
│       └── test (sheet)        # Test set filenames and labels
├── source data/
│   └── [reconstruction_type]/  # Raw .mat files
│       ├── patient001.mat
│       ├── patient002.mat
│       └── ...
└── npy/
    └── [processed_dataset]/    # Generated .npy files (after preprocessing)
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

## Excel File Format

The Excel file (`phase1exp.xlsx`) must contain three sheets with the following structure:

### Sheet Structure: "train", "val", "test"

| Column Name | Description | Format |
|-------------|-------------|--------|
| filename | Patient data filename (without extension) | String (e.g., "patient001") |
| label | Classification label | Integer (0 = benign, 1 = malignant) |

**Example:**
```
| filename    | label |
|-------------|-------|
| patient001  | 0     |
| patient002  | 1     |
| patient003  | 0     |
```

## Preprocessing Pipeline

### Configuration File: `dataset/config_data.py`

#### Step 1: Basic Parameters

```python
# Excel configuration
parser.add_argument('--excel_path', type=str, 
                    default='data/excel/phase1exp.xlsx',
                    help='Excel file path for dataset splits')

parser.add_argument('--sheet_names', type=str, nargs='+', 
                    default=['train', 'val', 'test'],
                    help='Sheet names in Excel file')
```

#### Step 2: Data Source Configuration

```python
# Reconstruction type
parser.add_argument('--single_layer', type=bool, default=False, 
                    help='True: 1-layer reconstruction, False: 9-layer reconstruction')

# Variable name in .mat files
parser.add_argument('--variable_name', type=str, default='img4D', 
                    help='Variable name in .mat file')

# Data paths
parser.add_argument('--data_folder', type=str,  
                    default='data/source data/fdDOT_DynamicFEM_ellips_height', 
                    help='Path to raw .mat files')

parser.add_argument('--cache_folder', type=str, 
                    default='data/npy/fdDOT_DynamicFEM_ellips_height_npy', 
                    help='Path to save processed .npy files')
```

#### Step 3: Preprocessing Methods

```python
# Preprocessing parameters (0 = disabled)
parser.add_argument('--interpolation_method', type=int, default=0, 
                    help='Interpolation method (0-5)')
parser.add_argument('--denoising_method', type=int, default=0, 
                    help='Denoising method (0-11)')
parser.add_argument('--normalization_method', type=int, default=0, 
                    help='Normalization method (0-2)')
parser.add_argument('--enhancement_method', type=int, default=0, 
                    help='Enhancement method (0-4)')
parser.add_argument('--edge_detection_method', type=int, default=0, 
                    help='Edge detection method (0-11)')
```

## Preprocessing Methods Reference

### Interpolation Methods (`--interpolation_method`)

| Value | Method |
|-------|--------|
| 0 | Nearest Neighbor |
| 1 | Bilinear |
| 2 | Quadratic |
| 3 | Cubic |
| 4 | Quartic |
| 5 | Quintic |

### Denoising Methods (`--denoising_method`)

| Value | Method |
|-------|--------|
| 0 | None (disabled) |
| 1 | Non-Local Means Denoising |
| 2 | Gaussian Smoothing |
| 3 | Median Filtering |
| 4 | Wiener Filter Deconvolution |
| 5 | Richardson-Lucy Deconvolution |
| 6 | Total Variation Denoising |
| 7 | Bilateral Filtering |
| 8 | Wavelet Denoising |
| 9 | Inpainting for Missing Pixels |
| 10 | Isotropic Diffusion |
| 11 | Anisotropic Diffusion |

### Normalization Methods (`--normalization_method`)

| Value | Method |
|-------|--------|
| 0 | None (disabled) |
| 1 | Min-Max Normalization |
| 2 | Z-score Normalization |

### Enhancement Methods (`--enhancement_method`)

| Value | Method |
|-------|--------|
| 0 | None (disabled) |
| 1 | CLAHE |
| 2 | Histogram Equalization |
| 3 | Horizontal Flip |
| 4 | Vertical Flip |

### Edge Detection Methods (`--edge_detection_method`)

| Value | Method |
|-------|--------|
| 0 | None (disabled) |
| 1 | Sobel |
| 2 | Canny |
| 3 | Prewitt |
| 4 | Roberts |
| 5 | Laplacian of Gaussian |
| 6 | Difference of Gaussians |
| 7 | Scharr |
| 8 | Gabor |
| 9 | Hessian |
| 10 | Morphological |
| 11 | Wavelet Transform |

## Running Data Preprocessing

### Execute preprocessing:

```bash
python dataset/data_npy.py
```

This will:
1. Read the Excel file to get train/val/test splits
2. Load corresponding .mat files
3. Apply selected preprocessing methods
4. Save processed data as .npy files:
   - `{sheet_name}_data.npy` - Processed image data
   - `{sheet_name}_labels.npy` - Classification labels
   - `{sheet_name}_names.npy` - Patient filenames

### Output Files

After preprocessing, you will have:
```
data/npy/{dataset_name}/
├── train_data.npy      # Shape: (N, 25, C, H, W) for 3-LED mode
├── train_labels.npy    # Shape: (N,)
├── train_names.npy     # Shape: (N,)
├── val_data.npy        # Shape: (N, 25, C, H, W) for 3-LED mode
├── val_labels.npy      # Shape: (N,)
├── val_names.npy       # Shape: (N,)
├── test_data.npy       # Shape: (N, 25, C, H, W) for 3-LED mode
├── test_labels.npy     # Shape: (N,)
└── test_names.npy      # Shape: (N,)
```

**Note:** The third dimension is 25 for 3-LED mode (standard breast sizes A-C) or 15 for 5-LED mode (larger sizes D+).

## Data Loading in Training

The processed .npy files are loaded by `dobi_dataset.py`:

```python
class CustomDataset(Dataset):
    def __init__(self, data_path, labels_path, names_path=None):
        self.data = np.load(data_path)
        self.labels = np.load(labels_path)
        self.names = np.load(names_path) if names_path else None
```

The dataset automatically:
- Converts single-channel data to 3-channel (for pretrained models)
- Returns tensors of shape: `(batch_size, 25, channels, height, width)` for 3-LED mode
- Provides patient names for tracking predictions
