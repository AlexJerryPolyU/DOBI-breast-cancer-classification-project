# Glossary and References

## Terminology

### Medical Imaging Terms

| Term | Full Name | Description |
|------|-----------|-------------|
| **fNIR** | Full Near-Infrared | Baseline NIR imaging capturing absolute tissue optical properties |
| **dNIR** | Difference Near-Infrared | Temporal differences between consecutive NIR frames |
| **DOBI** | Dynamic Optical Breast Imaging | Commercial name for the NIR imaging system |
| **BI-RADS** | Breast Imaging Reporting and Data System | Standardized classification scale (0-6) for breast lesions |
| **ROI** | Region of Interest | Specific area in medical image for focused analysis |
| **NIR** | Near-Infrared | Electromagnetic spectrum range (640-900 nm) used for tissue imaging |

### Machine Learning Terms

| Term | Full Name | Description |
|------|-----------|-------------|
| **CNN** | Convolutional Neural Network | Deep learning architecture for spatial feature extraction |
| **LSTM** | Long Short-Term Memory | Recurrent neural network for temporal sequence processing |
| **MIL** | Multiple Instance Learning | Weakly supervised learning treating sequences as bags of instances |
| **SE-block** | Squeeze-and-Excitation Block | Channel-wise attention mechanism for feature recalibration |
| **DenseNet** | Densely Connected Network | CNN architecture with dense skip connections |
| **RAdam** | Rectified Adam | Optimized adaptive learning rate algorithm |

### Performance Metrics

| Term | Formula | Range | Interpretation |
|------|---------|-------|----------------|
| **ACC** | (TP + TN) / Total | [0, 1] | Overall accuracy |
| **SEN** | TP / (TP + FN) | [0, 1] | Sensitivity (recall, true positive rate) |
| **SPE** | TN / (TN + FP) | [0, 1] | Specificity (true negative rate) |
| **AUC** | Area under ROC curve | [0, 1] | Discrimination ability |
| **MCC** | (TP×TN - FP×FN) / √(...) | [-1, 1] | Matthews Correlation Coefficient |
| **PPV** | TP / (TP + FP) | [0, 1] | Positive Predictive Value (precision) |
| **NPV** | TN / (TN + FN) | [0, 1] | Negative Predictive Value |

### Classification Terms

| Term | Description | Clinical Meaning |
|------|-------------|------------------|
| **TP** (True Positive) | Correctly identified malignant | Cancer correctly detected |
| **TN** (True Negative) | Correctly identified benign | Healthy correctly identified |
| **FP** (False Positive) | Incorrectly identified as malignant | False alarm (unnecessary biopsy) |
| **FN** (False Negative) | Incorrectly identified as benign | Missed cancer (dangerous) |

### Data Processing Terms

| Term | Description |
|------|-------------|
| **Interpolation** | Resampling technique to change image resolution |
| **Denoising** | Noise reduction in image data |
| **Normalization** | Scaling data to standard range |
| **Enhancement** | Improving image contrast or features |
| **Augmentation** | Creating variations of data for training |

## Data Formats

### Input Data Dimensions

#### 3-LED Mode (Small Breast Sizes A-C)

```
Raw .mat file:
├── Shape: (102, 128, 25, 3)
├── Dimension 1: 102 (image height)
├── Dimension 2: 128 (image width)
├── Dimension 3: 25 (LED scanning cycles)
└── Dimension 4: 3 (number of LEDs)
```

#### 5-LED Mode (Large Breast Sizes D+)

```
Raw .mat file:
├── Shape: (102, 128, 15, 5)
├── Dimension 1: 102 (image height)
├── Dimension 2: 128 (image width)
├── Dimension 3: 15 (LED scanning cycles)
└── Dimension 4: 5 (number of LEDs)
```

### Processed Data Dimensions

```
.npy files after preprocessing:
├── train_data.npy: (N, 25, C, H, W)
│   ├── N: Number of samples
│   ├── 25: Temporal frames (scanning cycles)
│   ├── C: Channels (1 or 3)
│   ├── H: Height (102)
│   └── W: Width (128)
├── train_labels.npy: (N,)
│   └── Values: 0 (benign) or 1 (malignant)
└── train_names.npy: (N,)
    └── Patient identifiers
```

### Model Tensor Dimensions

```
Model input/output:
├── Input: (B, 25, C, H, W)
│   ├── B: Batch size
│   ├── 25: Temporal frames
│   ├── C: Channels (1 or 3)
│   ├── H: Height (102)
│   └── W: Width (128)
├── DenseNet output: (B, 25, 1024)
├── LSTM output: (B, 25, 64)
├── Final hidden: (B, 64)
└── Output logits: (B, 1)
```

## Classification Criteria

### BI-RADS Score Mapping

```
BI-RADS Score → Binary Label
├── 0-4a: Benign (Label = 0)
│   ├── 0: Incomplete - Need additional imaging
│   ├── 1: Negative - No findings
│   ├── 2: Benign - Non-cancerous findings
│   ├── 3: Probably benign - <2% cancer risk
│   └── 4a: Low suspicion - 2-10% cancer risk
└── >4a: Malignant (Label = 1)
    ├── 4b: Moderate suspicion - 10-50% cancer risk
    ├── 4c: High suspicion - 50-95% cancer risk
    ├── 5: Highly suggestive - >95% cancer risk
    └── 6: Known biopsy-proven malignancy
```

## File Naming Conventions

### Dataset Names

```
Pattern: {reconstruction}_{layers}_npy_{methods}
Example: dDOT_9layers_DOBI_Recon_npy_interpolation_4_denoising_0_normalization_0_enhancement_0

Components:
├── dDOT / fDOT: Reconstruction type
├── 9layers / 1layer: Number of reconstruction layers
├── DOBI_Recon: Reconstruction method
├── interpolation_X: Interpolation method index
├── denoising_X: Denoising method index
├── normalization_X: Normalization method index
└── enhancement_X: Enhancement method index
```

### Model Checkpoint Names

```
Pattern: {module}_{class}_c1to3{bool}_att{bool}/cH{ch}_{extra}/
Example: model_CNNLSTMNet_c1to3False_attFalse/cH3_None/

Components:
├── model / lstm_mil / lambda_net: Model module
├── CNNLSTMNet: Model class
├── c1to3True/False: Channel conversion enabled
├── attTrue/False: Attention enabled
├── cH3: CNN input channels
└── extra parameters: Model-specific config
```

### Epoch Files

```
Pattern: epoch_{number}.pth
Example: epoch_078.pth

├── Number: Zero-padded 3-digit epoch number
└── Extension: .pth (PyTorch checkpoint)
```

## Method Indices Reference

### Interpolation Methods (0-5)

| Index | Method | Description |
|-------|--------|-------------|
| 0 | Nearest Neighbor | Fast, no smoothing |
| 1 | Bilinear | Linear interpolation |
| 2 | Quadratic | 2nd-order polynomial |
| 3 | Cubic | 3rd-order polynomial (smooth) |
| 4 | Quartic | 4th-order polynomial |
| 5 | Quintic | 5th-order polynomial |

### Denoising Methods (0-11)

| Index | Method | Best For |
|-------|--------|----------|
| 0 | None | No noise issues |
| 1 | Non-Local Means | Textured images |
| 2 | Gaussian | Gaussian noise |
| 3 | Median | Salt-and-pepper noise |
| 4 | Wiener | Known noise PSF |
| 5 | Richardson-Lucy | Deconvolution |
| 6 | Total Variation | Piecewise smooth |
| 7 | Bilateral | Edge-preserving |
| 8 | Wavelet | Multi-scale noise |
| 9 | Inpainting | Missing regions |
| 10 | Isotropic Diffusion | Uniform smoothing |
| 11 | Anisotropic Diffusion | Edge-preserving |

### Normalization Methods (0-2)

| Index | Method | Formula | Range |
|-------|--------|---------|-------|
| 0 | None | x | Original |
| 1 | Min-Max | (x - min) / (max - min) | [0, 1] |
| 2 | Z-score | (x - μ) / σ | Mean=0, Std=1 |

### Enhancement Methods (0-4)

| Index | Method | Purpose |
|-------|--------|---------|
| 0 | None | No enhancement |
| 1 | CLAHE | Adaptive contrast |
| 2 | Histogram Equalization | Global contrast |
| 3 | Horizontal Flip | Augmentation |
| 4 | Vertical Flip | Augmentation |

## Configuration Examples

### Minimal Configuration

```bash
# Simplest training setup
python train.py \
    --data_path data/npy/my_dataset \
    --num_epochs 100
```

### Standard Configuration

```bash
# Recommended baseline
python train.py \
    --data_path data/npy/phase1_i4_d0_n0_e0 \
    --model_module model \
    --model_class CNNLSTMNet \
    --Recon_num_layers 9 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --num_epochs 100 \
    --hidden_size 64 \
    --num_lstm_layers 7 \
    --metrics_file results.xlsx
```

### Advanced Configuration

```bash
# With attention and custom architecture
python train.py \
    --data_path data/npy/phase1_i4_d2_n2_e1 \
    --model_module lstm_mil \
    --model_class CNNLSTMNet \
    --use_attention \
    --Recon_num_layers 9 \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --num_epochs 200 \
    --hidden_size 128 \
    --num_lstm_layers 9 \
    --weight_decay 1e-5 \
    --metrics_file advanced_results.xlsx
```

## Performance Benchmarks

### Expected Training Metrics (Phase 1 Data)

| Model | Train MCC | Val MCC | Test MCC | Epochs to Converge |
|-------|-----------|---------|----------|-------------------|
| Base CNN-LSTM | 0.75-0.85 | 0.65-0.75 | 0.60-0.70 | 50-80 |
| + Attention | 0.78-0.88 | 0.68-0.78 | 0.63-0.73 | 60-90 |
| MIL Instance | 0.76-0.86 | 0.66-0.76 | 0.61-0.71 | 70-100 |
| Lambda Network | 0.77-0.87 | 0.67-0.77 | 0.62-0.72 | 55-85 |

### Clinical Performance Targets

| Metric | Minimum Target | Desired Target |
|--------|---------------|----------------|
| Sensitivity | ≥ 0.85 | ≥ 0.90 |
| Specificity | ≥ 0.65 | ≥ 0.75 |
| AUC | ≥ 0.85 | ≥ 0.90 |
| MCC | ≥ 0.60 | ≥ 0.70 |

## Design Patterns Used

### Creational Patterns

- **Factory Pattern**: Dynamic model instantiation (`create_model()`)
- **Builder Pattern**: Flexible model configuration composition

### Structural Patterns

- **Adapter Pattern**: Channel conversion (1→3)
- **Decorator Pattern**: Attention mechanisms wrapping base models

### Behavioral Patterns

- **Strategy Pattern**: Pluggable preprocessing algorithms
- **Template Method**: Consistent training and evaluation workflows
- **Observer Pattern**: Metrics tracking and logging

## References

### Scientific Publications

1. **DOBI Technology Overview**
   - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC3467859/
   - Topic: Dynamic optical breast imaging principles

2. **Recent Applications**
   - URL: https://www.sciencedirect.com/science/article/pii/S1687850725000408
   - Topic: Current research in diffuse optical imaging

### Key Technologies

**PyTorch**
- Deep learning framework
- URL: https://pytorch.org/

**DenseNet**
- Huang et al., "Densely Connected Convolutional Networks"
- CVPR 2017

**LSTM**
- Hochreiter & Schmidhuber, "Long Short-Term Memory"
- Neural Computation, 1997

**Multiple Instance Learning**
- Dietterich et al., "Solving the Multiple Instance Problem"
- Journal of Artificial Intelligence Research, 1997

### Libraries Used

**Core ML:**
- torch==2.5.0
- torchvision==0.20.0
- scikit-learn==1.5.2

**Scientific Computing:**
- numpy==1.26.4
- scipy==1.14.1
- pandas==2.2.3

**Image Processing:**
- scikit-image==0.24.0
- opencv-python==4.10.0.84
- PyWavelets==1.8.0
- SimpleITK==2.4.0

**Specialized:**
- einops==0.8.1
- lambda-networks==0.4.0
- fighting_attention==1.0.0

## Acronyms Quick Reference

| Acronym | Full Form |
|---------|-----------|
| ACC | Accuracy |
| AUC | Area Under Curve |
| BI-RADS | Breast Imaging Reporting and Data System |
| CNN | Convolutional Neural Network |
| DOBI | Dynamic Optical Breast Imaging |
| dNIR | Difference Near-Infrared |
| FC | Fully Connected |
| fNIR | Full Near-Infrared |
| FN | False Negative |
| FP | False Positive |
| GPU | Graphics Processing Unit |
| LSTM | Long Short-Term Memory |
| MCC | Matthews Correlation Coefficient |
| MIL | Multiple Instance Learning |
| NIR | Near-Infrared |
| NPV | Negative Predictive Value |
| PPV | Positive Predictive Value |
| ReLU | Rectified Linear Unit |
| ROC | Receiver Operating Characteristic |
| ROI | Region of Interest |
| SE | Squeeze-and-Excitation |
| SEN | Sensitivity |
| SPE | Specificity |
| TN | True Negative |
| TP | True Positive |

## Version Information

**Project Version:** 1.0
**Initial Release:** 2025-04-01
**Documentation Refined:** 2025-10-30
**Status:** Active Development

## Support and Contact

**Project Type:** Research Code
**License:** Open Source (following company bankruptcy)
**Purpose:** Advance research in Diffuse Optical Imaging

**For Issues:**
1. Review this documentation
2. Check source code comments
3. Verify data formats and preprocessing
4. Review example configurations
5. Test with minimal configuration first

## Appendix: Common Error Messages

### Data Loading Errors

```
KeyError: 'img4D'
→ Solution: Check variable_name parameter matches .mat file content

ValueError: shapes not aligned
→ Solution: Verify data preprocessing completed correctly

FileNotFoundError: [Errno 2] No such file
→ Solution: Check data_path and file naming conventions
```

### Training Errors

```
RuntimeError: CUDA out of memory
→ Solution: Reduce batch_size or use CPU

RuntimeError: Expected 5D tensor
→ Solution: Verify input shape is (B, T, C, H, W)

ValueError: Target size mismatch
→ Solution: Check label dimensions match predictions
```

### Model Loading Errors

```
KeyError: Unexpected key in state_dict
→ Solution: Verify model architecture matches checkpoint

RuntimeError: Error loading state_dict
→ Solution: Check model parameters match saved checkpoint
```

## End of Documentation

This completes the comprehensive documentation for the fNIR Base Model project. All information focuses on Phase 1 implementation, excluding Phase 2 and the 88 examples as requested.
