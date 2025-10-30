# System Architecture

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  Excel (splits) → .mat files → Preprocessing → .npy files       │
│  • Interpolation  • Denoising  • Normalization  • Enhancement   │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                     DATASET LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  CustomDataset (PyTorch)                                         │
│  • Channel conversion (1→3)  • Label loading  • Name tracking   │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL LAYER                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐       │
│  │   DenseNet   │→ │    LSTM      │→ │  Classifier     │       │
│  │  (Backbone)  │  │  (Temporal)  │  │  (FC+Dropout)   │       │
│  └──────────────┘  └──────────────┘  └─────────────────┘       │
│                                                                  │
│  Variants: MIL, Lambda, Wavelet, Attention-enhanced              │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   TRAINING & EVALUATION                          │
├─────────────────────────────────────────────────────────────────┤
│  • BCEWithLogitsLoss (pos_weight=2.2167)                        │
│  • RAdam Optimizer + CosineAnnealingLR                          │
│  • Metrics: ACC, SEN, SPE, AUC, MCC                             │
│  • Early stopping (MCC > 0.9)                                   │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                                  │
├─────────────────────────────────────────────────────────────────┤
│  • Model checkpoints (.pth)  • Metrics Excel  • Predictions     │
└─────────────────────────────────────────────────────────────────┘
```
![Alt text](https://github.com/AlexJerryPolyU/DOBI-breast-cancer-classification-project/blob/main/refined_docs/DOBI%20image%20classification%20algorithm%20design%20in%20one%20page.png)

## Core Components

### 1. Data Layer
- Manages data splitting through Excel files
- Converts raw .mat files to processed .npy format
- Applies preprocessing pipeline (interpolation, denoising, normalization, enhancement)

### 2. Dataset Layer
- PyTorch CustomDataset implementation
- Handles channel conversion for model compatibility
- Manages label loading and filename tracking

### 3. Model Layer
- DenseNet backbone for feature extraction
- LSTM layers for temporal sequence processing
- Classifier head with dropout regularization
- Multiple architecture variants available

### 4. Training & Evaluation Layer
- Binary cross-entropy loss with class weighting
- RAdam optimizer with cosine annealing learning rate schedule
- Comprehensive metrics tracking
- Early stopping based on MCC threshold

### 5. Output Layer
- Model checkpoint management
- Excel-based metrics reporting
- Prediction result storage

