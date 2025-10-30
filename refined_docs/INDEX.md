# 📚 fNIR Base Model - Documentation Quick Reference

## 🎯 Start Here

**New to this project?** → Start with [README.md](README.md)  
**Need quick setup?** → Jump to [Quick Start](#quick-start-3-steps)  
**Having issues?** → Check [Troubleshooting](#troubleshooting-quick-links)

---

## 📑 Documentation Index

### 📘 Getting Started
| File | Topics | When to Read |
|------|--------|-------------|
| **[README.md](README.md)** | Complete overview, index, quick start | First read |
| **[01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md)** | Background, imaging procedure, features | Understanding context |
| **[08_USAGE_EXAMPLES.md](08_USAGE_EXAMPLES.md)** | 6 complete workflows, recipes | First implementation |

### 🔧 Technical Implementation
| File | Topics | When to Read |
|------|--------|-------------|
| **[03_DATA_PIPELINE.md](03_DATA_PIPELINE.md)** | Data formats (**corrected**), preprocessing | Before data prep |
| **[04_CONFIGURATION.md](04_CONFIGURATION.md)** | All parameters, settings | Before training |
| **[05_MODEL_ARCHITECTURE.md](05_MODEL_ARCHITECTURE.md)** | Model options, selection guide | Choosing model |
| **[06_TRAINING_PIPELINE.md](06_TRAINING_PIPELINE.md)** | Training process, best practices | During training |
| **[07_INFERENCE_EVALUATION.md](07_INFERENCE_EVALUATION.md)** | Testing, threshold optimization | After training |
| **[10_TALOS_OPTIMIZATION.md](10_TALOS_OPTIMIZATION.md)** | Automated hyperparameter tuning | Optimizing performance |
| **[TALOS_QUICKSTART.md](TALOS_QUICKSTART.md)** | Quick start for Talos optimization | First-time Talos use |

### 📖 Reference Materials
| File | Topics | When to Read |
|------|--------|-------------|
| **[02_SYSTEM_ARCHITECTURE.md](02_SYSTEM_ARCHITECTURE.md)** | Architecture overview, data flow | Understanding system |
| **[09_GLOSSARY_REFERENCES.md](09_GLOSSARY_REFERENCES.md)** | Terminology, method indices | As reference |
| **[TALOS_README.md](TALOS_README.md)** | Complete Talos feature guide | Hyperparameter optimization |
| **[00_COMPLETION_SUMMARY.md](00_COMPLETION_SUMMARY.md)** | What was done, corrections | Understanding docs |

---

## 🚀 Quick Start (3 Steps)

### Step 1: Prepare Data
```bash
python dataset/data_npy.py \
    --excel_path data/excel/phase1exp.xlsx \
    --cache_folder data/npy/phase1_dataset \
    --interpolation_method 4
```
📖 Details: [03_DATA_PIPELINE.md](03_DATA_PIPELINE.md#running-data-preprocessing)

### Step 2: Train Model
```bash
python train.py \
    --data_path data/npy/phase1_dataset \
    --num_epochs 100 \
    --batch_size 16
```
📖 Details: [06_TRAINING_PIPELINE.md](06_TRAINING_PIPELINE.md#training-best-practices)

### Step 3: Evaluate
```bash
# Edit pro.py with your model path, then:
python pro.py
python cutoff.py
```
📖 Details: [07_INFERENCE_EVALUATION.md](07_INFERENCE_EVALUATION.md#batch-prediction-workflow)

---

## 🔍 Find Information Fast

### By Task

| What You Want to Do | Go To |
|---------------------|-------|
| **Understand the project** | [01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md) |
| **Check data format** | [03_DATA_PIPELINE.md](03_DATA_PIPELINE.md#input-data-format-requirements) |
| **Set up preprocessing** | [03_DATA_PIPELINE.md](03_DATA_PIPELINE.md#preprocessing-pipeline) |
| **Choose a model** | [05_MODEL_ARCHITECTURE.md](05_MODEL_ARCHITECTURE.md#model-selection-guide) |
| **Configure training** | [04_CONFIGURATION.md](04_CONFIGURATION.md#example-configurations) |
| **Optimize hyperparameters** | [10_TALOS_OPTIMIZATION.md](10_TALOS_OPTIMIZATION.md) or [TALOS_QUICKSTART.md](TALOS_QUICKSTART.md) |
| **Start training** | [06_TRAINING_PIPELINE.md](06_TRAINING_PIPELINE.md#training-loop-structure) |
| **Evaluate results** | [07_INFERENCE_EVALUATION.md](07_INFERENCE_EVALUATION.md#inference-script-propl) |
| **Optimize threshold** | [07_INFERENCE_EVALUATION.md](07_INFERENCE_EVALUATION.md#threshold-optimization-cutoffpl) |
| **Fix an error** | [08_USAGE_EXAMPLES.md](08_USAGE_EXAMPLES.md#troubleshooting-common-issues) |
| **Look up a term** | [09_GLOSSARY_REFERENCES.md](09_GLOSSARY_REFERENCES.md#terminology) |

### By Question

| Question | Answer Location |
|----------|----------------|
| What data format do I need? | [03_DATA_PIPELINE.md](03_DATA_PIPELINE.md#raw-data-format-mat-files) |
| How many LED modes are there? | [01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md#configuration-by-breast-size) |
| What's the correct dimension? | **3-LED: (102,128,25,3)** - See [03_DATA_PIPELINE.md](03_DATA_PIPELINE.md#raw-data-format-mat-files) |
| Which model should I use? | [05_MODEL_ARCHITECTURE.md](05_MODEL_ARCHITECTURE.md#model-selection-guide) |
| What preprocessing methods exist? | [03_DATA_PIPELINE.md](03_DATA_PIPELINE.md#preprocessing-methods-reference) |
| What are good hyperparameters? | [04_CONFIGURATION.md](04_CONFIGURATION.md#example-configurations) |
| How do I add a custom model? | [05_MODEL_ARCHITECTURE.md](05_MODEL_ARCHITECTURE.md#adding-new-model-architectures) |
| What metrics are tracked? | [06_TRAINING_PIPELINE.md](06_TRAINING_PIPELINE.md#evaluation-metrics) |
| How to optimize threshold? | [07_INFERENCE_EVALUATION.md](07_INFERENCE_EVALUATION.md#threshold-optimization-cutoffpl) |
| How to tune hyperparameters? | [10_TALOS_OPTIMIZATION.md](10_TALOS_OPTIMIZATION.md) or [TALOS_QUICKSTART.md](TALOS_QUICKSTART.md) |
| What does MCC mean? | [09_GLOSSARY_REFERENCES.md](09_GLOSSARY_REFERENCES.md#performance-metrics) |

---

## ⚠️ Important Corrections

### ✅ Data Dimensions (CORRECTED)

**OLD (Incorrect):**
- ❌ 23 scanning cycles
- ❌ 22 difference frames
- ❌ Shape (N, 23, C, H, W)

**NEW (Correct):**
- ✅ **25 scanning cycles** (3-LED mode)
- ✅ **24 difference frames** (dNIR)
- ✅ Shape **(N, 25, C, H, W)**

📖 Full details: [03_DATA_PIPELINE.md](03_DATA_PIPELINE.md#raw-data-format-mat-files)

### ✅ Phase 1 Only

**Included:**
- ✅ Phase 1 implementation
- ✅ train.py, pro.py, cutoff.py
- ✅ phase1exp.xlsx data

**Excluded:**
- ❌ Phase 2 (not covered)
- ❌ 88 examples (not covered)
- ❌ pro_88.py (not covered)

📖 Rationale: [README.md](README.md#phase-1-focus-current-documentation)

---

## 🎯 By User Type

### 👶 First-Time Users
1. Read [README.md](README.md)
2. Follow [08_USAGE_EXAMPLES.md](08_USAGE_EXAMPLES.md#workflow-1-complete-training-pipeline)
3. Keep [09_GLOSSARY_REFERENCES.md](09_GLOSSARY_REFERENCES.md) open for reference

### 👨‍💻 Developers
1. Review [02_SYSTEM_ARCHITECTURE.md](02_SYSTEM_ARCHITECTURE.md)
2. Study [05_MODEL_ARCHITECTURE.md](05_MODEL_ARCHITECTURE.md)
3. Reference [04_CONFIGURATION.md](04_CONFIGURATION.md) and [06_TRAINING_PIPELINE.md](06_TRAINING_PIPELINE.md)

### 🔬 Researchers
1. Understand [01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md)
2. Experiment with [08_USAGE_EXAMPLES.md](08_USAGE_EXAMPLES.md#workflow-2-experiment-with-different-preprocessing)
3. Analyze with [07_INFERENCE_EVALUATION.md](07_INFERENCE_EVALUATION.md)

### 🏥 Clinical Users
1. Start with [01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md)
2. Deploy using [08_USAGE_EXAMPLES.md](08_USAGE_EXAMPLES.md#workflow-6-clinical-deployment-simulation)
3. Understand metrics in [09_GLOSSARY_REFERENCES.md](09_GLOSSARY_REFERENCES.md#performance-metrics)

---

## 🛠️ Troubleshooting Quick Links

### Common Issues

| Problem | Solution |
|---------|----------|
| CUDA out of memory | [08_USAGE_EXAMPLES.md](08_USAGE_EXAMPLES.md#issue-cuda-out-of-memory) |
| Import errors | [08_USAGE_EXAMPLES.md](08_USAGE_EXAMPLES.md#issue-import-errors) |
| Data shape mismatch | [08_USAGE_EXAMPLES.md](08_USAGE_EXAMPLES.md#issue-data-shape-mismatch) |
| Low performance | [08_USAGE_EXAMPLES.md](08_USAGE_EXAMPLES.md#issue-low-performance) |
| Training not improving | [06_TRAINING_PIPELINE.md](06_TRAINING_PIPELINE.md#common-training-issues) |
| Evaluation errors | [07_INFERENCE_EVALUATION.md](07_INFERENCE_EVALUATION.md#common-evaluation-issues) |

---

## 📊 Data Format Quick Reference

### 3-LED Mode (Standard)
```python
Raw fNIR:    (102, 128, 25, 3)
Raw dNIR:    (102, 128, 24, 3)
Raw img4D:   (102, 128, 9, 25)
Processed:   (N, 25, C, 102, 128)
```

### 5-LED Mode (Large)
```python
Raw fNIR:    (102, 128, 15, 5)
Raw dNIR:    (102, 128, 14, 5)
Raw img4D:   (102, 128, 9, 15)
Processed:   (N, 15, C, 102, 128)
```

📖 Full specs: [03_DATA_PIPELINE.md](03_DATA_PIPELINE.md#input-data-format-requirements)

---

## 🧭 Configuration Quick Reference

### Minimal Training
```bash
python train.py --data_path data/npy/my_dataset --num_epochs 100
```

### Standard Training
```bash
python train.py \
    --data_path data/npy/phase1_i4_d0_n0_e0 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --num_epochs 100
```

### Advanced Training
```bash
python train.py \
    --model_module lstm_mil \
    --use_attention \
    --batch_size 32 \
    --num_epochs 200
```

📖 More examples: [04_CONFIGURATION.md](04_CONFIGURATION.md#configuration-examples)

---

## 📈 Expected Performance

| Model | Val MCC | Test MCC | Epochs |
|-------|---------|----------|--------|
| Base CNN-LSTM | 0.65-0.75 | 0.60-0.70 | 50-80 |
| + Attention | 0.68-0.78 | 0.63-0.73 | 60-90 |
| MIL Instance | 0.66-0.76 | 0.61-0.71 | 70-100 |

📖 Details: [README.md](README.md#performance-expectations)

---

## 🔗 External Resources

- **DOBI Technology:** https://pmc.ncbi.nlm.nih.gov/articles/PMC3467859/
- **Recent Research:** https://www.sciencedirect.com/science/article/pii/S1687850725000408
- **PyTorch:** https://pytorch.org/

📖 Full references: [09_GLOSSARY_REFERENCES.md](09_GLOSSARY_REFERENCES.md#references)

---

## 📝 Documentation Statistics

- **Total Files:** 14 documents (11 original + 3 Talos)
- **Total Content:** ~5,000 lines
- **Code Examples:** 120+
- **Tables:** 60+
- **Complete Workflows:** 6
- **Troubleshooting Guides:** 15+
- **New Features:** Talos hyperparameter optimization

---

## ✅ Quality Checklist

- [x] All data formats corrected (25 cycles, not 23)
- [x] Phase 2 and 88 examples excluded
- [x] Complete Phase 1 implementation coverage
- [x] Practical examples for all features
- [x] Comprehensive troubleshooting
- [x] Cross-referenced navigation
- [x] Production-ready documentation

---

## 🎓 Learning Path

### Beginner Path (2-3 hours)
1. [README.md](README.md) - 15 min
2. [01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md) - 15 min
3. [08_USAGE_EXAMPLES.md](08_USAGE_EXAMPLES.md) (Workflow 1) - 60 min
4. [09_GLOSSARY_REFERENCES.md](09_GLOSSARY_REFERENCES.md) - Reference

### Intermediate Path (4-6 hours)
1. Complete Beginner Path
2. [03_DATA_PIPELINE.md](03_DATA_PIPELINE.md) - 45 min
3. [04_CONFIGURATION.md](04_CONFIGURATION.md) - 30 min
4. [06_TRAINING_PIPELINE.md](06_TRAINING_PIPELINE.md) - 60 min
5. [08_USAGE_EXAMPLES.md](08_USAGE_EXAMPLES.md) (Workflows 2-3) - 90 min

### Advanced Path (8-10 hours)
1. Complete Intermediate Path
2. [02_SYSTEM_ARCHITECTURE.md](02_SYSTEM_ARCHITECTURE.md) - 30 min
3. [05_MODEL_ARCHITECTURE.md](05_MODEL_ARCHITECTURE.md) - 90 min
4. [07_INFERENCE_EVALUATION.md](07_INFERENCE_EVALUATION.md) - 60 min
5. [08_USAGE_EXAMPLES.md](08_USAGE_EXAMPLES.md) (All workflows) - 120 min

---

## 📞 Getting Help

**In order:**
1. Check this index for relevant document
2. Read the specific section
3. Try the troubleshooting guides
4. Review code comments in source files
5. Test with minimal configuration

---

**Last Updated:** 2025-10-30  
**Status:** Complete and Production-Ready  
**Focus:** Phase 1 Implementation Only  
**Version:** 1.1 (Refined with corrections)

---

<div align="center">

### 🚀 Ready to start? Open [README.md](README.md)!

</div>
