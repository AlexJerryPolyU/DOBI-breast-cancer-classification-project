# DOBI (Dynamic optical breast imaging) image Classification Model - Project Overview

## Project Description

This deep learning framework facilitates fNIR (functional Near-Infrared Spectroscopy) medical imaging analysis using Diffuse Optical Imaging via hybrid CNN-LSTM architectures. The system processes 4D medical imaging data in .mat format to enable binary classification tasks—specifically differentiating benign from malignant breast cancer diagnostics, where cases with a BI-RADS score greater than 4a are classified as malignant and all others as benign.

Following the bankruptcy of Zhejiang Dolby Medical Technology Co., Ltd. (DOBI Medical International, Inc.) in China, this codebase has been open-sourced to advance research and applications in Diffuse Optical Imaging (DOI).

**Further Reading:**
- https://pmc.ncbi.nlm.nih.gov/articles/PMC3467859/
- https://www.sciencedirect.com/science/article/pii/S1687850725000408

## Basic Imaging Procedure

The diffuse infrared imaging procedure for breast tissue employs LED lights that emit near-infrared (NIR) light at a wavelength of 640 nm, scanned sequentially from left to right across the target area.

### Configuration by Breast Size:
- **3-LED mode:** For smaller cup sizes (A–C)
- **5-LED mode:** For larger sizes (D and above)

### Image Acquisition:
Images are acquired sequentially with a 500 ms interval between captures to ensure comprehensive tissue coverage.

### Data Format:

The resulting raw data adopts a 4D format:

**For 3-LED mode:**
- Shape: **(102, 128, 25, 3)**
- Dimensions: (height, width, scanning_cycles, num_LEDs)

**For 5-LED mode:**
- Shape: **(102, 128, 15, 5)**
- Dimensions: (height, width, scanning_cycles, num_LEDs)

Where:
- **(102, 128)** = Image height and width
- **25 or 15** = Number of LED scanning cycles
- **3 or 5** = Number of LEDs in respective mode

**Note:** For data privacy reasons, this repository does not include sample datasets for testing. Users are encouraged to source their own compliant data for model evaluation and fine-tuning.

## Key Features

- **Phase 1 data support** for clinical data processing
- Flexible preprocessing pipeline with 11 denoising methods and 6 interpolation methods
- Multiple model architectures (standard CNN-LSTM, MIL variants, Lambda networks, wavelet-based)
- Automated hyperparameter search capabilities
- Support for both 1-layer and 9-layer reconstruction data
- Threshold optimization for sensitivity/specificity balance

## Project Status

**Project Type:** Research Code  
**Status:** Active Development  
**Last Updated:** 2025-04-01  
**Documentation Refined:** 2025-10-30

