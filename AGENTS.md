# AGENTS.md

> Context and instructions for AI coding agents working on this project

## Project Overview

**Zero-DCE (Zero-Reference Deep Curve Estimation)** is a deep learning approach for low-light image enhancement that doesn't require paired training data. This is a re-implementation in Keras 3 for a graduate-level Digital Image Processing course final project.

### Key Features
- **Zero-reference learning**: No need for paired low-light/normal-light images
- **Lightweight architecture**: DCE-Net has only 7 convolutional layers
- **Curve-based enhancement**: Learns pixel-wise tonal curves for dynamic range adjustment
- **Unsupervised training**: Uses 4 carefully designed loss functions

## Project Status

**Current State:** In active refactoring from monolithic script to modular architecture

- ✅ Original `zero_dce.py` contains complete working implementation (~537 lines)
- 🔄 Refactoring into modular components (in progress)
- 📁 Empty module files created: `dataset.py`, `loss.py`, `model.py`, `train.py`, `compare.py`

**See:** `REFACTOR_PLAN.md` for detailed refactoring roadmap

## Architecture

### Zero-DCE Framework
```
Input Image → DCE-Net → Curve Parameters → Iterative Enhancement → Output Image
                ↓
        Unsupervised Losses
        (guide training)
```

### DCE-Net Architecture
7-layer CNN with symmetrical skip connections:
- Input: Low-light RGB image (H×W×3)
- Conv layers 1-4: 32 filters, 3×3, ReLU
- Skip connections: Concatenate layers 4↔3, 5↔2, 6↔1
- Output: 24 parameter maps (8 iterations × 3 RGB channels), Tanh activation

### Loss Functions (Unsupervised)
1. **Color Constancy Loss** (weight: 5): Corrects color deviations
2. **Exposure Loss** (weight: 10): Prevents under/over-exposure
3. **Illumination Smoothness Loss** (weight: 200): Preserves monotonicity
4. **Spatial Consistency Loss** (weight: 1): Maintains spatial coherence

### Enhancement Process
Iteratively applies 8 curve adjustments:
```
x_{n+1} = x_n + r_n * (x_n^2 - x_n)
```
where `r_n` are learned curve parameters

## Module Structure

### Current Refactoring Target

```
redo-zero-dce-keras/
├── dataset.py          # Dataset loading and TensorFlow data pipelines
├── loss.py            # Four unsupervised loss functions
├── model.py           # DCE-Net architecture + ZeroDCE training model
├── train.py           # Training script with CLI arguments
├── compare.py         # Inference and comparison with classical methods
├── zero_dce.py        # Original monolithic implementation (reference)
├── lol_dataset/       # LOL Dataset (485 train + 15 test images)
│   ├── our485/low/    # Training low-light images
│   ├── our485/high/   # Training normal-light images (not used in training)
│   └── eval15/low/    # Test low-light images
├── weights/           # Trained model weights (to be created)
├── AGENTS.md          # This file
├── REFACTOR_PLAN.md   # Detailed refactoring guide
└── README.md          # Project documentation
```

### Module Responsibilities

#### `dataset.py`
- Load and preprocess images from LOL Dataset
- Create TensorFlow datasets for training/validation/testing
- Configurable paths, batch size, image size
- **Key functions:** `load_data()`, `data_generator()`, `get_dataset()`

#### `loss.py`
- All loss function implementations
- Pure functions (stateless) except `SpatialConsistencyLoss` class
- **Exports:** `color_constancy_loss()`, `exposure_loss()`, `illumination_smoothness_loss()`, `SpatialConsistencyLoss`

#### `model.py`
- DCE-Net architecture builder
- Enhancement function (applies curves)
- ZeroDCE training wrapper model
- **Key components:** `build_dce_net()`, `get_enhanced_image()`, `ZeroDCE` class

#### `train.py`
- CLI-based training script
- Uses argparse for configuration
- Integrates dataset, model, and losses
- Saves weights and plots training curves
- **Usage:** `python train.py --epochs 100 --batch-size 16`

#### `compare.py`
- Inference on single images
- Comparison with classical methods (AutoContrast, Histogram Eq, CLAHE, Gamma Correction)
- CLI for easy usage
- **Usage:** `python compare.py -i input.jpg -w weights.h5 -o output.jpg`

## Development Guidelines

### Code Style
- Follow PEP 8
- Use type hints where helpful
- Docstrings for all public functions/classes (Google style)
- Keep functions focused (single responsibility)

### Key Principles
1. **Modularity**: Each file has a single, clear purpose
2. **Reusability**: Functions should be usable independently
3. **Configurability**: Use arguments/configs instead of hardcoded values
4. **Maintainability**: Clear names, comments for complex logic
5. **Academic clarity**: Code should be educational (this is a course project)

### Import Structure
```python
# Standard library
import os
from typing import Tuple, Optional

# Third-party
import numpy as np
import tensorflow as tf
import keras
from keras import layers

# Local modules
from dataset import get_dataset
from loss import color_constancy_loss, SpatialConsistencyLoss
from model import build_dce_net, ZeroDCE
```

### Critical Implementation Details

#### Image Preprocessing
- Input images: Resize to 256×256 (configurable)
- Normalization: Divide by 255.0 to get [0, 1] range
- No data augmentation (as per original paper)

#### Training Configuration
- Default learning rate: 1e-4 (Adam optimizer)
- Batch size: 16 (adjust based on GPU memory)
- Epochs: 100 (original paper)
- Train/val split: 400/85 images from LOL Dataset

#### Loss Weights (Critical - Don't Change Without Testing)
```python
loss_illumination = 200 * illumination_smoothness_loss(output)
loss_spatial_constancy = 1 * spatial_constancy_loss(enhanced, original)
loss_color_constancy = 5 * color_constancy_loss(enhanced)
loss_exposure = 10 * exposure_loss(enhanced)
```

#### Model Weights
- Only save/load `dce_model` weights (not the wrapper)
- Use `.h5` format for compatibility
- Default path: `./weights/zero_dce_weights.h5`

## Common Tasks

### Running Training
```bash
# Basic training
python train.py

# Custom configuration
python train.py --epochs 100 --batch-size 16 --learning-rate 1e-4 \
                --dataset-path ./lol_dataset --save-path ./weights/model.h5
```

### Running Inference
```bash
# Enhance single image and compare methods
python compare.py -i ./test_image.jpg -w ./weights/zero_dce_weights.h5 -o ./enhanced.jpg

# Compare specific methods
python compare.py -i input.jpg -w weights.h5 --methods zero-dce autocontrast clahe
```

### Testing Modules
```bash
# Test dataset loading
python -c "from dataset import get_dataset; train, val, test = get_dataset(); print(train)"

# Test model building
python -c "from model import build_dce_net; model = build_dce_net(); model.summary()"

# Test loss functions
python -c "from loss import color_constancy_loss; import tensorflow as tf; x = tf.random.normal([1,256,256,3]); print(color_constancy_loss(x))"
```

## Refactoring Workflow

**Each refactoring step should:**
1. Be implemented completely before moving to the next
2. Be tested to ensure functionality
3. Be committed to git manually by the student
4. Follow the order in `REFACTOR_PLAN.md`

**Testing after each step:**
- Import the module successfully
- Run basic functionality tests
- Ensure no breaking changes to existing code

## Dependencies

### Required Packages
```toml
[project]
dependencies = [
    "tensorflow>=2.15.0",  # or tensorflow-gpu
    "keras>=3.0.0",
    "numpy>=1.24.0",
    "pillow>=10.0.0",
    "matplotlib>=3.7.0",
    "opencv-python>=4.8.0",  # for comparison methods
]
```

### Environment Setup
```bash
# Using uv (project uses uv)
uv sync

# Or using pip
pip install -r requirements.txt
```

## Dataset Information

### LOL Dataset Structure
- **Training set:** 485 image pairs in `our485/`
  - Low-light images: `our485/low/*.png`
  - Well-exposed images: `our485/high/*.png` (not used during training!)
- **Test set:** 15 image pairs in `eval15/`
  - Low-light images: `eval15/low/*.png`

### Download Instructions
```bash
wget https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip
unzip -q lol_dataset.zip && rm lol_dataset.zip
```

## Important Notes for AI Agents

### When Refactoring
- **Preserve exact logic**: Don't "improve" algorithms unless explicitly asked
- **Keep loss weights**: The specific weights (200, 10, 5, 1) are tuned
- **Maintain compatibility**: Ensure model weights from original code can be loaded
- **Test incrementally**: Each module should be tested before moving to the next

### When Debugging
- Check image shape: Should be (batch, height, width, 3)
- Verify value range: Images should be [0, 1] after preprocessing
- Loss values: Should decrease over training (total loss typically starts ~0.5-1.0)
- Memory issues: Reduce batch size if OOM errors occur

### When Adding Features
- Keep backward compatibility with existing trained models
- Document new hyperparameters and their default values
- Update this file and REFACTOR_PLAN.md accordingly
- Add CLI arguments for configurability

## References

### Paper
- **Title:** Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement
- **Authors:** Chunle Guo, Chongyi Li, et al.
- **Link:** https://arxiv.org/abs/2001.06826
- **Published:** CVPR 2020

### Original Implementation
- Keras example: https://keras.io/examples/vision/zero_dce/
- This project is based on the Keras 3 port by Soumik Rakshit

### Related Documentation
- `REFACTOR_PLAN.md`: Step-by-step refactoring instructions
- `README.md`: Project overview and usage guide

## Git Workflow

**Manual commits by student after each step:**
```bash
# After completing a refactoring step
git add <modified_files>
git commit -m "Refactor: Implement <module_name> - <description>"
```

**Agents should NOT:**
- Run git commands automatically
- Create commits without user approval
- Push to remote repositories

---

**Last Updated:** 2025-10-25  
**Maintained By:** Graduate students in Digital Image Processing course  
**For Questions:** Refer to `REFACTOR_PLAN.md` for detailed implementation steps
