# Zero-DCE Refactoring Plan

> Detailed step-by-step guide for refactoring `zero_dce.py` into modular components

## Overview

This document provides a detailed roadmap for refactoring the monolithic `zero_dce.py` (537 lines) into 5 modular files. Each step should be completed, tested, and committed before moving to the next.

## Prerequisites

- ✅ `zero_dce.py` exists with working implementation
- ✅ Empty module files created: `dataset.py`, `loss.py`, `model.py`, `train.py`, `compare.py`
- ✅ LOL Dataset downloaded and available at `./lol_dataset/`
- ✅ Development environment set up with required packages

## Progress Tracker

- ✅ **Step 1:** `dataset.py` - Complete (2025-10-25)
- ✅ **Step 2:** `loss.py` - Complete (2025-10-25)
- ✅ **Step 3:** `model.py` - Complete (2025-10-25)
- ✅ **Step 4:** `train.py` - Complete (2025-10-25)
- ✅ **Step 5:** `compare.py` - Complete (2025-10-25)
- ✅ **Step 6:** Documentation updates - Complete (2025-10-25)

## Refactoring Steps

---

## Step 1: Implement `dataset.py`

**Status:** ✅ Complete (2025-10-25)
**Estimated Time:** 20-30 minutes
**Dependencies:** None (foundation module)

### Components to Extract from `zero_dce.py`

**Lines to move:**
- Lines 76-78: Constants (`IMAGE_SIZE`, `BATCH_SIZE`, `MAX_TRAIN_IMAGES`)
- Lines 81-86: `load_data()` function
- Lines 89-93: `data_generator()` function
- Lines 96-102: Dataset path definitions and creation

### Implementation Requirements

```python
# dataset.py structure

import tensorflow as tf
from glob import glob
from typing import Tuple, Optional

# Configuration constants (make these function parameters)
DEFAULT_IMAGE_SIZE = 256
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_TRAIN_IMAGES = 400

def load_data(image_path: str, image_size: int = DEFAULT_IMAGE_SIZE) -> tf.Tensor:
    """Load and preprocess a single image.

    Args:
        image_path: Path to the image file
        image_size: Target size for resizing (height and width)

    Returns:
        Preprocessed image tensor normalized to [0, 1]
    """
    # Implementation from lines 81-86
    pass

def data_generator(
    low_light_images: list,
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE
) -> tf.data.Dataset:
    """Create TensorFlow dataset from image paths.

    Args:
        low_light_images: List of image file paths
        batch_size: Batch size for training
        image_size: Target image size

    Returns:
        TensorFlow Dataset object
    """
    # Implementation from lines 89-93
    pass

def get_dataset(
    dataset_path: str = "./lol_dataset",
    max_train_images: int = DEFAULT_MAX_TRAIN_IMAGES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE
) -> Tuple[tf.data.Dataset, tf.data.Dataset, list]:
    """Load and prepare train, validation, and test datasets.

    Args:
        dataset_path: Root path to LOL dataset
        max_train_images: Maximum number of images for training
        batch_size: Batch size for datasets
        image_size: Target image size

    Returns:
        Tuple of (train_dataset, val_dataset, test_image_paths)
    """
    # Implementation from lines 96-102
    # Add path validation
    # Return structured data
    pass
```

### Testing

```bash
# Test in Python REPL or script
python -c "
from dataset import get_dataset
train_ds, val_ds, test_paths = get_dataset()
print(f'Train dataset: {train_ds}')
print(f'Val dataset: {val_ds}')
print(f'Test images: {len(test_paths)}')
"
```

### Success Criteria
- ✅ Module imports without errors
- ✅ Returns valid TensorFlow datasets
- ✅ Images are normalized to [0, 1]
- ✅ Batch shapes are correct: (batch_size, image_size, image_size, 3)
- ✅ Test images list is not empty

### Implementation Summary

**Completed:** 2025-10-25

**What was implemented:**
- ✅ Extracted `load_data()` function from zero_dce.py (lines 81-86)
- ✅ Extracted `data_generator()` function from zero_dce.py (lines 89-93)
- ✅ Extracted `get_dataset()` function from zero_dce.py (lines 96-102)
- ✅ Added comprehensive docstrings (Google style) for all functions
- ✅ Added type hints for all parameters and return values
- ✅ Added path validation with helpful error messages
- ✅ Added edge case warnings (e.g., no validation images)
- ✅ Created `test_dataset.py` for regression testing

**Test Results:**
```
Train dataset: <_BatchDataset element_spec=TensorSpec(shape=(16, 256, 256, 3), dtype=tf.float32, name=None)>
Val dataset: <_BatchDataset element_spec=TensorSpec(shape=(16, 256, 256, 3), dtype=tf.float32, name=None)>
Test images: 15
Training batch shape: (16, 256, 256, 3)
Training batch value range: [0.000, 1.000]
```

### Git Commit
```bash
git add dataset.py
git commit -m "Refactor: Implement dataset.py - Data loading and preprocessing

- Extract load_data(), data_generator(), get_dataset() from zero_dce.py
- Add configurable parameters for image_size, batch_size
- Add path validation and error handling
- Add type hints and docstrings
"
```

---

## Step 2: Implement `loss.py`

**Status:** ✅ Complete (2025-10-25)
**Estimated Time:** 20-30 minutes
**Dependencies:** None (only TensorFlow/Keras)

### Components to Extract from `zero_dce.py`

**Lines to move:**
- Lines 192-202: `color_constancy_loss()` function
- Lines 214-217: `exposure_loss()` function
- Lines 228-239: `illumination_smoothness_loss()` function
- Lines 250-325: `SpatialConsistencyLoss` class

### Implementation Requirements

```python
# loss.py structure

import tensorflow as tf
import keras

def color_constancy_loss(x: tf.Tensor) -> tf.Tensor:
    """Compute color constancy loss.

    Measures the deviation between average values of RGB channels
    to correct potential color shifts in enhanced images.

    Args:
        x: Enhanced image tensor of shape (batch, height, width, 3)

    Returns:
        Color constancy loss value
    """
    # Implementation from lines 192-202
    pass

def exposure_loss(x: tf.Tensor, mean_val: float = 0.6) -> tf.Tensor:
    """Compute exposure control loss.

    Measures distance between average intensity of local regions
    and the target well-exposedness level.

    Args:
        x: Enhanced image tensor of shape (batch, height, width, 3)
        mean_val: Target exposure level (default: 0.6)

    Returns:
        Exposure control loss value
    """
    # Implementation from lines 214-217
    pass

def illumination_smoothness_loss(x: tf.Tensor) -> tf.Tensor:
    """Compute illumination smoothness loss.

    Preserves monotonicity between neighboring pixels by minimizing
    total variation in curve parameter maps.

    Args:
        x: Curve parameter maps of shape (batch, height, width, channels)

    Returns:
        Illumination smoothness loss value
    """
    # Implementation from lines 228-239
    pass

class SpatialConsistencyLoss(keras.losses.Loss):
    """Spatial consistency loss.

    Encourages spatial coherence by preserving contrast between
    neighboring regions across input and enhanced images.
    """

    def __init__(self, **kwargs):
        # Implementation from lines 251-265
        pass

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Implementation from lines 267-325
        pass
```

### Testing

```bash
# Test loss functions
python -c "
import tensorflow as tf
from loss import (
    color_constancy_loss,
    exposure_loss,
    illumination_smoothness_loss,
    SpatialConsistencyLoss
)

# Create dummy tensors
image = tf.random.uniform([2, 256, 256, 3], 0, 1)
params = tf.random.normal([2, 256, 256, 24])

# Test each loss
print('Color constancy loss:', color_constancy_loss(image))
print('Exposure loss:', exposure_loss(image))
print('Illumination smoothness loss:', illumination_smoothness_loss(params))

spatial_loss = SpatialConsistencyLoss()
print('Spatial consistency loss:', tf.reduce_mean(spatial_loss(image, image)))
"
```

### Success Criteria
- ✅ All functions return scalar loss values
- ✅ Loss values are non-negative
- ✅ SpatialConsistencyLoss class works with Keras API
- ✅ No runtime errors with typical input shapes

### Implementation Summary

**Completed:** 2025-10-25

**What was implemented:**
- ✅ Extracted `color_constancy_loss()` function from zero_dce.py (lines 192-202)
- ✅ Extracted `exposure_loss()` function from zero_dce.py (lines 214-217)
- ✅ Extracted `illumination_smoothness_loss()` function from zero_dce.py (lines 228-239)
- ✅ Extracted `SpatialConsistencyLoss` class from zero_dce.py (lines 250-325)
- ✅ Added comprehensive docstrings (Google style) for all functions and class
- ✅ Added type hints for all parameters and return values
- ✅ Added detailed explanations of loss computation in docstrings
- ✅ Created `test_loss.py` for regression testing

**Test Results:**
```
Color constancy loss: Non-negative scalar (range: 0.000003 to 0.000005)
Exposure loss (default mean_val=0.6): 0.010054
Illumination smoothness loss: 2088.177246
Spatial consistency loss (mean): 0.009522
✅ SpatialConsistencyLoss compatible with Keras API
✅ All loss values are non-negative
✅ Spatial loss with identical images: 0.000000 (as expected)
```

### Git Commit
```bash
git add loss.py
git commit -m "Refactor: Implement loss.py - Unsupervised loss functions

- Extract color_constancy_loss from zero_dce.py
- Extract exposure_loss with configurable mean_val
- Extract illumination_smoothness_loss
- Extract SpatialConsistencyLoss class
- Add type hints and comprehensive docstrings
"
```

---

## Step 3: Implement `model.py`

**Status:** ✅ Complete (2025-10-25)
**Estimated Time:** 30-40 minutes
**Dependencies:** Requires `loss.py` from Step 2

### Components to Extract from `zero_dce.py`

**Lines to move:**
- Lines 148-174: `build_dce_net()` function
- Lines 335-467: `ZeroDCE` class (entire implementation)
- Lines 366-383: `get_enhanced_image()` method

### Implementation Requirements

```python
# model.py structure

import tensorflow as tf
import keras
from keras import layers
from loss import (
    color_constancy_loss,
    exposure_loss,
    illumination_smoothness_loss,
    SpatialConsistencyLoss
)

def build_dce_net() -> keras.Model:
    """Build DCE-Net architecture.

    7-layer CNN with symmetrical skip connections:
    - Conv1-4: 32 filters, 3x3, ReLU
    - Skip connections at layers 4, 5, 6
    - Conv7: 24 filters (8 iterations × 3 channels), Tanh

    Returns:
        Keras Model that maps input images to curve parameter maps
    """
    # Implementation from lines 148-174
    pass

def get_enhanced_image(data: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
    """Apply curve enhancement iteratively.

    Applies 8 iterations of the learned curves:
        x_{n+1} = x_n + r_n * (x_n^2 - x_n)

    Args:
        data: Input low-light image (batch, h, w, 3)
        output: Curve parameters (batch, h, w, 24)

    Returns:
        Enhanced image (batch, h, w, 3)
    """
    # Implementation from lines 366-383
    # Extract as standalone function
    pass

class ZeroDCE(keras.Model):
    """Zero-DCE training model wrapper.

    Combines DCE-Net with unsupervised loss functions for training.
    """

    def __init__(self, **kwargs):
        # Implementation from lines 336-338
        pass

    def compile(self, learning_rate, **kwargs):
        # Implementation from lines 340-354
        pass

    @property
    def metrics(self):
        # Implementation from lines 356-364
        pass

    def call(self, data):
        # Implementation from lines 385-387
        pass

    def compute_losses(self, data, output):
        # Implementation from lines 389-410
        pass

    def train_step(self, data):
        # Implementation from lines 412-432
        pass

    def test_step(self, data):
        # Implementation from lines 434-448
        pass

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        # Implementation from lines 450-457
        pass

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        # Implementation from lines 459-466
        pass
```

### Testing

```bash
# Test model building
python -c "
from model import build_dce_net, get_enhanced_image, ZeroDCE
import tensorflow as tf

# Test DCE-Net
dce_net = build_dce_net()
print('DCE-Net summary:')
dce_net.summary()

# Test enhancement function
dummy_image = tf.random.uniform([1, 256, 256, 3], 0, 1)
dummy_params = tf.random.normal([1, 256, 256, 24])
enhanced = get_enhanced_image(dummy_image, dummy_params)
print(f'Enhanced shape: {enhanced.shape}')
print(f'Enhanced range: [{tf.reduce_min(enhanced):.3f}, {tf.reduce_max(enhanced):.3f}]')

# Test ZeroDCE model
zero_dce = ZeroDCE()
zero_dce.compile(learning_rate=1e-4)
output = zero_dce(dummy_image)
print(f'ZeroDCE output shape: {output.shape}')
"
```

### Success Criteria
- ✅ DCE-Net builds without errors
- ✅ Model has correct architecture (7 conv layers, 24 output channels)
- ✅ `get_enhanced_image()` works as standalone function
- ✅ ZeroDCE model compiles and runs inference
- ✅ Enhanced images are in valid range (can be outside [0,1] slightly)

### Implementation Summary

**Completed:** 2025-10-25

**What was implemented:**
- ✅ Extracted `build_dce_net()` function from zero_dce.py (lines 148-174)
- ✅ Extracted `get_enhanced_image()` as standalone function from zero_dce.py (lines 366-383)
- ✅ Extracted `ZeroDCE` class with all methods from zero_dce.py (lines 335-467)
- ✅ Integrated loss functions from loss.py module
- ✅ Added comprehensive docstrings (Google style) for all functions and class methods
- ✅ Added type hints for all parameters and return values
- ✅ Fixed Keras 3 compatibility for save_weights/load_weights (`.weights.h5` extension)
- ✅ Created `test_model.py` for comprehensive regression testing

**Test Results:**
```
DCE-Net Architecture:
  - Total parameters: 79,416 (310.22 KB)
  - Input shape: (None, None, None, 3)
  - Output shape: (None, None, None, 24)

Loss computation:
  - total_loss: 1875.081543
  - illumination_smoothness_loss: 1874.856201
  - spatial_constancy_loss: 0.003283
  - color_constancy_loss: 0.001706
  - exposure_loss: 0.220372

Weight save/load:
  - Max difference between outputs: 0.0000000000

Variable image sizes:
  - ✅ 128×128, 256×256, 512×512 all work correctly
```

### Git Commit
```bash
git add model.py
git commit -m "Refactor: Implement model.py - DCE-Net and ZeroDCE model

- Extract build_dce_net() architecture from zero_dce.py
- Extract get_enhanced_image() as standalone function
- Extract ZeroDCE training wrapper class
- Integrate loss functions from loss.py
- Add comprehensive docstrings and type hints
- Fix Keras 3 compatibility for weight saving/loading
"
```

---

## Step 4: Implement `train.py`

**Status:** ✅ Complete (2025-10-25)
**Estimated Time:** 30-40 minutes
**Dependencies:** Requires `dataset.py`, `model.py`, `loss.py`

### Components to Extract from `zero_dce.py`

**Lines to move:**
- Lines 47-60: Imports and environment setup
- Lines 473-475: Model instantiation and training
- Lines 478-493: Plotting functions

### Implementation Requirements

```python
# train.py structure

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import get_dataset
from model import ZeroDCE

def plot_training_history(history, save_dir: str = "./training_plots"):
    """Plot and save training history curves.

    Args:
        history: Keras History object from model.fit()
        save_dir: Directory to save plot images
    """
    metrics = [
        "total_loss",
        "illumination_smoothness_loss",
        "spatial_constancy_loss",
        "color_constancy_loss",
        "exposure_loss"
    ]

    Path(save_dir).mkdir(exist_ok=True)

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history[metric], label=metric)
        plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.title(f"Train and Validation {metric} Over Epochs")
        plt.legend()
        plt.grid()
        plt.savefig(f"{save_dir}/{metric}.png")
        plt.close()

    print(f"Training plots saved to {save_dir}/")

def main():
    parser = argparse.ArgumentParser(
        description="Train Zero-DCE model for low-light image enhancement"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./lol_dataset",
        help="Path to LOL dataset directory"
    )
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=400,
        help="Maximum number of images for training"
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image size (height and width)"
    )

    # Output arguments
    parser.add_argument(
        "--save-path",
        type=str,
        default="./weights/zero_dce.weights.h5",
        help="Path to save trained model weights"
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="./training_plots",
        help="Directory to save training plots"
    )

    args = parser.parse_args()

    # Create output directories
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    train_dataset, val_dataset, _ = get_dataset(
        dataset_path=args.dataset_path,
        max_train_images=args.max_train_images,
        batch_size=args.batch_size,
        image_size=args.image_size
    )

    print(f"Train Dataset: {train_dataset}")
    print(f"Validation Dataset: {val_dataset}")

    # Build and compile model
    print("\nBuilding Zero-DCE model...")
    zero_dce_model = ZeroDCE()
    zero_dce_model.compile(learning_rate=args.learning_rate)

    # Train model
    print(f"\nTraining for {args.epochs} epochs...")
    history = zero_dce_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs
    )

    # Save model weights
    print(f"\nSaving model weights to {args.save_path}...")
    zero_dce_model.save_weights(args.save_path)

    # Plot training history
    print("\nGenerating training plots...")
    plot_training_history(history, args.plot_dir)

    print("\n✅ Training completed successfully!")
    print(f"   Weights saved: {args.save_path}")
    print(f"   Plots saved: {args.plot_dir}/")

if __name__ == "__main__":
    main()
```

### Testing

```bash
# Test with minimal training (1 epoch)
python train.py --epochs 1 --batch-size 4 --save-path ./test_weights.h5

# Test full training (optional - takes time)
python train.py --epochs 100 --batch-size 16
```

### Success Criteria
- ✅ Script runs without errors
- ✅ Training progress is displayed
- ✅ Model weights are saved to specified path
- ✅ Training plots are generated and saved
- ✅ CLI arguments work correctly

### Implementation Summary

**Completed:** 2025-10-25

**What was implemented:**
- ✅ Created complete training script with environment setup (KERAS_BACKEND)
- ✅ Implemented `plot_training_history()` function for visualization
- ✅ Implemented `main()` function with comprehensive argparse CLI
- ✅ Integrated dataset, model, and loss modules seamlessly
- ✅ Added 8 configurable CLI arguments (dataset, training, and output parameters)
- ✅ Created automatic directory creation for weights and plots
- ✅ Added informative console output and success messages
- ✅ Comprehensive docstrings (Google style) for all functions

**Test Results:**
```
Training with 1 epoch (test run):
  - ✅ Script executed successfully
  - ✅ All 5 loss metrics tracked and displayed
  - ✅ Model weights saved (352KB .weights.h5 file)
  - ✅ All 5 training plots generated (total_loss, illumination_smoothness_loss,
       spatial_constancy_loss, color_constancy_loss, exposure_loss)
  - ✅ CLI help documentation displays correctly
  - ✅ Integration with dataset.py and model.py works flawlessly

Training progress output:
  - total_loss: 8.94 → 4.08 (training), 3.81 (validation)
  - illumination_smoothness_loss: 5.99 → 1.16
  - exposure_loss: 2.94 → 2.91
  - color_constancy_loss: 0.0004 → 0.0030
  - spatial_constancy_loss: 0.00007 → 0.00008
```

### Git Commit
```bash
git add train.py
git commit -m "Refactor: Implement train.py - Training script with CLI

- Extract training logic from zero_dce.py
- Add argparse for configurable training parameters
- Integrate dataset, model, and loss modules
- Add plot_training_history() for visualization
- Create weights and plots directories automatically
- Add comprehensive CLI help messages
"
```

---

## Step 5: Implement `compare.py`

**Status:** ✅ Complete (2025-10-25)
**Estimated Time:** 40-50 minutes
**Dependencies:** Requires `model.py`

### Components to Extract from `zero_dce.py`

**Lines to move:**
- Lines 509-516: `infer()` function
- Lines 500-506: `plot_results()` function
- Lines 529-536: Inference loop on test images

### New Components to Add

Classical enhancement methods for comparison:
- PIL AutoContrast (already in original)
- Histogram Equalization (OpenCV)
- CLAHE - Contrast Limited Adaptive Histogram Equalization (OpenCV)
- Gamma Correction

### Implementation Requirements

```python
# compare.py structure

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import keras

from model import ZeroDCE

def load_model_for_inference(weights_path: str) -> ZeroDCE:
    """Load trained Zero-DCE model for inference.

    Args:
        weights_path: Path to saved model weights (.h5 file)

    Returns:
        Loaded ZeroDCE model
    """
    model = ZeroDCE()
    model.load_weights(weights_path)
    return model

def enhance_with_zero_dce(image: Image.Image, model: ZeroDCE) -> Image.Image:
    """Enhance image using Zero-DCE model.

    Args:
        image: PIL Image (RGB)
        model: Trained ZeroDCE model

    Returns:
        Enhanced PIL Image
    """
    # Implementation from lines 509-516 (infer function)
    pass

def enhance_with_autocontrast(image: Image.Image) -> Image.Image:
    """Enhance using PIL AutoContrast."""
    return ImageOps.autocontrast(image)

def enhance_with_histogram_eq(image: Image.Image) -> Image.Image:
    """Enhance using histogram equalization."""
    img_array = np.array(image)
    img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(img_output)

def enhance_with_clahe(image: Image.Image, clip_limit: float = 2.0) -> Image.Image:
    """Enhance using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    img_array = np.array(image)
    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    img_output = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_output)

def enhance_with_gamma_correction(image: Image.Image, gamma: float = 2.2) -> Image.Image:
    """Enhance using gamma correction."""
    img_array = np.array(image) / 255.0
    img_corrected = np.power(img_array, 1.0 / gamma)
    img_output = (img_corrected * 255).astype(np.uint8)
    return Image.fromarray(img_output)

def compare_methods(
    input_path: str,
    weights_path: str,
    output_path: str = None,
    methods: list = None,
    save_individual: bool = False
):
    """Compare enhancement methods on a single image.

    Args:
        input_path: Path to input low-light image
        weights_path: Path to trained Zero-DCE weights
        output_path: Path to save comparison image (optional)
        methods: List of methods to compare (default: all)
        save_individual: Save individual enhanced images
    """
    # Load original image
    original_image = Image.open(input_path)

    # Default methods if not specified
    if methods is None:
        methods = ["zero-dce", "autocontrast", "histogram-eq", "clahe", "gamma"]

    # Load Zero-DCE model if needed
    model = None
    if "zero-dce" in methods:
        print(f"Loading Zero-DCE model from {weights_path}...")
        model = load_model_for_inference(weights_path)

    # Apply enhancement methods
    results = {"original": original_image}
    method_map = {
        "zero-dce": ("Zero-DCE", lambda: enhance_with_zero_dce(original_image, model)),
        "autocontrast": ("AutoContrast", lambda: enhance_with_autocontrast(original_image)),
        "histogram-eq": ("Histogram Eq", lambda: enhance_with_histogram_eq(original_image)),
        "clahe": ("CLAHE", lambda: enhance_with_clahe(original_image)),
        "gamma": ("Gamma Correction", lambda: enhance_with_gamma_correction(original_image))
    }

    print("Applying enhancement methods...")
    for method_key in methods:
        if method_key in method_map:
            method_name, enhance_fn = method_map[method_key]
            print(f"  - {method_name}")
            results[method_name] = enhance_fn()

    # Create comparison plot
    num_images = len(results)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    if num_images == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, results.items()):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()

    # Save or show
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n✅ Comparison saved to {output_path}")
    else:
        plt.show()

    plt.close()

    # Save individual images if requested
    if save_individual and output_path:
        output_dir = Path(output_path).parent / "individual"
        output_dir.mkdir(exist_ok=True)

        for title, img in results.items():
            if title != "original":
                save_path = output_dir / f"{Path(input_path).stem}_{title.lower().replace(' ', '_')}.png"
                img.save(save_path)

        print(f"Individual images saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(
        description="Compare Zero-DCE with classical image enhancement methods"
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to input low-light image"
    )
    parser.add_argument(
        "-w", "--weights",
        type=str,
        required=True,
        help="Path to trained Zero-DCE model weights (.h5 file)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save comparison image (if not specified, displays instead)"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        choices=["zero-dce", "autocontrast", "histogram-eq", "clahe", "gamma"],
        help="Enhancement methods to compare (default: all)"
    )
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Save individual enhanced images"
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.input).exists():
        print(f"❌ Error: Input image not found: {args.input}")
        return

    if not Path(args.weights).exists():
        print(f"❌ Error: Model weights not found: {args.weights}")
        return

    # Run comparison
    compare_methods(
        input_path=args.input,
        weights_path=args.weights,
        output_path=args.output,
        methods=args.methods,
        save_individual=args.save_individual
    )

if __name__ == "__main__":
    main()
```

### Testing

```bash
# Test with a sample image (requires trained weights)
python compare.py -i ./lol_dataset/eval15/low/1.png \
                  -w ./weights/zero_dce.weights.h5 \
                  -o ./comparison.png

# Test specific methods only
python compare.py -i ./lol_dataset/eval15/low/1.png \
                  -w ./weights/zero_dce.weights.h5 \
                  --methods zero-dce autocontrast clahe

# Test with individual saves
python compare.py -i ./test_image.jpg \
                  -w ./weights/model.h5 \
                  -o ./output/comparison.png \
                  --save-individual
```

### Success Criteria
- ✅ Script loads image and model correctly
- ✅ All enhancement methods run without errors
- ✅ Comparison visualization is generated
- ✅ Output is saved when specified
- ✅ CLI arguments work as expected
- ✅ Individual images saved when requested

### Implementation Summary

**Completed:** 2025-10-25

**What was implemented:**

**Phase 1: Initial Implementation**
- ✅ Extracted and adapted `infer()` function as `enhance_with_zero_dce()` from zero_dce.py
- ✅ Created `load_model_for_inference()` for model loading
- ✅ Implemented 4 classical enhancement methods in compare.py:
  * `enhance_with_autocontrast()` - PIL AutoContrast
  * `enhance_with_histogram_eq()` - OpenCV Histogram Equalization
  * `enhance_with_clahe()` - Contrast Limited Adaptive Histogram Equalization
  * `enhance_with_gamma_correction()` - Gamma correction (gamma=2.2)
- ✅ Implemented `compare_methods()` function for side-by-side visualization
- ✅ Added comprehensive CLI with argparse (5 arguments)
- ✅ Added individual image saving functionality
- ✅ Added opencv-python>=4.8.0 to project dependencies (pyproject.toml)
- ✅ Comprehensive docstrings (Google style) for all functions
- ✅ Created `test_compare.py` for comprehensive regression testing

**Phase 2: Reference Image Feature (Enhancement)**
- ✅ Added optional `-r/--reference` CLI argument
- ✅ Added `reference_path` parameter to `compare_methods()` function
- ✅ Display reference (ground truth) image from LOL dataset
- ✅ Position reference between original and enhanced images
- ✅ Graceful error handling for missing reference image
- ✅ Updated test suite with reference image test (Test 7)
- ✅ Updated CLI help with usage examples

**Phase 3: Modular Refactoring (Code Organization)**
- ✅ Created `classical_methods.py` module (143 lines)
  * Extracted all 4 classical enhancement methods from compare.py
  * Added `CLASSICAL_METHODS` dictionary for method registration
  * Added `get_available_methods()` helper function
  * Added `get_method_info()` helper function
  * Comprehensive docstrings for all functions
- ✅ Refactored `compare.py` (reduced from 320 to 243 lines, 24% reduction)
  * Imports from `classical_methods` module
  * Simplified method invocation using `CLASSICAL_METHODS` dictionary
  * Only contains Zero-DCE inference and visualization orchestration
  * Better separation of concerns
- ✅ Created `test_classical_methods.py` (186 lines, 9 test cases)
  * Tests all 4 enhancement methods independently
  * Tests CLASSICAL_METHODS dictionary structure
  * Tests helper functions
  * Tests with synthetic and real images
  * Tests custom parameters
- ✅ Updated `test_compare.py` to test module integration

**Test Results:**
```
test_compare.py: All 9 test cases passed
  ✅ Module imports successful (including cv2)
  ✅ Image loading works (600x400 RGB)
  ✅ All classical methods work:
     - AutoContrast: 600x400 pixels
     - Histogram Eq: 600x400 pixels
     - CLAHE: 600x400 pixels
     - Gamma Correction: 600x400 pixels
  ✅ Zero-DCE model loading and inference work
     - Output value range: [0, 251] (valid)
  ✅ Comparison visualization created (3.9MB PNG with 5 methods)
  ✅ Reference image inclusion works (2.9MB with reference)
  ✅ Individual image saving works (separate files per method)
  ✅ CLI help documentation complete

test_classical_methods.py: All 9 test cases passed
  ✅ Module imports work correctly
  ✅ CLASSICAL_METHODS dictionary properly structured
  ✅ get_available_methods() returns correct list
  ✅ get_method_info() provides complete information
  ✅ All 4 methods work on test images
  ✅ Custom parameters work (CLAHE clip_limit, Gamma gamma)
  ✅ All methods work on real LOL dataset images
  ✅ Methods callable via CLASSICAL_METHODS dictionary

Real-world test:
  - Input: ./lol_dataset/eval15/low/1.png
  - Reference: ./lol_dataset/eval15/high/1.png
  - Methods: Zero-DCE, AutoContrast, CLAHE
  - Output: 2.9MB comparison image with reference
  - All enhancements visually distinct and correct
```

**Key Features:**
- Supports 5 enhancement methods total (1 deep learning + 4 classical)
- Flexible CLI allows selecting specific methods
- Side-by-side comparison with original and optional reference image
- Optional individual image export to subdirectory
- Works with any image size (not restricted to 256x256)
- Comprehensive error handling and user feedback
- Modular architecture: classical methods can be used independently
- Easy to extend with new classical methods via CLASSICAL_METHODS registry

**Files Created:**
- `compare.py` - Main comparison tool (243 lines)
- `classical_methods.py` - Classical enhancement methods module (143 lines)
- `test_compare.py` - Test suite for compare.py (243 lines)
- `test_classical_methods.py` - Test suite for classical_methods.py (186 lines)

### Git Commits

**Commit 1: Initial Implementation**
```bash
git add compare.py test_compare.py pyproject.toml
git commit -m "Refactor: Implement compare.py - Inference and comparison tool

- Extract inference logic from zero_dce.py
- Add CLI with argparse for flexible usage
- Implement Zero-DCE enhancement function (enhance_with_zero_dce)
- Add 4 classical enhancement methods
- Add side-by-side comparison visualization
- Support saving individual enhanced images
- Add opencv-python>=4.8.0 to dependencies
- Create test_compare.py for regression testing
"
```

**Commit 2: Reference Image Feature**
```bash
git add compare.py test_compare.py
git commit -m "Feature: Add optional reference image to compare.py

- Add -r/--reference CLI argument for ground truth comparison
- Display reference image as 'Reference (Ground Truth)' in comparisons
- Position reference between original and enhanced images
- Add graceful error handling for missing reference image
- Update test suite with reference image test (Test 7)
- Update CLI help with usage examples
"
```

**Commit 3: Modular Refactoring**
```bash
git add classical_methods.py compare.py test_classical_methods.py test_compare.py
git commit -m "Refactor: Extract classical methods into separate module

- Create classical_methods.py with 4 enhancement methods
- Add CLASSICAL_METHODS dictionary for method registration
- Add get_available_methods() and get_method_info() helpers
- Refactor compare.py to use imported methods
- Reduce compare.py by 77 lines (24% reduction)
- Create test_classical_methods.py (9 tests, all passing)
- Update test_compare.py to test module integration
"
```

---

## Step 6: Update Documentation

**Status:** ✅ Complete (2025-10-25)
**Estimated Time:** 15-20 minutes
**Dependencies:** All previous steps completed

### Tasks

1. **Update `README.md`**
   - Document new modular structure
   - Add usage examples for `train.py` and `compare.py`
   - Update installation instructions
   - Add example commands

2. **Add note to `zero_dce.py`**
   - Keep file as reference
   - Add header explaining refactored structure
   - Point to new modular files

3. **Verify `pyproject.toml`**
   - Ensure all dependencies are listed
   - Add opencv-python if missing

### Implementation Summary

**Completed:** 2025-10-25

**What was implemented:**

1. ✅ **Created comprehensive README.md** (525 lines)
   - Complete project overview with badges (Python, TensorFlow, Keras)
   - Key features and architecture documentation
   - Detailed project structure tree
   - Step-by-step installation instructions (uv and pip)
   - LOL Dataset download instructions
   - Complete usage documentation:
     * Training with all CLI parameters explained
     * Inference and comparison with multiple examples
     * Example commands for different use cases
   - Architecture diagrams (ASCII art):
     * Zero-DCE framework flow
     * DCE-Net 7-layer architecture with skip connections
     * Loss function weights and formulas
     * Enhancement iteration formula
   - Module documentation for all 6 modules
   - Testing instructions for all test files
   - Training progress metrics table
   - Performance benchmarks (training and inference time)
   - Citation in BibTeX format
   - References to paper and dataset
   - Comprehensive troubleshooting section
   - Contributing guidelines

2. ✅ **Added reference note to zero_dce.py** (60 lines)
   - Added prominent ⚠️ warning at the top of the file
   - Explanation that this is the original monolithic implementation
   - Guidance to use modular implementation instead
   - Quick-start examples for train.py and compare.py
   - Complete modular structure listing
   - Links to documentation files (README, AGENTS, REFACTOR_PLAN)
   - Benefits of modular structure explained
   - Justification for keeping the original file
   - Clear call-to-action to use modular code

3. ✅ **Updated pyproject.toml**
   - Changed description from "Add your description here" to:
     "Zero-DCE: Zero-Reference Deep Curve Estimation for low-light image enhancement in Keras 3"
   - Verified all dependencies are present:
     * keras>=3.11.3
     * matplotlib>=3.10.7
     * numpy>=2.3.4
     * opencv-python>=4.11.0.86 (added in Step 5)
     * pillow>=12.0.0
     * pyqt6>=6.10.0
     * tensorflow[and-cuda]>=2.20.0

**Key Documentation Features:**

- **Comprehensive coverage:** Installation, usage, architecture, testing, troubleshooting
- **Educational focus:** Detailed explanations suitable for graduate course project
- **Practical examples:** Multiple usage scenarios with real commands
- **Visual aids:** ASCII diagrams for architecture understanding
- **Academic rigor:** Proper citations and references
- **User-friendly:** Clear structure, helpful error guidance, multiple entry points
- **Maintenance-friendly:** Modular documentation matches modular code

**Files Updated:**
- `README.md` - Complete documentation (empty → 525 lines)
- `zero_dce.py` - Added reference note header (+60 lines)
- `pyproject.toml` - Updated description
- `REFACTOR_PLAN.md` - Updated progress tracker

### Git Commit
```bash
git add README.md zero_dce.py pyproject.toml REFACTOR_PLAN.md
git commit -m "Docs: Update documentation for modular structure

- Create comprehensive README.md (525 lines)
  * Complete project overview with architecture diagrams
  * Detailed installation and usage instructions
  * All CLI parameters documented with examples
  * Testing, troubleshooting, and contributing sections
  * Proper citations and references
- Add reference note to zero_dce.py
  * Prominent warning that this is the original implementation
  * Guidance to use modular code instead
  * Quick-start examples and documentation links
- Update pyproject.toml description
- Mark Step 6 complete in REFACTOR_PLAN.md
"
```

---

## Verification Checklist

After completing all steps, verify:

- [ ] All modules import successfully
- [ ] `train.py` runs complete training cycle
- [ ] `compare.py` generates comparison images
- [ ] Model weights are compatible between modules
- [ ] CLI arguments work as documented
- [ ] No circular dependencies between modules
- [ ] All code is properly documented
- [ ] Git history shows logical commit progression

---

## Troubleshooting

### Common Issues

**Import errors:**
- Ensure all modules are in the same directory
- Check Python path includes project directory

**OOM errors during training:**
- Reduce `--batch-size` parameter
- Reduce `--image-size` if necessary

**Model weight loading fails:**
- Verify weight file path is correct
- Ensure weights were saved from DCE-Net (not wrapper)

**Comparison fails:**
- Check if opencv-python is installed: `pip install opencv-python`
- Verify input image format is supported (PNG, JPG)

---

## Next Steps After Refactoring

1. **Test thoroughly** with complete training run
2. **Document hyperparameters** and their effects
3. **Add unit tests** (optional but recommended)
4. **Experiment with loss weights** to improve results
5. **Try on custom images** outside LOL Dataset
6. **Write project report** for course submission

---

**Last Updated:** 2025-10-25
**Status:** ✅ Complete - All Steps 1-6 Finished
**Estimated Total Time:** 2.5-3.5 hours
**Time Spent:** ~220 minutes (all steps)

## Step 5 Additional Notes

Step 5 was implemented in three phases:
1. **Initial Implementation** - Core comparison functionality with classical methods
2. **Reference Image Feature** - Added optional ground truth comparison from LOL dataset
3. **Modular Refactoring** - Extracted classical methods into separate module for better code organization

This comprehensive approach resulted in a more maintainable, extensible, and well-tested codebase. The modular structure allows classical methods to be reused in other scripts and makes it easy to add new enhancement methods in the future.
