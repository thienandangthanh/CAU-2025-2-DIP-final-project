# Zero-DCE Refactoring Plan

> Detailed step-by-step guide for refactoring `zero_dce.py` into modular components

## Overview

This document provides a detailed roadmap for refactoring the monolithic `zero_dce.py` (537 lines) into 5 modular files. Each step should be completed, tested, and committed before moving to the next.

## Prerequisites

- ✅ `zero_dce.py` exists with working implementation
- ✅ Empty module files created: `dataset.py`, `loss.py`, `model.py`, `train.py`, `compare.py`
- ✅ LOL Dataset downloaded and available at `./lol_dataset/`
- ✅ Development environment set up with required packages

## Refactoring Steps

---

## Step 1: Implement `dataset.py`

**Status:** 🔄 Pending  
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

**Status:** 🔄 Pending  
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

**Status:** 🔄 Pending  
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

### Git Commit
```bash
git add model.py
git commit -m "Refactor: Implement model.py - DCE-Net and ZeroDCE model

- Extract build_dce_net() architecture from zero_dce.py
- Extract get_enhanced_image() as standalone function
- Extract ZeroDCE training wrapper class
- Integrate loss functions from loss.py
- Add comprehensive docstrings and type hints
"
```

---

## Step 4: Implement `train.py`

**Status:** 🔄 Pending  
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
        default="./weights/zero_dce_weights.h5",
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

**Status:** 🔄 Pending  
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
                  -w ./weights/zero_dce_weights.h5 \
                  -o ./comparison.png

# Test specific methods only
python compare.py -i ./lol_dataset/eval15/low/1.png \
                  -w ./weights/zero_dce_weights.h5 \
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

### Git Commit
```bash
git add compare.py
git commit -m "Refactor: Implement compare.py - Inference and comparison tool

- Extract inference logic from zero_dce.py
- Add CLI with argparse for flexible usage
- Implement Zero-DCE enhancement function
- Add classical methods: AutoContrast, Histogram Eq, CLAHE, Gamma
- Add side-by-side comparison visualization
- Support saving individual enhanced images
- Add input validation and error handling
"
```

---

## Step 6: Update Documentation

**Status:** 🔄 Pending  
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

### Example README Structure

```markdown
# Zero-DCE: Low-Light Image Enhancement

Re-implementation of Zero-Reference Deep Curve Estimation in Keras 3.

## Project Structure

- `dataset.py` - Data loading and preprocessing
- `loss.py` - Unsupervised loss functions
- `model.py` - DCE-Net architecture and ZeroDCE model
- `train.py` - Training script with CLI
- `compare.py` - Inference and comparison tool
- `zero_dce.py` - Original monolithic implementation (reference)

## Installation

[Add installation instructions]

## Usage

### Training
```bash
python train.py --epochs 100 --batch-size 16
```

### Inference
```bash
python compare.py -i input.jpg -w weights.h5 -o output.png
```

[Continue with more detailed documentation]
```

### Git Commit
```bash
git add README.md zero_dce.py pyproject.toml
git commit -m "Docs: Update documentation for modular structure

- Update README with new project structure
- Add usage examples for train.py and compare.py
- Add reference note to zero_dce.py
- Verify dependencies in pyproject.toml
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
**Status:** Ready for implementation  
**Estimated Total Time:** 2.5-3.5 hours
