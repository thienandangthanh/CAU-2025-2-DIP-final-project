# Zero-DCE Optimization Guide

> Strategies and techniques for improving Zero-DCE model performance

**Last Updated:** 2025-11-28
**Status:** Future optimization opportunities

## Table of Contents

1. [Quick Wins](#1-quick-wins-low-effort-high-impact)
2. [Hyperparameter Tuning](#2-hyperparameter-tuning)
3. [Loss Function Optimization](#3-loss-function-optimization)
4. [Training Strategies](#4-advanced-training-strategies)
5. [Architecture Modifications](#5-architecture-modifications)
6. [Data Augmentation](#6-data-augmentation)
7. [Additional Datasets](#7-additional-datasets)
8. [Optimizer Experiments](#8-optimizer-experiments)
9. [Recommended Optimization Path](#recommended-optimization-path)

---

## 1. Quick Wins (Low Effort, High Impact)

### Use All Training Data

**Current:** Only 400 out of 485 images used (see `train.py:74`)

**Optimization:**
```bash
python train.py --max-train-images 485
```

**Impact:** ~20% more training data, better generalization

---

### Train Longer

**Current:** Default 100 epochs

**Optimization:**
```bash
python train.py --epochs 150
# Or even 200 if validation loss hasn't plateaued
```

**Impact:** Better convergence if model is still improving

---

### Adjust Learning Rate

**Current:** `1e-4` (see `train.py:88`)

**Optimization:**
```bash
# More stable, better final performance
python train.py --learning-rate 5e-5

# Faster initial convergence
python train.py --learning-rate 2e-4
```

**Impact:** Can significantly affect convergence speed and final quality

---

## 2. Hyperparameter Tuning

### Batch Size Experiments

**Current:** 16 (see `train.py:83`)

**File to modify:** `train.py` (already has `--batch-size` argument)

**Options:**
```bash
# Larger batch = more stable gradients, needs more GPU memory
python train.py --batch-size 32

# Smaller batch = more frequent updates, noisier gradients
python train.py --batch-size 8
```

**Considerations:**
- Larger batch sizes may require proportional learning rate increase
- Monitor GPU memory usage
- Affects training time (larger batch = fewer steps per epoch)

---

### Image Resolution

**Current:** 256x256 (see `train.py:92`)

**Optimization:**
```bash
# Higher resolution for better detail preservation
python train.py --image-size 512

# Lower resolution for faster training/experimentation
python train.py --image-size 128
```

**Impact:**
- Higher resolution: Better detail, more GPU memory, slower training
- Lower resolution: Faster training, may miss fine details

**Warning:** Requires sufficient GPU memory for 512x512 images

---

## 3. Loss Function Optimization

### Current Loss Weights

**File:** `model.py:224-229`

```python
loss_illumination = 200 * illumination_smoothness_loss(output)
loss_spatial_constancy = 1 * spatial_constancy_loss(enhanced_image, data)
loss_color_constancy = 5 * color_constancy_loss(enhanced_image)
loss_exposure = 10 * exposure_loss(enhanced_image)
```

### Tuning Guidelines

| Loss Component | Current Weight | When to Increase | When to Decrease |
|----------------|----------------|------------------|------------------|
| Illumination Smoothness | 200 | Artifacts, unnatural curves | Too conservative enhancement |
| Spatial Constancy | 1 | Loss of detail, spatial distortion | Over-smoothing |
| Color Constancy | 5 | Color shifts, unnatural hues | Correct colors already |
| Exposure | 10 | Under/over-exposure issues | Correct brightness already |

### Suggested Experiments

**For brighter results:**
```python
loss_exposure = 15 * exposure_loss(enhanced_image)  # Increase from 10
```

**For better color preservation:**
```python
loss_color_constancy = 10 * color_constancy_loss(enhanced_image)  # Increase from 5
```

**For more aggressive enhancement:**
```python
loss_illumination = 150 * illumination_smoothness_loss(output)  # Decrease from 200
```

### Implementation

Add CLI arguments to `train.py`:
```python
parser.add_argument('--weight-illumination', type=float, default=200)
parser.add_argument('--weight-spatial', type=float, default=1)
parser.add_argument('--weight-color', type=float, default=5)
parser.add_argument('--weight-exposure', type=float, default=10)
```

Then pass to `ZeroDCE` model and modify `compute_losses()` method.

---

### Exposure Target Value

**Current:** 0.6 (see `loss.py:39`)

```python
def exposure_loss(x: tf.Tensor, mean_val: float = 0.6) -> tf.Tensor:
```

**Optimization:**
- `0.5`: Darker, more conservative enhancement
- `0.6`: Current (balanced)
- `0.7`: Brighter enhancement

**Implementation:** Add parameter to `exposure_loss()` and make configurable via CLI

---

## 4. Advanced Training Strategies

### Learning Rate Scheduling

**File to modify:** `train.py`

**Option 1: Reduce on Plateau**
```python
from keras.callbacks import ReduceLROnPlateau

lr_callback = ReduceLROnPlateau(
    monitor='val_total_loss',
    factor=0.5,              # Reduce by half
    patience=10,             # After 10 epochs without improvement
    min_lr=1e-7,            # Don't go below this
    verbose=1
)

history = zero_dce_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=args.epochs,
    callbacks=[lr_callback]
)
```

**Option 2: Cosine Decay**
```python
from keras.callbacks import LearningRateScheduler
import math

def cosine_decay(epoch, lr):
    """Cosine annealing schedule."""
    return args.learning_rate * 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))

lr_scheduler = LearningRateScheduler(cosine_decay, verbose=1)

history = zero_dce_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=args.epochs,
    callbacks=[lr_scheduler]
)
```

**Impact:** Helps escape local minima, better final convergence

---

### Early Stopping

**File to modify:** `train.py`

```python
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_total_loss',
    patience=20,                  # Stop if no improvement for 20 epochs
    restore_best_weights=True,    # Restore best weights, not final weights
    verbose=1
)

history = zero_dce_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=args.epochs,
    callbacks=[early_stop]
)
```

**Impact:** Prevents overfitting, saves training time

---

### Model Checkpointing

**File to modify:** `train.py`

```python
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    filepath='./weights/best_model_epoch{epoch:03d}_loss{val_total_loss:.4f}.weights.h5',
    monitor='val_total_loss',
    save_best_only=True,          # Only save when val_loss improves
    save_weights_only=True,
    verbose=1
)

history = zero_dce_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=args.epochs,
    callbacks=[checkpoint]
)
```

**Impact:** Never lose best model during training

---

### Combined Callback Strategy

```python
callbacks = [
    ReduceLROnPlateau(monitor='val_total_loss', factor=0.5, patience=10, min_lr=1e-7),
    EarlyStopping(monitor='val_total_loss', patience=20, restore_best_weights=True),
    ModelCheckpoint('./weights/best_model.weights.h5', monitor='val_total_loss', save_best_only=True)
]

history = zero_dce_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=args.epochs,
    callbacks=callbacks
)
```

---

## 5. Architecture Modifications

### Increase Model Capacity

**File to modify:** `model.py:20-79`

**Current:** 32 filters per convolutional layer

**Optimization:**
```python
# Change from 32 to 48 or 64
conv1 = layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="same")(input_img)
conv2 = layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv1)
conv3 = layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv2)
conv4 = layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv3)
# ... continue for all layers
```

**Impact:**
- More parameters = better representation capacity
- Slower training, more GPU memory
- May require more training data to avoid overfitting

**Trade-off:** Original paper uses 32 filters for lightweight design

---

### Add Batch Normalization

**File to modify:** `model.py:20-79`

**Implementation:**
```python
# Before (current)
conv1 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(input_img)

# After (with batch norm)
conv1 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same")(input_img)
conv1 = layers.BatchNormalization()(conv1)
conv1 = layers.Activation("relu")(conv1)
```

**Apply to all convolutional layers**

**Impact:**
- Faster convergence
- Higher learning rates possible
- Better generalization
- Slight increase in parameters

---

### Adjust Curve Iterations

**Current:** 8 iterations (24 channels = 8 iterations × 3 RGB)

**File to modify:** `model.py:75` and `model.py:82-124`

**Option 1: Fewer Iterations (Lighter Model)**
```python
# 6 iterations = 18 channels
x_r = layers.Conv2D(18, (3, 3), strides=(1, 1), activation="tanh", padding="same")(int_con3)
```

Then update `get_enhanced_image()`:
```python
def get_enhanced_image(data: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
    r1 = output[:, :, :, :3]
    r2 = output[:, :, :, 3:6]
    r3 = output[:, :, :, 6:9]
    r4 = output[:, :, :, 9:12]
    r5 = output[:, :, :, 12:15]
    r6 = output[:, :, :, 15:18]

    x = data + r1 * (tf.square(data) - data)
    x = x + r2 * (tf.square(x) - x)
    x = x + r3 * (tf.square(x) - x)
    x = x + r4 * (tf.square(x) - x)
    x = x + r5 * (tf.square(x) - x)
    enhanced_image = x + r6 * (tf.square(x) - x)

    return enhanced_image
```

**Option 2: More Iterations (More Refinement)**
```python
# 10 iterations = 30 channels
x_r = layers.Conv2D(30, (3, 3), strides=(1, 1), activation="tanh", padding="same")(int_con3)
```

**Trade-offs:**
- Fewer iterations: Faster inference, lighter model, may be less refined
- More iterations: Better refinement, slower inference, risk of overfitting

---

### Add Residual Connections

**File to modify:** `model.py:20-79`

**Concept:** Add residual connections similar to ResNet

```python
def build_dce_net() -> keras.Model:
    input_img = keras.Input(shape=[None, None, 3])

    conv1 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(input_img)
    conv2 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv1)

    # Add residual connection
    conv2 = layers.Add()([conv2, conv1])

    # Continue with more residual blocks...
```

**Impact:** Easier gradient flow, potentially better convergence

---

## 6. Data Augmentation

**Current:** No data augmentation (as per original paper)

**File to modify:** `dataset.py`

### Recommended Augmentations

```python
def augment_image(image):
    """Apply random augmentations to training images."""
    import tensorflow as tf

    # Random horizontal flip (50% chance)
    image = tf.image.random_flip_left_right(image)

    # Random vertical flip (50% chance)
    image = tf.image.random_flip_up_down(image)

    # Random rotation (±15 degrees)
    # Note: Requires custom implementation or tf-addons

    # Random brightness (±5%)
    image = tf.image.random_brightness(image, max_delta=0.05)

    # Random contrast (±5%)
    image = tf.image.random_contrast(image, lower=0.95, upper=1.05)

    # Random saturation (±5%)
    image = tf.image.random_saturation(image, lower=0.95, upper=1.05)

    # Random hue (±2%)
    image = tf.image.random_hue(image, max_delta=0.02)

    # Clip to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image
```

### Integration into Dataset Pipeline

**Modify `dataset.py`:**
```python
def data_generator(images: list, batch_size: int, image_size: int, augment: bool = False):
    """Generate batches of images with optional augmentation."""
    # ... existing code ...

    if augment:
        image = augment_image(image)

    # ... continue ...
```

**Modify `get_dataset()`:**
```python
def get_dataset(dataset_path, max_train_images=400, batch_size=16, image_size=256, augment_train=True):
    # ...
    train_dataset = train_dataset.map(lambda x: augment_image(x) if augment_train else x)
    # ...
```

**Impact:**
- Effectively increases dataset size
- Better generalization
- Reduces overfitting
- May slow down training slightly

**Note:** Some augmentations (rotation, heavy color changes) may not be appropriate for low-light enhancement

---

## 7. Additional Datasets

### Current Dataset

**LOL Dataset:**
- Training: 485 image pairs
- Testing: 15 image pairs
- Source: `lol_dataset/`

### Additional Datasets to Consider

#### 1. VE-LOL (Varied Exposure LOL)

**Size:** ~2,000 image pairs
**Download:** [VE-LOL GitHub](https://github.com/flyywh/CVPR-2020-Semi-Low-Light)
**Advantages:**
- More diverse lighting conditions
- Larger dataset for better generalization
- Similar structure to LOL

---

#### 2. SICE Dataset (Multi-Exposure)

**Size:** 4,413 images (589 scenes, 7-8 exposures each)
**Download:** [SICE Project Page](https://github.com/csjcai/SICE)
**Advantages:**
- Very large dataset
- Multi-exposure sequences
- Real-world scenes

**Note:** Would need to extract only low-light images

---

#### 3. SID (See-in-the-Dark)

**Size:** 5,094 raw images (indoor + outdoor)
**Download:** [SID Dataset](https://github.com/cchen156/Learning-to-See-in-the-Dark)
**Advantages:**
- Extremely low-light conditions
- Raw sensor data (high quality)
- Indoor and outdoor scenes

**Note:** Requires RAW image processing pipeline

---

#### 4. ExDark

**Size:** 7,363 images (12 object classes)
**Download:** [ExDark Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)
**Advantages:**
- Diverse low-light conditions
- Real-world scenarios
- Large dataset

**Note:** Designed for object detection, but usable for enhancement

---

### Dataset Integration Strategy

**Option 1: Merge Datasets**
```python
# Modify dataset.py to support multiple dataset paths
def get_dataset(dataset_paths: list, ...):
    all_images = []
    for path in dataset_paths:
        images = load_data(path)
        all_images.extend(images)
    # ... continue with merged images
```

**Option 2: Pre-train on Large Dataset, Fine-tune on LOL**
1. Train on SICE or VE-LOL (larger dataset)
2. Fine-tune on LOL dataset with lower learning rate
3. Better final performance on LOL test set

---

## 8. Optimizer Experiments

**Current:** Adam optimizer (see `model.py:162`)

```python
self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
```

### Alternative Optimizers

#### AdamW (Adam with Weight Decay)

**File to modify:** `model.py:162`

```python
self.optimizer = keras.optimizers.AdamW(
    learning_rate=learning_rate,
    weight_decay=1e-4  # L2 regularization
)
```

**Impact:** Better generalization, prevents overfitting

---

#### SGD with Momentum

```python
self.optimizer = keras.optimizers.SGD(
    learning_rate=learning_rate,
    momentum=0.9,
    nesterov=True
)
```

**Impact:** May escape sharp minima, potentially better final performance
**Note:** Usually requires higher learning rate (e.g., 1e-3) and LR scheduling

---

#### Lion Optimizer (Recent, State-of-Art)

```python
# Requires keras-cv or custom implementation
self.optimizer = keras.optimizers.Lion(
    learning_rate=learning_rate * 0.1  # Lion uses 10x smaller LR
)
```

**Impact:** Claimed to be more memory-efficient and performant than Adam

---

### Gradient Clipping

**File to modify:** `model.py:154-162`

```python
self.optimizer = keras.optimizers.Adam(
    learning_rate=learning_rate,
    clipnorm=1.0  # Clip gradients by norm
)
```

**Impact:** Stabilizes training, prevents gradient explosion

---

## 9. Mixed Precision Training

**File to modify:** `train.py` (before model creation)

```python
from keras import mixed_precision

# Enable mixed precision (FP16 + FP32)
mixed_precision.set_global_policy('mixed_float16')

# Build model as usual
zero_dce_model = ZeroDCE()
```

**Impact:**
- 2-3x faster training on modern GPUs
- Reduced memory usage
- No accuracy loss in most cases

**Requirements:** NVIDIA GPU with Tensor Cores (RTX series, V100, A100)

---

## Recommended Optimization Path

### Phase 1: Low-Hanging Fruit (1-2 hours)

**Goal:** Maximize performance with minimal code changes

1. Use all training data: `--max-train-images 485`
2. Train longer: `--epochs 150`
3. Experiment with learning rate: `--learning-rate 5e-5` or `2e-4`
4. Add learning rate scheduling (ReduceLROnPlateau)
5. Add early stopping and model checkpointing

**Expected improvement:** 10-20% better metrics

**Command to try:**
```bash
python train.py \
    --max-train-images 485 \
    --epochs 150 \
    --learning-rate 5e-5 \
    --batch-size 32
```

---

### Phase 2: Moderate Changes (2-4 hours)

**Goal:** Fine-tune hyperparameters and training process

6. Experiment with batch sizes: 8, 16, 32
7. Try different exposure target values: 0.5, 0.6, 0.7
8. Fine-tune loss weights (one at a time)
9. Experiment with AdamW optimizer
10. Add gradient clipping

**Expected improvement:** Additional 5-10% improvement

---

### Phase 3: Architecture & Data (4-8 hours)

**Goal:** Improve model capacity and data diversity

11. Increase model capacity (48 or 64 filters)
12. Add batch normalization to all conv layers
13. Implement data augmentation
14. Try mixed precision training
15. Experiment with 6 or 10 curve iterations

**Expected improvement:** 10-30% improvement (highly variable)

---

### Phase 4: Advanced Experiments (8+ hours)

**Goal:** Research-level improvements

16. Download and integrate VE-LOL or SICE dataset
17. Pre-train on large dataset, fine-tune on LOL
18. Add residual connections
19. Try ensemble of models (different random seeds)
20. Experiment with different loss formulations

**Expected improvement:** 20-50% improvement (dataset-dependent)

---

## Quick Experiment Template

Create a shell script `optimize_experiment.sh`:

```bash
#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Experiment 1: Baseline with all data
python train.py \
    --max-train-images 485 \
    --epochs 150 \
    --learning-rate 1e-4 \
    --batch-size 16 \
    --save-path ./weights/experiment_1_baseline.weights.h5 \
    --plot-dir ./training_plots/experiment_1

# Experiment 2: Lower learning rate
python train.py \
    --max-train-images 485 \
    --epochs 150 \
    --learning-rate 5e-5 \
    --batch-size 16 \
    --save-path ./weights/experiment_2_lr5e5.weights.h5 \
    --plot-dir ./training_plots/experiment_2

# Experiment 3: Larger batch
python train.py \
    --max-train-images 485 \
    --epochs 150 \
    --learning-rate 1e-4 \
    --batch-size 32 \
    --save-path ./weights/experiment_3_batch32.weights.h5 \
    --plot-dir ./training_plots/experiment_3

# Experiment 4: Higher resolution
python train.py \
    --max-train-images 485 \
    --epochs 150 \
    --learning-rate 1e-4 \
    --batch-size 8 \
    --image-size 512 \
    --save-path ./weights/experiment_4_res512.weights.h5 \
    --plot-dir ./training_plots/experiment_4

echo "All experiments completed!"
```

---

## Measuring Improvement

### Quantitative Metrics

**During Training:**
- Monitor validation loss trends
- Compare final loss values across experiments
- Check for overfitting (train vs. val loss gap)

**After Training:**
- Use eval15 test set for comparison
- Measure PSNR (Peak Signal-to-Noise Ratio)
- Measure SSIM (Structural Similarity Index)

### Qualitative Evaluation

- Visual inspection of enhanced images
- Check for artifacts, color shifts
- Evaluate on diverse lighting conditions
- User preference studies (if available)

### Create Evaluation Script

```python
# evaluate.py
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def evaluate_model(model_path, test_dataset):
    """Evaluate model on test dataset."""
    # Load model and generate enhancements
    # Compute PSNR and SSIM
    # Return metrics
    pass
```

---

## Important Notes

### Before Modifying Code

1. Always commit current working version to git
2. Create a new branch for experiments: `git checkout -b experiment/optimizer-tuning`
3. Document all changes in this file
4. Keep original hyperparameters as comments

### Tracking Experiments

Use a spreadsheet or experiment tracking tool (MLflow, Weights & Biases):

| Experiment | LR | Batch | Epochs | Filters | Loss Weights | Val Loss | Notes |
|------------|-----|-------|--------|---------|--------------|----------|-------|
| Baseline | 1e-4 | 16 | 100 | 32 | 200,1,5,10 | 0.0234 | Original |
| Exp-1 | 5e-5 | 16 | 150 | 32 | 200,1,5,10 | 0.0198 | Better! |
| Exp-2 | 1e-4 | 32 | 150 | 32 | 200,1,5,10 | 0.0245 | Worse |

### Warning: Don't Change Multiple Things at Once

Always change ONE hyperparameter at a time to understand its impact.

---

## References

### Papers on Optimization

- **Adam vs. AdamW:** "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
- **Learning Rate Scheduling:** "SGDR: Stochastic Gradient Descent with Warm Restarts" (Loshchilov & Hutter, 2017)
- **Batch Normalization:** "Batch Normalization: Accelerating Deep Network Training" (Ioffe & Szegedy, 2015)
- **Mixed Precision Training:** "Mixed Precision Training" (Micikevicius et al., 2018)

### Datasets

- LOL Dataset: https://daooshee.github.io/BMVC2018website/
- VE-LOL: https://github.com/flyywh/CVPR-2020-Semi-Low-Light
- SICE: https://github.com/csjcai/SICE
- SID: https://github.com/cchen156/Learning-to-See-in-the-Dark

---

## Next Steps

1. Read this guide and decide which optimizations to try
2. Start with Phase 1 (low-hanging fruit)
3. Document results in experiment tracking table
4. Gradually move to more advanced optimizations
5. Compare results and select best configuration

**Good luck optimizing!**
