# Zero-DCE: Low-Light Image Enhancement

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.11+-red.svg)](https://keras.io/)

Re-implementation of **Zero-Reference Deep Curve Estimation** in Keras 3 for low-light image enhancement. This project is part of a graduate-level Digital Image Processing course final project.

## Overview

**Zero-DCE** is a deep learning approach that enhances low-light images without requiring paired training data. It learns to estimate pixel-wise tonal curves through unsupervised training with carefully designed loss functions.

### Key Features

- 🚀 **Zero-reference learning**: No need for paired low-light/normal-light images during training
- 💡 **Lightweight architecture**: DCE-Net has only ~79K parameters (7 convolutional layers)
- 📈 **Curve-based enhancement**: Learns pixel-wise tonal curves for dynamic range adjustment
- 🎯 **Unsupervised training**: Uses 4 carefully designed loss functions
- 🔧 **Modular design**: Clean separation of concerns (dataset, model, loss, training, inference)
- 📊 **Comparison tools**: Built-in comparison with classical enhancement methods

## Project Structure

```
redo-zero-dce-keras/
├── dataset.py                  # Data loading and preprocessing
├── loss.py                     # Unsupervised loss functions
├── model.py                    # DCE-Net architecture and ZeroDCE model
├── train.py                    # Training script with CLI
├── compare.py                  # Inference and comparison tool
├── classical_methods.py        # Classical enhancement methods
├── zero_dce.py                 # Original monolithic implementation (reference)
├── test_*.py                   # Test suites for each module
├── lol_dataset/                # LOL Dataset (485 train + 15 test images)
│   ├── our485/low/             # Training low-light images
│   ├── our485/high/            # Training normal-light images (ground truth)
│   └── eval15/low/             # Test low-light images
├── weights/                    # Trained model weights
├── training_plots/             # Training curve visualizations
├── pyproject.toml              # Project dependencies
├── AGENTS.md                   # Instructions for AI coding agents
└── REFACTOR_PLAN.md            # Detailed refactoring roadmap
```

## Installation

### Prerequisites

- Python 3.13+
- CUDA-capable GPU (recommended for training)
- 4GB+ GPU memory for training

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd redo-zero-dce-keras
```

2. **Create virtual environment**
```bash
# Using uv (recommended)
uv venv

# Or using standard Python
python -m venv .venv
```

3. **Activate virtual environment**
```bash
# Linux/macOS (bash/zsh)
source .venv/bin/activate

# Linux/macOS (fish)
source .venv/bin/activate.fish

# Windows (Command Prompt)
.venv\Scripts\activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

4. **Install dependencies**
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Download LOL Dataset

The LoL (Low-Light) Dataset provides 485 training images and 15 test images.

```bash
wget https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip
unzip -q lol_dataset.zip && rm lol_dataset.zip
```

## Usage

### Training

Train the Zero-DCE model with default parameters:

```bash
python train.py
```

#### Training Options

```bash
python train.py \
  --epochs 100 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --image-size 256 \
  --dataset-path ./lol_dataset \
  --max-train-images 400 \
  --save-path ./weights/zero_dce.weights.h5 \
  --plot-dir ./training_plots
```

**Parameters:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 16)
- `--learning-rate`: Learning rate for Adam optimizer (default: 1e-4)
- `--image-size`: Image size (height and width) (default: 256)
- `--dataset-path`: Path to LOL dataset directory (default: ./lol_dataset)
- `--max-train-images`: Maximum number of images for training (default: 400)
- `--save-path`: Path to save trained model weights (default: ./weights/zero_dce.weights.h5)
- `--plot-dir`: Directory to save training plots (default: ./training_plots)

**Training output:**
- Model weights saved to `weights/zero_dce.weights.h5`
- Training plots saved to `training_plots/`:
  - `total_loss.png`
  - `illumination_smoothness_loss.png`
  - `spatial_constancy_loss.png`
  - `color_constancy_loss.png`
  - `exposure_loss.png`

### Inference and Comparison

Enhance a single image using trained Zero-DCE model:

```bash
python compare.py \
  -i ./lol_dataset/eval15/low/1.png \
  -w ./weights/zero_dce.weights.h5 \
  -o ./enhanced_comparison.png
```

#### Comparison Options

```bash
python compare.py \
  -i <input_image> \
  -w <weights_file> \
  -o <output_image> \
  -r <reference_image> \
  --methods zero-dce autocontrast histogram-eq clahe gamma \
  --save-individual
```

**Parameters:**
- `-i, --input`: Path to input low-light image (required)
- `-w, --weights`: Path to trained Zero-DCE model weights (required)
- `-o, --output`: Path to save comparison image (optional, displays if not provided)
- `-r, --reference`: Path to reference (ground truth) image for comparison (optional)
- `--methods`: Enhancement methods to compare (default: all)
  - `zero-dce`: Zero-DCE deep learning method
  - `autocontrast`: PIL AutoContrast
  - `histogram-eq`: OpenCV Histogram Equalization
  - `clahe`: Contrast Limited Adaptive Histogram Equalization
  - `gamma`: Gamma correction (gamma=2.2)
- `--save-individual`: Save individual enhanced images to subdirectory

#### Example with Reference Image

```bash
python compare.py \
  -i ./lol_dataset/eval15/low/1.png \
  -w ./weights/zero_dce.weights.h5 \
  -r ./lol_dataset/eval15/high/1.png \
  -o ./comparison_with_reference.png
```

#### Example with Specific Methods

```bash
python compare.py \
  -i ./test_image.jpg \
  -w ./weights/zero_dce.weights.h5 \
  --methods zero-dce autocontrast clahe
```

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

```
Input (H×W×3)
    ↓
Conv1 (32 filters, 3×3, ReLU) ──────────┐
    ↓                                    │
Conv2 (32 filters, 3×3, ReLU) ────┐     │
    ↓                              │     │
Conv3 (32 filters, 3×3, ReLU) ─┐  │     │
    ↓                           │  │     │
Conv4 (32 filters, 3×3, ReLU)  │  │     │
    ↓                           │  │     │
Conv5 (32 filters, 3×3, ReLU)  │  │     │
    ↓ ← Concat ← ← ← ← ← ← ← ← ┘  │     │
Conv6 (32 filters, 3×3, ReLU)     │     │
    ↓ ← Concat ← ← ← ← ← ← ← ← ← ─┘     │
Conv7 (32 filters, 3×3, ReLU)           │
    ↓ ← Concat ← ← ← ← ← ← ← ← ← ← ← ← ┘
Conv8 (24 filters, 3×3, Tanh)
    ↓
Output (H×W×24)
```

**Output channels:** 24 = 8 iterations × 3 RGB channels

### Loss Functions (Unsupervised)

The model is trained using 4 non-reference loss functions:

1. **Spatial Consistency Loss** (weight: 1)
   - Preserves contrast between neighboring regions
   - Ensures spatial coherence across input and enhanced images

2. **Color Constancy Loss** (weight: 5)
   - Corrects color deviations
   - Measures deviation between average values of RGB channels

3. **Exposure Loss** (weight: 10)
   - Prevents under/over-exposure
   - Measures distance from target well-exposedness level (0.6)

4. **Illumination Smoothness Loss** (weight: 200)
   - Preserves monotonicity between neighboring pixels
   - Minimizes total variation in curve parameter maps

**Total Loss:**
```
L_total = 200 × L_illum + 10 × L_exposure + 5 × L_color + 1 × L_spatial
```

### Enhancement Process

The learned curve parameters are applied iteratively (8 iterations):

```
x_{n+1} = x_n + r_n × (x_n² - x_n)
```

where:
- `x_n`: Image at iteration n
- `r_n`: Learned curve parameter at iteration n

## Module Documentation

### `dataset.py`

Handles data loading and preprocessing for the LOL Dataset.

**Key functions:**
- `load_data(image_path, image_size)`: Load and preprocess a single image
- `data_generator(low_light_images, batch_size, image_size)`: Create TensorFlow dataset
- `get_dataset(dataset_path, max_train_images, batch_size, image_size)`: Load train/val/test datasets

### `loss.py`

Implements the 4 unsupervised loss functions.

**Key components:**
- `color_constancy_loss(x)`: Measures color deviation
- `exposure_loss(x, mean_val)`: Controls exposure level
- `illumination_smoothness_loss(x)`: Preserves spatial smoothness
- `SpatialConsistencyLoss`: Keras loss class for spatial coherence

### `model.py`

Contains the DCE-Net architecture and Zero-DCE training wrapper.

**Key components:**
- `build_dce_net()`: Build 7-layer DCE-Net architecture
- `get_enhanced_image(data, output)`: Apply curve enhancement iteratively
- `ZeroDCE`: Training wrapper class that combines DCE-Net with loss functions

### `train.py`

Command-line training script with configurable parameters.

**Key functions:**
- `plot_training_history(history, save_dir)`: Generate training curve plots
- `main()`: Main training loop with argparse CLI

### `compare.py`

Inference and comparison tool with multiple enhancement methods.

**Key functions:**
- `load_model_for_inference(weights_path)`: Load trained model
- `enhance_with_zero_dce(image, model)`: Apply Zero-DCE enhancement
- `compare_methods(...)`: Compare multiple enhancement methods

### `classical_methods.py`

Classical image enhancement methods for comparison.

**Available methods:**
- `enhance_with_autocontrast(image)`: PIL AutoContrast
- `enhance_with_histogram_eq(image)`: Histogram Equalization
- `enhance_with_clahe(image, clip_limit)`: CLAHE
- `enhance_with_gamma_correction(image, gamma)`: Gamma correction

## Testing

Each module has a corresponding test file:

```bash
# Test dataset loading
python test_dataset.py

# Test loss functions
python test_loss.py

# Test model building
python test_model.py

# Test training (1 epoch)
python test_train.py

# Test comparison tool
python test_compare.py

# Test classical methods
python test_classical_methods.py
```

## Results

### Training Progress

Typical training progress on LOL Dataset (400 training images):

| Metric | Initial | After 100 epochs |
|--------|---------|------------------|
| Total Loss | ~8-10 | ~2-3 |
| Illumination Smoothness | ~6-8 | ~0.8-1.5 |
| Exposure Loss | ~2-3 | ~1.5-2.0 |
| Color Constancy | ~0.001 | ~0.002-0.003 |
| Spatial Consistency | ~0.0001 | ~0.0001-0.0002 |

### Visual Results

The model produces visually pleasing results that:
- Brighten dark regions without over-saturating bright areas
- Preserve color fidelity and natural appearance
- Maintain spatial coherence and edge details
- Outperform classical methods in most cases

## Performance

### Training Time

- **Hardware:** NVIDIA GPU (e.g., RTX 3080)
- **Time per epoch:** ~1-2 minutes (400 images, batch size 16)
- **Total training time:** ~1.5-3 hours (100 epochs)

### Inference Time

- **Single image (256×256):** ~0.1-0.2 seconds on GPU
- **Single image (512×512):** ~0.3-0.5 seconds on GPU

## Citation

If you use this code in your research, please cite the original Zero-DCE paper:

```bibtex
@inproceedings{guo2020zero,
  title={Zero-reference deep curve estimation for low-light image enhancement},
  author={Guo, Chunle and Li, Chongyi and Guo, Jichang and Loy, Chen Change and Hou, Junhui and Kwong, Sam and Cong, Runmin},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={1780--1789},
  year={2020}
}
```

## References

- **Paper:** [Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement](https://arxiv.org/abs/2001.06826) (CVPR 2020)
- **Original Keras Example:** [Keras Zero-DCE Tutorial](https://keras.io/examples/vision/zero_dce/)
- **LOL Dataset:** [Low-Light Image Dataset](https://huggingface.co/datasets/geekyrakshit/LoL-Dataset)
- **Adobe Curves:** [Curves adjustment in Adobe Photoshop](https://helpx.adobe.com/photoshop/using/curves-adjustment.html)

## License

This project is for educational purposes as part of a graduate Digital Image Processing course.

## Acknowledgments

- Original Zero-DCE implementation by Chunle Guo et al.
- Keras 3 port by [Soumik Rakshit](http://github.com/soumik12345)
- Course instructors and fellow students for feedback and guidance

## Troubleshooting

### Common Issues

**Import errors:**
- Ensure all modules are in the same directory
- Verify virtual environment is activated

**OOM (Out of Memory) errors during training:**
- Reduce `--batch-size` (try 8 or 4)
- Reduce `--image-size` (try 128 or 192)
- Check GPU memory usage with `nvidia-smi`

**Model weight loading fails:**
- Verify weight file path is correct
- Ensure weights were saved from the latest training
- Check file extension is `.weights.h5`

**Comparison script fails:**
- Verify opencv-python is installed: `pip install opencv-python`
- Check input image format is supported (PNG, JPG)
- Ensure weights file exists

**Training produces NaN losses:**
- Try reducing learning rate (`--learning-rate 1e-5`)
- Check dataset is properly loaded
- Verify images are normalized to [0, 1]

## Contributing

This is a course project, but suggestions and improvements are welcome! Please feel free to:
- Report bugs or issues
- Suggest enhancements
- Share your training results
- Contribute improvements to the code

## Contact

For questions related to this implementation, please refer to:
- `AGENTS.md` for detailed technical documentation
- `REFACTOR_PLAN.md` for development roadmap
- Course instructors for academic guidance
