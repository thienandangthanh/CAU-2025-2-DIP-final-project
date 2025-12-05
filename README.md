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
zero-dce-keras/
├── dataset.py                  # Data loading and preprocessing
├── loss.py                     # Unsupervised loss functions
├── model.py                    # DCE-Net architecture and ZeroDCE model
├── train.py                    # Training script with CLI
├── compare.py                  # Inference and comparison tool
├── classical_methods.py        # Classical enhancement methods
├── plot_utils.py               # Publication-quality plotting utilities
├── layout_utils.py             # Grid layout calculation utilities
├── zero_dce.py                 # Original monolithic implementation (reference)
├── gui_app.py                  # PyQt6 GUI application entry point
├── gui/                        # GUI components
│   ├── main_window.py          # Main application window
│   ├── widgets/                # Custom widgets (comparison grid, cells, etc.)
│   ├── dialogs/                # Dialog windows (preferences, method selection)
│   └── utils/                  # GUI utilities (enhancement runner, methods, etc.)
├── test_*.py                   # Test suites for each module
│   ├── test_dataset.py
│   ├── test_loss.py
│   ├── test_model.py
│   ├── test_train.py
│   ├── test_compare.py
│   └── test_classical_methods.py
├── tests/                      # Additional test modules (GUI tests)
├── lol_dataset/                # LOL Dataset (485 train + 15 test images)
│   ├── our485/                 # Training set (485 pairs)
│   │   ├── low/                # Training low-light images
│   │   └── high/               # Training normal-light images (ground truth)
│   └── eval15/                 # Test set (15 pairs)
│       ├── low/                # Test low-light images
│       └── high/               # Test normal-light images (ground truth)
├── weights/                    # Trained model weights
│   └── zero_dce.weights.h5     # Keras model weights
├── training_plots/             # Training curve visualizations
│   ├── total_loss.png
│   ├── illumination_smoothness_loss.png
│   ├── spatial_constancy_loss.png
│   ├── color_constancy_loss.png
│   └── exposure_loss.png
├── outputs/                    # Generated comparison outputs
├── docs/                       # Documentation
├── pyproject.toml              # Project dependencies and tool configuration
├── uv.lock                     # Lock file for dependencies
├── pytest.ini                  # Pytest configuration
├── .pre-commit-config.yaml     # Pre-commit hooks configuration
├── .python-version             # Python version specification
├── .gitignore                  # Git ignore rules
├── AGENTS.md                   # Instructions for AI coding agents
├── REFACTOR_PLAN.md            # Detailed refactoring roadmap
└── README.md                   # This file
```

## Installation

### Prerequisites

- [Python 3.13+](https://www.python.org/downloads/)
- [uv - Python project and package manager](https://docs.astral.sh/uv/getting-started/installation/)
- **For Linux/macOS:**
  - CUDA-capable GPU (recommended for training)
  - [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit)
  - [NVIDIA driver](https://www.nvidia.com/en-us/drivers/)
  - 4GB+ GPU memory for training
- **For Windows users:**
  - **Required:** [WSL2 (Windows Subsystem for Linux)](#windows-users-wsl2-required)
  - Native Windows is **not supported** due to OpenCV DLL issues and lack of TensorFlow GPU support

### Setup

1. **Change current directory**
```bash
cd <directory storing this source code>
```

2. **Create virtual environment**
```bash
# Using uv (recommended)
uv venv
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
# Using uv (installs both regular and dev dependencies)
uv sync
```

5. **(Optional) Install LaTeX for publication-quality plots**

For generating manuscript-ready figures with LaTeX-rendered text:

```bash
# Linux (Debian/Ubuntu)
sudo apt-get install texlive texlive-latex-extra cm-super dvipng

# Linux (Arch)
sudo pacman -S texlive-core texlive-bin texlive-latexextra

# Linux (Fedora/RHEL)
sudo dnf install texlive texlive-latex texlive-dvipng

# macOS (requires Homebrew)
brew install --cask mactex
# Or smaller BasicTeX distribution:
brew install --cask basictex

# Windows (in WSL2 Ubuntu)
sudo apt-get install texlive texlive-latex-extra cm-super dvipng
```

**Note:** LaTeX is optional. If not installed, plots will use standard matplotlib rendering with a warning message. This is fine for development and testing. LaTeX rendering is only needed for publication-quality figures in manuscripts.

### Windows Users: WSL2 (Required)

**Windows developers must use WSL2.** Native Windows is not supported due to:
- ❌ OpenCV DLL loading issues
- ❌ TensorFlow 2.11+ has no native GPU support (CPU-only)
- ❌ Python package compatibility problems

**WSL2 provides:**
- ✅ Native Linux environment with full CUDA GPU support
- ✅ No OpenCV DLL issues
- ✅ Full compatibility with all Python packages
- ✅ Fast GPU-accelerated training
- ✅ Same development experience as Linux/macOS

#### Installing WSL2

1. **Install WSL2** (requires Windows 10 version 2004+ or Windows 11)
   ```powershell
   # Run in PowerShell as Administrator
   wsl --install
   ```
   This will install Ubuntu by default. Restart your computer when prompted.

2. **Install NVIDIA CUDA on WSL2** (for GPU support)
   - Install [NVIDIA GPU Driver for WSL](https://developer.nvidia.com/cuda/wsl) on Windows (NOT inside WSL)
   - CUDA toolkit will be available automatically inside WSL2
   - Verify GPU is accessible in WSL:
     ```bash
     nvidia-smi
     ```

3. **Setup the project in WSL2**
   ```bash
   # Open WSL2 Ubuntu terminal
   cd /mnt/c/path/to/your/project  # or clone from git

   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Follow the regular Linux setup instructions above
   uv venv
   source .venv/bin/activate
   uv sync
   ```

4. **Run the application**
   ```bash
   uv run python gui_app.py
   ```
   Note: GUI applications require an X server on Windows. See [Running GUI apps](#running-gui-apps-in-wsl2) below.

#### Running GUI Apps in WSL2

For the PyQt6 GUI application, you have two options:

**Option 1: Windows 11 (Built-in WSLg support)**
- Windows 11 has built-in GUI support (WSLg)
- GUI apps work out of the box, no additional setup needed

**Option 2: Windows 10 (Requires X Server)**
1. Install [VcXsrv](https://sourceforge.net/projects/vcxsrv/) or [X410](https://x410.dev/) on Windows
2. Start the X server with "Disable access control" option
3. In WSL2, set the DISPLAY variable:
   ```bash
   export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
   ```
4. Run the application: `uv run python gui_app.py`

**Option 3: Use CLI tools only**
- For training and inference, you don't need the GUI
- Use `train.py` and `compare.py` command-line scripts instead

### Development Setup

For developers who want to contribute or modify the code, additional setup is required to ensure code quality and consistency.

#### Code Quality Tools

This project uses the following tools to maintain code quality:
- **Ruff** - Fast Python linter and formatter (replaces flake8, isort, black)
- **Pre-commit** - Git hooks framework for automatic code quality checks

#### Installing Pre-commit Hooks

After installing dependencies with `uv sync`, install the pre-commit hooks:

```bash
uv run pre-commit install
```

This will automatically run code quality checks before each commit:
- Trailing whitespace removal
- End-of-file fixes
- YAML/JSON/TOML validation
- Large file detection
- Merge conflict detection
- Debug statement detection
- Ruff linting with auto-fix
- Ruff formatting

#### Running Checks Manually

To manually run all pre-commit hooks on all files:

```bash
uv run pre-commit run --all-files
```

To run only Ruff linter:

```bash
uv run ruff check .
```

To run only Ruff formatter:

```bash
uv run ruff format .
```

#### Configuration Files

- `.pre-commit-config.yaml` - Pre-commit hooks configuration
- `pyproject.toml` - Tool configuration (Ruff settings, project metadata)
- `pytest.ini` - Pytest configuration

**Note:** Pre-commit hooks will automatically fix most formatting issues. If there are linting errors that cannot be auto-fixed, the commit will be blocked until you fix them manually.

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
Conv1 (32 filters, 3×3, ReLU) ─────────┐
    ↓                                  │
Conv2 (32 filters, 3×3, ReLU) ────┐    │
    ↓                             │    │
Conv3 (32 filters, 3×3, ReLU) ─┐  │    │
    ↓                          │  │    │
Conv4 (32 filters, 3×3, ReLU)  │  │    │
    ↓                          │  │    │
Conv5 (32 filters, 3×3, ReLU)  │  │    │
    ↓ ← Concat ← ← ← ← ← ← ← ← ┘  │    │
Conv6 (32 filters, 3×3, ReLU)     │    │
    ↓ ← Concat ← ← ← ← ← ← ← ← ← ─┘    │
Conv7 (32 filters, 3×3, ReLU)          │
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

### `plot_utils.py`

Utilities for consistent publication-quality plotting across all scripts.

**Key functions:**
- `configure_publication_style()`: Apply manuscript-ready styling to all matplotlib plots
- `check_latex_available()`: Detect if LaTeX is installed on the system
- `get_publication_style()`: Get matplotlib configuration dictionary

**Features:**
- Automatic LaTeX detection (enables LaTeX rendering if available)
- Consistent serif fonts and sizes for manuscripts
- Colorblind-friendly color palette
- Graceful fallback when LaTeX is not installed

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

- **Hardware:**
  - GPU: NVIDIA RTX 3070 8GB
  - CPU: Intel Core i7 12700
  - RAM: 32GB
- **Time per epoch:** ~2-15 seconds (400 images, batch size 16)
- **Total training time:** ~4 minutes (100 epochs)

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

- **Paper:** [Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement](https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.html) (CVPR 2020)
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

**Pre-commit hooks not running:**
- Verify hooks are installed: `uv run pre-commit install`
- Check `.git/hooks/pre-commit` exists and is executable
- Try running manually: `uv run pre-commit run --all-files`

**Pre-commit hooks blocking commit:**
- Review the error messages to see which checks failed
- Most formatting issues are auto-fixed; run `git add .` to stage the fixes
- For linting errors that can't be auto-fixed, manually fix the code
- Run `uv run pre-commit run --all-files` to verify all checks pass

### Windows-Specific Issues

**Native Windows is not supported.** If you're on Windows, use WSL2 as described in the [Windows Users: WSL2 (Required)](#windows-users-wsl2-required) section.

**WSL2-specific issues:**

**WSL2 installation fails:**
- Ensure Windows 10 version 2004+ or Windows 11
- Run PowerShell as Administrator
- Enable virtualization in BIOS if needed

**GPU not detected in WSL2:**
- Install [NVIDIA GPU Driver for WSL](https://developer.nvidia.com/cuda/wsl) on Windows host
- Verify with `nvidia-smi` inside WSL2
- Do NOT install CUDA toolkit inside WSL2 - it's included automatically

**GUI not working in WSL2:**
- **Windows 11:** WSLg provides built-in GUI support, should work out of the box
- **Windows 10:** Install X server (VcXsrv or X410) and set DISPLAY variable:
  ```bash
  export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}):0
  ```
- **Alternative:** Use CLI tools (`train.py`, `compare.py`) instead of GUI app

**Slow file I/O in WSL2:**
- Keep your project files in WSL2 filesystem (`/home/user/...`), not Windows filesystem (`/mnt/c/...`)
- Clone the repository inside WSL2 for best performance

## Contributing

This is a course project, but suggestions and improvements are welcome! Please feel free to:
- Report bugs or issues
- Suggest enhancements
- Share your training results
- Contribute improvements to the code

### Before Contributing

1. **Install dependencies:** Run `uv sync` to install all dependencies
2. **Install pre-commit hooks:** Run `uv run pre-commit install` to set up automatic code quality checks
3. **Run tests:** Ensure all tests pass before submitting changes
4. **Follow code style:** The pre-commit hooks will automatically format your code using Ruff

All commits will be automatically checked for code quality. Make sure to fix any linting errors before committing.

## Contact

For questions related to this implementation, please refer to:
- `AGENTS.md` for detailed technical documentation
- `REFACTOR_PLAN.md` for development roadmap
- Course instructors for academic guidance
